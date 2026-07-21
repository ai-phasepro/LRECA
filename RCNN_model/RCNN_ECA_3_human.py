
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POS_PATH = SCRIPT_DIR.parent / "Data" / "pos_dataset" / "human_pos_word_list.txt"
DEFAULT_NEG_PATH = SCRIPT_DIR.parent / "Data" / "neg_dataset" / "human_neg_word_list.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "human_10fold_outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train RCNN-ECA on human data with manual 10-fold CV")
    parser.add_argument("--pos-protein-dir", type=Path, default=DEFAULT_POS_PATH)
    parser.add_argument("--neg-protein-dir", type=Path, default=DEFAULT_NEG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--export-epoch", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pos-seed", type=int, default=20)
    parser.add_argument("--neg-seed", type=int, default=21)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None, help="Physical GPU index; omitted means CUDA device 0")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto/cuda require CUDA; cpu is intended for diagnostics only",
    )
    parser.add_argument("--max-folds", type=int, default=10, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not 1 <= args.export_epoch <= args.epochs:
        parser.error("--export-epoch must be between 1 and --epochs")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 1 <= args.max_folds <= 10:
        parser.error("--max-folds must be between 1 and 10")
    return args


def resolve_device(device_name, gpu_index):
    if gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    if device_name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run in a CUDA-enabled environment or use "
            "--device cpu only for a diagnostic run."
        )
    return torch.device("cuda:0")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_sequences(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    sequences = path.read_text(encoding="utf-8").splitlines()
    if not sequences:
        raise ValueError(f"Dataset is empty: {path}")
    empty_lines = [index + 1 for index, sequence in enumerate(sequences) if not sequence.replace(" ", "")]
    if empty_lines:
        raise ValueError(f"Empty protein sequence(s) in {path}: lines {empty_lines[:10]}")
    return sequences


def read_data(pos_path, neg_path, pos_seed, neg_seed):
    pos_sequences = read_sequences(pos_path)
    neg_sequences = read_sequences(neg_path)
    # Preserve mydata's RNG behavior, including leaving the global NumPy RNG at
    # the negative-class seed for the first train/validation shuffle.
    np.random.seed(pos_seed)
    np.random.shuffle(pos_sequences)
    np.random.seed(neg_seed)
    np.random.shuffle(neg_sequences)
    return pos_sequences, neg_sequences


def encode_sequences(train_val_sequences, test_sequences):
    counts = {}
    for sequence in train_val_sequences:
        for residue in sequence.replace(" ", ""):
            counts[residue] = counts.get(residue, 0) + 1
    vocabulary = {residue: index + 1 for index, residue in enumerate(counts)}

    def encode(sequence):
        encoded = []
        for residue in sequence.replace(" ", ""):
            index = vocabulary.get(residue)
            if index is None:
                raise ValueError(f"Test-only residue {residue!r} is absent from the training vocabulary")
            encoded.append(index)
        return encoded

    train_val_encoded = [encode(sequence) for sequence in train_val_sequences]
    test_encoded = [encode(sequence) for sequence in test_sequences]
    return train_val_encoded, test_encoded, vocabulary


def collate_fn(batch):
    batch.sort(key=lambda item: len(item[0]), reverse=True)
    lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)
    sequences = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return sequences, labels, lengths


class ProteinDataset(Dataset):
    def __init__(self, data, labels):
        if len(data) != len(labels):
            raise ValueError("Data and label counts do not match")
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class ECALayer(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, values, lengths):
        pooled = []
        for index in range(values.size(0)):
            unpadded = values[index, :, : lengths[index]].unsqueeze(0)
            pooled.append(self.avg_pool(unpadded))
        weights = torch.cat(pooled, dim=0)
        weights = self.conv(weights.transpose(-1, -2)).transpose(-1, -2)
        weights = self.sigmoid(weights)
        return values * weights.expand_as(values)


class RCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=1024, hidden_dim=100, dropout=0.2):
        super().__init__()
        self.eca = ECALayer(kernel_size=5)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim + embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, inputs, lengths):
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        packed_output, _ = self.lstm(packed)
        recurrent, _ = pad_packed_sequence(packed_output, batch_first=True)
        features = torch.cat((embedded, recurrent), dim=2)
        features = F.relu(features).permute(0, 2, 1)
        features = features + self.eca(features, lengths)
        features = self.global_max_pool(features).squeeze(-1)
        return self.classifier(F.relu(features))


def make_loader(encoded, labels, batch_size, shuffle):
    tensors = [torch.tensor(sequence, dtype=torch.long) for sequence in encoded]
    dataset = ProteinDataset(tensors, labels.astype(np.int64))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def calculate_metrics(labels, predictions, scores, loss):
    labels_np = np.asarray(labels)
    predictions_np = np.asarray(predictions)
    positives = predictions_np[labels_np == 1]
    negatives = predictions_np[labels_np == 0]
    sensitivity = float(np.mean(positives == 1)) if positives.size else 1.0
    specificity = float(np.mean(negatives == 0)) if negatives.size else 1.0
    return {
        "loss": loss,
        "acc": metrics.accuracy_score(labels_np, predictions_np),
        "precision": metrics.precision_score(labels_np, predictions_np, zero_division=0),
        "recall": metrics.recall_score(labels_np, predictions_np, zero_division=0),
        "f1_macro": metrics.f1_score(labels_np, predictions_np, average="macro", zero_division=0),
        "auc": metrics.roc_auc_score(labels_np, scores),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def run_loader(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    labels_all = []
    predictions_all = []
    scores_all = []
    loss_sum = 0.0
    sample_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, labels, lengths in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            outputs = model(inputs, lengths)
            loss = loss_fn(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            sample_count += batch_size
            probabilities = torch.softmax(outputs, dim=-1)[:, 1]
            labels_all.extend(labels.detach().cpu().tolist())
            predictions_all.extend(outputs.argmax(dim=1).detach().cpu().tolist())
            scores_all.extend(probabilities.detach().cpu().tolist())

    epoch_metrics = calculate_metrics(
        labels_all,
        predictions_all,
        scores_all,
        loss_sum / sample_count,
    )
    return epoch_metrics, labels_all, scores_all


def print_metrics(prefix, values):
    print(
        f"{prefix}: Correct: {values['acc']:.5f}, Precision: {values['precision']:.5f}, "
        f"R: {values['recall']:.5f}, F1(macro): {values['f1_macro']:.5f}, "
        f"AUC: {values['auc']:.5f}, loss: {values['loss']:.6f}",
        flush=True,
    )
    print(
        f"{prefix}: sen={values['sensitivity']:.5f}, "
        f"spe={values['specificity']:.5f}",
        flush=True,
    )


def split_fold(pos_sequences, neg_sequences, fold_index, val_fraction=0.1):
    pos_count = len(pos_sequences)
    neg_count = len(neg_sequences)
    pos_start = int(pos_count * fold_index / 10)
    neg_start = int(neg_count * fold_index / 10)
    pos_end = pos_count if fold_index == 9 else int(pos_count * (fold_index + 1) / 10)
    neg_end = neg_count if fold_index == 9 else int(neg_count * (fold_index + 1) / 10)

    test_pos = pos_sequences[pos_start:pos_end]
    test_neg = neg_sequences[neg_start:neg_end]
    train_val_pos = pos_sequences[:pos_start] + pos_sequences[pos_end:]
    train_val_neg = neg_sequences[:neg_start] + neg_sequences[neg_end:]

    # Match mydata: the global NumPy RNG is advanced and used to reshuffle each fold.
    np.random.shuffle(train_val_pos)
    np.random.shuffle(train_val_neg)
    val_pos_count = int(len(train_val_pos) * val_fraction)
    val_neg_count = int(len(train_val_neg) * val_fraction)
    val_pos, train_pos = train_val_pos[:val_pos_count], train_val_pos[val_pos_count:]
    val_neg, train_neg = train_val_neg[:val_neg_count], train_val_neg[val_neg_count:]

    return {
        "train": (train_neg + train_pos, np.r_[np.zeros(len(train_neg)), np.ones(len(train_pos))]),
        "val": (val_neg + val_pos, np.r_[np.zeros(len(val_neg)), np.ones(len(val_pos))]),
        "test": (test_neg + test_pos, np.r_[np.zeros(len(test_neg)), np.ones(len(test_pos))]),
    }


def main():
    args = parse_args()
    device = resolve_device(args.device, args.gpu)
    set_seed(args.seed)

    pos_sequences, neg_sequences = read_data(
        args.pos_protein_dir,
        args.neg_protein_dir,
        args.pos_seed,
        args.neg_seed,
    )
    if len(pos_sequences) != len(neg_sequences):
        raise ValueError(
            "The human experiment expects balanced classes, but found "
            f"{len(pos_sequences)} positive and {len(neg_sequences)} negative samples"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"rcnn_ECA_human_epoch{args.export_epoch}_roc.csv"
    pd.DataFrame(columns=["y_true", "y_score"]).to_csv(prediction_path, index=False)

    print(f"device={device}", flush=True)
    print(f"positive_samples={len(pos_sequences)} negative_samples={len(neg_sequences)}", flush=True)
    print(f"prediction_csv={prediction_path}", flush=True)

    exported_samples = 0
    covered_pos_indices = set()
    covered_neg_indices = set()

    for fold_index in range(args.max_folds):
        fold = split_fold(pos_sequences, neg_sequences, fold_index)
        train_sequences, train_labels = fold["train"]
        val_sequences, val_labels = fold["val"]
        test_sequences, test_labels = fold["test"]

        pos_start = int(len(pos_sequences) * fold_index / 10)
        neg_start = int(len(neg_sequences) * fold_index / 10)
        pos_end = len(pos_sequences) if fold_index == 9 else int(len(pos_sequences) * (fold_index + 1) / 10)
        neg_end = len(neg_sequences) if fold_index == 9 else int(len(neg_sequences) * (fold_index + 1) / 10)
        covered_pos_indices.update(range(pos_start, pos_end))
        covered_neg_indices.update(range(neg_start, neg_end))

        print(f"-------第 {fold_index + 1} fold-------", flush=True)
        print(
            f"train_pos={int(train_labels.sum())} train_neg={int((train_labels == 0).sum())} "
            f"val_pos={int(val_labels.sum())} val_neg={int((val_labels == 0).sum())} "
            f"test_pos={int(test_labels.sum())} test_neg={int((test_labels == 0).sum())}",
            flush=True,
        )

        train_val_sequences = train_sequences + val_sequences
        train_val_encoded, test_encoded, vocabulary = encode_sequences(
            train_val_sequences,
            test_sequences,
        )
        train_encoded = train_val_encoded[: len(train_sequences)]
        val_encoded = train_val_encoded[len(train_sequences) :]

        train_loader = make_loader(train_encoded, train_labels, args.batch_size, shuffle=True)
        val_loader = make_loader(val_encoded, val_labels, args.batch_size, shuffle=True)
        test_loader = make_loader(test_encoded, test_labels, args.batch_size, shuffle=True)

        model = RCNN(len(vocabulary) + 1, embedding_dim=1024, hidden_dim=100).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10)
        # Match mydata: reset RNGs after model construction and before loading batches.
        set_seed(args.seed)

        for epoch_index in range(args.epochs):
            epoch = epoch_index + 1
            print(f"-------第 {epoch} 轮训练开始-------", flush=True)
            train_metrics, _, _ = run_loader(model, train_loader, loss_fn, device, optimizer)
            val_metrics, _, _ = run_loader(model, val_loader, loss_fn, device)
            test_metrics, test_y_true, test_y_score = run_loader(model, test_loader, loss_fn, device)
            print_metrics("Train", train_metrics)
            print_metrics("val", val_metrics)
            print_metrics("Test", test_metrics)

            # Keep the original mydata schedule: step only after epoch index 70.
            if epoch_index > 70:
                scheduler.step(val_metrics["acc"])

            if epoch == args.export_epoch:
                pd.DataFrame(
                    {"y_true": test_y_true, "y_score": test_y_score}
                ).to_csv(
                    prediction_path,
                    mode="a",
                    header=False,
                    index=False,
                    float_format="%.4f",
                )
                exported_samples += len(test_y_true)

    expected_exported = sum(
        (len(pos_sequences) if index == 9 else int(len(pos_sequences) * (index + 1) / 10))
        - int(len(pos_sequences) * index / 10)
        + (len(neg_sequences) if index == 9 else int(len(neg_sequences) * (index + 1) / 10))
        - int(len(neg_sequences) * index / 10)
        for index in range(args.max_folds)
    )
    if exported_samples != expected_exported:
        raise RuntimeError(f"Exported {exported_samples} predictions; expected {expected_exported}")
    if args.max_folds == 10:
        if len(covered_pos_indices) != len(pos_sequences) or len(covered_neg_indices) != len(neg_sequences):
            raise RuntimeError("Ten-fold test coverage is incomplete")
        expected_total = len(pos_sequences) + len(neg_sequences)
        row_count = len(pd.read_csv(prediction_path))
        if row_count != expected_total:
            raise RuntimeError(f"Prediction CSV has {row_count} rows; expected {expected_total}")

    print(f"completed_folds={args.max_folds}", flush=True)
    print(f"exported_predictions={exported_samples}", flush=True)
    print(f"prediction_csv={prediction_path}", flush=True)


if __name__ == "__main__":
    main()
