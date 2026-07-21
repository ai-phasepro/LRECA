
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

# Capture the absolute script path before importing the legacy module below.
# That module changes cwd during import, which would otherwise make a relative
# __file__ resolve to Demo/code_for_model_testing/code_for_model_testing/....
_SCRIPT_PATH = Path(__file__)
if not _SCRIPT_PATH.is_absolute():
    _SCRIPT_PATH = (Path.cwd() / _SCRIPT_PATH).resolve()
else:
    _SCRIPT_PATH = _SCRIPT_PATH.resolve()
SCRIPT_DIR = _SCRIPT_PATH.parent
DEMO_DIR = SCRIPT_DIR.parent
PROJECT_DIR = DEMO_DIR.parent

# Reuse the exact 512-dimensional model implementation used by the matching
# personal/human checkpoint. That module changes cwd when imported, so every
# path below is resolved absolutely from this file.
from RCNN_ECA_personal_test import Mydata, RCNN, collate_fn


DEFAULT_TRAIN_POS = PROJECT_DIR / "Data" / "pos_dataset" / "human_pos_word_list.txt"
DEFAULT_TRAIN_NEG = PROJECT_DIR / "Data" / "neg_dataset" / "human_neg_word_list.txt"
DEFAULT_TEST_POS = DEMO_DIR / "test_dataset" / "pos_dataset" / "pos_word_list_human_test.txt"
DEFAULT_TEST_NEG = DEMO_DIR / "test_dataset" / "neg_dataset" / "neg_word_list_human_test.txt"
DEFAULT_MODEL = DEMO_DIR / "trained_model" / "human_1_RCNN_ECA_parallel_089-0.9802.pt"
DEFAULT_OUTPUT_DIR = DEMO_DIR / "classification_output" / "human_output"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Human RCNN-ECA classification demo")
    parser.add_argument("--train-pos", type=Path, default=DEFAULT_TRAIN_POS)
    parser.add_argument("--train-neg", type=Path, default=DEFAULT_TRAIN_NEG)
    parser.add_argument("--test-pos", type=Path, default=DEFAULT_TEST_POS)
    parser.add_argument("--test-neg", type=Path, default=DEFAULT_TEST_NEG)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_sequences(path, expected_count=None):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Sequence file not found: {path}")
    sequences = path.read_text(encoding="utf-8").splitlines()
    if expected_count is not None and len(sequences) != expected_count:
        raise ValueError(f"Expected {expected_count} sequences in {path}, found {len(sequences)}")
    if not sequences or any(not sequence.replace(" ", "") for sequence in sequences):
        raise ValueError(f"Empty sequence found in {path}")
    return sequences


def build_vocabulary(train_pos, train_neg):
    # Match the first Human training run: shuffle the full classes with seeds
    # 0/1, then enumerate residues by first occurrence in positive-then-negative
    # order, exactly as word2Num does in the existing personal classifier.
    train_pos = list(train_pos)
    train_neg = list(train_neg)
    np.random.seed(0)
    np.random.shuffle(train_pos)
    np.random.seed(1)
    np.random.shuffle(train_neg)

    vocabulary = {}
    for sequence in train_pos + train_neg:
        for residue in sequence.replace(" ", ""):
            if residue not in vocabulary:
                vocabulary[residue] = len(vocabulary) + 1
    return vocabulary


def encode_sequences(sequences, vocabulary):
    encoded = []
    for sequence_index, sequence in enumerate(sequences):
        values = []
        for residue in sequence.replace(" ", ""):
            if residue not in vocabulary:
                raise ValueError(
                    f"Unknown residue {residue!r} in test sequence {sequence_index + 1}"
                )
            values.append(vocabulary[residue])
        encoded.append(torch.tensor(values, dtype=torch.long))
    return encoded


def load_model(model_path, vocabulary, device):
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")

    expected_embedding_shape = (len(vocabulary) + 1, 512)
    actual_embedding_shape = tuple(state_dict["embedding.weight"].shape)
    if actual_embedding_shape != expected_embedding_shape:
        raise ValueError(
            f"Checkpoint embedding shape is {actual_embedding_shape}; "
            f"expected {expected_embedding_shape}"
        )

    model = RCNN(len(vocabulary) + 1, 512, 100, 1, True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def calculate_summary(y_true, y_pred, y_score):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    positives = y_pred[y_true == 1]
    negatives = y_pred[y_true == 0]
    return {
        "acc": metrics.accuracy_score(y_true, y_pred),
        "sen": float(np.mean(positives == 1)),
        "spe": float(np.mean(negatives == 0)),
        "auc": metrics.roc_auc_score(y_true, y_score),
    }


def main():
    args = parse_args()
    set_seed(1)
    device = resolve_device(args.device)

    train_pos = read_sequences(args.train_pos)
    train_neg = read_sequences(args.train_neg)
    test_pos = read_sequences(args.test_pos, expected_count=120)
    test_neg = read_sequences(args.test_neg, expected_count=120)
    vocabulary = build_vocabulary(train_pos, train_neg)
    if len(vocabulary) != 20:
        raise ValueError(f"Expected 20 amino-acid symbols, found {len(vocabulary)}")

    # Match the other four classifiers: negatives first (label 0), then
    # positives (label 1). shuffle=False guarantees one output row per sample.
    test_sequences = test_neg + test_pos
    test_labels = np.r_[np.zeros(120, dtype=np.int64), np.ones(120, dtype=np.int64)]
    test_encoded = encode_sequences(test_sequences, vocabulary)
    dataset = Mydata(test_encoded, torch.tensor(test_labels, dtype=torch.long))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model = load_model(args.model_path, vocabulary, device)

    y_true = []
    y_pred = []
    y_score = []
    with torch.inference_mode():
        for inputs, labels, lengths in loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            outputs = model(inputs, lengths)
            y_true.extend(labels.tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            y_score.extend(torch.softmax(outputs, dim=-1)[:, 1].cpu().tolist())

    if len(y_true) != 240 or set(y_true) != {0, 1}:
        raise RuntimeError("Classification did not produce 240 balanced predictions")
    summary = calculate_summary(y_true, y_pred, y_score)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roc_path = output_dir / "rcnn_ECA_human_test_roc_1.csv"
    result_path = output_dir / "result.csv"
    pd.DataFrame({"y_true": y_true, "y_score": y_score}).to_csv(
        roc_path, index=False, float_format="%.4f"
    )
    pd.DataFrame([summary], columns=["acc", "sen", "spe", "auc"]).to_csv(
        result_path, index=False, float_format="%.4f"
    )

    print(f"device={device}")
    print(f"test_pos=120 test_neg=120 predictions={len(y_true)}")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"roc_csv={roc_path}")
    print(f"result_csv={result_path}")


if __name__ == "__main__":
    main()
