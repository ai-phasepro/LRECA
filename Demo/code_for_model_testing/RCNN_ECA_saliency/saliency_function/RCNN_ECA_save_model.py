import os
import torch
from torch.utils.data import dataset, dataloader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
from torch import LongTensor, Tensor, from_numpy, max_pool1d, nn, unsqueeze,optim
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import copy

def readdata(pos_protein_dir, neg_protein_dir,  pos_seed, neg_seed):
    pos_protein_path = pos_protein_dir
    neg_protein_path = neg_protein_dir
    with open(pos_protein_path, 'r') as f:
        pos_word_list = f.read().splitlines()
    f.close
    with open(neg_protein_path, 'r') as f:
        neg_word_list = f.read().splitlines()
    f.close

    np.random.seed(pos_seed)  
    np.random.shuffle(pos_word_list)  
    np.random.seed(neg_seed)  
    np.random.shuffle(neg_word_list)
    pos_sequence = pos_word_list
    neg_sequence = neg_word_list
    pos_label = np.ones(shape=(len(pos_sequence,)))
    neg_label = np.zeros(shape=(len(neg_sequence,)))
    sequence = pos_sequence + neg_sequence
    label = np.hstack((pos_label, neg_label))
    return sequence, label
    
def word2Num(train, test, min=0, max=None, max_features=None):
    dic = {}
    count = {}
    for list in train:
        list = list.replace(' ', '')
        for word in list:
            count[word] = count.get(word, 0) + 1
    if min is not None:
        count = {word:value for word,value in count.items() if value>min}
    if max is not None:
        count = {word:value for word,value in count.items() if value<max}
    if  max_features is not None:
        temp = sorted(count.items(), key=lambda x:x[-1], reverse=True)[:max_features]
        count = dict(temp)
    for word in count:
        dic[word] = len(dic) + 1
    print(dic)
    Num = []
    for list in train:
        list = list.replace(' ', '')
        num = []
        for word in list:
            num.append(dic.get(word))
        Num.append(num)
    print(len(Num))
    Num2 = []
    for list in test:
        list = list.replace(' ', '')
        num2 = []
        for word in list:
            num2.append(dic.get(word))
        Num2.append(num2)
    print(len(Num2))  
    return Num, Num2, dic        


def collate_fn(data):    
    data.sort(key=lambda tuple: len(tuple[0]), reverse=True)
    data_length = [len(tuple[0]) for tuple in data]
    data_ten, data_label = [], []
    for tuple in data:
        data_ten.append(tuple[0])
        data_label.append(tuple[1])
    data_ten = pad_sequence(data_ten, batch_first=True,padding_value=0)
    data_label = torch.LongTensor(data_label)
    data_length = torch.LongTensor(data_length)
    return data_ten, data_label, data_length   
    

class Mydata(dataset.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __getitem__(self, idx):
        protein = self.data[idx]
        label = self.label[idx]
        return protein, label
    def __len__(self):
        assert len(self.data)==len(self.label)
        return len(self.data)


class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=5): # 3
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        # feature descriptor on the global spatial information
        b, e, t = x.size()
        
        for i in range(b):
            x_pack = x[i][: , : length[i]].unsqueeze(0)
            x_avg = self.avg_pool(x_pack)
            if i == 0:
                y = x_avg.clone()
            else:
                y = torch.cat((y, x_avg), dim=0)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, x, length):
        b, e , t = x.size()
        # Squeeze
        # y = self.avg_pool(x).view(b, e)
        for i in range(b):
            x_pack = x[i][: , : length[i]].unsqueeze(0)
            x_avg = self.avg_pool(x_pack)
            if i == 0:
                y = x_avg.clone()
            else:
                y = torch.cat((y, x_avg), dim=0)
        # Excitation
        y = self.fc(y.squeeze()).view(b, e, 1)
        # Fscale
        y = torch.mul(x, y)
        return y


class ChannelAttention(nn.Module):

    def __init__(self, k_size=5): # 3
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        # feature descriptor on the global spatial information
        b, e, t = x.size()
        
        for i in range(b):
            x_pack = x[i][: , : length[i]].unsqueeze(0)
            x_avg = self.avg_pool(x_pack)
            if i == 0:
                y_avg = x_avg.clone()
            else:
                y_avg = torch.cat((y_avg, x_avg), dim=0)
        
        y_max = self.max_pool(x)
        
        avg_out = self.conv(y_avg.transpose(-1, -2)).transpose(-1, -2)
        max_out = self.conv(y_max.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        out = self.sigmoid(avg_out + max_out)

        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv1d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, e, t = x.size()
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avgout, maxout], dim=1)
        y = self.conv(y)
        
        out = self.sigmoid(y)
        return x * out.expand_as(x)


class RCNN(nn.Module):
    def __init__(self, vocab_size, embedding_num, hidden_dim, num_layers, biFlag, dropout=0.2):
        super(RCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_num = embedding_num
        self.hiddern_dim = hidden_dim
        self.num_layers = num_layers
        if biFlag:
            self.bi_num = 2
        else:
            self.bi_num = 1
        self.biFlag = biFlag
        self.device = torch.device("cuda")
        self.ECABlock= ECALayer()
        self.embedding = nn.Embedding(vocab_size, embedding_num, padding_idx=0)   
        self.lstm = nn.LSTM(input_size= embedding_num, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=biFlag)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Sequential(
            nn.Dropout(dropout),

            nn.Linear(self.bi_num*hidden_dim + embedding_num, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,32), 
            nn.ReLU(),
            nn.Linear(32,2)
        )
        
    def forward(self, x, length):
        embed = self.embedding(x)
        x = pack_padded_sequence(embed, length.cpu(), batch_first=True)
        x, (ht,ct) = self.lstm(x)
        out, out_len = pad_packed_sequence(x, batch_first=True)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        
        out1 = self.ECABlock(out, length)
        out = out + out1
        
        out = self.globalmaxpool(out).squeeze()
        out = F.relu(out)
        out = self.linear(out)
        return out


def set_seed(seed):
    torch.manual_seed(seed)          
    torch.cuda.manual_seed(seed)      
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default = '../../../saliency_model')
    parser.add_argument('--pos_protein_dir', type=str, default = '../../../../Data/pos_dataset/pos_word_list_mydata_all_1507.txt')
    parser.add_argument('--neg_protein_dir', type=str, default = '../../../../Data/neg_dataset/neg_word_list_1479.txt')
    parser.add_argument('--learning_rate', type=float, default = 1e-3)
    parser.add_argument('--epoch', type=int, default = 89)
    args = parser.parse_args()
    
    device = torch.device("cuda")
    seed = 1
    set_seed(seed)
    pos_protein_dir = args.pos_protein_dir
    neg_protein_dir = args.neg_protein_dir
    save_path = args.save_path
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pos_seed = 0
    neg_seed = 1
    train_seq,train_label = readdata(pos_protein_dir, neg_protein_dir, pos_seed, neg_seed)

    print(len(train_seq))
    print(len(train_label))

    test_seq, test_label = train_seq.copy(), train_label.copy()
    train_num, test_num, vocab  = word2Num(train_seq, test_seq)
    train_data_size = len(train_num)
    test_data_size = len(test_num)

    train_ten = []
    for list in train_num:
        train_ten.append(torch.LongTensor(list))
    
    train_label_ten = from_numpy(train_label)
    train_label_ten = train_label_ten.type(torch.LongTensor)
    
    rcnn = RCNN(len(vocab)+1, 512, 100, 1, True)  
    rcnn = rcnn.to(device)
    print(rcnn)
    loss_fn = nn.CrossEntropyLoss()
    
    loss_fn = loss_fn.to(device)
    learning_rate = args.learning_rate
    optimizer = optim.Adam(rcnn.parameters(), lr=learning_rate, betas=(0.9,0.99))
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    
    train = Mydata(train_ten, train_label_ten)   
    set_seed(seed)
    train_dataloader = dataloader.DataLoader(dataset=train, batch_size=32,shuffle=True, collate_fn=collate_fn)

    epoch = args.epoch   
    
    for i in range(epoch):
        print("-------第 {} 轮训练开始-------".format(i+1))
            
        rcnn.train()
        total_labels = 0
        train_loss = 0.0
        y_true = []
        y_pre = []
        y_score = []
        for input, label, length in train_dataloader:
            input = input.to(device)
            label = label.to(device)
            length = length.to(device)
                
            output = rcnn(input, length)
                
            loss = loss_fn(output, label)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            train_loss += loss.item() * label.size(0)
            _, predicted = torch.max(output, 1)
            y_pre.extend(predicted.cpu())
            y_true.extend(label.cpu())
            y_score.extend(torch.softmax(output, dim=-1)[:,1].cpu().detach())
            total_labels += label.size(0)
            
        train_loss /= total_labels
        train_correct = metrics.accuracy_score(y_true, y_pre)
        train_F1 = metrics.f1_score(y_true, y_pre, average='macro')
        train_R = metrics.recall_score(y_true, y_pre)
        train_precision = metrics.precision_score(y_true, y_pre)
        train_auc = metrics.roc_auc_score(y_true, y_score)
        save_content = 'Train: Correct: %.5f, Precision: %.5f, R: %.5f, F1(macro): %.5f, AUC:%.5f, train_loss: %f' % \
                        (train_correct, train_precision, train_R, train_F1, train_auc, train_loss)
        print(save_content)
    
    save_model = rcnn
    save_name = save_path + 'mydata_1507_RCNN_ECA_parallel_{:03d}-{:.4f}.pt'.format((i+1), train_correct)
    torch.save(save_model.state_dict(),save_name)