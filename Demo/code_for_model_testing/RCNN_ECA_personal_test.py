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
import random

file_dir = os.path.dirname(os.path.abspath(__file__))
print(file_dir)
os.chdir(file_dir)

def readdata(pos_protein_dir, neg_protein_dir,  pos_seed, neg_seed):
    pos_protein_path = pos_protein_dir
    neg_protein_path = neg_protein_dir
    with open(pos_protein_path, 'r') as f:
        pos_word_list = f.read().splitlines()
    f.close
    with open(neg_protein_path, 'r') as f:
        neg_word_list = f.read().splitlines()
    f.close

    np.random.seed(pos_seed)  #0/3/7/8/14/20/27/29/34/39
    np.random.shuffle(pos_word_list)  
    np.random.seed(neg_seed)  #1/4/8/9/15/21/28/30/35/40
    np.random.shuffle(neg_word_list)
    pos_sequence = pos_word_list
    neg_sequence = neg_word_list
    pos_label = np.ones(shape=(len(pos_sequence,)))
    neg_label = np.zeros(shape=(len(neg_sequence,)))
    sequence = pos_sequence + neg_sequence
    label = np.hstack((pos_label, neg_label))
    return sequence, label

def readdata_test(pos_protein_dir, neg_protein_dir):
    pos_protein_path = pos_protein_dir
    neg_protein_path = neg_protein_dir
    with open(pos_protein_path, 'r') as f:
        pos_word_list = f.read().splitlines()
    f.close
    with open(neg_protein_path, 'r') as f:
        neg_word_list = f.read().splitlines()
    f.close

    pos_sequence = pos_word_list
    neg_sequence = neg_word_list
    return pos_sequence, neg_sequence

def readverifydata(verify_protein_path):
    verify_data = pd.read_excel(verify_protein_path,header=None)
    sequence = verify_data.iloc[:, 1].values.ravel()
    name = verify_data.iloc[:, 0].values.ravel()
    label = np.ones(shape=(sequence.shape))
    print('Sequence', sequence.shape[0])
    verify_seq = []
    for i in range(sequence.shape[0]):
        cur_s = ''.join(sequence[i])
        cur_s = cur_s.lower()
        cur_s = cur_s.strip()#去除空格符
        verify_divided_in_word = ' '.join(cur_s)
        verify_seq.append(verify_divided_in_word)
    return name, verify_seq, label
    
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

def collate_fn1(data):    
    data.sort(key=lambda tuple: len(tuple[0]), reverse=True)
    data_length = [len(tuple[0]) for tuple in data]
    data_ten, data_label, data_name = [], [], []
    for tuple in data:
        data_ten.append(tuple[0])
        data_label.append(tuple[1])
        data_name.append(tuple[2])
    data_ten = pad_sequence(data_ten, batch_first=True,padding_value=0)
    data_label = torch.LongTensor(data_label)
    data_name = torch.LongTensor(data_name)
    data_length = torch.LongTensor(data_length)
    return data_ten, data_label, data_length, data_name  
    

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
    
class Mydata_test(dataset.Dataset):
    def __init__(self, data, label, name):
        self.data = data
        self.label = label
        self.name = name
    def __getitem__(self, idx):
        protein = self.data[idx]
        label = self.label[idx]
        name = self.name[idx]
        return protein, label, name
    def __len__(self):
        assert len(self.data)==len(self.label)
        return len(self.data)


class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=5): # 3 5
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
        self.embedding = nn.Embedding(vocab_size, embedding_num, padding_idx=0)   # 需要添加padding_idx
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
    parser.add_argument('--pos_protein_dir', type=str, default = '../../Data/pos_dataset/pos_word_list_mydata_all_1507.txt')
    parser.add_argument('--neg_protein_dir', type=str, default = '../../Data/neg_dataset/neg_word_list_1479.txt')
    parser.add_argument('--pos_test_dir', type=str, default = '../test_dataset/2.xlsx')
    parser.add_argument('--neg_test_dir', type=str, default = '../test_dataset/2.xlsx')
    parser.add_argument('--model_path', type=str, default = '../trained_model/mydata_1507_RCNN_ECA_089-0.9930.pt')
    args = parser.parse_args()
    
    device = torch.device("cpu")
    seed = 1
    set_seed(seed)
    pos_protein_dir = args.pos_protein_dir
    neg_protein_dir = args.neg_protein_dir
    pos_test_dir = args.pos_test_dir
    neg_test_dir = args.neg_test_dir
    model_path = args.model_path

    pos_seed = 0
    neg_seed = 1
    
    test_name, test_seq, test_label = readverifydata(pos_test_dir)
    train_seq,train_label = readdata(pos_protein_dir, neg_protein_dir, pos_seed, neg_seed)

    # if not os.path.exists("../classification_output/personal_output"):
    #     os.makedirs("../classification_output/personal_output")
    # auc_save_csv = '../classification_output/personal_output/personal_test_roc_{}.csv'.format((i+1))
    result_save_csv = '../classification_output/personal_output/result.csv'
    # df_test = pd.DataFrame(columns=['y_true', 'y_score'])
    # df_test.to_csv(auc_save_csv, mode='w', index=False)   
    df_test = pd.DataFrame(columns=['acc', 'sen', 'spe', 'auc'])
    df_test.to_csv(result_save_csv, index=False)

    print(len(test_seq))
    print(len(test_label))

    train_num, test_num, w2n_vocab  = word2Num(train_seq, test_seq)

    n2w_vocab = {v:k for k,v in w2n_vocab.items()}
    proetin_vital_dict = {k:0 for k in w2n_vocab.keys()}
    pos_proetin_vital_dict = proetin_vital_dict.copy()
    neg_proetin_vital_dict = proetin_vital_dict.copy()
           
    train_data_size = len(train_num)
    test_data_size = len(test_num) 
    
    test_ten = []
    for singlelist in test_num:
        test_ten.append(torch.LongTensor(singlelist))
        
    test_name_ten = from_numpy(np.array(range(len(test_name))))
    
    test_label_ten = from_numpy(test_label)
    test_label_ten = test_label_ten.type(torch.LongTensor)
    test_name_ten = test_name_ten.type(torch.LongTensor)
            
    state_dict = torch.load(model_path)
    model = RCNN(len(w2n_vocab)+1, 512, 100, 1, True) 
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
            
    test = Mydata_test(test_ten, test_label_ten, test_name_ten)

    test_dataloader = dataloader.DataLoader(dataset=test, batch_size=32,shuffle=False, collate_fn=collate_fn1)
        
    test_loss = 0
    y_true_test = []
    y_pre_test = []
    y_score_test = []
    y_name_test = []
    total_labels_test = 0
    
    with torch.no_grad():
        for input, label, length, name in test_dataloader:
            input = input.to(device)
            label = label.to(device)
            length = length.to(device)

            output = model(input, length)
            _, predicted = torch.max(output,1)
            y_pre_test.extend(predicted.cpu())
            y_true_test.extend(label.cpu())
            y_name_test.extend(name)
            y_score_test.extend(torch.softmax(output, dim=-1)[:, 1].cpu().detach())
            total_labels_test += label.size(0)
                    
            test_correct = metrics.accuracy_score(y_true_test, y_pre_test)
            test_F1 = metrics.f1_score(y_true_test, y_pre_test, average='macro')
            test_R = metrics.recall_score(y_true_test, y_pre_test)
            test_precision = metrics.precision_score(y_true_test, y_pre_test)
                    
        if not os.path.exists("../classification_output/personal_output"):
                os.makedirs("../classification_output/personal_output")
        auc_save_csv = '../classification_output/personal_output/personal_test_roc.csv'
        df_test = pd.DataFrame(columns=['Name', 'Seq', 'y_true', 'y_score', 'y_pre'])
        df_test.to_csv(auc_save_csv,mode='w', index=False)    # rcnn_2使用全部数据， rcnn_1使用±668数据
        y_true_data = [i.item() for i in y_true_test]
        y_score_data = [i.item() for i in y_score_test]
        y_pre_data = [i.item() for i in y_pre_test]
        y_name_data = [i.item() for i in y_name_test]
        auc_dict = {'Name': [test_name[i] for i in y_name_data], 'Seq': [test_seq[i] for i in y_name_data] , 'y_true':y_true_data, 'y_score':y_score_data, 'y_pre': np.array(y_pre_data)}
        auc_score = pd.DataFrame(auc_dict)
        auc_score.to_csv(auc_save_csv, mode='a', header=False, index=False, float_format='%.4f')
