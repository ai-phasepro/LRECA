import os
import torch
from torch.utils.data import dataset, dataloader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
from torch import LongTensor, Tensor, from_numpy, max_pool1d, nn, unsqueeze,optim
import argparse
#from torchnlp.encoders.texts import StaticTokenizerEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import copy
# from torchsummaryX import summary

def readdata(root_dir, pos_protein_dir, neg_protein_dir, length, pos_seed, neg_seed):
    pos_protein_path = os.path.join(root_dir, pos_protein_dir)
    neg_protein_path = os.path.join(root_dir, neg_protein_dir)
    with open(pos_protein_path, 'r') as f:
        pos_word_list = f.read().splitlines()
    f.close
    with open(neg_protein_path, 'r') as f:
        neg_word_list = f.read().splitlines()
    f.close
    neg_word_list = neg_word_list[:length]  # #表示使用全部数据
    pos_word_list = pos_word_list[:length]  # #表示使用全部数据

    np.random.seed(pos_seed)  #0/3/7/8/14/20/27/29/34/39
    np.random.shuffle(pos_word_list)  
    np.random.seed(neg_seed)  #1/4/8/9/15/21/28/30/35/40
    np.random.shuffle(neg_word_list)
    pos_sequence = pos_word_list
    neg_sequence = neg_word_list
    # pos_label = np.ones(shape=(len(pos_sequence,)))
    # neg_label = np.zeros(shape=(len(neg_sequence,)))
    # sequence = pos_sequence + neg_sequence
    # label = np.hstack((pos_label, neg_label))
    # return sequence, label
    return pos_sequence, neg_sequence

def readdata_test(root_dir, pos_protein_dir, neg_protein_dir):
    pos_protein_path = os.path.join(root_dir, pos_protein_dir)
    neg_protein_path = os.path.join(root_dir, neg_protein_dir)
    with open(pos_protein_path, 'r') as f:
        pos_word_list = f.read().splitlines()
    f.close
    with open(neg_protein_path, 'r') as f:
        neg_word_list = f.read().splitlines()
    f.close

    pos_sequence = pos_word_list
    neg_sequence = neg_word_list
    return pos_sequence, neg_sequence

def writedata(root_dir, pos_test_dir, neg_test_dir, pos_sequence, neg_sequence):
    pos_protein_path = os.path.join(root_dir, pos_test_dir)
    neg_protein_path = os.path.join(root_dir, neg_test_dir)
    
    with open(pos_protein_path, 'w') as f:
        f.write('\n'.join(pos_sequence))
    f.close
    with open(neg_protein_path, 'w') as f:
        f.write('\n'.join(neg_sequence))
    f.close
    
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
    # a1, a2 = [], []
    # for list in train:
    #     list = list.replace(' ', '')
    #     a1.append(len(list))
    # for num in Num:
    #     a2.append(len(num))
    # print(a1 == a2)    
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
        # y = self.avg_pool(x).view(b,e,1)

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
        # y = self.avg_pool(x).view(b,e,1)
        
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
        self.embedding = nn.Embedding(vocab_size, embedding_num, padding_idx=0)   # 需要添加padding_idx
        self.lstm = nn.LSTM(input_size= embedding_num, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=biFlag)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Sequential(
            nn.Dropout(dropout),

            nn.Linear(self.bi_num*hidden_dim + embedding_num, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,32),  # 32
            # nn.Linear(256,2)
            nn.ReLU(),
            nn.Linear(32,2)  # 32
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
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__== '__main__':
    device = torch.device("cuda")
    seed = 1
    set_seed(seed)
    root_dir =  '../Data'
    pos_protein_dir = 'pos_dataset/pos_word_list_PhasepDB_high_throughput.txt'
    neg_protein_dir = 'neg_dataset/neg_word_list.txt'
    pos_test_dir = 'test_dataset/pos_dataset/pos_word_list_high_test.txt'
    neg_test_dir = 'test_dataset/neg_dataset/neg_word_list_high_test.txt'
    save_dir = './save_model_high'
    save_path = os.path.join(root_dir, save_dir)
    model_path = 'trained_model/model_high_2.pt'
    list_length = 1490 # pos:253, 592, 4644, 668, neg:1490


    # PhasepDB_high_throughput seed
    pos_seed_list = [20]           
    neg_seed_list = [21]


    # 十次实验
    for i in range(len(pos_seed_list)):
        pos_seed = pos_seed_list[i]
        neg_seed = neg_seed_list[i]
        pos_test_sequence,neg_test_sequence = readdata_test("./", pos_test_dir, neg_test_dir)
        pos_sequence, neg_sequence = readdata(root_dir, pos_protein_dir, neg_protein_dir, list_length, pos_seed, neg_seed)

        if not os.path.exists("./classification_output/dataset_RCNN_ECA_output/PhasepDB_high_throughput_output/RCNN_ECA_em1024_hidden128_128_32_output"):
            os.makedirs("./classification_output/dataset_RCNN_ECA_output/PhasepDB_high_throughput_output/RCNN_ECA_em1024_hidden128_128_32_output")
        auc_save_csv = './classification_output/dataset_RCNN_ECA_output/PhasepDB_high_throughput_output/RCNN_ECA_em1024_hidden128_128_32_output/rcnn_ECA_PDB_high_epoch100_roc_{}.csv'.format((i+1))
        result_save_csv = './classification_output/dataset_RCNN_ECA_output/PhasepDB_high_throughput_output/RCNN_ECA_em1024_hidden128_128_32_output/result.csv'
        df_test = pd.DataFrame(columns=['y_true', 'y_score'])
        df_test.to_csv(auc_save_csv, index=False)    # rcnn_2使用全部数据， rcnn_1使用±668数据
        df_test = pd.DataFrame(columns=['acc', 'sen', 'spe', 'auc'])
        df_test.to_csv(result_save_csv, index=False)

        
        neg_num = len(neg_test_sequence)
        pos_num = len(pos_test_sequence)
        print('pos_num=',pos_num)  # 253,592,1490,668,1507
        print('neg_num=',neg_num)  # 253,592,1490,668,1479
        
        neg_num = len(neg_sequence)
        pos_num = len(pos_sequence)

        start = 0.1
        interval = 0.1
        val_split = 0.1 #验证集占训练集比例
        
        total_tp = 0
        total_p = 0
        total_n = 0
        total_tn = 0

        save_sen = []
        save_spe = []
        save_acc = []

        fold = 0
        total_correct, total_F1, total_R, total_precision = [],[],[],[]


        test_pos_seq = pos_test_sequence
        test_neg_seq = neg_test_sequence
        test_pos_seq, test_neg_seq = readdata_test(root_dir, pos_test_dir, neg_test_dir)
        
        train_val_pos_seq = pos_sequence[:int(pos_num*start)] + pos_sequence[int(pos_num*(start+interval)):]
        train_val_neg_seq = neg_sequence[:int(neg_num * start)] + neg_sequence[int(neg_num * (start + interval)):]
        train_val_pos_num = len(train_val_pos_seq)  # 602
        train_val_neg_num = len(train_val_neg_seq)  # 602 
        set_seed(seed)  # 2022-7-2新加入
        np.random.shuffle(train_val_pos_seq)
        np.random.shuffle(train_val_neg_seq)
        val_pos_seq = train_val_pos_seq[:int(train_val_pos_num*val_split)]
        train_pos_seq = train_val_pos_seq[int(train_val_pos_num*val_split):]
        val_neg_seq = train_val_neg_seq[:int(train_val_neg_num*val_split)]
        train_neg_seq = train_val_neg_seq[int(train_val_neg_num*val_split):]

        # writedata(root_dir, pos_test_dir, neg_test_dir, test_pos_seq, test_neg_seq)

        test_y = np.hstack((np.zeros(shape=(len(test_neg_seq), )),
                        np.ones(shape=(len(test_pos_seq), ))))  # 66*2

        print('test_pos', test_y[test_y == 1].shape)     # 66 
        print('test_neg', test_y[test_y == 0].shape)     # 66

        train_seq = train_neg_seq + train_pos_seq
        val_seq = val_neg_seq + val_pos_seq
        train_val_seq = train_seq + val_seq
        test_seq = test_neg_seq + test_pos_seq

        _, test_num, vocab  = word2Num(train_seq, test_seq)
        test_data_size = len(test_num)
            
            
        test_ten = []
        for list in test_num:
            test_ten.append(torch.LongTensor(list))
            
        test_label_ten = from_numpy(test_y)
        test_label_ten = test_label_ten.type(torch.LongTensor)
            
        state_dict = torch.load(model_path)
        rcnn = RCNN(len(vocab)+1, 1024, 128, 1, True)  # 256,100,1  hidden128:256,128,1效果较差
        rcnn = rcnn.to(device)
        rcnn.load_state_dict(state_dict)
        rcnn.eval()
        print(rcnn)
            
        test = Mydata(test_ten, test_label_ten)
            

        set_seed(seed)
        test_dataloader = dataloader.DataLoader(dataset=test, batch_size=32,shuffle=True, collate_fn=collate_fn)
        
        test_loss = 0
        y_true_test = []
        y_pre_test = []
        y_score_test = []
        total_labels_test = 0
        with torch.no_grad():
            for input, label, length in test_dataloader:
                input = input.to(device)
                label = label.to(device)
                length = length.to(device)

                output = rcnn(input, length)
                _, predicted = torch.max(output,1)
                y_pre_test.extend(predicted.cpu())
                y_true_test.extend(label.cpu())
                y_score_test.extend(torch.softmax(output, dim=-1)[:, 1].cpu().detach())
                total_labels_test += label.size(0)
                    
                test_correct = metrics.accuracy_score(y_true_test, y_pre_test)
                test_F1 = metrics.f1_score(y_true_test, y_pre_test, average='macro')
                test_R = metrics.recall_score(y_true_test, y_pre_test)
                test_precision = metrics.precision_score(y_true_test, y_pre_test)
                test_auc = metrics.roc_auc_score(y_true_test, y_score_test)

                save_content = 'Test: Correct: %.5f, Precision: %.5f, R: %.5f, F1(macro): %.5f, AUC:%.5f, test_loss: %f' % \
                            (test_correct, test_precision, test_R, test_F1, test_auc, test_loss)
                print(save_content)
                    
                y_true_data = [i.item() for i in y_true_test]
                y_score_data = [i.item() for i in y_score_test]
                y_pre_data = [i.item() for i in y_pre_test]
                auc_dict = {'y_true':y_true_data, 'y_score':y_score_data}
                auc_score = pd.DataFrame(auc_dict)
                auc_score.to_csv(auc_save_csv, mode='a', header=False, index=False, float_format='%.4f')
                    
            p = np.array(y_pre_data)[np.array(y_true_data) == 1]
            tp = p[p == 1]
            n = np.array(y_pre_data)[np.array(y_true_data) == 0]
            tn = n[n == 0]
                    
            sen = tp.shape[0] / p.shape[0] if p.shape[0] > 0 else 1
            spe = tn.shape[0] / n.shape[0] if n.shape[0] > 0 else 1
            acc = (tp.shape[0] + tn.shape[0]) / (p.shape[0] + n.shape[0])
            auc = metrics.roc_auc_score(y_true_data, y_score_data)
            print('sen:', sen)
            print('spe:', spe)
            print('acc:', acc)
            print('auc:', auc)
                
            list1 = [test_loss, test_correct, test_F1, test_R, test_precision, (tn.shape[0] / n.shape[0] if n.shape[0] > 0 else 1), test_auc]
            test_dict = {'acc':[acc], 'sen':[sen], 'spe':[spe], 'auc':[auc]}
            list.extend(list1)
            data_test = pd.DataFrame([list])
            test_score = pd.DataFrame(test_dict)
            test_score.to_csv(result_save_csv, mode='a', header=False, index=False, float_format='%.4f')