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
from sklearn.preprocessing import StandardScaler
import copy
import math
# from torchsummaryX import summary

def readdata(root_dir, pos_protein_dir, neg_protein_dir,  pos_seed, neg_seed):
    pos_protein_path = os.path.join(root_dir, pos_protein_dir)
    neg_protein_path = os.path.join(root_dir, neg_protein_dir)
    with open(pos_protein_path, 'r') as f:
        pos_word_list = f.read().splitlines()
    f.close
    with open(neg_protein_path, 'r') as f:
        neg_word_list = f.read().splitlines()
    f.close
    # neg_word_list = neg_word_list[:length]  # #表示使用全部数据
    # pos_word_list = pos_word_list[:length]  # #表示使用全部数据

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
    # return pos_sequence, neg_sequence
    
def word2Num(train, test, min=0, max=None, max_features=None):
    dic = {}
    count = {}
    for singlelist in train:
        singlelist = singlelist.replace(' ', '')
        for word in singlelist:
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
    for singlelist in train:
        singlelist = singlelist.replace(' ', '')
        num = []
        for word in singlelist:
            num.append(dic.get(word))
        Num.append(num)
    print(len(Num))
    Num2 = []
    for singlelist in test:
        singlelist = singlelist.replace(' ', '')
        num2 = []
        for word in singlelist:
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
        # alpha = torch.FloatTensor([alpha])
        # self.alpha = nn.Parameter(alpha)
        self.ECABlock= ECALayer()
        # self.CABlock = ChannelAttention()
        # self.SABlock = SpatialAttention()
        # self.SEBlock = SEBlock(self.bi_num*hidden_dim + embedding_num)
        self.embedding = nn.Embedding(vocab_size, embedding_num, padding_idx=0)   # 需要添加padding_idx
        self.lstm = nn.LSTM(input_size= embedding_num, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=biFlag)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        # self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Sequential(
            nn.Dropout(dropout),

            nn.Linear(self.bi_num*hidden_dim + embedding_num, 128),
            # nn.Linear(self.bi_num*hidden_dim + embedding_num, 256),
            # nn.Linear(2 * (self.bi_num*hidden_dim + embedding_num), 128),
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
        bilstm_out, out_len = pad_packed_sequence(x, batch_first=True)
        out = torch.cat((embed, bilstm_out), 2)
        out_all = F.relu(out)
        out = out_all.permute(0, 2, 1)
        
        # out1 = self.SEBlock(out, length)
        
        out1 = self.ECABlock(out, length)
        out_feature = out + out1

        # out1 = self.CABlock(out, length)
        # out1 = self.SABlock(out1)
        # out = out + out1  # 残差结构
        
        # out = out * self.alpha+ out1  # 残差结构
        
        # out = self.SEBlock(out, length)
        # out = self.ECABlock(out, length)
        
        out = self.globalmaxpool(out_feature).squeeze()
        out = F.relu(out)
        out = self.linear(out)
        return out, out_all


def set_seed(seed):
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def create_cam(feature_maps, gradients, lengths, cam_list):
    for i in range(len(feature_maps)):
        length = lengths[i]
        feature_map = feature_maps[i][:, :length]
        gradient = gradients[i][:, :length]
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)						
        weight = np.mean(gradient, axis=1)							
        for i, w in enumerate(weight):
            cam += w * feature_map[i, :]							
        cam_list.append(cam)

def calculate_outputs_and_gradients(input, length, model, target_label_idx):
    gradient_list = []
    cam_list = []
    
    output, output_feature = model(input, length)

    if target_label_idx is None:
        target_label_idx = torch.argmax(output, 1).unsqueeze(1)
    index = torch.ones((output.size()[0], 1)).to(device) * target_label_idx
    index = index.long()
    output = output.gather(1, index)
    # clear grad

    gradient_feature = torch.autograd.grad(output, output_feature, torch.ones_like(output), True)
    gradient = gradient_feature[0]  
    gradsnp = gradient.detach().cpu().data.numpy()
    gradsnp = gradsnp.transpose(0, 2, 1)   # 调整(batch, length, channel) -> (batch, channel, length),如果已经是(b,c,l)不需要调整
    featuresnp = output_feature.detach().cpu().data.numpy()
    featuresnp = featuresnp.transpose(0, 2, 1)  # 调整(batch, length, channel) -> (batch, channel, length),如果已经是(b,c,l)不需要调整
    create_cam(featuresnp, gradsnp, length, cam_list)
    gradient_list.append(gradsnp)
    
    return gradient_list, cam_list, target_label_idx


def rescale_score_by_abs (score, max_score, min_score):
    """
    Normalize the relevance value (=score), accordingly to the extremal relevance values (max_score and min_score), 
    for visualization with a diverging colormap.
    i.e. rescale positive relevance to the range [0.5, 1.0], and negative relevance to the range [0.0, 0.5],
    using the highest absolute relevance for linear interpolation.
    """
    
    # CASE 1: positive AND negative scores occur --------------------
    if max_score>0 and min_score<0:
    
        if max_score >= abs(min_score):   # deepest color is positive
            if score>=0:
                return 0.5 + 0.5*(score/max_score)
            else:
                return 0.5 - 0.5*(abs(score)/max_score)

        else:                             # deepest color is negative
            if score>=0:
                return 0.5 + 0.5*(score/abs(min_score))
            else:
                return 0.5 - 0.5*(score/min_score)   
    
    # CASE 2: ONLY positive scores occur -----------------------------       
    elif max_score>0 and min_score>=0: 
        if max_score == min_score:
            return 1.0
        else:
            return 0.5 + 0.5*(score/max_score)
    
    # CASE 3: ONLY negative scores occur -----------------------------
    elif max_score<=0 and min_score<0: 
        if max_score == min_score:
            return 0.0
        else:
            return 0.5 - 0.5*(score/min_score)    
  
      
def getRGB (c_tuple):
    return "#%02x%02x%02x"%(int(c_tuple[0]*255), int(c_tuple[1]*255), int(c_tuple[2]*255))

     
def span_word (word, score, colormap):
    return "<span style=\"background-color:"+getRGB(colormap(score))+"\">"+word+"</span>"


def html_heatmap (words, scores, cmap_name="bwr"):
    """
    Return word-level heatmap in HTML format,
    with words being the singlelist of words (as string),
    scores the corresponding singlelist of word-level relevance values,
    and cmap_name the name of the matplotlib diverging colormap.
    """
    
    # colormap  = plt.get_cmap(cmap_name)
     
    assert len(words)==len(scores)
    max_s     = max(scores)
    min_s     = min(scores)
    
    output_text = ""
    
    for idx, w in enumerate(words):
        score       = rescale_score_by_abs(scores[idx], max_s, min_s)
        # output_text = output_text + span_word(w, score, colormap) + " "
    
    return output_text + "\n"


def sigmoid_function(x):
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig


# 计算一个蛋白质中各个氨基酸的score并统计各类氨基酸的score信息
def protein_statics(protein, gradient, protein_scorelist_dict):
    protein_score_list = []

    for idx, w in enumerate(protein):
        score = rescale_score_by_abs(gradient[idx], max(gradient), min(gradient))
        # score = sigmoid_function(gradient[idx])

        if protein_scorelist_dict.get(w) == 0:
            protein_scorelist_dict[w] = [score]
        else:
            protein_scorelist_dict.get(w).append(score)

        # protein_scorelist_dict[w] = protein_scorelist_dict.get(w) + score
        protein_score_list.append(score)
    score_np = np.array(protein_score_list)
    return score_np


# 统计每个尺度下的蛋白质信息并单独保存
def visualize_protein_gradient(protein_list, gradients, lengths, dictionary, true_label):
    for i in range(len(protein_list)):
        protein = protein_list[i]
        gradient = gradients[i][:lengths[i]]
        # display(HTML(html_heatmap(protein, gradient)))

        protein_scorelist_dict = dictionary.copy()  # 记录一个蛋白质中的每类氨基酸的score_list

        gradient_score = protein_statics(protein, gradient, protein_scorelist_dict)

        # protein_totalscore_dict = { k: np.sum(v_list) for k,v_list in protein_scorelist_dict.items()}  # 记录一个蛋白质中的每类氨基酸的总分
        protein_num_dict = { k: len(v_list) if v_list!=0 else 0 for k,v_list in protein_scorelist_dict.items()}    # 记录一个蛋白质中的每类氨基酸的个数
        protein_sort_num_dict = sorted(protein_num_dict.items(), key=lambda x: x[1], reverse=True)
        protein_sort_num_dict = { key[0]: key[1] for key in protein_sort_num_dict}
        
        protein_sort_singleacid_dict = {}  # 记录一个蛋白质中每类氨基酸的scorelist排序后前三个数据全部降序
        for k,v_list in protein_scorelist_dict.items():
            if v_list == 0:
                protein_sort_singleacid_dict[k] = [-1]*(3)
            elif len(v_list) >= 3:
                protein_sort_singleacid_dict[k] = sorted(v_list, reverse=True)[:3]
            else:
                protein_sort_singleacid_dict[k] = sorted(v_list, reverse=True)[:]+[-1]*(3-len(v_list))
        
        sorted_id = sorted(range(len(gradient_score)), key=lambda k: gradient_score[k], reverse=True)
        protein_sort_acid = np.array(list(protein))[sorted_id]
        protein_sort_score = gradient_score[sorted_id]  # 记录一个蛋白质中整个序列根据score排序后数据全部降序
        

        protein_sort_num_pd = pd.DataFrame(protein_sort_num_dict, index=['num'])
        protein_sort_singleacid_pd = pd.DataFrame(protein_sort_singleacid_dict)
        # protein_sort_singleacid_pd = pd.DataFrame.from_dict(protein_sort_singleacid_dict)
        protein_sort_protein_pd = pd.DataFrame([protein_sort_acid], index=['protein'])
        protein_sort_score_pd = pd.DataFrame([protein_sort_score], index=['score'])


        protein_split = ' '.join(protein).split()
        protein_pd = pd.DataFrame([protein_split])
        gradient_score_pd = pd.DataFrame([gradient_score])


        # 保存蛋白质的每个氨基酸和对应score
        save_dir_path = './output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/'
        if true_label[i] == 1:
            savepath = save_dir_path + 'mydata_1507_data/pos_sequence_score/RCNN_ECA_protein_score.csv'
            if not os.path.exists(os.path.dirname(savepath)):
                os.makedirs(os.path.dirname(savepath))
            protein_pd.to_csv(savepath, mode='a', header=False, index=False)
            gradient_score_pd.to_csv(savepath, mode='a', header=False, index=False, float_format='%.4f')
            # probability_pd.to_csv(savepath, mode='a', header=False, index=False, float_format='%.4f')
        elif true_label[i] == 0:  # 阴性为0
            savepath = save_dir_path + 'mydata_1507_data/neg_sequence_score/RCNN_ECA_protein_score.csv'
            if not os.path.exists(os.path.dirname(savepath)):
                os.makedirs(os.path.dirname(savepath))
            protein_pd.to_csv(savepath, mode='a', header=False, index=False)
            gradient_score_pd.to_csv(savepath, mode='a', header=False, index=False, float_format='%.4f')
            # probability_pd.to_csv(savepath, mode='a', header=False, index=False, float_format='%.4f')
        

        # 保存蛋白质的每类氨基酸score的总分、个数、平均值、标准差、按均值降序排序结果
        if true_label[i] == 1:  # 阳性为1
            savepath1 = save_dir_path + 'mydata_1507_data/pos_sequence_statics/RCNN_ECA_protein_statics.csv'
            if not os.path.exists(os.path.dirname(savepath1)):
                os.makedirs(os.path.dirname(savepath1))
            protein_pd.to_csv(savepath1, mode='a', header=False, index=False)
            protein_sort_num_pd.to_csv(savepath1, mode='a', header=True, index=True, float_format='%.4f')
            protein_sort_singleacid_pd.to_csv(savepath1, mode='a', header=True, index=True, float_format='%.4f')
            protein_sort_protein_pd.to_csv(savepath1, mode='a', header=False, index=False)
            protein_sort_score_pd.to_csv(savepath1, mode='a', header=False, index=False, float_format='%.4f')

        elif true_label[i] == 0:  # 阴性为0
            savepath1 = save_dir_path + 'mydata_1507_data/neg_sequence_statics/RCNN_ECA_protein_statics.csv'
            if not os.path.exists(os.path.dirname(savepath1)):
                os.makedirs(os.path.dirname(savepath1))
            protein_pd.to_csv(savepath1, mode='a', header=False, index=False)
            protein_sort_num_pd.to_csv(savepath1, mode='a', header=True, index=True, float_format='%.4f')
            protein_sort_singleacid_pd.to_csv(savepath1, mode='a', header=True, index=True, float_format='%.4f')
            protein_sort_protein_pd.to_csv(savepath1, mode='a', header=False, index=False)
            protein_sort_score_pd.to_csv(savepath1, mode='a', header=False, index=False, float_format='%.4f')




def num2word(input, length, dictionary):
    protein_list = []
    for i in range(input.size(0)):
        num = input[i][:length[i]]

        words = []
        for i in range(num.size(0)):
            words.append(dictionary.get(num[i].item()))
        protein = ''.join(words)
        protein_list.append(protein)
    
    return protein_list



if __name__== '__main__':
    device = torch.device("cuda")
    seed = 1
    set_seed(seed)
    root_dir = '.'
    pos_protein_dir = '../Data/processed_dataset/pos_dataset/pos_word_list_mydata_all_1507.txt'
    neg_protein_dir = '../Data/processed_dataset/neg_dataset/neg_word_list_1479.txt'
    
    pos_seed = 0
    neg_seed = 1
    train_seq,train_label = readdata(root_dir, pos_protein_dir, neg_protein_dir, pos_seed, neg_seed)
    
  
    print(len(train_seq))
    print(len(train_label))

    test_seq, test_label = train_seq.copy(), train_label.copy()
    train_num, test_num, w2n_vocab  = word2Num(train_seq, test_seq)

    n2w_vocab = {v:k for k,v in w2n_vocab.items()}

    proetin_vital_dict = {k:0 for k in w2n_vocab.keys()}

    pos_proetin_vital_dict = proetin_vital_dict.copy()
    neg_proetin_vital_dict = proetin_vital_dict.copy()

    train_data_size = len(train_num)
    test_data_size = len(test_num)
   
    
    train_ten = []
    for singlelist in train_num:
        train_ten.append(torch.LongTensor(singlelist))
    
    train_label_ten = from_numpy(train_label)
    train_label_ten = train_label_ten.type(torch.LongTensor)

    state_dict = torch.load('./saliency_model/mydata_1507_RCNN_ECA_089-0.9930.pt')
    model = RCNN(len(w2n_vocab)+1, 512, 100, 1, True)
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(model)
    
    train = Mydata(train_ten, train_label_ten) 
    train_dataloader = dataloader.DataLoader(dataset=train, batch_size=32,shuffle=False, collate_fn=collate_fn)
        
    total_labels = 0
    y_true = []
    y_pre = []
    gradients_list = []
    cams_list = []
    
    for input, label, length in train_dataloader:
        input = input.to(device)
        label = label.to(device)
        length = length.to(device)      

        protein_list = num2word(input, length, n2w_vocab)
        
        gradient_list, cam_list, pre_label = calculate_outputs_and_gradients(input, length, model, None)
        
        
        gradients_list.extend(gradient_list)
        cams_list.extend(cam_list)

        predicted_label = pre_label.reshape(pre_label.size(0))

        y_pre.extend(predicted_label.cpu())
        y_true.extend(label.cpu())
        total_labels += label.size(0)

        
        visualize_protein_gradient(protein_list, cam_list, length, proetin_vital_dict, label)
        

    train_correct = metrics.accuracy_score(y_true, y_pre)   
    print(train_correct)    

                
    
        
        
        
        
               
                
                      
                
                
            
            
            
            
            


            
