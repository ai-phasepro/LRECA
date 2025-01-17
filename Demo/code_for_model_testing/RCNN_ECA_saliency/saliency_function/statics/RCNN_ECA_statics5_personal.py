import numpy as np
import pandas as pd
import os
import csv

file_dir = os.path.dirname(os.path.abspath(__file__))
print(file_dir)
os.chdir(file_dir)

FUS_test_path='../../../../Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/personal/test_statics.csv'
save_file_path = '../../../../Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/personal/acid_avgscore.csv'
save_count_path = '../../../../Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/personal/acid_count.csv'
save_sum_path = '../../../../Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/personal/acid_sum.csv'

train_proteins_scores_all = []
with open(FUS_test_path, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        train_proteins_scores_all.append(line)
train_proteins_names = train_proteins_scores_all[::5] 
train_proteins = train_proteins_scores_all[1::5]
train_scores = train_proteins_scores_all[2::5]

acid_scoreList_dict = {}
acid_countList_dict = {}

for i, protein_name in enumerate(train_proteins_names):
    protein_name = protein_name[1:][0]
    protein = train_proteins[i][1:]
    score = list(map(float, train_scores[i][1:]))
    for j, acid in enumerate(protein):
        acid_score = score[j]
        if acid_scoreList_dict.get(acid) == None:
            acid_scoreList_dict[acid] = [acid_score]
        else:
            acid_scoreList_dict[acid].append(acid_score)
            
# count
for i, protein_name in enumerate(train_proteins_names):
    protein_name = protein_name[1:][0]
    protein = train_proteins[i][1:]
    for j, acid in enumerate(protein):
        if acid_countList_dict.get(acid) == None:
            acid_countList_dict[acid] = [1]
        else:
            acid_countList_dict[acid].append(1)

df_test = pd.DataFrame(columns=['acid', 'avg_score'])
df_test.to_csv(save_file_path, index=False)
for key, val in acid_scoreList_dict.items():
    val = np.mean(val)
    acid_score_dict = {'acid':[key], 'avg_score':[val]}
    acid_score_pd = pd.DataFrame(acid_score_dict)
    acid_score_pd.to_csv(save_file_path,  mode='a', header=False, index=False, float_format='%.4f')
acid_score = pd.read_csv(save_file_path)
acid_score = acid_score.sort_values(by=['avg_score'],ascending=[False])
print(acid_score)
os.remove(save_file_path)
df_test = pd.DataFrame(columns=['acid', 'avg_score'])
df_test.to_csv(save_file_path, index=False)
acid_score.to_csv(save_file_path, index=False)

df_test = pd.DataFrame(columns=['acid', 'count'])
df_test.to_csv(save_count_path, index=False)
for key, val in acid_countList_dict.items():
    val = np.sum(val)
    acid_count_dict = {'acid':[key], 'count':[val]}
    acid_count_pd = pd.DataFrame(acid_count_dict)
    acid_count_pd.to_csv(save_count_path,  mode='a', header=False, index=False, float_format='%.4f')
acid_count = pd.read_csv(save_count_path)
acid_count = acid_count.sort_values(by=['count'],ascending=[False])
print(acid_count)
os.remove(save_count_path)
df_test = pd.DataFrame(columns=['acid', 'count'])
df_test.to_csv(save_count_path, index=False)
acid_count.to_csv(save_count_path, index=False)

df_test = pd.DataFrame(columns=['acid', 'sum_score'])
df_test.to_csv(save_sum_path, index=False)
for key, val in acid_scoreList_dict.items():
    val = np.sum(val)
    acid_score_dict = {'acid':[key], 'sum_score':[val]}
    acid_score_pd = pd.DataFrame(acid_score_dict)
    acid_score_pd.to_csv(save_sum_path,  mode='a', header=False, index=False, float_format='%.4f')
acid_score = pd.read_csv(save_sum_path)
acid_score = acid_score.sort_values(by=['sum_score'],ascending=[False])
print(acid_score)
os.remove(save_sum_path)
df_test = pd.DataFrame(columns=['acid', 'sum_score'])
df_test.to_csv(save_sum_path, index=False)
acid_score.to_csv(save_sum_path, index=False)
print('success')