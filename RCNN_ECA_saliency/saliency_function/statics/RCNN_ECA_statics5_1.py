import numpy as np
import pandas as pd
import os
import csv


# FUS_train_path='./RCNN_ECA_saliency/output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/FUS_family/FUS_train_statics.csv'
# FUS_test_path='./RCNN_ECA_saliency/output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/FUS_family/FUS_test_statics.csv'
# FUS_train_path='./RCNN_ECA_saliency/output/gradCAM/gradCAM_noSoftmax_outFinal_protein_score/FUS_family/FUS_train_statics.csv'
FUS_test_path='../../output/gradCAM/gradCAM_noSoftmax_outFinal_protein_score/FUS_family/FUS_test_statics.csv'
# save_file_path = './RCNN_ECA_saliency/output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/FUS_family/FUS_train_acid_score.csv'
# save_file_path = './RCNN_ECA_saliency/output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/FUS_family/FUS_test_acid_score.csv'
# save_file_path = './RCNN_ECA_saliency/output/gradCAM/gradCAM_noSoftmax_outFinal_protein_score/FUS_family/FUS_train_acid_score.csv'
save_file_path = '../../output/gradCAM/gradCAM_noSoftmax_outFinal_protein_score/FUS_family/FUS_test_acid_score.csv'

# train_proteins_scores_all = []
# with open(FUS_train_path, 'r') as f:
#     reader = csv.reader(f)
#     for line in reader:
#         train_proteins_scores_all.append(line)
# train_proteins_names = train_proteins_scores_all[::5] 
# train_proteins = train_proteins_scores_all[1::5]
# train_scores = train_proteins_scores_all[2::5]

train_proteins_scores_all = []
with open(FUS_test_path, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        train_proteins_scores_all.append(line)
train_proteins_names = train_proteins_scores_all[::5][:-2]  # test去除最后两个多余蛋白质
train_proteins = train_proteins_scores_all[1::5][:-2]  
train_scores = train_proteins_scores_all[2::5][:-2]  


acid_scoreList_dict = {}

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
for key, val in acid_scoreList_dict.items():
    acid_score_pd = pd.DataFrame([val], index=[key])
    acid_score_pd.to_csv(save_file_path,  mode='a', header=False, index=True, float_format='%.4f')
print('success')



    
    



    