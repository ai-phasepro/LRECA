import numpy as np
import pandas as pd
import os
import csv


# FUS_train_path='./RCNN_ECA_saliency/output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/FUS_family/FUS_train_statics.csv'
# FUS_test_path='./RCNN_ECA_saliency/output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/FUS_family/FUS_test_statics.csv'
FUS_train_path='../../output/gradCAM/gradCAM_noSoftmax_outFinal_protein_score/FUS_family/FUS_train_statics.csv'
# FUS_test_path='./RCNN_ECA_saliency/output/gradCAM/gradCAM_noSoftmax_outFinal_protein_score/FUS_family/FUS_test_statics.csv'
# save_file_path = './RCNN_ECA_saliency/output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/FUS_family/FUS_train_segment_effect_1.csv'
# save_file_path = './RCNN_ECA_saliency/output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/FUS_family/FUS_test_segment_effect_1.csv'
save_file_path = '../../output/gradCAM/gradCAM_noSoftmax_outFinal_protein_score/FUS_family/FUS_train_segment_effect_1.csv'
# save_file_path = './RCNN_ECA_saliency/output/gradCAM/gradCAM_noSoftmax_outFinal_protein_score/FUS_family/FUS_test_segment_effect_1.csv'

train_proteins_scores_all = []
with open(FUS_train_path, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        train_proteins_scores_all.append(line)
train_proteins_names = train_proteins_scores_all[::5] 
train_proteins = train_proteins_scores_all[1::5]
train_scores = train_proteins_scores_all[2::5]

# train_proteins_scores_all = []
# with open(FUS_test_path, 'r') as f:
#     reader = csv.reader(f)
#     for line in reader:
#         train_proteins_scores_all.append(line)
# train_proteins_names = train_proteins_scores_all[::5][:-2]  # test去除最后两个多余蛋白质
# train_proteins = train_proteins_scores_all[1::5][:-2]  
# train_scores = train_proteins_scores_all[2::5][:-2]  


n = 10 # 序列十等分
seg_score_lists =[]
seg_effect_lists= []

for i, protein_name in enumerate(train_proteins_names):
    protein_name = protein_name[1:][0]
    protein = train_proteins[i][1:]
    score = list(map(float, train_scores[i][1:]))
    length = len(score)
    total_score = sum(score)
    seg_score_list = []
    seg_effect_list = []
    step = (int)(length/n)
    segments_score = []
    count = 0
    for j in range(0, length, step):
        count+=1
        if(count < n):
            segments_score.append(score[j:j+step])
        elif (count==n):
            segments_score.append(score[j:])
            break
    # segments_score = [score[i:i+step] if i+step<=length else score[i:] for i in range(0, length, step)]
    for segment_score in segments_score:
        seg_total_score = sum(segment_score)
        seg_effect = seg_total_score / total_score
        seg_score_list.append(seg_total_score)
        seg_effect_list.append(seg_effect)
    seg_score_lists.append(seg_score_list)
    seg_effect_lists.append(seg_effect_list)
    
    protein_pd = pd.DataFrame([protein], index=['protein'])
    score_pd = pd.DataFrame([score], index=['score'])
    seg_score_pd = pd.DataFrame([seg_score_list], index=['seg_score'])
    seg_effect_ppd = pd.DataFrame([seg_effect_list], index=['seg_effect'])

    protein_pd.to_csv(save_file_path, mode='a', header=False, index=True)
    score_pd.to_csv(save_file_path, mode='a', header=False, index=True, float_format='%.4f')
    seg_score_pd.to_csv(save_file_path,  mode='a', header=False, index=True, float_format='%.4f')
    seg_effect_ppd.to_csv(save_file_path,  mode='a', header=False, index=True, float_format='%.4f')

seg_effect_sum_list = []
all_score = sum([sum(seg_score_list) for seg_score_list in seg_score_lists])
for j in range(n):
    seg_score_sum = 0
    for i in range(len(seg_score_lists)):
        seg_score_sum+=seg_score_lists[i][j]
    seg_effect_sum = seg_score_sum / all_score
    seg_effect_sum_list.append(seg_effect_sum)
seg_effect_sum_pd = pd.DataFrame([seg_effect_sum_list], index=['total_seg_effect'])
seg_effect_sum_pd.to_csv(save_file_path,  mode='a', header=False, index=True, float_format='%.4f')

seg_score_lists_pd = pd.DataFrame(seg_score_lists, 
columns=['seg0_score','seg1_score','seg2_score','seg3_score','seg4_score','seg5_score','seg6_score','seg7_score','seg8_score','seg9_score'])
seg_effect_lists_pd = pd.DataFrame(seg_effect_lists, 
columns=['seg0_effect','seg1_effect','seg2_effect','seg3_effect','seg4_effect','seg5_effect','seg6_effect','seg7_effect','seg8_effect','seg9_effect'])
seg_score_lists_pd.to_csv(save_file_path,  mode='a', header=True, index=False, float_format='%.4f')
seg_effect_lists_pd.to_csv(save_file_path,  mode='a', header=True, index=False, float_format='%.4f')
print('success')



    
    



    