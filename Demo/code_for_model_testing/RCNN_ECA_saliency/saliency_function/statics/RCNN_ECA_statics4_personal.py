import numpy as np
import pandas as pd
import os
import csv


if __name__== '__main__':
    dir_path = '../../../../Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/'

    protein_class_path = dir_path + 'personal/'
    protein_file_name = 'pos_sequence_score.csv'
    
    protein_file_path = os.path.join(protein_class_path, protein_file_name)
    proteins_scores_all = []
    with open(protein_file_path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            proteins_scores_all.append(line)

    proteins = proteins_scores_all[::2]
    scores = proteins_scores_all[1::2]

    save_file_path = protein_class_path + '/segment_effect.csv'
    n = 10 # 序列十等分
    seg_score_lists =[]
    seg_effect_lists= []
    for i, protein in enumerate(proteins):
        score = list(map(float, scores[i]))
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
        seg_effect_pd = pd.DataFrame([seg_effect_list], index=['seg_effect'])

        protein_pd.to_csv(save_file_path, mode='a', header=False, index=True)
        score_pd.to_csv(save_file_path, mode='a', header=False, index=True, float_format='%.4f')
        seg_score_pd.to_csv(save_file_path,  mode='a', header=False, index=True, float_format='%.4f')
        seg_effect_pd.to_csv(save_file_path,  mode='a', header=False, index=True, float_format='%.4f')     
        
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