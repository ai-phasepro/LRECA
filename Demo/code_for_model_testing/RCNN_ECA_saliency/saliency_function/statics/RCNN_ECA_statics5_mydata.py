import numpy as np
import pandas as pd
import os
import csv


if __name__== '__main__':
    dir_path = 'Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/'

    protein_class = 'pos_sequence_score'

    protein_class_path = dir_path + 'mydata_1507_data/'+ protein_class
    protein_file_name = 'RCNN_ECA_protein_score.csv'
    
    protein_file_path = os.path.join(protein_class_path, protein_file_name)
    proteins_scores_all = []
    with open(protein_file_path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            proteins_scores_all.append(line)
    proteins = proteins_scores_all[::2]
    scores = proteins_scores_all[1::2]

    save_file_path = protein_class_path + '/acid_avgscore.csv'
    acid_scoreList_dict = {}
    for i, protein in enumerate(proteins):
        score = list(map(float, scores[i]))
        for j, acid in enumerate(protein):
            acid_score = score[j]
            if acid_scoreList_dict.get(acid) == None:
                acid_scoreList_dict[acid] = [acid_score]
            else:
                acid_scoreList_dict[acid].append(acid_score)
    
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
    acid_score_pd.to_csv(save_file_path,  mode='a', header=False, index=False, float_format='%.4f')   
    
    print('success')