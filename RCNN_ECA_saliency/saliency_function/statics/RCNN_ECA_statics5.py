import numpy as np
import pandas as pd
import os
import csv


if __name__== '__main__':

    # outAll outFinal
    dir_path = '../../output/gradCAM/gradCAM_noSoftmax_outFinal_protein_score/'

    # dir_path = './RCNN_ECA_saliency/output/gradCAM/gradCAM_noSoftmax_outFinal_protein_score/'

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

    save_file_path = protein_class_path + '/acid_score_1.csv'
    acid_scoreList_dict = {}
    for i, protein in enumerate(proteins):
        score = list(map(float, scores[i]))
        for j, acid in enumerate(protein):
            acid_score = score[j]
            if acid_scoreList_dict.get(acid) == None:
                acid_scoreList_dict[acid] = [acid_score]
            else:
                acid_scoreList_dict[acid].append(acid_score)
    
    for key, val in acid_scoreList_dict.items():
        # acid_score_pd = pd.DataFrame([val], index=[key])
        acid_score_pd = pd.DataFrame(val, columns=[key])
        acid_score_pd.to_csv(save_file_path,  mode='a', header=False, index=True, float_format='%.4f')     
    
    print('success')



    
    



    