import numpy as np
import pandas as pd
import os
import csv


def save_test_protein_statics(save_dir_path, name, protein, score):
    sorted_id = sorted(range(len(score)), key=lambda k: score[k], reverse=True)
    protein_sort_score = np.array(score)[sorted_id]  # 记录一个蛋白质中整个序列根据score排序后数据，阳性降序、阴性升序
    protein_sort_acid = np.array(protein)[sorted_id]


    protein_name = pd.Series([name], index=['name'])
    protein_pd = pd.DataFrame([protein], index=['protein'])
    score_pd = pd.DataFrame([score], index=['score'])
    protein_sort_protein_pd = pd.DataFrame([protein_sort_acid], index=['sort_protein'])
    protein_sort_score_pd = pd.DataFrame([protein_sort_score], index=['soct_score'])

    savepath1 = save_dir_path + 'mydata_1507_data/test_statics/RCNN_ECA_testprotein_statics.csv'

    protein_name.to_csv(savepath1, mode='a', header=False, index=True)
    protein_pd.to_csv(savepath1, mode='a', header=False, index=True)
    score_pd.to_csv(savepath1, mode='a', header=False, index=True, float_format='%.4f')
    protein_sort_protein_pd.to_csv(savepath1, mode='a', header=False, index=True)
    protein_sort_score_pd.to_csv(savepath1, mode='a', header=False, index=True, float_format='%.4f')


if __name__== '__main__':

    # outAll outFinal
    dir_path = '../../output/gradCAM/gradCAM_noSoftmax_outFinal_protein_score/'
    # dir_path = './RCNN_ECA_saliency/CAM_noSoftmax_outFinal_protein_score/'

    protein_class = 'pos_sequence_score'

    protein_class_path = dir_path + 'mydata_1507_data/'+ protein_class
    protein_file_name = 'RCNN_ECA_protein_score.csv'
    # test_protein_file_name = dir_path + 'mydata_1507_data/'+ 'mydata_1507_testprotein.xlsx'
    test_protein_file_name = '../../output/FUS_family_test.xlsx'
    
    protein_file_path = os.path.join(protein_class_path, protein_file_name)
    proteins_scores_all = []
    with open(protein_file_path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            proteins_scores_all.append(line)
    proteins = proteins_scores_all[::3]
    scores = proteins_scores_all[1::3]

    # proteins = proteins_scores_all[::2]
    # scores = proteins_scores_all[1::2]


    search_protein_dataset = [''.join(protein).upper() for protein in proteins]

    verify_data = pd.read_excel(test_protein_file_name,header=None)
    verify_proteins = verify_data.iloc[:, 1].values.ravel()
    verify_proteins_list = verify_proteins.tolist()
    verify_names = verify_data.iloc[:, 0].values.ravel()
    verify_names_list = verify_names.tolist()
    



    for idx in range(len(verify_proteins_list)):
        verify_protein = verify_proteins_list[idx]
        verify_name = verify_names_list[idx]
        if verify_protein in search_protein_dataset:
            searchindex = search_protein_dataset.index(verify_protein)
            searchprotein = proteins[searchindex]
            searchscore = scores[searchindex]
            # save_test_protein_statics(dir_path, verify_name, searchprotein, searchscore)
        else:
            print(verify_name)
            continue
    print('success')



    
    



    