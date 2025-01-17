import numpy as np
import pandas as pd
import os
import csv

file_dir = os.path.dirname(os.path.abspath(__file__))
print(file_dir)
os.chdir(file_dir)

def save_test_protein_statics(savepath1, name, protein, score):
    sorted_id = sorted(range(len(score)), key=lambda k: score[k], reverse=True)
    protein_sort_score = np.array(score)[sorted_id]  
    protein_sort_acid = np.array(protein)[sorted_id]

    protein_name = pd.Series([name], index=['name'])
    protein_pd = pd.DataFrame([protein], index=['protein'])
    score_pd = pd.DataFrame([score], index=['score'])
    protein_sort_protein_pd = pd.DataFrame([protein_sort_acid], index=['sort_protein'])
    protein_sort_score_pd = pd.DataFrame([protein_sort_score], index=['soct_score'])
    protein_name.to_csv(savepath1, mode='a', header=False, index=True)
    protein_pd.to_csv(savepath1, mode='a', header=False, index=True)
    score_pd.to_csv(savepath1, mode='a', header=False, index=True, float_format='%.4f')
    protein_sort_protein_pd.to_csv(savepath1, mode='a', header=False, index=True)
    protein_sort_score_pd.to_csv(savepath1, mode='a', header=False, index=True, float_format='%.4f')


if __name__== '__main__':
    dir_path = '../../../../Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/'

    protein_class_path = dir_path + 'test/'
    if not os.path.exists(os.path.dirname(protein_class_path)):
        os.makedirs(os.path.dirname(protein_class_path))
    protein_file_name = 'pos_sequence_score.csv'
    
    test_protein_file_name = '../../../../test_dataset/test.xlsx'

    protein_file_path = protein_class_path + protein_file_name
    proteins_scores_all = []
    with open(protein_file_path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            proteins_scores_all.append(line)
    proteins = proteins_scores_all[::2]
    scores = proteins_scores_all[1::2]

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
            save_test_protein_statics(protein_class_path + 'test_statics.csv', verify_name, searchprotein, searchscore)
        else:
            print(verify_name)
            continue
    print('success')



    
    



    