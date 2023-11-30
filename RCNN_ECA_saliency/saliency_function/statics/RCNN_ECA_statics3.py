import numpy as np
import pandas as pd
import os
import csv
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
def save_test_protein_statics(save_dir_path, name, protein, score):
    sorted_id = sorted(range(len(score)), key=lambda k: score[k], reverse=True)
    protein_sort_score = np.array(score)[sorted_id]  # 记录一个蛋白质中整个序列根据score排序后数据，阳性降序、阴性升序
    protein_sort_acid = np.array(protein)[sorted_id]


def get_acid_score_dict(protein, score):
    acid_score_dict = {}
    for i, acid in enumerate(protein):
        acid_score = float(score[i])
        if acid_score_dict.get(acid) == None:
            acid_score_dict[acid] = [acid_score]
        else:
            acid_score_dict[acid].append(acid_score)
    acid_score_dict = sorted(acid_score_dict.items())
    acid_score_dict = {k[0]: k[1] for k in acid_score_dict}
    acid_maxscore_dict = {k: max(v) for k, v in acid_score_dict.items()}
    acid_averagescore_dict = {k : sum(v)/len(v) for k, v in acid_score_dict.items()}
    return acid_score_dict, acid_maxscore_dict, acid_averagescore_dict


def is_norm_distribution(acid_score_dict):
    acid_score_norm_dict = {}
    for acid, score_list in acid_score_dict.items():
        u = np.mean(np.array(score_list))
        std = np.std(np.array(score_list))
        D, P = stats.kstest(score_list, 'norm', (u, std))
        acid_score_norm_dict[acid] = True if P >= 0.05 else False
    return acid_score_norm_dict

#  t检验方法
def compute_two_acid_ttest(acid_allprotein_score_dict):
    acid_allprotein_avgscore_dict = {k:np.average(v) for k,v in acid_allprotein_score_dict.items()}
    acid_allprotein_avgscore_sorted_tuple_list = sorted(acid_allprotein_avgscore_dict.items(), key=lambda x:x[1], reverse=True)
    acid_allprotein_avgscore_sorted_dict = {i[0]:i[1] for i in acid_allprotein_avgscore_sorted_tuple_list}
    dict_length = len(acid_allprotein_avgscore_sorted_dict)
    acid_allprotein_avgscore_sorted_pd = pd.DataFrame(data = np.full((dict_length, dict_length), -1), index=acid_allprotein_avgscore_sorted_dict.keys(), 
                                            columns=acid_allprotein_avgscore_sorted_dict.keys())
    
    acid_allprotein_score_sorted_tuple_list = sorted(acid_allprotein_score_dict.items(), key=lambda x:np.average(x[1]), reverse=True)
    acid_allprotein_score_sorted_dict = {i[0]:i[1] for i in acid_allprotein_score_sorted_tuple_list}
    allprotein_score_list = [v for v in acid_allprotein_score_sorted_dict.values()]
    allprotein_acid_list = [k for k in acid_allprotein_score_sorted_dict.keys()]
    two_acid_ttest_dict = {}

    for i in range(len(allprotein_score_list)):
        for j in range(i+1, len(allprotein_score_list)):
            acid1 = allprotein_score_list[i]
            acid2 = allprotein_score_list[j]
            ls, lp = stats.levene(acid1, acid2)
            if lp >= 0.05:
                rs, rp = stats.ttest_ind(acid1, acid2, equal_var=True)
            else:
                rs, rp = stats.ttest_ind(acid1, acid2, equal_var=False)
            if np.mean(acid1) > np.mean(acid2):
                ttest_key = allprotein_acid_list[i]+'>'+allprotein_acid_list[j]
                two_acid_ttest_dict[ttest_key] = (rs, rp/2)
                acid_allprotein_avgscore_sorted_pd.loc[allprotein_acid_list[i], allprotein_acid_list[j]] = rp/2
            elif np.mean(acid1) < np.mean(acid2):
                ttest_key = allprotein_acid_list[i]+'<'+allprotein_acid_list[j]
                two_acid_ttest_dict[ttest_key] = (rs, rp/2)
                acid_allprotein_avgscore_sorted_pd.loc[allprotein_acid_list[i], allprotein_acid_list[j]] = rp/2
            else:
                ttest_key = allprotein_acid_list[i]+'='+allprotein_acid_list[j]
                two_acid_ttest_dict[ttest_key] = (rs, rp)
                acid_allprotein_avgscore_sorted_pd.loc[allprotein_acid_list[i], allprotein_acid_list[j]] = rp
    return two_acid_ttest_dict, acid_allprotein_avgscore_sorted_pd


# 秩和检验方法
def compute_two_acid_ranksumtest(acid_allprotein_score_dict):
    allprotein_score_list = [v for v in acid_allprotein_score_dict.values()]
    allprotein_acid_list = [k for k in acid_allprotein_score_dict.keys()]
    
    # 产生每一个蛋白质的ranksum
    allprotein_score_array = np.asarray(allprotein_score_list)
    allprotein_score_array_flatten = allprotein_score_array.flatten()
    allprotein_score_array_flatten_index = np.argsort(allprotein_score_array_flatten)
    allprotein_score_array_flatten_rank = np.zeros_like(allprotein_score_array_flatten, dtype=int)
    for i, score_idx in enumerate(allprotein_score_array_flatten_index):
        allprotein_score_array_flatten_rank[score_idx] = i + 1
    allprotein_score_array_rank = allprotein_score_array_flatten_rank.reshape(allprotein_score_array.shape)
    

    allprotein_score_ranksum_list = [np.sum(v) for v in allprotein_score_array_rank]
    acid_allprotein_ranksum_dict = dict(zip(allprotein_acid_list, allprotein_score_ranksum_list))
    
    acid_allprotein_ranksum_sorted_tuple_list = sorted(acid_allprotein_ranksum_dict.items(), key=lambda x:x[1], reverse=True)
    acid_allprotein_ranksum_sorted_dict = {i[0]:i[1] for i in acid_allprotein_ranksum_sorted_tuple_list}
    dict_length = len(acid_allprotein_ranksum_sorted_dict)
    acid_allprotein_avgscore_sorted_pd = pd.DataFrame(data = np.full((dict_length, dict_length), -1), index=acid_allprotein_ranksum_sorted_dict.keys(), 
                                            columns=acid_allprotein_ranksum_sorted_dict.keys())
    
    sorted_id = sorted(range(len(allprotein_score_ranksum_list)), key=lambda k: allprotein_score_ranksum_list[k], reverse=True)
    allprotein_score_list = np.asarray(allprotein_score_list)[sorted_id].tolist()
    allprotein_acid_list = np.asarray(allprotein_acid_list)[sorted_id].tolist()

    two_acid_ranksumtest_dict = {}

    for i in range(len(allprotein_score_list)):
        for j in range(i+1, len(allprotein_score_list)):
            acid1 = allprotein_score_list[i]
            acid2 = allprotein_score_list[j]
            name1 = allprotein_acid_list[i]
            name2 = allprotein_acid_list[j]
            
            if  acid_allprotein_ranksum_sorted_dict[name1] > acid_allprotein_ranksum_sorted_dict[name2]:
                rs, rp = stats.mannwhitneyu(acid1, acid2, alternative='greater')
                sumtest_key = allprotein_acid_list[i]+'>'+allprotein_acid_list[j]
                
            elif acid_allprotein_ranksum_sorted_dict[name1] < acid_allprotein_ranksum_sorted_dict[name2]:
                rs, rp = stats.mannwhitneyu(acid1, acid2, alternative='less')
                sumtest_key = allprotein_acid_list[i]+'<'+allprotein_acid_list[j]
                
            else:
                rs, rp = stats.mannwhitneyu(acid1, acid2, alternative='two-sided')
                sumtest_key = allprotein_acid_list[i]+'='+allprotein_acid_list[j]
            two_acid_ranksumtest_dict[sumtest_key] = (rs, rp)
            acid_allprotein_avgscore_sorted_pd.loc[allprotein_acid_list[i], allprotein_acid_list[j]] = rp
    return two_acid_ranksumtest_dict, acid_allprotein_avgscore_sorted_pd


def drawAndSave_heatmap(save_path, max_score_sorted_pd, avg_score_sorted_pd):
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(80,60))
    max_x_ticks = max_score_sorted_pd.columns.values
    max_y_ticks = max_score_sorted_pd.index.values
    max_score_values = max_score_sorted_pd.values
    mask = np.zeros_like(max_score_values, dtype=bool)
    mask[np.tril_indices_from(mask)] = True
    sns.heatmap(max_score_values, mask=mask, ax=ax1, vmin=0.05, cmap="RdBu_r",
                annot=True, fmt='.4f', linewidths=.5, square=True, annot_kws={'fontsize':20})
    cbar = ax1.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    ax1.set_title('max_score_sorted_matrix', fontsize = 40)    
    ax1.set_xticklabels(max_x_ticks, fontsize=30)
    ax1.set_yticklabels(max_y_ticks, fontsize=30)


    avg_x_ticks = avg_score_sorted_pd.columns.values
    avg_y_ticks = avg_score_sorted_pd.index.values
    avg_score_values = avg_score_sorted_pd.values
    sns.heatmap(avg_score_values, mask=mask, ax=ax2,  vmin=0.05, cmap="RdBu_r",
                annot=True, fmt='.4f', linewidths=.5, square=True, annot_kws={'fontsize':20})
    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    ax2.set_title('avg_score_sorted_matrix', fontsize = 40)
    ax2.set_xticklabels(avg_x_ticks, fontsize=30)
    ax2.set_yticklabels(avg_y_ticks, fontsize=30)

    plt.show()
    figure = f.get_figure()
    figure.savefig(save_path+'heatmap.jpg')


def drawAndSave_allheatmap(save_path, ttest_avgscore_sorted_pd, ttest_maxscore_sorted_pd, 
                            ranksumtest_avgscore_sorted_pd, ranksumtest_maxscore_sorted_pd):
    f, axes = plt.subplots(nrows=2, ncols=2, figsize=(120,100))
    ax1, ax2, ax3, ax4 = axes[0][0], axes[0][1], axes[1][0], axes[1][1]
    mask = np.zeros_like(ttest_avgscore_sorted_pd.values, dtype=bool)
    mask[np.tril_indices_from(mask)] = True
    # ax1
    sns.heatmap(ttest_avgscore_sorted_pd.values, mask=mask, ax=ax1, vmin=0.05, cmap="RdBu_r",
                annot=True, fmt='.4f', linewidths=.5, square=True, annot_kws={'fontsize':20})
    cbar = ax1.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    ax1.set_title('ttest_avgscore_sorted_matrix', fontsize = 40)    
    ax1.set_xticklabels(ttest_avgscore_sorted_pd.columns.values, fontsize=30)
    ax1.set_yticklabels(ttest_avgscore_sorted_pd.index.values, fontsize=30)

    # ax2
    sns.heatmap(ttest_maxscore_sorted_pd.values, mask=mask, ax=ax2,  vmin=0.05, cmap="RdBu_r",
                annot=True, fmt='.4f', linewidths=.5, square=True, annot_kws={'fontsize':20})
    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    ax2.set_title('ttest_maxscore_sorted_matrix', fontsize = 40)
    ax2.set_xticklabels(ttest_maxscore_sorted_pd.columns.values, fontsize=30)
    ax2.set_yticklabels(ttest_maxscore_sorted_pd.index.values, fontsize=30)

    #ax3
    sns.heatmap(ranksumtest_avgscore_sorted_pd.values, mask=mask, ax=ax3,  vmin=0.05, cmap="RdBu_r",
                annot=True, fmt='.4f', linewidths=.5, square=True, annot_kws={'fontsize':20})
    cbar = ax3.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    ax3.set_title('ranksumtest_avgscore_sorted_matrix', fontsize = 40)
    ax3.set_xticklabels(ranksumtest_avgscore_sorted_pd.columns.values, fontsize=30)
    ax3.set_yticklabels(ranksumtest_avgscore_sorted_pd.index.values, fontsize=30)

    #ax4
    sns.heatmap(ranksumtest_maxscore_sorted_pd.values, mask=mask, ax=ax4,  vmin=0.05, cmap="RdBu_r",
                annot=True, fmt='.4f', linewidths=.5, square=True, annot_kws={'fontsize':20})
    cbar = ax4.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    ax4.set_title('ranksumtest_maxscore_sorted_matrix', fontsize = 40)
    ax4.set_xticklabels(ranksumtest_maxscore_sorted_pd.columns.values, fontsize=30)
    ax4.set_yticklabels(ranksumtest_maxscore_sorted_pd.index.values, fontsize=30)

    plt.show()
    figure = f.get_figure()
    figure.savefig(save_path+'heatmap.jpg')


def save_test_dict(save_path, two_acid_test_dict, score_sorted_pd):
    two_acid_test_pd = pd.DataFrame.from_dict(two_acid_test_dict, orient='index', columns=['statistic','pvalue'])
    two_acid_test_pd.to_csv(save_path+'dict_sorted.csv', float_format='%.4f')
    score_sorted_pd.to_excel(save_path+'DataFrame_sorted.xlsx', float_format='%.4f')




if __name__== '__main__':

    dir_path = '../../output/gradCAM/gradCAM_noSoftmax_outFinal_protein_score/'
    # dir_path = './RCNN_ECA_saliency/RCNN_ECA_saliency_CAM_noSoftmax_outFinal_protein_score/'

    test_protein_dir_path = dir_path + 'mydata_1507_data/test_statics/'
    test_protein_file_name = test_protein_dir_path + 'RCNN_ECA_testprotein_statics.csv'
    
    proteins_scores_all = []
    with open(test_protein_file_name, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            proteins_scores_all.append(line)
    proteins_names = proteins_scores_all[::5][:-1]  # 删去最后一个TDP-43蛋白质
    proteins = proteins_scores_all[1::5][:-1]
    scores = proteins_scores_all[2::5][:-1]

    acid_allprotein_maxscore_dict = {}
    acid_allprotein_avgscore_dict = {}
    for idx, protein_name in enumerate(proteins_names):
        protein_name = protein_name[1:]
        protein = proteins[idx][1:]
        score = scores[idx][1:]
        acid_score_dict, acid_maxscore_dict, acid_averagescore_dict = get_acid_score_dict(protein, score)
        # acid_score_norm_dict = is_norm_distribution(acid_score_dict)
        for acid, score in acid_maxscore_dict.items():
            if acid_allprotein_maxscore_dict.get(acid) == None:
                acid_allprotein_maxscore_dict[acid] = [score]
            else:
                acid_allprotein_maxscore_dict[acid].append(score)
        
        for acid, score in acid_averagescore_dict.items():
            if acid_allprotein_avgscore_dict.get(acid) == None:
                acid_allprotein_avgscore_dict[acid] = [score]
            else:
                acid_allprotein_avgscore_dict[acid].append(score)

    acid_allprotein_maxscore_norm_dict = is_norm_distribution(acid_allprotein_maxscore_dict)
    two_acid_maxscore_ttest_dict, ttest_maxscore_sorted_pd = compute_two_acid_ttest(acid_allprotein_maxscore_dict)
    two_acid_maxscore_ranksumtest_dict, ranksumtest_maxscore_sorted_pd = compute_two_acid_ranksumtest(acid_allprotein_maxscore_dict)
    
    acid_allprotein_avgscore_norm_dict = is_norm_distribution(acid_allprotein_avgscore_dict)
    two_acid_avgscore_ttest_dict, ttest_avgscore_sorted_pd = compute_two_acid_ttest(acid_allprotein_avgscore_dict)
    two_acid_avgscore_ranksumtest_dict, ranksumtest_avgscore_sorted_pd = compute_two_acid_ranksumtest(acid_allprotein_avgscore_dict)

    # save_test_dict(test_protein_dir_path+ 'two_acid_maxscore_ttest_', two_acid_maxscore_ttest_dict, ttest_maxscore_sorted_pd)
    # save_test_dict(test_protein_dir_path+ 'two_acid_avgscore_ttest_', two_acid_avgscore_ttest_dict, ttest_avgscore_sorted_pd)

    save_test_dict(test_protein_dir_path+ 'two_acid_maxscore_sumtest_', two_acid_maxscore_ranksumtest_dict, ranksumtest_maxscore_sorted_pd)
    save_test_dict(test_protein_dir_path+ 'two_acid_avgscore_sumtest_', two_acid_avgscore_ranksumtest_dict, ranksumtest_avgscore_sorted_pd)

    # drawAndSave_heatmap(test_protein_dir_path+ 'two_acid_ttest_', ttest_maxscore_sorted_pd, ttest_avgscore_sorted_pd)
    # drawAndSave_heatmap(test_protein_dir_path+ 'two_acid_sumtest_', ranksumtest_maxscore_sorted_pd, ranksumtest_avgscore_sorted_pd)
    drawAndSave_allheatmap(test_protein_dir_path+ 'two_acid_test_', ttest_avgscore_sorted_pd, ttest_maxscore_sorted_pd,
                            ranksumtest_avgscore_sorted_pd, ranksumtest_maxscore_sorted_pd)
    print('success')



