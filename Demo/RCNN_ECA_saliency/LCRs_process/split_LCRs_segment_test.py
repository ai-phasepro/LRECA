import pandas as pd
import csv
import os
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import seaborn as sns
from IPython import display
import scipy 
from scipy import signal
import heapq


def save_density(savepath, proteins_names_list, proteins_list, scores_list, density_list):
    for i, protein_name in enumerate(proteins_names_list):
        name = pd.Series(protein_name, index=['name'])
        protein = pd.DataFrame([proteins_list[i]], index=['protein'])
        score = pd.DataFrame([scores_list[i]], index=['score'])
        density_array = np.exp(density_list[i])  
        smooth_density_array = signal.savgol_filter(density_array,50,3)
        process_density_array = -1 * (smooth_density_array - np.max(smooth_density_array))
        process_density = pd.DataFrame([process_density_array], index=['process_density'])
        name.to_csv(savepath, mode='a', header=False, index=True)
        protein.to_csv(savepath, mode='a', header=False, index=True)
        score.to_csv(savepath, mode='a',header=False, index=True, float_format='%.4f')
        process_density.to_csv(savepath, mode='a', header=False, index=True, float_format='%.4f')


def draw_score(savepath, proteins_names_list, proteins_list, scores_list):
    for i, protein_name in enumerate(proteins_names_list):
        plt.figure()
        plt.plot(np.arange(len(scores_list[i])), scores_list[i])
        plt.xlabel('protein')
        plt.ylabel('score')
        plt.show()
        plt.savefig(savepath+ i +'.png')


def draw_density(savepath, proteins_names_list, proteins_list, density_list):
    for i, protein_name in enumerate(proteins_names_list):
        plt.figure()
        plt.plot(np.arange(density_list[i].shape[0]), density_list[i])
        plt.xlabel('protein')
        plt.ylabel('density')
        plt.show()
        plt.savefig(savepath+ i +'.png')


def draw_density_heatmap(savepath, proteins_names_list, proteins_list, density_list):
    proteins_length = [len(i) for i in proteins_list]
    fig_row, fig_col = 3, 5
    
    fig, ax = plt.subplots(fig_row, fig_col, constrained_layout=True, figsize=(320, 240))
    
    for i, protein_name in enumerate(proteins_names_list):
        protein_array = np.asarray(proteins_list[i])
        density_array = density_list[i]
        array_length = protein_array.shape[0]
        array_max1 = np.power(10,len(str(array_length))-1)
        protein_array_pad_length =  (int)(np.ceil(array_length/array_max1) * array_max1)
        padding_array = np.full((protein_array_pad_length-array_length,), 0)
        protein_array_pad = np.hstack([protein_array, padding_array]).reshape(-1, 100)
        density_array_pad = np.hstack([density_array, padding_array]).reshape(-1, 100)

        mask = np.zeros_like(density_array_pad)
        mask_row_index, mask_col_index = array_length//array_max1, array_length%array_max1
        mask[mask_row_index,mask_col_index:] = True
        
        h = sns.heatmap(density_array_pad.reshape(-1,100), mask=mask, fmt=".4f", cmap="RdBu_r",annot=True, 
                    linewidths=.5, square=True, annot_kws={'fontsize':8}, cbar=False, ax=ax[i//fig_col][i%fig_col])
        h.set_title(protein_name[0], fontsize = 40)
    figure = fig.get_figure()
    figure.savefig(savepath)        


def compute_density_diff(proteins_names_list, proteins_list, density_list):
    density_diff_list = []
    for i, protein_name in enumerate(proteins_names_list):
        protein_array = np.asarray(proteins_list[i])
        density_array = density_list[i]
        dx = 1
        density_diff = np.diff(density_array)/dx
        density_diff_list.append(density_diff) 


def find_sequence_segmentpoint(process_density_array, valley_xlimit=20, valley_ylimit=0.15, topk=3):
    peaks,_ = signal.find_peaks(process_density_array, prominence=0.1)          
    if 0 not in peaks:
        peaks = np.insert(peaks, 0, 0)
    if (process_density_array.shape[0]-1) not in peaks:
        peaks = np.append(peaks, process_density_array.shape[0]-1)
        
    valley_list = []
    for i in range(peaks.shape[0]-1):
        local_min_value = np.min(process_density_array[peaks[i]:peaks[i+1]+1])
        if local_min_value <= valley_ylimit:
            local_min_index = np.where(process_density_array[peaks[i]:peaks[i+1]+1] == local_min_value)
            valley_list.append(local_min_index[0][0]+peaks[i])

    if 0 not in valley_list:
        valley_list.insert(0,0)
    if (process_density_array.shape[0]-1) not in valley_list:
        valley_list.append(process_density_array.shape[0]-1)
    
    segment_list = []
    left, mid = valley_list[0], valley_list[1]
    for right in valley_list[2:]:
        if (process_density_array[mid] < process_density_array[left] or process_density_array[mid] < process_density_array[right]):
            segment_list.append(left)
            left = mid
        mid = right
    segment_list.extend([left, mid])
    
    return peaks, np.asarray(segment_list)

def find_max_segment(proteins_names_list, proteins_list, density_list, savepath):
    max_protein_segment_list, max_segment_idx_list = [],[]
    density_segments_list, process_density_segments_list, protein_segments_list = [],[], []
    index_list, score_list, area_list = [], [], [] 
    for i, protein_name in enumerate(proteins_names_list):
        protein_array = np.asarray(proteins_list[i])
        density_array = density_list[i]
        smooth_density_array = signal.savgol_filter(density_array,50,3)
        process_density_array = -1 * (smooth_density_array - np.max(smooth_density_array))
        peaks, valleys  = find_sequence_segmentpoint(process_density_array, 20, np.max(process_density_array), 3)
        density_segments, process_density_segments, protein_segments, segment_score, segment_area = [],[],[],[],[]
        max_density = -float('inf')  
        max_area = -float('inf')
        max_segment_length = 0  
        for j in range(valleys.shape[0]-1):
            density_segment_array = density_array[valleys[j]:valleys[j+1]]
            process_density_segment_array = process_density_array[valleys[j]:valleys[j+1]]
            protein_segment_array = protein_array[valleys[j]:valleys[j+1]]
            density_segments.append(density_segment_array.tolist())
            process_density_segments.append(process_density_segment_array.tolist())
            protein_segments.append(protein_segment_array.tolist())
            avg_score = np.sum(process_density_segment_array)
            segment_score.append(avg_score)
            segment_area.append(np.trapz(process_density_segment_array, dx=1.0))  

            if((max_density<avg_score) or ((max_density==avg_score)and(max_segment_length<process_density_segment_array.shape[0]))):
                max_segment_idx = str(j)
                max_segment_length = process_density_segment_array.shape[0]
                max_density = avg_score
                max_protein_segment = protein_segment_array.tolist()
        index_list.append(valleys.tolist())
        max_segment_idx_list.append(max_segment_idx)
        max_protein_segment_list.append(max_protein_segment)
        
        density_segments_list.append(density_segments)
        process_density_segments_list.append(process_density_segments)
        protein_segments_list.append(protein_segments)
        score_list.append(segment_score)
        area_list.append(segment_area)
        
        plt.figure(figsize=(20,20))
        plt.subplot(2,1,1)
        plt.plot(np.arange(density_array.shape[0]), density_array)
        plt.vlines([valleys[int(max_segment_idx)],valleys[int(max_segment_idx)+1]-1], np.min(density_array), np.max(density_array), linestyles='dashed', colors='red')
        plt.xlabel('protein')
        plt.ylabel('density')
        plt.title(protein_name[0]+'_density_max_segment')

        plt.subplot(2,1,2)
        plt.plot(np.arange(process_density_array.shape[0]), process_density_array, color = "yellow")
        plt.scatter(peaks, process_density_array[peaks], color="red")
        plt.scatter(valleys, process_density_array[valleys], color="green")
        plt.xlabel('protein')
        plt.ylabel('process_density')
        plt.title(protein_name[0]+'_density_peaks_and_valleys')
        plt.savefig(savepath+'density_segment_picture/LCRs_true_density_segment_{}.png'.format(i))
        print("picture_{} success".format(i))

    return index_list, max_segment_idx_list, max_protein_segment_list, protein_segments_list, density_segments_list, process_density_segments_list, score_list, area_list




def save_segment(savepath, max_segment_idx_list, max_protein_segment_list, proteins_names, protein_segments_list, density_segments_list, process_density_segments_list, scores_list, areas_list, indexs_list):
    for i, protein_name in enumerate(proteins_names):
        protein_segments = protein_segments_list[i]
        density_segments = density_segments_list[i]
        process_density_segments = process_density_segments_list[i]
        scores = scores_list[i]
        areas = areas_list[i]
        indexs = indexs_list[i]

        max_protein_segment = max_protein_segment_list[i]
        max_segment_idx = max_segment_idx_list[i]

        name = pd.Series(protein_name, index=['name'])    
        max_idx = pd.Series(max_segment_idx, index=['max_segment_idx'])    
        max_head_tail = pd.DataFrame([[indexs[int(max_segment_idx)],indexs[int(max_segment_idx)+1]-1]], index=['max_segment_head_tail_index'])  
        max_protein = pd.DataFrame([max_protein_segment], index=['max_segment_protein'])  

        name.to_csv(savepath, mode='a', header=False, index=True)
        max_idx.to_csv(savepath, mode='a', header=False, index=True)
        max_head_tail.to_csv(savepath, mode='a',header=False, index=True, float_format='%d')
        max_protein.to_csv(savepath, mode='a', header=False, index=True)

        for j, protein_segment in enumerate(protein_segments):
            segment_idx = pd.Series(str(j), index=['segment_idx'])  
            seg_protein = pd.DataFrame([protein_segment], index=['segment_protein'])  
            seg_density = pd.DataFrame([density_segments[j]], index=['segment_density'])  
            process_seg_density = pd.DataFrame([process_density_segments[j]], index=['process_segment_density'])  
            seg_score = pd.Series(scores[j], index=['segment_score'])  
            seg_area = pd.Series(areas[j], index=['segment_area'])     
            seg_head_tail = pd.DataFrame([[indexs[j],indexs[j+1]-1]], index=['segment_head_tail_index'])  
            
            segment_idx.to_csv(savepath, mode='a', header=False, index=True)
            seg_score.to_csv(savepath, mode='a',header=False, index=True, float_format='%.4f')
            seg_area.to_csv(savepath, mode='a',header=False, index=True, float_format='%.4f')
            seg_head_tail.to_csv(savepath, mode='a',header=False, index=True, float_format='%d')
            seg_protein.to_csv(savepath, mode='a', header=False, index=True)
            seg_density.to_csv(savepath, mode='a',header=False, index=True, float_format='%.4f')
            process_seg_density.to_csv(savepath, mode='a',header=False, index=True, float_format='%.4f')


if __name__== '__main__':
    FUS_test_path='Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/test/test_statics.csv'
    train_proteins_scores_all = []
    with open(FUS_test_path, 'r') as f:
        reader = csv.reader(f)
        train_proteins_scores_all.extend(reader)
    train_proteins_names = train_proteins_scores_all[::5] 
    train_proteins = train_proteins_scores_all[1::5]  
    train_scores = train_proteins_scores_all[2::5]  

    train_proteins_names = [i[1:] for i in train_proteins_names]
    train_proteins = [i[1:] for i in train_proteins]
    train_scores = [list(map(float, i[1:])) for i in train_scores]

    log_density_list, density_list = [], []
    for i, protein_name in enumerate(train_proteins_names):
        protein_name = protein_name[0]
        protein = train_proteins[i]
        score = train_scores[i]
        score_array = np.asarray(score).reshape((len(score),-1))

        params = {"bandwidth": np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(score_array)
        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
        kde = grid.best_estimator_
        log_density =kde.score_samples(score_array)
        log_density_list.append(log_density)
        density_list.append(np.exp(log_density))

    segment_savepath = 'Saliency_output/LCRs_process/test/density_segment/density_segment_train/'
    density_savepath = 'Saliency_output/LCRs_process/test/density_map/LCRs_protein_densitymap_train'
    if not os.path.exists('Saliency_output/LCRs_process/test/density_map/'):
        os.makedirs('Saliency_output/LCRs_process/test/density_map/')
    save_density(density_savepath +'.csv', train_proteins_names, train_proteins, train_scores, log_density_list)

    print("ALL success")
    