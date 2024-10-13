import os
import time
import pandas as pd
import numpy as np
import argparse

def readverifydata(verify_protein_path):
    verify_data = pd.read_excel(verify_protein_path,header=None)
    sequence = verify_data.iloc[:, 1].values.ravel()
    name = verify_data.iloc[:, 0].values.ravel()
    label = np.ones(shape=(sequence.shape))
    print('Sequence', sequence.shape[0])

    return name, sequence, label

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = None)
parser.add_argument('--seq', type=str, default = None)
args = parser.parse_args()

if args.dataset:
    dataname = args.dataset

    test_name, test_seq, _ = readverifydata("../../test_dataset/" + dataname +".xlsx")
elif args.seq:
    test_seq = args.seq
    test_name = "test_seq"
    dataname = test_name

for idx, seq in enumerate(test_seq):
    pd_dict = {'name':[test_name[idx], test_name[idx]], 'seq':[seq, seq]}
    test_pd = pd.DataFrame(pd_dict)
    test_pd.to_excel("../../Saliency_output/test.xlsx", header=False, index=False)
    os.system("python ./saliency_function/verify/RCNN_ECA_saliency_verify_gradCAM_fortest.py")
    os.system("python ./saliency_function/statics/RCNN_ECA_statics2_fortest.py")
    os.system("python ./saliency_function/statics/RCNN_ECA_statics5_fortest.py")
    os.system("python ./LCRs_process/split_LCRs_segment_forsingle.py")
    os.rename("../../Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/forsingle/test_acid_avgscore.csv", "../../Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/forsingle/%s_acid_avgscore.csv" % test_name[idx])
    os.rename("../../Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/forsingle/test_acid_sumscore.csv", "../../Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/forsingle/%s_acid_sumscore.csv" % test_name[idx])
    os.rename("../../Saliency_output/LCRs_process/forsingle/density_segment/true_density_segment_statics.csv", "../../Saliency_output/LCRs_process/forsingle/density_segment/%s_true_density_segment_statics.csv" % test_name[idx])
    os.rename("../../Saliency_output/LCRs_process/forsingle/density_segment/max_segment.csv", "../../Saliency_output/LCRs_process/forsingle/density_segment/%s_max_segment.csv" % test_name[idx])
    os.rename("../../Saliency_output/LCRs_process/forsingle/density_map/LCRs_protein_densitymap_forsingle.csv", "../../Saliency_output/LCRs_process/forsingle/density_map/%s_LCRs_protein_densitymap.csv" % test_name[idx])

os.rename("../../Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/forsingle/test_statics.csv", "../../Saliency_output/gradCAM/gradCAM_noSoftmax_outAll_protein_score/forsingle/%s_statics.csv" % dataname)