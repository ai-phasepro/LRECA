# LRECA

## Files

This code contains two parts: protein phase separation characterization and traceability.

The RCNN_model folder contains the characterization code.

Specifically, it contains the code for LLPS and PDB, phaspDB_reviewed and PDB, phaspDB_highthroughput and PDB, mydata_1507 and PDB dataset training.

The RCNN_ECA_saliency folder contains the feature traceability code. It specifically contains the LCRs_process, saliency_functi, and save_model folders.

The LCRs_process folder contains read_LCRs_segment.py for comparative analysis with existing LCRs segments, and split_LCRs_segment.py for generating segments based on the probability density of the contribution value at each position of the amino acid sequence and highlighting the amino acid segment with the largest contribution value;

The method folder in the saliency_functio folder contains the code for generating the contribution value at each position of the amino acid sequence, the statics folder contains the code for statistically analyzing the contribution value at each position of the amino acid sequence, and the RCNN_ECA_save_model.py file is used to save the trained model and parameters;

The save_model folder saves the trained models and parameters for characterization detection.

## Test_Dataset

./processed_dataset contains processed data of three public datasets that are used in this paper, including LLPS, phaspDB_reviewed, phasepDB_highthroughput. 

You can also visit our [website](http://www.ai-phasepro.pro/) for the whole datasets

## Run Code

### Install Requirement

Code run with python=3.8&torch=2.1.1+cu118

~~~python
conda --name protein --file requirements.txt
~~~

### To run our model

~~~python
mkdir classification_output/dataset_RCNN_ECA_output/mydata_all_1507_output/RCNN_ECA_em1024_128_32_output
cd RCNN_model # output for mydata
python RCNN_ECA_3_mydata.py
~~~

### Run with other datasets

```python
mkdir classification_output/dataset_RCNN_ECA_output/LLPS_output/r3/RF_output # output for LLPS
mkdir classification_output/dataset_RCNN_ECA_output/PhasepDB_Reviewed_output/RCNN_ECA_em1024_128_32_output # output for PhasepDB_Reviewed
mkdir classification_output/dataset_RCNN_ECA_output/PhasepDB_high_throughput_output/RCNN_ECA_em1024_hidden128_128_32_output # output for PhasepDB_high_throughput
cd RCNN_model
python RCNN_ECA_3_LLPS.py # for LLPS
python RCNN_ECA_3_R.py    # for PhasepDB_Reviewed
python RCNN_ECA_3_high.py # for PhasepDB_high_throughput
```

### Saliency

__save LRECA model__

```python
cd ECNN_ECA_saliency/saliency_function
python RCNN_ECA_save_model.py
```

__Get protein score and statics result__

```python
cd ECNN_ECA_saliency/saliency_function/method
python RCNN_ECA_sliency_gradCAM.py
```

__process statics result (t-test、rank-sum-test)__

```python
cd ECNN_ECA_saliency/saliency_function/statics
├─statics
│      RCNN_ECA_statics2.py
│      RCNN_ECA_statics2_1.py
│      RCNN_ECA_statics3.py
│      RCNN_ECA_statics3_1.py
│      RCNN_ECA_statics4.py
│      RCNN_ECA_statics4_1.py
│      RCNN_ECA_statics5.py
│      RCNN_ECA_statics5_1.py
│      RCNN_ECA_statics_length_distribution.py
```

__LCRs split__

```python
cd ECNN_ECA_saliency/LCRs_process
python split_LCRs_segment.py
```

