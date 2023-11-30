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

