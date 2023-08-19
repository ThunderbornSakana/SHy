# SHy
Implementation for the paper: Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction.

## Requirements
- Python 3.9.13
- PyTorch 1.13.1
- Pyro 1.8.4
- torch_scatter 2.1.0+pt113cu116
- torch_sparse 0.6.16+pt113cu116
- torch_geometric 2.2.0
- DHG 0.9.3

## Data Downloading and Preprocessing
First, make the following directories:
- `./data/RAW/MIMIC_III`
- `./data/MIMIC_III`
- `./data/MIMIC_IV/binary_test_x_slices`
- `./data/MIMIC_IV/binary_train_x_slices`
### Experiments on MIMIC-III
Download the following MIMIC-III data files from [PhysioNet](https://physionet.org/content/mimiciii/1.4/) to the directory `./data/RAW/MIMIC_III`:
- ADMISSIONS.csv
- DIAGNOSES_ICD.csv
- D_ICD_DIAGNOSES.csv

Run the preprocessing notebook for MIMIC-III: `./src/iii_preprocessing.ipynb`.

### Experiments on MIMIC-IV
Download the following MIMIC-IV data files from [PhysioNet](https://physionet.org/content/mimiciv/1.0/) to the directory `./data/RAW/MIMIC_IV`:
- admissions.csv
- diagnoses_icd.csv
- d_icd_diagnoses.csv

Run the preprocessing notebook for MIMIC-IV: `./src/iv_preprocessing.ipynb`.

## Model Training and Evaluation
First, make the following directories:
- `./saved_models`
- `./training_logs`
### Experiments on MIMIC-III
In `./src`, run the following command:
```bash
python -u main.py --temperature 1.0 1.0 1.0 1.0 1.0 --add_ratio 0.2 0.2 0.2 0.2 0.2 --loss_weight 1.0 0.003 0.00025 0.0 0.04
```
The model checkpoint at each epoch will be saved in `./saved_models`. When the training is done, the results will be saved in `./training_logs`.

### Experiments on MIMIC-IV
In `./src`, run the following command:
```bash
python -u main.py --dataset_name 'MIMIC_IV' --temperature 1.0 1.0 1.0 1.0 1.0 --add_ratio 0.2 0.2 0.2 0.2 0.2 --loss_weight 1.0 0.003 0.00025 0.0 0.04
```
The model checkpoint at each epoch will be saved in `./saved_models`. When the training is done, the results will be saved in `./training_logs`.
