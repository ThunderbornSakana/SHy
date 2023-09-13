# Fall 2023 COMP 576 Assignment 0 Report
Student: Leisheng Yu

## Task 1
```bash
     active environment : base
    active env location : /Users/leishengyu/opt/anaconda3
            shell level : 1
       user config file : /Users/leishengyu/.condarc
 populated config files : /Users/leishengyu/.condarc
          conda version : 4.14.0
    conda-build version : 3.21.5
         python version : 3.9.7.final.0
       virtual packages : __osx=10.16=0
                          __unix=0=0
                          __archspec=1=x86_64
       base environment : /Users/leishengyu/opt/anaconda3  (writable)
      conda av data dir : /Users/leishengyu/opt/anaconda3/etc/conda
  conda av metadata url : None
           channel URLs : https://repo.anaconda.com/pkgs/main/osx-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/osx-64
                          https://repo.anaconda.com/pkgs/r/noarch
          package cache : /Users/leishengyu/opt/anaconda3/pkgs
                          /Users/leishengyu/.conda/pkgs
       envs directories : /Users/leishengyu/opt/anaconda3/envs
                          /Users/leishengyu/.conda/envs
               platform : osx-64
             user-agent : conda/4.14.0 requests/2.26.0 CPython/3.9.7 Darwin/22.5.0 OSX/10.16
                UID:GID : 501:20
             netrc file : /Users/leishengyu/.netrc
           offline mode : False
```

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
