# Multimodal RBD Sleep Analysis

Classification of polysomnography recordings from the [CAP Sleep Database](https://physionet.org/content/capslpdb/1.0.0/) into healthy, other sleep disorders and REM sleep behavior disorder (RBD). The features are computed from EEG, EOG and chin EMG. Sleep stages are scored automatically with [YASA](https://github.com/raphaelvallat/yasa) and [GSSC](https://github.com/jshanna100/gssc) and compared with the manual scoring of the database.

## Installation

```bash
pip install -r requirements.txt
```

## Data

```bash
python scripts/download_data.py
python scripts/organize_data.py
```

The recordings are stored in `data/capslpdb`. The dataset is only read by the notebooks 01 and 02.

The notebooks 03 and 04 use the CSV files in `outputs`, so the download is not needed to reconstruct the results.

## Usage

Run the notebooks in `notebooks` in order, from 01 to 04.

## Project structure

```
notebooks/
├── 01_data_quality.ipynb            recording checks and channel selection
├── 02_preprocessing.ipynb           sleep staging and feature extraction
├── 03_split_and_distribution.ipynb  data splits and feature distributions
├── 04_model_comparison.ipynb        model comparison, training and test
├── edf_quality.py                   EDF reading and signal quality checks
├── sleep_staging.py                 sleep staging with YASA and GSSC
└── result_cache.py                  cache for results per recording
scripts/                             download and organization of the dataset
outputs/
├── 01_data_quality/                 channel selections
├── 02_preprocessing/                features, sleep stages and figures
├── 03_split_and_distribution/       splits and figures
└── 04_model_comparison/             figures and trained models
```
