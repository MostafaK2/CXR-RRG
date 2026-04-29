# Radiology Report Generation

This project explores three approaches to generating radiology reports from chest X-ray images:

1. **DenseNet121 → Transformer Decoder**
2. **DenseNet121 + BioClinicalBERT → Cross-Attention Fusion → Transformer Decoder**
3. **Swin-S + BioClinicalBERT → Cross-Attention Fusion → Transformer Decoder**

---

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16 GB+ RAM

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/TODO/your-repo.git
cd your-repo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

### Download CheXpert Plus

- Find the dataset at: `https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x?v=next` or `https://aimi.stanford.edu/datasets/chexpert-plus`
- Download all CSV files and save them to disk:
  - `report_fixed.json`, `impression_fixed.json`, `findings_fixed.json`, `df_chexpert_plus_240401`

> **License:** Stanford Research Usage Agreement

### Preprocessing Pipeline

**Step 1.** Extract and resize images from the dataset:

```bash
python extract.py <name_of_your_zipped_file>
```
This will create a **/PNG folder** with the dataset in there. You dont have to touch this folder. Place it anywhere you'd like.


**Step 2.** Update the preprocessing configuration file:
<br> From the root, run
```bash
cd preprocess && python preprocess.py
```
or do it manually
```bash
cd preprocess
python preprocess.py
```

This will generate the following output structure:

```
filtered_data/         # Filtered CSV files
├── train/
│   ├── train.csv
|   ├── train_cleaned.csv
│   └── train.h5
└── valid/
    ├── valid.csv
    ├── valid_cleaned.csv
    └── valid.h5
    
```

After preprocessing, update the data paths in your configuration files to point to these outputs.

---

## CheXbert Evaluation (Optional)

This step is only required if you want to run label-based F1 evaluation at the end.

**Step 1.** Download the CheXbert model weights: 

Note down the location you downloaded the weights

```bash
wget https://huggingface.co/StanfordAIMI/RRG_scorers/resolve/main/chexbert.pth
```
---

## Training

### Prerequisites
Configurations files for each model is located at: 
- Baseline: `configs/captioner-conf/main.yml`
- Multimodal Densenet: `configs/multimodal_conf/main.yml`
- Multimodal Densenet: `configs/multimodal_swin/main.yml`

```
eval:
  chestXbertModelWeights: <UPDATE>
  findings_label_path: <UPDATE>
  impression_label_path: <UPDATE>
  reports_label_path: <UPDATE>

data:
  csv_file: <UPDATE>
  h5_file: <UPDATE>

checkpoint:
  model_checkpoint_path: <UPDATE>
  model_save_name: <UPDATE>
```

### Running Training
Baseline
```bash
python exp1_baseline_captioning/train.py 
python exp2_multimodal/train.py 
python exp3_multimodal_swin/train.py 
```
