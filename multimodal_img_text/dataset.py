import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import logging

from utils.metrics import CHEXBERT_LABELS

def load_and_split(csv_path: str, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42, logger: logging = None):
    """
    Splits by unique patient ID to prevent data leakage.
    Same patient will never appear in more than one split.
    """
    logger.info(f"Dataset is being split into train-test-split ({1 - val_size}, {val_size}, {test_size})")
    df = pd.read_csv(csv_path)
    patients = df['deid_patient_id'].unique().tolist()
    

    # First split off test set
    train_val_patients, test_patients = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed
    )

    # Then split remaining into train/val
    relative_val_size = val_size / (1.0 - test_size)  # adjust val proportion
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=relative_val_size,
        random_state=seed
    )

    train_df = df[df['deid_patient_id'].isin(train_patients)]
    val_df   = df[df['deid_patient_id'].isin(val_patients)]
    test_df  = df[df['deid_patient_id'].isin(test_patients)]

    logger.info(f"Patients — train: {len(train_patients)} | val: {len(val_patients)} | test: {len(test_patients)}")
    logger.info(f"Samples  — train: {len(train_df)}     | val: {len(val_df)}     | test: {len(test_df)}")

    return train_df, val_df, test_df



class CXRDataset(Dataset):
    def __init__(self, 
                 df_reports: pd.DataFrame,
                 df_labels: pd.DataFrame, 
                 h5_path: str, 
                 vocab: dict, 
                 bos: str, 
                 eos: str, 
                 unk: str,
                 finding: str,
                 impression: str, 
                 tokenizer = None, 
                 transform = None):
        
        self.df_reports = df_reports
        self.indices  = df_reports.index.tolist()  # original CSV index = h5 position

        self.df_labels = df_labels.set_index("path_no_ext")  # easy lookup

        self.h5_path  = h5_path
        self.transform = transform
        self.tokenizer = tokenizer
        self.h5_file  = None  # lazy open for multiprocessing safety

        self.vocab = vocab
        self.bos = bos
        self.eos = eos
        self.unk = unk

        self.impression_col = "section_impression_gen"
        self.findings_col = "section_finding_gen"

        self.clinical_his_col = "clinical_history"
 

    # edit to add special tokens like <impression> <findings> and contruct it 
    def _add_bos_eos(self, token_ids):
       ids = [self.vocab[self.bos]] + token_ids + [self.vocab[self.eos]]
       return ids

    def _get_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        return self.h5_file
    
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

    def __len__(self):
        return len(self.df_reports)

    def __getitem__(self, idx):
        row    = self.df_reports.iloc[idx]  # Row from the dataframe
        h5_idx = self.indices[idx]  # original index → correct h5 position

        # Label Dataframe
        path_no_ext = row["path_no_ext"]
        label_row = self.df_labels.loc[path_no_ext, CHEXBERT_LABELS]
        labels = torch.tensor(label_row.to_numpy(), dtype=torch.float32)

        # Getting the img (non normalized) and normalizing it (Image net)
        img = torch.tensor(self._get_h5()['images'][h5_idx])  # (1, 224, 224)
        if self.transform:
            img = self.transform(img)


        # Only doing impression NOW (TEST) --> change to Findings + Impression
        report = row[self.impression_col]  # adjust to your exact column name
        if self.tokenizer: 
            encodings = self.tokenizer.encode(report)
            report = self._add_bos_eos(encodings.ids)
        report = torch.tensor(report, dtype=torch.long)
        src_seq  = report[:-1]   # Everyting except last, shifted by one
        tgt_seq = report[1:]     # Everything except first, shifted by one   


        # Everything outs
        img = img
        src_seq = src_seq # Teacher forcing
        tgt_seq = tgt_seq # output target
        labels = labels   # additional supervision with labels
        
        clinical_his_texts = row[self.clinical_his_col] if not pd.isna(row[self.clinical_his_col]) else ""

        return img, src_seq, tgt_seq, clinical_his_texts, labels
    

# if __name__ == "__main__":
#     findings_label_path = "/home/public/mkamal/dataset/filtered_data/finding_labels.csv"
#     impression_label_path = "/home/public/mkamal/dataset/filtered_data/impression_labels.csv"
#     reports_label_path = "/home/public/mkamal/dataset/filtered_data/reports_label.csv"
#     reports_csv_file =  "/home/public/mkamal/dataset/filtered_data/train/train_cleaned.csv"
    
#     import pandas as pd
#     import numpy as np
#     import random
#     reports_df = pd.read_csv(reports_csv_file)

#     def reorder_labels_df(path: str) -> pd.DataFrame:
#         labels_df = pd.read_csv(path)
#         CHEXBERT_LABELS = [
#             "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
#             "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
#             "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
#             "Support Devices", "No Finding",
#         ]
#         mapping = {
#             np.nan: 0,   # NaN -> 0
#             1.0: 1,      # 1.0 -> 1
#             0.0: 0,      # 0.0 -> 2
#             -1.0: 0      # -1.0 -> 3

#         }
#         for col in CHEXBERT_LABELS:
#             labels_df[col] = labels_df[col].map(mapping)
#         # Since map does not directly handle NaN keys well, fix NaNs separately
#         for col in CHEXBERT_LABELS:
#             labels_df[col] = labels_df[col].fillna(0).astype(int)

#         return labels_df
    
#     label_df = reorder_labels_df(reports_label_path)
    
#     from transformers import PreTrainedTokenizerFast

#     tokenizer = PreTrainedTokenizerFast(
#         tokenizer_object=None,
#         bos_token="<bos>",
#         eos_token="<eos>",
#         unk_token="<unk>",
#         pad_token="<pad>"
#     )

#     vocab = {
#         "<bos>": 1,
#         "<eos>": 2,
#         "<unk>": 3,

#     }

#     ds = CXRDataset(
#         reports_df,
#         label_df,
#         "/home/public/mkamal/dataset/filtered_data/train/train.h5",
#         vocab, 
#         "<bos>", 
#         "<eos>",
#         "<unk>",
#         tokenizer=tokenizer,

#     )

#     print(ds[0])


