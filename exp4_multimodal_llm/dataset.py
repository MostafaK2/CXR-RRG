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


class Dataset_for_llm_model(Dataset):
    def __init__(self, 
                 df_reports: pd.DataFrame,
                 df_labels: pd.DataFrame, 
                 h5_path: str, 
                 tokenizer,
                 max_len: int = 256,
                 transform=None):
        
        self.df_reports = df_reports
        self.indices    = df_reports.index.tolist() # original CSV index = h5 position
        self.df_labels  = df_labels.set_index("path_no_ext")  # easy lookup
        self.h5_path    = h5_path
        self.transform  = transform
        self.tokenizer  = tokenizer
        self.max_len    = max_len
        self.h5_file    = None

        self.impression_col   = "section_impression_gen"
        self.findings_col     = "section_findings_gen"
        self.clinical_his_col = "clinical_history"
 

    def _construct_report(self, finding_text, impression_text):
        has_finding = not pd.isna(finding_text)
        if has_finding:
            text = f"Findings: {finding_text} Impression: {impression_text}"
        else:
            text = f"Impression: {impression_text}"
        return text
    

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

        # Image
        img = torch.tensor(self._get_h5()['images'][h5_idx])
        if self.transform:
            img = self.transform(img)

        # CheXbert labels
        path_no_ext = row["path_no_ext"]
        label_row   = self.df_labels.loc[path_no_ext, CHEXBERT_LABELS]
        labels      = torch.tensor(label_row.to_numpy(), dtype=torch.float32)

        # Report — BioGPT tokenizer handles BOS/EOS/shifting
        report_text = self._construct_report(
            row[self.findings_col],
            row[self.impression_col]
        )

        tokenized = self.tokenizer(
            report_text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        clinical_history = row[self.clinical_his_col] if not pd.isna(row[self.clinical_his_col]) else ""

        return img, tokenized.input_ids.squeeze(0), tokenized.attention_mask.squeeze(0), clinical_history, labels
    