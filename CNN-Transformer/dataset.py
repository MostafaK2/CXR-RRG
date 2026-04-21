import h5py
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

import logging


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
    def __init__(self, df: pd.DataFrame, h5_path: str, vocab, bos, eos, unk, tokenizer = None, transform = None):
        self.df       = df
        self.indices  = df.index.tolist()  # original CSV index = h5 position
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
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        h5_idx = self.indices[idx]  # original index → correct h5 position

        # Getting the img (non normalized) and normalizing it (Image net)
        img = torch.tensor(self._get_h5()['images'][h5_idx])  # (1, 224, 224)
        if self.transform:
            img = self.transform(img)

        report = row[self.impression_col]  # adjust to your exact column name
        
        if self.tokenizer: 
            encodings = self.tokenizer.encode(report)
            report = self._add_bos_eos(encodings.ids)
        
        report = torch.tensor(report, dtype=torch.long)
        src_seq  = report[:-1]   # x[:, 0:T-1]
        tgt_seq = report[1:]

        return img, src_seq, tgt_seq
