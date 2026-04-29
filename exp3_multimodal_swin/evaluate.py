import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import shutil

from utils.config import load_config, _find_root
from utils.metrics import evaluate_metric_batched, evaluate_metric_batched_for_error_analysis, generation_diversity, hallucination_rate

import torch
import numpy as np
import pandas as pd
import random

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tokenizers import Tokenizer

from exp2_multimodal.dataset import load_and_split, CXRDataset
from model import Multimodal_Memory

from utils.logginghelpers import log_chexbert_f1_summary, save_training_results

# ----------------------- LOGGING ---------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/evaluate.log", mode='w')
    ]
)
logger = logging.getLogger()


# ----------------------- CONSTANTS -------------------
CHEXBERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding",
]

# Your current zero / near-zero F1 classes from the latest results
RARE_CLASSES = [
    "Enlarged Cardiomediastinum",  # F1: 0.000
    "Lung Lesion",                 # F1: 0.000
    "Pneumonia",                   # F1: 0.000
    "Fracture",                    # F1: 0.000
    "Pneumothorax",                # F1: 0.060
    "Pleural Other",               # F1: 0.091
    "Consolidation",               # F1: 0.089
]


# ----------------------- RARE CLASS FILTER -----------
def filter_rare_class_samples(df: pd.DataFrame,
                               label_df: pd.DataFrame,
                               rare_classes: list = None,
                               min_samples_per_class: int = None) -> pd.DataFrame:
    """
    Returns the subset of df whose reports contain at least one
    of the rare / low-F1 findings.

    Args:
        df:                    your split df (valid_df / test_df)
        label_df:              CheXBert label DataFrame
        rare_classes:          list of label column names to filter on.
                               Defaults to RARE_CLASSES above.
        min_samples_per_class: if set, caps samples per class so the
                               eval set stays balanced across rare classes.
    Returns:
        Filtered DataFrame — only rows that have at least one rare finding.
    """
    if rare_classes is None:
        rare_classes = RARE_CLASSES

    aligned   = label_df.loc[label_df.index.isin(df.index)]
    rare_mask = aligned[rare_classes].any(axis=1)
    rare_idx  = aligned[rare_mask].index
    filtered  = df.loc[df.index.isin(rare_idx)].copy()

    logger.info(f"Rare class filter: {len(filtered)}/{len(df)} samples kept")
    for cls in rare_classes:
        if cls in aligned.columns:
            n = int(aligned.loc[rare_idx, cls].sum())
            logger.info(f"  {cls:35s}: {n} positive samples")

    if min_samples_per_class is not None:
        keep_idx = set()
        for cls in rare_classes:
            if cls not in aligned.columns:
                continue
            cls_pos = aligned.loc[rare_idx][aligned.loc[rare_idx, cls] == 1].index
            keep_idx.update(cls_pos[:min_samples_per_class].tolist())
        filtered = df.loc[df.index.isin(keep_idx)].copy()
        logger.info(f"  After per-class cap ({min_samples_per_class}): "
                    f"{len(filtered)} samples")

    return filtered


# ----------------------- ARGS ------------------------
def get_args():
    ap = argparse.ArgumentParser(description="CXR Report Generator — Evaluation Only")
    ap.add_argument('--config',      type=str,  default=None,
                    help='Path to config file')
    ap.add_argument('--checkpoint',  type=str,  default=None,
                    help='Path to .pt checkpoint to evaluate')
    ap.add_argument('--split',       type=str,  default='valid',
                    choices=['valid', 'test', 'both'],
                    help='Which split to evaluate')
    ap.add_argument('--num_samples', type=int,  default=None,
                    help='Number of samples for CheXBert eval (ignored with --rare_only)')
    ap.add_argument('--device',      type=str,  default=None,
                    help='cuda / cpu / mps')
    ap.add_argument('--seed',        type=int,  default=None)
    ap.add_argument('--rare_only',   action='store_true',
                    help='Evaluate only on samples containing rare/low-F1 findings')
    ap.add_argument('--rare_cap',    type=int,  default=None,
                    help='Max samples per rare class when using --rare_only')
    return ap.parse_args()


# ----------------------- HELPERS ---------------------
def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_sequence(sequences, max_len, pad_value):
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        padded[i, :seq_len] = seq[:seq_len]
    return padded


def reorder_labels_df(path: str) -> pd.DataFrame:
    labels_df = pd.read_csv(path)
    mapping = {np.nan: 0, 1.0: 1, 0.0: 0, -1.0: 0}
    for col in CHEXBERT_LABELS:
        labels_df[col] = labels_df[col].map(mapping).fillna(0).astype(int)
    return labels_df


# ----------------------- MAIN ------------------------
def main():
    args = get_args()

    DEFAULT = os.path.join(_find_root(), 'configs', 'multimodal_label_conf', 'main.yml')
    config  = load_config(args.config, default_config=DEFAULT, logger=logger)

    if args.device:
        config['training']['device'] = args.device
    if args.seed:
        config['reproducibility']['seed'] = args.seed

    DEVICE = args.device or pick_device()
    SEED   = config['reproducibility']['seed']
    seed_everything(SEED)
    logger.info(f"Device: {DEVICE} | Seed: {SEED}")

    # ── Special tokens ────────────────────────────────────────────────────
    BOS        = config['special_tokens']['bos']
    EOS        = config['special_tokens']['eos']
    PAD        = config['special_tokens']['pad']
    UNK        = config['special_tokens']['unk']
    FINDING    = config['special_tokens']['finding']
    IMPRESSION = config['special_tokens']['impression']

    # ── Data split ────────────────────────────────────────────────────────
    train_df, valid_df, test_df = load_and_split(
        config["data"]["csv_file"],
        val_size  = config["data"]["valid_sz"],
        test_size = config["data"]["test_sz"],
        seed      = SEED,
        logger    = logger
    )

    # ── Tokenizer — load from checkpoint save_dir ─────────────────────────
    BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
    _save_dir      = config['checkpoint']['save_dir']
    if not os.path.isabs(_save_dir):
        _save_dir  = os.path.join(BASE_DIR, _save_dir)
    tokenizer_path = os.path.join(_save_dir, "bpe_tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    word2idx  = tokenizer.get_vocab()
    config["model"]["vocab_size"] = tokenizer.get_vocab_size()
    config["model"]["pad_id"]     = word2idx[PAD]
    logger.info(f"Tokenizer loaded from {tokenizer_path} | "
                f"Vocab: {tokenizer.get_vocab_size()}")

    # ── collate_fn ────────────────────────────────────────────────────────
    def collate_fn(batch):
        img_tens, src_seqs, tgt_seqs, clinical_text, labels = zip(*batch)
        img_tens   = torch.stack(img_tens)
        max_src    = max(len(s) for s in src_seqs)
        max_tgt    = max(len(t) for t in tgt_seqs)
        padded_src = pad_sequence(src_seqs, max_src, word2idx[PAD])
        padded_tgt = pad_sequence(tgt_seqs, max_tgt, word2idx[PAD])
        return img_tens, padded_src, padded_tgt, list(clinical_text), torch.stack(labels)

    # ── Transform ─────────────────────────────────────────────────────────
    imagenet_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    label_df = reorder_labels_df(config['eval']['reports_label_path'])

    # ── Rare class filter ─────────────────────────────────────────────────
    if args.rare_only:
        logger.info("=== RARE CLASS MODE: filtering to low-F1 findings only ===")
        valid_df = filter_rare_class_samples(
            valid_df, label_df, min_samples_per_class=args.rare_cap
        )
        test_df  = filter_rare_class_samples(
            test_df,  label_df, min_samples_per_class=args.rare_cap
        )

    # ── Build datasets ────────────────────────────────────────────────────
    ds_kwargs = dict(
        df_labels       = label_df,
        h5_path         = config["data"]["h5_file"],
        vocab           = word2idx,
        bos             = BOS, eos = EOS, unk = UNK,
        finding         = FINDING, impression = IMPRESSION,
        decoder_max_len = config['model']['decoder_max_len'],
        tokenizer       = tokenizer,
        transform       = imagenet_transform,
    )
    valid_ds = CXRDataset(df_reports=valid_df, **ds_kwargs)
    test_ds  = CXRDataset(df_reports=test_df,  **ds_kwargs)

    BATCH    = config['training']['batch_size']
    valid_dl = DataLoader(valid_ds, batch_size=BATCH, shuffle=False,
                          collate_fn=collate_fn, num_workers=4)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                          collate_fn=collate_fn, num_workers=4)

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt_path = (args.checkpoint or
                 os.path.join(config['checkpoint']['model_checkpoint_path'],
                              config['checkpoint']['model_save_name']))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    hp    = ckpt['hyperparams']
    model = Multimodal_Memory(**hp).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logger.info(f"Checkpoint: epoch {ckpt['epoch']+1}, "
                f"valid loss {ckpt['valid_loss']:.4f}")

    # ── Evaluate ──────────────────────────────────────────────────────────---------------------------------------------

    split      = args.split
    n_valid    = len(valid_df) if args.rare_only else args.num_samples
    n_test     = len(test_df)  if args.rare_only else args.num_samples

    valid_bleu = valid_meteor = valid_chexbert = None
    test_bleu  = test_meteor  = test_chexbert  = None

    if split in ('valid', 'both'):
        logger.info(f"─── Validation set ({n_valid} samples) ───")
        valid_bleu, valid_meteor, valid_chexbert, generated_list, reference_list, gt_labels = evaluate_metric_batched_for_error_analysis(
            model, valid_df, valid_ds, tokenizer, word2idx,
            config, DEVICE,
            num_samples = n_valid,
            batch_size=BATCH,
            labels_path = config["eval"]["reports_label_path"]
        )

        result = generation_diversity(generated_list, save_dir=config["checkpoint"]["save_dir"] )
        result = hallucination_rate(generated_list, gt_labels, save_dir=config["checkpoint"]["save_dir"] )

        logger.info(f"Valid BLEU (1/2/4): {valid_bleu}")
        logger.info(f"Valid METEOR:       {valid_meteor:.4f}")
        log_chexbert_f1_summary(valid_chexbert, logger)

    if split in ('test', 'both'):
        logger.info(f"─── Test set ({n_test} samples) ───")
        test_bleu, test_meteor, test_chexbert, generated_list, reference_list, gt_labels = evaluate_metric_batched_for_error_analysis(
            model, test_df, test_ds, tokenizer, word2idx,
            config, DEVICE,
            num_samples = n_test,
            batch_size=BATCH,
            labels_path = config["eval"]["reports_label_path"]
        )
        logger.info(f"Test BLEU (1/2/4): {test_bleu}")
        logger.info(f"Test METEOR:       {test_meteor:.4f}")
        log_chexbert_f1_summary(test_chexbert, logger)

    # ── Save results ──────────────────────────────────────────────────────
    save_path = _save_dir
    os.makedirs(save_path, exist_ok=True)

    class _FakeConf:
        def __init__(self, c):
            self.D_MODEL              = c['model']['d_model']
            self.DROPOUT              = c['model']['dropout']
            self.IMG_ENC_BACKBONE     = c['model']['img_enc_backbone']
            self.BERT_MODEL           = c['model']['bert_model']
            self.EPOCHS               = c['training']['epochs']
            self.BATCH_SIZE           = c['training']['batch_size']
            self.LR                   = c['training']['learning_rate']
            self.GRAD_CLIP            = c['training']['grad_clip']
            self.WEIGHT_DECAY         = c['training']['weight_decay']
            self.PATIENCE             = c['training']['patience']
            self.LABEL_SMOOTHING      = c['training']['label_smoothing']
            self.SAVE_DIR             = c['checkpoint']['save_dir']
            self.MODEL_CHKPT_SAVE_DIR = c['checkpoint']['model_checkpoint_path']

    save_training_results(
        save_path          = save_path,
        conf               = _FakeConf(config),
        model              = model,
        best_epoch         = ckpt['epoch'],
        best_valid_loss    = ckpt['valid_loss'],
        valid_corpus_bleu  = valid_bleu,
        valid_meteor_score = valid_meteor,
        valid_chexpert_f1s = valid_chexbert,
        test_corpus_bleu   = test_bleu,
        test_meteor_score  = test_meteor,
        test_chexpert_f1s  = test_chexbert,
        train_losses       = [],
        valid_losses       = [],
    )

    # Save log — use distinct name in rare mode
    for handler in logger.handlers:
        handler.flush()
    log_name  = "evaluate_rare.log" if args.rare_only else "evaluate.log"
    if os.path.exists("logs/evaluate.log"):
        shutil.copy("logs/evaluate.log", os.path.join(save_path, log_name))

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()