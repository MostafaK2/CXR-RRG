import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import math
import shutil

from utils.config import load_config, _find_root
from utils.metrics import evaluate_metric, evaluate_metric_batched

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
import torch.nn.functional as F

import torchvision.transforms as transforms
import pandas as pd
import numpy as np

import tqdm

import random

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from exp2_multimodal.dataset import load_and_split, CXRDataset

import yaml

from model import Multimodal_Memory

from utils.logginghelpers import log_chexbert_f1_summary, save_training_results

import matplotlib.pyplot as plt


# ----------------------- LOGGING ---------------------

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    handlers=[
        logging.StreamHandler(),                 # print to console
        logging.FileHandler("logs/exp3_ablation_heads.log", mode='w')    # save to file
    ]
)
logger = logging.getLogger()

# --------------------- Parsing Arguments ---------- NEED HEAVY EDITING FOR NOW JUST KEEP IT as Load config works
def get_args():
    ap = argparse.ArgumentParser(description="Chest X-ray Report Generator Training")

    # ---- Config ----
    ap.add_argument('--config', type=str, default=None, help='Path to config file')

    # ---- Data ----
    ap.add_argument('--data_dir',    type=str, default=None, help='Path to dataset')
    ap.add_argument('--num_workers', type=int, default=None, help='Dataloader workers')
    ap.add_argument('--max_len',     type=int, default=None, help='Max caption token length')

    # ---- Model ----
    ap.add_argument('--d_model',  type=int,   default=None, help='Model dimension')
    ap.add_argument('--n_heads',  type=int,   default=None, help='Attention heads')
    ap.add_argument('--n_layers', type=int,   default=None, help='Decoder layers')
    ap.add_argument('--d_ff',     type=int,   default=None, help='Feedforward dimension')
    ap.add_argument('--dropout',  type=float, default=None, help='Dropout rate')

    # ---- Training ----
    ap.add_argument('--epochs',       type=int,   default=None,  help='Number of epochs')
    ap.add_argument('--batch_size',   type=int,   default=None,  help='Batch size')
    ap.add_argument('--lr',           type=float, default=None,  help='Learning rate')
    ap.add_argument('--decay',        type=float, default=None,  help='Weight decay')
    ap.add_argument('--grad_clip',    type=float, default=None,  help='Gradient clipping')
    ap.add_argument('--warmup_steps', type=int,   default=None,  help='LR warmup steps')
    ap.add_argument('--seed',         type=int,   default=None,  help='Random seed')
    ap.add_argument('--device',       type=str,   default=None,  help='cuda or cpu')
    ap.add_argument('--resume',       type=str,   default=None,  help='Checkpoint path to resume from')

    # ---- Logging ----
    ap.add_argument('--save_dir',   type=str, default=None, help='Checkpoint save directory')
    ap.add_argument('--run_name',   type=str, default=None, help='Run name for wandb/tensorboard')
    ap.add_argument('--log_every',  type=int, default=None, help='Log loss every N steps')
    ap.add_argument('--eval_every', type=int, default=None, help='Evaluate every N epochs')

    return ap.parse_args()

def override_config(args, config):
    overrides = {
        # Training
        'epochs':       ('training',        'epochs'),
        'batch_size':   ('training',        'batch_size'),
        'lr':           ('training',        'learning_rate'),
        'decay':        ('training',        'weight_decay'),
        'grad_clip':    ('training',        'grad_clip'),
        'warmup_steps': ('training',        'warmup_steps'),
        'device':       ('training',        'device'),
        # Data
        'data_dir':     ('data',            'data_dir'),
        'num_workers':  ('data',            'num_workers'),
        'max_len':      ('data',            'max_len'),
        # Model
        'd_model':      ('model',           'd_model'),
        'n_heads':      ('model',           'n_heads'),
        'n_layers':     ('model',           'n_layers'),
        'd_ff':         ('model',           'd_ff'),
        'dropout':      ('model',           'dropout'),
        # Reproducibility
        'seed':         ('reproducibility', 'seed'),
        # Checkpoint
        'save_dir':     ('checkpoint',      'save_dir'),
        'resume':       ('checkpoint',      'resume'),
        # Logging
        'run_name':     ('logging',         'run_name'),
        'log_every':    ('logging',         'log_every'),
        'eval_every':   ('logging',         'eval_every'),
    }

    for arg_name, (section, key) in overrides.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            config[section][key] = val
            logger.info(f"[CLI Override] {section}.{key}: {val}")

    return config
args = get_args()

DEFAULT = os.path.join(_find_root(), 'configs', 'multimodal_swin', 'main.yml')
config  = load_config(args.config, default_config=DEFAULT, logger=logger)
config = override_config(args,config)



# ---------------- Device ----------------  COPY THIS
def pick_device():
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon / Metal
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = pick_device()
logger.info(f"Using device : {DEVICE}")

# ---------------- Reproducibility ---------------- COPY THIS
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = config["reproducibility"]["seed"]
seed_everything(config["reproducibility"]["seed"])

# --------------------------------  Preprocessing and splitting the data ------------------------------------

BOS        = config['special_tokens']['bos']
EOS        = config['special_tokens']['eos']
PAD        = config['special_tokens']['pad']
UNK        = config['special_tokens']['unk']
FINDING    = config['special_tokens']['finding']
IMPRESSION = config['special_tokens']['impression']

def build_tokenizer(train_df, caption_col, max_len, special_tokens, vocab_size=10000):
    """
    Trains BPE tokenizer on train split only.
    """
    logger.info("Training BPE tokenizer...")

    _, _, _, unk_token, _, _ = special_tokens

    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    train_captions = train_df[caption_col].tolist()
    tokenizer.train_from_iterator(train_captions, trainer=trainer)

    word2idx = tokenizer.get_vocab()
    logger.info(f"Vocabulary size: {len(word2idx)}")

    return word2idx, tokenizer

train_df, valid_df, test_df = load_and_split(config["data"]["csv_file"], val_size=config["data"]["valid_sz"], test_size=config["data"]["test_sz"], seed=SEED, logger=logger)

word2idx, tokenizer = build_tokenizer(   # just trains the tokenizer
    train_df,
    caption_col="section_impression_gen",
    max_len=config["data"]["max_len"],
    special_tokens=[PAD, BOS, EOS, UNK, FINDING, IMPRESSION],
    vocab_size=config['model']['vocab_size']
)

logger.info(f"Vocab size is {tokenizer.get_vocab_size()}")
logger.info("Tokenizer has been completed")
logger.info("Updating Configuration...")

config["model"]["vocab_size"] = tokenizer.get_vocab_size()
config["model"]["pad_id"] = word2idx[PAD]
logger.info("  configuration updated")
logger.info(
    f"PAD: {word2idx[PAD]}  "
    f"BOS: {word2idx[BOS]}  "
    f"EOS: {word2idx[EOS]}  "
    f"UNK: {word2idx[UNK]}  "
    f"FINDING: {word2idx[FINDING]}  "
    f"IMPRESSION: {word2idx[IMPRESSION]}"
)


# --------------------------------------- Dataset Helper Function (EDIT) --------------------------------
def pad_sequence(sequences, max_len, pad_value): 
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), pad_value, dtype=torch.long)

    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        padded[i, :seq_len] = seq[:seq_len]
    return padded

def collate_fn(batch):  
    img_tens, src_seqs, tgt_seqs, clinical_text, labels = zip(*batch)
    img_tens = torch.stack(img_tens) # stack the images (B, 3, 224, 224)

    src_lens = torch.tensor([len(s) for s in src_seqs])
    tgt_lens = torch.tensor([len(t) for t in tgt_seqs])

    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)

    padded_src = pad_sequence(src_seqs, max_src_len, word2idx[PAD])
    padded_tgt = pad_sequence(tgt_seqs, max_tgt_len, word2idx[PAD])

    # new stuff
    clinical_text = list(clinical_text)
    labels = torch.stack(labels)

    return img_tens, padded_src, padded_tgt, clinical_text, labels

def reorder_labels_df(path: str) -> pd.DataFrame:
    labels_df = pd.read_csv(path)
    CHEXBERT_LABELS = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
        "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
        "Support Devices", "No Finding",
    ]
    mapping = {
        np.nan: 0,   # NaN -> 0
        1.0: 1,      # 1.0 -> 1
        0.0: 0,      # 0.0 -> 2
        -1.0: 0      # -1.0 -> 3

    }
    for col in CHEXBERT_LABELS:
        labels_df[col] = labels_df[col].map(mapping)
    # Since map does not directly handle NaN keys well, fix NaNs separately
    for col in CHEXBERT_LABELS:
        labels_df[col] = labels_df[col].fillna(0).astype(int)

    return labels_df

# Training Helpers
# ----------------- Training Helper functions ------------------- 
def train_epoch(model, dataloader, optimizer, criterion, device,  clip_grad=1.0, warmup_scheduler=None):
    model.train()
    total_loss = 0.0
    for img, src, tgt, clincal_text, labels in tqdm.tqdm(dataloader,"train"):
        img, src, tgt = img.to(device), src.to(device), tgt.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        logits = model(img, clincal_text, src)  # (B, T, V)
        B, T, V = logits.shape

        loss = criterion(logits.view(B * T, V), tgt.view(B * T))
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item()
        
        if warmup_scheduler is not None:
            warmup_scheduler.step()
            
    avg_loss = total_loss / len(dataloader)

    nll = float(avg_loss)
    ppl = float(math.exp(nll))
    return nll, ppl

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for img, src, tgt, clinical_text, labels in tqdm.tqdm(dataloader, "Evaluating"):
            img, src, tgt = img.to(device), src.to(device), tgt.to(device)
            labels = labels.to(device)

            logits = model(img, clinical_text, src)  # (B, T, V)
            B, T, V = logits.shape
            
            loss = criterion(logits.view(B * T, V), tgt.view(B * T))
            
            total_loss += loss.item()
    
    nll = total_loss / len(dataloader)
    ppl = math.exp(nll)
    return nll, ppl


# ---------------- MODEL CONFIGURATIONS -------------------
# Creating a config class for better readibility in the code
class Config:
    DEVICE = DEVICE

    #  ------ Datasets Paths ------------------------------------------
    H5_PATH = config["data"]["h5_file"]
    CSV_PATH = config["data"]["csv_file"]

    MIN_FREQ = config['data']['min_freq']

    # ---------- Special tokens --------------------------
    BOS = config['special_tokens']['bos']
    PAD = config['special_tokens']['pad']
    EOS = config['special_tokens']['eos']
    UNK = config['special_tokens']['unk']
    FINDING = config['special_tokens']['finding']
    IMPRESSION = config['special_tokens']['impression']
    # findings, impresssoin add later if needed

    # ------ Model Parameters -------------------------------------------
    VOCAB_SIZE = int(config['model']['vocab_size'])   # Will be set during training
    D_MODEL = int(config['model']['d_model'])

        # Transformer Decoder
    DECODER_N_HEADS = int(config['model']['decoder_n_heads'])
    DECODER_FF_DIM = int(config['model']['decoder_ff_dim'])
    DECODER_MAX_LEN = int(config['model']['decoder_max_len'])
    DECODER_LAYERS = int(config['model']['decoder_layers'])
    PAD_ID = int(config['model']['pad_id'])
       
        # CROSS ATTN FUSION
    FUSION_HEADS = int(config['model']['fusion_heads'])
    FUSION_FF_DIM = int(config['model']['fusion_ff_dim'])
     
        # IMG ENCODER
    IMG_ENC_BACKBONE = str(config['model']['img_enc_backbone'])
    IMG_ENC_FREEZE_LAYER = int(config['model']['img_enc_freeze_layer'])
    USE_FPN = bool(config['model']['use_fpn'])
    FPN_DIM = int(config['model']['fpn_dim'])
    FPN_SCALE = int(config['model']['fpn_scale'])


        # Text encoder
    BERT_MODEL = str(config['model']['bert_model'])
    BERT_FREEZE_LAYER = int(config['model']['bert_freeze_layer'])
    BERT_MAX_LENGTH = int(config['model']['bert_max_length'])
     
     # Dropout
    DROPOUT = float(config['model']['dropout'])

    # --------- Hyperparameters -----------------------------
    EPOCHS = int(config['training']['epochs'])
    BATCH_SIZE = int(config['training']['batch_size'])
    LR = float(config['training']['learning_rate'])
    GRAD_CLIP = float(config['training']['grad_clip'])
    WEIGHT_DECAY =  float(config['training']['weight_decay'])
    PATIENCE = int(config['training']['patience'])
    LABEL_SMOOTHING = float(config['training']['label_smoothing'])
    

    # scheduler
    WARMUP_STEPS = int(config['training']['warmup_steps'])

    # ----------- CHECKPOINTING AND RESULT --------------------
    SAVE_DIR = config['checkpoint']['save_dir']
    MODEL_CHKPT_SAVE_DIR = config['checkpoint']['model_checkpoint_path']


def main():
    # - - - -- - - - - Configurations and results folders - --  -- -- 

    conf = Config()
    logger.info(f"{conf.VOCAB_SIZE}, {tokenizer.get_vocab_size()}, finding: {conf.FINDING}: {word2idx[conf.FINDING]} Impression: {conf.IMPRESSION}: {word2idx[conf.IMPRESSION]}")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_path = conf.SAVE_DIR
    if not os.path.isabs(save_path):
        save_path = os.path.join(BASE_DIR, save_path)

    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, "best_config.yml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saving all results, model configurations, results in {save_path}")

    # - - - - - -  - - - - - - - - - END - - - -- -- - - - -- - - -- - --

    model = Multimodal_Memory(
        d_model=conf.D_MODEL,
        # IMG Encoder
        img_enc_backbone=conf.IMG_ENC_BACKBONE,
        img_enc_freeze_layers=conf.IMG_ENC_FREEZE_LAYER,
        use_fpn=conf.USE_FPN,
        fpn_dim = conf.FPN_DIM,
        fpn_scale=conf.FPN_SCALE,
        # text encoder
        bert_model=conf.BERT_MODEL,
        bert_freeze_layers=conf.BERT_FREEZE_LAYER,
        bert_max_length=conf.BERT_MAX_LENGTH,
        # Fusion
        fusion_heads = conf.FUSION_HEADS,
        fusion_ff_dim = conf.FUSION_FF_DIM,

        # Decoder
        vocab_size = conf.VOCAB_SIZE, # will be set
        decoder_layers = conf.DECODER_LAYERS,
        decoder_heads = conf.DECODER_N_HEADS,
        decoder_ff_dim = conf.DECODER_FF_DIM,
        decoder_max_len = conf.DECODER_MAX_LEN,
        pad_id = conf.PAD_ID,

        dropout = conf.DROPOUT
    )
    model = model.to(conf.DEVICE)

    optimizer = AdamW(model.parameters(), lr=conf.LR, weight_decay=conf.WEIGHT_DECAY)
    optimizer = AdamW([
        # Frozen or slow — pretrained encoders
        {"params": model.image_encoder.parameters(),"lr": conf.LR * 0.1}, # 10x smaller
        {"params": model.text_encoder.parameters(), "lr": conf.LR * 0.1}, # 10x smaller
        # Faster — fusion and decoder are trained from scratch
        {"params": model.fusion.parameters(), "lr": conf.LR},
        {"params": model.decoder.parameters(),"lr": conf.LR},
    ], weight_decay=conf.WEIGHT_DECAY)

    # CE Loss (prev)
    criterion = nn.CrossEntropyLoss(ignore_index=conf.PAD_ID, label_smoothing=conf.LABEL_SMOOTHING)

    # Parameter Calculation
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Model initialized sucessfully")
    logger.info(f"  - Total params:     {total_params:,}")
    logger.info(f"  - Trainable params: {trainable_params:,}")
    logger.info(f"  - Non-trainable:    {total_params - trainable_params:,}")

    # ------------------- Datasets and Dataloader --------------------------------
    imagenet_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # grayscale -> 3 channels
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    ## CHANGE IF NEEDED CHANGE IF NEEDED CHANGE IF NEEDED CHANGE IF NEEDED CHANGE IF NEEDED CHANGE IF NEEDED
    label_df = reorder_labels_df(config['eval']['reports_label_path'])

    ds_kwargs = {
        "df_labels": label_df,
        "h5_path": conf.H5_PATH,
        "vocab": word2idx,
        "bos": conf.BOS,
        "eos": conf.EOS,
        "unk": conf.UNK,
        "finding": conf.FINDING,
        "impression": conf.IMPRESSION,
        "decoder_max_len": conf.DECODER_MAX_LEN,
        "tokenizer": tokenizer,
        "transform": imagenet_transform,
    }

    # Datasets
    train_ds = CXRDataset(df_reports = train_df,**ds_kwargs)
    valid_ds = CXRDataset(df_reports = valid_df, **ds_kwargs)
    test_ds = CXRDataset(df_reports = test_df, **ds_kwargs)

    # Dataloaders
    train_dl = DataLoader(train_ds, batch_size=conf.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8)
    valid_dl = DataLoader(valid_ds, batch_size=conf.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8)
    test_dl = DataLoader(test_ds, batch_size=conf.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8)
    logger.info(f"Datasets initialized | train: {len(train_ds):,} samples | valid: {len(valid_ds):,} samples | test: {len(test_ds):,} samples")
    
    # Scheduler (Warmup + Cosine Anealing): Warmup(~0 -> LR) -> Then -> CosineAnealing (LR  -> ~0)
    #  ----------------  Scheduler  -----------------
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=1/conf.WARMUP_STEPS, 
        end_factor=1.0, 
        total_iters=conf.WARMUP_STEPS)
    
    epoch_by_warmup = (conf.WARMUP_STEPS // len(train_dl))
    remaining_epochs = conf.EPOCHS - (epoch_by_warmup)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs)

    # ----------------  Scheduler End  -----------------
    # Loss Curve lists
    tl_list, vl_list = [], []
    tp_list, vp_list = [], []

    # Stop loss 
    best_valid_loss  = float("inf")
    patience_counter = 0
    patience = conf.PATIENCE
    best_model_save_path = os.path.join(conf.MODEL_CHKPT_SAVE_DIR, config['checkpoint']['model_save_name'])

    from utils.lr_finder import LRFinder

    # # ── LR Finder — run once, then comment out ───────────────────────────────
    # finder = LRFinder(
    #     model     = model,
    #     optimizer = optimizer,
    #     criterion = criterion,
    #     device    = conf.DEVICE,
    # )

    # optimal_lr = finder.find(
    #     dataloader = train_dl,
    #     min_lr     = 1e-7,
    #     max_lr     = 1e-1,
    #     num_steps  = 200,
    #     save_path  = os.path.join(save_path, "lr_finder.png"),
    # )

    # ------------------------------------- TRAINING START ----------------------------------------------------------
    logger.info("======= " + "Starting Training " + ("=" * 60))
    
    for epoch in range(conf.EPOCHS):
        if epoch > epoch_by_warmup:
            warmup_scheduler = None
        
        train_nll, train_ppl = train_epoch(model, train_dl, optimizer, criterion, DEVICE, warmup_scheduler=warmup_scheduler, clip_grad=conf.GRAD_CLIP)
        valid_nll,  valid_ppl  = evaluate(model,valid_dl,criterion,DEVICE)
        
        # Early Stopping
        if valid_nll < best_valid_loss:
            best_valid_loss  = valid_nll
            patience_counter = 0


            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparams': {
                    # Core
                    'd_model':            conf.D_MODEL,
                    'dropout':            conf.DROPOUT,
                    # IMG encoder
                    'img_enc_backbone':       conf.IMG_ENC_BACKBONE,
                    'img_enc_freeze_layers':  conf.IMG_ENC_FREEZE_LAYER,
                    'use_fpn':                conf.USE_FPN,
                    'fpn_dim':                conf.FPN_DIM,
                    'fpn_scale':              conf.FPN_SCALE,

                    # Text encoder
                    'bert_model':         conf.BERT_MODEL,
                    'bert_freeze_layers': conf.BERT_FREEZE_LAYER,
                    'bert_max_length':    conf.BERT_MAX_LENGTH,
                    # Fusion
                    'fusion_heads':       conf.FUSION_HEADS,
                    'fusion_ff_dim':      conf.FUSION_FF_DIM,
                    # Decoder
                    'vocab_size':         conf.VOCAB_SIZE,
                    'decoder_layers':     conf.DECODER_LAYERS,
                    'decoder_heads':      conf.DECODER_N_HEADS,
                    'decoder_ff_dim':     conf.DECODER_FF_DIM,
                    'decoder_max_len':    conf.DECODER_MAX_LEN,
                    'pad_id':             conf.PAD_ID,

                },
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch':                epoch,
                'valid_loss':           best_valid_loss,
            }, best_model_save_path)

            print(f"    At epoch: {epoch+1}, best model saved at {best_model_save_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        if epoch > epoch_by_warmup:
            cosine_scheduler.step()
            
        # Metricss
        tl_list.append(train_nll); vl_list.append(valid_nll)
        tp_list.append(train_ppl);  vp_list.append(valid_ppl)

        logger.info(
            "Epoch %d/%d | Train Loss=%.4f | Train PPL=%.2f | Valid Loss=%.4f | Valid PPL=%.2f",
            epoch + 1,
            conf.EPOCHS,
            train_nll,
            train_ppl,
            valid_nll,
            valid_ppl,
        )

    # ── Plots ─────────────────────────────────────────────────────────────────────
    from utils.plotting import plot_train_validation_curve
    plot_train_validation_curve(tl_list=tl_list, vl_list=vl_list, tp_list=tp_list, vp_list=vp_list, save_path=save_path)
    logger.info(f"Plots saved in /{save_path}/training_curves.png")
    logger.info("Training & Evaluating completed!")
        
        
    # # ── Evaluation Test & Valid Dataset ─────────────────────────────────────────────────────────────────────
    logger.info("======= " + "Starting Evaluating " + ("=" * 60))
    
    ckpt = torch.load(best_model_save_path, weights_only=False)
    hp   = ckpt['hyperparams']
    model = Multimodal_Memory(**hp).to(conf.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']+1} with valid loss {ckpt['valid_loss']:.4f}")


    logger.info("======= Calculating Evaluation BLEU Scores =======")
    cpbleus_valid, avg_valid_meteor, chexbert_res_valid = evaluate_metric_batched(
        model,
        valid_df, 
        valid_ds, 
        tokenizer, 
        word2idx,
        config, 
        conf.DEVICE,
        batch_size=conf.BATCH_SIZE,
        num_samples=None,
        labels_path=config["eval"]["reports_label_path"] # switch to reports_label_path
    )

    logger.info(f"Validation BLeU scores (BLEU-1, BLEU-2, BLEU-4): {cpbleus_valid}")
    logger.info(f"Validation METEOR score: {avg_valid_meteor}")
    log_chexbert_f1_summary(chexbert_res_valid, logger)


    # ---------------------------- Training and Evaluating Complete ------------------------------------------ #

    # Saving to result.txt
    save_training_results(
        # General Stuff
        save_path = save_path,
        conf = conf,
        model = model,
        best_epoch = ckpt["epoch"],

        # Evaluation Metric Valid
        best_valid_loss=best_valid_loss,
        valid_corpus_bleu = cpbleus_valid, 
        valid_meteor_score=avg_valid_meteor,
        valid_chexpert_f1s=chexbert_res_valid,

        # Evaluation Metric test
        test_corpus_bleu = None, 
        test_meteor_score=None,
        test_chexpert_f1s=None,

        train_losses=tl_list,
        valid_losses=vl_list,
    )

    tokenizer_file = os.path.join(save_path, "bpe_tokenizer.json")
    tokenizer.save(tokenizer_file)

    logger.info(f"BPE Tokenizer saved to: {tokenizer_file}")

    ## LAST STEP ## 
    # Flush handlers and save log file
    for handler in logger.handlers:
        handler.flush()
    
    temp_log = "/home/grad/masters/2025/mkamal/mkamal/cxr_report_gen/logs/exp3_ablation_heads.log"
    final_log = os.path.join(save_path, "train.log")
    if os.path.exists(temp_log):
        shutil.copy(temp_log, final_log)

if __name__ == "__main__":
   main()