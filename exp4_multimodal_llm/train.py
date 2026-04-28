import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import math
import shutil

from utils.config import load_config, _find_root
from utils.metrics import evaluate_metric, evaluate_metric_llm

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

from exp2_multimodal.dataset import load_and_split
from dataset import Dataset_for_llm_model

from model import Radiology_llm
import yaml

from transformers import AutoTokenizer

from utils.logginghelpers import log_chexbert_f1_summary, save_training_results

import matplotlib.pyplot as plt


# ----------------------- LOGGING ---------------------

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    handlers=[
        logging.StreamHandler(),                 # print to console
        logging.FileHandler("logs/multimodal_llm_train.log", mode='w')    # save to file
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

DEFAULT = os.path.join(_find_root(), 'configs', 'multimodal_llm', 'main.yml')
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

# ---------------- LABEL DATA ---------------- COPY THIS
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


def collate_fn(batch):  
    img_tens, report_ids, report_mask, clinical_history, labels = zip(*batch)
    img_tens = torch.stack(img_tens) # stack the images (B, 3, 224, 224)
    report_ids = torch.stack(report_ids)
    report_mask = torch.stack(report_mask)
    # new stuff
    clinical_history = list(clinical_history)
    labels = torch.stack(labels)

    return img_tens, report_ids, report_mask, clinical_history, labels
# Training Helpers
# ----------------- Training Helper functions ------------------- 
def train_epoch(model, dataloader, optimizer, criterion, device,  clip_grad=1.0, warmup_scheduler=None):
    model.train()
    total_loss = 0.0
    for img, report_ids, report_mask, clincal_text, labels in tqdm.tqdm(dataloader,"train"):
        images      = img.to(device)
        report_ids  = report_ids.to(device)
        report_mask = report_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        out = model(images, clincal_text, report_ids, report_mask)  # (B, T, V)
        loss = out.loss
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        total_loss += loss.item()
        
             

    nll = total_loss / len(dataloader)
    ppl = math.exp(nll)
    return nll, ppl

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
         for img, report_ids, report_mask, clincal_text, labels in tqdm.tqdm(dataloader, "Evaluating"):
            images      = img.to(device)
            
            report_ids  = report_ids.to(device)
            report_mask = report_mask.to(device)
            labels = labels.to(device)

            out = model(images, clincal_text, report_ids, report_mask)  # (B, T, V)
            loss = out.loss
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
    MAX_LEN = config['data']['max_len']

    # ------ Model Parameters -------------------------------------------
    IMG_ENC_DIM = int(config['model']['img_enc_dim'])

       # LLM DECODER
    LLM_MODEL_NAME = str(config['model']['model_name'])
    MAX_NEW_TOKEN = int(config['model']['max_new_tokens'])
    
        # IMG ENCODER
    IMG_ENC_BACKBONE = str(config['model']['img_enc_backbone'])
    IMG_ENC_FREEZE_LAYER = int(config['model']['img_enc_freeze_layer'])


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
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_path = conf.SAVE_DIR
    if not os.path.isabs(save_path):
        save_path = os.path.join(BASE_DIR, save_path)

    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, "best_config.yml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saving all results, model configurations, results in {save_path}")

    # - - - - - -  - - - - - - - - - MODEL DEFINITION - - - -- -- - - - -- - - -- - --

    model = Radiology_llm(
        img_enc_backbone = conf.IMG_ENC_BACKBONE,
        img_enc_dim = conf.IMG_ENC_DIM,
        img_enc_freeze_layer = conf.IMG_ENC_FREEZE_LAYER,
        dropout = conf.DROPOUT,
        max_new_tokens = conf.MAX_NEW_TOKEN,
        model_name =  conf.LLM_MODEL_NAME
    ).to(conf.DEVICE)

    optimizer = AdamW(model.parameters(), lr=conf.LR, weight_decay=conf.WEIGHT_DECAY)
    # CE Loss (prev)
    criterion = nn.CrossEntropyLoss(label_smoothing=conf.LABEL_SMOOTHING)

    # Parameter Calculation
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Model initialized sucessfully")
    logger.info(f"  - Total params:     {total_params:,}")
    logger.info(f"  - Trainable params: {trainable_params:,}")
    logger.info(f"  - Non-trainable:    {total_params - trainable_params:,}")

    # ── Resume from checkpoint ──────────────────────────────────────────────────
    start_epoch = 0
    if config['checkpoint'].get('resume'):
        resume_path = config['checkpoint']['resume']
        if os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location=conf.DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_valid_loss = ckpt['valid_loss']
            logger.info(f"Resumed from {resume_path} (epoch {ckpt['epoch']+1}, valid loss {ckpt['valid_loss']:.4f})")
        else:
            logger.warning(f"Resume path not found: {resume_path}")

    # ---------------- Splitting the data ------------------------------------
    train_df, valid_df, test_df = load_and_split(config["data"]["csv_file"], val_size=config["data"]["valid_sz"], test_size=config["data"]["test_sz"], seed=SEED, logger=logger)
    
    tokenizer = AutoTokenizer.from_pretrained(conf.LLM_MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # grayscale -> 3 channels
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    ## CHANGE IF NEEDED CHANGE IF NEEDED CHANGE IF NEEDED CHANGE IF NEEDED CHANGE IF NEEDED CHANGE IF NEEDED
    label_df = reorder_labels_df(config['eval']['reports_label_path'])

    train_dataset = Dataset_for_llm_model(
        df_reports = train_df,
        df_labels  = label_df,
        h5_path    = conf.H5_PATH,
        tokenizer  = tokenizer,
        max_len    = conf.MAX_LEN,
        transform  = transform,
    )

    valid_dataset = Dataset_for_llm_model(
        df_reports = valid_df,
        df_labels  = label_df,
        h5_path    = conf.H5_PATH,
        tokenizer  = tokenizer,
        max_len    = conf.MAX_LEN,
        transform  = transform,
    )

    test_dataset = Dataset_for_llm_model(
        df_reports = test_df,
        df_labels  = label_df,
        h5_path    = conf.H5_PATH,
        tokenizer  = tokenizer,
        max_len    = conf.MAX_LEN,
        transform  = transform,
    )

    train_dl = DataLoader(train_dataset, batch_size=conf.BATCH_SIZE,shuffle=True,  collate_fn=collate_fn, num_workers=4)
    valid_dl = DataLoader(valid_dataset, batch_size=conf.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_dl  = DataLoader(test_dataset,  batch_size=conf.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

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


    # # ── LR Finder — run once, then comment out ───────────────────────────────
    # from utils.lr_finder import LRFinder
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
                    'img_enc_backbone'      : conf.IMG_ENC_BACKBONE,
                    'img_enc_dim'           : conf.IMG_ENC_DIM,
                    'img_enc_freeze_layer'  : conf.IMG_ENC_FREEZE_LAYER,
                    'dropout'               : conf.DROPOUT,
                    'max_new_tokens'        : conf.MAX_NEW_TOKEN,
                    'model_name'            : conf.LLM_MODEL_NAME

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
        


    # # ── Plots ─────────────────────────────────────────────────────────────────────
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    # ax1.plot(tl_list, label='Train Loss', marker='o')
    # ax1.plot(vl_list, label='Valid Loss', marker='s')
    # ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    # ax1.set_title('Training and Validation Loss', fontweight='bold')
    # ax1.legend(); ax1.grid(True, alpha=0.3)

    # ax2.plot(tp_list, label='Train PPL', marker='o')
    # ax2.plot(vp_list, label='Valid PPL', marker='s')
    # ax2.set_xlabel('Epoch'); ax2.set_ylabel('Perplexity')
    # ax2.set_title('Training and Validation Perplexity', fontweight='bold')
    # ax2.legend(); ax2.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.savefig(save_path + '/training_curves.png', dpi=150, bbox_inches='tight')
    # plt.show()
    # logger.info("Plots saved in /results/training_curves.png")

    # logger.info("Training & Evaluating completed!")

    # # ── Evaluation Test & Valid Dataset ─────────────────────────────────────────────────────────────────────
    logger.info("======= " + "Starting Evaluating " + ("=" * 60))
    
    ckpt = torch.load(best_model_save_path, weights_only=False)
    hp   = ckpt['hyperparams']
    model = Radiology_llm(**hp).to(conf.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']+1} with valid loss {ckpt['valid_loss']:.4f}")


    logger.info("======= Calculating Evaluation BLEU Scores =======")
    cpbleus_valid, avg_valid_meteor, chexbert_res_valid = evaluate_metric_llm(
        model = model,
        df    = valid_df, 
        dataset = valid_dataset, 
        config = config, 
        device = conf.DEVICE,
        num_samples=None,
        batch_size = conf.BATCH_SIZE,
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
    # LAST STEP ## 
    # Flush handlers and save log file
    for handler in logger.handlers:
        handler.flush()
    
    temp_log = "/home/grad/masters/2025/mkamal/mkamal/cxr_report_gen/logs/multimodal_llm_train.log"
    final_log = os.path.join(save_path, "train.log")
    if os.path.exists(temp_log):
        shutil.copy(temp_log, final_log)

if __name__ == "__main__":
   main()