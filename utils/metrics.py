import re
import warnings
from dataclasses import dataclass, field
from typing import Optional
 
import pandas as pd
import numpy as np
from pandas import DataFrame

import tqdm
import torch

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

import torch
from transformers import BertTokenizer
from .bert_labeler import bert_labeler

import warnings
warnings.filterwarnings("ignore")

## ------------------------------- CONSTANTS -------------------------
CHEXBERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding",
]

CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}
                    #        Baseline                Main
MODEL_CLASS_NAMES = ["ChestXrayReportGenerator", "ChestXrayMRG"]


# -------------------------------- Evaluation BLEU & METEOR helpers -------------------------------------
def setup_nltk_resources():
    resources = [
        ('tokenizers/punkt_tab', 'punkt_tab'), 
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource: {package_name}")
            nltk.download(package_name)


## ---------------------------------------------- BLEU AND METEOR SCORING -------------------------------------- ##

def calculate_sentence_bleu_score(reference, hypothesis):
    """
    Calculates BLEU-4 score for a single sentence pair.
    Uses SmoothingFunction to handle short sentences/low overlap.
    """
    # Tokenize strings into lists of words
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    
    # Use method1 smoothing to avoid 0.0 scores for very short sentences
    smoothie = SmoothingFunction().method1
    
    score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
    return score

def calculate_meteor_score(reference, hypothesis):
    """
    Calculates METEOR for a single reference.
    """
    # Tokenize
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    
    score = meteor_score([ref_tokens], hyp_tokens)
    return score

def calculate_corpus_bleu(references, hypotheses):
    refs_tokenized = [[nltk.word_tokenize(ref.lower())] for ref in references]
    hyps_tokenized = [nltk.word_tokenize(hyp.lower()) for hyp in hypotheses]

    smoothie = SmoothingFunction().method1

    bleu1 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    return [bleu1, bleu2, bleu3, bleu4]
    
# -------------------------------------------- ChestXbert F1 Scoring -----------------------------------------
def _extract_ground_truth_labels(label_df, img_path):
    match = label_df[label_df["path_no_ext"] == img_path]
    if match.empty:
        return None
    # If only one row should exist
    row = match.iloc[0]
    gt_labels = row.drop(labels=["path_no_ext", "path_to_image"]).to_dict()
    return gt_labels


def reorder_labels_df(path: str) -> DataFrame:
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
        0.0: 2,      # 0.0 -> 2
        -1.0: 3      # -1.0 -> 3

    }
    for col in CHEXBERT_LABELS:
        labels_df[col] = labels_df[col].map(mapping)
    # Since map does not directly handle NaN keys well, fix NaNs separately
    for col in CHEXBERT_LABELS:
        labels_df[col] = labels_df[col].fillna(0).astype(int)

    return labels_df

@torch.no_grad()
def evaluate_with_chesxbert(generated_texts, ground_truth_labels, bert_checkpoint_path, batch_sz = 64, device="cuda"):
    # Load
    tokenizer  = BertTokenizer.from_pretrained("bert-base-uncased")
    model      = bert_labeler()

    # Replace with bert_checkpoint_path
    checkpoint = torch.load(bert_checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(generated_texts), batch_sz), desc="CheXbert Evaluation"):
            batch_texts = generated_texts[i:i+batch_sz]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(
                source_padded=input_ids,
                attention_mask=attention_mask,
            )
            preds = torch.stack([o.argmax(dim=-1) for o in outputs], dim=1).cpu().numpy()
            all_preds.append(preds)
    

    y_pred = np.vstack(all_preds)

    y_true = np.array([
        [d[label] for label in CHEXBERT_LABELS]
        for d in ground_truth_labels
    ])

    # ── Binarise — Convention 1: positive class only (Smit et al. 2020) ──────
    # class 1 = positive mention; everything else (blank/neg/uncertain) = 0
    y_true_bin = (y_true == 1).astype(int)   # (N, 14)
    y_pred_bin = (y_pred == 1).astype(int)   # (N, 14)

    # ── Per-label F1 then macro average ──────────────────────────────────────
    results = {}
    precisions, recalls, f1s = [], [], []

    for i, label in enumerate(CHEXBERT_LABELS):
        tp = int(((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 1)).sum())
        fp = int(((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 1)).sum())
        fn = int(((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 0)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        results[f"{label}_precision"] = round(prec, 4)
        results[f"{label}_recall"]    = round(rec,  4)
        results[f"{label}_f1"]        = round(f1,   4)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    results["macro_precision"] = round(sum(precisions) / len(precisions), 4)
    results["macro_recall"]    = round(sum(recalls)    / len(recalls),    4)
    results["macro_f1"]        = round(sum(f1s)        / len(f1s),        4)
    return results

# ---------------------------------------------------------------
@torch.no_grad()
def generate_report(model, sample, tokenizer, word2idx, config, device="cuda"):
    """Generates a caption for a single image using greedy search."""
    model.eval()
     
    ## \START MODULAR ACROSS DIFFERENT CLASSES CODE (DATA)
    modelclassname = model.__class__.__name__
    if MODEL_CLASS_NAMES[0] == modelclassname: image, _, _ = sample; max_len = config["model"]["max_len"]
    elif MODEL_CLASS_NAMES[1] == modelclassname: image, _, _, clinical_hist, _ = sample; max_len = config["model"]["decoder_max_len"]
    ## \END MODULAR ACROSS DIFFERENT CLASSES CODE END

    image = image.to(device).unsqueeze(0) # (1, 3, 224, 224)
    
    # Start with <BOS>
    bos_id = word2idx[config['special_tokens']['bos']]
    eos_id = word2idx[config['special_tokens']['eos']]
    generated_ids = [bos_id]

    for _ in range(max_len):
        src_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)

        ## \START MODULAR ACROSS DIFFERENT CLASSES CODE (LOGITS)
        if MODEL_CLASS_NAMES[0] == modelclassname: logits = model(image, src_tensor)
        elif MODEL_CLASS_NAMES[1] == modelclassname: logits = model(image, clinical_hist, src_tensor)
        ## \END MODULAR ACROSS DIFFERENT CLASSES CODE
        
        # Get next token (last position in sequence)
        next_token = logits[0, -1, :].argmax().item()
        generated_ids.append(next_token)
        
        if next_token == eos_id:
            break

    # Decode IDs to string, skipping special tokens
    report = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return report

def evaluate_metric(model, 
                    train_df, 
                    dataset, 
                    tokenizer, 
                    word2idx, 
                    config, 
                    device, 
                    num_samples=None, 
                    labels_path = None):
    
    """Iterates through dataset and calculates average BLEU score."""
    model.eval()
    
    label_df = reorder_labels_df(labels_path)
    meteor_scores = []
    
    # If num_samples is None, evaluate the whole dataset
    indices = range(len(dataset)) if num_samples is None else range(min(num_samples, len(dataset)))

    reference_list, generated_list = [], []
    gt_labels = [] # N, 14 labels GT

    for i in tqdm.tqdm(indices, desc="BLEU & METEOR Scoring"):
        
        sample = dataset[i]

        # Refrence and Generated Texts (BLEU)
        reference_text = train_df.iloc[i]["section_impression_gen"]
        generated_text = generate_report(model, sample, tokenizer, word2idx, config, device=device)
        reference_list.append(reference_text)
        generated_list.append(generated_text)
        # print(f"{generated_text}\n {CHEXBERT_LABELS} \n{sample[4]}\n\n")
        # METEOR
        meteor_score = calculate_meteor_score(reference_text, generated_text)
        meteor_scores.append(meteor_score)

        # Labels
        img_path = train_df.iloc[i]["path_no_ext"]
        gt_label = _extract_ground_truth_labels(label_df ,img_path)
        gt_labels.append(gt_label)


    ## EVALUATION OF CHESTXBERT
    chexbert_res = evaluate_with_chesxbert(generated_list, gt_labels, config["eval"]["chestXbertModelWeights"], batch_sz = 64, device=device)

    cpbleus = calculate_corpus_bleu(reference_list, generated_list)
    avg_meteor  = sum(meteor_scores)/len(meteor_scores)

    return cpbleus, avg_meteor, chexbert_res



