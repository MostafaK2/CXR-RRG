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
from torch.utils.data import DataLoader


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
MODEL_CLASS_NAMES = ["ChestXrayReportGenerator", "ChestXrayMRG", "Multimodal_Memory", "Multimodal_Memory_Real"]

REPORT_SECTION_CHOSEN = "report_gen"


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


## ---------------------------------------------- METRICS CALCULATORS -------------------------------------- ##

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
    
# ------- ChestXbert F1 Scoring -----------------------
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

# --------------------------------------------------- GENERATE REPORTS ------------
# Generate reports
@torch.no_grad()
def generate_report(model, sample, tokenizer, word2idx, config, device="cuda"):
    """Generates a caption for a single image using greedy search."""
    model.eval()
     
    ## \START MODULAR ACROSS DIFFERENT CLASSES CODE (DATA)
    modelclassname = model.__class__.__name__
    if MODEL_CLASS_NAMES[0] == modelclassname: image, _, _ = sample; max_len = config["model"]["max_len"]
    elif MODEL_CLASS_NAMES[1] == modelclassname: image, _, _, clinical_hist, _ = sample; max_len = config["model"]["decoder_max_len"]
    elif MODEL_CLASS_NAMES[2] == modelclassname: image, _, _, clinical_hist, _ = sample; max_len = config["model"]["decoder_max_len"]
    elif MODEL_CLASS_NAMES[3] == modelclassname: image, _, _, clinical_hist, _ = sample; max_len = config["model"]["decoder_max_len"] 

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
        elif MODEL_CLASS_NAMES[2] == modelclassname: logits = model(image, clinical_hist, src_tensor)
        elif MODEL_CLASS_NAMES[3] == modelclassname: logits, labels = model(image, clinical_hist, src_tensor)
        ## \END MODULAR ACROSS DIFFERENT CLASSES CODE
        
        # Get next token (last position in sequence)
        next_token = logits[0, -1, :].argmax().item()
        generated_ids.append(next_token)
        
        if next_token == eos_id:
            break

    # Decode IDs to string, skipping special tokens
    report = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return report

@torch.no_grad()
def generate_report_batched(model, samples, tokenizer, word2idx, config, device="cuda"):
    """Generates a caption for a single image using greedy search."""
    model.eval()
     
    ## \START MODULAR ACROSS DIFFERENT CLASSES CODE (DATA)
    modelclassname = model.__class__.__name__
    if MODEL_CLASS_NAMES[0] == modelclassname: images, _, _ = samples; max_len = config["model"]["decoder_max_len"]
    elif MODEL_CLASS_NAMES[1] == modelclassname: images, clinical_hist, _ = samples; max_len = config["model"]["decoder_max_len"]
    elif MODEL_CLASS_NAMES[2] == modelclassname: images, clinical_hist, _ = samples; max_len = config["model"]["decoder_max_len"]
    elif MODEL_CLASS_NAMES[3] == modelclassname: images, clinical_hist, _ = samples; max_len = config["model"]["decoder_max_len"] 
    ## \END MODULAR ACROSS DIFFERENT CLASSES CODE END

    B = images.size(0)
    images = images.to(device) # (1, 3, 224, 224)
    
    # Start with <BOS>
    bos_id = word2idx[config['special_tokens']['bos']]
    eos_id = word2idx[config['special_tokens']['eos']]
    pad_id = eos_id # for finished sequence

    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished  = torch.zeros(B, dtype=torch.bool, device=device)  # which rows are done


    for _ in range(max_len):
        ## \START MODULAR ACROSS DIFFERENT CLASSES CODE (LOGITS)
        if MODEL_CLASS_NAMES[0] == modelclassname: logits = model(images, generated)
        elif MODEL_CLASS_NAMES[1] == modelclassname: logits = model(images, clinical_hist, generated)
        elif MODEL_CLASS_NAMES[2] == modelclassname: logits = model(images, clinical_hist, generated)
        elif MODEL_CLASS_NAMES[3] == modelclassname: logits, _ = model(images, clinical_hist, generated)
        ## \END MODULAR ACROSS DIFFERENT CLASSES CODE
        
        # Greedy: pick highest-logit token at the last position
        next_tokens = logits[:, -1, :].argmax(dim=-1)  # (B,)

        # Finished sequences emit PAD instead of a real token
        next_tokens[finished] = pad_id

        # Append to running sequence
        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)  # (B, T+1)

        # Mark newly finished sequences
        finished |= (next_tokens == eos_id)

        if finished.all():
            break
        
    # ── 4. Decode ───────────────────────────────────────────────────────────
    reports = []
    for i in range(B):
        ids = generated[i].tolist()
        reports.append(tokenizer.decode(ids, skip_special_tokens=True))

    return reports



# Evaluate Metric with batching
def evaluate_metric_batched(model, 
                    train_df, 
                    dataset, 
                    tokenizer, 
                    word2idx, 
                    config, 
                    device, 
                    num_samples=None, 
                    labels_path = None,
                    batch_size = 64):
    
    """Iterates through dataset and calculates average BLEU score."""
    
    model.eval()
    label_df = reorder_labels_df(labels_path)

    if num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(num_samples, len(dataset))))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    reference_list, generated_list, gt_labels = [], [], []
    meteor_scores = []
    global_i = 0

    for batch in tqdm.tqdm(loader, desc="BLEU & METEOR Scoring"):
            batch_generated = generate_report_batched(
                model, batch, tokenizer, word2idx, config, device=device
            )

            batch_df  = train_df.iloc[global_i : global_i + len(batch_generated)]
            refs      = batch_df[REPORT_SECTION_CHOSEN].tolist()
            img_paths = batch_df["path_no_ext"].tolist()

            reference_list.extend(refs)
            generated_list.extend(batch_generated)
            meteor_scores.extend(calculate_meteor_score(r, g) for r, g in zip(refs, batch_generated))
            gt_labels.extend(_extract_ground_truth_labels(label_df, p) for p in img_paths)

            global_i += len(batch_generated)

    chexbert_res = evaluate_with_chesxbert(
        generated_list, gt_labels,
        config["eval"]["chestXbertModelWeights"],
        batch_sz=64, device=device
    )
    cpbleus    = calculate_corpus_bleu(reference_list, generated_list)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    return cpbleus, avg_meteor, chexbert_res




# Evaluate Metric
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
        reference_text = train_df.iloc[i][REPORT_SECTION_CHOSEN]
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


## ------------------------------------- COALLATE FN FOR BATCHED 





# --------------------------------------- Dataset Helper Function (EDIT) --------------------------------


def collate_fn(batch):  
    img_tens, _, _, clinical_text, labels = zip(*batch)
    img_tens = torch.stack(img_tens) # stack the images (B, 3, 224, 224)
    clinical_text = list(clinical_text)
    labels = torch.stack(labels)
    return img_tens, clinical_text, labels


@torch.no_grad()
def evaluate_metric_llm(model,
                         df,
                         dataset,
                         config,
                         device,
                         labels_path=None,
                         num_samples=None,
                         batch_size=16):
    model.eval()
    # ----llm ----

    def collate_fn(batch):  
        img_tens, report_ids, report_mask, clinical_history, labels = zip(*batch)
        img_tens = torch.stack(img_tens) # stack the images (B, 3, 224, 224)
        report_ids = torch.stack(report_ids)
        report_mask = torch.stack(report_mask)
        # new stuff
        clinical_history = list(clinical_history)
        labels = torch.stack(labels)
        return img_tens, report_ids, report_mask, clinical_history, labels
    label_df = reorder_labels_df(labels_path)

    if num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(num_samples, len(dataset))))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    reference_list, generated_list = [], []
    meteor_scores = []
    gt_labels     = []

    for batch_idx, batch in enumerate(tqdm.tqdm(loader, desc="Evaluating")):
        
        images, report_ids, _, texts, _ = batch
        images     = images.to(device)
        report_ids = report_ids.to(device)  
        generated = model.generate(images, texts)  # List[str] of length B
        generated_list.extend(generated)

        references = model.tokenizer.batch_decode(report_ids, skip_special_tokens=True)
        reference_list.extend(references)

        for ref, gen in zip(references, generated):
            meteor_scores.append(calculate_meteor_score(ref, gen))

        global_start = batch_idx * batch_size
        for j in range(len(images)):
            global_i = global_start + j
            img_path = df.iloc[global_i]["path_no_ext"]
            gt_labels.append(_extract_ground_truth_labels(label_df, img_path))

    chexbert_res = evaluate_with_chesxbert(
        generated_list, gt_labels,
        config["eval"]["chestXbertModelWeights"],
        batch_sz=64, device=device
    )
    cpbleus    = calculate_corpus_bleu(reference_list, generated_list)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    return cpbleus, avg_meteor, chexbert_res


# ------------------------------------------------------ERROR ANALYSIS HELPERS. -------------------

from collections import Counter
import json
import os
import numpy as np

def generation_diversity(generated_list, save_dir=None):
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Generation Diversity")
    print("=" * 60)
 
    total      = len(generated_list)
    unique     = len(set(generated_list))
    uniqueness = 100 * unique / total
 
    all_words    = " ".join(generated_list).split()
    unique_words = len(set(all_words))
    total_words  = len(all_words)
    ttr          = unique_words / total_words if total_words > 0 else 0.0
 
    lengths = [len(r.split()) for r in generated_list]
    counter = Counter(generated_list)
    top5    = counter.most_common(5)
 
    print(f"\nTotal reports:      {total}")
    print(f"Unique reports:     {unique} ({uniqueness:.1f}%)")
    print(f"Type-Token Ratio:   {ttr:.4f}")
    print(f"Avg length:         {np.mean(lengths):.1f} words (std={np.std(lengths):.1f})")
 
    if uniqueness < 20:
        print("❌ Severe mode collapse")
    elif uniqueness < 50:
        print("⚠️  Moderate collapse")
    else:
        print("✅ Reasonable diversity")
 
    print(f"\nTop 5 repeated:")
    for report, count in top5:
        print(f"  [{count:4d}x | {100*count/total:4.1f}%] {report[:120]}")
 
    results = {
        "total":             total,
        "unique":            unique,
        "uniqueness_pct":    uniqueness,
        "type_token_ratio":  ttr,
        "avg_length":        float(np.mean(lengths)),
        "std_length":        float(np.std(lengths)),
        "top5_repeated":     [(r[:120], c) for r, c in top5],
    }
 
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "generation_diversity.json"), "w") as f:
            json.dump(results, f, indent=2)
 
    return results
 

 # Evaluate Metric with batching
def evaluate_metric_batched_for_error_analysis(model, 
                    train_df, 
                    dataset, 
                    tokenizer, 
                    word2idx, 
                    config, 
                    device, 
                    num_samples=None, 
                    labels_path = None,
                    batch_size = 64):
    
    """Iterates through dataset and calculates average BLEU score."""
    
    model.eval()
    label_df = reorder_labels_df(labels_path)

    if num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(num_samples, len(dataset))))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    reference_list, generated_list, gt_labels = [], [], []
    meteor_scores = []
    global_i = 0

    for batch in tqdm.tqdm(loader, desc="BLEU & METEOR Scoring"):
            batch_generated = generate_report_batched(
                model, batch, tokenizer, word2idx, config, device=device
            )

            batch_df  = train_df.iloc[global_i : global_i + len(batch_generated)]
            refs      = batch_df[REPORT_SECTION_CHOSEN].tolist()
            img_paths = batch_df["path_no_ext"].tolist()

            reference_list.extend(refs)
            generated_list.extend(batch_generated)
            meteor_scores.extend(calculate_meteor_score(r, g) for r, g in zip(refs, batch_generated))
            gt_labels.extend(_extract_ground_truth_labels(label_df, p) for p in img_paths)

            global_i += len(batch_generated)

    chexbert_res = evaluate_with_chesxbert(
        generated_list, gt_labels,
        config["eval"]["chestXbertModelWeights"],
        batch_sz=64, device=device
    )
    cpbleus    = calculate_corpus_bleu(reference_list, generated_list)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    return cpbleus, avg_meteor, chexbert_res, generated_list, reference_list, gt_labels




# HALUCINATE RATES
LABEL_KEYWORDS = {
    "Enlarged Cardiomediastinum": ["enlarged cardiomediastinum", "widened mediastinum"],
    "Cardiomegaly":               ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
    "Lung Opacity":               ["opacity", "opacification", "haziness"],
    "Lung Lesion":                ["lesion", "nodule", "mass"],
    "Edema":                      ["edema", "oedema", "vascular congestion"],
    "Consolidation":              ["consolidation"],
    "Pneumonia":                  ["pneumonia", "pneumonic"],
    "Atelectasis":                ["atelectasis", "atelectatic"],
    "Pneumothorax":               ["pneumothorax"],
    "Pleural Effusion":           ["effusion", "pleural fluid"],
    "Pleural Other":              ["pleural thickening", "pleural disease"],
    "Fracture":                   ["fracture", "rib fracture"],
    "Support Devices":            ["support device", "pacemaker", "tube", "line", "catheter"],
    "No Finding":                 ["no acute", "no finding", "unremarkable", "normal"],
}
 
 
def _extract_mentions(report_text):
    report_lower = report_text.lower()
    mentioned = set()
    for label, keywords in LABEL_KEYWORDS.items():
        for kw in keywords:
            if kw in report_lower:
                mentioned.add(label)
                break
    return mentioned
 
 
def _extract_positive_gt_labels(gt_label_dict):
    if gt_label_dict is None:
        return set()
    return {label for label, val in gt_label_dict.items() if val == 1}
 
 
def hallucination_rate(generated_list, gt_labels, save_dir=None):
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Hallucination Rate")
    print("=" * 60)
 
    per_label_hallucinations = {label: 0 for label in CHEXBERT_LABELS}
    per_label_generated      = {label: 0 for label in CHEXBERT_LABELS}
    hallucination_rates      = []
    any_hallucination        = []
    examples                 = []
 
    for gen, gt in zip(generated_list, gt_labels):
        if gt is None:
            continue
 
        gen_mentions = _extract_mentions(gen)
        gt_positives = _extract_positive_gt_labels(gt)
        hallucinated = gen_mentions - gt_positives
 
        for label in gen_mentions:
            per_label_generated[label] += 1
        for label in hallucinated:
            per_label_hallucinations[label] += 1
 
        rate = len(hallucinated) / len(gen_mentions) if len(gen_mentions) > 0 else 0.0
        hallucination_rates.append(rate)
        any_hallucination.append(1 if len(hallucinated) > 0 else 0)
 
        if len(hallucinated) > 0 and len(examples) < 10:
            examples.append({
                "generated":    gen[:200],
                "gt_positives": list(gt_positives),
                "hallucinated": list(hallucinated),
            })
 
    mean_rate      = float(np.mean(hallucination_rates))
    pct_any_halluc = float(np.mean(any_hallucination)) * 100
 
    print(f"\nSamples evaluated:                {len(hallucination_rates)}")
    print(f"Mean hallucination rate:          {mean_rate*100:.1f}%")
    print(f"Reports with any hallucination:   {pct_any_halluc:.1f}%")
 
    print(f"\n{'Label':<30} {'Generated':>10} {'Hallucinated':>13} {'Rate':>8}")
    print("-" * 65)
 
    per_label_results = {}
    for label in CHEXBERT_LABELS:
        g = per_label_generated[label]
        h = per_label_hallucinations[label]
        r = h / g if g > 0 else 0.0
        per_label_results[label] = {"generated": g, "hallucinated": h, "rate": round(r, 4)}
        print(f"{label:<30} {g:>10} {h:>13} {r:>8.3f}")
 
    print(f"\nWorst examples:")
    for i, ex in enumerate(examples):
        print(f"\n  [{i+1}] Hallucinated: {ex['hallucinated']}")
        print(f"       GT positives: {ex['gt_positives']}")
        print(f"       Generated:    {ex['generated'][:150]}")
 
    results = {
        "mean_hallucination_rate":          mean_rate,
        "pct_reports_with_hallucination":   pct_any_halluc,
        "per_label":                        per_label_results,
        "examples":                         examples,
    }
 
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "hallucination_rate.json"), "w") as f:
            json.dump(results, f, indent=2)
 
    return results