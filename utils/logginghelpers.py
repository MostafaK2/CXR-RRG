
from .metrics import CHEXBERT_LABELS
import logging
import os
import json

def log_chexbert_f1_summary(results: dict, logger:logging) -> None:
    logger.info("CheXbert F1 Summary (only positive (1) labels)")
    logger.info("%-32s  %9s  %9s  %6s", "Label", "Precision", "Recall", "F1")
    for label in CHEXBERT_LABELS:
        logger.info(
            "%-32s  %9.3f  %9.3f  %6.3f",
            label,
            results[f"{label}_precision"],
            results[f"{label}_recall"],
            results[f"{label}_f1"],
        )
    logger.info(
        "%-32s  %9.3f  %9.3f  %6.3f",
        "MACRO AVERAGE",
        results["macro_precision"],
        results["macro_recall"],
        results["macro_f1"],
    )


def save_training_results(
    save_path,
    conf,
    model,
    best_epoch,

    best_valid_loss,
    valid_corpus_bleu,
    valid_meteor_score,
    valid_chexpert_f1s,

    test_corpus_bleu,
    test_meteor_score,
    test_chexpert_f1s,

    train_losses,
    valid_losses,
):
    os.makedirs(save_path, exist_ok=True)

    results_path = os.path.join(save_path, "results.txt")

    with open(results_path, "w") as f:
        f.write("===== TRAINING RESULTS =====\n\n")

        # General Info
        f.write("---- GENERAL ----\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Validation Loss: {best_valid_loss:.4f}\n")

        if model is not None:
            f.write(f"Model Class: {model.__class__.__name__}\n")

        f.write("\n")

        # Config
        f.write("---- CONFIG ----\n")
        if isinstance(conf, dict):
            for key, value in conf.items():
                f.write(f"{key}: {value}\n")
        else:
            f.write(str(conf) + "\n")

        f.write("\n")

        # Validation Metrics
        f.write("---- VALIDATION METRICS ----\n")
        f.write(f"BLEU Scores: {valid_corpus_bleu}\n")
        f.write(f"METEOR Score: {valid_meteor_score}\n")

        if valid_chexpert_f1s is not None:
            f.write("CheXbert Metrics:\n")
            for key, value in valid_chexpert_f1s.items():
                f.write(f"  {key}: {value}\n")

        f.write("\n")

        # Test Metrics
        f.write("---- TEST METRICS ----\n")
        f.write(f"BLEU Scores: {test_corpus_bleu}\n")
        f.write(f"METEOR Score: {test_meteor_score}\n")

        if test_chexpert_f1s is not None:
            f.write("CheXbert Metrics:\n")
            for key, value in test_chexpert_f1s.items():
                f.write(f"  {key}: {value}\n")

        f.write("\n")

        # Loss Curves
        f.write("---- LOSSES ----\n")
        f.write(f"Train Losses: {train_losses}\n")
        f.write(f"Validation Losses: {valid_losses}\n")

    return f"Training results saved to: {results_path}"


