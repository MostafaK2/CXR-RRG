"""
Learning Rate Finder for ChestXrayMRG
======================================
Implements the LR Range Test (Smith, 2017) — runs a mini training loop
increasing LR exponentially from min_lr to max_lr, records loss at each step,
then plots the loss curve so you can pick the optimal LR.

The optimal LR is typically:
  - The LR just BEFORE the loss starts to diverge (steepest downward slope)
  - Usually 1-2 orders of magnitude below the minimum loss LR

Usage:
    from utils.lr_finder import LRFinder

    finder = LRFinder(
        model     = model,
        optimizer = optimizer,
        criterion = criterion,
        device    = device,
    )

    optimal_lr = finder.find(
        dataloader   = train_dl,
        min_lr       = 1e-7,
        max_lr       = 1e-1,
        num_steps    = 200,
        smooth_beta  = 0.98,
        save_path    = "results/lr_finder.png",
    )
"""

import os
import copy
import math
import logging

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class LRFinder:
    """
    Learning Rate Range Test.

    Parameters
    ----------
    model      : your ChestXrayMRG model
    optimizer  : AdamW optimizer (will be reset after finding)
    criterion  : CrossEntropyLoss
    device     : cuda / cpu
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device:    str = "cuda",
    ):
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device

        # Store original states so we can reset after the run
        self._model_state     = copy.deepcopy(model.state_dict())
        self._optimizer_state = copy.deepcopy(optimizer.state_dict())

        self.lrs:         list[float] = []
        self.losses:      list[float] = []
        self.smooth_loss: list[float] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Main find method
    # ─────────────────────────────────────────────────────────────────────────

    def find(
        self,
        dataloader,
        min_lr:      float = 1e-7,
        max_lr:      float = 1e-1,
        num_steps:   int   = 200,
        smooth_beta: float = 0.98,   # exponential smoothing for loss curve
        diverge_threshold: float = 4.0,  # stop if loss > diverge_threshold * best
        save_path:   str   = "lr_finder.png",
    ) -> float:
        """
        Run the LR range test.

        Parameters
        ----------
        dataloader         : your train_dl — will iterate cyclically
        min_lr             : starting LR (very small)
        max_lr             : ending LR (large, expect divergence here)
        num_steps          : number of LR steps to test
        smooth_beta        : exponential smoothing factor for loss curve
        diverge_threshold  : stop early if loss spikes above best * this factor
        save_path          : where to save the LR curve plot

        Returns
        -------
        float : suggested optimal LR
        """
        logger.info(f"LR Finder: scanning {min_lr:.2e} → {max_lr:.2e} over {num_steps} steps")

        # Multiplicative factor per step
        lr_mult = (max_lr / min_lr) ** (1.0 / (num_steps - 1))

        # Set starting LR
        self._set_lr(min_lr)

        self.lrs     = []
        self.losses  = []
        self.smooth_loss = []

        best_loss    = float("inf")
        avg_loss     = 0.0
        data_iter    = iter(dataloader)

        self.model.train()

        for step in range(num_steps):
            # ── Get next batch (cycle through dataloader) ─────────────────
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch     = next(data_iter)

            # ── Unpack — matches your collate_fn output ───────────────────
            # collate_fn returns: img, padded_src, padded_tgt, clinical_text, labels
            img, src, tgt, clinical_text, labels = batch
            img = img.to(self.device)
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            # clinical_text stays as list of strings — text encoder handles it
            # labels not used here (no cls loss in finder — keeps it simple)

            # ── Forward + loss ────────────────────────────────────────────
            self.optimizer.zero_grad()

            logits = self.model(img, clinical_text, src)   # (B, T, V)
            B, T, V = logits.shape
            loss = self.criterion(logits.view(B * T, V), tgt.view(B * T))

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"  Step {step}: NaN/Inf loss detected, stopping early")
                break

            loss_val = loss.item()

            # ── Exponential smoothing ──────────────────────────────────────
            avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * loss_val
            smoothed = avg_loss / (1 - smooth_beta ** (step + 1))   # bias correction

            # ── Track best and check divergence ───────────────────────────
            if smoothed < best_loss:
                best_loss = smoothed

            if step > 10 and smoothed > diverge_threshold * best_loss:
                logger.info(f"  Divergence detected at step {step}, LR={self._get_lr():.2e}")
                break

            # ── Record ────────────────────────────────────────────────────
            current_lr = self._get_lr()
            self.lrs.append(current_lr)
            self.losses.append(loss_val)
            self.smooth_loss.append(smoothed)

            if step % 20 == 0:
                logger.info(f"  Step {step:>3}/{num_steps} | LR={current_lr:.2e} | Loss={loss_val:.4f} | Smooth={smoothed:.4f}")

            # ── Backward ──────────────────────────────────────────────────
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # ── Increase LR for next step ─────────────────────────────────
            self._set_lr(current_lr * lr_mult)

        # ── Reset model and optimizer to original state ───────────────────
        self._reset()
        logger.info("  Model and optimizer reset to pre-finder state ✓")

        # ── Find optimal LR ───────────────────────────────────────────────
        optimal_lr = self._find_optimal_lr()

        # ── Plot ──────────────────────────────────────────────────────────
        self._plot(save_path, optimal_lr)

        logger.info(f"\n  Suggested LR : {optimal_lr:.2e}")
        logger.info(f"  Typical range: {optimal_lr/10:.2e}  to  {optimal_lr:.2e}")
        logger.info(f"  For warmup target LR, use: {optimal_lr:.2e}")

        return optimal_lr

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _find_optimal_lr(self) -> float:
        """
        Find LR at the steepest negative gradient of the smoothed loss curve.
        This is the point of fastest descent — ideal starting LR.
        """
        if len(self.smooth_loss) < 5:
            logger.warning("Too few steps recorded to suggest LR")
            return self.lrs[len(self.lrs) // 2] if self.lrs else 1e-4

        losses = np.array(self.smooth_loss)
        lrs    = np.array(self.lrs)

        # Gradient of smoothed loss w.r.t. log(LR)
        log_lrs   = np.log10(lrs)
        gradients = np.gradient(losses, log_lrs)

        # Steepest descent = most negative gradient
        # Skip first 5 and last 5 steps to avoid boundary artifacts
        trim       = 5
        grad_trim  = gradients[trim:-trim]
        lrs_trim   = lrs[trim:-trim]

        if len(grad_trim) == 0:
            return lrs[len(lrs) // 2]

        min_grad_idx = np.argmin(grad_trim)
        optimal_lr   = float(lrs_trim[min_grad_idx])

        return optimal_lr

    def _plot(self, save_path: str, optimal_lr: float):
        """Plot raw loss, smoothed loss, and mark the suggested LR."""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Learning Rate Finder", fontweight="bold", fontsize=14)

        # ── Left: loss vs LR (log scale) ──────────────────────────────────
        ax1.plot(self.lrs, self.losses,      alpha=0.3, color="steelblue", label="Raw loss")
        ax1.plot(self.lrs, self.smooth_loss, color="steelblue",            label="Smoothed loss", linewidth=2)
        ax1.axvline(optimal_lr, color="red", linestyle="--", linewidth=1.5,
                    label=f"Suggested LR: {optimal_lr:.2e}")
        ax1.set_xscale("log")
        ax1.set_xlabel("Learning Rate (log scale)")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss vs Learning Rate")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ── Right: loss gradient vs LR ────────────────────────────────────
        if len(self.smooth_loss) > 10:
            log_lrs   = np.log10(self.lrs)
            gradients = np.gradient(np.array(self.smooth_loss), log_lrs)
            ax2.plot(self.lrs, gradients, color="darkorange", linewidth=2)
            ax2.axvline(optimal_lr, color="red", linestyle="--", linewidth=1.5,
                        label=f"Suggested LR: {optimal_lr:.2e}")
            ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
            ax2.set_xscale("log")
            ax2.set_xlabel("Learning Rate (log scale)")
            ax2.set_ylabel("Loss Gradient")
            ax2.set_title("Loss Gradient vs Learning Rate\n(optimal = most negative)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  LR finder plot saved: {save_path}")

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _reset(self):
        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(self._optimizer_state)

    def summary(self):
        """Print a summary table of LR vs smoothed loss."""
        if not self.lrs:
            print("No results — run .find() first")
            return
        print(f"\n{'Step':>5}  {'LR':>12}  {'Smooth Loss':>12}")
        print("─" * 35)
        step_size = max(1, len(self.lrs) // 20)
        for i in range(0, len(self.lrs), step_size):
            print(f"{i:>5}  {self.lrs[i]:>12.2e}  {self.smooth_loss[i]:>12.4f}")