"""Shared multi-task (masked-BCE) training loop.

Used by the ChemBERTa models on the multi-task classification datasets
(HIV / Tox21). A model plugs in by providing:

    model.module                      the nn.Module (for params / train-eval mode)
    model.build_loader(smiles, labels, batch_size, shuffle) -> iterable of batches
    model.forward_batch(batch, device) -> (logits[B,T], y[B,T], mask[B,T])  on device

where ``y`` has NaN replaced by 0 and ``mask`` is 1 where the label is present.
The monitored metric is mean ROC-AUC over tasks; loss is mean masked BCE.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from .metrics import multitask_classification_metrics
from .optim import build_optimizer
from .utils import get_logger

logger = get_logger(__name__)

MONITOR = ("roc_auc", True)  # (name, higher_is_better)


@torch.no_grad()
def _eval_split(model, smiles, labels, batch_size, device, bce) -> dict:
    model.module.eval()
    scores, tot_loss, n = [], 0.0, 0.0
    for batch in model.build_loader(smiles, labels, batch_size, shuffle=False):
        logits, y, mask = model.forward_batch(batch, device)
        tot_loss += (bce(logits, y) * mask).sum().item()
        n += mask.sum().item()
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
    scores = np.concatenate(scores, axis=0)
    out = multitask_classification_metrics(labels, scores)
    out = {"roc_auc": out["roc_auc"], "prc_auc": out["prc_auc"],
           "loss": tot_loss / max(1.0, n)}
    return out


def train_multitask(model, splits, *, cfg, device, study=False):
    """Train ``model`` on ``splits`` (dict with 'train','valid', optional 'test').

    Each split is ``(smiles, labels[N,T])``. ``cfg`` is a TrainConfig. When
    ``study`` is True, per-epoch train+test metrics are also recorded and early
    stopping is disabled (for full curves).
    """
    module = model.module.to(device)
    bs = cfg.batch_size
    tr_s, tr_y = splits["train"]
    va_s, va_y = splits["valid"]
    test_split = splits.get("test")

    optimizer = build_optimizer(module.parameters(), name=cfg.optimizer, lr=cfg.lr,
                                weight_decay=cfg.weight_decay,
                                bgf_lambda=cfg.bgf_lambda, bgf_alpha=cfg.bgf_alpha)
    if cfg.optimizer.lower() in ("bgf", "bgf_adamw"):
        logger.info("  optimizer=BGF+AdamW (lambda=%g, alpha=%g)", cfg.bgf_lambda, cfg.bgf_alpha)
    scheduler = None
    steps_per_epoch = max(1, math.ceil(len(tr_s) / bs))
    if cfg.scheduler == "linear":
        from transformers import get_linear_schedule_with_warmup
        total = steps_per_epoch * cfg.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(cfg.warmup_ratio * total), total)

    bce = nn.BCEWithLogitsLoss(reduction="none")
    name, higher = MONITOR
    best_metric = -np.inf
    best_state = copy.deepcopy(module.state_dict())
    best_epoch, patience_left = -1, cfg.patience
    history = []

    for epoch in range(cfg.epochs):
        module.train()
        running, seen = 0.0, 0
        for batch in model.build_loader(tr_s, tr_y, bs, shuffle=True):
            logits, y, mask = model.forward_batch(batch, device)
            loss = (bce(logits, y) * mask).sum() / mask.sum().clamp(min=1.0)
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running += loss.item() * logits.size(0)
            seen += logits.size(0)
        train_loss = running / max(1, seen)

        val = _eval_split(model, va_s, va_y, bs, device, bce)
        metric = val[name]
        improved = metric > best_metric if higher else metric < best_metric
        if improved:
            best_metric, best_epoch = metric, epoch
            best_state = copy.deepcopy(module.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1

        row = {"epoch": epoch, "train_loss": train_loss,
               "val_roc_auc": val["roc_auc"], "val_prc_auc": val["prc_auc"],
               "val_loss": val["loss"]}
        if study:
            cap = cfg.study_train_cap  # subsample train-metric eval on big datasets
            extra = [("train", (tr_s[:cap], tr_y[:cap]))]
            if test_split is not None:
                extra.append(("test", test_split))
            for nm, (s, y_) in extra:
                e = _eval_split(model, s, y_, bs, device, bce)
                row[f"{nm}_roc_auc"] = e["roc_auc"]
                row[f"{nm}_prc_auc"] = e["prc_auc"]
                row[f"{nm}_loss"] = e["loss"]
        history.append(row)
        logger.info("  epoch %2d/%d  loss=%.4f  val_roc_auc=%.4f%s",
                    epoch + 1, cfg.epochs, train_loss, metric, "  *" if improved else "")
        if not study and patience_left <= 0:
            logger.info("  early stop @ epoch %d (best val_roc_auc=%.4f @ %d)",
                        epoch + 1, best_metric, best_epoch + 1)
            break

    module.load_state_dict(best_state)
    return {"history": history, "best_metric": best_metric, "best_epoch": best_epoch + 1}


@torch.no_grad()
def predict_multitask(model, smiles, batch_size, device) -> np.ndarray:
    """Return per-task probabilities ``[N, T]``."""
    model.module.eval()
    dummy = np.full((len(smiles), model.n_tasks), 0.0)
    scores = []
    for batch in model.build_loader(smiles, dummy, batch_size, shuffle=False):
        logits, _, _ = model.forward_batch(batch, device)
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(scores, axis=0)
