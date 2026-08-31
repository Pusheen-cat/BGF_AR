"""Shared training / early-stopping loop for the two neural models.

Both the BiLSTM and the ChemBERTa model expose an ``nn.Module`` whose
``forward(**inputs)`` returns a single logit per molecule, plus a torch
``Dataset`` carrying a ``.labels`` array and a ``collate`` function that yields
``(inputs_dict, labels_tensor)``. Everything else (optimizer, scheduler, loss,
validation-based model selection) lives here so the two models stay tiny and
behave identically.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..metrics import MONITOR, compute_metrics
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class TargetScaler:
    """Standardize regression targets; identity for classification."""

    mean: float = 0.0
    std: float = 1.0

    @classmethod
    def fit(cls, y) -> "TargetScaler":
        y = np.asarray(y, dtype=float)
        std = float(y.std())
        return cls(mean=float(y.mean()), std=std if std > 1e-8 else 1.0)

    def transform(self, y):
        return (np.asarray(y, dtype=float) - self.mean) / self.std

    def inverse(self, y):
        return np.asarray(y, dtype=float) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    scheduler: str = "none"  # 'none' | 'linear'
    grad_clip: float | None = None
    patience: int = 10
    num_workers: int = 0
    optimizer: str = "adamw"  # 'adamw' | 'bgf'
    bgf_lambda: float = 0.01
    bgf_alpha: float = 0.9
    study_train_cap: int = 5000  # cap train-metric eval rows in study curves

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in fields})


def _make_loader(dataset, collate, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        drop_last=False,
    )


def _to_device(inputs: dict, device) -> dict:
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def predict_scores(module, loader, task_type, device, scaler: TargetScaler | None = None):
    module.eval()
    chunks = []
    for inputs, _ in loader:
        logits = module(**_to_device(inputs, device)).squeeze(-1)
        if task_type == "classification":
            scores = torch.sigmoid(logits)
        else:
            scores = logits
        chunks.append(scores.detach().float().cpu().numpy())
    scores = np.concatenate(chunks) if chunks else np.array([], dtype=float)
    if task_type == "regression" and scaler is not None:
        scores = scaler.inverse(scores)
    return scores


@torch.no_grad()
def eval_with_loss(module, loader, labels, task_type, device, scaler, loss_fn):
    """Metrics + mean loss for one split (loss in training space; scaled for reg)."""
    module.eval()
    chunks, tot_loss, n = [], 0.0, 0
    for inputs, lab in loader:
        logits = module(**_to_device(inputs, device)).squeeze(-1)
        y = lab.to(device).float()
        y_t = (y - scaler.mean) / scaler.std if (task_type == "regression" and scaler) else y
        tot_loss += loss_fn(logits, y_t).item() * y.size(0)
        n += y.size(0)
        scores = torch.sigmoid(logits) if task_type == "classification" else logits
        chunks.append(scores.detach().float().cpu().numpy())
    scores = np.concatenate(chunks) if chunks else np.array([], dtype=float)
    if task_type == "regression" and scaler is not None:
        scores = scaler.inverse(scores)
    out = compute_metrics(task_type, labels, scores)
    out["loss"] = tot_loss / max(1, n)
    return out


def build_monitor_sets(builder, study_eval):
    """For study mode, turn {name: (smiles, labels)} into {name: Dataset}.

    Returns ``(monitor_sets, early_stop)``: when ``study_eval`` is given we also
    disable early stopping so the full per-epoch curve is recorded.
    """
    if not study_eval:
        return None, True
    return {name: builder(s, l) for name, (s, l) in study_eval.items()}, False


def train_loop(
    module,
    train_set,
    valid_set,
    collate,
    *,
    task_type: str,
    cfg: TrainConfig,
    device,
    scaler: TargetScaler | None = None,
    monitor_sets: dict | None = None,   # study mode: extra splits to log each epoch
    early_stop: bool = True,
):
    module.to(device)
    train_loader = _make_loader(train_set, collate, cfg.batch_size, True, cfg.num_workers)
    train_eval_loader = _make_loader(train_set, collate, cfg.batch_size, False, cfg.num_workers)
    valid_loader = (
        _make_loader(valid_set, collate, cfg.batch_size, False, cfg.num_workers)
        if valid_set is not None
        else None
    )

    from ..optim import build_optimizer

    optimizer = build_optimizer(
        module.parameters(), name=cfg.optimizer, lr=cfg.lr,
        weight_decay=cfg.weight_decay, bgf_lambda=cfg.bgf_lambda, bgf_alpha=cfg.bgf_alpha,
    )
    if cfg.optimizer.lower() in ("bgf", "bgf_adamw"):
        logger.info("  optimizer=BGF+AdamW (lambda=%g, alpha=%g)", cfg.bgf_lambda, cfg.bgf_alpha)
    scheduler = None
    total_steps = max(1, len(train_loader) * cfg.epochs)
    if cfg.scheduler == "linear":
        from transformers import get_linear_schedule_with_warmup

        warmup = int(cfg.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    loss_fn = (
        torch.nn.BCEWithLogitsLoss()
        if task_type == "classification"
        else torch.nn.MSELoss()
    )

    monitor_name, higher_better = MONITOR[task_type]
    best_metric = -np.inf if higher_better else np.inf
    best_state = copy.deepcopy(module.state_dict())
    best_epoch = -1
    patience_left = cfg.patience
    history: list[dict] = []

    eval_loader = valid_loader if valid_loader is not None else train_eval_loader
    eval_labels = valid_set.labels if valid_set is not None else train_set.labels

    for epoch in range(cfg.epochs):
        module.train()
        running, seen = 0.0, 0
        for inputs, labels in train_loader:
            y = labels.to(device).float()
            if task_type == "regression" and scaler is not None:
                y = (y - scaler.mean) / scaler.std
            optimizer.zero_grad()
            logits = module(**_to_device(inputs, device)).squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        train_loss = running / max(1, seen)

        val = eval_with_loss(module, eval_loader, eval_labels, task_type, device, scaler, loss_fn)
        metric = val[monitor_name]
        improved = metric > best_metric if higher_better else metric < best_metric
        if improved:
            best_metric, best_epoch = metric, epoch
            best_state = copy.deepcopy(module.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1

        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"val_{k}": v for k, v in val.items()})
        if monitor_sets:  # study mode: also log train/test (etc.) splits per epoch
            for name, ds in monitor_sets.items():
                loader = _make_loader(ds, collate, cfg.batch_size, False, cfg.num_workers)
                mm = eval_with_loss(module, loader, ds.labels, task_type, device, scaler, loss_fn)
                row.update({f"{name}_{k}": v for k, v in mm.items()})
        history.append(row)
        logger.info(
            "  epoch %2d/%d  loss=%.4f  val_%s=%.4f%s",
            epoch + 1, cfg.epochs, train_loss, monitor_name, metric,
            "  *" if improved else "",
        )
        if early_stop and patience_left <= 0:
            logger.info(
                "  early stop @ epoch %d (best val_%s=%.4f @ epoch %d)",
                epoch + 1, monitor_name, best_metric, best_epoch + 1,
            )
            break

    module.load_state_dict(best_state)
    return {"best_metric": best_metric, "best_epoch": best_epoch + 1, "history": history}
