"""Evaluation metrics and multi-seed aggregation."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

# Metrics used to pick the best epoch during training, and whether larger is better.
MONITOR = {
    "classification": ("roc_auc", True),
    "regression": ("rmse", False),
}


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """ROC-AUC and PRC-AUC (average precision) for binary classification."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "prc_auc": float(average_precision_score(y_true, y_score)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def multitask_classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Mean ROC-AUC / PRC-AUC over tasks (MoleculeNet convention).

    ``y_true`` is ``[N, T]`` with NaN for missing labels; degenerate tasks
    (a single class present) are skipped, matching DeepChem/MoleculeNet scoring.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim == 1:
        y_true, y_score = y_true[:, None], y_score[:, None]
    rocs, prcs = [], []
    for t in range(y_true.shape[1]):
        yt, ys = y_true[:, t], y_score[:, t]
        m = ~np.isnan(yt)
        yt, ys = yt[m].astype(int), ys[m]
        if yt.size == 0 or len(np.unique(yt)) < 2:
            continue
        rocs.append(roc_auc_score(yt, ys))
        prcs.append(average_precision_score(yt, ys))
    return {
        "roc_auc": float(np.mean(rocs)) if rocs else float("nan"),
        "prc_auc": float(np.mean(prcs)) if prcs else float("nan"),
        "n_tasks_scored": len(rocs),
    }


def compute_metrics(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if task_type == "classification":
        if np.asarray(y_true).ndim == 2 and np.asarray(y_true).shape[1] > 1:
            m = multitask_classification_metrics(y_true, y_pred)
            return {"roc_auc": m["roc_auc"], "prc_auc": m["prc_auc"]}
        return classification_metrics(y_true, y_pred)
    return regression_metrics(y_true, y_pred)


_NON_METRIC_KEYS = {"seed", "n_tasks", "epoch"}


def aggregate(
    rows: Sequence[dict[str, float]], exclude: set[str] | None = None
) -> dict[str, dict[str, float]]:
    """Mean / std for every numeric metric key across per-seed metric dicts."""
    if not rows:
        return {}
    skip = _NON_METRIC_KEYS | (exclude or set())
    keys = [k for k in rows[0] if k not in skip and isinstance(rows[0][k], (int, float))]
    summary: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=float)
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return summary


def format_summary(summary: dict[str, dict[str, float]]) -> str:
    return "  ".join(
        f"{k}={v['mean']:.4f}±{v['std']:.4f}" for k, v in summary.items()
    )
