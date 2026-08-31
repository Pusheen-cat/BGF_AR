"""Multi-label AUROC / AUPRC metrics with safe handling of degenerate labels.

A label column that contains only one class (all-0 or all-1 in the evaluated
split) has an undefined AUROC/AUPRC and is *skipped* (recorded as NaN and counted
in ``n_skipped``).  Aggregate scores use ``nanmean`` / ``nanmedian``.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute per-label and aggregate metrics.

    Parameters
    ----------
    y_true : ``[N, L]`` binary ground truth.
    y_score : ``[N, L]`` predicted probabilities (post-sigmoid) or scores.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    n, L = y_true.shape

    per_auroc = np.full(L, np.nan, dtype=np.float64)
    per_auprc = np.full(L, np.nan, dtype=np.float64)
    prevalence = y_true.mean(axis=0)

    n_skipped = 0
    for i in range(L):
        yt = y_true[:, i]
        pos = yt.sum()
        if pos == 0 or pos == n:  # only one class present -> undefined
            n_skipped += 1
            continue
        ys = y_score[:, i]
        try:
            per_auroc[i] = roc_auc_score(yt, ys)
            per_auprc[i] = average_precision_score(yt, ys)
        except ValueError:
            n_skipped += 1

    return {
        "mean_auroc": float(np.nanmean(per_auroc)) if L - n_skipped else float("nan"),
        "mean_auprc": float(np.nanmean(per_auprc)) if L - n_skipped else float("nan"),
        "median_auroc": float(np.nanmedian(per_auroc)) if L - n_skipped else float("nan"),
        "median_auprc": float(np.nanmedian(per_auprc)) if L - n_skipped else float("nan"),
        "per_label_auroc": per_auroc,
        "per_label_auprc": per_auprc,
        "prevalence": prevalence.astype(np.float64),
        "n_labels": int(L),
        "n_skipped": int(n_skipped),
        "n_evaluated": int(L - n_skipped),
        "n_samples": int(n),
    }


def summarize_metrics(metrics: dict) -> str:
    """One-line human summary of the aggregate scores."""
    return (
        f"AUROC mean={metrics['mean_auroc']:.4f} median={metrics['median_auroc']:.4f} | "
        f"AUPRC mean={metrics['mean_auprc']:.4f} median={metrics['median_auprc']:.4f} | "
        f"labels={metrics['n_evaluated']}/{metrics['n_labels']} "
        f"(skipped {metrics['n_skipped']})"
    )
