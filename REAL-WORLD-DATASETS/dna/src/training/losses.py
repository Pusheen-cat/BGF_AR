"""Multi-label loss and optional positive-class weighting.

Training uses raw logits with ``BCEWithLogitsLoss`` (numerically stable; sigmoid
is applied by the loss internally, never in the model's forward pass).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_pos_weight(pos_counts, n_total, clip: float = 100.0) -> np.ndarray:
    """``pos_weight = num_negative / num_positive`` per label (clipped).

    Labels with zero positives get weight ``clip`` (they will be skipped in
    metrics anyway).  Returned as float32 ``[L]``.
    """
    pos = np.asarray(pos_counts, dtype=np.float64)
    neg = float(n_total) - pos
    pos_safe = np.clip(pos, 1.0, None)
    w = neg / pos_safe
    return np.clip(w, 0.0, clip).astype(np.float32)


def build_loss(
    use_pos_weight: bool = False,
    pos_weight: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Return a configured ``BCEWithLogitsLoss``."""
    if use_pos_weight and pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    return nn.BCEWithLogitsLoss()
