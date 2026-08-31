"""Training / evaluation loop for the DeepSEA sequence models.

Features
--------
* ``BCEWithLogitsLoss`` on raw logits (sigmoid only at evaluation time).
* ``AdamW`` + gradient clipping.
* Optional mixed precision (``torch.amp``) — auto-disabled on CPU.
* Best checkpoint selected by **validation mean AUPRC**.
* Per-epoch history + best metrics returned/saved.
"""
from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import build_loss
from .metrics import compute_metrics, summarize_metrics
from .optim import build_optimizer
from .utils import count_parameters, save_checkpoint


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        device: torch.device,
        logger=None,
        pos_weight: np.ndarray | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger
        tcfg = config.get("training", {})

        self.epochs = int(tcfg.get("epochs", 10))
        self.grad_clip = float(tcfg.get("gradient_clip", 1.0))
        self.use_amp = bool(tcfg.get("use_amp", True)) and device.type == "cuda"

        self.criterion = build_loss(
            use_pos_weight=bool(tcfg.get("use_pos_weight", False)),
            pos_weight=pos_weight,
            device=device,
        )
        # Base optimizer is AdamW; ``optimizer: bgf`` wraps BGF on top of AdamW
        # (grad pre-conditioning + AdamW step). All other hyperparameters (lr,
        # weight_decay, default betas/eps) are identical across the two.
        self.optimizer_name = str(tcfg.get("optimizer", "adamw")).lower()
        self.optimizer = build_optimizer(
            self.model.parameters(),
            name=self.optimizer_name,
            lr=float(tcfg.get("lr", 1e-3)),
            weight_decay=float(tcfg.get("weight_decay", 1e-5)),
            bgf_lambda=float(tcfg.get("bgf_lambda", 0.01)),
            bgf_alpha=float(tcfg.get("bgf_alpha", 0.9)),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.history: list[dict[str, Any]] = []
        self.best_metric = -np.inf          # best val mean AUPRC
        self.best_epoch = -1
        self.best_state: dict | None = None

        self._log(
            f"model params: {count_parameters(self.model):,} | "
            f"device={device} | amp={self.use_amp} | epochs={self.epochs} | "
            f"optimizer={self.optimizer_name}"
        )

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg, flush=True)

    # ------------------------------------------------------------------ #
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total, n_batches = 0.0, 0
        t0 = time.time()
        for bidx, (x, y) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type,
                                dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            self.scaler.scale(loss).backward()
            if self.grad_clip and self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total += float(loss.detach())
            n_batches += 1
            if bidx == 0 and epoch == 1:
                self._log(f"  [shape check] x={tuple(x.shape)} "
                          f"logits={tuple(logits.shape)} y={tuple(y.shape)}")
        avg = total / max(n_batches, 1)
        # SafeAdamW/BGF skip a step on any non-finite gradient (instead of
        # corrupting params / poisoning state); surface how often per epoch.
        skip_note = ""
        skipped = getattr(self.optimizer, "num_skipped", None)
        if skipped is not None:
            delta = skipped - getattr(self, "_skipped_prev", 0)
            self._skipped_prev = skipped
            if delta:
                skip_note = f", skipped_steps={delta}"
        self._log(f"epoch {epoch}: train_loss={avg:.4f} "
                  f"({time.time()-t0:.1f}s, {n_batches} batches{skip_note})")
        return avg

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, compute_full_metrics: bool = True,
                 max_batches: int | None = None) -> dict:
        """Run the model over ``loader`` and return loss + metrics.

        Applies sigmoid to logits before scoring (never during training).
        """
        self.model.eval()
        losses, ys, ps = [], [], []
        for bidx, (x, y) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            yb = y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type,
                                dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, yb)
            losses.append(float(loss))
            ps.append(torch.sigmoid(logits.float()).cpu().numpy())
            ys.append(y.numpy())
            if max_batches and bidx + 1 >= max_batches:
                break
        y_true = np.concatenate(ys, axis=0)
        y_score = np.concatenate(ps, axis=0)
        out = {"loss": float(np.mean(losses))}
        if compute_full_metrics:
            out.update(compute_metrics(y_true, y_score))
        return out

    # ------------------------------------------------------------------ #
    def fit(self, train_loader: DataLoader, valid_loader: DataLoader | None,
            checkpoint_path: str | None = None) -> dict:
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            record = {"epoch": epoch, "train_loss": train_loss}

            if valid_loader is not None:
                val = self.evaluate(valid_loader, compute_full_metrics=True)
                record.update({
                    "val_loss": val["loss"],
                    "val_mean_auroc": val["mean_auroc"],
                    "val_mean_auprc": val["mean_auprc"],
                    "val_median_auroc": val["median_auroc"],
                    "val_median_auprc": val["median_auprc"],
                    "val_n_skipped": val["n_skipped"],
                })
                self._log(f"epoch {epoch}: val_loss={val['loss']:.4f} | "
                          + summarize_metrics(val))

                metric = val["mean_auprc"]
                if np.isfinite(metric) and metric > self.best_metric:
                    self.best_metric = metric
                    self.best_epoch = epoch
                    self.best_state = copy.deepcopy(self.model.state_dict())
                    if checkpoint_path:
                        save_checkpoint(
                            checkpoint_path, self.model, self.config,
                            extra={"epoch": epoch,
                                   "val_mean_auprc": metric,
                                   "val_metrics": _json_safe(val)},
                        )
                    self._log(f"  * new best val mean AUPRC={metric:.4f} "
                              f"(epoch {epoch}) -> saved {checkpoint_path}")
            else:
                # no validation: keep the last epoch as "best"
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, self.model, self.config,
                                    extra={"epoch": epoch})

            self.history.append(record)

        # Safeguard: if the validation metric was never finite (e.g. a diverged
        # run), no checkpoint was saved above. Persist the last-epoch weights so
        # downstream evaluation never fails on a missing file.
        if self.best_state is None:
            self._log("  ! val metric never finite -> saving last-epoch "
                      "fallback checkpoint")
            self.best_state = copy.deepcopy(self.model.state_dict())
            self.best_epoch = self.epochs
            if checkpoint_path:
                save_checkpoint(checkpoint_path, self.model, self.config,
                                extra={"epoch": self.epochs,
                                       "note": "fallback: no finite val metric"})

        # restore best weights in-memory for any post-fit evaluation
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_mean_auprc": (None if not np.isfinite(self.best_metric)
                                    else self.best_metric),
        }


def _json_safe(metrics: dict) -> dict:
    """Drop bulky per-label arrays for compact in-checkpoint storage."""
    keep = {"loss", "mean_auroc", "mean_auprc", "median_auroc", "median_auprc",
            "n_labels", "n_skipped", "n_evaluated", "n_samples"}
    return {k: v for k, v in metrics.items() if k in keep}
