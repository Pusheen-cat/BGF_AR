"""Single-run study entry: one (task, model, optimizer, seed) with full
per-epoch train/val/test curves recorded (no early stopping).

Used by the AdamW-vs-BGF comparison study. Writes ``history.csv`` (per-epoch
loss + metrics for all three splits) and ``final.json`` (the test metric at the
best-validation epoch + the last epoch) into ``--outdir``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import Config
from .data import load_splits
from .metrics import MONITOR
from .models import build_model
from .utils import ensure_dir, get_logger, set_seed

logger = get_logger("smiles_pp.study")


def run(task, model, optimizer, seed, outdir, *, epochs=None, device="auto",
        bgf_lambda=0.01, bgf_alpha=0.9, config_dir="configs"):
    cfg = Config.from_yaml(Path(config_dir) / f"{task}_{model}.yaml")
    cfg.train["optimizer"] = optimizer
    cfg.train["bgf_lambda"] = bgf_lambda
    cfg.train["bgf_alpha"] = bgf_alpha
    cfg.train["device"] = device
    if epochs:
        cfg.train["epochs"] = epochs

    set_seed(seed)
    bundle = load_splits(cfg.dataset, cfg.data.get("cache_dir", "data"))
    m = build_model(cfg, seed)
    m.fit(
        bundle.train.smiles, bundle.train.labels,
        bundle.valid.smiles, bundle.valid.labels,
        study_eval={
            "train": (bundle.train.smiles, bundle.train.labels),
            "test": (bundle.test.smiles, bundle.test.labels),
        },
    )

    hist = pd.DataFrame(m.history_)
    outdir = ensure_dir(outdir)
    hist.to_csv(outdir / "history.csv", index=False)

    monitor, higher = MONITOR[cfg.task_type]
    valcol = f"val_{monitor}"
    best_idx = hist[valcol].idxmax() if higher else hist[valcol].idxmin()
    best = hist.loc[best_idx].to_dict()
    splits = ("train_", "val_", "test_")
    final = {
        "pipeline": "SMILES", "dataset": task, "task_type": cfg.task_type,
        "model": model, "optimizer": optimizer, "seed": seed,
        "monitor": monitor, "epochs_run": int(len(hist)),
        "best_epoch": int(best["epoch"]) + 1,
        "best": {k: float(v) for k, v in best.items()
                 if any(k.startswith(p) for p in splits)},
        "last": {k: float(hist.iloc[-1][k]) for k in hist.columns if k != "epoch"},
    }
    (outdir / "final.json").write_text(json.dumps(final, indent=2))
    logger.info("[SMILES %s/%s/%s seed=%d] best_epoch=%d  test_%s@best=%.4f",
                task, model, optimizer, seed, final["best_epoch"],
                monitor, final["best"][f"test_{monitor}"])
    return final


def main(argv=None):
    p = argparse.ArgumentParser(prog="smiles_pp.study")
    p.add_argument("--task", required=True,
                   choices=["bbbp", "esol", "hiv", "tox21"])
    p.add_argument("--model", required=True,
                   choices=["chemberta", "chemberta2", "chemberta3"])
    p.add_argument("--optimizer", required=True, choices=["adamw", "bgf"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--bgf-lambda", type=float, default=0.01)
    p.add_argument("--bgf-alpha", type=float, default=0.9)
    a = p.parse_args(argv)
    run(a.task, a.model, a.optimizer, a.seed, a.outdir, epochs=a.epochs,
        device=a.device, bgf_lambda=a.bgf_lambda, bgf_alpha=a.bgf_alpha)


if __name__ == "__main__":
    main()
