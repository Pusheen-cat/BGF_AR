#!/usr/bin/env python
"""Evaluate a trained checkpoint on a split and dump metrics.

    python scripts/evaluate.py --checkpoint outputs/checkpoints/rnn.pt \
        --data data/processed/deepsea.h5 --split test

Writes:
    outputs/metrics/<name>_<split>_metrics.json   aggregate scores
    outputs/metrics/<name>_<split>_per_label.csv  per-label AUROC/AUPRC/prevalence
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import make_dataloaders  # noqa: E402
from src.models import build_model  # noqa: E402
from src.training.metrics import summarize_metrics  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.training.utils import (  # noqa: E402
    get_device,
    load_checkpoint,
    save_json,
    setup_logger,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    metrics_dir = os.path.join(args.outdir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    logger = setup_logger("deepsea-eval")
    device = get_device(args.device)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    name = args.name or os.path.splitext(os.path.basename(args.checkpoint))[0]
    logger.info(f"checkpoint={args.checkpoint} model={cfg['model'].get('type')} "
                f"(trained epoch={ckpt.get('epoch')})")

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])

    loaders = make_dataloaders(
        args.data, batch_size=args.batch_size, num_workers=args.num_workers,
        reverse_complement_augmentation=False, splits=(args.split,),
    )
    if args.split not in loaders:
        logger.error(f"split '{args.split}' not present in {args.data}")
        return 1
    logger.info(f"evaluating on {args.split} N={len(loaders[args.split].dataset)}")

    trainer = Trainer(model, cfg, device, logger=logger)
    metrics = trainer.evaluate(loaders[args.split], compute_full_metrics=True)
    logger.info(summarize_metrics(metrics))

    # aggregate json (per-label arrays stripped out into the CSV below)
    agg = {k: v for k, v in metrics.items()
           if k not in ("per_label_auroc", "per_label_auprc", "prevalence")}
    agg.update({"checkpoint": os.path.abspath(args.checkpoint),
                "data": os.path.abspath(args.data),
                "split": args.split,
                "model_type": cfg["model"].get("type"),
                "name": name})
    json_path = os.path.join(metrics_dir, f"{name}_{args.split}_metrics.json")
    save_json(json_path, agg)

    per_label = pd.DataFrame({
        "label_index": np.arange(metrics["n_labels"]),
        "auroc": metrics["per_label_auroc"],
        "auprc": metrics["per_label_auprc"],
        "prevalence": metrics["prevalence"],
    })
    csv_path = os.path.join(metrics_dir, f"{name}_{args.split}_per_label.csv")
    per_label.to_csv(csv_path, index=False)

    logger.info(f"saved aggregate -> {json_path}")
    logger.info(f"saved per-label -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
