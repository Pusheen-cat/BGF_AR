#!/usr/bin/env python
"""Train one DeepSEA sequence model (RNN / LSTM / Stack-RNN / Tape-RNN).

    python scripts/train.py --config configs/rnn_bgf.yaml \
        --data data/processed/deepsea.h5 --outdir outputs/rnn_bgf/seed0 --seed 0

Optimizer: the config's ``training.optimizer`` is ``adamw`` or ``bgf``. For BGF
this release fixes ``bgf_alpha = 0.95`` (used without hyperparameter search); the
``--bgf_alpha`` / ``--bgf_lambda`` flags below let a caller set them explicitly
(the reproduction and lambda-sweep scripts use these).

Writes:
    outputs/checkpoints/<name>.pt         best-by-val-mean-AUPRC checkpoint
    outputs/metrics/<name>_history.json   per-epoch history + best summary
    outputs/logs/<name>.log               training log
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import make_dataloaders  # noqa: E402
from src.data.deepsea_dataset import DeepSEADataset  # noqa: E402
from src.models import build_model  # noqa: E402
from src.training import Trainer  # noqa: E402
from src.training.losses import compute_pos_weight  # noqa: E402
from src.training.utils import (  # noqa: E402
    get_device,
    load_config,
    save_json,
    set_seed,
    setup_logger,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="processed .h5 file")
    ap.add_argument("--name", default=None,
                    help="run name (default: config filename stem)")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    ap.add_argument("--epochs", type=int, default=None, help="override config")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    # BGF hyperparameters. Left as None -> use the value in the config. This
    # release fixes bgf_alpha = 0.95 (no hyperparameter search); bgf_lambda is the
    # EMA smoothing factor swept only by the RNN/LSTM lambda-comparison script.
    ap.add_argument("--bgf_alpha", type=float, default=None,
                    help="override training.bgf_alpha (release default: 0.95)")
    ap.add_argument("--bgf_lambda", type=float, default=None,
                    help="override training.bgf_lambda (default: 0.01)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.bgf_alpha is not None:
        cfg["training"]["bgf_alpha"] = args.bgf_alpha
    if args.bgf_lambda is not None:
        cfg["training"]["bgf_lambda"] = args.bgf_lambda

    name = args.name or os.path.splitext(os.path.basename(args.config))[0]
    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    metrics_dir = os.path.join(args.outdir, "metrics")
    log_dir = os.path.join(args.outdir, "logs")
    for d in (ckpt_dir, metrics_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    logger = setup_logger("deepsea", os.path.join(log_dir, f"{name}.log"))
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(args.device)

    logger.info(f"run='{name}'  config={args.config}  data={args.data}")
    logger.info(f"training config: {cfg['training']}")

    tcfg = cfg["training"]
    loaders = make_dataloaders(
        args.data,
        batch_size=int(tcfg.get("batch_size", 128)),
        num_workers=int(tcfg.get("num_workers", 4)),
        reverse_complement_augmentation=bool(
            tcfg.get("reverse_complement_augmentation", True)
        ),
        splits=("train", "valid"),
    )
    if "train" not in loaders:
        logger.error("no train split found in the HDF5 file")
        return 1
    logger.info(f"train N={len(loaders['train'].dataset)} "
                f"valid N={len(loaders['valid'].dataset) if 'valid' in loaders else 0}")

    # optional positive-class weighting from training labels
    pos_weight = None
    if bool(tcfg.get("use_pos_weight", False)):
        logger.info("computing pos_weight from training labels ...")
        train_ds = DeepSEADataset(args.data, "train")
        pos_weight = compute_pos_weight(
            train_ds.positive_counts(), len(train_ds)
        )
        logger.info(f"pos_weight: min={pos_weight.min():.2f} "
                    f"max={pos_weight.max():.2f} mean={pos_weight.mean():.2f}")

    model = build_model(cfg)
    trainer = Trainer(model, cfg, device, logger=logger, pos_weight=pos_weight)

    ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
    summary = trainer.fit(
        loaders["train"], loaders.get("valid"), checkpoint_path=ckpt_path
    )

    history_path = os.path.join(metrics_dir, f"{name}_history.json")
    save_json(history_path, {
        "name": name,
        "config": cfg,
        "data": os.path.abspath(args.data),
        "model_type": cfg["model"].get("type"),
        **summary,
    })
    logger.info(f"best epoch={summary['best_epoch']} "
                f"val_mean_auprc={summary['best_val_mean_auprc']}")
    logger.info(f"saved checkpoint -> {ckpt_path}")
    logger.info(f"saved history    -> {history_path}")
    logger.info("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
