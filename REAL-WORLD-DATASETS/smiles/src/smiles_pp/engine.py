"""Seed loop, CSV reporting and the train/eval/predict entry points."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import load_splits
from .metrics import aggregate, compute_metrics, format_summary
from .models import build_model, load_model
from .preprocess import canonical_smiles
from .utils import ensure_dir, get_logger, save_json, set_seed

logger = get_logger(__name__)


def _ckpt_path(config: Config, seed: int) -> Path:
    return config.ckpt_dir / f"seed_{seed}"


def train(config: Config, seeds=None, device: str | None = None) -> dict:
    """Train one model across seeds, evaluate on test, and write CSV results."""
    bundle = load_splits(config.dataset, config.data.get("cache_dir", "data"))
    seeds = list(seeds) if seeds else list(config.seeds)
    logger.info(
        "=== %s | dataset=%s model=%s | train/valid/test = %d/%d/%d | seeds=%s ===",
        config.name, config.dataset, config.model_type,
        len(bundle.train), len(bundle.valid), len(bundle.test), seeds,
    )

    rows: list[dict] = []
    for seed in seeds:
        set_seed(seed)
        if device:
            config.train["device"] = device
        model = build_model(config, seed)
        model.fit(
            bundle.train.smiles, bundle.train.labels,
            bundle.valid.smiles, bundle.valid.labels,
        )
        ckpt = _ckpt_path(config, seed)
        model.save(ckpt)
        preds = model.predict(bundle.test.smiles)
        metrics = compute_metrics(config.task_type, bundle.test.labels, preds)
        row = {
            "name": config.name,
            "dataset": config.dataset,
            "model": config.model_type,
            "task_type": config.task_type,
            "seed": seed,
            **metrics,
        }
        rows.append(row)
        logger.info(
            "[%s seed=%d] TEST  %s  (ckpt=%s)",
            config.name, seed,
            "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()), ckpt,
        )

    summary = aggregate(rows)
    _write_results(config, rows, summary)
    logger.info("=== %s SUMMARY (n=%d): %s ===", config.name, len(seeds), format_summary(summary))
    return {"per_seed": rows, "summary": summary}


def evaluate(config: Config, seed: int = 0, device: str | None = None) -> dict:
    """Load a saved checkpoint and evaluate it on the test split."""
    if device:
        config.train["device"] = device
    bundle = load_splits(config.dataset, config.data.get("cache_dir", "data"))
    model = load_model(config, _ckpt_path(config, seed))
    preds = model.predict(bundle.test.smiles)
    metrics = compute_metrics(config.task_type, bundle.test.labels, preds)
    logger.info("[%s seed=%d] EVAL  %s", config.name, seed,
                "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def predict(config: Config, smiles: list[str], seed: int = 0,
            device: str | None = None) -> pd.DataFrame:
    """Predict on arbitrary SMILES with a saved checkpoint."""
    if device:
        config.train["device"] = device
    model = load_model(config, _ckpt_path(config, seed))

    canon, valid_mask = [], []
    for smi in smiles:
        c = canonical_smiles(smi)
        valid_mask.append(c is not None)
        canon.append(c if c is not None else smi)

    preds = np.full(len(smiles), np.nan)
    valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
    if valid_idx:
        valid_preds = model.predict([canon[i] for i in valid_idx])
        for i, p in zip(valid_idx, valid_preds):
            preds[i] = p

    col = "probability" if config.task_type == "classification" else "prediction"
    return pd.DataFrame(
        {"smiles": smiles, "canonical_smiles": canon, "valid": valid_mask, col: preds}
    )


# -- CSV reporting ------------------------------------------------------------
def _write_results(config: Config, rows: list[dict], summary: dict) -> None:
    results_dir = ensure_dir(config.results_dir)

    per_seed = pd.DataFrame(rows)
    per_seed_path = results_dir / f"{config.name}_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    flat = {
        "name": config.name,
        "dataset": config.dataset,
        "model": config.model_type,
        "task_type": config.task_type,
        "n_seeds": len(rows),
        "seeds": "|".join(str(r["seed"]) for r in rows),
    }
    for metric, stats in summary.items():
        flat[f"{metric}_mean"] = stats["mean"]
        flat[f"{metric}_std"] = stats["std"]
    summary_df = pd.DataFrame([flat])
    summary_df.to_csv(results_dir / f"{config.name}_summary.csv", index=False)

    # Append to / refresh the global leaderboard.
    all_path = results_dir / "all_results.csv"
    if all_path.exists():
        prev = pd.read_csv(all_path)
        prev = prev[prev["name"] != config.name]
        summary_df = pd.concat([prev, summary_df], ignore_index=True)
    summary_df.to_csv(all_path, index=False)

    save_json({"config": config.to_dict(), "per_seed": rows, "summary": summary},
              results_dir / f"{config.name}_run.json")
    logger.info("Wrote results: %s , %s , %s",
                per_seed_path.name, f"{config.name}_summary.csv", "all_results.csv")
