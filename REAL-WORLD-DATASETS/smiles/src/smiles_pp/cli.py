"""Command-line interface: ``prepare`` / ``train`` / ``eval`` / ``predict``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .data import build_cache
from .utils import get_logger

logger = get_logger("smiles_pp.cli")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", required=True, help="Path to a YAML config file")
    p.add_argument("--device", default=None, help="cuda | cuda:0 | cpu | auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smiles-pp",
        description="Reproducible SMILES molecular-property prediction pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Download + scaffold-split + cache a dataset")
    p_prep.add_argument("--dataset", "-d", required=True,
                        choices=["bbbp", "esol", "hiv", "tox21"])
    p_prep.add_argument("--cache-dir", default="data")
    p_prep.add_argument("--force", action="store_true", help="Rebuild even if cached")

    p_train = sub.add_parser("train", help="Train across seeds and write CSV results")
    _add_common(p_train)
    p_train.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Override the seeds listed in the config")
    p_train.add_argument("--optimizer", choices=["adamw", "bgf"], default=None,
                         help="Override train.optimizer (bgf = BGF grad filter + AdamW)")
    p_train.add_argument("--bgf-lambda", type=float, default=None,
                         help="BGF EMA weight lambda (default 0.01)")
    p_train.add_argument("--bgf-alpha", type=float, default=None,
                         help="BGF low-freq grad weight alpha (default 0.9)")

    p_eval = sub.add_parser("eval", help="Evaluate a saved checkpoint on the test split")
    _add_common(p_eval)
    p_eval.add_argument("--seed", type=int, default=0)

    p_pred = sub.add_parser("predict", help="Predict on SMILES from a file")
    _add_common(p_pred)
    p_pred.add_argument("--seed", type=int, default=0)
    p_pred.add_argument("--input", "-i", required=True,
                        help="Text file with one SMILES per line")
    p_pred.add_argument("--output", "-o", default=None, help="Output CSV path")
    return parser


def _apply_optimizer_overrides(config, args) -> None:
    """Let --optimizer / --bgf-* override the config's train block."""
    if getattr(args, "optimizer", None) is not None:
        config.train["optimizer"] = args.optimizer
    if getattr(args, "bgf_lambda", None) is not None:
        config.train["bgf_lambda"] = args.bgf_lambda
    if getattr(args, "bgf_alpha", None) is not None:
        config.train["bgf_alpha"] = args.bgf_alpha


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "prepare":
        build_cache(args.dataset, args.cache_dir, force=args.force)
        return 0

    # The remaining commands are import-heavy (torch); import lazily.
    from . import engine

    config = Config.from_yaml(args.config)

    if args.command == "train":
        _apply_optimizer_overrides(config, args)
        engine.train(config, seeds=args.seeds, device=args.device)
    elif args.command == "eval":
        engine.evaluate(config, seed=args.seed, device=args.device)
    elif args.command == "predict":
        smiles = [ln.strip() for ln in Path(args.input).read_text().splitlines() if ln.strip()]
        df = engine.predict(config, smiles, seed=args.seed, device=args.device)
        out = args.output or f"{config.name}_predictions.csv"
        df.to_csv(out, index=False)
        logger.info("Wrote %d predictions to %s", len(df), out)
        with __import__("pandas").option_context("display.max_rows", 20):
            print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
