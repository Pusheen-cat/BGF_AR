#!/usr/bin/env python
"""Compare SMILES / ChemBERTa results: AdamW vs BGF (alpha=0.95), paired per seed.

For every (dataset, ChemBERTa variant, metric) it reports mean +/- std over seeds
(sample std, ddof=1), Delta = BGF - AdamW (raw signed), the direction-aware
"better" optimizer, BGF win-rate, and a paired t-test. The per-seed metric is the
test score at the best-validation epoch (final.json["best"]["test_<metric>"]),
matching how the study records results.

Reads outputs/<task>/<variant>/{adamw,bgf}/seed*/final.json (produced by
reproduce_smiles.py). Writes results_smiles_compare.{md,csv}.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from scipy import stats

VARIANTS = {"chemberta": "ChemBERTa-1 (77M-MTR)",
            "chemberta2": "ChemBERTa-2 (77M-MLM)",
            "chemberta3": "ChemBERTa-3 (10M-MTR)"}
# dataset -> (task type, metrics, higher_is_better)
TASKS = {"esol":  ("regression",     [("rmse", False), ("mae", False)]),
         "bbbp":  ("classification", [("roc_auc", True), ("prc_auc", True)]),
         "hiv":   ("classification", [("roc_auc", True), ("prc_auc", True)]),
         "tox21": ("classification", [("roc_auc", True), ("prc_auc", True)])}


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def load_arm(root: Path, metric: str):
    """seed -> best-epoch test metric for every final.json under <root>/seed*."""
    out = {}
    if not root.exists():
        return out
    for sd in sorted(root.glob("seed*")):
        fj = sd / "final.json"
        if fj.exists():
            v = json.loads(fj.read_text()).get("best", {}).get(f"test_{metric}")
            if finite(v):
                out[int(sd.name[4:])] = v
    return out


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--outroot", default=str(repo / "outputs"))
    ap.add_argument("--out", default=str(repo / "results_smiles_compare"))
    args = ap.parse_args()
    outroot = Path(args.outroot)

    md = ["# SMILES / ChemBERTa — AdamW vs. BGF (alpha=0.95)\n",
          "Fine-tuning three pretrained ChemBERTa encoders (settings 1/2/3) on four "
          "MoleculeNet datasets, scaffold split. Paired over seeds; Delta = BGF - AdamW "
          "(RMSE/MAE lower-is-better, ROC/PRC-AUC higher-is-better; the 'better' column "
          "resolves direction). p = paired t-test. * p<0.05, ** p<0.01, *** p<0.001.\n"]
    csv = ["dataset,variant,metric,n,adamw_mean,adamw_std,bgf_mean,bgf_std,delta,better,bgf_win_rate,p_value"]
    n_bgf = n_adamw = n_total = 0
    for task, (ttype, metrics) in TASKS.items():
        md += [f"## {task.upper()} ({ttype})\n",
               "| ChemBERTa | metric | n | AdamW | BGF | Delta | better | win | p |",
               "|---|---|---|---|---|---|---|---|---|"]
        for variant, vlabel in VARIANTS.items():
            for m, higher in metrics:
                a = load_arm(outroot / task / variant / "adamw", m)
                b = load_arm(outroot / task / variant / "bgf", m)
                seeds = sorted(set(a) & set(b))
                if len(seeds) < 2:
                    continue
                xa = [a[s] for s in seeds]; xb = [b[s] for s in seeds]
                delta = st.mean([v - u for u, v in zip(xa, xb)])
                improve = [(v - u) if higher else (u - v) for u, v in zip(xa, xb)]
                win = sum(1 for d in improve if d > 0) / len(improve)
                better = "BGF" if st.mean(improve) > 0 else "AdamW"
                p = float(stats.ttest_rel(xb, xa).pvalue)
                am, asd = st.mean(xa), st.stdev(xa)
                bm, bsd = st.mean(xb), st.stdev(xb)
                md.append(f"| {vlabel} | {m} | {len(seeds)} | {am:.4f}±{asd:.4f} | "
                          f"{bm:.4f}±{bsd:.4f} | {delta:+.4f} | {better} | "
                          f"{win*100:.0f}% | {p:.3g}{stars(p)} |")
                csv.append(f"{task},{variant},{m},{len(seeds)},{am:.6f},{asd:.6f},"
                           f"{bm:.6f},{bsd:.6f},{delta:+.6f},{better},{win:.2f},{p:.6g}")
                n_total += 1
                if p < 0.05:
                    n_bgf += better == "BGF"; n_adamw += better == "AdamW"
        md.append("")
    md.append(f"**Summary:** BGF significantly better on {n_bgf}/{n_total} comparisons, "
              f"AdamW on {n_adamw}/{n_total} (p<0.05).")
    Path(args.out + ".md").write_text("\n".join(md) + "\n")
    Path(args.out + ".csv").write_text("\n".join(csv) + "\n")
    print("\n".join(md))
    print(f"[written] {args.out}.md and {args.out}.csv")


if __name__ == "__main__":
    raise SystemExit(main())
