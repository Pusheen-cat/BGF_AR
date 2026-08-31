#!/usr/bin/env python
"""Compare DNA (DeepSEA) results and write summary tables.

Two modes:

  --mode compare  (default)
      Paired AdamW-vs-BGF comparison per architecture. For each metric it reports
      mean +/- std over seeds, Delta = BGF - AdamW, a paired t-test and a Wilcoxon
      signed-rank test. Only seed-pairs finite on both arms are used (divergence-
      aware). Reads outputs/<arch>_{adamw,bgf}/seed*/metrics/run_test_metrics.json.

  --mode lambda
      Per-lambda summary for RNN/LSTM (mean +/- std over seeds), for the sweep
      produced by reproduce_lambda_rnn_lstm.py. Reads outputs_lambda/<arch>/lam*/.

Both metrics are higher-is-better, so positive Delta favors BGF.
Writes <out>.md and <out>.csv (default: results_dna_compare / results_dna_lambda).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from scipy import stats

METRICS = ["mean_auroc", "mean_auprc", "median_auroc", "median_auprc"]
ARCHS = ["rnn", "lstm", "stack_rnn", "tape_rnn"]


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def load_seeds(root: Path):
    """seed -> test-metrics dict, for every run under <root>/seed*."""
    out = {}
    if not root.exists():
        return out
    for sd in sorted(root.glob("seed*")):
        fp = sd / "metrics" / "run_test_metrics.json"
        if fp.exists():
            out[int(sd.name[4:])] = json.loads(fp.read_text())
    return out


def paired(a: dict, b: dict, metric: str):
    """Paired stats over seeds finite on both arms. Returns dict or None."""
    seeds = sorted(set(a) & set(b))
    xa = [a[s].get(metric) for s in seeds]
    xb = [b[s].get(metric) for s in seeds]
    pairs = [(u, v) for u, v in zip(xa, xb) if finite(u) and finite(v)]
    if len(pairs) < 2:
        return None
    xa = [u for u, _ in pairs]; xb = [v for _, v in pairs]
    p = float(stats.ttest_rel(xb, xa).pvalue)
    try:
        w = float(stats.wilcoxon(xb, xa).pvalue)
    except ValueError:
        w = float("nan")
    return dict(n=len(pairs), adamw=st.mean(xa), adamw_sd=st.stdev(xa),
                bgf=st.mean(xb), bgf_sd=st.stdev(xb),
                delta=st.mean([v - u for u, v in zip(xa, xb)]), p=p, w=w)


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


def mode_compare(outroot: Path, out: str):
    rows = []
    md = ["# DNA (DeepSEA) — AdamW vs. BGF (alpha=0.95)\n",
          "Paired over seeds; Delta = BGF - AdamW (both metrics higher-is-better, "
          "so positive favors BGF). t-p = paired t-test, W-p = Wilcoxon. "
          "* p<0.05, ** p<0.01, *** p<0.001.\n"]
    for arch in ARCHS:
        a = load_seeds(outroot / f"{arch}_adamw")
        b = load_seeds(outroot / f"{arch}_bgf")
        if not a or not b:
            continue
        md += [f"## {arch}\n",
               "| metric | n | AdamW | BGF | Delta | t-p | W-p |",
               "|---|---|---|---|---|---|---|"]
        for m in METRICS:
            r = paired(a, b, m)
            if r is None:
                continue
            md.append(f"| {m} | {r['n']} | {r['adamw']:.4f}±{r['adamw_sd']:.4f} | "
                      f"{r['bgf']:.4f}±{r['bgf_sd']:.4f} | {r['delta']:+.4f} | "
                      f"{r['p']:.3g}{stars(r['p'])} | {r['w']:.3g} |")
            rows.append((arch, m, r["n"], r["adamw"], r["adamw_sd"], r["bgf"],
                         r["bgf_sd"], r["delta"], r["p"], r["w"]))
        md.append("")
    Path(out + ".md").write_text("\n".join(md) + "\n")
    with open(out + ".csv", "w") as f:
        f.write("arch,metric,n,adamw_mean,adamw_std,bgf_mean,bgf_std,delta,ttest_p,wilcoxon_p\n")
        for r in rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print("\n".join(md))
    print(f"[written] {out}.md and {out}.csv")


def mode_lambda(outroot: Path, out: str):
    md = ["# DNA (DeepSEA) — BGF bgf_lambda sweep (RNN & LSTM, alpha=0.95)\n",
          "Mean +/- std over seeds. Only RNN and LSTM are covered.\n"]
    csv = ["arch,lambda,metric,n,mean,std"]
    for arch in ["rnn", "lstm"]:
        base = outroot / arch
        if not base.exists():
            continue
        lam_dirs = sorted(base.glob("lam*"))
        md += [f"## {arch}\n",
               "| lambda | " + " | ".join(METRICS) + " | n |",
               "|---|" + "---|" * (len(METRICS) + 1)]
        for ld in lam_dirs:
            lam = ld.name[3:].replace("p", ".")
            seeds = load_seeds(ld)
            cells = []
            n_last = 0
            for m in METRICS:
                vals = [d[m] for d in seeds.values() if finite(d.get(m))]
                if vals:
                    mean = st.mean(vals); sd = st.stdev(vals) if len(vals) > 1 else 0.0
                    cells.append(f"{mean:.4f}±{sd:.4f}"); n_last = len(vals)
                    csv.append(f"{arch},{lam},{m},{len(vals)},{mean:.6f},{sd:.6f}")
                else:
                    cells.append("—")
            md.append(f"| {lam} | " + " | ".join(cells) + f" | {n_last} |")
        md.append("")
    Path(out + ".md").write_text("\n".join(md) + "\n")
    Path(out + ".csv").write_text("\n".join(csv) + "\n")
    print("\n".join(md))
    print(f"[written] {out}.md and {out}.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["compare", "lambda"], default="compare")
    ap.add_argument("--outroot", default=None, help="run outputs dir (defaults per mode)")
    ap.add_argument("--out", default=None, help="output file stem (defaults per mode)")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    if args.mode == "compare":
        outroot = Path(args.outroot) if args.outroot else repo / "outputs"
        mode_compare(outroot, args.out or str(repo / "results_dna_compare"))
    else:
        outroot = Path(args.outroot) if args.outroot else repo / "outputs_lambda"
        mode_lambda(outroot, args.out or str(repo / "results_dna_lambda"))


if __name__ == "__main__":
    raise SystemExit(main())
