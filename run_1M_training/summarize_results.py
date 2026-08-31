"""Summary table of the 1M-step momentum experiment (see run_1M.py).

Scans the results written by run_1M.py and, for every setting
(momentum x architecture x method), identifies the BEST generalization result
across the three seeds and prints/writes a summary table.

Per-run metrics (from the run's ``logs`` file):
  * train_fit    - mean end-of-training accuracy over the TRAINING lengths
                   1..sequence_length (40), from the range evaluation.  Used
                   as a fitting guard: a run that never fitted the training
                   distribution has no meaningful generalization score.
  * final_valid  - mean per-step validation accuracy (length 100) over the
                   last --ma_window (50) training steps: the end-of-training
                   generalization accuracy.  DEFAULT metric.
  * max_valid    - maximum of the --ma_window-step moving average of the
                   validation accuracy over the whole run (best
                   generalization ever reached during training).
  * range_score  - mean end-of-training accuracy over the UNSEEN lengths
                   sequence_length+1..100, from the range evaluation.

Selection per setting: "best" = the maximum of --metric across the seeds
whose train_fit >= --min_train_acc (0.98).  If no seed passes the guard the
maximum over all seeds is reported and marked with '*'.  The table also
reports mean +- std over all found seeds.

Outputs:
  * a per-momentum table on stdout,
  * <save_dir>/runs_1M.csv     - every run's metrics (long form),
  * <save_dir>/summary_1M.csv  - the summary table.

Usage:  python summarize_results.py             # reads ./results_1M
        python summarize_results.py --metric range_score
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

METRICS = ['final_valid', 'max_valid', 'range_score']

parser = argparse.ArgumentParser(description='summary table of the 1M-step momentum experiment')
parser.add_argument('--save_dir', default=os.path.join(HERE, 'results_1M'), type=str)
parser.add_argument('--metric', default='final_valid', choices=METRICS,
                    help='generalization metric shown in the table')
parser.add_argument('--ma_window', default=50, type=int,
                    help='moving-average window (steps) for the validation-accuracy metrics')
parser.add_argument('--min_train_acc', default=0.98, type=float,
                    help='fitting guard: seeds below this train_fit are excluded from "best"')
args = parser.parse_args()


def parse_folder(folder):
    """folder name -> (method, momentum) or None if it is not a run_1M folder."""
    if '-m' not in folder:
        return None
    head, mom = folder.rsplit('-m', 1)
    try:
        momentum = float(mom)
    except ValueError:
        return None
    if head.startswith('baseline-'):
        return 'Adam', momentum
    if head.startswith('ours_balance-'):
        return 'BGF', momentum
    return None


def run_metrics(logs_path):
    """The per-run metrics dict, computed from one 'logs' file."""
    with open(logs_path) as f:
        logs = json.load(f)
    step_log = logs['step_log']
    steps = sorted(int(k) for k in step_log if str(k).isdigit())
    v_a = np.array([step_log[str(s)]['v_a'] for s in steps], dtype=float)
    ma = np.convolve(v_a, np.ones(args.ma_window) / args.ma_window, mode='valid')

    seq_len = logs['setting'].get('sequence_length', 40)
    lengths = sorted(int(k) for k in logs['range_eval'])
    acc = np.array([logs['range_eval'][str(l)]['final_acc'] for l in lengths], dtype=float)
    train_mask = np.array(lengths) <= seq_len

    return dict(
        train_fit=float(acc[train_mask].mean()),
        final_valid=float(v_a[-args.ma_window:].mean()),
        max_valid=float(ma.max()) if len(ma) else float(v_a.mean()),
        range_score=float(acc[~train_mask].mean()),
        seed=int(logs['setting']['seed']),
        architecture=logs['setting']['architecture'],
    )


# ---- collect every run ----------------------------------------------------- #
runs = []          # list of dicts: method, momentum, architecture, seed, metrics
for logs_path in sorted(glob.glob(os.path.join(args.save_dir, '*', '*', '*', '*', 'logs'))):
    folder = logs_path.split(os.sep)[-5]
    parsed = parse_folder(folder)
    if parsed is None:
        continue
    method, momentum = parsed
    m = run_metrics(logs_path)
    runs.append(dict(method=method, momentum=momentum, folder=folder, **m))

if not runs:
    raise SystemExit(f'no run_1M logs found under {args.save_dir}')

momentums = sorted({r['momentum'] for r in runs})
architectures = [a for a in ['rnn', 'lstm', 'stack_rnn', 'tape_rnn']
                 if any(r['architecture'] == a for r in runs)]
methods = ['Adam', 'BGF']

with open(os.path.join(args.save_dir, 'runs_1M.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['method', 'momentum', 'architecture', 'seed',
                                      'train_fit'] + METRICS + ['folder'])
    w.writeheader()
    for r in sorted(runs, key=lambda r: (r['momentum'], r['method'], r['architecture'], r['seed'])):
        w.writerow({k: r[k] for k in w.fieldnames})

# ---- summarise: best over seeds per (momentum, architecture, method) ------- #
def summarise(method, momentum, architecture):
    sel = [r for r in runs if r['method'] == method and r['momentum'] == momentum
           and r['architecture'] == architecture]
    if not sel:
        return None
    vals = np.array([r[args.metric] for r in sel])
    fit = [r for r in sel if r['train_fit'] >= args.min_train_acc]
    pool = fit if fit else sel
    best_run = max(pool, key=lambda r: r[args.metric])
    return dict(n=len(sel), n_fit=len(fit),
                best=best_run[args.metric], best_seed=best_run['seed'],
                guarded=bool(fit), mean=float(vals.mean()), std=float(vals.std()))


print(f'{len(runs)} runs found under {args.save_dir}')
print(f"metric: {args.metric}  (fitting guard: train_fit >= {args.min_train_acc}; "
      f"'*' = no seed passed the guard, best over all seeds shown)\n")

summary_rows = []
for momentum in momentums:
    print(f'==== momentum {momentum} ' + '=' * 46)
    header = (f"{'architecture':<14}" +
              ''.join(f"| {m + ' best':>12} {'mean+-std':>15} {'seeds':>6} " for m in methods))
    print(header)
    print('-' * len(header))
    for architecture in architectures:
        line = f'{architecture:<14}'
        for method in methods:
            s = summarise(method, momentum, architecture)
            if s is None:
                line += f"| {'-':>12} {'-':>15} {'0':>6} "
                continue
            mark = '' if s['guarded'] else '*'
            line += (f"| {s['best']:.3f} (s{s['best_seed']}){mark:<1} "
                     f"{s['mean']:.3f}+-{s['std']:.3f} {s['n_fit']}/{s['n']:>2}   ")
            summary_rows.append(dict(momentum=momentum, architecture=architecture,
                                     method=method, metric=args.metric,
                                     best=round(s['best'], 4), best_seed=s['best_seed'],
                                     fit_guard_passed=s['guarded'],
                                     mean=round(s['mean'], 4), std=round(s['std'], 4),
                                     seeds_fit=s['n_fit'], seeds_found=s['n']))
        print(line)
    print()

out_csv = os.path.join(args.save_dir, 'summary_1M.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['momentum', 'architecture', 'method', 'metric', 'best',
                                      'best_seed', 'fit_guard_passed', 'mean', 'std',
                                      'seeds_fit', 'seeds_found'])
    w.writeheader()
    w.writerows(summary_rows)
print(f'wrote {os.path.join(args.save_dir, "runs_1M.csv")}')
print(f'wrote {out_csv}')
