"""Summary of the fine-grained Adam-momentum sweep (see run_momentum_sweep.py).

For every run it computes, from the per-step training log:

  * best_test_acc  - the maximum of the --ma_window-step (50) moving average
                     of the per-step validation accuracy (sequence length 100)
                     over the whole run: the best test accuracy ever reached.
  * steps_to_90    - the training step at which the trailing --ma_window-step
                     moving average of the TRAINING accuracy first reaches 90%.
  * steps_to_95    - the same for 95%.

and prints one table per architecture (momentum rows), reporting the mean
over the seeds; for the step counts, seeds that never reach the threshold are
excluded from the mean and the reached/found seed count is shown.

Outputs:
  * per-architecture tables on stdout,
  * <save_dir>/runs_momentum.csv    - every run's metrics (long form),
  * <save_dir>/summary_momentum.csv - the summary table.

Usage:  python summarize_momentum.py            # reads ./results_momentum
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='summary of the Adam momentum sweep')
parser.add_argument('--save_dir', default=os.path.join(HERE, 'results_momentum'), type=str)
parser.add_argument('--ma_window', default=50, type=int,
                    help='moving-average window (steps) for all three metrics')
parser.add_argument('--thresholds', nargs='+', type=float, default=[0.90, 0.95],
                    help='training-accuracy thresholds for the steps-to-reach metrics')
args = parser.parse_args()


def parse_folder(folder):
    """baseline-{lr}-{steps}-m{momentum} -> momentum, or None."""
    if not folder.startswith('baseline') or '-m' not in folder:
        return None
    try:
        return float(folder.rsplit('-m', 1)[1])
    except ValueError:
        return None


def run_metrics(logs_path):
    with open(logs_path) as f:
        logs = json.load(f)
    step_log = logs['step_log']
    steps = sorted(int(k) for k in step_log if str(k).isdigit())
    t_a = np.array([step_log[str(s)]['t_a'] for s in steps], dtype=float)
    v_a = np.array([step_log[str(s)]['v_a'] for s in steps], dtype=float)
    kernel = np.ones(args.ma_window) / args.ma_window
    t_ma = np.convolve(t_a, kernel, mode='valid')
    v_ma = np.convolve(v_a, kernel, mode='valid')

    out = dict(seed=int(logs['setting']['seed']),
               architecture=logs['setting']['architecture'],
               best_test_acc=float(v_ma.max()) if len(v_ma) else float(v_a.mean()))
    for thr in args.thresholds:
        hits = np.where(t_ma >= thr)[0]
        # the step at which the trailing MA first reaches the threshold
        out[f'steps_to_{round(thr * 100)}'] = \
            int(steps[hits[0] + args.ma_window - 1]) if len(hits) else None
    return out


# ---- collect every run ----------------------------------------------------- #
runs = []
for logs_path in sorted(glob.glob(os.path.join(args.save_dir, '*', '*', '*', '*', 'logs'))):
    folder = logs_path.split(os.sep)[-5]
    momentum = parse_folder(folder)
    if momentum is None:
        continue
    runs.append(dict(momentum=momentum, folder=folder, **run_metrics(logs_path)))

if not runs:
    raise SystemExit(f'no momentum-sweep logs found under {args.save_dir}')

architectures = [a for a in ['rnn', 'tape_rnn', 'lstm', 'stack_rnn']
                 if any(r['architecture'] == a for r in runs)]
momentums = sorted({r['momentum'] for r in runs})
step_keys = [f'steps_to_{round(thr * 100)}' for thr in args.thresholds]

with open(os.path.join(args.save_dir, 'runs_momentum.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['architecture', 'momentum', 'seed',
                                      'best_test_acc'] + step_keys + ['folder'])
    w.writeheader()
    for r in sorted(runs, key=lambda r: (r['architecture'], r['momentum'], r['seed'])):
        w.writerow({k: r[k] for k in w.fieldnames})

# ---- per-architecture tables ----------------------------------------------- #
print(f'{len(runs)} runs found under {args.save_dir}  '
      f'(moving-average window {args.ma_window} steps)\n')

summary_rows = []
for architecture in architectures:
    print(f'==== {architecture} ' + '=' * (60 - len(architecture)))
    header = f"{'momentum':>9} {'best_test_acc':>15} " + ''.join(
        f"{k + ' (reached)':>22} " for k in step_keys) + f"{'seeds':>6}"
    print(header)
    print('-' * len(header))
    for momentum in momentums:
        sel = [r for r in runs if r['architecture'] == architecture
               and r['momentum'] == momentum]
        if not sel:
            continue
        accs = [r['best_test_acc'] for r in sel]
        row = dict(architecture=architecture, momentum=momentum, seeds_found=len(sel),
                   best_test_acc_mean=round(float(np.mean(accs)), 4),
                   best_test_acc_std=round(float(np.std(accs)), 4))
        line = f'{momentum:>9} {np.mean(accs):>7.3f}+-{np.std(accs):<6.3f} '
        for k in step_keys:
            reached = [r[k] for r in sel if r[k] is not None]
            row[f'{k}_mean'] = round(float(np.mean(reached)), 1) if reached else None
            row[f'{k}_reached'] = len(reached)
            txt = f'{np.mean(reached):>12.0f} ({len(reached)}/{len(sel)})' if reached \
                else f"{'-':>12} (0/{len(sel)})"
            line += f'{txt:>22} '
        line += f'{len(sel):>6}'
        print(line)
        summary_rows.append(row)
    print()

out_csv = os.path.join(args.save_dir, 'summary_momentum.csv')
fieldnames = ['architecture', 'momentum', 'best_test_acc_mean', 'best_test_acc_std'] + \
    [f'{k}_{suffix}' for k in step_keys for suffix in ('mean', 'reached')] + ['seeds_found']
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(summary_rows)
print(f'wrote {os.path.join(args.save_dir, "runs_momentum.csv")}')
print(f'wrote {out_csv}')
