"""Chomsky-level summary of the sharpness measures: Adam vs BGF, paired t-test.

Aggregates the ``sharpness.json`` files written by sharp_eval.py.  For each of
the three sharpness measures (low_pass, fim, shannon_entropy) and each Chomsky
hierarchy level, the Adam and BGF results of the four recurrent models (rnn,
lstm, stack_rnn, tape_rnn) and all tasks of that level are PAIRED on
(task, architecture, seed).  If either member of a pair is NaN (sharpness
could not be calculated) the pair is excluded.  The remaining matched pairs
give:

    * the mean sharpness of Adam and of BGF within the level, and
    * the p-value of a paired t-test (scipy.stats.ttest_rel) between them:
        Adam-Regular          vs.  BGF-Regular
        Adam-Context-Free     vs.  BGF-Context-Free
        Adam-Context-Sensitive vs. BGF-Context-Sensitive

Adam runs are the ``baseline-*`` folders, BGF runs the ``ours_balance-*``
folders.  If several folders of one method contain the same
(task, architecture, seed) - e.g. several learning rates or BGF weights -
restrict them with --adam_folders/--bgf_folders; otherwise the
alphabetically first folder is used (with a warning).

The evaluated sharpness value is the training-distribution one by default
(--phase train); use ``--phase val --length 100`` for a fixed length.

Usage:  python analyze_sharpness.py --save_dir ../run_experiments/results
Output: per-metric tables on stdout + <save_dir>/sharpness_summary.csv
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import ttest_rel

METRICS = ['low_pass', 'fim', 'shannon_entropy']
ARCHITECTURES = ['rnn', 'lstm', 'stack_rnn', 'tape_rnn']

# The 15 tasks by Chomsky hierarchy level (as in tasks/{regular,dcf,cs}/).
CHOMSKY_LEVELS = {
    'Regular': ['modular_arithmetic', 'parity_check', 'even_pairs', 'cycle_navigation'],
    'Context-Free': ['modular_arithmetic_brackets', 'reverse_string', 'solve_equation',
                     'stack_manipulation'],
    'Context-Sensitive': ['binary_addition', 'binary_multiplication', 'bucket_sort',
                          'compute_sqrt', 'duplicate_string', 'missing_duplicate_string',
                          'odds_first'],
}

parser = argparse.ArgumentParser(description='Adam-vs-BGF sharpness summary by Chomsky level')
parser.add_argument('--save_dir', default='../run_experiments/results', type=str,
                    help='results root containing the sharpness.json files')
parser.add_argument('--phase', default='train', choices=['train', 'val'],
                    help='which evaluation distribution to analyse')
parser.add_argument('--length', default=100, type=int,
                    help="sequence length when --phase val")
parser.add_argument('--adam_folders', nargs='+', type=str, default=None,
                    help='restrict the Adam runs to these folders')
parser.add_argument('--bgf_folders', nargs='+', type=str, default=None,
                    help='restrict the BGF runs to these folders')
args = parser.parse_args()


def metric_value(mtr, metric):
    """The analysed scalar of one run; NaN if absent or not computable."""
    entry = mtr.get(metric)
    if entry is None:
        return float('nan')
    if args.phase == 'train':
        v = entry.get('train')
    else:
        v = entry.get('val', {}).get(str(args.length), entry.get('val', {}).get(args.length))
    return float('nan') if v is None else float(v)


# ---- collect: method -> (task, architecture, seed) -> metric values -------- #
values = {'Adam': {}, 'BGF': {}}
key_folder = {'Adam': {}, 'BGF': {}}
folders_used = {'Adam': set(), 'BGF': set()}
for path in sorted(glob.glob(os.path.join(args.save_dir, '*', '*', '*', '*', 'sharpness.json'))):
    # layout: <save_dir>/<folder>/<architecture>/<task>/<seedN_date>/sharpness.json
    parts = path.split(os.sep)
    task, architecture, folder = parts[-3], parts[-4], parts[-5]
    if folder.startswith('baseline'):
        method = 'Adam'
        if args.adam_folders is not None and folder not in args.adam_folders:
            continue
    elif folder.startswith('ours_balance'):
        method = 'BGF'
        if args.bgf_folders is not None and folder not in args.bgf_folders:
            continue
    else:
        continue
    if architecture not in ARCHITECTURES:
        continue
    with open(path) as f:
        mtr = json.load(f)
    key = (task, architecture, int(mtr['seed']))
    if key in values[method]:
        print(f'WARNING: duplicate run for {method} {key}: keeping the one from '
              f'{key_folder[method][key]}, ignoring {folder} '
              f'(restrict with --adam_folders/--bgf_folders)')
        continue
    values[method][key] = {m: metric_value(mtr, m) for m in METRICS}
    key_folder[method][key] = folder
    folders_used[method].add(folder)

n_adam, n_bgf = len(values['Adam']), len(values['BGF'])
if n_adam == 0 or n_bgf == 0:
    raise SystemExit(f'found {n_adam} Adam and {n_bgf} BGF sharpness.json runs under '
                     f'{args.save_dir} - run sharp_eval.py first')
print(f"runs found: Adam {n_adam} (folders {sorted(folders_used['Adam'])}), "
      f"BGF {n_bgf} (folders {sorted(folders_used['BGF'])})")
phase_txt = 'train distribution' if args.phase == 'train' else f'validation length {args.length}'
print(f'analysed value: {phase_txt}\n')

# ---- pair, exclude NaN pairs, t-test per metric and level ------------------ #
rows = []
for metric in METRICS:
    print(f'==== {metric} ' + '=' * (58 - len(metric)))
    header = (f"{'level':<18} {'mean Adam':>12} {'mean BGF':>12} {'p (paired t)':>13} "
              f"{'pairs':>7} {'excluded':>9}")
    print(header)
    print('-' * len(header))
    for level, tasks in CHOMSKY_LEVELS.items():
        adam_vals, bgf_vals = [], []
        n_total = 0
        for key in sorted(set(values['Adam']) & set(values['BGF'])):
            task, architecture, _seed = key
            if task not in tasks:
                continue
            n_total += 1
            a = values['Adam'][key][metric]
            b = values['BGF'][key][metric]
            if np.isfinite(a) and np.isfinite(b):     # NaN in either -> drop the pair
                adam_vals.append(a)
                bgf_vals.append(b)
        n_used = len(adam_vals)
        if n_used >= 2:
            p = float(ttest_rel(adam_vals, bgf_vals).pvalue)
        else:
            p = float('nan')
        mean_a = float(np.mean(adam_vals)) if n_used else float('nan')
        mean_b = float(np.mean(bgf_vals)) if n_used else float('nan')
        print(f'{level:<18} {mean_a:>12.6g} {mean_b:>12.6g} {p:>13.4g} '
              f'{n_used:>7} {n_total - n_used:>9}')
        rows.append(dict(metric=metric, level=level, phase=phase_txt,
                         mean_adam=mean_a, mean_bgf=mean_b, p_value=p,
                         pairs_used=n_used, pairs_excluded=n_total - n_used))
    print()

out_csv = os.path.join(args.save_dir, 'sharpness_summary.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['metric', 'level', 'phase', 'mean_adam', 'mean_bgf',
                                      'p_value', 'pairs_used', 'pairs_excluded'])
    w.writeheader()
    w.writerows(rows)
print(f'wrote {out_csv}')
