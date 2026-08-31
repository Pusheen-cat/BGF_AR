"""Reproduce the paper's toy experiments (spurious-correlation ablations).

The file paper_toy_configs.json lists every (method, task, architecture, lr,
weights, spurious probabilities) configuration that was actually run for the
paper:
  * label  - label concatenation: the one-hot true label is appended to the
             input with probability alpha (else a random label); 14 tasks x
             4 recurrent architectures; methods adam / bgf / add.
  * length - input-length confounding: training batches are resampled so the
             label correlates with the (binned) input length at strength
             alpha; 7 single-output tasks x 4 recurrent architectures.

This script replays those configurations (x 3 seeds by default).  Runs are
enumerated with 1-based indices; execute a slice with --start/--end and list
all indices with --dry_run.  To run a configuration that is not part of the
paper grid, call spu_parser.py / spu_len_parser.py directly.

Examples:
    python run_toy.py --experiment label  --dry_run
    python run_toy.py --experiment label  --cuda 0 --start 1 --end 501
    python run_toy.py --experiment length --cuda 1 --tasks parity_check
    python run_toy.py --experiment label  --methods bgf --spu_probs 0.9 0.99
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

MANIFEST = {'label': 'label_concatenation', 'length': 'length_confounding'}
PARSER_SCRIPT = {'label': 'spu_parser.py', 'length': 'spu_len_parser.py'}
OPTIM = {'adam': 'none', 'bgf': 'ours_balance', 'add': 'ours_add'}
FOLDER_PREFIX = {'adam': 'baseline', 'bgf': 'ours_balance', 'add': 'ours_add'}

parser = argparse.ArgumentParser(description='replay the paper toy-experiment grid')
parser.add_argument('--experiment', required=True, choices=['label', 'length'])
parser.add_argument('--cuda', default=0, type=int, help='CUDA device ID')
parser.add_argument('--start', default=1, type=int, help='1-based index of the first run to execute')
parser.add_argument('--end', default=1_000_000_000, type=int, help='runs with index < end are executed')
parser.add_argument('--save_dir', default=None, type=str,
                    help='root results directory (default ./results_toy_<experiment>)')
parser.add_argument('--training_steps', default=100_000, type=int)
parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
parser.add_argument('--methods', nargs='+', type=str, default=['adam', 'bgf', 'add'],
                    choices=['adam', 'bgf', 'add'], help='restrict to these methods')
parser.add_argument('--tasks', nargs='+', type=str, default=None, help='restrict to these tasks')
parser.add_argument('--architectures', nargs='+', type=str, default=None, help='restrict to these architectures')
parser.add_argument('--spu_probs', nargs='+', type=float, default=None,
                    help='override the per-config spurious probabilities of the manifest')
parser.add_argument('--dry_run', action='store_true', help='print every command instead of running')
args = parser.parse_args()

save_dir = args.save_dir or f'./results_toy_{args.experiment}'
script = os.path.join(HERE, PARSER_SCRIPT[args.experiment])

with open(os.path.join(HERE, 'paper_toy_configs.json')) as f:
    configs = json.load(f)[MANIFEST[args.experiment]]

commands = []
for cfg in configs:
    if cfg['method'] not in args.methods:
        continue
    if args.tasks is not None and cfg['task'] not in args.tasks:
        continue
    if args.architectures is not None and cfg['architecture'] not in args.architectures:
        continue
    probs = args.spu_probs if args.spu_probs is not None else cfg['spu_probs']
    lr, weight_a, weight_b = cfg['lr'], cfg['weight_a'], cfg['weight_b']
    for spu_prob in probs:
        # Unlike the internal experiment code, the spurious probability is part
        # of the folder name, so runs at different strengths never mix.
        if cfg['method'] == 'adam':
            folder_name = f'baseline-{lr}-p{spu_prob}-{args.training_steps}'
        else:
            folder_name = (f'{FOLDER_PREFIX[cfg["method"]]}-{lr}-{weight_a}-{weight_b}'
                           f'-p{spu_prob}-{args.training_steps}')
        for seed in args.seeds:
            commands.append(
                f"python {script} --cuda {args.cuda} --folder_name {folder_name} "
                f"--training_steps {args.training_steps} --seed {seed} "
                f"--task {cfg['task']} --architecture {cfg['architecture']} "
                f"--optim {OPTIM[cfg['method']]} --weight_a {weight_a} --weight_b {weight_b} "
                f"--lr {lr} --spu_prob {spu_prob} --save_dir {save_dir} ")

total = len(commands)
print(f'Grid size: {total} runs; executing slice [{args.start}, {min(args.end, total + 1)})')
for idx, cmd in enumerate(commands, start=1):
    if idx < args.start or idx >= args.end:
        continue
    print(f'\n#### Run {idx}/{total}')
    print(cmd)
    if not args.dry_run:
        os.system(cmd)
