"""Fine-grained Adam-momentum sweep on the MD task (RNN and Tape-RNN).

Trains missing_duplicate_string (MD) with plain Adam while sweeping the
momentum (beta1) over 15 values from 0.01 to 0.99, for the rnn and tape_rnn
architectures.  Experimental details follow the paper's momentum analysis:
learning rate 5e-4, seeds {0, 1, 2}, 1,000,000 training steps.

Grid: 2 architectures x 15 momenta x 3 seeds = 90 runs.  Runs are enumerated
with 1-based indices and the slice [--start, --end) is executed sequentially
on the GPU given by --cuda; use --dry_run to list the grid and the
restriction flags to run a subset.  Result folders embed the momentum as an
``-m{momentum}`` suffix.

Summarise the finished runs with ``summarize_momentum.py``.

Examples:
    python run_momentum_sweep.py --dry_run                     # list all 90 runs
    python run_momentum_sweep.py --cuda 0 --start 1 --end 46   # split across GPUs
    python run_momentum_sweep.py --cuda 1 --start 46
    python run_momentum_sweep.py --cuda 0 --architectures rnn --momentums 0.9 0.95
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASIC_PARSER = os.path.join(os.path.dirname(HERE), 'run_experiments', 'basic_parser.py')

TASK = 'missing_duplicate_string'
ARCHITECTURES = ['rnn', 'tape_rnn']
MOMENTUMS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
             0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
SEEDS = [0, 1, 2]

parser = argparse.ArgumentParser(description='fine-grained Adam momentum sweep on the MD task')
parser.add_argument('--cuda', default=0, type=int, help='CUDA device ID')
parser.add_argument('--start', default=1, type=int, help='1-based index of the first run to execute')
parser.add_argument('--end', default=1_000_000_000, type=int, help='runs with index < end are executed')
parser.add_argument('--save_dir', default=os.path.join(HERE, 'results_momentum'), type=str)
parser.add_argument('--training_steps', default=1_000_000, type=int)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--architectures', nargs='+', type=str, default=ARCHITECTURES)
parser.add_argument('--momentums', nargs='+', type=float, default=MOMENTUMS)
parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
parser.add_argument('--dry_run', action='store_true', help='print every command instead of running')
args = parser.parse_args()

save_dir = os.path.abspath(args.save_dir)

commands = []
for architecture in args.architectures:
    for momentum in args.momentums:
        folder = f'baseline-{args.lr}-{args.training_steps}-m{momentum}'
        for seed in args.seeds:
            commands.append(
                f'python {BASIC_PARSER} --cuda {args.cuda} --folder_name {folder} '
                f'--training_steps {args.training_steps} --seed {seed} --task {TASK} '
                f'--architecture {architecture} --lr {args.lr} --momentum {momentum} '
                f'--save_dir {save_dir} --optim none ')

total = len(commands)
print(f'Grid size: {total} runs; executing slice [{args.start}, {min(args.end, total + 1)})')
for idx, cmd in enumerate(commands, start=1):
    if idx < args.start or idx >= args.end:
        continue
    print(f'\n#### Run {idx}/{total}')
    print(cmd)
    if not args.dry_run:
        os.system(cmd)
