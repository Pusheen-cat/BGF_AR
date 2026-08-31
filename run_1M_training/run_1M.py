"""1M-step momentum experiment: Adam vs BGF on the MD task, 4 recurrent models.

Trains missing_duplicate_string (MD) for 1,000,000 steps with the four
recurrent architectures (rnn, lstm, stack_rnn, tape_rnn), for both methods
(Adam baseline and BGF) and three values of Adam's momentum (beta1) - 0.8,
0.9, 0.95.  BGF is built on top of Adam, so the momentum applies to both
methods.  BGF's weights are fixed to (a, b) = (0.95, 0.05); learning rate
5e-4; seeds 0, 1, 2.

Grid: 2 methods x 3 momenta x 4 architectures x 3 seeds = 72 runs.  Runs are
enumerated with 1-based indices and the slice [--start, --end) is executed
sequentially on the GPU given by --cuda (as in run_experiments/); use
--dry_run to list the grid and the restriction flags to run a subset.
Result folders embed the momentum as an ``-m{momentum}`` suffix.

After the runs finish, summarise them with ``summarize_results.py``.

Examples:
    python run_1M.py --dry_run                       # list all 72 runs
    python run_1M.py --cuda 0 --start 1 --end 37     # first half
    python run_1M.py --cuda 1 --start 37             # second half
    python run_1M.py --cuda 0 --methods bgf --momentums 0.9
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASIC_PARSER = os.path.join(os.path.dirname(HERE), 'run_experiments', 'basic_parser.py')

TASK = 'missing_duplicate_string'
ARCHITECTURES = ['rnn', 'lstm', 'stack_rnn', 'tape_rnn']
MOMENTUMS = [0.8, 0.9, 0.95]
SEEDS = [0, 1, 2]

parser = argparse.ArgumentParser(description='1M-step Adam/BGF momentum experiment on the MD task')
parser.add_argument('--cuda', default=0, type=int, help='CUDA device ID')
parser.add_argument('--start', default=1, type=int, help='1-based index of the first run to execute')
parser.add_argument('--end', default=1_000_000_000, type=int, help='runs with index < end are executed')
parser.add_argument('--save_dir', default=os.path.join(HERE, 'results_1M'), type=str)
parser.add_argument('--training_steps', default=1_000_000, type=int)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--methods', nargs='+', type=str, default=['adam', 'bgf'],
                    choices=['adam', 'bgf'])
parser.add_argument('--momentums', nargs='+', type=float, default=MOMENTUMS)
parser.add_argument('--architectures', nargs='+', type=str, default=ARCHITECTURES)
parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
parser.add_argument('--weight_a', default=0.95, type=float, help='BGF raw-gradient weight (fixed in the paper)')
parser.add_argument('--weight_b', default=0.05, type=float, help='BGF low-frequency weight')
parser.add_argument('--dry_run', action='store_true', help='print every command instead of running')
args = parser.parse_args()

save_dir = os.path.abspath(args.save_dir)

commands = []
for momentum in args.momentums:
    for method in args.methods:
        if method == 'adam':
            folder = f'baseline-{args.lr}-{args.training_steps}-m{momentum}'
            optim = '--optim none '
        else:
            folder = (f'ours_balance-{args.lr}-{args.weight_a}-{args.weight_b}-'
                      f'{args.training_steps}-m{momentum}')
            optim = f'--optim ours_balance --weight_a {args.weight_a} --weight_b {args.weight_b} '
        for architecture in args.architectures:
            for seed in args.seeds:
                commands.append(
                    f'python {BASIC_PARSER} --cuda {args.cuda} --folder_name {folder} '
                    f'--training_steps {args.training_steps} --seed {seed} --task {TASK} '
                    f'--architecture {architecture} --lr {args.lr} --momentum {momentum} '
                    f'--save_dir {save_dir} ' + optim)

total = len(commands)
print(f'Grid size: {total} runs; executing slice [{args.start}, {min(args.end, total + 1)})')
for idx, cmd in enumerate(commands, start=1):
    if idx < args.start or idx >= args.end:
        continue
    print(f'\n#### Run {idx}/{total}')
    print(cmd)
    if not args.dry_run:
        os.system(cmd)
