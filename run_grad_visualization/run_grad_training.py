"""Training stage of the gradient-frequency visualization experiment.

Launches the two runs behind the paper's gradient-FFT figures - the Adam
baseline and BGF - on the Solve Equation task with the LSTM model, with
``--save_grad_signal 1`` so that 500 sampled gradient coordinates are recorded
at EVERY training step (full-tree gradient dumps would require excessive
storage; see training/grad_signal.py).  Each run writes
``<run_dir>/grad_signal.pkl`` next to its ``logs`` file; the figures are then
generated with ``plot_grad_fft.py``.

Defaults reproduce the paper's setting: lr 5e-4, seed 2, 100k steps and BGF
weights (a, b) = (0.9, 0.1).

Examples:
    python run_grad_training.py --cuda 0                  # both runs
    python run_grad_training.py --cuda 0 --methods adam   # Adam run only
    python run_grad_training.py --cuda 1 --methods bgf    # BGF run only
    python run_grad_training.py --dry_run
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASIC_PARSER = os.path.join(os.path.dirname(HERE), 'run_experiments', 'basic_parser.py')

parser = argparse.ArgumentParser(description='train the Adam and BGF runs of the gradient-frequency experiment')
parser.add_argument('--cuda', default=0, type=int, help='CUDA device ID')
parser.add_argument('--save_dir', default=os.path.join(HERE, 'results_grad'), type=str,
                    help='root directory for the two runs')
parser.add_argument('--methods', nargs='+', type=str, default=['adam', 'bgf'],
                    choices=['adam', 'bgf'], help='which of the two runs to launch')
parser.add_argument('--task', default='solve_equation', type=str,
                    help='task (the paper figure uses solve_equation)')
parser.add_argument('--architecture', default='lstm', type=str,
                    help='architecture (the paper figure uses lstm)')
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--seed', default=2, type=int)
parser.add_argument('--training_steps', default=100_000, type=int)
parser.add_argument('--weight_a', default=0.9, type=float, help='BGF raw-gradient weight')
parser.add_argument('--weight_b', default=0.1, type=float, help='BGF low-frequency weight')
parser.add_argument('--dry_run', action='store_true', help='print the commands instead of running')
args = parser.parse_args()

save_dir = os.path.abspath(args.save_dir)

commands = []
for method in args.methods:
    if method == 'adam':
        folder = f'baseline-{args.lr}-{args.training_steps}'
        optim = '--optim none '
    else:
        folder = f'ours_balance-{args.lr}-{args.weight_a}-{args.weight_b}-{args.training_steps}'
        optim = f'--optim ours_balance --weight_a {args.weight_a} --weight_b {args.weight_b} '
    commands.append(
        f'python {BASIC_PARSER} --cuda {args.cuda} --folder_name {folder} '
        f'--training_steps {args.training_steps} --seed {args.seed} --task {args.task} '
        f'--architecture {args.architecture} --lr {args.lr} --save_dir {save_dir} '
        f'--save_grad_signal 1 ' + optim)

for idx, cmd in enumerate(commands, start=1):
    print(f'\n#### Run {idx}/{len(commands)}')
    print(cmd)
    if not args.dry_run:
        os.system(cmd)
