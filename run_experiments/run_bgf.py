"""Queue-based BGF (optim ours_balance) over the full paper grid.

Grid: 3 learning rates x 4 weight pairs x 3 seeds x 15 tasks x 8 architectures
      = 4320 runs.
The queue size (lambda) defaults to 100 and can be changed with --queue_size.

Example (all runs on GPU 0):        python run_bgf.py --cuda 0
Example (single weight pair):       python run_bgf.py --cuda 0 --weight_a 0.7 --weight_b 0.3
Example (queue-size ablation):      python run_bgf.py --cuda 0 --queue_size 300
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from sweep_utils import BGF_WEIGHTS, base_command, launch, sweep_argparser

parser = sweep_argparser('Queue-based BGF sweep (optim ours_balance)')
parser.add_argument('--weight_a', default=None, type=float, help='sweep only this raw-gradient weight')
parser.add_argument('--weight_b', default=None, type=float, help='sweep only this low-frequency weight')
parser.add_argument('--queue_size', default=100, type=int, help='queue size (lambda) of the gradient queue')
args = parser.parse_args()

if (args.weight_a is None) != (args.weight_b is None):
    raise ValueError('--weight_a and --weight_b must be given together')
weights = BGF_WEIGHTS if args.weight_a is None else [(args.weight_a, args.weight_b)]

# The queue size is part of the folder name only when it deviates from the
# paper default, so default runs keep the paper's folder naming scheme.
queue_tag = '' if args.queue_size == 100 else f'-q{args.queue_size}'

commands = []
for lr in args.lrs:
    for (weight_a, weight_b) in weights:
        for task in args.tasks:
            for seed in args.seeds:
                for architecture in args.architectures:
                    folder_name = f'ours_balance-{lr}-{weight_a}-{weight_b}-{args.training_steps}{queue_tag}'
                    commands.append(base_command(args, folder_name, task, architecture, seed, lr)
                                    + f'--optim ours_balance --weight_a {weight_a} --weight_b {weight_b} '
                                    + f'--queue_size {args.queue_size} ')

launch(commands, args)
