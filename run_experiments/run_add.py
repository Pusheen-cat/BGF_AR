"""ADD ablation (optim ours_add) over the full paper grid.

ADD is the naive, non-balanced variant of BGF:
    g_add = alpha * g + beta * g_low     with alpha = 1 and beta > 0,
i.e. it omits BGF's balancing / norm-matching step (alpha + beta != 1 in
general).  The paper sweeps beta over {0.5, 1.0, 2.0, 3.0}.

Grid: 3 learning rates x 4 betas x 3 seeds x 15 tasks x 8 architectures
      = 4320 runs.
Example (all runs on GPU 0):        python run_add.py --cuda 0
Example (single beta):              python run_add.py --cuda 0 --betas 2.0
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from sweep_utils import ADD_BETAS, base_command, launch, sweep_argparser

parser = sweep_argparser('ADD sweep (optim ours_add, weight_a = 1)')
parser.add_argument('--betas', nargs='+', type=float, default=ADD_BETAS,
                    help='low-frequency weights beta (weight_b); weight_a is fixed to 1')
parser.add_argument('--queue_size', default=100, type=int, help='queue size (lambda) of the gradient queue')
args = parser.parse_args()

weight_a = 1.0
queue_tag = '' if args.queue_size == 100 else f'-q{args.queue_size}'

commands = []
for lr in args.lrs:
    for beta in args.betas:
        for task in args.tasks:
            for seed in args.seeds:
                for architecture in args.architectures:
                    folder_name = f'ours_add-{lr}-{weight_a}-{beta}-{args.training_steps}{queue_tag}'
                    commands.append(base_command(args, folder_name, task, architecture, seed, lr)
                                    + f'--optim ours_add --weight_a {weight_a} --weight_b {beta} '
                                    + f'--queue_size {args.queue_size} ')

launch(commands, args)
