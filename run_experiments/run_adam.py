"""Adam baseline over the full paper grid.

Grid: 3 learning rates x 3 seeds x 15 tasks x 8 architectures = 1080 runs.
Example (all runs on GPU 0):        python run_adam.py --cuda 0
Example (first half only):          python run_adam.py --cuda 0 --end 541
Example (list the grid):            python run_adam.py --dry_run
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from sweep_utils import base_command, launch, sweep_argparser

args = sweep_argparser('Adam baseline sweep (optim none)').parse_args()

commands = []
for lr in args.lrs:
    for task in args.tasks:
        for seed in args.seeds:
            for architecture in args.architectures:
                folder_name = f'baseline-{lr}-{args.training_steps}'
                commands.append(base_command(args, folder_name, task, architecture, seed, lr)
                                + '--optim none ')

launch(commands, args)
