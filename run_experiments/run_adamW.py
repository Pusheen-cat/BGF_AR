"""AdamW baseline over the full paper grid.

Identical to run_adam.py except that the underlying optimizer is AdamW
(decoupled weight decay, default 1e-4 via --weight_decay).  Results go to
./results_adamw by default (folders ``baseline_adamw-...``) so they stay
separate from the Adam results.

Grid: 3 learning rates x 3 seeds x 15 tasks x 8 architectures = 1080 runs.
Example (all runs on GPU 0):        python run_adamW.py --cuda 0
Example (list the grid):            python run_adamW.py --dry_run
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from sweep_utils import base_command, launch, sweep_argparser

parser = sweep_argparser('AdamW baseline sweep (optim none, base_optim adamw)')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='decoupled weight decay')
parser.set_defaults(save_dir='./results_adamw')
args = parser.parse_args()

commands = []
for lr in args.lrs:
    for task in args.tasks:
        for seed in args.seeds:
            for architecture in args.architectures:
                folder_name = f'baseline_adamw-{lr}-{args.training_steps}'
                commands.append(base_command(args, folder_name, task, architecture, seed, lr)
                                + f'--optim none --base_optim adamw --weight_decay {args.weight_decay} ')

launch(commands, args)
