"""EMA-based BGF on top of AdamW (optim ours_ema, base_optim adamw).

Identical to run_bgf_ema.py except that the underlying optimizer is AdamW
(decoupled weight decay, default 1e-4 via --weight_decay).  Results go to
./results_adamw by default (folders ``ours_ema_adamw-...``) so they stay
separate from the Adam results.

Grid: 3 learning rates x 4 weight pairs x 3 seeds x 15 tasks x 8 architectures
      = 4320 runs, at a single EMA smoothing factor (default 0.98,
      configurable with --ema_sm).

Example (all runs on GPU 0):        python run_bgf_ema_adamW.py --cuda 0
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from sweep_utils import BGF_WEIGHTS, base_command, launch, sweep_argparser

parser = sweep_argparser('EMA-based BGF sweep on AdamW (optim ours_ema, base_optim adamw)')
parser.add_argument('--weight_a', default=None, type=float, help='sweep only this raw-gradient weight')
parser.add_argument('--weight_b', default=None, type=float, help='sweep only this low-frequency weight')
parser.add_argument('--ema_sm', default=0.98, type=float, help='EMA smoothing factor')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='decoupled weight decay')
parser.set_defaults(save_dir='./results_adamw')
args = parser.parse_args()

if (args.weight_a is None) != (args.weight_b is None):
    raise ValueError('--weight_a and --weight_b must be given together')
weights = BGF_WEIGHTS if args.weight_a is None else [(args.weight_a, args.weight_b)]

commands = []
for lr in args.lrs:
    for (weight_a, weight_b) in weights:
        for task in args.tasks:
            for seed in args.seeds:
                for architecture in args.architectures:
                    folder_name = (f'ours_ema_adamw-{args.ema_sm}-{lr}-{weight_a}-{weight_b}-'
                                   f'{args.training_steps}')
                    commands.append(base_command(args, folder_name, task, architecture, seed, lr)
                                    + f'--optim ours_ema --weight_a {weight_a} --weight_b {weight_b} '
                                    + f'--ema_sm {args.ema_sm} '
                                    + f'--base_optim adamw --weight_decay {args.weight_decay} ')

launch(commands, args)
