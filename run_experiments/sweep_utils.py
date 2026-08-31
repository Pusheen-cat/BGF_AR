"""Shared helpers for the run-all sweep scripts (run_adam / run_bgf / run_bgf_ema / run_add).

Each sweep script enumerates its full experiment grid as a list of
basic_parser.py commands and then executes the 1-based slice
[--start, --end) sequentially on the GPU selected with --cuda.  To split a
sweep over several GPUs or machines, launch the same script several times
with disjoint --start/--end ranges (use --dry_run to list all indices first).
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASIC_PARSER = os.path.join(HERE, 'basic_parser.py')

# The 15 tasks of the paper's main experiments.
TASKS = ['modular_arithmetic', 'parity_check', 'even_pairs', 'cycle_navigation',
         'modular_arithmetic_brackets', 'reverse_string', 'missing_duplicate_string',
         'duplicate_string', 'binary_addition', 'binary_multiplication', 'compute_sqrt',
         'odds_first', 'solve_equation', 'stack_manipulation', 'bucket_sort']

# 4 recurrent architectures + 4 transformer encoders, as in the paper.
ARCHITECTURES = ['rnn', 'lstm', 'stack_rnn', 'tape_rnn',
                 'transformer_encoder_sincos', 'transformer_encoder_none',
                 'transformer_encoder_alibi', 'transformer_encoder_relative']

LEARNING_RATES = [5e-4, 3e-4, 1e-4]
SEEDS = [0, 1, 2]
TRAINING_STEPS = 100_000

# BGF weight pairs (weight_a = raw gradient, weight_b = low-frequency gradient,
# weight_a + weight_b = 1).
BGF_WEIGHTS = [(0.7, 0.3), (0.8, 0.2), (0.9, 0.1), (0.95, 0.05)]

# ADD low-frequency weights beta (weight_a is fixed to 1, so the two gradient
# components are NOT balanced; see the paper's ablation).
ADD_BETAS = [0.5, 1.0, 2.0, 3.0]


def sweep_argparser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cuda', default=0, type=int, help='CUDA device ID')
    parser.add_argument('--start', default=1, type=int, help='1-based index of the first run to execute')
    parser.add_argument('--end', default=1_000_000_000, type=int, help='runs with index < end are executed')
    parser.add_argument('--save_dir', default='./results', type=str, help='root directory for results')
    parser.add_argument('--training_steps', default=TRAINING_STEPS, type=int)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--lrs', nargs='+', type=float, default=LEARNING_RATES)
    parser.add_argument('--tasks', nargs='+', type=str, default=TASKS)
    parser.add_argument('--architectures', nargs='+', type=str, default=ARCHITECTURES)
    parser.add_argument('--save_param', default=0, type=int,
                        help='1 saves parameter checkpoints (incl. the final step) - required for long_eval.py')
    parser.add_argument('--dry_run', action='store_true', help='print every command instead of running')
    return parser


def base_command(args, folder_name, task, architecture, seed, lr):
    return (f"python {BASIC_PARSER} --cuda {args.cuda} --folder_name {folder_name} "
            f"--training_steps {args.training_steps} --seed {seed} --task {task} "
            f"--architecture {architecture} --lr {lr} --save_dir {args.save_dir} "
            f"--save_param {args.save_param} ")


def launch(commands, args):
    total = len(commands)
    print(f'Grid size: {total} runs; executing slice [{args.start}, {min(args.end, total + 1)})')
    for idx, cmd in enumerate(commands, start=1):
        if idx < args.start or idx >= args.end:
            continue
        print(f'\n#### Run {idx}/{total}')
        print(cmd)
        if not args.dry_run:
            os.system(cmd)
