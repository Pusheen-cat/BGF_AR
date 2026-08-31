"""Standalone long-range generalization evaluation (the paper's up-to-length-1000 figure).

This script is fully independent of the training pipeline and of the standard
end-of-training evaluation over lengths 1-100: it only needs a finished run
directory containing a saved checkpoint.  It rebuilds the model, loads
``<run_dir>/params/param_s<step>.pt`` (written when training ran with
``--save_param 1``) and evaluates accuracy on EVERY sequence length
``1..--max_length`` (default 1000).  The default task is
``missing_duplicate_string`` (MD), the task shown in the paper's figure.

Prerequisite (training with checkpointing enabled):
    cd ../run_experiments
    python run_adam.py --tasks missing_duplicate_string --save_param 1 ...
    python run_bgf.py  --tasks missing_duplicate_string --save_param 1 ...

The script scans <--save_dir>/<folder>/<architecture>/<task>/<seedN_date>/ for
checkpoints (or evaluates a single run given with --run_dir) and writes
long_range_eval_s<step>.json into each run directory:
    {"accuracy_per_length": [acc at length 1, ..., acc at length max_length],
     "max_length": ..., "checkpoint_step": ..., "task": ..., "architecture": ...,
     "folder": ...}

Examples:
    python long_eval.py --save_dir ../run_experiments/results --cuda 0
    python long_eval.py --save_dir ../run_experiments/results --dry_run
    python long_eval.py --save_dir ../run_experiments/results --tasks missing_duplicate_string parity_check
    python long_eval.py --run_dir ../run_experiments/results/baseline-0.0005-100000/lstm/missing_duplicate_string/seed0_<date>
"""
import argparse
import glob
import json
import os
import random
import sys

parser = argparse.ArgumentParser(description='evaluate trained checkpoints on long sequence lengths')
parser.add_argument('--cuda', default=0, type=int, help='CUDA device ID')
parser.add_argument('--save_dir', default='../run_experiments/results', type=str,
                    help='results root used during training')
parser.add_argument('--run_dir', default=None, type=str, help='evaluate only this run directory')
parser.add_argument('--tasks', nargs='+', type=str, default=['missing_duplicate_string'],
                    help='tasks to evaluate (default: missing_duplicate_string, the paper figure task)')
parser.add_argument('--architectures', nargs='+', type=str, default=None, help='restrict to these architectures')
parser.add_argument('--folders', nargs='+', type=str, default=None, help='restrict to these method folders')
parser.add_argument('--max_length', default=1000, type=int, help='evaluate lengths 1..max_length')
parser.add_argument('--checkpoint_step', default=100_000, type=int,
                    help='training step of the checkpoint to load (= --training_steps of the run)')
parser.add_argument('--total_batch_size', default=512, type=int, help='test samples per length')
parser.add_argument('--sub_batch_size', default=128, type=int, help='sub-batch size to avoid memory overflow')
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--memory_cell_size', default=8, type=int)
parser.add_argument('--memory_size', default=40, type=int)
parser.add_argument('--is_autoregressive', default=False, type=bool)
parser.add_argument('--start', default=1, type=int, help='1-based index of the first run to evaluate')
parser.add_argument('--end', default=1_000_000_000, type=int, help='runs with index < end are evaluated')
parser.add_argument('--overwrite', action='store_true', help='re-evaluate runs that already have a result file')
parser.add_argument('--dry_run', action='store_true', help='list the runs instead of evaluating them')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

out_name = f'long_range_eval_s{args.checkpoint_step}.json'


def find_runs():
    """Returns (run_dir, folder, architecture, task) for every matching run."""
    if args.run_dir is not None:
        run_dir = os.path.normpath(os.path.abspath(args.run_dir))
        # run directories are laid out as <root>/<folder>/<architecture>/<task>/<seedN_date>
        seed_dir, task, architecture, folder = run_dir.split(os.sep)[-1:-5:-1]
        return [(run_dir, folder, architecture, task)]
    runs = []
    for ckpt in sorted(glob.glob(os.path.join(
            args.save_dir, '*', '*', '*', '*', 'params', f'param_s{args.checkpoint_step}.pt'))):
        run_dir = os.path.dirname(os.path.dirname(ckpt))
        seed_dir, task, architecture, folder = run_dir.split(os.sep)[-1:-5:-1]
        if task not in args.tasks:
            continue
        if args.architectures is not None and architecture not in args.architectures:
            continue
        if args.folders is not None and folder not in args.folders:
            continue
        runs.append((run_dir, folder, architecture, task))
    return runs


def build_model(task, architecture):
    """Rebuilds the model exactly as in training/main.py."""
    from BGF_AR.training import constants, utils
    import haiku as hk

    if architecture in ['tape_rnn']:
        architecture_params = {'hidden_size': args.hidden_size, 'memory_cell_size': args.memory_cell_size,
                               'memory_size': args.memory_size}
    elif architecture in ['stack_rnn', 'stack_lstm']:
        architecture_params = {'hidden_size': args.hidden_size, 'stack_cell_size': args.memory_cell_size}
    elif architecture in ['lstm', 'rnn']:
        architecture_params = {'hidden_size': args.hidden_size}
    elif 'transformer' in architecture:
        architecture_params = None
    else:
        raise NotImplementedError(f'unknown architecture {architecture!r}')

    computation_steps_mult = 0
    single_output = task.output_length(10) == 1
    if 'transformer' not in architecture:
        model = constants.MODEL_BUILDERS[architecture](
            output_size=task.output_size,
            return_all_outputs=True,
            **architecture_params)
    else:
        model = constants.MODEL_BUILDERS[architecture](
            output_size=task.output_size,
            return_all_outputs=True)
    if args.is_autoregressive:
        if 'transformer' not in architecture:
            model = utils.make_model_with_targets_as_input(model, computation_steps_mult)
        model = utils.add_sampling_to_autoregressive_model(model, single_output)
    else:
        model = utils.make_model_with_empty_targets(
            model, task, computation_steps_mult, single_output)
    return hk.transform(model)


def evaluate_lengths(model, params, task):
    """Accuracy at every length 1..--max_length (same protocol as the standard
    range evaluation: fixed seed 42, --total_batch_size samples per length)."""
    import haiku as hk
    import jax
    import jax.numpy as jnp
    import numpy as np
    import tqdm

    random.seed(42)
    np.random.seed(42)
    rng_seq = hk.PRNGSequence(42)

    if args.is_autoregressive:
        apply_fn = jax.jit(model.apply, static_argnames=('sample',))
    else:
        apply_fn = jax.jit(model.apply)

    def accuracy_fn(output, target):
        mask = task.accuracy_mask(target)
        return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)

    accuracies = []
    for length in tqdm.tqdm(range(1, args.max_length + 1)):
        # Clear the jit cache: every length compiles its own function, which
        # would otherwise accumulate over the 1..max_length sweep.
        apply_fn.clear_cache()
        sub_accuracies = []
        for _ in range(args.total_batch_size // args.sub_batch_size):
            batch = task.sample_batch(next(rng_seq), args.sub_batch_size, length)
            if args.is_autoregressive:
                outputs = apply_fn(params, next(rng_seq), batch['input'],
                                   jnp.empty_like(batch['output']), sample=True)
            else:
                outputs = apply_fn(params, next(rng_seq), batch['input'])
            sub_accuracies.append(float(np.mean(accuracy_fn(outputs, batch['output']))))
        accuracies.append(float(np.mean(sub_accuracies)))
    return accuracies


runs = find_runs()
total = len(runs)
print(f'Found {total} run(s) with a param_s{args.checkpoint_step}.pt checkpoint; '
      f'evaluating slice [{args.start}, {min(args.end, total + 1)}) at lengths 1..{args.max_length}')

for idx, (run_dir, folder, architecture, task_name) in enumerate(runs, start=1):
    if idx < args.start or idx >= args.end:
        continue
    out_path = os.path.join(run_dir, out_name)
    done = os.path.isfile(out_path)
    print(f'\n#### Eval {idx}/{total}  {folder}/{architecture}/{task_name}  ->  {out_path}'
          + ('  [exists, skipping]' if done and not args.overwrite else ''))
    if args.dry_run or (done and not args.overwrite):
        continue

    import pickle
    from BGF_AR.training import constants

    task = constants.TASK_BUILDERS[task_name]()
    model = build_model(task, architecture)
    with open(os.path.join(run_dir, 'params', f'param_s{args.checkpoint_step}.pt'), 'rb') as f:
        params = pickle.load(f)

    accuracies = evaluate_lengths(model, params, task)
    with open(out_path, 'w') as f:
        json.dump({'accuracy_per_length': accuracies,
                   'max_length': args.max_length,
                   'checkpoint_step': args.checkpoint_step,
                   'task': task_name,
                   'architecture': architecture,
                   'folder': folder}, f)
    mid = min(100, args.max_length)
    print(f'wrote {out_path}  (acc@{mid}={accuracies[mid - 1]:.3f}, '
          f'acc@{args.max_length}={accuracies[-1]:.3f})')
