"""Loss-landscape sharpness evaluation of trained checkpoints.

For every finished run (trained with ``--save_param 1`` so that
``<run_dir>/params/param_s<step>.pt`` exists) this script rebuilds the model,
loads the final checkpoint and computes the paper's three sharpness measures
(see flat_measures.py):

    low_pass  -  low-pass-filter-based sharpness E_eps[L(theta+eps)]
    fim       -  Fisher-information-based sharpness theta^T H theta
    shannon_entropy - Shannon entropy of the output distribution

Each measure is evaluated on the training distribution ('train') and on every
--val_len sequence length, and written to ``<run_dir>/sharpness.json``
together with the checkpoint's loss/accuracy.  Aggregate the resulting files
with ``analyze_sharpness.py``.

Only the (non-autoregressive) recurrent architectures used by the paper's
sharpness analysis are supported: rnn, lstm, stack_rnn, tape_rnn.

Prerequisite (Adam + BGF runs with checkpointing):
    cd ../run_experiments
    python run_adam.py --cuda 0 --architectures rnn lstm stack_rnn tape_rnn --save_param 1
    python run_bgf.py  --cuda 0 --architectures rnn lstm stack_rnn tape_rnn --save_param 1

Examples:
    python sharp_eval.py --save_dir ../run_experiments/results --cuda 0
    python sharp_eval.py --save_dir ../run_experiments/results --dry_run
    python sharp_eval.py --save_dir ../run_experiments/results --start 1 --end 91 --cuda 0
"""
import argparse
import copy
import glob
import json
import os
import pickle
import random
import sys

parser = argparse.ArgumentParser(description='sharpness measures of trained checkpoints')
parser.add_argument('--cuda', default=0, type=int, help='CUDA device ID')
parser.add_argument('--save_dir', default='../run_experiments/results', type=str,
                    help='results root used during training')
parser.add_argument('--run_dir', default=None, type=str, help='evaluate only this run directory')
parser.add_argument('--folders', nargs='+', type=str, default=None, help='restrict to these method folders')
parser.add_argument('--tasks', nargs='+', type=str, default=None, help='restrict to these tasks')
parser.add_argument('--architectures', nargs='+', type=str,
                    default=['rnn', 'lstm', 'stack_rnn', 'tape_rnn'],
                    help='architectures to evaluate (recurrent models only)')
parser.add_argument('--checkpoint_step', default=100_000, type=int,
                    help='training step of the checkpoint to load (= --training_steps of the run)')
parser.add_argument('--sharp_batch', default=128, type=int, help='batch size per Monte-Carlo step')
parser.add_argument('--sharp_steps', default=100, type=int, help='Monte-Carlo steps per loss estimate')
parser.add_argument('--mcmc_itr', default=100, type=int, help='parameter samples of the low-pass measure')
parser.add_argument('--sigma', default=0.01, type=float, help='perturbation scale of the low-pass measure')
parser.add_argument('--val_len', nargs='+', type=int, default=[100],
                    help='validation sequence lengths to evaluate at (besides the training distribution)')
parser.add_argument('--start', default=1, type=int, help='1-based index of the first run to evaluate')
parser.add_argument('--end', default=1_000_000_000, type=int, help='runs with index < end are evaluated')
parser.add_argument('--overwrite', action='store_true', help='re-evaluate runs that already have a result file')
parser.add_argument('--dry_run', action='store_true', help='list the runs instead of evaluating them')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import flat_measures  # noqa: E402  (sibling module; jax-free until called)

OUT_NAME = 'sharpness.json'


def find_runs():
    """(run_dir, folder, architecture, task) for every matching finished run."""
    if args.run_dir is not None:
        run_dir = os.path.normpath(os.path.abspath(args.run_dir))
        seed_dir, task, architecture, folder = run_dir.split(os.sep)[-1:-5:-1]
        return [(run_dir, folder, architecture, task)]
    runs = []
    for ckpt in sorted(glob.glob(os.path.join(
            args.save_dir, '*', '*', '*', '*', 'params', f'param_s{args.checkpoint_step}.pt'))):
        run_dir = os.path.dirname(os.path.dirname(ckpt))
        seed_dir, task, architecture, folder = run_dir.split(os.sep)[-1:-5:-1]
        if args.tasks is not None and task not in args.tasks:
            continue
        if architecture not in args.architectures:
            continue
        if args.folders is not None and folder not in args.folders:
            continue
        runs.append((run_dir, folder, architecture, task))
    return runs


def build(setting):
    """Rebuilds (task, curriculum, model) from a run's saved 'setting' dict."""
    from BGF_AR.training import constants, utils
    from BGF_AR.training import curriculum as curriculum_lib
    import haiku as hk

    if setting.get('is_autoregressive'):
        raise NotImplementedError('sharpness evaluation supports the non-autoregressive '
                                  'recurrent models only')
    architecture = setting['architecture']
    hidden_size = setting.get('hidden_size', 256)
    if architecture == 'tape_rnn':
        architecture_params = {'hidden_size': hidden_size,
                               'memory_cell_size': setting.get('memory_cell_size', 8),
                               'memory_size': setting.get('memory_size', 40)}
    elif architecture in ['stack_rnn', 'stack_lstm']:
        architecture_params = {'hidden_size': hidden_size,
                               'stack_cell_size': setting.get('memory_cell_size', 8)}
    elif architecture in ['lstm', 'rnn']:
        architecture_params = {'hidden_size': hidden_size}
    else:
        raise NotImplementedError(f'unsupported architecture {architecture!r}')

    task = constants.TASK_BUILDERS[setting['task']]()
    curriculum = curriculum_lib.UniformCurriculum(
        values=list(range(1, setting.get('sequence_length', 40) + 1)))
    model = constants.MODEL_BUILDERS[architecture](
        output_size=task.output_size, return_all_outputs=True, **architecture_params)
    model = utils.make_model_with_empty_targets(
        model, task, computation_steps_mult=0, single_output=task.output_length(10) == 1)
    return task, curriculum, hk.transform(model)


class ModelForSharp:
    """Trained model + task wrapper the sharpness measures operate on.

    Provides Monte-Carlo loss evaluation (``compute_loss``) and a
    Hessian-vector product over the flattened parameter vector (``hvp``);
    batches are drawn from a per-call fresh PRNG sequence seeded with the
    run's own seed, so every measure - and the matched Adam/BGF runs - sees
    the same data.
    """

    def __init__(self, model, param_trained, setting, task, curriculum):
        self.model = model
        self.param_trained = param_trained
        self.setting = setting
        self.task = task
        self.curriculum = curriculum
        self.sharp_steps = args.sharp_steps
        self._seed = setting['seed']

        # loss / accuracy exactly as in training/main.py
        import jax.numpy as jnp

        def loss_fn(output, target):
            return jnp.mean(jnp.sum(task.pointwise_loss_fn(output, target), axis=-1)), {}

        def accuracy_fn(output, target):
            mask = task.accuracy_mask(target)
            return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)

        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn

        # flattened-parameter bookkeeping for the Hessian-vector product
        self.dim = 0
        self.list_to_dict = []
        for key, layer in param_trained.items():
            for key_, w_b in layer.items():
                self.dim += w_b.size
                self.list_to_dict.append([key, key_, self.dim - w_b.size, self.dim, w_b.shape])

        import jax

        @jax.jit
        def _eval(param, rng, inp, out):
            output = model.apply(param, rng, inp)
            return loss_fn(output, out)[0], accuracy_fn(output, out)
        self._eval = _eval

        @jax.jit
        def _hvp_single(flat_param, vec, rng, inp, out):
            def loss_of_flat(flat):
                pdict = {}
                for key, key_, lo, hi, shape in self.list_to_dict:
                    pdict.setdefault(key, {})[key_] = flat[lo:hi].reshape(shape)
                output = model.apply(pdict, rng, inp)
                return loss_fn(output, out)[0]
            return jax.grad(lambda f: jnp.vdot(jax.grad(loss_of_flat)(f), vec))(flat_param)
        self._hvp_single = _hvp_single

    def fresh_rng_seq(self):
        import haiku as hk
        rng_seq = hk.PRNGSequence(self._seed)
        rng_seq.reserve(min(10_000, 4 * self.sharp_steps))
        return rng_seq

    def sample_batch(self, rng_seq, step, length=None):
        len_ = self.curriculum.sample_sequence_length(step) if length is None else length
        return self.task.sample_batch(next(rng_seq), length=len_, batch_size=args.sharp_batch)

    def compute_loss(self, param=None, length=None):
        """(mean loss, mean accuracy) over sharp_steps Monte-Carlo batches."""
        import numpy as np
        if param is None:
            param = self.param_trained
        rng_seq = self.fresh_rng_seq()
        losses, accs = [], []
        for step in range(self.sharp_steps):
            batch = self.sample_batch(rng_seq, step, length)
            loss, acc = self._eval(param, next(rng_seq), batch['input'], batch['output'])
            losses.append(float(loss))
            accs.append(float(acc))
        return float(np.mean(losses)), float(np.mean(accs))

    def hvp(self, vec, length=None):
        """Mean H*vec over sharp_steps batches; H = loss Hessian w.r.t. the
        flattened parameter vector."""
        import jax.numpy as jnp
        import numpy as np
        array_param = []
        for _, layer in self.param_trained.items():
            for _, w_b in layer.items():
                array_param.append(np.array(copy.deepcopy(w_b)).flatten())
        flat_param = jnp.array(np.concatenate(array_param))
        vec = jnp.array(vec)
        rng_seq = self.fresh_rng_seq()
        out = None
        for step in range(self.sharp_steps):
            batch = self.sample_batch(rng_seq, step, length)
            hv = self._hvp_single(flat_param, vec, next(rng_seq),
                                  batch['input'], batch['output'])
            out = hv if out is None else out + hv
        return np.asarray(out / self.sharp_steps)


def evaluate_run(run_dir):
    import numpy as np

    with open(os.path.join(run_dir, 'logs')) as f:
        setting = json.load(f)['setting']
    with open(os.path.join(run_dir, 'params', f'param_s{args.checkpoint_step}.pt'), 'rb') as f:
        param_trained = pickle.load(f)

    random.seed(setting['seed'])
    np.random.seed(setting['seed'])

    task, curriculum, model = build(setting)
    model_func = ModelForSharp(model, param_trained, setting, task, curriculum)

    mtr = {'config': dict(checkpoint_step=args.checkpoint_step, sharp_batch=args.sharp_batch,
                          sharp_steps=args.sharp_steps, mcmc_itr=args.mcmc_itr,
                          sigma=args.sigma, val_len=args.val_len),
           'optim': setting.get('optim', 'none'), 'seed': setting['seed']}

    mtr['train_loss'], mtr['train_acc'] = model_func.compute_loss()
    mtr['val_loss'], mtr['val_acc'] = {}, {}
    for len_ in args.val_len:
        mtr['val_loss'][len_], mtr['val_acc'][len_] = model_func.compute_loss(length=len_)

    mtr['shannon_entropy'] = {'train': flat_measures.shannon_entropy(model_func),
                              'val': {len_: flat_measures.shannon_entropy(model_func, length=len_)
                                      for len_ in args.val_len}}
    print('  done - shannon_entropy')

    mtr['fim'] = {'train': flat_measures.fim(model_func),
                  'val': {len_: flat_measures.fim(model_func, length=len_)
                          for len_ in args.val_len}}
    print('  done - fim')

    np.random.seed(setting['seed'])          # reproducible low-pass perturbations
    mtr['low_pass'] = {'train': flat_measures.low_pass(model_func, args.sigma, args.mcmc_itr),
                       'val': {len_: flat_measures.low_pass(model_func, args.sigma,
                                                            args.mcmc_itr, length=len_)
                               for len_ in args.val_len}}
    print('  done - low_pass')
    return mtr


runs = find_runs()
total = len(runs)
print(f'Found {total} run(s) with a param_s{args.checkpoint_step}.pt checkpoint; '
      f'evaluating slice [{args.start}, {min(args.end, total + 1)})')

for idx, (run_dir, folder, architecture, task_name) in enumerate(runs, start=1):
    if idx < args.start or idx >= args.end:
        continue
    out_path = os.path.join(run_dir, OUT_NAME)
    done = os.path.isfile(out_path)
    print(f'\n#### Eval {idx}/{total}  {folder}/{architecture}/{task_name}  ->  {out_path}'
          + ('  [exists, skipping]' if done and not args.overwrite else ''))
    if args.dry_run or (done and not args.overwrite):
        continue
    mtr = evaluate_run(run_dir)
    with open(out_path, 'w') as f:
        json.dump(mtr, f)
    print(f'wrote {out_path}')
