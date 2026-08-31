# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Training loop for the input-length confounding toy experiment.

Training batches are rejection-sampled (spu_len_input.add_spruious) so that,
with probability spu_prob, the label matches a length-dependent pseudo-label
(input length binned into output_size classes) - i.e. sequence length and
label are spuriously correlated at strength spu_prob.  The inputs themselves
are unmodified, so validation batches and the final length-range evaluation
use the STANDARD samplers, where the correlation is absent.  This experiment
only supports single-output tasks (task.output_length == 1).
"""

import dataclasses
import random
from typing import Tuple, List, Callable, Mapping, Optional, Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

from BGF_AR.tasks import task as task_lib
from BGF_AR.training import curriculum as curriculum_lib
from BGF_AR.training import range_evaluation as range_evaluation
from BGF_AR.run_toy_experiment.spu_len_input import *
from BGF_AR.training.update_fn import _update_parameters_valid
from BGF_AR.training.update_fn import _update_parameters_valid_filter_balance_pre,_update_parameters_valid_filter_balance
from BGF_AR.training.update_fn import _update_parameters_valid_filter_add,_update_parameters_valid_filter_add_pre

import pickle
import os

_LossMetrics = Optional[Mapping[str, jnp.ndarray]]
_LossFn = Callable[[chex.Array, chex.Array], Tuple[float, _LossMetrics]]
_AccuracyFn = Callable[[chex.Array, chex.Array], float]
_ModelApplyFn = Callable[..., chex.Array]
_MAX_RNGS_RESERVE = 50000

@dataclasses.dataclass
class ClassicTrainingParams:
  """Parameters needed to train classical architectures."""
  seed: int  # Used to sample during forward pass (e.g. from final logits).
  model_init_seed: int  # Used to initialize model parameters.
  training_steps: int
  log_frequency: int

  task: task_lib.GeneralizationTask
  length_curriculum: curriculum_lib.Curriculum
  batch_size: int

  model: hk.Transformed
  loss_fn: Callable[[jnp.ndarray, jnp.ndarray], Tuple[float, _LossMetrics]]
  learning_rate: float
  weight_a:float
  weight_b:float
  optim: str

  spu_last:bool
  spu_prob:float

  test_model: Optional[hk.Transformed] = None
  max_grad_norm: float = 1.
  is_autoregressive: bool = False

  compute_full_range_test: bool = False
  range_test_total_batch_size: int = 512
  range_test_sub_batch_size: int = 64
  max_range_test_length: int = 100

  accuracy_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                 jnp.ndarray]] = None

  valid_seed: int = 0
  valid_length: int = 100

  # queue size (lambda) of the queue-based BGF / ADD gradient queue
  filter_step: int = 100

class TrainingWorker:
  """Training worker."""

  def __init__(self,
               training_params: ClassicTrainingParams,
               use_tqdm: bool,
               save_dir: str):
    """Initializes the worker.

    Args:
      training_params: The training parameters.
      use_tqdm: Whether to add a progress bar to stdout.
    """
    self._training_params = training_params
    self._use_tqdm = use_tqdm
    self._save_dir = save_dir

  def run(
      self
  ) -> Tuple[List[Mapping[str, Any]], Optional[List[Mapping[str, Any]]],
             chex.ArrayTree]:
    """Trains the model with the provided config.

    Returns:
      Results (various training and validation metrics), module parameters
      and router parameters.
    """
    training_params = self._training_params
    rngs_reserve = min(_MAX_RNGS_RESERVE, training_params.training_steps)

    random.seed(training_params.seed)
    np.random.seed(training_params.seed)
    rng_seq = hk.PRNGSequence(training_params.seed)
    rng_seq.reserve(rngs_reserve)

    results = []
    model = training_params.model
    task = training_params.task
    length_curriculum = training_params.length_curriculum

    optimizer = optax.chain(
        optax.clip_by_global_norm(training_params.max_grad_norm),
        optax.adam(training_params.learning_rate))


    dummy_batch = task.sample_batch(
        next(rng_seq), length=10, batch_size=training_params.batch_size)
    model_init_rng_key = jax.random.PRNGKey(training_params.model_init_seed)

    if training_params.is_autoregressive:
      params = model.init(
          model_init_rng_key,
          dummy_batch["input"],
          dummy_batch["output"],
          sample=False)
    else:
      params = model.init(model_init_rng_key, dummy_batch["input"])

    opt_state = optimizer.init(params)
    self._params, self._step = params, 0
    results = {}
    steps = range(training_params.training_steps + 1)
    if self._use_tqdm:
      steps = tqdm.tqdm(steps)


    grad_dq = []
    filter_step = training_params.filter_step

    param_save_freq =  [10, 100, 1000, 10000, 100000, 500000, 1000000]

    param_dir = os.path.join(self._save_dir, "params")
    os.makedirs(param_dir, exist_ok=True)

    for step in steps:
      # Randomness handled by either python.random or numpy.
      length = length_curriculum.sample_sequence_length(step)
      # Randomness handled by either jax, python.random or numpy.

      # Training batch with the length<->label correlation injected.
      train_batch = add_spruious(task, rng_seq, length, training_params.batch_size, training_params.spu_prob)

      # Validation batch from the standard sampler (no correlation).
      valid_batch = task.sample_batch(
          next(rng_seq), length=self._training_params.valid_length, batch_size=training_params.range_test_sub_batch_size)

      if training_params.optim == 'none':
          params, opt_state, (
              train_loss, train_metrics, train_accuracy, valid_loss, valid_accuracy, raw_grads, filtered_grads) = _update_parameters_valid(
              params=params,
              rng_key=next(rng_seq),
              batch=train_batch,
              valid_batch=valid_batch,
              model_apply_fn=model.apply,
              loss_fn=training_params.loss_fn,
              accuracy_fn=training_params.accuracy_fn,
              optimizer=optimizer,
              opt_state=opt_state,
              is_autoregressive=training_params.is_autoregressive)

      elif training_params.optim == 'ours_balance':
          if step < filter_step:
              params, grad_dq, opt_state, (
                  train_loss, train_metrics, train_accuracy, valid_loss, valid_accuracy, raw_grads,
                  filtered_grads) = _update_parameters_valid_filter_balance_pre(
                  weight_a=training_params.weight_a,
                  weight_b=training_params.weight_b,
                  grad_dq=grad_dq,
                  params=params,
                  rng_key=next(rng_seq),
                  batch=train_batch,
                  valid_batch=valid_batch,
                  model_apply_fn=model.apply,
                  loss_fn=training_params.loss_fn,
                  accuracy_fn=training_params.accuracy_fn,
                  optimizer=optimizer,
                  opt_state=opt_state,
                  is_autoregressive=training_params.is_autoregressive, )
              if step +1 == filter_step:
                  grad_dq = jax.tree_util.tree_map(lambda *args: jnp.stack(args, axis=0),*grad_dq)

          else:
              params, grad_dq, opt_state, (
                  train_loss, train_metrics, train_accuracy, valid_loss, valid_accuracy, raw_grads,
                  filtered_grads) = _update_parameters_valid_filter_balance(
                  weight_a=training_params.weight_a,
                  weight_b=training_params.weight_b,
                  grad_dq=grad_dq,
                  params=params,
                  rng_key=next(rng_seq),
                  batch=train_batch,
                  valid_batch=valid_batch,
                  model_apply_fn=model.apply,
                  loss_fn=training_params.loss_fn,
                  accuracy_fn=training_params.accuracy_fn,
                  optimizer=optimizer,
                  opt_state=opt_state,
                  is_autoregressive=training_params.is_autoregressive, )


      elif training_params.optim == 'ours_add':
          if step < filter_step:
              params, grad_dq, opt_state, (
                  train_loss, train_metrics, train_accuracy, valid_loss, valid_accuracy, raw_grads,
                  filtered_grads) = _update_parameters_valid_filter_add_pre(
                  weight_a=training_params.weight_a,
                  weight_b=training_params.weight_b,
                  grad_dq=grad_dq,
                  params=params,
                  rng_key=next(rng_seq),
                  batch=train_batch,
                  valid_batch=valid_batch,
                  model_apply_fn=model.apply,
                  loss_fn=training_params.loss_fn,
                  accuracy_fn=training_params.accuracy_fn,
                  optimizer=optimizer,
                  opt_state=opt_state,
                  is_autoregressive=training_params.is_autoregressive, )
              if step +1 == filter_step:
                  grad_dq = jax.tree_util.tree_map(lambda *args: jnp.stack(args, axis=0),*grad_dq)

          else:
              params, grad_dq, opt_state, (
                  train_loss, train_metrics, train_accuracy, valid_loss, valid_accuracy, raw_grads,
                  filtered_grads) = _update_parameters_valid_filter_add(
                  weight_a=training_params.weight_a,
                  weight_b=training_params.weight_b,
                  grad_dq=grad_dq,
                  params=params,
                  rng_key=next(rng_seq),
                  batch=train_batch,
                  valid_batch=valid_batch,
                  model_apply_fn=model.apply,
                  loss_fn=training_params.loss_fn,
                  accuracy_fn=training_params.accuracy_fn,
                  optimizer=optimizer,
                  opt_state=opt_state,
                  is_autoregressive=training_params.is_autoregressive, )

      else:
          # The toy experiments of the paper use Adam ('none'), queue-based
          # BGF ('ours_balance') and ADD ('ours_add') only.
          raise NotImplementedError(f'optim {training_params.optim!r} is not supported '
                                    f'in the toy experiments')



      self._params, self._step = params, step

      if step in param_save_freq:
          with open(os.path.join(param_dir, f"param_s{step}_spu.pt"), "wb") as f:
              pickle.dump(params, f)

      results[step] = {'t_l':train_loss.item(), 't_a':train_accuracy.item(), 'v_l':valid_loss.item(), 'v_a':valid_accuracy.item()}

      # We need to access this private attribute since the default reserve size
      # can not be edited yet.
      if not rng_seq._subkeys:  # pylint: disable=protected-access
        rng_seq.reserve(rngs_reserve)

    eval_results = None
    if training_params.compute_full_range_test:
      eval_params = range_evaluation.EvaluationParams(
          model=training_params.test_model or model,
          params=params,
          accuracy_fn=training_params.accuracy_fn,
          sample_batch=task.sample_batch,
          max_test_length=training_params.max_range_test_length,
          total_batch_size=training_params.range_test_total_batch_size,
          sub_batch_size=training_params.range_test_sub_batch_size,
          is_autoregressive=training_params.is_autoregressive,
      )
      eval_results = range_evaluation.range_evaluation(
          eval_params, use_tqdm=False)

    return results, eval_results, params
