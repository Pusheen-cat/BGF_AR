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

"""Training loop for base generalization experiments."""
import copy
import dataclasses
import functools
import random
from typing import Tuple, List, Callable, Mapping, Optional, Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import time

from BGF_AR.tasks import task as task_lib
from BGF_AR.training import curriculum as curriculum_lib
from BGF_AR.training import range_evaluation
from BGF_AR.training.update_fn import _update_parameters_valid_filter,_update_parameters_valid
from BGF_AR.training.update_fn import _update_parameters_valid_filter_balance_pre,_update_parameters_valid_filter_balance
from BGF_AR.training.update_fn import _update_parameters_valid_filter_add,_update_parameters_valid_filter_add_pre
from BGF_AR.training.update_fn import _update_parameters_ema_balance,_update_parameters_ema_balance_pre

import pickle
import os
from collections import deque

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
  sm: float

  save_param: int
  save_grad: int

  optim: str
  test_model: Optional[hk.Transformed] = None
  max_grad_norm: float = 1.
  is_autoregressive: bool = False

  compute_full_range_test: bool = False
  range_test_total_batch_size: int = 512
  range_test_sub_batch_size: int = 64
  max_range_test_length: int = 100

  accuracy_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None

  ## add by bong
  valid_seed: int = 0
  valid_length: int = 100

  ## add by jisoo
  filter_step: int = 100

  momentum: float = 0.9

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

    # optimizer = optax.chain( #TODO original
    #     optax.clip_by_global_norm(training_params.max_grad_norm),
    #     optax.adam(training_params.learning_rate))
    optimizer = optax.chain( # TODO rebuttal
        optax.clip_by_global_norm(training_params.max_grad_norm),
        optax.adam(training_params.learning_rate, training_params.momentum))
    #Todo adam --> adamw

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


    #early_stoppper = []
    grad_dq = [] #deque(maxlen=training_params.filter_step)
    # grad_dq = deque()  # Try
    #grad_dq = None  # Try
    filter_step = training_params.filter_step

    param_save_freq =  [0, 10, 100, 1000, 10000, 100000, 500000, 1000000]


    last_save_step = 1
    for save_step in param_save_freq:
        if save_step<=training_params.training_steps:
            last_save_step = save_step
    grad_save_window =  1000
    grad_save_freq = []
    valid_acc_tmp = []
    #calc__ = True
    save__=100000


    for freq in param_save_freq[2:]:
        grad_save_freq.extend(range(int(freq-grad_save_window), int(freq+grad_save_window)))

    runtime_pk = []
    start = time.time()
    for step in steps:
      if step%1000 == 0:
          runtime_pk.append(time.time()-start)
      # Randomness handled by either python.random or numpy.
      length = length_curriculum.sample_sequence_length(step)
      # Randomness handled by either jax, python.random or numpy.
      train_batch = task.sample_batch(
          next(rng_seq), length=length, batch_size=training_params.batch_size)
      valid_batch = task.sample_batch(
          next(rng_seq), length=self._training_params.valid_length, batch_size=training_params.range_test_sub_batch_size)
      if training_params.optim == 'ours_lpf':
          params, grad_dq, opt_state, (
              train_loss, train_metrics, train_accuracy, valid_loss, valid_accuracy, raw_grads,
              filtered_grads) = _update_parameters_valid_filter(
                  filter_step=filter_step,
                  grad_dq=grad_dq,
                  params=params,
                  rng_key=next(rng_seq),
                  batch=train_batch,
                  valid_batch=valid_batch, # added by bong
                  model_apply_fn=model.apply,
                  loss_fn=training_params.loss_fn,
                  accuracy_fn=training_params.accuracy_fn,
                  optimizer=optimizer,
                  opt_state=opt_state,
                  is_autoregressive=training_params.is_autoregressive,)

      elif training_params.optim == 'none':
          params, opt_state, (
              train_loss, train_metrics, train_accuracy, valid_loss, valid_accuracy, raw_grads,
              filtered_grads) = _update_parameters_valid(
              params=params,
              rng_key=next(rng_seq),
              batch=train_batch,
              valid_batch=valid_batch,  # added by bong
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
                  valid_batch=valid_batch,  # added by bong
                  model_apply_fn=model.apply,
                  loss_fn=training_params.loss_fn,
                  accuracy_fn=training_params.accuracy_fn,
                  optimizer=optimizer,
                  opt_state=opt_state,
                  is_autoregressive=training_params.is_autoregressive, )
              if step +1 == filter_step:
                  # check! whether the output is 1st dim concated 100 tensors
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
                  valid_batch=valid_batch,  # added by bong
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
                  valid_batch=valid_batch,  # added by bong
                  model_apply_fn=model.apply,
                  loss_fn=training_params.loss_fn,
                  accuracy_fn=training_params.accuracy_fn,
                  optimizer=optimizer,
                  opt_state=opt_state,
                  is_autoregressive=training_params.is_autoregressive, )
              if step +1 == filter_step:
                  # check! whether the output is 1st dim concated 100 tensors
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
                  valid_batch=valid_batch,  # added by bong
                  model_apply_fn=model.apply,
                  loss_fn=training_params.loss_fn,
                  accuracy_fn=training_params.accuracy_fn,
                  optimizer=optimizer,
                  opt_state=opt_state,
                  is_autoregressive=training_params.is_autoregressive, )


      elif training_params.optim == 'ours_ema':
          if step < filter_step:
              params, grad_dq, opt_state, (
                  train_loss, train_metrics, train_accuracy, valid_loss, valid_accuracy, raw_grads,
                  filtered_grads) = _update_parameters_ema_balance_pre(
                  weight_a=training_params.weight_a,
                  weight_b=training_params.weight_b,
                  sm=training_params.sm,
                  grad_dq=grad_dq,
                  params=params,
                  rng_key=next(rng_seq),
                  batch=train_batch,
                  valid_batch=valid_batch,  # added by bong
                  model_apply_fn=model.apply,
                  loss_fn=training_params.loss_fn,
                  accuracy_fn=training_params.accuracy_fn,
                  optimizer=optimizer,
                  opt_state=opt_state,
                  is_autoregressive=training_params.is_autoregressive, )

          else:
              params, grad_dq, opt_state, (
                  train_loss, train_metrics, train_accuracy, valid_loss, valid_accuracy, raw_grads,
                  filtered_grads) = _update_parameters_ema_balance(
                  weight_a=training_params.weight_a,
                  weight_b=training_params.weight_b,
                  sm=training_params.sm,
                  grad_dq=grad_dq,
                  params=params,
                  rng_key=next(rng_seq),
                  batch=train_batch,
                  valid_batch=valid_batch,  # added by bong
                  model_apply_fn=model.apply,
                  loss_fn=training_params.loss_fn,
                  accuracy_fn=training_params.accuracy_fn,
                  optimizer=optimizer,
                  opt_state=opt_state,
                  is_autoregressive=training_params.is_autoregressive, )

      else:
          raise NotImplementedError



      self._params, self._step = params, step

      param_dir = os.path.join(self._save_dir, "params")
      raw_grad_dir = os.path.join(self._save_dir, "raw_grads")

      os.makedirs(param_dir, exist_ok=True)
      os.makedirs(raw_grad_dir, exist_ok=True)

      if training_params.save_param == 1:
          if step in param_save_freq:
              with open(os.path.join(param_dir, f"param_s{step}.pt"), "wb") as f:
                  pickle.dump(params, f)

              # Save for further training
              if step == last_save_step:
                  with open(os.path.join(param_dir, f"opt_state_s{step}.pt"), "wb") as f:
                      pickle.dump(opt_state, f)

                  if training_params.optim.startswith('ours'):
                      with open(os.path.join(param_dir, f"grad_dq_s{step}.pt"), "wb") as f:
                          pickle.dump(grad_dq, f)

      elif training_params.save_param == 2:
          if step<=10:
              with open(os.path.join(param_dir, f"param_s{step}.pt"), "wb") as f:
                  pickle.dump(params, f)
          elif step<=100:
              if step%3==0:
                  with open(os.path.join(param_dir, f"param_s{step}.pt"), "wb") as f:
                      pickle.dump(params, f)
          elif step<=1000:
              if step%10==0:
                  with open(os.path.join(param_dir, f"param_s{step}.pt"), "wb") as f:
                      pickle.dump(params, f)
          elif step<=10000:
              if step%30==0:
                  with open(os.path.join(param_dir, f"param_s{step}.pt"), "wb") as f:
                      pickle.dump(params, f)
          elif step<=100000:
              if step%100==0:
                  with open(os.path.join(param_dir, f"param_s{step}.pt"), "wb") as f:
                      pickle.dump(params, f)

      elif training_params.save_param == 3:
          if step <= 10000:
              if step % 10 == 0:
                  with open(os.path.join(param_dir, f"param_s{step}.pt"), "wb") as f:
                      pickle.dump(params, f)
          elif step <= 50000:
              if step % 20 == 0:
                  with open(os.path.join(param_dir, f"param_s{step}.pt"), "wb") as f:
                      pickle.dump(params, f)
          elif step <= 100000:
              if step % 100 == 0:
                  with open(os.path.join(param_dir, f"param_s{step}.pt"), "wb") as f:
                      pickle.dump(params, f)

          # # Save for further training
          # if step == training_params.training_steps:
          #     with open(os.path.join(param_dir, f"opt_state_s{step}.pt"), "wb") as f:
          #         pickle.dump(opt_state, f)
          #
          #     if training_params.optim.startswith('ours'):
          #         with open(os.path.join(param_dir, f"grad_dq_s{step}.pt"), "wb") as f:
          #             pickle.dump(grad_dq, f)
      elif training_params.save_param == 10:
          with open(os.path.join(param_dir, f"param_s{step}.pt"), "wb") as f:
              pickle.dump(params, f)


      ### PREV ### : Save gradient
      # if step in grad_save_freq:
      #     with open(os.path.join(raw_grad_dir, f"rawgrad_s{step}.pt"), "wb") as f:
      #         pickle.dump(raw_grads, f)
      #
      #     if filtered_grads is not None:
      #         filter_grad_dir = os.path.join(self._save_dir, "filtered_grads")
      #         os.makedirs(filter_grad_dir, exist_ok=True)
      #         with open(os.path.join(filter_grad_dir, f"filtergrad_s{step}.pt"), "wb") as f:
      #             pickle.dump(filtered_grads, f)

      ## NOW 0514 ### : Save gradient (Untill reach 95% acc)
      if training_params.save_grad==1:
          if calc__ == True:
              valid_acc_tmp.append(valid_accuracy.item())
              if len(valid_acc_tmp)>50:
                  valid_acc_tmp = valid_acc_tmp[-50:]
                  avg_val_acc = sum(valid_acc_tmp)/len(valid_acc_tmp)
                  if avg_val_acc > 0.95:
                      first_step = copy.deepcopy(step)
                      if first_step >5000:
                          save__ = max(3000, int(first_step*1.5))
                      else:
                          save__ = int(first_step * 1.2)
                      calc__ = False

          if step<save__:
              with open(os.path.join(raw_grad_dir, f"rawgrad_s{step}.pt"), "wb") as f:
                  pickle.dump(raw_grads, f)

              if filtered_grads is not None:
                  filter_grad_dir = os.path.join(self._save_dir, "filtered_grads")
                  os.makedirs(filter_grad_dir, exist_ok=True)
                  with open(os.path.join(filter_grad_dir, f"filtergrad_s{step}.pt"), "wb") as f:
                      pickle.dump(filtered_grads, f)

      elif training_params.save_grad==10:
          with open(os.path.join(raw_grad_dir, f"rawgrad_s{step}.pt"), "wb") as f:
              pickle.dump(raw_grads, f)

          if filtered_grads is not None:
              filter_grad_dir = os.path.join(self._save_dir, "filtered_grads")
              os.makedirs(filter_grad_dir, exist_ok=True)
              with open(os.path.join(filter_grad_dir, f"filtergrad_s{step}.pt"), "wb") as f:
                  pickle.dump(filtered_grads, f)

      #print(f'step: {step}, train loss: {train_loss}, train acc: {train_accuracy}, valid loss: {valid_loss}, valid acc: {valid_accuracy}')
      results[step] = {'t_l':train_loss.item(), 't_a':train_accuracy.item(), 'v_l':valid_loss.item(), 'v_a':valid_accuracy.item()}

      # We need to access this private attribute since the default reserve size
      # can not be edited yet.
      if not rng_seq._subkeys:  # pylint: disable=protected-access
        rng_seq.reserve(rngs_reserve)

    eval_results = None
    results['runtime_per_k'] = runtime_pk
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
      eval_results = range_evaluation.range_evaluation(eval_params, use_tqdm=False)

    return results, eval_results, params
