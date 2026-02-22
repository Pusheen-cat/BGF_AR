"""Training loop for base generalization experiments."""

import dataclasses
import functools
from typing import Tuple, List, Callable, Mapping, Optional, Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from BGF_AR.tasks import task as task_lib


_LossMetrics = Optional[Mapping[str, jnp.ndarray]]
_LossFn = Callable[[chex.Array, chex.Array], Tuple[float, _LossMetrics]]
_AccuracyFn = Callable[[chex.Array, chex.Array], float]
_ModelApplyFn = Callable[..., chex.Array]
_MAX_RNGS_RESERVE = 50000

'''
def dq_mean(dict_list):
    mean_dq = {}
    for outer_key in dict_list[0].keys():
        inner_dict = {}
        for inner_key in dict_list[0][outer_key].keys():
            # print([d[outer_key][inner_key] for d in dict_list])
            inner_dict[inner_key] = jnp.mean(jnp.array([d[outer_key][inner_key] for d in dict_list]), axis=0)
        mean_dq[outer_key] = inner_dict #jnp.mean(jnp.array([d[outer_key][inner_key] for d in dict_list]), axis=0)
    return mean_dq

def adj_dict_mean(dic_a, dic_b, a=0.5, b=0.5): # a: raw grad
    mean_dict = {}
    for outer_key in dic_a.keys():
        inner_dict = {}
        for inner_key in dic_a[outer_key].keys():
            # print([d[outer_key][inner_key] for d in dict_list])
            tmpa = dic_a[outer_key][inner_key]
            tmpb = dic_b[outer_key][inner_key]
            adj = jnp.linalg.norm(tmpa) / jnp.linalg.norm(tmpb)
            inner_dict[inner_key] = a*tmpa+adj*b*tmpb
        mean_dict[outer_key] = inner_dict #jnp.mean(jnp.array([d[outer_key][inner_key] for d in dict_list]), axis=0)
    return mean_dict

def add_dict_mean(dic_a, dic_b, a=1.0, b=1.0): # a: raw grad
    mean_dict = {}
    for outer_key in dic_a.keys():
        inner_dict = {}
        for inner_key in dic_a[outer_key].keys():
            # print([d[outer_key][inner_key] for d in dict_list])
            tmpa = dic_a[outer_key][inner_key]
            tmpb = dic_b[outer_key][inner_key]
            inner_dict[inner_key] = a*tmpa+b*tmpb
        mean_dict[outer_key] = inner_dict #jnp.mean(jnp.array([d[outer_key][inner_key] for d in dict_list]), axis=0)
    return mean_dict
'''
def dq_mean(dict_list, length = 100):
    return jax.tree_util.tree_map(lambda *args: sum(args)/length, *dict_list)

def adj_dict_mean(dic_a, dic_b, a=0.5, b=0.5): # a: raw grad
    return jax.tree_util.tree_map(lambda x, y: a*x + (jnp.linalg.norm(x) / jnp.linalg.norm(y))*b*y, dic_a, dic_b)

def add_dict_mean(dic_a, dic_b, a=1.0, b=1.0): # a: raw grad
    return jax.tree_util.tree_map(lambda x, y: (a * x) + (b * y), dic_a, dic_b)

def ema_add(new, old, sm = 0.98): # a: raw grad
    return jax.tree_util.tree_map(lambda x, y: ((1-sm) * x) + (sm * y), new, old)


### New
def dq_mean_whole(deque): #check! whether the output is actually mean of 100 grads
    return jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), deque)

def dq_append_whole(deque, new): #check! whether it pops oldest grad and append the new one
    return jax.tree_util.tree_map(lambda x, y: jnp.r_[x[1:],[y]], deque, new)


def _apply_loss_and_metrics_fn(
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    model_apply_fn: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    is_autoregressive: bool = False,
) -> Tuple[float, Tuple[_LossMetrics, float]]:
  """Computes the model output and applies the loss function.

  Depending on whether a model is autoregressive or not, it will have a
  different number of input parameters (i.e., autoregressive models also require
  the targets as an input).

  Args:
    params: The model parameters.
    rng_key: The prng key to use for random number generation.
    batch: The data (consists of both inputs and outputs).
    model_apply_fn: The model function that converts inputs into outputs.
    loss_fn: A function that computes the loss for a batch of logits and labels.
    accuracy_fn: A function that computes the accuracy for a batch of logits and
      labels.
    is_autoregressive: Whether the model is autoregressive or not.

  Returns:
    The loss of the model for the batch of data, extra loss metrics and the
    accuracy, if accuracy_fn is not None.
  """
  if is_autoregressive:
    outputs = model_apply_fn(
        params, rng_key, batch["input"], batch["output"], sample=False)
  else:
    outputs = model_apply_fn(params, rng_key, batch["input"])
  loss, loss_metrics = loss_fn(outputs, batch["output"])
  if accuracy_fn is not None:
    accuracy = accuracy_fn(outputs, batch["output"])
  else:
    accuracy = None
  return loss, (loss_metrics, accuracy)


@functools.partial(
    jax.jit,
    static_argnames=(
        "filter_step",
        "model_apply_fn",
        "loss_fn",
        "accuracy_fn",
        "optimizer",
        "is_autoregressive",
    ),
)
def _update_parameters_valid_filter(
    filter_step: int,
    grad_dq,
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    valid_batch: task_lib.Batch,
    model_apply_fn: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    is_autoregressive: bool = False,
) -> Tuple[hk.Params, optax.OptState, Tuple[float, _LossMetrics, float]]:
  """Applies a single SGD update step to the model parameters.

  Args:
    update_step: current training step
    filter_step: time step to apply low-pass filter
    grad_dq: python deque containing gradients
    params: The model parameters.
    rng_key: The prng key to use for random number generation.
    batch: The data (consists of both inputs and outputs).
    model_apply_fn: The model function that converts inputs into outputs.
    loss_fn: A function that computes the loss for a batch of logits and labels.
    accuracy_fn: A function that computes the accuracy for a batch of logits and
      labels.
    optimizer: The optimizer that computes the updates from the gradients of the
      `loss_fn` with respect to the `params` and the previous `opt_state`.
    opt_state: The optimizer state, e.g., momentum for each variable when using
      Adam.
    is_autoregressive: Whether the model is autoregressive or not.

  Returns:
    The updated parameters, the new optimizer state, and the loss, loss metrics
    and accuracy.
  """
  ##### Here model runs validation first (before parameter update) start
  if is_autoregressive:
      outputs = model_apply_fn(
          params,
          rng_key,
          valid_batch['input'],
          jnp.empty_like(valid_batch['output']),
          sample=True)
  else:
      outputs = model_apply_fn(params, rng_key, valid_batch['input'])

  valid_loss, valid_loss_metrics = loss_fn(outputs, valid_batch["output"])
  if accuracy_fn is not None:
      valid_accuracy = accuracy_fn(outputs, valid_batch["output"])
  else:
      valid_accuracy = None
  ##### Here model runs validation first (before parameter update) end

  (loss, (metrics, accuracy)), grads = jax.value_and_grad(
      _apply_loss_and_metrics_fn,
      has_aux=True)(params, rng_key, batch, model_apply_fn, loss_fn,
                    accuracy_fn, is_autoregressive)
  grad_dq.append(grads)
  if len(grad_dq) <= filter_step:
      new_grads = None
      updates, new_opt_state = optimizer.update(grads, opt_state)
  else:
      grad_dq.pop(0)
      new_grads = dq_mean(grad_dq)
      # new_grads = temp_grads
      updates, new_opt_state = optimizer.update(new_grads, opt_state)

  new_params = optax.apply_updates(params, updates)
  return new_params, grad_dq, new_opt_state, (loss, metrics, accuracy, valid_loss, valid_accuracy, grads, new_grads)

@functools.partial(
    jax.jit,
    static_argnames=(
        "model_apply_fn",
        "loss_fn",
        "accuracy_fn",
        "optimizer",
        "is_autoregressive",
    ),
)
def _update_parameters_valid(
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    valid_batch: task_lib.Batch,
    model_apply_fn: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    is_autoregressive: bool = False,
) -> Tuple[hk.Params, optax.OptState, Tuple[float, _LossMetrics, float]]:
  """Applies a single SGD update step to the model parameters.

  Args:
    params: The model parameters.
    rng_key: The prng key to use for random number generation.
    batch: The data (consists of both inputs and outputs).
    model_apply_fn: The model function that converts inputs into outputs.
    loss_fn: A function that computes the loss for a batch of logits and labels.
    accuracy_fn: A function that computes the accuracy for a batch of logits and
      labels.
    optimizer: The optimizer that computes the updates from the gradients of the
      `loss_fn` with respect to the `params` and the previous `opt_state`.
    opt_state: The optimizer state, e.g., momentum for each variable when using
      Adam.
    is_autoregressive: Whether the model is autoregressive or not.

  Returns:
    The updated parameters, the new optimizer state, and the loss, loss metrics
    and accuracy.
  """

  ##### Here model runs validation first (before parameter update) start
  if is_autoregressive:
      outputs = model_apply_fn(
          params,
          rng_key,
          valid_batch['input'],
          jnp.empty_like(valid_batch['output']),
          sample=True)
  else:
      outputs = model_apply_fn(params, rng_key, valid_batch['input'])
  # jax.debug.print("outputs = {}", outputs)
  # jax.debug.print("valid_batch = {}", valid_batch["output"])

  valid_loss, valid_loss_metrics = loss_fn(outputs, valid_batch["output"])
  if accuracy_fn is not None:
      valid_accuracy = accuracy_fn(outputs, valid_batch["output"])
  else:
      valid_accuracy = None
  ##### Here model runs validation first (before parameter update) end

  (loss, (metrics, accuracy)), grads = jax.value_and_grad(
      _apply_loss_and_metrics_fn,
      has_aux=True)(params, rng_key, batch, model_apply_fn, loss_fn,
                    accuracy_fn, is_autoregressive)
  updates, new_opt_state = optimizer.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)
  return new_params, new_opt_state, (loss, metrics, accuracy, valid_loss, valid_accuracy, grads, None)



def _update_parameters_valid_filter_balance_pre( #2:37:55
    weight_a: float,
    weight_b: float,
    grad_dq,
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    valid_batch: task_lib.Batch,
    model_apply_fn: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    is_autoregressive: bool = False,
) -> Tuple[hk.Params, optax.OptState, Tuple[float, _LossMetrics, float]]:
  ##### Here model runs validation first (before parameter update) start
  if is_autoregressive:
      outputs = model_apply_fn(
          params,
          rng_key,
          valid_batch['input'],
          jnp.empty_like(valid_batch['output']),
          sample=True)
  else:
      outputs = model_apply_fn(params, rng_key, valid_batch['input'])
  valid_loss, valid_loss_metrics = loss_fn(outputs, valid_batch["output"])
  if accuracy_fn is not None:
      valid_accuracy = accuracy_fn(outputs, valid_batch["output"])
  else:
      valid_accuracy = None
  ##### Here model runs validation first (before parameter update) end
  (loss, (metrics, accuracy)), grads = jax.value_and_grad(
      _apply_loss_and_metrics_fn,
      has_aux=True)(params, rng_key, batch, model_apply_fn, loss_fn,
                    accuracy_fn, is_autoregressive)
  grad_dq.append(grads) #This makes time delay
  new_grads = None
  updates, new_opt_state = optimizer.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)

  return new_params, grad_dq, new_opt_state, (loss, metrics, accuracy, valid_loss, valid_accuracy, grads, new_grads)



@functools.partial(
    jax.jit,
    static_argnames=(
        "weight_a",
        "weight_b",
        "model_apply_fn",
        "loss_fn",
        "accuracy_fn",
        "optimizer",
        "is_autoregressive",
    ),
    # donate_argnames=(
    #     "grad_dq"
    # )
)

def _update_parameters_valid_filter_balance(
    weight_a: float,
    weight_b: float,
    grad_dq,
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    valid_batch: task_lib.Batch,
    model_apply_fn: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    is_autoregressive: bool = False,
) -> Tuple[hk.Params, optax.OptState, Tuple[float, _LossMetrics, float]]:
  ##### Here model runs validation first (before parameter update) start
  if is_autoregressive:
      outputs = model_apply_fn(
          params,
          rng_key,
          valid_batch['input'],
          jnp.empty_like(valid_batch['output']),
          sample=True)
  else:
      outputs = model_apply_fn(params, rng_key, valid_batch['input'])
  valid_loss, valid_loss_metrics = loss_fn(outputs, valid_batch["output"])
  if accuracy_fn is not None:
      valid_accuracy = accuracy_fn(outputs, valid_batch["output"])
  else:
      valid_accuracy = None
  ##### Here model runs validation first (before parameter update) end

  (loss, (metrics, accuracy)), grads = jax.value_and_grad(
      _apply_loss_and_metrics_fn,
      has_aux=True)(params, rng_key, batch, model_apply_fn, loss_fn,
                    accuracy_fn, is_autoregressive)

  grad_dq = dq_append_whole(grad_dq, grads)
  new_grads = adj_dict_mean(grads, dq_mean_whole(grad_dq), weight_a, weight_b)
  updates, new_opt_state = optimizer.update(new_grads, opt_state)
  new_params = optax.apply_updates(params, updates)

  return new_params, grad_dq, new_opt_state, (loss, metrics, accuracy, valid_loss, valid_accuracy, grads, new_grads)


#TODO rebuttal
def _update_parameters_ema_balance_pre( #2:37:55
    weight_a: float,
    weight_b: float,
    sm:float,
    grad_dq,
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    valid_batch: task_lib.Batch,
    model_apply_fn: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    is_autoregressive: bool = False,
) -> Tuple[hk.Params, optax.OptState, Tuple[float, _LossMetrics, float]]:
  ##### Here model runs validation first (before parameter update) start
  if is_autoregressive:
      outputs = model_apply_fn(
          params,
          rng_key,
          valid_batch['input'],
          jnp.empty_like(valid_batch['output']),
          sample=True)
  else:
      outputs = model_apply_fn(params, rng_key, valid_batch['input'])
  valid_loss, valid_loss_metrics = loss_fn(outputs, valid_batch["output"])
  if accuracy_fn is not None:
      valid_accuracy = accuracy_fn(outputs, valid_batch["output"])
  else:
      valid_accuracy = None
  ##### Here model runs validation first (before parameter update) end
  (loss, (metrics, accuracy)), grads = jax.value_and_grad(
      _apply_loss_and_metrics_fn,
      has_aux=True)(params, rng_key, batch, model_apply_fn, loss_fn,
                    accuracy_fn, is_autoregressive)
  if isinstance(grad_dq, list):
      grad_dq = grads
  else:
      grad_dq = ema_add(grads, grad_dq, sm)
  new_grads = None
  updates, new_opt_state = optimizer.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)

  return new_params, grad_dq, new_opt_state, (loss, metrics, accuracy, valid_loss, valid_accuracy, grads, new_grads)


@functools.partial(
    jax.jit,
    static_argnames=(
        "weight_a",
        "weight_b",
        "sm",
        "model_apply_fn",
        "loss_fn",
        "accuracy_fn",
        "optimizer",
        "is_autoregressive",
    ),
    # donate_argnames=(
    #     "grad_dq"
    # )
)

def _update_parameters_ema_balance(
    weight_a: float,
    weight_b: float,
    sm: float,
    grad_dq,
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    valid_batch: task_lib.Batch,
    model_apply_fn: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    is_autoregressive: bool = False,
) -> Tuple[hk.Params, optax.OptState, Tuple[float, _LossMetrics, float]]:
  ##### Here model runs validation first (before parameter update) start
  if is_autoregressive:
      outputs = model_apply_fn(
          params,
          rng_key,
          valid_batch['input'],
          jnp.empty_like(valid_batch['output']),
          sample=True)
  else:
      outputs = model_apply_fn(params, rng_key, valid_batch['input'])
  valid_loss, valid_loss_metrics = loss_fn(outputs, valid_batch["output"])
  if accuracy_fn is not None:
      valid_accuracy = accuracy_fn(outputs, valid_batch["output"])
  else:
      valid_accuracy = None
  ##### Here model runs validation first (before parameter update) end

  (loss, (metrics, accuracy)), grads = jax.value_and_grad(
      _apply_loss_and_metrics_fn,
      has_aux=True)(params, rng_key, batch, model_apply_fn, loss_fn,
                    accuracy_fn, is_autoregressive)

  grad_dq = ema_add(grads, grad_dq, sm)
  new_grads = adj_dict_mean(grads, grad_dq, weight_a, weight_b)
  updates, new_opt_state = optimizer.update(new_grads, opt_state)
  new_params = optax.apply_updates(params, updates)

  return new_params, grad_dq, new_opt_state, (loss, metrics, accuracy, valid_loss, valid_accuracy, grads, new_grads)


def _update_parameters_valid_filter_add_pre(
    weight_a: float,
    weight_b: float,
    grad_dq,
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    valid_batch: task_lib.Batch,
    model_apply_fn: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    is_autoregressive: bool = False,
) -> Tuple[hk.Params, optax.OptState, Tuple[float, _LossMetrics, float]]:
  """Applies a single SGD update step to the model parameters.

  Args:
    update_step: current training step
    filter_step: time step to apply low-pass filter
    grad_dq: python deque containing gradients
    params: The model parameters.
    rng_key: The prng key to use for random number generation.
    batch: The data (consists of both inputs and outputs).
    model_apply_fn: The model function that converts inputs into outputs.
    loss_fn: A function that computes the loss for a batch of logits and labels.
    accuracy_fn: A function that computes the accuracy for a batch of logits and
      labels.
    optimizer: The optimizer that computes the updates from the gradients of the
      `loss_fn` with respect to the `params` and the previous `opt_state`.
    opt_state: The optimizer state, e.g., momentum for each variable when using
      Adam.
    is_autoregressive: Whether the model is autoregressive or not.

  Returns:
    The updated parameters, the new optimizer state, and the loss, loss metrics
    and accuracy.
  """
  ##### Here model runs validation first (before parameter update) start
  if is_autoregressive:
      outputs = model_apply_fn(
          params,
          rng_key,
          valid_batch['input'],
          jnp.empty_like(valid_batch['output']),
          sample=True)
  else:
      outputs = model_apply_fn(params, rng_key, valid_batch['input'])

  valid_loss, valid_loss_metrics = loss_fn(outputs, valid_batch["output"])
  if accuracy_fn is not None:
      valid_accuracy = accuracy_fn(outputs, valid_batch["output"])
  else:
      valid_accuracy = None
  ##### Here model runs validation first (before parameter update) end

  (loss, (metrics, accuracy)), grads = jax.value_and_grad(
      _apply_loss_and_metrics_fn,
      has_aux=True)(params, rng_key, batch, model_apply_fn, loss_fn,
                    accuracy_fn, is_autoregressive)
  grad_dq.append(grads)
  new_grads = None
  updates, new_opt_state = optimizer.update(grads, opt_state)

  new_params = optax.apply_updates(params, updates)
  return new_params, grad_dq, new_opt_state, (loss, metrics, accuracy, valid_loss, valid_accuracy, grads, new_grads)


@functools.partial(
    jax.jit,
    static_argnames=(
        "weight_a",
        "weight_b",
        "model_apply_fn",
        "loss_fn",
        "accuracy_fn",
        "optimizer",
        "is_autoregressive",
    ),
    # donate_argnames=(
    #     "grad_dq"
    # )

)
def _update_parameters_valid_filter_add(
    weight_a: float,
    weight_b: float,
    grad_dq,
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    valid_batch: task_lib.Batch,
    model_apply_fn: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    is_autoregressive: bool = False,
) -> Tuple[hk.Params, optax.OptState, Tuple[float, _LossMetrics, float]]:
  """Applies a single SGD update step to the model parameters.

  Args:
    update_step: current training step
    filter_step: time step to apply low-pass filter
    grad_dq: python deque containing gradients
    params: The model parameters.
    rng_key: The prng key to use for random number generation.
    batch: The data (consists of both inputs and outputs).
    model_apply_fn: The model function that converts inputs into outputs.
    loss_fn: A function that computes the loss for a batch of logits and labels.
    accuracy_fn: A function that computes the accuracy for a batch of logits and
      labels.
    optimizer: The optimizer that computes the updates from the gradients of the
      `loss_fn` with respect to the `params` and the previous `opt_state`.
    opt_state: The optimizer state, e.g., momentum for each variable when using
      Adam.
    is_autoregressive: Whether the model is autoregressive or not.

  Returns:
    The updated parameters, the new optimizer state, and the loss, loss metrics
    and accuracy.
  """
  ##### Here model runs validation first (before parameter update) start
  if is_autoregressive:
      outputs = model_apply_fn(
          params,
          rng_key,
          valid_batch['input'],
          jnp.empty_like(valid_batch['output']),
          sample=True)
  else:
      outputs = model_apply_fn(params, rng_key, valid_batch['input'])

  valid_loss, valid_loss_metrics = loss_fn(outputs, valid_batch["output"])
  if accuracy_fn is not None:
      valid_accuracy = accuracy_fn(outputs, valid_batch["output"])
  else:
      valid_accuracy = None
  ##### Here model runs validation first (before parameter update) end

  (loss, (metrics, accuracy)), grads = jax.value_and_grad(
      _apply_loss_and_metrics_fn,
      has_aux=True)(params, rng_key, batch, model_apply_fn, loss_fn,
                    accuracy_fn, is_autoregressive)

  grad_dq = dq_append_whole(grad_dq, grads)
  new_grads = add_dict_mean(grads, dq_mean_whole(grad_dq), weight_a, weight_b)
  updates, new_opt_state = optimizer.update(new_grads, opt_state)
  new_params = optax.apply_updates(params, updates)

  return new_params, grad_dq, new_opt_state, (loss, metrics, accuracy, valid_loss, valid_accuracy, grads, new_grads)

