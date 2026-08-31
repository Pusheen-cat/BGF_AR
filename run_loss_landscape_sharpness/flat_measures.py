"""The three loss-landscape sharpness measures reported in the paper.

Each function takes a ``ModelForSharp`` wrapper (see sharp_eval.py) exposing
the trained parameters, the task/model pair and Monte-Carlo loss evaluation,
and returns one scalar.  ``length=None`` evaluates on the training
distribution (curriculum lengths 1..sequence_length); an integer evaluates on
that fixed sequence length.

  * ``low_pass``        - low-pass-filter-based sharpness: Monte-Carlo
                          estimate of the Gaussian-smoothed loss
                          E_eps[L(theta + eps)], eps ~ N(0, sigma^2 I).
  * ``fim``             - Fisher-information-based sharpness:
                          theta^T H theta with the loss Hessian H applied via
                          a Hessian-vector product at the trained parameters.
  * ``shannon_entropy`` - Shannon-entropy-based sharpness: mean Shannon
                          entropy -E[sum_c p_c log p_c] of the model's output
                          distribution over sampled batches.
"""
import copy

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import tqdm


def low_pass(model_func, sigma=0.01, mcmc_itr=100, length=None):
    """Average loss under Gaussian parameter perturbations of scale sigma."""
    out = 0.0
    theta_star = model_func.param_trained
    for _ in tqdm.tqdm(range(mcmc_itr)):
        tmp_theta = copy.deepcopy(theta_star)
        for key, value in theta_star.items():
            for key_, _ in value.items():
                tmp_theta[key][key_] += np.random.normal(
                    size=tmp_theta[key][key_].shape, scale=sigma)
        out += model_func.compute_loss(tmp_theta, length=length)[0]
    return out / mcmc_itr


def fim(model_func, length=None):
    """theta^T (H theta) at the trained parameters (Fisher-Rao-norm style)."""
    array_param = []
    for _, layer in model_func.param_trained.items():
        for _, w_b in layer.items():
            array_param.append(np.array(copy.deepcopy(w_b)).flatten())
    flatten_param = np.concatenate(array_param)
    return float(flatten_param.T @ model_func.hvp(flatten_param, length=length))


def shannon_entropy(model_func, length=None):
    """Mean Shannon entropy of the output distribution over sampled batches."""
    res = 0.0
    rng_seq = model_func.fresh_rng_seq()
    for step in range(model_func.sharp_steps):
        batch = model_func.sample_batch(rng_seq, step, length)
        outputs = model_func.model.apply(
            model_func.param_trained, next(rng_seq), batch['input'])
        res += jnp.mean(jnp.sum(jnn.softmax(outputs) * jnn.log_softmax(outputs), axis=-1))
    return float(-res / model_func.sharp_steps)
