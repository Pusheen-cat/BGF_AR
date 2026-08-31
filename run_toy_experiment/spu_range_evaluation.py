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

"""Length-range evaluation for the label-concatenation toy experiment.

Identical to BGF_AR.training.range_evaluation except that every test batch is
extended with an UNCORRELATED random cue channel (spurious_corr.add_random),
so the reported accuracies measure the true rule, not cue copying.
"""

import dataclasses
import random
from typing import Any, List, Mapping, Callable


from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from BGF_AR.run_toy_experiment.spurious_corr import *


_Batch = Mapping[str, jnp.ndarray]


@dataclasses.dataclass
class EvaluationParams:
  """The parameters used for range evaluation of networks."""
  model: hk.Transformed
  params: hk.Params

  accuracy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
  sample_batch: Callable[[jnp.ndarray, int, int], _Batch]

  spu_last: bool

  max_test_length: int
  total_batch_size: int
  sub_batch_size: int  # We use this to avoid memory overflow.

  is_autoregressive: bool = False


def range_evaluation(
    eval_params: EvaluationParams,
    use_tqdm: bool = False,
) -> List[Mapping[str, Any]]:
  """Evaluates the model on longer, never seen strings and log the results.

  Args:
    eval_params: The evaluation parameters, see above.
    use_tqdm: Whether to use a progress bar with tqdm.

  Returns:
    The list of dicts containing the accuracies.
  """
  model = eval_params.model
  params = eval_params.params

  random.seed(42)
  np.random.seed(42)
  rng_seq = hk.PRNGSequence(42)

  if eval_params.is_autoregressive:
    apply_fn = jax.jit(model.apply, static_argnames=('sample',))
  else:
    apply_fn = jax.jit(model.apply)

  results = {}
  lengths = range(1, eval_params.max_test_length + 1)
  if use_tqdm:
    lengths = tqdm.tqdm(lengths)
  for length in lengths:
    # We need to clear the cache of jitted functions, to avoid overflow as we
    # are jitting len(lengths) ones, which can be a lot.
    apply_fn.clear_cache()
    sub_accuracies = []
    for _ in range(eval_params.total_batch_size // eval_params.sub_batch_size):
      batch = eval_params.sample_batch(
          next(rng_seq), eval_params.sub_batch_size, length)
      batch = add_random(batch, last=eval_params.spu_last)

      if eval_params.is_autoregressive:
        outputs = apply_fn(
            params,
            next(rng_seq),
            batch['input'],
            jnp.empty_like(batch['output']),
            sample=True)
      else:
        outputs = apply_fn(params, next(rng_seq), batch['input'])

      sub_accuracies.append(
          float(np.mean(eval_params.accuracy_fn(outputs, batch['output']))))
    results[length] = {'final_acc': np.mean(sub_accuracies).item()}
  return results
