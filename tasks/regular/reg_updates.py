# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Automata tasks for generalization (JAX version) - Final State Prediction."""

import functools
from typing import Mapping, Optional, Tuple, List, Union

import chex
import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import nn as jnn
import numpy as np
import itertools

from BGF_AR.tasks import task


# ==============================================================================
# 1. Gridworld Task / checked
# ==============================================================================
class GridworldTask(task.GeneralizationTask):
    """Gridworld Automaton Task (Final Position)."""

    def __init__(self, *args, n: int = 9, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n = n
        self.S = n - 1

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def sample_batch(self, rng: chex.PRNGKey, batch_size: int, length: int) -> Mapping[str, chex.Array]:
        x = jrandom.randint(rng, shape=(batch_size, length), minval=0, maxval=2)
        moves = jnp.where(x == 0, -1, 1)

        def step_fn(carry, move):
            new_pos = jnp.clip(carry + move, 0, self.S)
            return new_pos, None  # no per-step outputs recorded (only the final state is needed)

        # The scan carry is the final state
        final_pos, _ = jax.lax.scan(step_fn, jnp.zeros(batch_size, dtype=jnp.int32), moves.T)

        # No dimension expansion: keep (batch,) rather than final_pos[:, None] -> (batch, 1)
        y_final = final_pos

        return {
            'input': jnn.one_hot(x, num_classes=self.input_size),
            'output': jnn.one_hot(y_final, num_classes=self.output_size),  # (batch, class)
        }

    @property
    def input_size(self) -> int: return 2

    @property
    def output_size(self) -> int: return self.n

    def output_length(self, input_length: int) -> int: return 1


# ==============================================================================
# 2. ABAB Task
# ==============================================================================
class ABABTask(task.GeneralizationTask):
    """ABAB Automaton Task (Final State)."""

    def __init__(self, *args, prob_abab: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prob_abab = prob_abab
        self.transition_table = jnp.array([
            [4, 1], [2, 4], [4, 3], [0, 4], [4, 4]
        ])

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def sample_batch(self, rng: chex.PRNGKey, batch_size: int, length: int) -> Mapping[str, chex.Array]:
        rng_seq, rng_type = jrandom.split(rng)

        x_rand = jrandom.randint(rng_seq, shape=(batch_size, length), minval=0, maxval=2)
        pattern = jnp.tile(jnp.array([0, 1]), (length // 2 + 1))[:length]
        x_abab = jnp.tile(pattern, (batch_size, 1))

        probs = jrandom.uniform(rng_type, shape=(batch_size, 1))
        x = jnp.where(probs < self.prob_abab, x_abab, x_rand)

        def step_fn(state, input_token):
            return self.transition_table[state, input_token], None

        init_state = jnp.full((batch_size,), 3, dtype=jnp.int32)
        final_state, _ = jax.lax.scan(step_fn, init_state, x.T)

        # No dimension expansion: keep y_final as (batch,)
        y_final = final_state

        return {
            'input': jnn.one_hot(x, num_classes=self.input_size),
            'output': jnn.one_hot(y_final, num_classes=self.output_size),
        }

    @property
    def input_size(self) -> int: return 2

    @property
    def output_size(self) -> int: return 5

    def output_length(self, input_length: int) -> int: return 1


# ==============================================================================
# 3. Adder Task
# ==============================================================================
class AdderTask(task.GeneralizationTask):
    """Adder Automaton Task (Final Output Label)."""

    def __init__(self, *args, n_addends: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_addends = n_addends
        self._input_vocab = 2 ** n_addends
        self._output_vocab = n_addends * n_addends + n_addends

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def sample_batch(self, rng: chex.PRNGKey, batch_size: int, length: int) -> Mapping[str, chex.Array]:
        x = jrandom.randint(rng, shape=(batch_size, length), minval=0, maxval=self._input_vocab)

        # Popcount logic for column sum
        def get_column_sum(val):
            return sum([(val >> i) & 1 for i in range(self.n_addends)])

        lookup = jnp.array([get_column_sum(i) for i in range(self._input_vocab)])
        column_sums = lookup[x]

        def step_fn(carry, col_sum):
            curr_sum = col_sum + carry
            output_digit = curr_sum % self.n_addends
            new_carry = curr_sum // self.n_addends
            label = output_digit + self.n_addends * new_carry
            return new_carry, label

        # The adder predicts the label (digit + carry) of the last step
        _, y_seq = jax.lax.scan(step_fn, jnp.zeros(batch_size, dtype=jnp.int32), column_sums.T)

        y_final = y_seq[-1, :]  # Last time step (batch,)

        # No dimension expansion: y_final stays (batch,)

        return {
            'input': jnn.one_hot(x, num_classes=self.input_size),
            'output': jnn.one_hot(y_final, num_classes=self.output_size),
        }

    @property
    def input_size(self) -> int: return self._input_vocab

    @property
    def output_size(self) -> int: return self._output_vocab

    def output_length(self, input_length: int) -> int: return 1


# ==============================================================================
# 4. FlipFlop Task
# ==============================================================================
class FlipFlopTask(task.GeneralizationTask):
    """FlipFlop Automaton Task (Final Memory State)."""

    def __init__(self, *args, n: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n = n
        self._vocab_size = n + 1

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def sample_batch(self, rng: chex.PRNGKey, batch_size: int, length: int) -> Mapping[str, chex.Array]:
        rng_val, rng_action = jrandom.split(rng)
        is_write = jrandom.bernoulli(rng_val, p=0.5, shape=(batch_size, length))
        writes = jrandom.randint(rng_action, shape=(batch_size, length), minval=1, maxval=self.n + 1)
        x = jnp.where(is_write, writes, 0)

        def step_fn(memory, action):
            new_memory = jnp.where(action > 0, action, memory)
            return new_memory, None

        final_memory, _ = jax.lax.scan(step_fn, jnp.zeros(batch_size, dtype=jnp.int32), x.T)

        # No dimension expansion: keep y_final as (batch,)
        y_final = final_memory

        return {
            'input': jnn.one_hot(x, num_classes=self.input_size),
            'output': jnn.one_hot(y_final, num_classes=self.output_size),
        }

    @property
    def input_size(self) -> int: return self._vocab_size

    @property
    def output_size(self) -> int: return self._vocab_size

    def output_length(self, input_length: int) -> int: return 1


# ==============================================================================
# 5. Dihedral Task
# ==============================================================================
class DihedralTask(task.GeneralizationTask):
    """Dihedral Group Task (Final State)."""

    def __init__(self, *args, n: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n = n
        self._output_vocab = 2 * n

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def sample_batch(self, rng: chex.PRNGKey, batch_size: int, length: int) -> Mapping[str, chex.Array]:
        x = jrandom.randint(rng, shape=(batch_size, length), minval=0, maxval=2)

        toggles = (x == 0).astype(jnp.int32)
        toggle_status = jnp.sum(toggles, axis=-1) % 2

        # cumsum is needed to track the accumulated direction (intermediate values
        # are required even though only the final one is output); efficient and parallel
        intermediate_toggles = jnp.cumsum(toggles, axis=-1) % 2
        directions = jnp.power(-1, intermediate_toggles)  # (batch, length)
        drives = (x == 1).astype(jnp.int32) * directions
        final_positions = jnp.sum(drives, axis=-1) % self.n

        y_final = self.n * toggle_status + final_positions

        # No dimension expansion: y_final stays (batch,)

        return {
            'input': jnn.one_hot(x, num_classes=self.input_size),
            'output': jnn.one_hot(y_final, num_classes=self.output_size),
        }

    @property
    def input_size(self) -> int: return 2

    @property
    def output_size(self) -> int: return self._output_vocab

    def output_length(self, input_length: int) -> int: return 1


# ==============================================================================
# 6. Symmetric Task
# ==============================================================================
class SymmetricTask(task.GeneralizationTask):
    """Symmetric Group Task (Final Permutation)."""

    def __init__(self, *args, n: int = 4, n_actions: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n = n
        self.n_actions = n_actions

        generators = []
        generators.append(np.arange(n))  # 0: Id
        generators.append(np.array(list(range(1, n)) + [0]))  # 1: Shift
        generators.append(np.array([1, 0] + list(range(2, n))))  # 2: Swap

        while len(generators) < n_actions:
            generators.append(np.random.permutation(n))

        self.generators = jnp.array(generators)

        perms = list(itertools.permutations(range(n)))
        self.perm_to_idx_table = jnp.zeros([n] * n, dtype=jnp.int32)
        for i, p in enumerate(perms):
            self.perm_to_idx_table = self.perm_to_idx_table.at[p].set(i)
        self._output_vocab = len(perms)

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def sample_batch(self, rng: chex.PRNGKey, batch_size: int, length: int) -> Mapping[str, chex.Array]:
        x = jrandom.randint(rng, shape=(batch_size, length), minval=0, maxval=self.n_actions)

        def step_fn(curr_perm, action_idx):
            action_perm = self.generators[action_idx]
            new_perm = curr_perm[action_perm]
            return new_perm, None

        init_perms = jnp.tile(jnp.arange(self.n, dtype=jnp.int32), (batch_size, 1))

        def scan_over_seq(carry, inputs):
            return jax.lax.scan(step_fn, carry, inputs)

        # Batched with vmap; only the final state (final_perms) is used
        final_perms, _ = jax.vmap(scan_over_seq)(init_perms, x)

        # Convert the final permutation to an index
        # vmap over batch
        def get_label(perm): return self.perm_to_idx_table[tuple(perm)]

        y_final = jax.vmap(get_label)(final_perms)

        # No dimension expansion: y_final stays (batch,)

        return {
            'input': jnn.one_hot(x, num_classes=self.input_size),
            'output': jnn.one_hot(y_final, num_classes=self.output_size),
        }

    @property
    def input_size(self) -> int:
        return self.n_actions

    @property
    def output_size(self) -> int:
        return self._output_vocab

    def output_length(self, input_length: int) -> int:
        return 1


# ==============================================================================
# 7. Alternating Task
# ==============================================================================
class AlternatingTask(SymmetricTask):
    """Alternating Group Task (Inherits from Symmetric)."""

    def __init__(self, *args, n: int = 5, **kwargs) -> None:
        kwargs['n'] = n
        generators = [np.arange(n)]
        for idx in range(2, n):
            perm = list(range(n))
            perm[0], perm[1], perm[idx] = perm[1], perm[idx], perm[0]
            generators.append(np.array(perm))

        kwargs['n_actions'] = len(generators)
        super().__init__(*args, **kwargs)
        self.generators = jnp.array(generators)


# ==============================================================================
# 8. Quaternion Task
# ==============================================================================
class QuaternionTask(task.GeneralizationTask):
    """Quaternion Group Task (Final State)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_actions = 4
        self._output_vocab = 8

        pos = [0, 1, 2, 3, 1, 4, 3, 6, 2, 7, 4, 1, 3, 2, 5, 4]
        neg = [(x + 4) % 8 for x in pos]
        self.table = jnp.array(pos + neg).reshape(8, 4)

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def sample_batch(self, rng: chex.PRNGKey, batch_size: int, length: int) -> Mapping[str, chex.Array]:
        x = jrandom.randint(rng, shape=(batch_size, length), minval=0, maxval=self.n_actions)

        def step_fn(state, action):
            return self.table[state, action], None

        final_state, _ = jax.lax.scan(step_fn, jnp.zeros(batch_size, dtype=jnp.int32), x.T)

        # No dimension expansion: keep y_final as (batch,)
        y_final = final_state

        return {
            'input': jnn.one_hot(x, num_classes=self.input_size),
            'output': jnn.one_hot(y_final, num_classes=self.output_size),
        }

    @property
    def input_size(self) -> int: return self.n_actions

    @property
    def output_size(self) -> int: return self._output_vocab

    def output_length(self, input_length: int) -> int: return 1


# ==============================================================================
# 11. Permutation Reset Task
# ==============================================================================
class PermutationResetTask(task.GeneralizationTask):
    """Permutation Reset Automaton (Final State)."""

    def __init__(self, *args, n: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n = n
        self.n_states = 24  # 4!
        self.generators = [
            np.array(list(range(1, n)) + [0]),
            np.array([1, 0] + list(range(2, n)))
        ]
        self.n_gens = len(self.generators)
        self.n_actions = self.n_states + self.n_gens

        self.gen_arrays = jnp.array(self.generators)
        perms = list(itertools.permutations(range(n)))
        self.int2perm = jnp.array(perms)
        self.perm_to_idx_table = jnp.zeros([n] * n, dtype=jnp.int32)
        for i, p in enumerate(perms):
            self.perm_to_idx_table = self.perm_to_idx_table.at[p].set(i)

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def sample_batch(self, rng: chex.PRNGKey, batch_size: int, length: int) -> Mapping[str, chex.Array]:
        x = jrandom.randint(rng, shape=(batch_size, length), minval=0, maxval=self.n_actions)

        def step_fn(curr_perm, action):
            # Clamp the index into the valid range so that generator actions
            # do not cause out-of-bounds errors
            safe_reset_idx = jnp.minimum(action, self.n_states - 1)
            reset_perm = self.int2perm[safe_reset_idx]

            # Generator index computation (kept defensive)
            gen_idx = action - self.n_states
            safe_gen_idx = jnp.maximum(0, gen_idx)
            safe_gen = self.gen_arrays[safe_gen_idx]

            # Apply
            apply_perm = curr_perm[safe_gen]

            # Selection logic
            is_reset = action < self.n_states

            # Choose reset_perm for a reset action, apply_perm otherwise
            new_perm = jnp.where(is_reset, reset_perm, apply_perm)

            return new_perm, None

        init_perms = jnp.tile(jnp.arange(self.n, dtype=jnp.int32), (batch_size, 1))

        def scan_over_seq(carry, inputs):
            return jax.lax.scan(step_fn, carry, inputs)

        final_perms, _ = jax.vmap(scan_over_seq)(init_perms, x)

        def get_label(perm): return self.perm_to_idx_table[tuple(perm)]

        y_final = jax.vmap(get_label)(final_perms)

        # No dimension expansion: y_final stays (batch,)

        return {
            'input': jnn.one_hot(x, num_classes=self.input_size),
            'output': jnn.one_hot(y_final, num_classes=self.output_size),
        }

    @property
    def input_size(self) -> int: return self.n_actions

    @property
    def output_size(self) -> int: return self.n_states

    def output_length(self, input_length: int) -> int: return 1


# ==============================================================================
# Dataset Map
# ==============================================================================
update_regulars_map = {
    'abab': ABABTask,
    'add': AdderTask,
    'alternating': AlternatingTask,
    'dihedral': DihedralTask,
    'flipflop': FlipFlopTask,
    'gridworld': GridworldTask,
    'quaternion': QuaternionTask,
    'symmetric': SymmetricTask,
    'permutation_reset': PermutationResetTask
}