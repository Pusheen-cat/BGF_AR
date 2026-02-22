# Copyright 2022 DeepMind Technologies Limited
# ... (License header omitted) ...

"""Bounded Dyck Task: Valid Prefixes, Fixed Depth, Output Length 1."""

import functools
from typing import Mapping

import chex
import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import nn as jnn
import numpy as np

from BGF_AR.tasks import task


class DyckTask(task.GeneralizationTask):
    """
    Bounded Dyck Task implementation.
    - Max depth limited (FSM based).
    - Valid Prefix allowed (e.g., '((' is Valid).
    - Output Length is ALWAYS 1 (Final state only).
    """

    def __init__(self, *args,
                 k: int = 2,
                 max_depth: int = 2,
                 variant: str = 'full_state',
                 error_prob: float = 0.2,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k = k
        self.max_depth = max_depth
        self.variant = variant
        self.error_prob = error_prob

        self._input_vocab = 2 * k

        # ----------------------------------------------------------------
        # 1. State Space Construction
        # ----------------------------------------------------------------
        self.states = []

        def generate_states(current_stack):
            self.states.append(tuple(current_stack))
            if len(current_stack) < self.max_depth:
                for bracket_type in range(k):
                    generate_states(current_stack + [bracket_type])

        generate_states([])

        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.num_valid_states = len(self.states)
        self.error_state_idx = self.num_valid_states

        # ----------------------------------------------------------------
        # 2. Build Transition Table
        # ----------------------------------------------------------------
        table = np.zeros((self.num_valid_states + 1, self._input_vocab), dtype=np.int32)

        for idx, stack in enumerate(self.states):
            # Open Brackets
            for open_b in range(self.k):
                if len(stack) < self.max_depth:
                    new_stack = tuple(list(stack) + [open_b])
                    table[idx, open_b] = self.state_to_idx[new_stack]
                else:
                    table[idx, open_b] = self.error_state_idx  # Overflow

            # Close Brackets
            for close_b_offset in range(self.k):
                close_b_input = self.k + close_b_offset
                if len(stack) > 0 and stack[-1] == close_b_offset:
                    new_stack = tuple(list(stack)[:-1])
                    table[idx, close_b_input] = self.state_to_idx[new_stack]
                else:
                    table[idx, close_b_input] = self.error_state_idx  # Mismatch/Underflow

        table[self.error_state_idx, :] = self.error_state_idx
        self.transition_table = jnp.array(table)

        # ----------------------------------------------------------------
        # 3. Build Output Mapping Tables
        # ----------------------------------------------------------------
        # stack_top variant mapping
        top_mapping = np.zeros(self.num_valid_states + 1, dtype=np.int32)
        for idx, stack in enumerate(self.states):
            if len(stack) == 0:
                top_mapping[idx] = 0  # Empty
            else:
                top_mapping[idx] = stack[-1] + 1  # Type + 1
        top_mapping[self.error_state_idx] = self.k + 1  # Error
        self.state_top_map_table = jnp.array(top_mapping)

    @property
    def input_size(self) -> int:
        return self._input_vocab

    @property
    def output_size(self) -> int:
        if self.variant == 'full_state':
            # Valid states + 1 Error state
            return self.num_valid_states + 1
        elif self.variant == 'stack_top':
            # Empty(1) + Types(k) + Error(1)
            return self.k + 2
        elif self.variant == 'validity':
            # Valid(0) vs Error(1)
            return 2
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def output_length(self, input_length: int) -> int:
        return 1

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def sample_batch(self, rng: chex.PRNGKey, batch_size: int, length: int) -> Mapping[str, chex.Array]:
        rng_valid, rng_noise, rng_mix = jrandom.split(rng, 3)

        # ==========================================
        # 1. Valid Generator (Prefix Allowed)
        # ==========================================
        def valid_step(carry, _):
            stack, ptr, key, step_idx = carry
            key, subkey = jrandom.split(key)

            # 끝날 때 스택을 비우지 않아도 됨 (Valid Prefix 허용)
            must_push = (ptr == 0)
            must_pop = (ptr == self.max_depth)  # Overflow 방지를 위한 강제 Pop

            random_pop = jrandom.bernoulli(subkey, p=0.5)
            do_pop = must_pop | (random_pop & (~must_push))

            # Push
            new_open_type = jrandom.randint(subkey, shape=(), minval=0, maxval=self.k)
            # Pop
            safe_ptr_idx = jnp.maximum(0, ptr - 1)
            expected_open = stack[safe_ptr_idx]
            token_pop = expected_open + self.k

            token = jnp.where(do_pop, token_pop, new_open_type)

            new_stack = jnp.where(do_pop, stack, stack.at[ptr].set(new_open_type))
            new_ptr = jnp.where(do_pop, ptr - 1, ptr + 1)

            return (new_stack, new_ptr, key, step_idx + 1), token

        init_stack = jnp.zeros((batch_size, length), dtype=jnp.int32)
        init_ptr = jnp.zeros((batch_size,), dtype=jnp.int32)
        init_idx = jnp.zeros((batch_size,), dtype=jnp.int32)
        rng_valid_split = jrandom.split(rng_valid, batch_size)

        _, x_valid = jax.lax.scan(
            lambda c, _: jax.vmap(valid_step)(c, None),
            (init_stack, init_ptr, rng_valid_split, init_idx),
            None,
            length=length
        )
        x_valid = x_valid.T

        # ==========================================
        # 2. Noise & Mix
        # ==========================================
        x_noise = jrandom.randint(rng_noise, shape=(batch_size, length), minval=0, maxval=self._input_vocab)
        use_noise = jrandom.bernoulli(rng_mix, p=self.error_prob, shape=(batch_size, 1))
        x = jnp.where(use_noise, x_noise, x_valid)

        # ==========================================
        # 3. Labeling (Final State Only)
        # ==========================================
        def fsm_step(state_idx, input_token):
            # vmap lookup
            next_state = jax.vmap(lambda s, t: self.transition_table[s, t])(state_idx, input_token)
            return next_state, None

        init_state = jnp.zeros((batch_size,), dtype=jnp.int32)

        # scan의 두번째 리턴(history)은 무시하고, 첫번째 리턴(final carry)만 사용
        final_state_idx, _ = jax.lax.scan(fsm_step, init_state, x.T)

        # ==========================================
        # 4. Generate Output (Length = 1)
        # ==========================================
        # final_state_idx Shape: (batch_size,)

        if self.variant == 'full_state':
            # 마지막 상태의 State Index 그대로 사용
            y = final_state_idx

        elif self.variant == 'stack_top':
            # 마지막 상태의 Top Element 매핑
            y = self.state_top_map_table[final_state_idx]

        elif self.variant == 'validity':
            # Error 상태가 아니면 모두 Valid (0)
            y = jnp.where(
                final_state_idx != self.error_state_idx,
                0,  # Valid (including non-empty prefixes)
                1  # Invalid
            )

        # [수정됨] 차원 확장 제거
        # 기존: y = y[:, None]  -> Shape: (batch, 1)
        # 수정: y 그대로 사용   -> Shape: (batch,)

        # One-hot encoding 적용
        # y가 (batch,) 이므로 one_hot 결과는 (batch, output_size)가 됩니다.
        output = jnn.one_hot(y, num_classes=self.output_size)

        return {
            'input': jnn.one_hot(x, num_classes=self.input_size),
            'output': output,
        }


# Wrapper classes for the dataset_map
class BoundedDyckTask_simple(DyckTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, variant='full_state', **kwargs)


class DyckStackTopTask_simple(DyckTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, variant='stack_top', **kwargs)


class DyckRecognitionTask_simple(DyckTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, variant='validity', error_prob = 0.5, **kwargs)

# Add to dataset map
# dataset_map['dyck_bounded'] = BoundedDyckTask_simple
# dataset_map['dyck_top'] = DyckStackTopTask_simple
# dataset_map['dyck_valid'] = DyckRecognitionTask_simple