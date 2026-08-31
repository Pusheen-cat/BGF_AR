# Copyright 2022 DeepMind Technologies Limited
# ... (License header omitted) ...

"""Dynamic Dyck Language Tasks with Controlled Error Rate."""

import functools
from typing import Mapping

import chex
import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import nn as jnn
import numpy as np

from BGF_AR.tasks import task


class DynamicDyckTask(task.GeneralizationTask):
    """
    Dynamic Dyck Task with controllable error rate logic.
    """

    def __init__(self, *args,
                 k: int = 2,
                 variant: str = 'stack_top',
                 error_prob: float = 0.1,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k = k
        self.variant = variant
        self.error_prob = error_prob
        self._input_vocab = 2 * k + 1
        self.TOKEN_PAD = self.k
        self.TOKEN_ERROR = self.k + 1

    @property
    def input_size(self) -> int:
        return self._input_vocab

    @property
    def output_size(self) -> int:
        if self.variant == 'stack_top':
            return self.k + 2
        elif self.variant == 'recognition':
            return 2
        elif self.variant == 'full_state':
            return self.k + 2
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def output_length(self, input_length: int) -> int:
        if self.variant == 'full_state':
            return input_length
        return 1

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def sample_batch(self, rng: chex.PRNGKey, batch_size: int, length: int) -> Mapping[str, chex.Array]:
        rng_reset, rng_noise, rng_mix = jrandom.split(rng, 3)

        # 1. Decide noise mode (per batch element)
        # Decide upfront whether each sequence will be a noise or a valid sequence.
        use_noise = jrandom.bernoulli(rng_mix, p=self.error_prob, shape=(batch_size,))

        # 2. Fused step function
        # Performs data generation (Generate) and state validation (Simulate) at once
        def fused_step(carry, _):
            stack, ptr, is_valid, key, is_noise_seq = carry
            key, subkey_gen, subkey_noise = jrandom.split(key, 3)

            # --- [A] Generate valid token candidate ---
            # Push if the stack is empty, otherwise pop with 50% probability
            must_push = (ptr == 0)
            random_action = jrandom.bernoulli(subkey_gen, p=0.5)
            do_pop = (~must_push) & random_action

            # Token to push (random bracket)
            new_open_type = jrandom.randint(subkey_gen, shape=(), minval=0, maxval=self.k)
            # Token to pop (closing bracket matching the stack top)
            safe_ptr_idx = jnp.maximum(0, ptr - 1)
            expected_open = stack[safe_ptr_idx]
            token_pop = expected_open + self.k

            valid_token = jnp.where(do_pop, token_pop, new_open_type)

            # --- [B] Generate noise token candidate ---
            # Never generates PAD (2*k); sampled only among the brackets (0 ~ 2k-1)
            noise_token = jrandom.randint(subkey_noise, shape=(), minval=0, maxval=2 * self.k)

            # --- [C] Token selection ---
            # Select according to the pre-computed use_noise flag
            token = jnp.where(is_noise_seq, noise_token, valid_token)

            # --- [D] Simulator Logic (State Update) ---
            # Immediately update the stack state with the selected token
            is_open = (token < self.k)
            is_close = (token >= self.k) & (token < 2 * self.k)
            close_type = token - self.k

            # Push Update
            new_stack_push = stack.at[ptr].set(token)
            new_ptr_push = ptr + 1

            # Pop Update
            top_val = stack[ptr - 1]
            match = (top_val == close_type)
            valid_pop = (ptr > 0) & match
            new_ptr_pop = ptr - 1

            # Apply Update logic
            next_stack = jnp.where(is_open, new_stack_push, stack)
            next_ptr = jnp.where(is_open, new_ptr_push, ptr)
            next_ptr = jnp.where(is_close, jnp.maximum(0, new_ptr_pop), next_ptr)

            # Validity Update
            current_step_invalid = is_close & (~valid_pop)
            next_is_valid = is_valid & (~current_step_invalid)

            # Return
            # Emit the stack and ptr at every step for the full_state labeling
            return (next_stack, next_ptr, next_is_valid, key, is_noise_seq), (token, next_stack, next_ptr)

        # Init variables
        init_stack = jnp.zeros((batch_size, length), dtype=jnp.int32)
        init_ptr = jnp.zeros((batch_size,), dtype=jnp.int32)
        init_valid = jnp.ones((batch_size,), dtype=jnp.bool_)
        rng_split = jrandom.split(rng_reset, batch_size)

        # 3. Single Scan Execution
        final_carry, (x_seq, stack_seq, ptr_seq) = jax.lax.scan(
            lambda c, _: jax.vmap(fused_step)(c, None),
            (init_stack, init_ptr, init_valid, rng_split, use_noise),
            None,
            length=length
        )

        final_stack, final_ptr, final_valid, _, _ = final_carry

        # (time, batch, ...) -> (batch, time, ...)
        x = x_seq.T

        # 4. Label Generation
        if self.variant == 'full_state':
            # Uses final_stack (final state) instead of stack_seq (history).
            # final_stack shape: (batch_size, length)
            # final_ptr shape: (batch_size,)

            # 1. Masking Logic
            # Only stack indices (0, 1, 2...) below the current pointer (final_ptr) hold valid values.
            idxs = jnp.arange(length)[None, :]  # (1, length)
            ptr_exp = final_ptr[:, None]  # (batch, 1)
            stack_mask = idxs < ptr_exp  # (batch, length) -> True at valid stack positions

            # 2. Label generation
            # Valid positions take the stack value, the rest are PAD
            seq_label = jnp.where(stack_mask, final_stack, self.TOKEN_PAD)

            # 3. Error Sequence Handling (invalid case)
            # If any rule was violated during simulation (final_valid == False),
            # the output is a sequence of one error token followed by PADs.
            error_seq = jnp.full((batch_size, length), self.TOKEN_PAD, dtype=jnp.int32)
            error_seq = error_seq.at[:, 0].set(self.TOKEN_ERROR)

            # 4. Final Output Selection
            y = jnp.where(final_valid[:, None], seq_label, error_seq)

            # shape: (batch, length, output_size)
            output = jnn.one_hot(y, num_classes=self.output_size)

        elif self.variant == 'stack_top':
            safe_ptr_idx = jnp.maximum(0, final_ptr - 1)
            top_elements = jax.vmap(lambda s, p: s[p])(final_stack, safe_ptr_idx)

            label = jnp.where(
                ~final_valid, self.k + 1,
                jnp.where(
                    final_ptr == 0, 0,
                    top_elements + 1
                )
            )
            output = jnn.one_hot(label, num_classes=self.output_size)

        elif self.variant == 'recognition':
            label = jnp.where(final_valid, 0, 1)
            output = jnn.one_hot(label, num_classes=self.output_size)

        return {
            'input': jnn.one_hot(x, num_classes=self.input_size),
            'output': output,
        }

# Wrapper classes remain the same
# Wrapper classes (Fixed)
class DyckFullStateTask_complex(DynamicDyckTask):
    # Takes error_prob as an explicit argument
    def __init__(self, *args, error_prob: float = 0.2, **kwargs):
        super().__init__(*args, variant='full_state', error_prob=error_prob, **kwargs)

    def accuracy_mask(self, target: chex.Array) -> chex.Array:
        """
        Computes mask that ignores everything after the first PAD token.
        We include the first PAD token because predicting the 'end of stack' is important.

        Args:
          target: Target tokens of shape `(batch_size, output_length, output_size)`.

        Returns:
          The mask of shape `(batch_size, output_length)`.
        """
        # 1. One-hot -> index conversion
        target_idx = jnp.argmax(target, axis=-1)

        # 2. Locate PAD tokens
        # TOKEN_PAD equals self.k
        is_pad = (target_idx == self.TOKEN_PAD)

        # 3. Cumulative sum
        # Example: [A, B, PAD, PAD, PAD] -> is_pad: [0, 0, 1, 1, 1] -> cumsum: [0, 0, 1, 2, 3]
        pad_cumsum = jnp.cumsum(is_pad, axis=-1)

        # 4. Build the mask
        # cumsum == 0: no PAD seen yet (valid stack data)
        # cumsum == 1: the first PAD (marks the end of the stack, must be predicted too)
        # cumsum >= 2: trailing PADs that do not matter
        mask = (pad_cumsum <= 1)

        return mask


class DyckStackTopTask_complex(DynamicDyckTask):
    def __init__(self, *args, error_prob: float = 0.2, **kwargs):
        super().__init__(*args, variant='stack_top', error_prob=error_prob, **kwargs)


class DyckRecognitionTask_complex(DynamicDyckTask):
    def __init__(self, *args, error_prob: float = 0.5, **kwargs):
        super().__init__(*args, variant='recognition', error_prob=error_prob, **kwargs)


# =========================================================
# Main Execution Block
# =========================================================
if __name__ == "__main__":
    # Helper function for visualization
    def decode_input(seq, k):
        # 0: (, 1: [, 2: ), 3: ], 4: PAD, 5: ERR
        tokens = []
        for token in seq:
            if token < k:
                tokens.append(f"Open{token}")
            elif token < 2 * k:
                tokens.append(f"Clos{token - k}")
            elif token == 2 * k:
                tokens.append("PAD")
            else:
                tokens.append("ERR")
        return " ".join(tokens)


    # Settings
    K = 2
    BATCH_SIZE = 5
    LENGTH = 8
    KEY = jrandom.PRNGKey(42)

    print(f"=== Dyck Task Demo (K={K}, Length={LENGTH}) ===\n")

    # 1. Full State variant test
    print(">>> 1. Full State Variant (Sequence-to-Sequence)")
    task_full = DyckFullStateTask_complex(k=K, error_prob=0.2)
    batch_full = task_full.sample_batch(KEY, BATCH_SIZE, LENGTH)

    in_full = jnp.argmax(batch_full['input'], axis=-1)
    out_full = jnp.argmax(batch_full['output'], axis=-1)

    for i in range(BATCH_SIZE):
        print(f"Batch {i}:")
        print(f"  Input:  {decode_input(in_full[i], K)}")
        # The output represents the stack state (type IDs)
        print(f"  Target: {out_full[i]}")
    print("-" * 50)

    # 2. Stack Top variant test
    print("\n>>> 2. Stack Top Variant (Sequence-to-Label)")
    # Raise error_prob to 0.5 to generate invalid cases more often
    task_top = DyckStackTopTask_complex(k=K, error_prob=0.5)
    batch_top = task_top.sample_batch(KEY, BATCH_SIZE, LENGTH)

    in_top = jnp.argmax(batch_top['input'], axis=-1)
    out_top = jnp.argmax(batch_top['output'], axis=-1)

    for i in range(BATCH_SIZE):
        # Label meaning: 0=Empty, 1~K=Top Type, K+1=Invalid
        label_str = "Invalid" if out_top[i] == K + 1 else (f"Top is {out_top[i] - 1}" if out_top[i] > 0 else "Empty")
        print(f"Batch {i}: {decode_input(in_top[i], K)}")
        print(f"  -> Label: {out_top[i]} ({label_str})")
    print("-" * 50)

    # 3. Recognition variant test
    print("\n>>> 3. Recognition Variant (Binary Classification)")
    task_rec = DyckRecognitionTask_complex(k=K, error_prob=0.5)
    batch_rec = task_rec.sample_batch(KEY, BATCH_SIZE, LENGTH)

    in_rec = jnp.argmax(batch_rec['input'], axis=-1)
    out_rec = jnp.argmax(batch_rec['output'], axis=-1)

    for i in range(BATCH_SIZE):
        # Label meaning: 0=Valid, 1=Invalid
        status = "Valid" if out_rec[i] == 0 else "Invalid"
        print(f"Batch {i}: {decode_input(in_rec[i], K)}")
        print(f"  -> Result: {out_rec[i]} ({status})")