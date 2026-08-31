"""Per-step recording of a small sample of gradient coordinates.

Saving the full gradient tree at every training step is prohibitively large
(model size x 100k steps per run), so - as in the paper's gradient-frequency
experiment - only ``total_samples`` (default 500) scalar gradient coordinates
are recorded.  The coordinates are drawn once from the initial parameter tree
with a fixed seed and reused at every step, so each of them yields a
time-series "gradient signal" of length ``training_steps + 1`` that can be
analysed in the frequency domain (see ``run_grad_visualization/``).

The sampling scheme reproduces the paper's selection exactly: per leaf at
least 5 elements for biases ('b') and 20 for weights ('w'), the remainder
allocated proportionally to leaf size, drawn uniformly without replacement
with ``numpy.random.default_rng(seed)``.  Because the selection depends only
on the parameter-tree structure and the seed, two runs of the same
architecture sample the SAME coordinates, which makes their signals directly
comparable per coordinate.

Recorded signals are written to ``<run_dir>/grad_signal.pkl``:
    {'raw':      float32 (total_samples, training_steps + 1),  column i == step i,
     'filtered': float32 (total_samples, T') or None,  column i == step filtered_start_step + i,
     'raw_start_step': 0,
     'filtered_start_step': int or None,
     'positions': [(path tuple of str keys, index array into the flattened leaf), ...]}

``raw`` is the raw loss gradient of every step.  ``filtered`` is the gradient
the optimizer actually applied when it differs from the raw one (the BGF
low-frequency-enhanced gradient of ``ours_balance``/``ours_add``/``ours_ema``);
it starts at the step the gradient filter becomes active (``filter_step``,
default 100) and is None for plain Adam runs.
"""
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np


def _key_name(path_entry):
    """A tree_flatten_with_path entry -> its plain string key."""
    return getattr(path_entry, 'key', path_entry)


def sample_positions(params, total_samples=500, seed=42):
    """Samples ``total_samples`` coordinate positions from a parameter tree.

    Returns a list of (path, indices) with ``path`` a tuple of string keys
    into the (haiku) parameter dict and ``indices`` positions into the
    flattened leaf array.
    """
    rng = np.random.default_rng(seed)
    leaves = jax.tree_util.tree_flatten_with_path(params)[0]

    sizes = [leaf.size for _, leaf in leaves]
    total_size = sum(sizes)

    # minimum number of samples per leaf, by parameter kind
    min_counts = []
    for path, _ in leaves:
        last_key = _key_name(path[-1])
        if last_key == 'b':
            min_counts.append(5)
        elif last_key == 'w':
            min_counts.append(20)
        else:
            min_counts.append(1)

    # initial allocation: at least the minimum, otherwise proportional to size
    counts = [
        max(min(min_c, s), int(total_samples * s / total_size))
        for s, min_c in zip(sizes, min_counts)
    ]

    # adjust to the exact total (respecting the minimum counts)
    def largest_idx():
        return int(np.argmax([c - m for c, m in zip(counts, min_counts)]))

    while sum(counts) > total_samples:
        i = largest_idx()
        if counts[i] > min_counts[i]:
            counts[i] -= 1
        else:
            break

    while sum(counts) < total_samples:
        i = int(np.argmax(sizes))
        counts[i] += 1

    positions = []
    for (path, leaf), k in zip(leaves, counts):
        k = min(k, leaf.size)
        idx = rng.choice(leaf.size, size=k, replace=False)
        positions.append((tuple(_key_name(p) for p in path), idx))
    return positions


def make_extractor(positions):
    """Jitted function mapping a gradient tree to the (total_samples,) vector."""
    def _extract(tree):
        values = []
        for path, idx in positions:
            leaf = tree
            for key in path:
                leaf = leaf[key]
            values.append(jnp.reshape(leaf, (-1,))[idx])
        return jnp.concatenate(values)
    return jax.jit(_extract)


class GradSignalRecorder:
    """Collects the sampled gradient coordinates over a whole training run."""

    SAVE_EVERY = 10_000     # periodically rewrite the pickle as crash insurance

    def __init__(self, save_dir, params, training_steps,
                 total_samples=500, seed=42):
        self._save_dir = save_dir
        self._positions = sample_positions(params, total_samples, seed)
        self._extract = make_extractor(self._positions)
        self._n = sum(len(idx) for _, idx in self._positions)
        # the training loop executes steps 0 .. training_steps inclusive
        self._raw = np.zeros((self._n, training_steps + 1), dtype=np.float32)
        self._num_steps = training_steps + 1
        self._filtered = None
        self._filtered_start = None

    def record(self, step, raw_grads, filtered_grads):
        self._raw[:, step] = np.asarray(self._extract(raw_grads))
        if filtered_grads is not None:
            if self._filtered is None:
                self._filtered_start = step
                self._filtered = np.zeros(
                    (self._n, self._num_steps - step), dtype=np.float32)
            self._filtered[:, step - self._filtered_start] = np.asarray(
                self._extract(filtered_grads))
        if step > 0 and step % self.SAVE_EVERY == 0:
            self.save()

    def save(self):
        payload = {
            'raw': self._raw,
            'filtered': self._filtered,
            'raw_start_step': 0,
            'filtered_start_step': self._filtered_start,
            'positions': [(path, np.asarray(idx))
                          for path, idx in self._positions],
        }
        with open(os.path.join(self._save_dir, 'grad_signal.pkl'), 'wb') as f:
            pickle.dump(payload, f)
