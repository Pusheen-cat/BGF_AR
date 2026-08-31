"""One-hot encoding / decoding utilities for DNA sequences.

The DeepSEA bundle already ships *one-hot* encoded sequences, so training and
evaluation never call these helpers.  They are used by ``prepare_hdf5.py`` (only
if a raw ``.mat`` turns out to be string / integer encoded) and by
``predict.py`` (to encode a FASTA/plain-text sequence at inference time).

Channel convention
------------------
We use ``BASES = "ACGT"`` (index 0..3).  DeepSEA's internal channel order is
documented as ``A, G, C, T``; because we consume the *pre-encoded* one-hot
arrays directly for training, the exact ordering is irrelevant to the model.
It only matters when encoding raw text here, so keep the same ``BASES`` string
end-to-end if you feed raw sequences.

Reverse-complement note
------------------------
For BOTH the ``ACGT`` and ``AGCT`` channel orderings, the base-complement
permutation is exactly the reverse of the channel axis (A<->T, C<->G / G<->C).
Hence the reverse complement of a ``[4, L]`` one-hot tensor is simply a flip
along *both* the channel axis and the length axis -- order agnostic.
"""
from __future__ import annotations

import numpy as np

BASES = "ACGT"
_BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
# lowercase + common ambiguity code N handled as all-zero column
for _b, _i in list(_BASE_TO_IDX.items()):
    _BASE_TO_IDX[_b.lower()] = _i


def one_hot_encode(seq: str, length: int | None = None) -> np.ndarray:
    """Encode a DNA string into a ``[4, L]`` float32 one-hot array.

    Unknown characters (``N``, ambiguity codes, padding) become all-zero
    columns.  If ``length`` is given the sequence is truncated / zero-padded on
    the right to exactly ``length`` columns.
    """
    if length is None:
        length = len(seq)
    x = np.zeros((4, length), dtype=np.float32)
    for j, ch in enumerate(seq[:length]):
        idx = _BASE_TO_IDX.get(ch)
        if idx is not None:
            x[idx, j] = 1.0
    return x


def one_hot_decode(x: np.ndarray) -> str:
    """Inverse of :func:`one_hot_encode`. All-zero columns decode to ``N``."""
    x = np.asarray(x)
    if x.shape[0] != 4:
        raise ValueError(f"expected [4, L] one-hot, got shape {x.shape}")
    idx = x.argmax(axis=0)
    has = x.sum(axis=0) > 0
    return "".join(BASES[i] if h else "N" for i, h in zip(idx, has))


def integer_encode_to_onehot(codes: np.ndarray, length: int | None = None) -> np.ndarray:
    """Convert integer-encoded sequence(s) (0..3, other=gap) to one-hot ``[4, L]``.

    ``codes`` may be shape ``[L]`` (single) or ``[N, L]`` (batch); returns
    ``[4, L]`` or ``[N, 4, L]`` respectively.
    """
    codes = np.asarray(codes)
    single = codes.ndim == 1
    if single:
        codes = codes[None]
    n, L = codes.shape
    if length is None:
        length = L
    out = np.zeros((n, 4, length), dtype=np.float32)
    for i in range(n):
        for j in range(min(L, length)):
            c = int(codes[i, j])
            if 0 <= c < 4:
                out[i, c, j] = 1.0
    return out[0] if single else out


def reverse_complement_onehot(x: np.ndarray) -> np.ndarray:
    """Reverse complement of a ``[4, L]`` (or batched ``[N, 4, L]``) one-hot.

    Implemented as a flip along the channel axis and the length axis.  Works for
    numpy arrays; a torch-tensor variant lives in ``deepsea_dataset`` for the
    augmentation path.
    """
    x = np.asarray(x)
    if x.ndim == 2:
        return x[::-1, ::-1].copy()
    if x.ndim == 3:
        return x[:, ::-1, ::-1].copy()
    raise ValueError(f"expected [4, L] or [N, 4, L], got shape {x.shape}")
