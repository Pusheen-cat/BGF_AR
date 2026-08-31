"""Robust readers for DeepSEA ``.mat`` files.

The bundle mixes two on-disk formats:

* ``train.mat`` is MATLAB v7.3 == HDF5 -> read with **h5py**.  h5py returns
  arrays with the axis order *reversed* relative to MATLAB, so ``trainxdata``
  typically appears as ``[1000, 4, N]`` and ``traindata`` as ``[919, N]``.
* ``valid.mat`` / ``test.mat`` are the older MATLAB v5 format -> read with
  **scipy.io.loadmat**, giving ``[N, 4, 1000]`` and ``[N, 919]``.

Rather than hard-code any of this, we detect the sequence axes (the one of size
4 = channels, the one of size 1000 = length) and the label axis (size 919), and
normalise everything to ``X:[N,4,1000]`` / ``y:[N,919]``.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

SEQ_LEN = 1000
N_CHANNELS = 4
N_LABELS = 919


# --------------------------------------------------------------------------- #
# format detection / lazy handles
# --------------------------------------------------------------------------- #
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def is_hdf5(path: str) -> bool:
    """True if the file is HDF5 (covers MATLAB v7.3).

    MATLAB v7.3 files are HDF5 with a 512-byte descriptive user-block prepended,
    so the 8-byte HDF5 signature appears at offset 512 rather than 0.  HDF5
    user blocks are always at an offset that is a power of two >= 512, so we
    probe offset 0 and those boundaries.
    """
    try:
        with open(path, "rb") as fh:
            for off in (0, 512, 1024, 2048, 4096, 8192):
                fh.seek(off)
                if fh.read(8) == _HDF5_MAGIC:
                    return True
    except OSError:
        return False
    return False


class MatArray:
    """Uniform lazy wrapper around one array inside a .mat file.

    Exposes ``.shape``, ``.dtype`` and ``read(sl)`` where ``sl`` slices the
    FIRST axis of the *original* on-disk array.  Supports numpy arrays (scipy)
    and h5py datasets alike.
    """

    def __init__(self, obj: Any):
        self._obj = obj
        self.shape = tuple(int(s) for s in obj.shape)
        self.dtype = obj.dtype

    def read(self, sl: slice | None = None) -> np.ndarray:
        if sl is None:
            return np.asarray(self._obj[:])
        return np.asarray(self._obj[sl])


class MatFile:
    """Open a .mat file (either format) and expose its arrays by key."""

    def __init__(self, path: str):
        self.path = path
        self._h5 = None
        self._scipy = None
        if is_hdf5(path):
            import h5py

            self._h5 = h5py.File(path, "r")
            self._keys = [
                k for k in self._h5.keys() if not k.startswith("#")
            ]
        else:
            from scipy.io import loadmat

            self._scipy = loadmat(path)
            self._keys = [k for k in self._scipy.keys() if not k.startswith("__")]

    @property
    def format(self) -> str:
        return "hdf5(v7.3)" if self._h5 is not None else "matlab(v5)"

    def keys(self) -> list[str]:
        return list(self._keys)

    def array(self, key: str) -> MatArray:
        if self._h5 is not None:
            return MatArray(self._h5[key])
        return MatArray(self._scipy[key])

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
# array-role identification
# --------------------------------------------------------------------------- #
def looks_like_sequence(shape: tuple[int, ...]) -> bool:
    return len(shape) == 3 and (N_CHANNELS in shape) and (SEQ_LEN in shape)


def looks_like_labels(shape: tuple[int, ...]) -> bool:
    return len(shape) == 2 and (N_LABELS in shape)


def find_roles(mat: MatFile) -> dict[str, str]:
    """Guess which key is the sequence array and which is the label array."""
    roles: dict[str, str] = {}
    seq_key = lbl_key = None
    for k in mat.keys():
        shape = mat.array(k).shape
        if looks_like_sequence(shape):
            # prefer the largest such array (train has millions of rows)
            if seq_key is None or _n_elems(mat.array(k).shape) > _n_elems(
                mat.array(seq_key).shape
            ):
                seq_key = k
        elif looks_like_labels(shape):
            if lbl_key is None or _n_elems(mat.array(k).shape) > _n_elems(
                mat.array(lbl_key).shape
            ):
                lbl_key = k
    # fallback: match on substrings of common DeepSEA key names
    if seq_key is None:
        seq_key = _guess_by_name(mat, ("xdata", "seq", "x"))
    if lbl_key is None:
        lbl_key = _guess_by_name(mat, ("data", "label", "y"), exclude=seq_key)
    if seq_key:
        roles[seq_key] = "sequence"
    if lbl_key:
        roles[lbl_key] = "labels"
    return roles


def _guess_by_name(mat: MatFile, needles, exclude=None):
    for k in mat.keys():
        if k == exclude:
            continue
        kl = k.lower()
        if any(nd in kl for nd in needles):
            return k
    return None


def _n_elems(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


# --------------------------------------------------------------------------- #
# axis normalisation
# --------------------------------------------------------------------------- #
def seq_axis_map(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Return ``(n_axis, c_axis, l_axis)`` for a 3-D sequence array."""
    if len(shape) != 3:
        raise ValueError(f"sequence array must be 3-D, got {shape}")
    dims = list(shape)
    c_axis = dims.index(N_CHANNELS)
    # length axis: a dim equal to SEQ_LEN that is not the channel axis
    l_axis = next(i for i, d in enumerate(dims) if d == SEQ_LEN and i != c_axis)
    n_axis = ({0, 1, 2} - {c_axis, l_axis}).pop()
    return n_axis, c_axis, l_axis


def label_axis_map(shape: tuple[int, ...]) -> tuple[int, int]:
    """Return ``(n_axis, label_axis)`` for a 2-D label array."""
    if len(shape) != 2:
        raise ValueError(f"label array must be 2-D, got {shape}")
    label_axis = list(shape).index(N_LABELS)
    n_axis = 1 - label_axis
    return n_axis, label_axis


def to_ncl(chunk: np.ndarray, n_axis: int, c_axis: int, l_axis: int) -> np.ndarray:
    """Move a (sliced-along-n) chunk to ``[n, 4, 1000]`` float32."""
    x = np.moveaxis(chunk, (n_axis, c_axis, l_axis), (0, 1, 2))
    return np.ascontiguousarray(x, dtype=np.float32)


def to_nl(chunk: np.ndarray, n_axis: int, label_axis: int) -> np.ndarray:
    """Move a (sliced-along-n) label chunk to ``[n, 919]`` float32."""
    y = np.moveaxis(chunk, (n_axis, label_axis), (0, 1))
    return np.ascontiguousarray(y, dtype=np.float32)


def n_samples(shape: tuple[int, ...], n_axis: int) -> int:
    return int(shape[n_axis])
