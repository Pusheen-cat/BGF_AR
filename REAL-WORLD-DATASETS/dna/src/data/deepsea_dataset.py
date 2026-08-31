"""HDF5-backed PyTorch Dataset for the DeepSEA-style data.

The processed file (see ``scripts/prepare_hdf5.py``) has the layout::

    train/X  float32 [N, 4, 1000]     train/y  float32 [N, 919]
    valid/X  float32 [N, 4, 1000]     valid/y  float32 [N, 919]
    test/X   float32 [N, 4, 1000]     test/y   float32 [N, 919]

The file handle is opened lazily *per worker* so the Dataset is safe to use with
``num_workers > 0`` (h5py handles cannot be shared across forked processes).
"""
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DeepSEADataset(Dataset):
    """Reads ``<split>/X`` and ``<split>/y`` from an HDF5 file on demand.

    Parameters
    ----------
    h5_path : str
        Path to the processed HDF5 file.
    split : {"train", "valid", "test"}
    reverse_complement_augmentation : bool
        If True (only meaningful for training), each item is independently
        replaced by its reverse complement with probability 0.5.
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        reverse_complement_augmentation: bool = False,
    ) -> None:
        super().__init__()
        self.h5_path = str(h5_path)
        self.split = split
        self.rc_aug = bool(reverse_complement_augmentation)
        self._h5: h5py.File | None = None

        # Read shapes once up front (cheap, no data loaded) and then close.
        with h5py.File(self.h5_path, "r") as f:
            if split not in f:
                raise KeyError(
                    f"split '{split}' not found in {self.h5_path}; "
                    f"available groups: {list(f.keys())}"
                )
            self._n = int(f[f"{split}/X"].shape[0])
            self.x_shape = tuple(f[f"{split}/X"].shape[1:])
            self.y_shape = tuple(f[f"{split}/y"].shape[1:])
        self.n_outputs = int(self.y_shape[0])

    # -- lazy per-worker file handle -------------------------------------
    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        f = self._file()
        x = np.asarray(f[f"{self.split}/X"][idx], dtype=np.float32)  # [4, 1000]
        y = np.asarray(f[f"{self.split}/y"][idx], dtype=np.float32)  # [919]
        if self.rc_aug and np.random.rand() < 0.5:
            # reverse complement = flip channel axis + flip length axis
            x = x[::-1, ::-1].copy()
        return torch.from_numpy(x), torch.from_numpy(y)

    # h5py handles are not picklable; drop the handle on pickling so that
    # DataLoader workers reopen the file themselves.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def positive_counts(self) -> np.ndarray:
        """Return per-label positive counts over this split (for pos_weight)."""
        with h5py.File(self.h5_path, "r") as f:
            y = f[f"{self.split}/y"]
            counts = np.zeros(self.n_outputs, dtype=np.float64)
            step = 100_000
            for start in range(0, y.shape[0], step):
                counts += np.asarray(y[start : start + step]).sum(axis=0)
        return counts


def make_dataloaders(
    h5_path: str,
    batch_size: int = 128,
    num_workers: int = 4,
    reverse_complement_augmentation: bool = True,
    splits=("train", "valid", "test"),
    pin_memory: bool | None = None,
):
    """Build a dict of DataLoaders for the requested splits.

    Only the training split shuffles and receives RC augmentation.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loaders = {}
    for split in splits:
        try:
            ds = DeepSEADataset(
                h5_path,
                split=split,
                reverse_complement_augmentation=(
                    reverse_complement_augmentation and split == "train"
                ),
            )
        except KeyError:
            # A split may legitimately be absent (e.g. debug files may still
            # have all three, but be defensive).
            continue
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=(num_workers > 0),
        )
    return loaders
