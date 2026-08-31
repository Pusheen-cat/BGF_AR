"""MoleculeNet loading, scaffold splitting and on-disk caching.

The heavy / network-bound work (download + featurize + scaffold split via
DeepChem) happens once in :func:`build_cache`. Training and the per-seed loops
then read the cheap CSV cache through :func:`load_splits`.

Single-task datasets (BBBP, ESOL) cache a single ``label`` column and expose 1-D
labels. Multi-task datasets (HIV=1, Tox21=12, ClinTox=2 tasks) cache one
``t<k>`` column per task (NaN = missing label) plus a ``*.meta.json`` sidecar,
and expose 2-D ``[N, T]`` labels with a NaN mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess import canonical_smiles, clean_dataset
from .utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger(__name__)

# DeepChem MoleculeNet loader for each supported dataset key.
_DATASETS = {
    "bbbp": {"loader": "load_bbbp", "task_type": "classification", "multitask": False},
    "esol": {"loader": "load_delaney", "task_type": "regression", "multitask": False},
    "hiv": {"loader": "load_hiv", "task_type": "classification", "multitask": True},
    "tox21": {"loader": "load_tox21", "task_type": "classification", "multitask": True},
    "clintox": {"loader": "load_clintox", "task_type": "classification", "multitask": True},
}

SPLITS = ("train", "valid", "test")


@dataclass
class Split:
    smiles: list[str]
    labels: np.ndarray  # (N,) single-task or (N, T) multi-task (NaN = missing)

    def __len__(self) -> int:
        return len(self.smiles)


@dataclass
class DatasetBundle:
    name: str
    task_type: str
    train: Split
    valid: Split
    test: Split
    n_tasks: int = 1
    task_names: list[str] = field(default_factory=list)

    @property
    def multitask(self) -> bool:
        return self.n_tasks > 1 or self.train.labels.ndim == 2

    def split(self, which: str) -> Split:
        return getattr(self, which)


def cache_path(name: str, cache_dir: str | Path) -> Path:
    return Path(cache_dir) / f"{name}_scaffold.csv"


def _meta_path(name: str, cache_dir: str | Path) -> Path:
    return Path(cache_dir) / f"{name}_scaffold.meta.json"


def _load_molnet(name, splitter, cache_dir):
    import deepchem as dc

    loader = getattr(dc.molnet, _DATASETS[name]["loader"])
    dc_dir = str(Path(cache_dir) / f"_deepchem_{name}")
    try:
        return loader(featurizer="ECFP", splitter=splitter, transformers=[],
                      reload=True, data_dir=dc_dir, save_dir=dc_dir)
    except TypeError:
        logger.warning("Loaded %s with default transformers (fallback path)", name)
        return loader(featurizer="ECFP", splitter=splitter)


def build_cache(name, cache_dir="data", *, splitter="scaffold", isomeric=True,
                force=False) -> Path:
    """Download via DeepChem, scaffold-split, canonicalize and cache to CSV."""
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(_DATASETS)}")
    out = cache_path(name, cache_dir)
    if out.exists() and not force:
        logger.info("Cache already exists: %s", out)
        return out
    ensure_dir(cache_dir)

    multitask = _DATASETS[name]["multitask"]
    logger.info("Loading %s from MoleculeNet (splitter=%s, multitask=%s)...",
                name, splitter, multitask)
    tasks, datasets, _ = _load_molnet(name, splitter, cache_dir)
    train, valid, test = datasets

    frames = []
    for split_name, ds in zip(SPLITS, (train, valid, test)):
        smiles = [str(s) for s in ds.ids]
        if not multitask:
            labels = np.asarray(ds.y).reshape(len(smiles), -1)[:, 0]
            canon, lab, dropped = clean_dataset(smiles, labels, isomeric=isomeric)
            frames.append(pd.DataFrame({"smiles": canon, "label": lab, "split": split_name}))
        else:
            y = np.asarray(ds.y, dtype=float).reshape(len(smiles), -1)
            w = np.asarray(ds.w, dtype=float).reshape(len(smiles), -1)
            y = np.where(w == 0, np.nan, y)            # missing labels -> NaN
            canon, keep = [], []
            for i, s in enumerate(smiles):
                c = canonical_smiles(s, isomeric=isomeric)
                if c is not None:
                    canon.append(c)
                    keep.append(i)
            dropped = len(smiles) - len(canon)
            yk = y[keep]
            frame = pd.DataFrame({"smiles": canon, "split": split_name})
            for t in range(yk.shape[1]):
                frame[f"t{t}"] = yk[:, t]
            frames.append(frame)
        logger.info("  %-5s: %d molecules (%d dropped)", split_name, len(canon), dropped)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False)
    if multitask:
        save_json({"task_type": "classification", "n_tasks": len(tasks),
                   "task_names": [str(t) for t in tasks], "multitask": True},
                  _meta_path(name, cache_dir))
    logger.info("Wrote cache: %s (%d rows, %d tasks)", out, len(df), len(tasks) if multitask else 1)
    return out


def load_splits(name, cache_dir="data") -> DatasetBundle:
    """Read the cached CSV into a :class:`DatasetBundle` (building it if absent)."""
    out = cache_path(name, cache_dir)
    if not out.exists():
        logger.info("Cache missing for %s; building it now.", name)
        build_cache(name, cache_dir)
    df = pd.read_csv(out)

    meta_p = _meta_path(name, cache_dir)
    if meta_p.exists():
        meta = load_json(meta_p)
        n_tasks, task_names = meta["n_tasks"], meta["task_names"]
        task_type = meta["task_type"]
        task_cols = [f"t{t}" for t in range(n_tasks)]
    else:
        n_tasks, task_names = 1, [name]
        task_type = _DATASETS[name]["task_type"]
        task_cols = None

    parts = {}
    for split_name in SPLITS:
        sub = df[df["split"] == split_name]
        if task_cols is None:
            labels = sub["label"].to_numpy(dtype=np.float64)
        else:
            labels = sub[task_cols].to_numpy(dtype=np.float64)  # (N, T), NaN = missing
        parts[split_name] = Split(smiles=sub["smiles"].astype(str).tolist(), labels=labels)

    return DatasetBundle(
        name=name, task_type=task_type,
        train=parts["train"], valid=parts["valid"], test=parts["test"],
        n_tasks=n_tasks, task_names=task_names,
    )
