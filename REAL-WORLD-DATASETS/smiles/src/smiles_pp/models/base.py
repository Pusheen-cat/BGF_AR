"""Common model interface shared by all three model families."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np


class BaseModel(ABC):
    """Uniform ``fit / predict / save / load`` interface.

    ``predict`` always returns a 1-D float array:

    * classification -> probability of the positive class
    * regression     -> predicted value in the target's original units
    """

    task_type: str  # "classification" | "regression"

    @abstractmethod
    def fit(
        self,
        train_smiles: Sequence[str],
        train_labels: np.ndarray,
        valid_smiles: Sequence[str] | None = None,
        valid_labels: np.ndarray | None = None,
    ) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, smiles: Sequence[str]) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseModel":
        ...


def build_model(config, seed: int) -> BaseModel:
    """Factory dispatching on ``config.model_type``.

    This public release covers the ChemBERTa AdamW-vs-BGF comparison only, so the
    single supported ``model.type`` is ``chemberta`` (three pretrained checkpoints,
    selected via ``model.pretrained`` in the YAML config).
    """
    mtype = config.model_type
    if mtype == "chemberta":
        from .chemberta import ChemBERTaModel

        return ChemBERTaModel(config, seed)
    raise ValueError(f"Unknown model.type: {mtype!r} (this release supports 'chemberta')")


def load_model(config, path: str | Path) -> BaseModel:
    mtype = config.model_type
    if mtype == "chemberta":
        from .chemberta import ChemBERTaModel

        return ChemBERTaModel.load(path)
    raise ValueError(f"Unknown model.type: {mtype!r} (this release supports 'chemberta')")
