"""Typed configuration loaded from YAML files."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Container for a single experiment configuration.

    The nested sections (``task``, ``data``, ``model``, ``train``, ``output``)
    are kept as plain dicts so that each model type can carry its own
    hyper-parameters without forcing a rigid schema.
    """

    name: str
    task: dict[str, Any]
    data: dict[str, Any]
    model: dict[str, Any]
    train: dict[str, Any] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    output: dict[str, Any] = field(default_factory=dict)
    path: str | None = None

    # -- convenience accessors -------------------------------------------------
    @property
    def task_type(self) -> str:
        """``'classification'`` or ``'regression'``."""
        return self.task["type"]

    @property
    def dataset(self) -> str:
        return self.data["dataset"]

    @property
    def model_type(self) -> str:
        return self.model["type"]

    @property
    def results_dir(self) -> Path:
        return Path(self.output.get("results_dir", "results"))

    @property
    def ckpt_dir(self) -> Path:
        return Path(self.output.get("ckpt_dir", "checkpoints")) / self.name

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        _validate(raw, path)
        return cls(
            name=raw.get("name", path.stem),
            task=raw["task"],
            data=raw["data"],
            model=raw["model"],
            train=raw.get("train", {}),
            seeds=list(raw.get("seeds", [0, 1, 2])),
            output=raw.get("output", {}),
            path=str(path),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "data": self.data,
            "model": self.model,
            "train": self.train,
            "seeds": self.seeds,
            "output": self.output,
        }


def _validate(raw: dict, path: Path) -> None:
    for key in ("task", "data", "model"):
        if key not in raw:
            raise ValueError(f"{path}: missing required top-level section '{key}'")
    if raw["task"].get("type") not in ("classification", "regression"):
        raise ValueError(
            f"{path}: task.type must be 'classification' or 'regression', "
            f"got {raw['task'].get('type')!r}"
        )
    if "dataset" not in raw["data"]:
        raise ValueError(f"{path}: data.dataset is required")
    if "type" not in raw["model"]:
        raise ValueError(f"{path}: model.type is required")
