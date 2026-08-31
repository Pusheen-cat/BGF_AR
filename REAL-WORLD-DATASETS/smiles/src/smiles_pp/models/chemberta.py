"""Fine-tuned HuggingFace ChemBERTa encoder (DeepChem/ChemBERTa-77M-MTR)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..config import Config
from ..multitask import predict_multitask, train_multitask
from ..utils import ensure_dir, get_device, get_logger, load_json, save_json
from ._torch_trainer import (
    TargetScaler,
    TrainConfig,
    build_monitor_sets,
    predict_scores,
    train_loop,
)
from .base import BaseModel

logger = get_logger(__name__)


class _HFDataset(Dataset):
    def __init__(self, smiles: Sequence[str], labels):
        self.smiles = list(smiles)
        self.labels = np.asarray(labels, dtype=np.float32) if labels is not None \
            else np.zeros(len(self.smiles), dtype=np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]


def _make_collate(tokenizer, max_length: int):
    def collate(batch):
        smis, labels = zip(*batch)
        enc = tokenizer(
            list(smis),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        return inputs, torch.tensor(labels, dtype=torch.float32)

    return collate


class _HFWrapper(nn.Module):
    """Thin wrapper exposing a single logit per molecule for the shared trainer."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids, attention_mask):
        out = self.hf_model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits.squeeze(-1)


class ChemBERTaModel(BaseModel):
    def __init__(self, config: Config, seed: int = 0):
        self.config = config
        self.task_type = config.task_type
        self.seed = seed
        m = config.model
        self.pretrained = m.get("pretrained", "DeepChem/ChemBERTa-77M-MTR")
        self.max_length = int(config.data.get("max_length", 256))
        self.dropout = float(m.get("dropout", 0.1))
        self.device = get_device(config.train.get("device"))
        self.tokenizer = None
        self.module: _HFWrapper | None = None
        self.scaler: TargetScaler | None = None
        self.n_tasks = 1
        self.multitask = False

    def _build(self, num_labels=1):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.pretrained)
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained,
            num_labels=num_labels,
            hidden_dropout_prob=self.dropout,
            ignore_mismatched_sizes=True,
        )
        return tokenizer, _HFWrapper(hf_model)

    # -- multi-task plumbing (used by smiles_pp.multitask.train_multitask) -----
    def build_loader(self, smiles, labels, batch_size, shuffle):
        from torch.utils.data import DataLoader

        ds = _HFDataset(smiles, labels)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self._mt_collate)

    def _mt_collate(self, batch):
        smis, labs = zip(*batch)
        enc = self.tokenizer(list(smis), padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        y = torch.tensor(np.stack(labs), dtype=torch.float32)  # [B, T] (may hold NaN)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}, y

    def forward_batch(self, batch, device):
        inputs, y = batch
        logits = self.module.hf_model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
        ).logits  # [B, T]
        y = y.to(device)
        mask = (~torch.isnan(y)).float()
        return logits, torch.nan_to_num(y, nan=0.0), mask

    def fit(self, train_smiles, train_labels, valid_smiles=None, valid_labels=None,
            study_eval=None):
        cfg = TrainConfig.from_dict(self.config.train)
        self.multitask = np.asarray(train_labels).ndim == 2
        if self.multitask:
            self.n_tasks = int(np.asarray(train_labels).shape[1])
            self.tokenizer, self.module = self._build(num_labels=self.n_tasks)
            logger.info("ChemBERTa(multitask, %d tasks): %s, device=%s",
                        self.n_tasks, self.pretrained, self.device)
            splits = {"train": (train_smiles, train_labels),
                      "valid": (valid_smiles, valid_labels)}
            if study_eval and "test" in study_eval:
                splits["test"] = study_eval["test"]
            info = train_multitask(self, splits, cfg=cfg, device=self.device,
                                   study=study_eval is not None)
            self.history_ = info["history"]
            logger.info("ChemBERTa best val roc_auc=%.4f @ epoch %d",
                        info["best_metric"], info["best_epoch"])
            return self

        # single-task path (BBBP / ESOL)
        self.tokenizer, self.module = self._build(num_labels=1)
        logger.info("ChemBERTa: %s, device=%s", self.pretrained, self.device)
        self.scaler = (
            TargetScaler.fit(train_labels) if self.task_type == "regression" else None
        )
        train_set = _HFDataset(train_smiles, train_labels)
        valid_set = _HFDataset(valid_smiles, valid_labels) if valid_smiles is not None else None
        collate = _make_collate(self.tokenizer, self.max_length)
        monitor_sets, early_stop = build_monitor_sets(
            lambda s, l: _HFDataset(s, l), study_eval)
        info = train_loop(
            self.module, train_set, valid_set, collate,
            task_type=self.task_type, cfg=cfg, device=self.device, scaler=self.scaler,
            monitor_sets=monitor_sets, early_stop=early_stop,
        )
        self.history_ = info["history"]
        logger.info("ChemBERTa best val metric=%.4f @ epoch %d", info["best_metric"], info["best_epoch"])
        return self

    def predict(self, smiles):
        if self.multitask:
            return predict_multitask(self, smiles, 64, self.device)
        dataset = _HFDataset(smiles, None)
        collate = _make_collate(self.tokenizer, self.max_length)
        from ._torch_trainer import _make_loader

        loader = _make_loader(dataset, collate, 64, False, 0)
        return predict_scores(self.module, loader, self.task_type, self.device, self.scaler)

    def save(self, path):
        path = ensure_dir(path)
        self.module.hf_model.save_pretrained(path / "hf_model")
        self.tokenizer.save_pretrained(path / "hf_model")
        save_json(
            {
                "task_type": self.task_type,
                "pretrained": self.pretrained,
                "max_length": self.max_length,
                "dropout": self.dropout,
                "multitask": self.multitask,
                "n_tasks": self.n_tasks,
                "scaler": self.scaler.to_dict() if self.scaler else None,
            },
            path / "meta.json",
        )

    @classmethod
    def load(cls, path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        meta = load_json(path / "meta.json")
        obj = cls.__new__(cls)
        obj.task_type = meta["task_type"]
        obj.pretrained = meta["pretrained"]
        obj.max_length = meta["max_length"]
        obj.dropout = meta["dropout"]
        obj.multitask = meta.get("multitask", False)
        obj.n_tasks = meta.get("n_tasks", 1)
        obj.scaler = TargetScaler(**meta["scaler"]) if meta["scaler"] else None
        obj.device = get_device()
        obj.tokenizer = AutoTokenizer.from_pretrained(path / "hf_model")
        hf_model = AutoModelForSequenceClassification.from_pretrained(path / "hf_model")
        obj.module = _HFWrapper(hf_model).to(obj.device)
        return obj
