"""Vanilla (Elman) RNN sequence classifier.

    [B, 4, 1000]
      -> transpose to [B, 1000, 4]
      -> linear projection to proj_dim
      -> bidirectional nn.RNN
      -> mean (or last) pooling over time
      -> linear head -> raw logits [B, 919]
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        proj_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        n_outputs: int = 919,
        pooling: str = "mean",
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "last"):
            raise ValueError(f"pooling must be 'mean' or 'last', got {pooling}")
        self.pooling = pooling
        self.proj = nn.Linear(input_dim, proj_dim)
        self.rnn = nn.RNN(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, 1000] -> [B, 1000, 4]
        x = x.transpose(1, 2)
        x = self.proj(x)               # [B, T, proj_dim]
        out, _ = self.rnn(x)           # [B, T, out_dim]
        feat = out.mean(dim=1) if self.pooling == "mean" else out[:, -1, :]
        feat = self.dropout(feat)
        return self.head(feat)         # raw logits [B, n_outputs]
