"""Tape-RNN sequence classifier (PyTorch port of the DeepMind Haiku core).

Port of ``TapeRNNCore`` / ``TapeInputLengthJumpCore`` from "Neural Networks and
the Chomsky Hierarchy" (DeepMind, 2024). A vanilla RNN cell is augmented with a
differentiable tape (memory): at every step the cell reads the current tape cell,
writes a value to it (via a small MLP), and emits a softmax over head-movement
actions that linearly mixes the shifted tapes.

``TapeInputLengthJumpCore`` has 5 actions:
    write & stay, write & move left, write & move right,
    write & jump ``input_length`` left, write & jump ``input_length`` right.

The wrapper is intentionally identical in structure to ``rnn.py`` / ``lstm.py``
(input projection -> multi-layer, optionally bidirectional recurrent stack ->
pool -> dropout -> head); the ONLY difference from the plain RNN is the internal
recurrence, where each directional layer is a tape-augmented cell.

    [B, 4, 1000]
      -> transpose to [B, 1000, 4]
      -> linear projection to proj_dim
      -> num_layers x (fwd tape-RNN [+ bwd tape-RNN]) with inter-layer dropout
      -> mean (or last) pooling
      -> linear head -> raw logits [B, 919]

Note on the tape update: the JAX code builds permutation matrices ``roll(eye, s)``
and applies them with ``einsum('mM,bMc->bmc', ...)``. Left-multiplying by
``roll(eye, s, axis=0)`` is exactly ``torch.roll(mem, shifts=s, dims=<mem axis>)``,
so we use ``torch.roll`` directly (same result, far cheaper). ``jax.vmap`` over the
tape axis becomes an explicit ``n_tapes`` dimension.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

# TapeInputLengthJumpCore action order: [stay, left, right, jump_left, jump_right]
_NUM_ACTIONS = 5


def _tape_shifts(input_length: int) -> list[int]:
    """Head-movement shifts, in the same order as the softmax action logits."""
    return [0, -1, 1, -input_length, input_length]


def _update_memory(memory: torch.Tensor, actions: torch.Tensor,
                   write_values: torch.Tensor, input_length: int) -> torch.Tensor:
    """Differentiable tape update.

    Args:
        memory: current tape, ``[B, n_tapes, memory_size, cell]``.
        actions: action probabilities ``[B, n_tapes, 5]``.
        write_values: values written to the current cell ``[B, n_tapes, cell]``.
        input_length: sequence length used by the jump actions.

    Returns:
        the new tape, same shape as ``memory``.
    """
    # write into the current (slot 0) cell, keep the rest
    memory_with_write = torch.cat(
        [write_values.unsqueeze(2), memory[:, :, 1:, :]], dim=2)   # [B, N, M, cell]

    # each action shifts the (written) tape along the memory axis
    ops = [torch.roll(memory_with_write, shifts=s, dims=2)
           for s in _tape_shifts(input_length)]
    ops = torch.stack(ops, dim=0)                                  # [A, B, N, M, cell]

    # action-probability-weighted combination of the shifted tapes
    return torch.einsum("abnmc,bna->bnmc", ops, actions)


def _build_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    """hk.nets.MLP equivalent: ReLU between layers, no activation on the output."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class _TapeRNNLayer(nn.Module):
    """One directional tape-augmented RNN layer: ``[B, T, in] -> [B, T, hidden]``."""

    def __init__(self, input_size: int, hidden_size: int, memory_cell_size: int,
                 memory_size: int, n_tapes: int, mlp_layers_size: Sequence[int],
                 nonlinearity: str, input_length: int | None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_cell_size = memory_cell_size
        self.memory_size = memory_size
        self.n_tapes = n_tapes
        self.input_length = input_length
        self.cell = nn.RNNCell(input_size + n_tapes * memory_cell_size,
                               hidden_size, nonlinearity=nonlinearity)
        self.readout = _build_mlp(hidden_size, tuple(mlp_layers_size),
                                  n_tapes * memory_cell_size)
        self.action_heads = nn.ModuleList(
            [nn.Linear(hidden_size, _NUM_ACTIONS) for _ in range(n_tapes)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        input_length = self.input_length if self.input_length is not None else T
        h = x.new_zeros(B, self.hidden_size)
        memory = x.new_zeros(B, self.n_tapes, self.memory_size, self.memory_cell_size)
        outs = []
        for t in range(T):
            current = memory[:, :, 0, :].reshape(
                B, self.n_tapes * self.memory_cell_size)
            inp = torch.cat([x[:, t, :], current], dim=-1)
            h = self.cell(inp, h)          # vanilla RNN: output == state == h
            write_values = self.readout(h).reshape(
                B, self.n_tapes, self.memory_cell_size)
            actions = torch.stack(
                [torch.softmax(head(h), dim=-1) for head in self.action_heads],
                dim=1)                     # [B, n_tapes, 5]
            memory = _update_memory(memory, actions, write_values, input_length)
            outs.append(h)
        return torch.stack(outs, dim=1)    # [B, T, hidden]


class TapeRNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        proj_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        memory_cell_size: int = 16,
        memory_size: int = 30,
        n_tapes: int = 1,
        mlp_layers_size: Sequence[int] = (64, 64),
        inner_nonlinearity: str = "tanh",   # DeepMind default is "relu"; tanh is
                                            # the stable choice over 1000 steps.
        dropout: float = 0.3,
        n_outputs: int = 919,
        pooling: str = "mean",
        input_length: int | None = None,    # jump distance; defaults to seq len T
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "last"):
            raise ValueError(f"pooling must be 'mean' or 'last', got {pooling}")
        if inner_nonlinearity not in ("tanh", "relu"):
            raise ValueError("inner_nonlinearity must be 'tanh' or 'relu'")
        self.pooling = pooling
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.proj = nn.Linear(input_dim, proj_dim)
        self.layers = nn.ModuleList()
        in_size = proj_dim
        for _ in range(num_layers):
            directions = nn.ModuleList([
                _TapeRNNLayer(in_size, hidden_size, memory_cell_size, memory_size,
                              n_tapes, mlp_layers_size, inner_nonlinearity,
                              input_length)
                for _ in range(self.num_directions)
            ])
            self.layers.append(directions)
            in_size = hidden_size * self.num_directions

        self.inter_dropout = (nn.Dropout(dropout)
                              if (num_layers > 1 and dropout > 0) else None)
        out_dim = hidden_size * self.num_directions
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)              # [B, 4, T] -> [B, T, 4]
        x = self.proj(x)                   # [B, T, proj_dim]
        for li, directions in enumerate(self.layers):
            if self.bidirectional:
                fwd = directions[0](x)
                bwd = directions[1](x.flip(1)).flip(1)
                x = torch.cat([fwd, bwd], dim=-1)
            else:
                x = directions[0](x)
            if self.inter_dropout is not None and li < self.num_layers - 1:
                x = self.inter_dropout(x)
        feat = x.mean(dim=1) if self.pooling == "mean" else x[:, -1, :]
        feat = self.dropout(feat)
        return self.head(feat)             # raw logits [B, n_outputs]
