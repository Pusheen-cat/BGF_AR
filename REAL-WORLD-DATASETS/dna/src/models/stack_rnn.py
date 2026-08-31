"""Stack-RNN sequence classifier (PyTorch port of the DeepMind Haiku core).

Port of ``StackRNNCore`` from Joulin & Mikolov (2015, https://arxiv.org/abs/1503.01007)
as used in "Neural Networks and the Chomsky Hierarchy" (DeepMind, 2024). A
differentiable stack is bolted onto a vanilla RNN cell: at every step the cell
reads the top of the stack, and its output emits (a) a push vector and (b) a
softmax over {PUSH, POP, NO-OP} that linearly mixes the next stack state.

The wrapper is intentionally identical in structure to ``rnn.py`` / ``lstm.py``
(input projection -> multi-layer, optionally bidirectional recurrent stack ->
pool -> dropout -> head); the ONLY difference from the plain RNN is the internal
recurrence, where each directional layer is a stack-augmented cell instead of a
bare ``nn.RNN``. This mirrors how ``nn.RNN(num_layers=L, bidirectional=True)``
stacks layers and concatenates the two directions.

    [B, 4, 1000]
      -> transpose to [B, 1000, 4]
      -> linear projection to proj_dim
      -> num_layers x (fwd stack-RNN [+ bwd stack-RNN]) with inter-layer dropout
      -> mean (or last) pooling
      -> linear head -> raw logits [B, 919]

The stack update is a faithful, batched (over ``n_stacks``) translation of the JAX
``_update_stack``; the JAX ``jax.vmap`` over the stack axis becomes an explicit
leading ``n_stacks`` dimension here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# PUSH, POP, NO_OP
_NUM_ACTIONS = 3


def _update_stack(stack: torch.Tensor, actions: torch.Tensor,
                  push_value: torch.Tensor) -> torch.Tensor:
    """Differentiable stack update.

    Args:
        stack: current stack, ``[B, n_stacks, stack_size, cell]``.
        actions: action probabilities ``[B, n_stacks, 3]`` (PUSH, POP, NO_OP).
        push_value: vector to push ``[B, n_stacks, cell]``.

    Returns:
        the new stack, same shape as ``stack``.
    """
    # --- top of the stack (slot 0) ---
    push_a = actions[..., 0:1]                 # [B, N, 1] -> broadcast over cell
    pop_a = actions[..., 1:2]
    noop_a = actions[..., 2:3]
    pop_value_top = stack[:, :, 1, :]          # element below the top
    noop_value_top = stack[:, :, 0, :]         # current top
    top_new = push_a * push_value + pop_a * pop_value_top + noop_a * noop_value_top
    top_new = top_new.unsqueeze(2)             # [B, N, 1, cell]

    # --- rest of the stack (slots 1..stack_size-1) ---
    push_a_r = actions[..., 0].unsqueeze(-1).unsqueeze(-1)   # [B, N, 1, 1]
    pop_a_r = actions[..., 1].unsqueeze(-1).unsqueeze(-1)
    noop_a_r = actions[..., 2].unsqueeze(-1).unsqueeze(-1)
    push_value_r = stack[:, :, :-1, :]                        # move elements down
    zeros = stack.new_zeros(stack.shape[0], stack.shape[1], 1, stack.shape[3])
    pop_value_r = torch.cat([stack[:, :, 2:, :], zeros], dim=2)  # move elements up
    noop_value_r = stack[:, :, 1:, :]                        # unchanged
    rest_new = push_a_r * push_value_r + pop_a_r * pop_value_r + noop_a_r * noop_value_r

    return torch.cat([top_new, rest_new], dim=2)


class _StackRNNLayer(nn.Module):
    """One directional stack-augmented RNN layer: ``[B, T, in] -> [B, T, hidden]``."""

    def __init__(self, input_size: int, hidden_size: int, stack_cell_size: int,
                 stack_size: int, n_stacks: int, nonlinearity: str) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.stack_cell_size = stack_cell_size
        self.stack_size = stack_size
        self.n_stacks = n_stacks
        self.cell = nn.RNNCell(input_size + n_stacks * stack_cell_size,
                               hidden_size, nonlinearity=nonlinearity)
        self.push = nn.Linear(hidden_size, n_stacks * stack_cell_size)
        self.action = nn.Linear(hidden_size, n_stacks * _NUM_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_size)
        stacks = x.new_zeros(B, self.n_stacks, self.stack_size, self.stack_cell_size)
        outs = []
        for t in range(T):
            top = stacks[:, :, 0, :].reshape(B, self.n_stacks * self.stack_cell_size)
            inp = torch.cat([x[:, t, :], top], dim=-1)
            h = self.cell(inp, h)          # vanilla RNN: output == state == h
            push_values = self.push(h).reshape(B, self.n_stacks, self.stack_cell_size)
            actions = torch.softmax(
                self.action(h).reshape(B, self.n_stacks, _NUM_ACTIONS), dim=-1)
            stacks = _update_stack(stacks, actions, push_values)
            outs.append(h)
        return torch.stack(outs, dim=1)    # [B, T, hidden]


class StackRNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        proj_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        stack_cell_size: int = 16,
        stack_size: int = 30,
        n_stacks: int = 1,
        inner_nonlinearity: str = "tanh",   # DeepMind default is "relu"; tanh is
                                            # the stable choice over 1000 steps.
        dropout: float = 0.3,
        n_outputs: int = 919,
        pooling: str = "mean",
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
                _StackRNNLayer(in_size, hidden_size, stack_cell_size,
                               stack_size, n_stacks, inner_nonlinearity)
                for _ in range(self.num_directions)
            ])
            self.layers.append(directions)
            in_size = hidden_size * self.num_directions

        # inter-layer dropout, matching nn.RNN(dropout=...) semantics
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
