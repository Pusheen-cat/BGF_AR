"""Model registry + config-driven factory.

Every model consumes ``[B, 4, 1000]`` and returns raw logits ``[B, n_outputs]``.
"""
from __future__ import annotations

from .lstm import LSTMClassifier
from .rnn import RNNClassifier
from .stack_rnn import StackRNNClassifier
from .tape_rnn import TapeRNNClassifier

MODEL_TYPES = {
    "rnn": RNNClassifier,
    "lstm": LSTMClassifier,
    "stack_rnn": StackRNNClassifier,
    "tape_rnn": TapeRNNClassifier,
}

# which model kwargs each type accepts (everything else in the config's `model`
# block is ignored, so a single YAML schema can carry extra keys harmlessly)
_ALLOWED = {
    "rnn": {"input_dim", "proj_dim", "hidden_size", "num_layers",
            "bidirectional", "dropout", "pooling", "nonlinearity"},
    "lstm": {"input_dim", "proj_dim", "hidden_size", "num_layers",
             "bidirectional", "dropout", "pooling"},
    "stack_rnn": {"input_dim", "proj_dim", "hidden_size", "num_layers",
                  "bidirectional", "stack_cell_size", "stack_size", "n_stacks",
                  "inner_nonlinearity", "dropout", "pooling"},
    "tape_rnn": {"input_dim", "proj_dim", "hidden_size", "num_layers",
                 "bidirectional", "memory_cell_size", "memory_size", "n_tapes",
                 "mlp_layers_size", "inner_nonlinearity", "dropout", "pooling",
                 "input_length"},
}


def build_model(config: dict):
    """Instantiate a model from a parsed config dict.

    Expects ``config['model']['type']`` in {rnn, lstm, stack_rnn, tape_rnn} and
    ``config['n_outputs']`` (defaults to 919).
    """
    model_cfg = dict(config.get("model", {}))
    mtype = model_cfg.pop("type", None)
    if mtype not in MODEL_TYPES:
        raise ValueError(
            f"unknown model type {mtype!r}; choose from {list(MODEL_TYPES)}"
        )
    n_outputs = int(config.get("n_outputs", 919))
    kwargs = {k: v for k, v in model_cfg.items() if k in _ALLOWED[mtype]}
    return MODEL_TYPES[mtype](n_outputs=n_outputs, **kwargs)


__all__ = [
    "RNNClassifier",
    "LSTMClassifier",
    "StackRNNClassifier",
    "TapeRNNClassifier",
    "MODEL_TYPES",
    "build_model",
]
