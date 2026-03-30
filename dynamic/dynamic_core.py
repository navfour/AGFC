"""Core hyperparameter utilities for dynamic FC-GAGA experiments."""

from __future__ import annotations

from typing import Any, Dict, NamedTuple

import tensorflow as tf

import os


hyperparams_defaults = {
    "dataset": "metr-la",
    "repeat": list(range(20)),
    "epochs": [200],
    "steps_per_epoch": [45],
    "block_layers": 3,
    "hidden_units": 128,
    "blocks": 2,
    "horizon": 5,
    "history_length": 10,
    "init_learning_rate": 5e-4,
    "decay_steps": 8,
    "decay_rate": 0.7,
    "batch_size": 4,
    "weight_decay": 1e-5,
    "node_id_dim": 64,
    "num_nodes": 207,
    "num_stacks": [3],
    "epsilon": 10,
    "use_adaptive_graph": True,
    "aptonly": False,
    "randomadj": False,
    "apt_size": 10,
    "adjacency_path": os.path.join("data", "cooccurrence_adj_mx.pkl"),
    "graph_layer_type": "gcn",
    "enable_graph_conv": True,
    "use_static_graph": False,
    "graph_mode": "hybrid",
    "adjacency_fusion": "last",
    "adaptive_init": "last",
    "use_time_gate": False,
}


class Parameters(NamedTuple):
    dataset: str
    repeat: int
    epochs: int
    steps_per_epoch: int
    block_layers: int
    hidden_units: int
    blocks: int
    horizon: int
    history_length: int
    init_learning_rate: float
    decay_steps: int
    decay_rate: float
    batch_size: int
    weight_decay: float
    node_id_dim: int
    num_nodes: int
    num_stacks: int
    epsilon: float
    use_adaptive_graph: bool
    aptonly: bool
    randomadj: bool
    apt_size: int
    adjacency_path: str
    graph_layer_type: str
    enable_graph_conv: bool
    use_static_graph: bool
    graph_mode: str
    adjacency_fusion: str
    adaptive_init: str
    use_time_gate: bool


def serialize_hyperparameters(params: Parameters) -> Dict[str, Any]:
    if isinstance(params, Parameters):
        return dict(params._asdict())
    return params


def deserialize_hyperparameters(param_config: Dict[str, Any]) -> Parameters:
    if isinstance(param_config, Parameters):
        return param_config
    enriched = dict(param_config)
    for key, default in hyperparams_defaults.items():
        if key not in enriched:
            if isinstance(default, list):
                enriched[key] = default[0] if default else None
            else:
                enriched[key] = default
    addaptadj = enriched.pop("addaptadj", None)
    use_adaptive = enriched.get("use_adaptive_graph")
    if use_adaptive is None:
        use_adaptive = addaptadj
    if use_adaptive is None:
        use_adaptive = hyperparams_defaults.get("use_adaptive_graph", True)
    enriched["use_adaptive_graph"] = use_adaptive
    if "aptonly" not in enriched:
        enriched["aptonly"] = hyperparams_defaults.get("aptonly", False)
    if "randomadj" not in enriched:
        enriched["randomadj"] = hyperparams_defaults.get("randomadj", False)
    return Parameters(**enriched)


class FcBlock(tf.keras.layers.Layer):
    """Fully connected block identical to baseline implementation."""

    def __init__(self, hyperparams: Parameters, input_size: int, output_size: int, **kw):
        super().__init__(**kw)
        self.hyperparams = hyperparams
        self.input_size = input_size
        self.output_size = output_size
        self.fc_layers = []
        for i in range(hyperparams.block_layers):
            self.fc_layers.append(
                tf.keras.layers.Dense(hyperparams.hidden_units,
                                      activation=tf.nn.relu,
                                      kernel_regularizer=tf.keras.regularizers.l2(hyperparams.weight_decay),
                                      name=f"fc_{i}")
            )
        self.forecast = tf.keras.layers.Dense(self.output_size, activation=None, name="forecast")
        self.backcast = tf.keras.layers.Dense(self.input_size, activation=None, name="backcast")

    def call(self, inputs, training=False):
        h = self.fc_layers[0](inputs)
        for i in range(1, self.hyperparams.block_layers):
            h = self.fc_layers[i](h)
        backcast = tf.keras.activations.relu(inputs - self.backcast(h))
        return backcast, self.forecast(h)
