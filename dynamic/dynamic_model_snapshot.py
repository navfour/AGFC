"""Models and trainers for dynamic (snapshot/adaptive) FC-GAGA experiments."""

from __future__ import annotations

import os
from itertools import product
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf

from dynamic.dynamic_core import FcBlock, Parameters
from dynamic.dynamic_adaptive_layers import DynamicAdaptiveAdjacencyLayer
from dynamic.dynamic_metrics import DynamicMetricsCallback


class DynamicFcGagaLayer(tf.keras.layers.Layer):
    """Layer that can consume snapshot adjacencies or learn its own adaptive graph."""

    def __init__(self,
                 hyperparams: Parameters,
                 input_size: int,
                 output_size: int,
                 num_nodes: int,
                 graph_layer_type: str = "none",
                 enable_graph_conv: bool = True,
                 adj_layer: Optional[DynamicAdaptiveAdjacencyLayer] = None,
                 base_graph: Optional[np.ndarray] = None,
                 adjacency_fusion: str = "last",
                 graph_mode: str = "hybrid",
                 use_time_gate: bool = False,
                 **kw):
        super().__init__(**kw)
        self.hyperparams = hyperparams
        self.num_nodes = num_nodes
        self.input_size = input_size
        self.output_size = output_size
        self.graph_layer_type = graph_layer_type.lower()
        self.enable_graph_conv = enable_graph_conv
        self.adj_layer = adj_layer
        self.base_graph = tf.constant(base_graph, dtype=tf.float32) if base_graph is not None else None
        self.adjacency_fusion = adjacency_fusion.lower() if adjacency_fusion else "last"
        self.graph_mode = graph_mode.lower() if graph_mode else "hybrid"
        self.use_time_gate = use_time_gate
        if self.graph_layer_type not in ("none", "gcn", "gat", "graphsage"):
            raise ValueError(f"Unsupported graph_layer_type: {self.graph_layer_type}")

        self.graph_gcn_dense = None
        self.graphsage_dense = None
        self.gat_dense = None
        self.gat_attention = None
        self.gat_leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        if self.graph_layer_type == "gcn":
            self.graph_gcn_dense = tf.keras.layers.Dense(
                self.input_size, use_bias=False, name=f"{self.name}_graph_gcn")
        elif self.graph_layer_type == "graphsage":
            self.graphsage_dense = tf.keras.layers.Dense(
                self.input_size, activation=tf.nn.relu, name=f"{self.name}_graph_sage")
        elif self.graph_layer_type == "gat":
            self.gat_dense = tf.keras.layers.Dense(
                self.input_size, use_bias=False, name=f"{self.name}_graph_gat")
            self.gat_attention = tf.keras.layers.Dense(
                1, use_bias=False, name=f"{self.name}_graph_gat_attention")

        self.blocks = []
        for i in range(self.hyperparams.blocks):
            self.blocks.append(
                FcBlock(hyperparams=hyperparams,
                        input_size=self.input_size,
                        output_size=hyperparams.horizon,
                        name=f"block_{i}")
            )

        self.node_id_em = tf.keras.layers.Embedding(
            input_dim=self.num_nodes,
            output_dim=self.hyperparams.node_id_dim,
            embeddings_initializer='uniform',
            input_length=self.num_nodes,
            name="dept_id_em",
            embeddings_regularizer=tf.keras.regularizers.l2(hyperparams.weight_decay))

        if self.use_time_gate:
            self.time_gate1 = tf.keras.layers.Dense(
                hyperparams.hidden_units,
                activation=tf.keras.activations.relu,
                name="time_gate1")
            self.time_gate2 = tf.keras.layers.Dense(
                hyperparams.horizon,
                activation=None,
                name="time_gate2")
            self.time_gate3 = tf.keras.layers.Dense(
                hyperparams.history_length,
                activation=None,
                name="time_gate3")
        else:
            self.time_gate1 = None
            self.time_gate2 = None
            self.time_gate3 = None
        self.decay_logit = None

    def build(self, input_shape):
        if self.adjacency_fusion == "learned_decay":
            self.decay_logit = self.add_weight(
                shape=(),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                name=f"{self.name}_adj_decay_logit")
        super().build(input_shape)

    def call(self,
             history_in,
             node_id_in,
             time_of_day_in,
             adjacency_in=None,
             training=False):
        node_id = self.node_id_em(node_id_in)
        node_id = tf.squeeze(node_id, axis=-2)

        if self.use_time_gate:
            time_gate = self.time_gate1(tf.concat([node_id, time_of_day_in], axis=-1))
            time_gate_forward = self.time_gate2(time_gate)
            time_gate_backward = self.time_gate3(time_gate)
        else:
            batch_size = tf.shape(history_in)[0]
            time_gate_forward = tf.zeros(
                (batch_size, self.num_nodes, self.hyperparams.horizon),
                dtype=history_in.dtype)
            time_gate_backward = tf.zeros_like(history_in)

        history_in = history_in / (1.0 + time_gate_backward)

        # Determine if we have any adjacency source (for independent mode, this will be None)
        graph_adj = None

        # Hybrid mode: both snapshot and adaptive layer exist
        if adjacency_in is not None and self.adj_layer is not None:
            snapshot = tf.cast(adjacency_in, tf.float32)
            # Pass snapshot to adaptive layer for fusion
            graph_adj = tf.cast(self.adj_layer(snapshot), tf.float32)

        # Snapshot only mode
        elif adjacency_in is not None:
            graph_adj = tf.cast(adjacency_in, tf.float32)

        # AdaptiveOnly mode
        elif self.adj_layer is not None:
            graph_adj = tf.cast(self.adj_layer(None), tf.float32)

        # Static base graph (fallback)
        elif self.base_graph is not None:
            graph_adj = self.base_graph

        # Independent mode: no graph adjacency available
        # Use zero-padding for cross-node features to maintain dimension compatibility
        if graph_adj is None:
            level = tf.reduce_max(history_in, axis=-1, keepdims=True)
            history = tf.math.divide_no_nan(history_in, level)

            # Create zero tensor as placeholder for all_node_history to maintain input_size
            shape = history_in.get_shape().as_list()
            batch_size = tf.shape(history_in)[0]
            all_node_history = tf.zeros(
                shape=[batch_size, self.num_nodes, self.num_nodes * shape[2]],
                dtype=history.dtype)

            # Concatenate in same order as standard mode to maintain dimension
            history = tf.concat([history, all_node_history], axis=-1)
            history = tf.concat([history, node_id], axis=-1)

        elif self.graph_layer_type == "none":
            # None mode: use graph adjacency for pre-aggregated cross-node features
            # Graph aggregation happens BEFORE FC blocks
            if graph_adj.shape.rank == 4:
                graph_adj = self._fuse_adjacency_stack(graph_adj)
            graph_adj = self._ensure_batch_dimension(graph_adj, tf.shape(history_in)[0])
            node_embeddings_dp = graph_adj[..., tf.newaxis]

            level = tf.reduce_max(history_in, axis=-1, keepdims=True)
            history = tf.math.divide_no_nan(history_in, level)

            shape = history_in.get_shape().as_list()
            all_node_history = tf.tile(
                history_in[:, tf.newaxis, :, :],
                multiples=[1, self.num_nodes, 1, 1])
            all_node_history = all_node_history * node_embeddings_dp
            all_node_history = tf.reshape(
                all_node_history,
                shape=[-1, self.num_nodes, self.num_nodes * shape[2]])
            all_node_history = tf.math.divide_no_nan(all_node_history - level, level)
            all_node_history = tf.where(all_node_history > 0, all_node_history, 0.0)
            history = tf.concat([history, all_node_history], axis=-1)
            history = tf.concat([history, node_id], axis=-1)

        else:
            # GNN modes (gcn/gat/graphsage): no pre-aggregation
            # Let the graph convolution layer handle neighbor aggregation
            if graph_adj.shape.rank == 4:
                graph_adj = self._fuse_adjacency_stack(graph_adj)
            graph_adj = self._ensure_batch_dimension(graph_adj, tf.shape(history_in)[0])

            level = tf.reduce_max(history_in, axis=-1, keepdims=True)
            history = tf.math.divide_no_nan(history_in, level)

            # Only concatenate self features and node_id (no all_node_history)
            history = tf.concat([history, node_id], axis=-1)

            # Apply graph convolution to learn neighbor aggregation
            if self.enable_graph_conv:
                history = self._apply_graph_layer(history, graph_adj)

        backcast, forecast_out = self.blocks[0](history)
        for i in range(1, self.hyperparams.blocks):
            backcast, forecast_block = self.blocks[i](backcast)
            forecast_out = forecast_out + forecast_block
        forecast_out = forecast_out[:, :, :self.hyperparams.horizon]
        forecast = forecast_out * level
        forecast = (1.0 + time_gate_forward) * forecast
        return backcast, forecast

    def _normalize_adjacency(self, adj_matrix, add_self_loops=True):
        """
        Normalize adjacency matrix for GCN/GraphSAGE.

        Note on double normalization in adaptive-only mode:
        When using adaptive graphs, the input is already softmax-normalized (row sums = 1).
        We still normalize here because:
        1. We need to add self-loops to the adjacency matrix
        2. After adding self-loops (A' = A + I), row sums change from 1.0 to ~2.0
        3. Re-normalization ensures row sums return to 1.0 for proper message passing

        This is intentional design, not a bug. The softmax in the adaptive layer is
        part of the "graph generation" process, while this normalization is part of
        the GCN message-passing mechanism.

        Args:
            adj_matrix: Input adjacency matrix (raw counts or softmax probabilities)
            add_self_loops: Whether to add self-loops before normalization

        Returns:
            Row-normalized adjacency matrix
        """
        adj = tf.cast(adj_matrix, tf.float32)
        if add_self_loops:
            eye = tf.eye(self.num_nodes, dtype=adj.dtype)
            adj = adj + eye[tf.newaxis, ...]
        degree = tf.reduce_sum(adj, axis=-1, keepdims=True)
        degree = tf.where(degree > 0, degree, tf.ones_like(degree))
        return adj / degree

    def _apply_graph_layer(self, features, adj_matrix):
        if self.graph_layer_type == "gcn":
            return self._apply_gcn(features, adj_matrix)
        if self.graph_layer_type == "graphsage":
            return self._apply_graphsage(features, adj_matrix)
        if self.graph_layer_type == "gat":
            return self._apply_gat(features, adj_matrix)
        return features

    def _apply_gcn(self, features, adj_matrix):
        transformed = self.graph_gcn_dense(features)
        # Hybrid mode: adj_matrix already normalized with self-loops in adaptive layer
        if self.graph_mode == "hybrid":
            return tf.einsum("bij,bjf->bif", adj_matrix, transformed)
        # Other modes: need normalization + self-loops
        norm_adj = self._normalize_adjacency(adj_matrix, add_self_loops=True)
        return tf.einsum("bij,bjf->bif", norm_adj, transformed)

    def _apply_graphsage(self, features, adj_matrix):
        # Hybrid mode: adj_matrix already normalized with self-loops
        if self.graph_mode == "hybrid":
            neighbor_info = tf.einsum("bij,bjf->bif", adj_matrix, features)
        else:
            norm_adj = self._normalize_adjacency(adj_matrix, add_self_loops=True)
            neighbor_info = tf.einsum("bij,bjf->bif", norm_adj, features)
        combined = tf.concat([features, neighbor_info], axis=-1)
        return self.graphsage_dense(combined)

    def _apply_gat(self, features, adj_matrix):
        h = self.gat_dense(features)
        adj = tf.cast(adj_matrix, tf.float32)
        if adj.shape.rank == 2:
            adj = adj[tf.newaxis, ...]
        eye = tf.eye(self.num_nodes, dtype=adj.dtype)
        adj = adj + eye[tf.newaxis, ...]
        mask = tf.where(adj > 0, 0.0, -1e9)
        h_i = tf.tile(tf.expand_dims(h, 2), [1, 1, self.num_nodes, 1])
        h_j = tf.tile(tf.expand_dims(h, 1), [1, self.num_nodes, 1, 1])
        logits = self.gat_attention(tf.concat([h_i, h_j], axis=-1))
        logits = tf.squeeze(self.gat_leaky_relu(logits), axis=-1)
        logits = logits + mask
        attention = tf.nn.softmax(logits, axis=-1)
        attention = tf.where(tf.math.is_finite(attention), attention, tf.zeros_like(attention))
        return tf.einsum("bij,bjf->bif", attention, h)

    def _fuse_adjacency_stack(self, adj_stack: tf.Tensor) -> tf.Tensor:
        if self.adjacency_fusion == "learned_decay":
            weights = self._decay_weights(tf.shape(adj_stack)[1])
            fused = tf.tensordot(adj_stack, weights, axes=([1], [0]))
            return fused
        return tf.reduce_mean(adj_stack, axis=1)

    def _decay_weights(self, stack_len: tf.Tensor) -> tf.Tensor:
        stack_len = tf.cast(stack_len, tf.int32)
        if self.decay_logit is None:
            ones = tf.ones((stack_len,), dtype=tf.float32)
            return ones / tf.cast(stack_len, tf.float32)
        alpha = tf.sigmoid(self.decay_logit)
        positions = tf.cast(tf.range(stack_len), tf.float32)
        reversed_positions = tf.reverse(positions, axis=[0])
        weights = tf.pow(alpha, reversed_positions)
        weights_sum = tf.reduce_sum(weights)
        weights = tf.math.divide_no_nan(weights, weights_sum)
        return weights

    def _ensure_batch_dimension(self, adj_matrix, batch_size):
        adj = tf.cast(adj_matrix, tf.float32)
        if adj.shape.rank == 2:
            adj = adj[tf.newaxis, ...]
        current = tf.shape(adj)[0]
        def tile_to_batch():
            multiples = tf.concat([[batch_size], tf.ones(tf.rank(adj) - 1, dtype=tf.int32)], axis=0)
            return tf.tile(adj, multiples)
        adj = tf.cond(
            tf.equal(current, batch_size),
            lambda: adj,
            lambda: tf.cond(tf.equal(current, 1), tile_to_batch, lambda: adj))
        return adj


class DynamicFcGaga:
    """Builds a Keras model that handles snapshot or adaptive adjacency."""

    def __init__(self,
                 hyperparams: Parameters,
                 name: str = 'dynamic_fcgaga',
                 logdir: str = 'logs_dynamic',
                 num_nodes: int = 100,
                 *,
                 adj_layer: Optional[DynamicAdaptiveAdjacencyLayer] = None,
                 base_graph: Optional[np.ndarray] = None,
                 expect_adjacency_input: bool = True,
                 adjacency_is_stacked: bool = False):
        self.hyperparams = hyperparams
        self.name = name
        self.logdir = logdir
        self.num_nodes = num_nodes
        self.adj_layer = adj_layer
        self.base_graph = base_graph
        self.expect_adjacency_input = expect_adjacency_input
        self.adjacency_is_stacked = adjacency_is_stacked

        # Compute input_size based on graph_layer_type
        graph_layer_type = getattr(hyperparams, "graph_layer_type", "none")
        if graph_layer_type in ("gcn", "gat", "graphsage"):
            # GNN modes: no pre-aggregation, let GNN handle neighbor aggregation
            self.input_size = (
                self.hyperparams.history_length +
                self.hyperparams.node_id_dim
            )
        else:
            # None or Independent modes: include all_node_history dimension
            self.input_size = (
                self.hyperparams.history_length +
                self.hyperparams.node_id_dim +
                self.num_nodes * self.hyperparams.history_length
            )

        self.fcgaga_layers = []
        for i in range(hyperparams.num_stacks):
            self.fcgaga_layers.append(
                DynamicFcGagaLayer(
                    hyperparams=hyperparams,
                    input_size=self.input_size,
                    output_size=hyperparams.horizon,
                    num_nodes=self.num_nodes,
                    graph_layer_type=getattr(hyperparams, "graph_layer_type", "none"),
                    enable_graph_conv=getattr(hyperparams, "enable_graph_conv", True),
                    adj_layer=self.adj_layer,
                    base_graph=self.base_graph,
                    adjacency_fusion=getattr(hyperparams, "adjacency_fusion", "last"),
                    graph_mode=getattr(hyperparams, "graph_mode", "hybrid"),
                    use_time_gate=getattr(hyperparams, "use_time_gate", False),
                    name=f"dynamic_fcgaga_{i}")
            )
        self.history_roll_layers = []
        if hyperparams.num_stacks > 1:
            roll_horizon = hyperparams.horizon
            hist_len = hyperparams.history_length
            if roll_horizon >= hist_len:
                def roll_fn_factory(_idx):
                    def _roll(inputs):
                        _, new_vals = inputs
                        return new_vals[..., -hist_len:]
                    return _roll
            else:
                def roll_fn_factory(_idx):
                    def _roll(inputs):
                        prev_hist, new_vals = inputs
                        trim = prev_hist[..., roll_horizon:]
                        return tf.concat([trim, new_vals], axis=-1)
                    return _roll
            for i in range(hyperparams.num_stacks - 1):
                self.history_roll_layers.append(
                    tf.keras.layers.Lambda(roll_fn_factory(i), name=f"history_roll_{i}"))

        inputs, outputs = self.get_model()
        self.inputs = inputs
        self.outputs = outputs
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    def get_model(self):
        history_in = tf.keras.layers.Input(
            shape=(self.num_nodes, self.hyperparams.history_length), name='history')
        time_of_day_in = tf.keras.layers.Input(
            shape=(self.num_nodes, self.hyperparams.history_length), name='time_of_day')
        node_id_in = tf.keras.layers.Input(
            shape=(self.num_nodes, 1), dtype=tf.uint16, name='node_id')
        adjacency_in = None
        if self.expect_adjacency_input:
            if self.adjacency_is_stacked and self.hyperparams.adjacency_fusion == "learned_decay":
                adj_shape = (self.hyperparams.history_length, self.num_nodes, self.num_nodes)
            else:
                adj_shape = (self.num_nodes, self.num_nodes)
            adjacency_in = tf.keras.layers.Input(
                shape=adj_shape, name='adjacency')

        history_buffer = history_in
        backcast, forecast = self.fcgaga_layers[0](
            history_in=history_buffer,
            node_id_in=node_id_in,
            time_of_day_in=time_of_day_in,
            adjacency_in=adjacency_in)
        if self.history_roll_layers:
            history_buffer = self.history_roll_layers[0]([history_buffer, forecast])
        for idx, layer in enumerate(self.fcgaga_layers[1:], start=1):
            backcast, forecast_graph = layer(
                history_in=history_buffer,
                node_id_in=node_id_in,
                time_of_day_in=time_of_day_in,
                adjacency_in=adjacency_in)
            forecast = forecast + forecast_graph
            if idx < len(self.history_roll_layers):
                history_buffer = self.history_roll_layers[idx]([history_buffer, forecast_graph])
        forecast = forecast / self.hyperparams.num_stacks
        forecast = tf.keras.layers.Lambda(
            lambda t: tf.where(tf.math.is_nan(t), tf.zeros_like(t), t),
            name="forecast_nan_to_zero")(forecast)
        forecast = tf.keras.layers.Lambda(lambda t: t, name="targets")(forecast)

        inputs = {
            'history': history_in,
            'node_id': node_id_in,
            'time_of_day': time_of_day_in,
        }
        if self.expect_adjacency_input:
            inputs['adjacency'] = adjacency_in
        outputs = forecast
        return inputs, outputs


class DynamicTrainer:
    """Trainer that feeds snapshot or adaptive adjacency into DynamicFcGaga."""

    def __init__(self,
                 hyperparams: Parameters,
                 logdir: str,
                 adjacency_splits: Optional[Dict[str, np.ndarray]],
                 adj_layer: Optional[DynamicAdaptiveAdjacencyLayer],
                 base_graph: Optional[np.ndarray],
                 expect_adjacency_input: bool,
                 adjacency_is_stacked: bool):
        inp = dict(hyperparams._asdict())
        values = [v if isinstance(v, list) else [v] for v in inp.values()]
        self.hyperparams = [Parameters(**dict(zip(inp.keys(), v))) for v in product(*values)]
        inp_lists = {k: v for k, v in inp.items() if isinstance(v, list)}
        values = [v for v in inp_lists.values()]
        variable_values = [dict(zip(inp_lists.keys(), v)) for v in product(*values)]
        folder_names = []
        for d in variable_values:
            folder_names.append(';'.join(['%s=%s' % (key, value) for (key, value) in d.items()]))
        self.history = []
        self.forecasts = []
        self.models = []
        self.logdir = logdir
        self.folder_names = folder_names
        self.best_weight_paths = []
        self.adjacency_splits = adjacency_splits
        self.adj_layer = adj_layer
        self.base_graph = base_graph
        self.expect_adjacency_input = expect_adjacency_input
        self.adjacency_is_stacked = adjacency_is_stacked
        for i, h in enumerate(self.hyperparams):
            self.models.append(DynamicFcGaga(
                hyperparams=h,
                name=f"dynamic_fcgaga_model_{i}",
                logdir=os.path.join(self.logdir, folder_names[i]),
                num_nodes=h.num_nodes,
                adj_layer=self.adj_layer,
                base_graph=self.base_graph,
                expect_adjacency_input=self.expect_adjacency_input,
                adjacency_is_stacked=self.adjacency_is_stacked))

    @staticmethod
    def _build_node_ids(num_samples: int, num_nodes: int) -> np.ndarray:
        base = np.arange(num_nodes, dtype=np.uint16).reshape(1, num_nodes, 1)
        return np.tile(base, (num_samples, 1, 1))

    def _batch_generator(self,
                         dataset,
                         hyperparams: Parameters,
                         split: str) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        x_data = dataset.data[f"x_{split}"]
        y_data = dataset.data[f"y_{split}"][..., 0]
        adjacency_data = None
        if self.expect_adjacency_input and self.adjacency_splits:
            adjacency_data = self.adjacency_splits[split]
        num_nodes = dataset.num_nodes
        while True:
            ts_idxs = np.random.choice(np.arange(len(x_data)), size=hyperparams.batch_size, replace=True)
            history = x_data[ts_idxs][..., 0]
            time_of_day = x_data[ts_idxs][..., 1]
            labels = y_data[ts_idxs]
            node_ids = np.tile(
                np.arange(num_nodes, dtype=np.uint16).reshape(1, num_nodes, 1),
                (hyperparams.batch_size, 1, 1))
            weights = np.all(labels > 0, axis=-1, keepdims=False).astype(np.float32)
            weights = weights / np.prod(weights.shape)
            inputs = {
                "history": history,
                "node_id": node_ids,
                "time_of_day": time_of_day,
            }
            if self.expect_adjacency_input and adjacency_data is not None:
                inputs["adjacency"] = adjacency_data[ts_idxs]
            yield inputs, labels, weights

    def fit(self, dataset, verbose: int = 1):
        for i, hyperparams in enumerate(self.hyperparams):
            if verbose > 0:
                print(f"[Dynamic] Fitting model {i + 1} of {len(self.hyperparams)}, {self.folder_names[i]}")
            boundary_step = hyperparams.epochs // 10 if hyperparams.epochs >= 10 else 1
            boundary_start = hyperparams.epochs - boundary_step * hyperparams.decay_steps - 1
            boundaries = list(range(boundary_start, hyperparams.epochs, boundary_step))
            values = list(hyperparams.init_learning_rate * hyperparams.decay_rate ** np.arange(0, len(boundaries) + 1))
            scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=boundaries, values=values)

            def lr_schedule(epoch, lr):
                new_lr = scheduler(epoch)
                if isinstance(new_lr, tf.Tensor):
                    new_lr = tf.keras.backend.get_value(new_lr)
                return float(new_lr)

            lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)
            metrics = DynamicMetricsCallback(dataset=dataset,
                                             adjacency_splits=self.adjacency_splits,
                                             expect_adjacency_input=self.expect_adjacency_input)
            best_dir = os.path.join(self.models[i].logdir, "best_checkpoint")
            os.makedirs(best_dir, exist_ok=True)
            best_weights_path = os.path.join(best_dir, "weights.weights.h5")
            best_ckpt = tf.keras.callbacks.ModelCheckpoint(
                filepath=best_weights_path,
                save_best_only=True,
                save_weights_only=True,
                monitor='mae_val',
                mode='min',
                verbose=0)

            # Early stopping to prevent overfitting and save training time
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='mae_val',
                patience=40,
                mode='min',
                verbose=1,
                restore_best_weights=False  # ModelCheckpoint already handles this
            )

            self.models[i].model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams.init_learning_rate),
                loss=tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM),
                loss_weights=1.0)

            fit_output = self.models[i].model.fit(
                self._batch_generator(dataset, hyperparams, split="train"),
                callbacks=[lr_callback, metrics, best_ckpt, early_stop],
                epochs=hyperparams.epochs,
                steps_per_epoch=hyperparams.steps_per_epoch,
                verbose=verbose)
            self.history.append(fit_output.history)
            if os.path.exists(best_weights_path):
                self.models[i].model.load_weights(best_weights_path)
            self.best_weight_paths.append(best_weights_path)
