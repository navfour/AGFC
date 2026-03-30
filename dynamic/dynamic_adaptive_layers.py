"""Adaptive adjacency utilities dedicated to dynamic experiments."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


class DynamicAdaptiveAdjacencyLayer(tf.keras.layers.Layer):
    """
    Lightweight copy of the baseline adaptive adjacency layer.

    Supports three fusion modes:
    - 'none': Pure adaptive (no base graph)
    - 'static': Adaptive + static base graph (pre-combined)
    - 'hybrid': Adaptive + dynamic snapshot (fused at runtime)
    """

    def __init__(self,
                 num_nodes: int,
                 apt_size: int,
                 base_adj: Optional[np.ndarray],
                 combine_with_base: bool = True,
                 svd_init_adj: Optional[np.ndarray] = None,
                 fusion_mode: str = "static",  # New parameter
                 fusion_weight_init: Optional[np.ndarray] = None,  # Node-wise fusion weights for hybrid mode
                 **kw):
        super().__init__(**kw)
        self.num_nodes = num_nodes
        self.apt_size = apt_size
        self.base_adj_np = base_adj
        self.combine_with_base = combine_with_base
        self.fusion_mode = fusion_mode  # 'none', 'static', 'hybrid'
        self.fusion_weight_init = fusion_weight_init  # Save for build()
        init_source = svd_init_adj if svd_init_adj is not None else base_adj
        self.initial_nodevecs = self._svd_init(init_source) if init_source is not None else (None, None)
        self.base_adj = None

    def build(self, input_shape):
        if self.base_adj_np is not None:
            base_adj_init = tf.constant_initializer(self.base_adj_np.astype(np.float32))
            self.base_adj = self.add_weight(
                name="base_adjacency",
                shape=self.base_adj_np.shape,
                initializer=base_adj_init,
                trainable=False,
                dtype=tf.float32,
            )
        else:
            self.base_adj = None

        init1, init2 = self.initial_nodevecs
        if init1 is not None and init2 is not None:
            initializer1 = tf.constant_initializer(init1)
            initializer2 = tf.constant_initializer(init2)
        else:
            initializer1 = tf.keras.initializers.GlorotUniform()
            initializer2 = tf.keras.initializers.GlorotUniform()

        self.nodevec1 = self.add_weight(
            shape=(self.num_nodes, self.apt_size),
            initializer=initializer1,
            trainable=True,
            name="nodevec1"
        )
        self.nodevec2 = self.add_weight(
            shape=(self.apt_size, self.num_nodes),
            initializer=initializer2,
            trainable=True,
            name="nodevec2"
        )

        # Node-wise fusion weights for hybrid mode
        if self.fusion_mode == "hybrid" and self.fusion_weight_init is not None:
            # Initialize in log space for optimization stability and to ensure positive values
            init_logits = np.log(self.fusion_weight_init.astype(np.float32) + 1.0)

            self.fusion_weights = self.add_weight(
                shape=(self.num_nodes,),
                initializer=tf.keras.initializers.Constant(init_logits),
                trainable=True,
                name="fusion_weights"
            )
        else:
            self.fusion_weights = None

        super().build(input_shape)

    def call(self, inputs=None):
        """
        Compute final adjacency matrix.

        Args:
            inputs: Optional snapshot adjacency for hybrid mode
                   - None: use static base or pure adaptive
                   - Tensor: snapshot adjacency to fuse with adaptive

        Returns:
            Final adjacency matrix (num_nodes, num_nodes)
        """
        # Compute adaptive component
        adaptive = tf.matmul(self.nodevec1, self.nodevec2)
        adaptive = tf.nn.softmax(tf.nn.relu(adaptive), axis=1)

        # Fusion logic based on mode
        if self.fusion_mode == "hybrid" and inputs is not None:
            # Hybrid mode: Node-wise learnable fusion
            # Formula: row_norm(base + diag(m)·adaptive + I)
            # where m is learned node-wise weight vector
            snapshot = inputs  # Raw co-occurrence counts

            # Convert fusion weights from log space to positive values
            m = tf.exp(self.fusion_weights)  # (N,)
            m_expanded = m[:, tf.newaxis]  # (N, 1)

            # Node-wise scaling: diag(m) @ adaptive = adaptive * m[:, None]
            scaled_adaptive = adaptive * m_expanded  # (N, N)

            # Handle batch dimension
            if len(snapshot.shape) == 3:
                scaled_adaptive = tf.expand_dims(scaled_adaptive, axis=0)  # (1, N, N)

            # Fusion: base + diag(m)·adaptive
            fused = snapshot + scaled_adaptive

            # Add self-loops
            eye = tf.eye(self.num_nodes, dtype=fused.dtype)
            if len(snapshot.shape) == 3:
                eye = eye[tf.newaxis, ...]
            fused_with_loop = fused + eye

            # Row normalization (ONLY ONE TIME for hybrid mode)
            degree = tf.reduce_sum(fused_with_loop, axis=-1, keepdims=True)
            normalized = fused_with_loop / (degree + 1e-8)

            return normalized  # Returns normalized adjacency with self-loops, GCN uses directly

        elif self.fusion_mode == "static" and self.base_adj is not None and self.combine_with_base:
            # Static mode: use pre-stored base_adj
            return (adaptive + self.base_adj) / 2.0
        else:
            # Pure adaptive mode
            return adaptive

    def _svd_init(self, base_adj: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Initialize node embeddings using SVD decomposition of the base adjacency matrix.

        Args:
            base_adj: Raw adjacency matrix (without self-loops or normalization)

        Returns:
            (nodevec1, nodevec2): Two embedding matrices for adaptive graph, or (None, None) on failure

        Raises:
            ValueError: If matrix rank is insufficient (indicates data quality issues)
        """
        if base_adj is None:
            return None, None
        try:
            u, s, v = np.linalg.svd(base_adj)
            actual_rank = len(s)

            # Check if matrix has sufficient rank
            if actual_rank < self.apt_size:
                raise ValueError(
                    f"SVD initialization failed: Matrix rank ({actual_rank}) < apt_size ({self.apt_size}).\n"
                    f"This indicates the adjacency matrix is rank-deficient.\n"
                    f"Possible causes:\n"
                    f"  - Isolated nodes (no connections)\n"
                    f"  - Zero rows/columns in co-occurrence matrix\n"
                    f"  - Data quality issues (too sparse, duplicate rows, etc.)\n"
                    f"Solutions:\n"
                    f"  1. Check your co-occurrence data for quality issues\n"
                    f"  2. Reduce apt_size to <= {actual_rank}\n"
                    f"  3. Remove isolated nodes or use a denser subset of data"
                )

            # Extract top apt_size singular components (no padding needed)
            rank = self.apt_size
            u = u[:, :rank]
            v = v[:rank, :]
            s_root = np.sqrt(s[:rank])

            nodevec1 = (u * s_root).astype(np.float32)
            nodevec2 = (s_root[:, np.newaxis] * v).astype(np.float32)

            return nodevec1, nodevec2
        except ValueError:
            # Re-raise ValueError with our custom message
            raise
        except Exception as e:
            # Other errors (numerical issues, etc.)
            print(f"⚠️  SVD initialization failed with unexpected error: {e}")
            print(f"   Falling back to random initialization.")
            return None, None
