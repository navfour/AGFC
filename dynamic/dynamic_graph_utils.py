"""Graph utility functions for loading adjacency matrices within dynamic runs."""

from __future__ import annotations

import os
import pickle
from typing import Any, Optional

import numpy as np


def _append_adj_matrices(container: list[np.ndarray], obj: Any) -> None:
    if isinstance(obj, dict):
        if "adj_mx" in obj:
            _append_adj_matrices(container, obj["adj_mx"])
            return
        for value in obj.values():
            _append_adj_matrices(container, value)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _append_adj_matrices(container, item)
        return
    arr = np.asarray(obj)
    if arr.ndim < 2:
        return
    if arr.ndim == 2:
        container.append(arr)
        return
    for idx in range(arr.shape[0]):
        _append_adj_matrices(container, arr[idx])


def _extract_adj_matrix(obj: Any) -> np.ndarray:
    matrices: list[np.ndarray] = []
    _append_adj_matrices(matrices, obj)
    if not matrices:
        raise ValueError("No valid adjacency matrix found in object.")
    return matrices[0]


def _ensure_shape(adj: np.ndarray, num_nodes: Optional[int]) -> np.ndarray:
    if num_nodes is None or adj.shape[0] == num_nodes:
        return adj
    trimmed = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    min_nodes = min(num_nodes, adj.shape[0])
    trimmed[:min_nodes, :min_nodes] = adj[:min_nodes, :min_nodes]
    return trimmed


def _row_normalize(adj: np.ndarray) -> np.ndarray:
    adj = np.maximum(adj, 0.0)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    degree = adj.sum(axis=1, keepdims=True)
    degree[degree == 0] = 1.0
    return adj / degree


def load_cooccurrence_adjacency(path: str, num_nodes: Optional[int] = None) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    adj = _extract_adj_matrix(obj)
    if adj.ndim == 3:
        adj = adj[0]
    adj = adj.astype(np.float32)
    adj = _ensure_shape(adj, num_nodes)
    return _row_normalize(adj)


def load_cooccurrence_adjacency_raw(path: str, num_nodes: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Load co-occurrence adjacency matrix WITHOUT self-loops or normalization.

    This is used for analysis and visualization purposes to show the raw
    topic co-occurrence data as it was collected.

    Args:
        path: Path to the pickle file containing adjacency matrix
        num_nodes: Expected number of nodes (for shape validation)

    Returns:
        Raw adjacency matrix (num_nodes, num_nodes) or None if file not found
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    adj = _extract_adj_matrix(obj)
    if adj.ndim == 3:
        adj = adj[0]
    adj = adj.astype(np.float32)
    adj = _ensure_shape(adj, num_nodes)
    # Do NOT apply _row_normalize() - return raw co-occurrence counts
    return adj


__all__ = ["load_cooccurrence_adjacency", "load_cooccurrence_adjacency_raw"]
