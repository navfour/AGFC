"""Utility module for loading and serving graph snapshots."""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dynamic.dynamic_graph_utils import load_cooccurrence_adjacency, load_cooccurrence_adjacency_raw


@dataclasses.dataclass
class SlidingWindowSnapshotConfig:
    """Configuration describing how to pull snapshot adjacencies."""

    csv_path: str
    adjacency_dir: str
    history_length: int
    horizon: int
    adjacency_pattern: str = "subfield_cooccur_{year}.pkl"
    min_year: int = 1975


class SlidingWindowLastSnapshotProvider:
    """Selects the last snapshot inside each sliding window as the preset graph."""

    def __init__(self, config: SlidingWindowSnapshotConfig, *, num_nodes: int):
        self.config = config
        self.num_nodes = num_nodes
        self._df_index = self._load_index(config.csv_path)
        self._sample_years = self._compute_sample_years()
        self._sample_year_sequences = self._compute_sample_year_sequences()
        self._year_cache: Dict[int, np.ndarray] = {}
        self._year_cache_raw: Dict[int, np.ndarray] = {}  # Cache for raw matrices
        self._split_bounds = self._compute_split_bounds(len(self._sample_years))
        self._full_stack_cache: Optional[np.ndarray] = None
        self._full_history_stack_cache: Optional[np.ndarray] = None
        self._decay_alpha = 0.5

    @staticmethod
    def _load_index(csv_path: str) -> pd.Index:
        df = pd.read_csv(csv_path, index_col=0)
        # Treat integer years like "1975" as calendar years, not nanosecond offsets.
        df.index = pd.to_datetime(df.index.astype(str), format="%Y")
        return df.index

    def _compute_sample_years(self) -> List[int]:
        """Map every training sample to the year of its last observation."""
        history = self.config.history_length
        horizon = self.config.horizon
        total_points = len(self._df_index)
        min_t = history - 1
        max_t = total_points - horizon
        if max_t <= min_t:
            raise ValueError(
                f"Not enough samples for history={history} horizon={horizon} "
                f"(total points={total_points})."
            )
        years: List[int] = []
        for t in range(min_t, max_t):
            years.append(int(self._df_index[t].year))
        return years

    def _compute_sample_year_sequences(self) -> List[List[int]]:
        """For each sample, list the years of all timesteps within the window."""
        history = self.config.history_length
        horizon = self.config.horizon
        total_points = len(self._df_index)
        min_t = history - 1
        max_t = total_points - horizon
        if max_t <= min_t:
            return []
        sequences: List[List[int]] = []
        for t in range(min_t, max_t):
            start_idx = t - (history - 1)
            seq_years: List[int] = []
            for offset in range(history):
                year = int(self._df_index[start_idx + offset].year)
                seq_years.append(year)
            sequences.append(seq_years)
        return sequences

    @staticmethod
    def _compute_split_bounds(total_samples: int) -> Dict[str, Tuple[int, int]]:
        # Must match the split logic in dynamic_generate_training_data.py
        num_test = 5  # Fixed 5 samples for test set
        num_train = round(total_samples * 0.8)  # 80% train set
        num_val = total_samples - num_test - num_train  # Remaining for validation
        bounds = {
            "train": (0, num_train),
            "val": (num_train, num_train + num_val),
            "test": (total_samples - num_test, total_samples),
        }
        return bounds

    def _load_year_matrix(self, year: int) -> np.ndarray:
        """
        Load adjacency matrix for a given year (RAW version without preprocessing).

        Returns raw co-occurrence counts. Normalization and self-loops should be
        added in the model layer, not here.
        """
        target_year = max(year, self.config.min_year)
        if target_year in self._year_cache:
            return self._year_cache[target_year]
        filename = self.config.adjacency_pattern.format(year=target_year)
        path = os.path.join(self.config.adjacency_dir, filename)
        adj = load_cooccurrence_adjacency_raw(path, num_nodes=self.num_nodes)
        if adj is None:
            raise FileNotFoundError(f"Unable to load adjacency for year={target_year} from {path}")
        self._year_cache[target_year] = adj
        return adj

    def _load_year_matrix_raw(self, year: int) -> np.ndarray:
        """Load raw adjacency matrix without self-loops or normalization."""
        target_year = max(year, self.config.min_year)
        if target_year in self._year_cache_raw:
            return self._year_cache_raw[target_year]
        filename = self.config.adjacency_pattern.format(year=target_year)
        path = os.path.join(self.config.adjacency_dir, filename)
        adj = load_cooccurrence_adjacency_raw(path, num_nodes=self.num_nodes)
        if adj is None:
            raise FileNotFoundError(f"Unable to load raw adjacency for year={target_year} from {path}")
        self._year_cache_raw[target_year] = adj
        return adj

    def _build_full_stack(self) -> np.ndarray:
        if self._full_stack_cache is None:
            matrices = [self._load_year_matrix(year) for year in self._sample_years]
            self._full_stack_cache = np.stack(matrices, axis=0)
        return self._full_stack_cache

    def _build_full_history_stack(self) -> np.ndarray:
        if self._full_history_stack_cache is None:
            history_mats: List[np.ndarray] = []
            for year_seq in self._sample_year_sequences:
                seq_mats = [self._load_year_matrix(max(year, self.config.min_year)) for year in year_seq]
                history_mats.append(np.stack(seq_mats, axis=0))
            if not history_mats:
                raise ValueError("No history adjacency matrices could be built.")
            self._full_history_stack_cache = np.stack(history_mats, axis=0)
        return self._full_history_stack_cache

    def _default_decay_weights(self, length: int) -> np.ndarray:
        if length <= 0:
            raise ValueError("Cannot compute decay weights for empty history length.")
        positions = np.arange(length, dtype=np.float32)
        reversed_positions = positions[::-1]
        weights = np.power(self._decay_alpha, reversed_positions)
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            weights = np.ones_like(weights) / length
        else:
            weights = weights / weights_sum
        return weights.astype(np.float32)

    def _collapse_history_stack(self, history_stack: np.ndarray, fusion_mode: str) -> np.ndarray:
        fusion = fusion_mode.lower()
        if fusion == "sum":
            return np.sum(history_stack, axis=1)
        if fusion == "mean":
            return np.mean(history_stack, axis=1)
        if fusion == "learned_decay":
            weights = self._default_decay_weights(history_stack.shape[1])
            return np.tensordot(history_stack, weights, axes=([1], [0]))
        raise ValueError(f"Unsupported fusion mode '{fusion_mode}'.")

    def build_split_arrays(self, fusion_mode: str = "last") -> Dict[str, np.ndarray]:
        """Returns adjacency tensors aligned with train/val/test windows."""
        fusion = (fusion_mode or "last").lower()
        if fusion == "last":
            base_stack = self._build_full_stack()
        else:
            history_stack = self._build_full_history_stack()
            if fusion == "learned_decay":
                base_stack = history_stack
            else:
                base_stack = self._collapse_history_stack(history_stack, fusion_mode)
        split_arrays: Dict[str, np.ndarray] = {}
        for split, (start, end) in self._split_bounds.items():
            split_arrays[split] = base_stack[start:end]
        return split_arrays

    def fused_mean_adjacency(self,
                             split: str = "train",
                             fusion_mode: str = "last") -> np.ndarray:
        bounds = self._split_bounds.get(split)
        if bounds is None:
            raise ValueError(f"Unknown split: {split}")
        start, end = bounds
        if start >= end:
            raise ValueError(f"No samples available for split {split} to compute mean adjacency.")
        fusion = (fusion_mode or "last").lower()
        if fusion == "last":
            stack = self._build_full_stack()
        else:
            history_stack = self._build_full_history_stack()
            if fusion == "learned_decay":
                stack = self._collapse_history_stack(history_stack, fusion_mode)
            else:
                stack = self._collapse_history_stack(history_stack, fusion_mode)
        return np.mean(stack[start:end], axis=0)

    def fused_mean_adjacency_raw(self,
                                 split: str = "train",
                                 fusion_mode: str = "last") -> np.ndarray:
        """
        Compute mean adjacency using raw matrices (no self-loops, no normalization).

        This is used for analysis and visualization to show the original co-occurrence data.

        Args:
            split: Data split ('train', 'val', 'test')
            fusion_mode: How to combine historical snapshots ('last', 'mean', 'sum', 'learned_decay')

        Returns:
            Mean adjacency matrix (num_nodes, num_nodes)
        """
        bounds = self._split_bounds.get(split)
        if bounds is None:
            raise ValueError(f"Unknown split: {split}")
        start, end = bounds
        if start >= end:
            raise ValueError(f"No samples available for split {split} to compute mean adjacency.")

        fusion = (fusion_mode or "last").lower()

        # Build stack using raw matrices
        if fusion == "last":
            # Last snapshot in each window
            matrices = [self._load_year_matrix_raw(year) for year in self._sample_years]
            stack = np.stack(matrices, axis=0)
        else:
            # History-based fusion
            history_mats: List[np.ndarray] = []
            for year_seq in self._sample_year_sequences:
                seq_mats = [self._load_year_matrix_raw(max(year, self.config.min_year)) for year in year_seq]
                history_mats.append(np.stack(seq_mats, axis=0))
            history_stack = np.stack(history_mats, axis=0)

            if fusion == "learned_decay":
                stack = self._collapse_history_stack(history_stack, fusion_mode)
            else:
                stack = self._collapse_history_stack(history_stack, fusion_mode)

        return np.mean(stack[start:end], axis=0)
