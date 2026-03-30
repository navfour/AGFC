"""Entry script for experimenting with dynamic/snapshot-based adjacency settings."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import csv

from dynamic.dynamic_dataset import Dataset
from dynamic.dynamic_utils import masked_mae_np, masked_mape_np, masked_rmse_np, masked_r2_np
from dynamic.dynamic_adaptive_layers import DynamicAdaptiveAdjacencyLayer
from dynamic.dynamic_graph_provider import (
    SlidingWindowLastSnapshotProvider,
    SlidingWindowSnapshotConfig,
)
from dynamic.dynamic_model_snapshot import DynamicTrainer
from dynamic.dynamic_core import Parameters, hyperparams_defaults, serialize_hyperparameters
from dynamic.dynamic_graph_utils import _row_normalize


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(BASE_DIR, "data")
DEFAULT_DATA_DIR = os.path.join(DEFAULT_DATA_ROOT, "oa_2")
DEFAULT_CSV = os.path.join(DEFAULT_DATA_DIR, "oa_2_covert.csv")
DEFAULT_ADJ_DIR = os.path.join(DEFAULT_DATA_DIR, "oa_2_coo_pkl")
DEFAULT_ADJ_PATTERN = "2_cooccur_{year}.pkl"
DEFAULT_LOGDIR = os.path.join(BASE_DIR, "logs_dynamic0130")
DEFAULT_RESULTS = os.path.join(BASE_DIR, "dynamic_results0130")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic FC-GAGA with snapshot or adaptive graphs.")
    parser.add_argument("--data_csv", default=DEFAULT_CSV, help="CSV with citation time series.")
    parser.add_argument("--adjacency_dir", default=DEFAULT_ADJ_DIR, help="Directory containing snapshot adjacencies.")
    parser.add_argument("--adjacency_pattern",
                        default=DEFAULT_ADJ_PATTERN,
                        help="Filename pattern for snapshot adjacencies (use {year}).")
    parser.add_argument("--history_length", type=int, default=5, help="Sliding window length.")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon.")
    parser.add_argument("--logdir", default=DEFAULT_LOGDIR, help="Training log directory.")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS, help="Directory for exported artifacts.")
    parser.add_argument("--dataset_name", default="citation_ai_dynamic", help="Dataset cache folder name.")
    parser.add_argument("--graph_mode",
                        choices=["basegraph", "adaptiveonly", "hybrid", "independent"],
                        default="hybrid",
                        help="basegraph: only use base graph (time-varying snapshots); "
                             "adaptiveonly: only use adaptive graph; "
                             "hybrid: use both base graph and adaptive graph; "
                             "independent: no graph structure.")
    parser.add_argument("--adjacency_fusion",
                        choices=["last", "sum", "mean", "learned_decay"],
                        default="mean",
                        help="How to fuse base graphs within sliding window (effective for basegraph/hybrid modes). "
                             "last: use last snapshot; sum: sum all; mean: average; learned_decay: learn decay weights.")
    parser.add_argument("--adaptive_init",
                        choices=["last", "sum", "mean", "random"],
                        default="last",
                        help="How to initialize adaptive graph (effective for adaptiveonly/hybrid modes). "
                             "last/sum/mean: use SVD of fused base graphs; random: random initialization.")
    parser.add_argument("--use_time_gate",
                        action="store_true",
                        help="Enable time-gating (per-node temporal scaling). Disabled by default.")
    parser.add_argument("--graph_layer_type",
                        choices=["none", "gcn", "gat", "graphsage"],
                        default="graphsage",
                        help="Graph convolution type used inside DynamicFcGagaLayer.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Override steps per epoch.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--apt_size", type=int, default=None, help="Override adaptive adjacency rank.")
    parser.add_argument("--repeat", type=int, default=None, help="Override repeat index.")
    return parser.parse_args()


def ensure_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def build_node_ids(num_samples: int, num_nodes: int) -> np.ndarray:
    base = np.arange(num_nodes, dtype=np.uint16).reshape(1, num_nodes, 1)
    return np.tile(base, (num_samples, 1, 1))


def extract_targets(prediction):
    return prediction['targets'] if isinstance(prediction, dict) else prediction


def compute_metrics(prediction: np.ndarray, labels: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    metrics = {
        "MAE": float(masked_mae_np(prediction, labels, null_val=0)),
        "MAPE": float(masked_mape_np(prediction, labels, null_val=0)),
        "RMSE": float(masked_rmse_np(prediction, labels, null_val=0)),
        "R2": float(masked_r2_np(prediction, labels, null_val=0)),
    }
    per_horizon: Dict[str, Dict[str, float]] = {}
    for horizon_idx in range(prediction.shape[-1]):
        key = f"h{horizon_idx + 1}"
        per_horizon[key] = {
            "MAE": float(masked_mae_np(prediction[..., horizon_idx], labels[..., horizon_idx], null_val=0)),
            "MAPE": float(masked_mape_np(prediction[..., horizon_idx], labels[..., horizon_idx], null_val=0)),
            "RMSE": float(masked_rmse_np(prediction[..., horizon_idx], labels[..., horizon_idx], null_val=0)),
            "R2": float(masked_r2_np(prediction[..., horizon_idx], labels[..., horizon_idx], null_val=0)),
        }
    return metrics, per_horizon


def _patch_tensorflow_dictwrapper_bug() -> bool:
    """
    TensorFlow 2.13+ on Python 3.12 can crash when tf.saved_model.save scans
    trackable dict wrappers. We defensively patch the offending runtime check.
    """
    try:
        # Delayed import keeps the module optional outside TensorFlow contexts.
        from tensorflow.python.framework import tensor_util as _tensor_util  # type: ignore
    except Exception:
        return False

    if getattr(_tensor_util, "_fcgaga_safe_is_tf_type", False):
        return True

    original_is_tf_type = _tensor_util.is_tf_type

    def _safe_is_tf_type(value):
        try:
            return original_is_tf_type(value)
        except TypeError as exc:
            # _DictWrapper objects trigger a Python 3.12 inspect crash; surface False instead.
            if "_DictWrapper" in str(exc):
                return False
            raise

    _tensor_util.is_tf_type = _safe_is_tf_type
    if hasattr(tf, "is_tensor"):
        tf.is_tensor = _safe_is_tf_type
    _tensor_util._fcgaga_safe_is_tf_type = True
    return True


def export_saved_model(model: tf.keras.Model, export_dir: str) -> None:
    """Wrapper around tf.saved_model.save with a fallback patch for Python 3.12."""
    try:
        tf.saved_model.save(model, export_dir)
        return
    except TypeError as exc:
        if "_DictWrapper" not in str(exc):
            raise
        print("Encountered TensorFlow _DictWrapper serialization bug; applying compatibility patch.")
    if not _patch_tensorflow_dictwrapper_bug():
        raise RuntimeError("Unable to patch TensorFlow's is_tf_type for _DictWrapper handling.")
    tf.saved_model.save(model, export_dir)


def align_adjacency(dataset: Dataset, adjacency_splits: Dict[str, np.ndarray]) -> None:
    for split in ("train", "val", "test"):
        expected = dataset.data[f"x_{split}"].shape[0]
        actual = adjacency_splits[split].shape[0]
        if expected != actual:
            raise ValueError(
                f"Adjacency samples for split '{split}' mismatch "
                f"(expected {expected}, got {actual}).")


def create_hyperparams(dataset: Dataset,
                       history_length: int,
                       horizon: int,
                       *,
                       use_adaptive_graph: bool,
                       aptonly: bool,
                       graph_mode: str,
                       adjacency_fusion: str,
                       adaptive_init: str,
                       use_time_gate: bool,
                       graph_layer_type: str,
                       epochs: Optional[int],
                       steps_per_epoch: Optional[int],
                       batch_size: Optional[int],
                       apt_size: Optional[int],
                       repeat: Optional[int]) -> Parameters:
    params = dict(hyperparams_defaults)
    params.update({
        "dataset": dataset.name,
        "history_length": history_length,
        "horizon": horizon,
        "num_nodes": dataset.num_nodes,
        "use_adaptive_graph": use_adaptive_graph,
        "aptonly": aptonly,
        "use_static_graph": False,
        "graph_mode": graph_mode,
        "adjacency_fusion": adjacency_fusion,
        "adaptive_init": adaptive_init,
        "use_time_gate": use_time_gate,
        "graph_layer_type": graph_layer_type,
    })
    if epochs is not None:
        params["epochs"] = epochs
    if steps_per_epoch is not None:
        params["steps_per_epoch"] = steps_per_epoch
    if batch_size is not None:
        params["batch_size"] = batch_size
    if apt_size is not None:
        params["apt_size"] = apt_size
    if repeat is not None:
        params["repeat"] = repeat
    return Parameters(**params)


def load_year_timestamps(csv_path: str) -> list[str]:
    if not os.path.exists(csv_path):
        print(f"⚠️  WARNING: CSV file not found: {csv_path}")
        return []
    df = pd.read_csv(csv_path, index_col=0)

    # Extract years from index
    timestamps = []
    for i in df.index:
        s = str(i)
        # If index value is a 4-digit year (1975, 1976, etc.), use it directly
        if s.isdigit() and len(s) == 4:
            timestamps.append(s)
        elif len(s) >= 4:
            # Try to extract year from datetime string (e.g., "1975-01-01" -> "1975")
            try:
                idx_dt = pd.to_datetime(s)
                timestamps.append(idx_dt.strftime("%Y"))
            except:
                # Fallback: take first 4 characters
                timestamps.append(s[:4])
        else:
            timestamps.append(s)

    print(f"✓ Loaded {len(timestamps)} timestamps from CSV")
    if len(timestamps) > 0:
        print(f"  Sample timestamps: {timestamps[:3]} ... {timestamps[-3:]}")
    return timestamps


def load_node_labels(csv_path: str) -> list[str]:
    if not os.path.exists(csv_path):
        print(f"⚠️  WARNING: CSV file not found: {csv_path}")
        return []
    df = pd.read_csv(csv_path, index_col=0)
    labels = list(df.columns)
    print(f"✓ Loaded {len(labels)} node labels from CSV")
    return labels


def export_prediction_csv(path: str,
                          prediction: np.ndarray,
                          labels: np.ndarray,
                          timestamps: list[str],
                          sample_indices: np.ndarray,
                          history_length: int,
                          node_labels: list[str]) -> None:
    num_samples, num_nodes, num_horizons = prediction.shape
    min_t = history_length - 1
    y_offsets = np.arange(1, num_horizons + 1)

    # Diagnostic information
    max_global_idx = sample_indices[-1] if len(sample_indices) > 0 else -1
    max_ts_idx = max_global_idx + min_t + num_horizons
    print(f"\n{'='*70}")
    print(f"CSV Export Diagnostics:")
    print(f"{'='*70}")
    print(f"  Timestamps length: {len(timestamps)}")
    print(f"  Num samples: {num_samples}")
    print(f"  Sample indices range: [{sample_indices[0]}, {sample_indices[-1]}]")
    print(f"  History length: {history_length}")
    print(f"  Num horizons: {num_horizons}")
    print(f"  Max ts_idx will be: {max_ts_idx}")
    print(f"  Out of bounds: {max_ts_idx >= len(timestamps)}")
    print(f"{'='*70}\n")

    rows = []
    missing_year_count = 0
    for sample_idx in range(num_samples):
        global_idx = sample_indices[sample_idx]
        base_t = global_idx + min_t
        for node_idx in range(num_nodes):
            for horizon_idx, y_offset in enumerate(y_offsets):
                ts_idx = base_t + y_offset
                year = ""
                if 0 <= ts_idx < len(timestamps):
                    ts_value = str(timestamps[ts_idx])
                    year = ts_value[:4] if len(ts_value) >= 4 else ts_value
                else:
                    missing_year_count += 1
                    # Use "N/A" instead of empty string to avoid 1970 display
                    year = "N/A"
                node_label = node_labels[node_idx] if node_idx < len(node_labels) else str(node_idx)
                rows.append({
                    "nodeid": node_label,
                    "year": year,
                    "true": float(labels[sample_idx, node_idx, horizon_idx]),
                    "predict": float(prediction[sample_idx, node_idx, horizon_idx]),
                })

    if missing_year_count > 0:
        print(f"⚠️  WARNING: {missing_year_count} rows have missing year data (marked as 'N/A')")
        print(f"   This means ts_idx exceeded timestamps length ({len(timestamps)})")
        print(f"   Check if your CSV data matches the dataset split configuration.\n")

    pd.DataFrame(rows).to_csv(path, index=False)


def export_node_error_csv(path: str,
                          prediction: np.ndarray,
                          labels: np.ndarray,
                          node_labels: list[str]) -> None:
    num_nodes = prediction.shape[1]
    rows = []
    for node_idx in range(num_nodes):
        preds = prediction[:, node_idx, :]
        truths = labels[:, node_idx, :]
        rmse = float(masked_rmse_np(preds, truths, null_val=0))
        mse = float(rmse ** 2)
        mae = float(masked_mae_np(preds, truths, null_val=0))
        mape = float(masked_mape_np(preds, truths, null_val=0))
        r2 = float(masked_r2_np(preds, truths, null_val=0))
        node_label = node_labels[node_idx] if node_idx < len(node_labels) else str(node_idx)
        rows.append({
            "nodeid": node_label,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def export_config_json(path: str,
                       args_namespace: argparse.Namespace,
                       hyperparams: Parameters,
                       metadata: Dict[str, str]) -> None:
    args_dict = {}
    for key, value in vars(args_namespace).items():
        if isinstance(value, (np.integer, np.int64)):
            args_dict[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            args_dict[key] = float(value)
        else:
            args_dict[key] = value
    payload = {
        "metadata": metadata,
        "cli_args": args_dict,
        "hyperparameters": serialize_hyperparameters(hyperparams),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def append_results_index(csv_path: str, row: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    fieldnames = [
        "run_folder",
        "timestamp",
        "graph_mode",
        "adjacency_fusion",
        "adaptive_init",
        "use_time_gate",
        "graph_layer_type",
        "repeat",
        "history_length",
        "horizon",
        "apt_size",
        "val_mae",
        "test_mae",
        "test_rmse",
        "test_mape",
        "test_r2",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    args.data_csv = resolve_path(args.data_csv)
    args.adjacency_dir = resolve_path(args.adjacency_dir)
    args.logdir = resolve_path(args.logdir)
    args.results_dir = resolve_path(args.results_dir)
    ensure_directory(args.logdir)
    ensure_directory(args.results_dir)

    dataset = Dataset(
        name=args.dataset_name,
        horizon=args.horizon,
        history_length=args.history_length,
        path=os.path.dirname(args.data_csv),
        data_csv=args.data_csv)

    snapshot_config = SlidingWindowSnapshotConfig(
        csv_path=args.data_csv,
        adjacency_dir=args.adjacency_dir,
        history_length=args.history_length,
        horizon=args.horizon,
        adjacency_pattern=args.adjacency_pattern,
        min_year=1975)
    provider = SlidingWindowLastSnapshotProvider(
        snapshot_config,
        num_nodes=dataset.num_nodes)

    graph_mode = args.graph_mode

    # Independent mode: no graph structure at all
    if graph_mode == "independent":
        use_base_graph = False
        use_adaptive_graph = False
        aptonly_flag = False
        adjacency_splits = {}
        adaptive_base_adj = None

        print("=" * 70)
        print("Running in INDEPENDENT mode: nodes are modeled independently without graph structure.")
        print("=" * 70)

        # Force graph_layer_type to 'none' for independent mode
        if args.graph_layer_type != "none":
            print()
            print("⚠️  AUTO-CORRECTION: Independent Mode Configuration")
            print("-" * 70)
            print(f"检测到: graph_mode=independent + graph_layer_type={args.graph_layer_type}")
            print()
            print("说明:")
            print("  - Independent 模式下无邻接矩阵，不应使用图卷积")
            print("  - 自动修正为: graph_layer_type=none")
            print("-" * 70)
            print()
            args.graph_layer_type = "none"  # 强制修正

        print("✓ 配置: independent + none (节点完全独立建模)")
        print("=" * 70)
        print()
    else:
        # Graph-based modes: basegraph, adaptiveonly, hybrid
        use_base_graph = graph_mode in ("basegraph", "hybrid")
        use_adaptive_graph = graph_mode in ("adaptiveonly", "hybrid")
        aptonly_flag = graph_mode == "adaptiveonly"

        # Build adjacency splits (time-varying base graphs) if using base graph
        adjacency_splits = provider.build_split_arrays(fusion_mode=args.adjacency_fusion) if use_base_graph else {}
        if use_base_graph:
            align_adjacency(dataset, adjacency_splits)
            print(f"✓ Base graph enabled: adjacency_fusion={args.adjacency_fusion}")

        # Get base graph for adaptive initialization if needed
        adaptive_base_adj = None
        svd_init_adj = None
        if use_adaptive_graph and args.adaptive_init != "random":
            # Get raw matrix for SVD initialization (no self-loops, no normalization)
            svd_init_adj = provider.fused_mean_adjacency_raw(
                split="train",
                fusion_mode=args.adaptive_init
            )

            # If hybrid/static mode, prepare normalized version for graph fusion
            if not aptonly_flag:
                adaptive_base_adj = _row_normalize(svd_init_adj)

            print(f"✓ Adaptive graph enabled: adaptive_init={args.adaptive_init} (SVD on raw matrix)")
        elif use_adaptive_graph:
            print(f"✓ Adaptive graph enabled: adaptive_init=random")

    hyperparams = create_hyperparams(dataset,
                                     args.history_length,
                                     args.horizon,
                                     use_adaptive_graph=use_adaptive_graph,
                                     aptonly=aptonly_flag,
                                     graph_mode=graph_mode,
                                     adjacency_fusion=args.adjacency_fusion,
                                     adaptive_init=args.adaptive_init,
                                     use_time_gate=args.use_time_gate,
                                     graph_layer_type=args.graph_layer_type,
                                     epochs=args.epochs,
                                     steps_per_epoch=args.steps_per_epoch,
                                     batch_size=args.batch_size,
                                     apt_size=args.apt_size,
                                     repeat=args.repeat)

    adj_layer = None
    base_graph_for_layer = None

    # Only create adaptive adjacency layer for graph-based modes
    if use_adaptive_graph:
        mix_adj = None if aptonly_flag else adaptive_base_adj  # Normalized version for fusion
        base_graph_for_layer = adaptive_base_adj if (not aptonly_flag and adaptive_base_adj is not None) else None

        # Determine fusion mode based on graph configuration
        if aptonly_flag:
            fusion_mode = "none"  # Pure adaptive, no base graph
        elif use_base_graph:
            fusion_mode = "hybrid"  # Dynamic fusion with time-varying snapshots
        else:
            fusion_mode = "static"  # Static base graph only

        # Prepare node-wise fusion weight initialization (only for hybrid mode)
        fusion_weight_init = None
        if fusion_mode == "hybrid":
            # Compute row sums of training base graph for initialization
            # Use the original snapshot graphs rather than svd_init_adj
            # This ensures fusion_weight_init is independent of adaptive graph initialization
            mean_snapshot_graph = provider.fused_mean_adjacency_raw(
                split="train",
                fusion_mode=args.adjacency_fusion
            )
            fusion_weight_init = np.sum(mean_snapshot_graph, axis=1)  # (num_nodes,)
            print(f"✓ Hybrid mode: Node-wise fusion weights initialized from snapshot graphs")
            print(f"  Weight range: [{np.min(fusion_weight_init):.2f}, {np.max(fusion_weight_init):.2f}]")
            print(f"  Weight mean: {np.mean(fusion_weight_init):.2f}")

        adj_layer = DynamicAdaptiveAdjacencyLayer(
            num_nodes=dataset.num_nodes,
            apt_size=hyperparams.apt_size,
            base_adj=mix_adj,              # For fusion: normalized version (with self-loops)
            combine_with_base=mix_adj is not None,
            svd_init_adj=svd_init_adj,     # For SVD: raw version (original co-occurrence counts)
            fusion_mode=fusion_mode,       # Explicitly set fusion mode
            fusion_weight_init=fusion_weight_init,  # Node-wise fusion weights for hybrid mode
            name="dynamic_adaptive_adj")

    adjacency_is_stacked = use_base_graph and args.adjacency_fusion == "learned_decay"
    trainer = DynamicTrainer(
        hyperparams=hyperparams,
        logdir=args.logdir,
        adjacency_splits=adjacency_splits if use_base_graph else {},
        adj_layer=adj_layer,
        base_graph=base_graph_for_layer,
        expect_adjacency_input=use_base_graph,
        adjacency_is_stacked=adjacency_is_stacked)
    trainer.fit(dataset=dataset)

    best_model_idx = None
    best_model_val = np.inf
    best_epoch_indices = []
    for i, history in enumerate(trainer.history):
        mae_val = history['mae_val']
        early_stop_idx = int(np.argmin(mae_val))
        best_epoch_indices.append(early_stop_idx)
        if mae_val[early_stop_idx] < best_model_val:
            best_model_val = mae_val[early_stop_idx]
            best_model_idx = i

    if best_model_idx is None:
        print("No trained dynamic model available.")
        return

    best_model = trainer.models[best_model_idx]
    best_hparams = trainer.hyperparams[best_model_idx]

    test_history = dataset.data['x_test'][..., 0]
    test_time_of_day = dataset.data['x_test'][..., 1]
    test_targets = dataset.data['y_test'][..., 0]
    node_ids = build_node_ids(test_history.shape[0], dataset.num_nodes)
    prediction_inputs = {
        "history": test_history,
        "node_id": node_ids,
        "time_of_day": test_time_of_day,
    }
    adjacency_test = None
    if use_base_graph:
        adjacency_test = adjacency_splits["test"]
        prediction_inputs["adjacency"] = adjacency_test

    prediction_raw = best_model.model.predict(
        prediction_inputs,
        batch_size=best_hparams.batch_size,
        verbose=0)
    prediction = extract_targets(prediction_raw)

    metrics_overall, per_horizon_metrics = compute_metrics(prediction, test_targets)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_descriptor_parts = [
        f"mode={graph_mode}",
        f"fusion={args.adjacency_fusion}",
        f"adaptive_init={args.adaptive_init if use_adaptive_graph else 'na'}",
        f"timegate={'on' if args.use_time_gate else 'off'}",
        f"graph={getattr(best_hparams, 'graph_layer_type', 'none')}",
    ]
    run_descriptor = "__".join(run_descriptor_parts)
    export_basename = f"{trainer.folder_names[best_model_idx]}__{run_descriptor}_{timestamp}"
    export_dir = ensure_directory(os.path.join(args.results_dir, export_basename))

    config_metadata = {
        "timestamp": timestamp,
        "run_folder": export_basename,
        "graph_mode": graph_mode,
        "adjacency_fusion": args.adjacency_fusion,
        "adaptive_init": args.adaptive_init if use_adaptive_graph else "na",
        "use_time_gate": bool(args.use_time_gate),
        "graph_layer_type": getattr(best_hparams, "graph_layer_type", "none"),
        "best_model_folder": trainer.folder_names[best_model_idx],
        "best_epoch": int(best_epoch_indices[best_model_idx]) + 1,
    }
    export_config_json(os.path.join(export_dir, "config.json"), args, best_hparams, config_metadata)

    np.save(os.path.join(export_dir, "y_true.npy"), test_targets)
    np.save(os.path.join(export_dir, "y_pred.npy"), prediction)
    if use_base_graph and adjacency_test is not None:
        np.save(os.path.join(export_dir, "adjacency_test.npy"), adjacency_test)
    if use_base_graph and "train" in adjacency_splits:
        np.save(os.path.join(export_dir, "adjacency_train.npy"), adjacency_splits["train"])

    # Save learned_decay coefficients if using learned_decay fusion mode
    if use_base_graph and args.adjacency_fusion == "learned_decay":
        fcgaga_layer = best_model.fcgaga_layers[0]
        decay_logit_var = getattr(fcgaga_layer, "decay_logit", None)
        if decay_logit_var is not None:
            decay_logit_value = float(tf.keras.backend.get_value(decay_logit_var))
            alpha_value = 1.0 / (1.0 + np.exp(-decay_logit_value))  # sigmoid

            # Compute weights for history_length time steps
            history_len = args.history_length
            positions = np.arange(history_len, dtype=np.float32)
            reversed_positions = positions[::-1]  # Reverse: most recent gets alpha^0=1
            weights = np.power(alpha_value, reversed_positions)
            weights = weights / np.sum(weights)  # Normalize

            # Save decay parameters
            decay_params = {
                "decay_logit": decay_logit_value,
                "alpha": alpha_value,
                "weights": weights,
                "history_length": history_len
            }
            np.save(os.path.join(export_dir, "learned_decay_params.npy"), decay_params)

            # Also save as human-readable JSON
            decay_params_json = {
                "decay_logit": float(decay_logit_value),
                "alpha": float(alpha_value),
                "weights": weights.tolist(),
                "history_length": int(history_len),
                "description": "Learned decay weights for temporal graph fusion (most recent to oldest)"
            }
            with open(os.path.join(export_dir, "learned_decay_params.json"), "w") as f:
                json.dump(decay_params_json, f, indent=2)

            print(f"\n{'='*70}")
            print(f"Learned Decay Parameters:")
            print(f"{'='*70}")
            print(f"  decay_logit: {decay_logit_value:.6f}")
            print(f"  alpha (decay rate): {alpha_value:.6f}")
            print(f"  Temporal weights (recent → old):")
            for i, w in enumerate(weights):
                print(f"    t-{history_len-i-1}: {w:.6f}")
            print(f"{'='*70}\n")
    if use_adaptive_graph:
        adj_layer_ref = getattr(best_model.fcgaga_layers[0], "adj_layer", None)
        if adj_layer_ref is not None:
            # Save the fused adjacency (what's actually used in training)
            adaptive_adj_fused = tf.keras.backend.get_value(adj_layer_ref(None))
            np.save(os.path.join(export_dir, "adaptive_adj.npy"), adaptive_adj_fused)

            # Extract and save PURE adaptive component: softmax(relu(E1 @ E2))
            # This is what the model actually learned, before fusion with base graph
            nodevec1 = tf.keras.backend.get_value(adj_layer_ref.nodevec1)
            nodevec2 = tf.keras.backend.get_value(adj_layer_ref.nodevec2)
            adaptive_pure = np.matmul(nodevec1, nodevec2)
            adaptive_pure = np.maximum(adaptive_pure, 0)  # ReLU
            # Row-wise softmax
            exp_adaptive = np.exp(adaptive_pure - np.max(adaptive_pure, axis=1, keepdims=True))
            adaptive_pure = exp_adaptive / np.sum(exp_adaptive, axis=1, keepdims=True)
            np.save(os.path.join(export_dir, "adaptive_adj_pure.npy"), adaptive_pure)

            # Export node-wise fusion weights (only for hybrid mode)
            if graph_mode == "hybrid" and hasattr(adj_layer_ref, 'fusion_weights') and adj_layer_ref.fusion_weights is not None:
                fusion_weights_logit = tf.keras.backend.get_value(adj_layer_ref.fusion_weights)
                m_values = np.exp(fusion_weights_logit)

                # Get training base graph row sums
                train_base_raw = provider.fused_mean_adjacency_raw(
                    split="train",
                    fusion_mode=args.adjacency_fusion
                )
                base_row_sums = np.sum(train_base_raw, axis=1)

                # Compute contribution ratios
                base_contrib = base_row_sums / (base_row_sums + m_values + 1)
                adaptive_contrib = m_values / (base_row_sums + m_values + 1)

                # Save analysis
                fusion_analysis = {
                    'm_values': m_values,
                    'm_init': fusion_weight_init if 'fusion_weight_init' in locals() else base_row_sums,
                    'base_row_sums': base_row_sums,
                    'base_contribution': base_contrib,
                    'adaptive_contribution': adaptive_contrib,
                }
                np.save(os.path.join(export_dir, "fusion_weights_analysis.npy"), fusion_analysis)

                # Print statistics
                print(f"\n{'='*70}")
                print(f"Learned Node-wise Fusion Weights (Hybrid Mode):")
                print(f"{'='*70}")
                print(f"  m values - mean: {np.mean(m_values):.2f}, std: {np.std(m_values):.2f}")
                print(f"  m values - range: [{np.min(m_values):.2f}, {np.max(m_values):.2f}]")
                print(f"  m_init - mean: {np.mean(base_row_sums):.2f}")
                print(f"  Base contribution - mean: {np.mean(base_contrib)*100:.2f}%")
                print(f"  Adaptive contribution - mean: {np.mean(adaptive_contrib)*100:.2f}%")
                print(f"{'='*70}\n")

        if base_graph_for_layer is not None:
            # Save normalized version (with self-loops, used for training)
            np.save(os.path.join(export_dir, "base_graph.npy"), base_graph_for_layer)

            # Save raw version (without self-loops, for analysis/visualization)
            if svd_init_adj is not None:
                # Use the raw matrix we already loaded for SVD (avoids recomputation)
                np.save(os.path.join(export_dir, "base_graph_raw.npy"), svd_init_adj)
            elif args.adaptive_init != "random":
                # Fallback: reload if svd_init_adj wasn't created
                raw_base = provider.fused_mean_adjacency_raw(split="train",
                                                             fusion_mode=args.adjacency_fusion)
                np.save(os.path.join(export_dir, "base_graph_raw.npy"), raw_base)
    train_samples = dataset.data['x_train'].shape[0]
    val_samples = dataset.data['x_val'].shape[0]
    test_samples = dataset.data['x_test'].shape[0]
    total_samples = train_samples + val_samples + test_samples
    test_global_indices = np.arange(total_samples - test_samples, total_samples)

    print(f"\n{'='*70}")
    print(f"Dataset Split Summary:")
    print(f"{'='*70}")
    print(f"  Train samples: {train_samples}")
    print(f"  Val samples: {val_samples}")
    print(f"  Test samples: {test_samples}")
    print(f"  Total samples: {total_samples}")
    print(f"  History length: {best_hparams.history_length}")
    print(f"  Horizon: {best_hparams.horizon}")
    print(f"{'='*70}\n")

    timestamps = load_year_timestamps(args.data_csv)
    node_labels = load_node_labels(args.data_csv)
    export_prediction_csv(
        os.path.join(export_dir, "prediction_by_node_year.csv"),
        prediction,
        test_targets,
        timestamps,
        test_global_indices,
        best_hparams.history_length,
        node_labels)
    export_node_error_csv(
        os.path.join(export_dir, "node_error_summary.csv"),
        prediction,
        test_targets,
        node_labels)

    metrics_payload = {
        "best_model_folder": trainer.folder_names[best_model_idx],
        "best_epoch": int(best_epoch_indices[best_model_idx]) + 1,
        "val_mae": float(best_model_val),
        "test_metrics": metrics_overall,
        "per_horizon_metrics": per_horizon_metrics,
    }
    with open(os.path.join(export_dir, "metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    results_index_row = {
        "run_folder": export_basename,
        "timestamp": timestamp,
        "graph_mode": graph_mode,
        "adjacency_fusion": args.adjacency_fusion,
        "adaptive_init": args.adaptive_init if use_adaptive_graph else "na",
        "use_time_gate": "on" if args.use_time_gate else "off",
        "graph_layer_type": getattr(best_hparams, "graph_layer_type", "none"),
        "repeat": str(best_hparams.repeat),
        "history_length": str(best_hparams.history_length),
        "horizon": str(best_hparams.horizon),
        "apt_size": str(getattr(best_hparams, "apt_size", 10)),
        "val_mae": f"{best_model_val:.6f}",
        "test_mae": f"{metrics_overall['MAE']:.6f}",
        "test_rmse": f"{metrics_overall['RMSE']:.6f}",
        "test_mape": f"{metrics_overall['MAPE']:.6f}",
        "test_r2": f"{metrics_overall['R2']:.6f}",
    }
    append_results_index(os.path.join(args.results_dir, "results_index.csv"), results_index_row)

    if trainer.best_weight_paths:
        best_weight_src = trainer.best_weight_paths[best_model_idx]
        if os.path.exists(best_weight_src):
            shutil.copy2(best_weight_src, os.path.join(export_dir, "best_weights.weights.h5"))

    saved_model_dir = os.path.join(export_dir, "dynamic_saved_model")
    export_saved_model(best_model.model, saved_model_dir)
    print(f"Dynamic snapshot run complete. Artifacts saved to {export_dir}")


if __name__ == "__main__":
    main()
