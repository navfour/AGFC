"""Metrics callbacks that feed adjacency inputs to the model."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from dynamic.dynamic_utils import masked_mae_np, masked_mape_np, masked_rmse_np


class DynamicMetricsCallback(tf.keras.callbacks.Callback):
    """Metrics reporting for dynamic models that may or may not require adjacencies."""

    def __init__(self, dataset, adjacency_splits, expect_adjacency_input: bool = True):
        super().__init__()
        self.validation_data = dataset.get_sequential_batch(
            batch_size=len(dataset.data['x_val']),
            split='val').__next__()
        self.test_data = dataset.get_sequential_batch(
            batch_size=len(dataset.data['x_test']),
            split='test').__next__()
        self.expect_adjacency_input = expect_adjacency_input
        self.validation_adj = None
        self.test_adj = None
        if self.expect_adjacency_input:
            self.validation_adj = adjacency_splits['val']
            self.test_adj = adjacency_splits['test']

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_inputs = {
            "history": self.validation_data["x"][..., 0],
            "node_id": self.validation_data["node_id"],
            "time_of_day": self.validation_data["x"][..., 1],
        }
        test_inputs = {
            "history": self.test_data["x"][..., 0],
            "node_id": self.test_data["node_id"],
            "time_of_day": self.test_data["x"][..., 1],
        }
        if self.expect_adjacency_input:
            val_inputs["adjacency"] = self.validation_adj
            test_inputs["adjacency"] = self.test_adj

        prediction_val = self._extract_targets(self.model.predict(val_inputs))
        prediction_test = self._extract_targets(self.model.predict(test_inputs))

        logs['mae_val'] = masked_mae_np(preds=prediction_val, labels=self.validation_data['y'], null_val=0)
        logs['mae_test'] = masked_mae_np(preds=prediction_test, labels=self.test_data['y'], null_val=0)

        for h in range(prediction_test.shape[-1]):
            logs[f'mae_val_h{h + 1}'] = masked_mae_np(
                preds=prediction_val[..., h], labels=self.validation_data['y'][..., h], null_val=0)
            logs[f'mae_test_h{h + 1}'] = masked_mae_np(
                preds=prediction_test[..., h], labels=self.test_data['y'][..., h], null_val=0)
            logs[f'mape_val_h{h + 1}'] = masked_mape_np(
                preds=prediction_val[..., h], labels=self.validation_data['y'][..., h], null_val=0)
            logs[f'rmse_val_h{h + 1}'] = masked_rmse_np(
                preds=prediction_val[..., h], labels=self.validation_data['y'][..., h], null_val=0)
            logs[f'mape_test_h{h + 1}'] = masked_mape_np(
                preds=prediction_test[..., h], labels=self.test_data['y'][..., h], null_val=0)
            logs[f'rmse_test_h{h + 1}'] = masked_rmse_np(
                preds=prediction_test[..., h], labels=self.test_data['y'][..., h], null_val=0)

    @staticmethod
    def _extract_targets(prediction):
        return prediction['targets'] if isinstance(prediction, dict) else prediction
