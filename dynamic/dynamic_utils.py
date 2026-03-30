"""Utility functions for masking metrics (copied for dynamic use)."""

from __future__ import annotations

import numpy as np


def masked_mae_np(preds, labels, null_val=np.nan):
    mask = np.not_equal(labels, null_val)
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    loss = np.abs(preds - labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), 0, loss)
    return np.mean(loss)


def masked_mape_np(preds, labels, null_val=np.nan):
    mask = np.not_equal(labels, null_val)
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    loss = np.abs((preds - labels) / labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), 0, loss)
    return np.mean(loss) * 100


def masked_rmse_np(preds, labels, null_val=np.nan):
    mask = np.not_equal(labels, null_val)
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = np.where(np.isnan(loss), 0, loss)
    return np.sqrt(np.mean(loss))


def masked_r2_np(preds, labels, null_val=np.nan):
    mask = np.not_equal(labels, null_val)
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    labels_mean = np.mean(labels[mask.astype(bool)])
    total = (labels - labels_mean) ** 2
    total = total * mask
    total = np.where(np.isnan(total), 0, total)
    unexplained = (preds - labels) ** 2
    unexplained = unexplained * mask
    unexplained = np.where(np.isnan(unexplained), 0, unexplained)
    total_sum = np.sum(total)
    if total_sum == 0:
        return 0.0
    return 1 - np.sum(unexplained) / total_sum
