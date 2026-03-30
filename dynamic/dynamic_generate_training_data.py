"""Generate train/val/test splits from a CSV time series (copied for dynamic use)."""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


def generate_graph_seq2seq_io_data(
        df: pd.DataFrame,
        x_offsets: np.ndarray,
        y_offsets: np.ndarray,
        add_time_in_day: bool = False,
        add_day_in_week: bool = False,
):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    df = pd.read_csv(args.data_csv, index_col=0)
    df.index = pd.to_datetime(df.index)
    zero_mask = (df > 0).astype(np.float32)
    df = df.replace(0, np.nan)
    df = df.fillna(method='ffill')
    df = df.fillna(0.0)
    x_offsets = np.sort(np.arange(1 - args.history_length, 1, 1))
    y_offsets = np.sort(np.arange(1, 1 + args.horizon, 1))
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
    )
    x_mask, y_mask = generate_graph_seq2seq_io_data(
        zero_mask,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
    )
    num_samples = x.shape[0]
    num_test = 5  # Fixed 5 samples for test set
    num_train = round(num_samples * 0.8)  # 80% train set
    num_val = num_samples - num_test - num_train  # Remaining for validation

    x_train, y_train = x[:num_train], y[:num_train] * y_mask[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val] * y_mask[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:] * y_mask[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        os.makedirs(args.output_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s-history-%d-horizon-%d.npz" % (cat, args.history_length, args.horizon)),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output_dir", type=str, default="data/", help="Output directory.")
        parser.add_argument("--data_csv",
                            type=str,
                            default="data/citation_cs/citation_cs_small.csv",
                            help="Input CSV with time series.")
        parser.add_argument("--horizon", type=int, default=12)
        parser.add_argument("--history_length", type=int, default=12)
        args = parser.parse_args()
    generate_train_val_test(args)


if __name__ == "__main__":
    main()
