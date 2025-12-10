from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    timestamp_col: str = "num__timestamp_ts",
    test_days: int = 20,
    val_days: int = 20,
    gap_days: int = 0,
) -> TimeSplit:
    """
    Deterministic time-based split with a gap to reduce leakage.
    Returns TimeSplit(train, val, test).
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"{timestamp_col} not in dataframe")

    ts = pd.to_numeric(df[timestamp_col], errors="coerce")
    t_max = ts.max()
    gap = pd.Timedelta(days=gap_days).total_seconds()
    cut_test = t_max - pd.Timedelta(days=test_days).total_seconds()
    cut_val = cut_test - pd.Timedelta(days=val_days).total_seconds()

    train = df[ts < cut_val - gap]
    val = df[(ts >= cut_val) & (ts < cut_test - gap)]
    test = df[ts >= cut_test]

    for name, part in (("train", train), ("val", val), ("test", test)):
        if part.empty:
            raise ValueError(f"{name} split is empty; adjust window sizes")

    return TimeSplit(train=train, val=val, test=test)



def describe_split(split: TimeSplit, target: str) -> Dict[str, float]:
    """
    Quick summary: sizes and positive rates for each partition.
    """
    def _stats(part: pd.DataFrame) -> Tuple[int, int, float]:
        y = part[target]
        return len(part), int(y.sum()), float(y.mean())

    return {
        "train_n": _stats(split.train)[0],
        "train_pos": _stats(split.train)[1],
        "train_pos_rate": _stats(split.train)[2],
        "val_n": _stats(split.val)[0],
        "val_pos": _stats(split.val)[1],
        "val_pos_rate": _stats(split.val)[2],
        "test_n": _stats(split.test)[0],
        "test_pos": _stats(split.test)[1],
        "test_pos_rate": _stats(split.test)[2],
    }
