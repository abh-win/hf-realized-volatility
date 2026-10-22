"""Time-series cross-validation with purging and embargo (López de Prado, 2018, ch. 7).

Samples are (stock, time_id) pairs; all stocks of a time_id move together, so
splits are made over time_ids in chronological order. Each test fold is a
contiguous block of time. Training data excludes `purge` time_ids immediately
before the test block (their target windows run into it) and an `embargo`
fraction of time_ids right after it (serial correlation leaks backwards).
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd


def chronological(order: pd.Series) -> np.ndarray:
    """time_ids sorted by their rank."""
    return order.sort_values().index.to_numpy()


def holdout_split(order: pd.Series, test_frac: float = 0.2, gap: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Last `test_frac` of time as the final test set, with a purge gap."""
    t = chronological(order)
    n_test = int(round(len(t) * test_frac))
    return t[: len(t) - n_test - gap], t[len(t) - n_test:]


def purged_kfold(time_ids: np.ndarray, order: pd.Series, n_splits: int = 5, purge: int = 1,
                 embargo: float = 0.01) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yields (train_time_ids, test_time_ids) over the given time_ids."""
    t = order.reindex(time_ids).sort_values().index.to_numpy()
    n = len(t)
    emb = int(math.ceil(embargo * n))
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    for k in range(n_splits):
        lo, hi = bounds[k], bounds[k + 1]
        test = t[lo:hi]
        keep = np.ones(n, dtype=bool)
        keep[max(0, lo - purge): min(n, hi + emb)] = False
        yield t[keep], test
