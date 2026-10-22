"""Loading Optiver-format data into dense per-second grids.

Directory layout (Optiver Realized Volatility Prediction, Kaggle 2021):
    book_train.parquet/stock_id=<k>/*.parquet
        time_id, seconds_in_bucket, bid_price1, ask_price1, [bid_price2, ask_price2,]
        bid_size1, ask_size1, [bid_size2, ask_size2]
    trade_train.parquet/stock_id=<k>/*.parquet
        time_id, seconds_in_bucket, price, size, order_count, [signed_size]
    train.csv: stock_id, time_id, target
    time_ids.csv (optional): true chronological order (crypto / synthetic data)

Optiver stores the book only when it changes; we forward-fill onto a full
600-second grid so every feature can be computed with array operations.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

N_SEC = 600
BOOK_COLS = ["bid_price1", "ask_price1", "bid_size1", "ask_size1", "bid_price2", "ask_price2",
             "bid_size2", "ask_size2"]


@dataclass
class StockGrid:
    stock_id: int
    time_ids: np.ndarray                 # (n,) sorted
    book: dict[str, np.ndarray]          # column -> (n, 600), forward filled
    trade_size: np.ndarray               # (n, 600), 0 where no trade
    trade_price: np.ndarray              # (n, 600), NaN where no trade
    trade_count: np.ndarray              # (n, 600)
    signed_size: np.ndarray | None = None  # (n, 600) taker-signed volume if available
    has_l2: bool = field(default=False)


def stock_ids(data_dir: str) -> list[int]:
    dirs = glob.glob(os.path.join(data_dir, "book_train.parquet", "stock_id=*"))
    return sorted(int(d.rsplit("=", 1)[1]) for d in dirs)


def _ffill_rows(a: np.ndarray) -> np.ndarray:
    """Forward fill NaNs along axis 1, then back fill leading NaNs."""
    mask = np.isfinite(a)
    idx = np.where(mask, np.arange(a.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = a[np.arange(a.shape[0])[:, None], idx]
    # leading NaNs: use first valid value of the row
    first = np.argmax(mask, axis=1)
    lead = np.arange(a.shape[1])[None, :] < first[:, None]
    out = np.where(lead, a[np.arange(a.shape[0]), first][:, None], out)
    return out


def _to_grid(df: pd.DataFrame, col: str, rows: np.ndarray, n: int, fill=np.nan) -> np.ndarray:
    g = np.full((n, N_SEC), fill, dtype=np.float64)
    g[rows, df["seconds_in_bucket"].to_numpy(int)] = df[col].to_numpy(np.float64)
    return g


def load_stock(data_dir: str, stock_id: int, time_ids: np.ndarray | None = None) -> StockGrid:
    book = pd.read_parquet(os.path.join(data_dir, "book_train.parquet", f"stock_id={stock_id}"))
    if time_ids is None:
        time_ids = np.sort(book["time_id"].unique())
    pos = {t: i for i, t in enumerate(time_ids)}
    book = book[book["time_id"].isin(pos)]
    rows = book["time_id"].map(pos).to_numpy()
    n = len(time_ids)
    has_l2 = "bid_price2" in book.columns
    grids = {}
    for c in BOOK_COLS:
        if c in book.columns:
            grids[c] = _ffill_rows(_to_grid(book, c, rows, n))

    tdir = os.path.join(data_dir, "trade_train.parquet", f"stock_id={stock_id}")
    if os.path.isdir(tdir):
        tr = pd.read_parquet(tdir)
        tr = tr[tr["time_id"].isin(pos)]
    else:
        tr = pd.DataFrame(columns=["time_id", "seconds_in_bucket", "price", "size", "order_count"])
    trows = tr["time_id"].map(pos).to_numpy()
    size = _to_grid(tr, "size", trows, n, 0.0)
    price = _to_grid(tr, "price", trows, n, np.nan)
    count = _to_grid(tr, "order_count", trows, n, 0.0)
    signed = _to_grid(tr, "signed_size", trows, n, 0.0) if "signed_size" in tr.columns else None
    return StockGrid(stock_id, np.asarray(time_ids), grids, size, price, count, signed, has_l2)


def load_targets(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "train.csv"))


def load_time_order(data_dir: str) -> pd.Series | None:
    """time_id -> chronological rank, if the dataset ships one."""
    path = os.path.join(data_dir, "time_ids.csv")
    if not os.path.exists(path):
        return None
    m = pd.read_csv(path)
    if "start_ms" in m.columns:
        first = m.groupby("time_id")["start_ms"].min()
    else:
        first = m.groupby("time_id")["order"].min()
    return first.rank(method="first").astype(int) - 1
