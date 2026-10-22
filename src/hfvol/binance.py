"""Binance USD-M futures (public data.binance.vision archive) -> Optiver format.

Each UTC day is cut into 72 non-overlapping 20-minute blocks. The first 10
minutes of a block become one Optiver "bucket" (book + trade data,
seconds_in_bucket 0..599); the target is the realized volatility of the
1-second WAP over the following 10 minutes - the same definition as the
Optiver competition. time_id = day_index * 72 + block, so time_ids are
chronological (the true order is kept in time_ids.csv for purged CV).

Book data is the best bid/ask (level 1) from `bookTicker`, sampled at the
last update of every second; Binance's archive has no level-2 history, so
the level-2 columns are absent and level-2 features are skipped for this
dataset. Prices are normalised by the WAP at the first second of each bucket,
like Optiver's anonymised prices. Trades come from `aggTrades` aggregated per
second (VWAP price, size, order_count, plus signed taker volume).
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/daily"
BLOCK = 1200
BUCKET = 600
BLOCKS_PER_DAY = 86400 // BLOCK


def _fetch_zip_csv(url: str, usecols, names=None) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            head = f.read(64)
        has_header = not head[:1].isdigit()
        with z.open(name) as f:
            return pd.read_csv(f, usecols=usecols, header=0 if has_header else None, names=names,
                               engine="c")


def _book_per_second(df: pd.DataFrame, day_start_ms: int) -> pd.DataFrame:
    sec = ((df["transaction_time"].to_numpy() - day_start_ms) // 1000).astype(np.int64)
    df = df.assign(sec=sec)
    df = df[(df.sec >= 0) & (df.sec < 86400)]
    last = df.groupby("sec").last()
    full = last.reindex(np.arange(86400)).ffill().bfill()
    return full


def _trades_per_second(df: pd.DataFrame, day_start_ms: int) -> pd.DataFrame:
    sec = ((df["transact_time"].to_numpy() - day_start_ms) // 1000).astype(np.int64)
    px = df["price"].to_numpy(float)
    qty = df["quantity"].to_numpy(float)
    sign = np.where(df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"]), -1.0, 1.0)
    t = pd.DataFrame({"sec": sec, "pq": px * qty, "size": qty, "signed": sign * qty})
    t = t[(t.sec >= 0) & (t.sec < 86400)]
    g = t.groupby("sec").agg(pq=("pq", "sum"), size=("size", "sum"), signed_size=("signed", "sum"),
                             order_count=("size", "size"))
    g["price"] = g["pq"] / g["size"]
    return g.drop(columns="pq")


@dataclass
class DayResult:
    book: pd.DataFrame
    trade: pd.DataFrame
    target: pd.DataFrame
    meta: pd.DataFrame


def process_day(symbol: str, date: str, day_index: int) -> DayResult:
    day_start_ms = int(pd.Timestamp(date, tz="UTC").value // 1_000_000)
    bt = _fetch_zip_csv(f"{BASE}/bookTicker/{symbol}/{symbol}-bookTicker-{date}.zip",
                        usecols=["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty",
                                 "transaction_time"])
    book = _book_per_second(bt, day_start_ms)
    tr = _fetch_zip_csv(f"{BASE}/aggTrades/{symbol}/{symbol}-aggTrades-{date}.zip",
                        usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
    trades = _trades_per_second(tr, day_start_ms)

    bid, ask = book["best_bid_price"].to_numpy(), book["best_ask_price"].to_numpy()
    bq, aq = book["best_bid_qty"].to_numpy(), book["best_ask_qty"].to_numpy()
    wap = (bid * aq + ask * bq) / (bq + aq)

    books, trade_rows, targets, metas = [], [], [], []
    for blk in range(BLOCKS_PER_DAY):
        s0 = blk * BLOCK
        tid = day_index * BLOCKS_PER_DAY + blk
        norm = wap[s0]
        sl = slice(s0, s0 + BUCKET)
        books.append(pd.DataFrame({
            "time_id": tid,
            "seconds_in_bucket": np.arange(BUCKET, dtype=np.int16),
            "bid_price1": bid[sl] / norm, "ask_price1": ask[sl] / norm,
            "bid_size1": bq[sl], "ask_size1": aq[sl],
        }))
        tw = trades.loc[(trades.index >= s0) & (trades.index < s0 + BUCKET)]
        if len(tw):
            trade_rows.append(pd.DataFrame({
                "time_id": tid, "seconds_in_bucket": (tw.index - s0).astype(np.int16),
                "price": tw["price"].to_numpy() / norm, "size": tw["size"].to_numpy(),
                "order_count": tw["order_count"].to_numpy(), "signed_size": tw["signed_size"].to_numpy(),
            }))
        fut = np.log(wap[s0 + BUCKET: s0 + BLOCK])
        targets.append((tid, float(np.sqrt(np.sum(np.diff(fut) ** 2)))))
        metas.append((tid, date, blk, day_start_ms + s0 * 1000, float(norm), float(np.median(np.diff(np.unique(bid))[:50])) if len(np.unique(bid)) > 1 else np.nan))
    return DayResult(
        book=pd.concat(books, ignore_index=True),
        trade=pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame(),
        target=pd.DataFrame(targets, columns=["time_id", "target"]),
        meta=pd.DataFrame(metas, columns=["time_id", "date", "block", "start_ms", "price_level", "tick"]),
    )


def write_day(out_dir: str, stock_id: int, date: str, res: DayResult) -> None:
    bdir = os.path.join(out_dir, "book_train.parquet", f"stock_id={stock_id}")
    tdir = os.path.join(out_dir, "trade_train.parquet", f"stock_id={stock_id}")
    os.makedirs(bdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    res.book.to_parquet(os.path.join(bdir, f"{date}.parquet"), index=False)
    if len(res.trade):
        res.trade.to_parquet(os.path.join(tdir, f"{date}.parquet"), index=False)
    t = res.target.assign(stock_id=stock_id)[["stock_id", "time_id", "target"]]
    t.to_csv(os.path.join(out_dir, "targets", f"{stock_id}_{date}.csv"), index=False)
    res.meta.assign(stock_id=stock_id).to_csv(os.path.join(out_dir, "meta", f"{stock_id}_{date}.csv"), index=False)
