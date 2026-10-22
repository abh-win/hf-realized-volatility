"""Synthetic market data in the exact Optiver competition format.

Used to exercise the full pipeline (including level-2 features) when the
Kaggle data is not available. The generator encodes the stylised facts the
models are supposed to exploit:

* Stochastic volatility with a factor structure: log-vol = stock base level
  + beta * market factor + sector factor + idiosyncratic factor, each an AR(1)
  across consecutive 20-minute blocks (volatility clustering, cross-sectional
  co-movement, clusters of similar stocks).
* Within a block the instantaneous volatility follows a mean-reverting OU
  path at 1-second resolution, so the last seconds of the feature window
  carry more information about the next 10 minutes (the reason for
  multi-scale windows).
* Price jumps (Poisson) and microstructure noise from discrete ticks and
  bid-ask bounce in trade prices.
* An order-book imbalance process that predicts short-term returns, and
  depth that thins out when volatility is high.
* Real price levels on a 0.01 tick grid. Prices are normalised per bucket as
  in Optiver, but the normalised tick size 0.01 / P reveals the price level,
  so time_ids (shuffled, like Optiver's) can be put back into order.

Target = realized volatility of the 1-second WAP over the 10 minutes following
the feature window, as in the competition.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

SEC = 600
TICK = 0.01


def _ar1(rng, n, phi, sigma, size=None):
    shape = (n,) if size is None else (size, n)
    x = np.zeros(shape)
    eps = rng.normal(0, sigma, shape)
    for t in range(1, n):
        x[..., t] = phi * x[..., t - 1] + eps[..., t]
    return x


def generate(out_dir: str, n_stocks: int = 30, n_times: int = 1200, n_sectors: int = 4, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    market = _ar1(rng, n_times, 0.97, 0.10)
    sector = _ar1(rng, n_times, 0.95, 0.08, size=n_sectors)
    sector_of = rng.integers(0, n_sectors, n_stocks)
    base = rng.normal(np.log(0.0028), 0.30, n_stocks)
    beta = rng.uniform(0.7, 1.3, n_stocks)
    price0 = np.exp(rng.uniform(np.log(15), np.log(400), n_stocks))
    trade_rate = rng.uniform(0.05, 0.6, n_stocks)
    depth_scale = rng.uniform(50, 500, n_stocks)

    perm = rng.permutation(n_times)  # chronological index -> anonymised time_id
    targets, truth = [], pd.DataFrame({"time_id": perm, "order": np.arange(n_times)})
    for i in range(n_stocks):
        idio = _ar1(rng, n_times, 0.85, 0.15)
        level = base[i] + beta[i] * market + sector[sector_of[i]] + idio  # log 10-min RV
        sig_block = np.exp(level)                                         # (T,)

        # 1-second log-vol OU around the block level, over feature+target halves.
        n = 2 * SEC
        theta, s_ou = 1 / 200.0, 0.04
        ou = np.zeros((n_times, n))
        ou[:, 0] = rng.normal(0, s_ou / np.sqrt(2 * theta), n_times)
        shocks = rng.normal(0, s_ou, (n_times, n))
        for s in range(1, n):
            ou[:, s] = (1 - theta) * ou[:, s - 1] + shocks[:, s]
        # Regime drift inside the block: the target half can move away from the level.
        drift = rng.normal(0, 0.15, n_times)[:, None] * np.linspace(0, 1, n)[None, :]
        sig = sig_block[:, None] * np.exp(ou + drift) / np.sqrt(SEC)       # per-second vol

        imb_lat = np.zeros((n_times, n))
        e = rng.normal(0, 0.35, (n_times, n))
        for s in range(1, n):
            imb_lat[:, s] = 0.9 * imb_lat[:, s - 1] + e[:, s]
        imb = np.tanh(imb_lat)
        ret = sig * rng.standard_normal((n_times, n)) + 0.25 * sig * np.roll(imb, 1, axis=1)
        jumps = rng.random((n_times, n)) < 0.0015
        ret += jumps * rng.normal(0, 6, (n_times, n)) * sig

        # Real price level evolves block to block with the block volatility.
        logp_block = np.log(price0[i]) + np.cumsum(sig_block * rng.standard_normal(n_times) * 1.5)
        logp = logp_block[:, None] + np.cumsum(ret, axis=1)
        mid = np.exp(logp)
        spread_ticks = 1 + rng.poisson(np.clip(sig * np.sqrt(SEC) / (TICK / mid) * 0.02, 0, 8))
        bid1 = np.floor((mid - spread_ticks * TICK / 2) / TICK) * TICK
        ask1 = bid1 + spread_ticks * TICK
        bid2 = bid1 - TICK * (1 + rng.poisson(0.4, (n_times, n)))
        ask2 = ask1 + TICK * (1 + rng.poisson(0.4, (n_times, n)))
        depth = depth_scale[i] * np.exp(-0.6 * (np.log(sig * np.sqrt(SEC)) - base[i])) * \
            rng.lognormal(0, 0.4, (n_times, n))
        bs1 = np.maximum(1, np.round(depth * (1 + imb) / 2))
        as1 = np.maximum(1, np.round(depth * (1 - imb) / 2))
        bs2 = np.maximum(1, np.round(depth * rng.uniform(0.5, 1.5, (n_times, n))))
        as2 = np.maximum(1, np.round(depth * rng.uniform(0.5, 1.5, (n_times, n))))
        wap = (bid1 * as1 + ask1 * bs1) / (bs1 + as1)

        # Target: realized vol of WAP over the second half.
        lw = np.log(wap[:, SEC:])
        rv_target = np.sqrt(np.sum(np.diff(lw, axis=1) ** 2, axis=1))

        norm = wap[:, :1]
        f = slice(0, SEC)
        cols = {
            "bid_price1": bid1[:, f] / norm, "ask_price1": ask1[:, f] / norm,
            "bid_price2": bid2[:, f] / norm, "ask_price2": ask2[:, f] / norm,
            "bid_size1": bs1[:, f], "ask_size1": as1[:, f],
            "bid_size2": bs2[:, f], "ask_size2": as2[:, f],
        }
        # Keep only seconds where the book changed (Optiver stores updates only).
        stacked = np.stack([cols[c] for c in cols], axis=-1)
        changed = np.ones((n_times, SEC), dtype=bool)
        changed[:, 1:] = np.any(stacked[:, 1:] != stacked[:, :-1], axis=-1)
        tt, ss = np.nonzero(changed)
        book = pd.DataFrame({"time_id": perm[tt].astype(np.int32), "seconds_in_bucket": ss.astype(np.int16)})
        for c in cols:
            book[c] = cols[c][tt, ss].astype(np.float32 if "size" in c else np.float64)
        bdir = os.path.join(out_dir, "book_train.parquet", f"stock_id={i}")
        os.makedirs(bdir, exist_ok=True)
        book.sort_values(["time_id", "seconds_in_bucket"]).to_parquet(os.path.join(bdir, "part-0.parquet"), index=False)

        lam = trade_rate[i] * (sig[:, f] * np.sqrt(SEC) / np.exp(base[i])) ** 0.8
        cnt = rng.poisson(lam)
        tt, ss = np.nonzero(cnt)
        buy = rng.random(len(tt)) < (1 + imb[tt, ss]) / 2
        price = np.where(buy, ask1[tt, ss], bid1[tt, ss]) / norm[tt, 0]
        size = np.round(rng.lognormal(3, 1, len(tt)) * cnt[tt, ss])
        trade = pd.DataFrame({"time_id": perm[tt].astype(np.int32), "seconds_in_bucket": ss.astype(np.int16),
                              "price": price, "size": size, "order_count": cnt[tt, ss].astype(np.int32)})
        tdir = os.path.join(out_dir, "trade_train.parquet", f"stock_id={i}")
        os.makedirs(tdir, exist_ok=True)
        trade.sort_values(["time_id", "seconds_in_bucket"]).to_parquet(os.path.join(tdir, "part-0.parquet"), index=False)

        targets.append(pd.DataFrame({"stock_id": i, "time_id": perm, "target": rv_target}))

    pd.concat(targets).sort_values(["stock_id", "time_id"]).to_csv(os.path.join(out_dir, "train.csv"), index=False)
    # The true chronology is withheld from the pipeline (like Optiver) and only
    # used to score time-order recovery.
    truth.to_csv(os.path.join(out_dir, "truth_time_order.csv"), index=False)
    pd.DataFrame({"stock_id": range(n_stocks), "sector": sector_of}).to_csv(
        os.path.join(out_dir, "truth_sectors.csv"), index=False)
