"""Per-bucket feature engineering from 1-second book and trade grids.

Feature groups (column prefixes in brackets):

* Realized measures [rv1_, rv2_, bv1_, jump1_, rsp1_, rsn1_, rq1]
  RV from the level-1 WAP (and level-2 WAP when depth is available),
  bipower variation, jump variation RV-BV, signed semivariances, quarticity.
* Noise-robust estimators [tsrv1, rk1]
  Two-scale RV and the Parzen realized kernel; tick-level WAP returns carry
  bid-ask bounce, which inflates plain RV.
* Microstructure [spread_, imb1_, imb2_, ofi_, depth_, wapbal, kyle, amihud, range, n_upd]
  Relative spread level/dynamics, L1/L2 depth imbalance, Cont-Kukanov-Stoikov
  (2014) order-flow imbalance, depth, |WAP1-WAP2|, Kyle's lambda (price impact
  of signed volume), Amihud illiquidity, high-low range, book update count.
* Trades [trade_]
  Trade-second count, volume, order count, trade-price RV, signed-volume share.
* Multi-scale windows: most of the above over the last 600/500/300/200/100
  seconds (suffix _w{W}) - the HAR idea applied inside the bucket.
* tick_norm / price_level: the smallest normalised price increment reveals the
  real price level (tick / P), used to recover time ordering across buckets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import realized as R
from .data import StockGrid

WINDOWS = (600, 500, 300, 200, 100)


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return np.where(np.isfinite(out), out, 0.0)


def wap(bid, ask, bs, as_):
    return (bid * as_ + ask * bs) / (bs + as_)


def order_flow_imbalance(bid, ask, bs, as_):
    """Cont, Kukanov & Stoikov (2014) OFI increments between consecutive seconds."""
    b0, b1 = bid[:, :-1], bid[:, 1:]
    a0, a1 = ask[:, :-1], ask[:, 1:]
    q0b, q1b = bs[:, :-1], bs[:, 1:]
    q0a, q1a = as_[:, :-1], as_[:, 1:]
    return (b1 >= b0) * q1b - (b1 <= b0) * q0b - (a1 <= a0) * q1a + (a1 >= a0) * q0a


def tick_size(bid, ask):
    """Smallest positive gap between distinct prices seen in each bucket."""
    p = np.sort(np.concatenate([bid, ask], axis=1), axis=1)
    d = np.diff(p, axis=1)
    d = np.where(d > 1e-9, d, np.inf)
    t = d.min(axis=1)
    return np.where(np.isfinite(t), t, np.nan)


def lee_ready_sign(trade_price, mid):
    s = np.sign(trade_price - mid)
    s = np.where(np.isfinite(s), s, 0.0)
    # Tick rule where the trade is at the mid: compare with the previous trade price.
    prev = pd.DataFrame(trade_price.T).ffill().shift(1).to_numpy().T
    tick = np.sign(trade_price - prev)
    tick = np.where(np.isfinite(tick), tick, 0.0)
    return np.where(s == 0, tick, s)


def stock_features(g: StockGrid) -> pd.DataFrame:
    b = g.book
    bid1, ask1, bs1, as1 = b["bid_price1"], b["ask_price1"], b["bid_size1"], b["ask_size1"]
    w1 = wap(bid1, ask1, bs1, as1)
    lw1 = np.log(w1)
    r1 = R._clean(np.diff(lw1, axis=1))  # (n, 599)
    f: dict[str, np.ndarray] = {}

    if g.has_l2:
        bid2, ask2, bs2, as2 = b["bid_price2"], b["ask_price2"], b["bid_size2"], b["ask_size2"]
        w2 = wap(bid2, ask2, bs2, as2)
        r2 = R._clean(np.diff(np.log(w2), axis=1))

    spread = _safe_div(ask1, bid1) - 1.0
    imb1 = _safe_div(bs1 - as1, bs1 + as1)
    depth1 = np.log1p(bs1 + as1)
    ofi = np.concatenate([np.zeros((r1.shape[0], 1)), order_flow_imbalance(bid1, ask1, bs1, as1)], axis=1)
    mean_depth = np.maximum((bs1 + as1).mean(axis=1, keepdims=True) / 2, 1e-9)
    ofi_n = ofi / mean_depth
    if g.has_l2:
        imb2 = _safe_div(bs1 + bs2 - as1 - as2, bs1 + bs2 + as1 + as2)

    tsize = g.trade_size
    tcount = g.trade_count
    has_trade = tsize > 0

    for W in WINDOWS:
        rw = r1[:, -(W - 1):]
        sl = slice(600 - W, 600)
        s = f"_w{W}"
        f["rv1" + s] = R.realized_volatility(rw)
        f["bv1" + s] = np.sqrt(R.bipower_variation(rw))
        f["jump1" + s] = np.sqrt(R.jump_variation(rw))
        rsp, rsn = R.semivariances(rw)
        f["rsp1" + s], f["rsn1" + s] = np.sqrt(rsp), np.sqrt(rsn)
        if g.has_l2:
            f["rv2" + s] = R.realized_volatility(r2[:, -(W - 1):])
            f["imb2" + s] = imb2[:, sl].mean(axis=1)
        f["spread_mean" + s] = spread[:, sl].mean(axis=1)
        f["imb1" + s] = imb1[:, sl].mean(axis=1)
        f["ofi" + s] = ofi_n[:, sl].sum(axis=1)
        f["ofi_abs" + s] = np.abs(ofi_n[:, sl]).sum(axis=1)
        f["trade_size" + s] = np.log1p(tsize[:, sl].sum(axis=1))
        f["trade_secs" + s] = has_trade[:, sl].sum(axis=1)
        f["trade_orders" + s] = tcount[:, sl].sum(axis=1)

    # Full-window only.
    f["rq1"] = R.realized_quarticity(r1)
    f["tsrv1"] = np.sqrt(R.tsrv(lw1, K=10))
    f["rk1"] = np.sqrt(R.realized_kernel(r1))
    f["spread_std"] = spread.std(axis=1)
    f["spread_trend"] = spread[:, 300:].mean(axis=1) - spread[:, :300].mean(axis=1)
    f["depth_mean"] = depth1.mean(axis=1)
    f["depth_trend"] = depth1[:, 300:].mean(axis=1) - depth1[:, :300].mean(axis=1)
    f["range"] = np.log(w1.max(axis=1) / w1.min(axis=1))
    f["ret_sum"] = lw1[:, -1] - lw1[:, 0]
    f["n_upd"] = (np.abs(np.diff(bid1, axis=1)) + np.abs(np.diff(ask1, axis=1)) +
                  np.abs(np.diff(bs1, axis=1)) + np.abs(np.diff(as1, axis=1)) > 0).sum(axis=1)
    if g.has_l2:
        f["wapbal"] = np.abs(w1 - w2).mean(axis=1)
        f["spread2_mean"] = (_safe_div(ask2, bid2) - 1.0).mean(axis=1)
        f["depth2_mean"] = np.log1p(bs2 + as2).mean(axis=1)

    # Trade-based measures.
    mid = (bid1 + ask1) / 2
    tp = g.trade_price
    tp_ff = pd.DataFrame(tp.T).ffill().bfill().to_numpy().T
    tr_ret = R._clean(np.diff(np.log(tp_ff), axis=1))
    f["trade_rv"] = R.realized_volatility(tr_ret)
    if g.signed_size is not None:
        sv = g.signed_size
    else:
        sv = lee_ready_sign(tp, mid) * tsize
    f["trade_signed_share"] = _safe_div(sv.sum(axis=1), tsize.sum(axis=1))
    dollar = np.where(has_trade, np.nan_to_num(tp) * tsize, 0.0)
    f["amihud"] = _safe_div(np.abs(r1).sum(axis=1), dollar.sum(axis=1) + 1e-9)
    x = sv[:, 1:]
    xc = x - x.mean(axis=1, keepdims=True)
    rc = r1 - r1.mean(axis=1, keepdims=True)
    f["kyle"] = _safe_div((xc * rc).sum(axis=1), (xc * xc).sum(axis=1)) * 1e4

    tick = tick_size(bid1, ask1)
    f["tick_norm"] = tick
    f["price_level"] = _safe_div(0.01, tick)

    df = pd.DataFrame(f)
    df.insert(0, "time_id", g.time_ids)
    df.insert(0, "stock_id", g.stock_id)
    return df


def sequence_features(g: StockGrid, bins: int = 60) -> np.ndarray:
    """Per-second series pooled into `bins` bins for the 1-D CNN: (n, C, bins)."""
    b = g.book
    bid1, ask1, bs1, as1 = b["bid_price1"], b["ask_price1"], b["bid_size1"], b["ask_size1"]
    w1 = wap(bid1, ask1, bs1, as1)
    r = np.concatenate([np.zeros((w1.shape[0], 1)), R._clean(np.diff(np.log(w1), axis=1))], axis=1)
    step = 600 // bins
    def pool(x, fn=np.sum):
        return fn(x.reshape(x.shape[0], bins, step), axis=2)
    ofi = np.concatenate([np.zeros((r.shape[0], 1)), order_flow_imbalance(bid1, ask1, bs1, as1)], axis=1)
    depth = np.maximum((bs1 + as1).mean(axis=1, keepdims=True), 1e-9)
    chans = [
        np.log(pool(r * r) + 1e-12),
        np.log(pool(_safe_div(ask1, bid1) - 1.0, np.mean) + 1e-8),
        pool(_safe_div(bs1 - as1, bs1 + as1), np.mean),
        pool(ofi / depth),
        np.log1p(pool(g.trade_size)),
    ]
    return np.stack(chans, axis=1).astype(np.float32)
