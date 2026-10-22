"""Forecast comparison: Diebold-Mariano tests, Model Confidence Set, SHAP.

Losses are squared percentage errors ((p - y) / y)^2, i.e. the per-sample
RMSPE contribution. Because all stocks in a time_id share market shocks,
per-sample losses are cross-sectionally dependent; we therefore average the
loss over stocks within each time_id and run the tests on the resulting time
series (in chronological order), with Newey-West (HAC) variance for serial
correlation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def loss_series(df: pd.DataFrame, preds: dict[str, np.ndarray], order: pd.Series) -> pd.DataFrame:
    """time_id-aggregated squared percentage error per model, chronologically sorted."""
    y = df["target"].to_numpy()
    L = pd.DataFrame({m: ((p - y) / y) ** 2 for m, p in preds.items()})
    L["time_id"] = df["time_id"].to_numpy()
    agg = L.groupby("time_id").mean()
    return agg.loc[order.reindex(agg.index).sort_values().index]


def diebold_mariano(l1: np.ndarray, l2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM test of equal predictive accuracy, HAC variance, Harvey-Leybourne-Newbold
    small-sample correction. Returns (statistic, two-sided p-value);
    statistic < 0 means model 1 has lower loss."""
    d = np.asarray(l1) - np.asarray(l2)
    T = len(d)
    dbar = d.mean()
    lag = max(h - 1, int(math.floor(T ** (1 / 3))))
    dc = d - dbar
    gamma = [np.dot(dc[k:], dc[:T - k]) / T for k in range(lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    if var <= 0:
        return 0.0, 1.0
    dm = dbar / math.sqrt(var / T)
    hln = math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return float(stat), float(p)


def dm_matrix(losses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = list(losses.columns)
    S = pd.DataFrame(index=models, columns=models, dtype=float)
    P = pd.DataFrame(index=models, columns=models, dtype=float)
    for a in models:
        for b in models:
            if a == b:
                continue
            S.loc[a, b], P.loc[a, b] = diebold_mariano(losses[a].to_numpy(), losses[b].to_numpy())
    return S, P


def model_confidence_set(losses: pd.DataFrame, size: float = 0.10, reps: int = 2000, block: int = 10,
                         seed: int = 0) -> pd.DataFrame:
    """Hansen, Lunde & Nason (2011) MCS via arch's block-bootstrap implementation."""
    from arch.bootstrap import MCS
    mcs = MCS(losses, size=size, reps=reps, block_size=block, method="R", seed=seed)
    mcs.compute()
    pv = mcs.pvalues.rename(columns={"Pvalue": "mcs_pvalue"})
    pv["in_mcs"] = pv.index.isin(mcs.included)
    return pv.sort_values("mcs_pvalue", ascending=False)


FEATURE_GROUPS = {
    "realized (RV/BV/jumps/semivariance/RQ)": lambda c: c.split("_w")[0] in ("rv1", "bv1", "jump1", "rsp1", "rsn1") or c == "rq1",
    "noise-robust (TSRV, realized kernel)": lambda c: c in ("tsrv1", "rk1"),
    "level-2 (WAP2, L2 imbalance/spread/depth)": lambda c: c.startswith(("rv2", "imb2", "wapbal", "spread2", "depth2")),
    "microstructure (spread, imbalance, OFI, depth, Kyle, Amihud)": lambda c: c.startswith(
        ("spread_", "imb1", "ofi", "depth_", "kyle", "amihud", "range", "n_upd", "ret_sum")),
    "trades": lambda c: c.startswith("trade_"),
    "cross-sectional (market aggregates, clusters)": lambda c: c.endswith(("_tmean", "_tstd", "_tratio")) or c.startswith("clu_"),
    "nearest-neighbour time_ids": lambda c: c.startswith("nn"),
    "lags across buckets": lambda c: c.startswith("lag"),
    "price level / tick": lambda c: c in ("tick_norm", "price_level"),
    "stock id": lambda c: c == "stock_id",
}


def feature_group(col: str) -> str:
    # Cross-sectional transforms of other features belong to the cross-sectional group.
    if col.endswith(("_tmean", "_tstd", "_tratio")) or col.startswith("clu_"):
        return "cross-sectional (market aggregates, clusters)"
    for name, pred in FEATURE_GROUPS.items():
        if pred(col):
            return name
    return "other"


def multiscale_columns(cols) -> list[str]:
    return [c for c in cols if any(c.endswith(f"_w{w}") for w in (500, 300, 200, 100)) and
            not c.endswith(("_tmean", "_tstd", "_tratio"))]


def shap_importance(model, X: pd.DataFrame, max_rows: int = 3000, seed: int = 0) -> pd.DataFrame:
    import shap
    Xs = X.sample(min(max_rows, len(X)), random_state=seed)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    imp = pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp["group"] = imp["feature"].map(feature_group)
    return imp.sort_values("mean_abs_shap", ascending=False), sv, Xs
