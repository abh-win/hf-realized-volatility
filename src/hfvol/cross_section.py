"""Cross-sectional and cross-time features.

* Time-id aggregates: market-wide volatility state (mean/std across stocks of
  key features per time_id, leave-one-out) and each stock's ratio to it.
* Stock clusters: KMeans on the correlation matrix of log-RV across time_ids
  (stocks whose volatility co-moves), then cluster-mean features per time_id.
* Time-order recovery: Optiver shuffled time_ids, but the normalised tick size
  reveals each stock's real price level (P = tick / tick_norm). Treating each
  time_id as a point in the space of log price levels across stocks, prices
  trace a continuous path through time, so a 1-D Isomap embedding of the k-NN
  graph recovers the chronology (up to direction).
* Nearest-neighbour time_ids: features averaged over the k time_ids closest in
  price space - in the competition setting this pools information from
  adjacent periods (a key trick in the winning solutions). In the *strict*
  setting only neighbours earlier in time are allowed, so no future
  information can leak into a forecast.
* Lags: once an order is known (true or recovered), classic HAR-style lags
  across buckets: previous bucket's RV, mean over the last 6 and 72 buckets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors

AGG_COLS = ["rv1_w600", "rv1_w300", "rv1_w100", "tsrv1", "spread_mean_w600", "ofi_abs_w600", "trade_size_w600"]


def add_time_aggregates(df: pd.DataFrame, cols=AGG_COLS) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    g = df.groupby("time_id")[cols]
    s, n = g.transform("sum"), g.transform("count")
    loo_mean = (s - df[cols]) / (n - 1).clip(lower=1)  # leave-one-out market mean
    std = g.transform("std")
    out = df.copy()
    for c in cols:
        out[f"{c}_tmean"] = loo_mean[c]
        out[f"{c}_tstd"] = std[c]
        out[f"{c}_tratio"] = df[c] / loo_mean[c].replace(0, np.nan)
    return out


def stock_clusters(df: pd.DataFrame, k: int = 4, seed: int = 0) -> pd.Series:
    piv = np.log(df.pivot(index="time_id", columns="stock_id", values="rv1_w600").clip(lower=1e-8))
    piv = piv.fillna(piv.mean())
    corr = np.corrcoef(piv.to_numpy().T)
    k = min(k, len(piv.columns))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(corr)
    return pd.Series(labels, index=piv.columns, name="cluster")


def add_cluster_features(df: pd.DataFrame, clusters: pd.Series, cols=("rv1_w600", "rv1_w300", "tsrv1")) -> pd.DataFrame:
    out = df.copy()
    out["cluster"] = out["stock_id"].map(clusters)
    for c in cols:
        if c in out.columns:
            out[f"clu_{c}_mean"] = out.groupby(["time_id", "cluster"])[c].transform("mean")
    return out


def price_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """time_id x stock matrix of standardised log price levels."""
    piv = np.log(df.pivot(index="time_id", columns="stock_id", values="price_level").replace(0, np.nan))
    piv = piv.fillna(piv.median())
    return (piv - piv.mean()) / piv.std().replace(0, 1)


def recover_time_order(df: pd.DataFrame, k: int = 10) -> pd.Series:
    """1-D Isomap embedding of time_ids in price-level space. Returns time_id -> rank.

    Prices trace a long, curved path through time; Isomap's geodesic distances
    follow the path, whereas a Laplacian eigenmap (spectral seriation) folds
    long curves (|Spearman| 0.62 vs 0.99999 on the synthetic data).
    """
    P = price_matrix(df)
    emb = Isomap(n_neighbors=k, n_components=1).fit_transform(P.to_numpy())[:, 0]
    return pd.Series(pd.Series(emb, index=P.index).rank(method="first").astype(int) - 1, name="rank")


def order_quality(recovered: pd.Series, truth: pd.Series) -> float:
    """|Spearman rho| between recovered and true ranks (direction is arbitrary)."""
    common = recovered.index.intersection(truth.index)
    rho = spearmanr(recovered.loc[common], truth.loc[common]).statistic
    return float(abs(rho))


def add_neighbor_features(df: pd.DataFrame, k_list=(5, 20), order: pd.Series | None = None,
                          cols=("rv1_w600", "tsrv1")) -> pd.DataFrame:
    """Mean of each stock's features over its nearest time_ids in price space.

    If `order` is given (strict mode) only neighbours strictly earlier in that
    order are used.
    """
    P = price_matrix(df)
    tids = P.index.to_numpy()
    X = P.to_numpy()
    kmax = max(k_list)
    n_query = kmax + 1 if order is None else min(len(X), 4 * kmax + 1)
    nn = NearestNeighbors(n_neighbors=n_query).fit(X)
    _, idx = nn.kneighbors(X)
    neigh = idx[:, 1:]
    if order is not None:
        rank = order.reindex(tids).to_numpy()
        valid = rank[neigh] < rank[:, None]
    else:
        valid = np.ones_like(neigh, dtype=bool)

    out = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        piv = df.pivot(index="time_id", columns="stock_id", values=c).reindex(tids)
        V = piv.to_numpy()                       # (T, S)
        mkt = np.nanmean(V, axis=1)              # market level per time_id
        for k in k_list:
            # first k valid neighbours per row
            cs = np.cumsum(valid, axis=1)
            take = valid & (cs <= k)
            cnt = np.maximum(take.sum(axis=1), 1)
            own = np.einsum("tn,tns->ts", take, np.nan_to_num(V[neigh])) / cnt[:, None]
            own[take.sum(axis=1) == 0] = np.nan
            m = (take * mkt[neigh]).sum(axis=1) / cnt
            m[take.sum(axis=1) == 0] = np.nan
            own_df = pd.DataFrame(own, index=tids, columns=piv.columns).stack().rename(f"nn{k}_{c}")
            out = out.merge(own_df.reset_index().rename(columns={"level_0": "time_id"}),
                            on=["time_id", "stock_id"], how="left")
            out = out.merge(pd.DataFrame({"time_id": tids, f"nn{k}_mkt_{c}": m}), on="time_id", how="left")
    return out


def add_lag_features(df: pd.DataFrame, order: pd.Series, col: str = "rv1_w600") -> pd.DataFrame:
    """HAR lags across buckets in the given chronological order."""
    out = df.copy()
    out["_rank"] = out["time_id"].map(order)
    out = out.sort_values(["stock_id", "_rank"])
    g = out.groupby("stock_id")[col]
    out[f"lag1_{col}"] = g.shift(1)
    out[f"lag6_{col}"] = g.transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    out[f"lag72_{col}"] = g.transform(lambda s: s.shift(1).rolling(72, min_periods=1).mean())
    return out.drop(columns="_rank").sort_values(["stock_id", "time_id"]).reset_index(drop=True)
