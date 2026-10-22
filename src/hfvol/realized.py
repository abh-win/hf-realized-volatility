"""Realized volatility estimators, vectorised over buckets.

Every function takes a 2-D array of log returns, shape (n_buckets, n_returns),
on a regular (1-second) grid, and returns one value per bucket. NaNs are
treated as zero returns (no price change).

References
- Andersen, Bollerslev, Diebold & Labys (2003): realized variance.
- Barndorff-Nielsen & Shephard (2004): bipower variation; jumps = RV - BV.
- Barndorff-Nielsen, Kinnebrock & Shephard (2010): realized semivariances.
- Bollerslev, Patton & Quaedvlieg (2016): realized quarticity (HARQ).
- Zhang, Mykland & Aït-Sahalia (2005): two-scale realized variance (TSRV).
- Barndorff-Nielsen, Hansen, Lunde & Shephard (2008, 2009): realized kernels
  with the Parzen kernel and the optimal bandwidth H* = c* xi^(4/5) n^(3/5).
"""

from __future__ import annotations

import numpy as np

MU1 = np.sqrt(2.0 / np.pi)  # E|Z| for Z ~ N(0,1)


def _clean(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    return np.where(np.isfinite(r), r, 0.0)


def log_returns(price: np.ndarray) -> np.ndarray:
    """Log returns along axis 1 of a (n_buckets, n_seconds) price grid."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(price)
    return _clean(np.diff(lp, axis=1))


def realized_variance(r: np.ndarray) -> np.ndarray:
    r = _clean(r)
    return np.sum(r * r, axis=1)


def realized_volatility(r: np.ndarray) -> np.ndarray:
    return np.sqrt(realized_variance(r))


def bipower_variation(r: np.ndarray) -> np.ndarray:
    """BV = mu1^-2 * n/(n-1) * sum |r_t||r_{t-1}| - robust to (finite) jumps."""
    r = np.abs(_clean(r))
    n = r.shape[1]
    s = np.sum(r[:, 1:] * r[:, :-1], axis=1)
    return s / (MU1 ** 2) * n / max(n - 1, 1)


def jump_variation(r: np.ndarray) -> np.ndarray:
    """Jump component max(RV - BV, 0)."""
    return np.maximum(realized_variance(r) - bipower_variation(r), 0.0)


def semivariances(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(RS+, RS-): squared returns split by sign ("good" and "bad" volatility)."""
    r = _clean(r)
    pos = np.sum(np.where(r > 0, r * r, 0.0), axis=1)
    neg = np.sum(np.where(r < 0, r * r, 0.0), axis=1)
    return pos, neg


def realized_quarticity(r: np.ndarray) -> np.ndarray:
    """RQ = n/3 * sum r^4 (estimates integrated quarticity)."""
    r = _clean(r)
    n = r.shape[1]
    return n / 3.0 * np.sum(r ** 4, axis=1)


def tsrv(log_price: np.ndarray, K: int = 10) -> np.ndarray:
    """Two-scale realized variance (Zhang, Mykland & Aït-Sahalia 2005).

    RV_avg averages K sparse realized variances (sampling every K-th point,
    offsets 0..K-1); the dense RV estimates the noise. With n dense returns and
    nbar = (n - K + 1) / K:
        TSRV = (1 - nbar/n)^-1 * (RV_avg - nbar/n * RV_dense)
    which removes the bias 2 n E[eps^2] that microstructure noise adds to RV.
    """
    lp = np.asarray(log_price, dtype=np.float64)
    n = lp.shape[1] - 1
    rv_dense = realized_variance(np.diff(lp, axis=1))
    rv_avg = np.zeros(lp.shape[0])
    for k in range(K):
        sub = lp[:, k::K]
        rv_avg += realized_variance(np.diff(sub, axis=1))
    rv_avg /= K
    nbar = (n - K + 1) / K
    est = (rv_avg - nbar / n * rv_dense) / (1.0 - nbar / n)
    return np.maximum(est, 0.0)


def _parzen(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    return np.where(x <= 0.5, 1 - 6 * x ** 2 + 6 * x ** 3, np.where(x <= 1.0, 2 * (1 - x) ** 3, 0.0))


def realized_kernel(r: np.ndarray, H: np.ndarray | int | None = None, h_max: int = 40) -> np.ndarray:
    """Parzen realized kernel  RK = g0 + 2 sum_{h=1..H} k(h/(H+1)) g_h,  g_h = sum r_t r_{t-h}.

    If H is None it is chosen per bucket from BNHLS (2009): xi^2 = omega^2 / IV
    with omega^2 = RV_dense / (2n) (noise variance) and IV approximated by a
    sparse (20-step) RV; H* = 3.5134 * xi^(4/5) * n^(3/5), capped at h_max.
    """
    r = _clean(r)
    nb, n = r.shape
    if H is None:
        rv = realized_variance(r)
        omega2 = rv / (2.0 * n)
        cum = np.concatenate([np.zeros((nb, 1)), np.cumsum(r, axis=1)], axis=1)
        sparse = np.diff(cum[:, ::20], axis=1)
        iv = np.maximum(realized_variance(sparse), 1e-18)
        xi = np.sqrt(np.maximum(omega2 / iv, 1e-12))
        H = np.clip(np.ceil(3.5134 * xi ** 0.8 * n ** 0.6), 1, h_max).astype(int)
    H = np.broadcast_to(np.asarray(H), (nb,))
    out = np.sum(r * r, axis=1)
    for h in range(1, int(H.max()) + 1):
        gamma = np.sum(r[:, h:] * r[:, :-h], axis=1)
        w = _parzen(h / (H + 1.0))
        out += 2.0 * w * gamma
    return np.maximum(out, 0.0)
