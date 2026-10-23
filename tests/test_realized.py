import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hfvol import realized as R  # noqa: E402
from hfvol.cv import holdout_split, purged_kfold  # noqa: E402
from hfvol.evaluation import diebold_mariano  # noqa: E402
from hfvol.features import order_flow_imbalance, tick_size  # noqa: E402
from hfvol.models import rmspe, stack_weights  # noqa: E402

rng = np.random.default_rng(0)


def brownian(nb=2000, n=600, sigma=0.003):
    """Log prices with integrated variance sigma^2 over the bucket."""
    r = rng.normal(0, sigma / np.sqrt(n), (nb, n))
    return np.concatenate([np.zeros((nb, 1)), np.cumsum(r, axis=1)], axis=1)


def test_rv_unbiased_without_noise():
    lp = brownian()
    rv = R.realized_variance(np.diff(lp, axis=1))
    assert abs(rv.mean() / 0.003 ** 2 - 1) < 0.02


def test_bipower_matches_rv_without_jumps_and_detects_jumps():
    lp = brownian()
    r = np.diff(lp, axis=1)
    assert abs(R.bipower_variation(r).mean() / R.realized_variance(r).mean() - 1) < 0.03
    r_j = r.copy()
    r_j[:, 300] += 0.004  # one jump per bucket, variance 1.6e-5 >> IV 9e-6
    jv = R.jump_variation(r_j)
    assert abs(jv.mean() - 0.004 ** 2) / 0.004 ** 2 < 0.15
    # BV stays close to the diffusive part; the jump leaks in only through the
    # two neighbouring products |J||r| (~ 2 J E|r| pi/2, ~13% of IV at n = 600).
    bv = R.bipower_variation(r_j).mean()
    assert abs(bv / 0.003 ** 2 - 1) < 0.2
    assert R.realized_variance(r_j).mean() - bv > 0.8 * 0.004 ** 2


def test_semivariances_sum_to_rv():
    r = np.diff(brownian(100), axis=1)
    p, n = R.semivariances(r)
    assert np.allclose(p + n, R.realized_variance(r))


def test_quarticity_matches_theory():
    r = np.diff(brownian(), axis=1)
    # For constant volatility, IQ = sigma^4 (per unit time = bucket)
    assert abs(R.realized_quarticity(r).mean() / 0.003 ** 4 - 1) < 0.05


def test_tsrv_and_kernel_remove_noise_bias():
    lp = brownian()
    noisy = lp + rng.normal(0, 6e-5, lp.shape)  # microstructure noise
    r = np.diff(noisy, axis=1)
    iv = 0.003 ** 2
    rv_bias = R.realized_variance(r).mean() / iv - 1
    assert rv_bias > 0.2  # plain RV is badly biased upward (2 n omega^2)
    assert abs(R.tsrv(noisy, K=10).mean() / iv - 1) < 0.1
    assert abs(R.realized_kernel(r).mean() / iv - 1) < 0.1


def test_ofi_signs():
    bid = np.array([[100.0, 100.0, 101.0]])
    ask = np.array([[101.0, 101.0, 102.0]])
    bs = np.array([[10.0, 15.0, 5.0]])
    as_ = np.array([[10.0, 10.0, 8.0]])
    e = order_flow_imbalance(bid, ask, bs, as_)
    assert e[0, 0] == 5.0    # bid queue grew by 5, ask unchanged
    assert e[0, 1] == 5 + 10 # bid moved up (new queue 5); ask moved up (old ask queue 10 removed)


def test_tick_size_recovers_price_level():
    P, tick = 37.25, 0.01
    levels = (P + tick * np.array([[0, 1, 2, -1, 3]])) / P
    assert abs(tick_size(levels, levels + tick / P)[0] - tick / P) < 1e-9


def test_purged_kfold_has_gaps_and_covers_everything():
    order = pd.Series(np.arange(100), index=np.arange(100) * 7)  # time_id -> rank
    seen = []
    for tr, te in purged_kfold(order.index.to_numpy(), order, n_splits=5, purge=2, embargo=0.03):
        rtr, rte = order[tr].to_numpy(), order[te].to_numpy()
        assert not set(tr) & set(te)
        assert np.all((rtr < rte.min() - 2) | (rtr > rte.max() + 3))
        seen.extend(te)
    assert sorted(seen) == sorted(order.index)
    tr, te = holdout_split(order, 0.2, gap=1)
    assert order[tr].max() + 1 < order[te].min()


def test_dm_detects_better_forecast():
    y = rng.lognormal(-6, 0.5, 500)
    good = y * (1 + rng.normal(0, 0.1, 500))
    bad = y * (1 + rng.normal(0, 0.3, 500))
    stat, p = diebold_mariano(((good - y) / y) ** 2, ((bad - y) / y) ** 2)
    assert stat < 0 and p < 1e-6
    _, p_same = diebold_mariano(((good - y) / y) ** 2, ((good - y) / y) ** 2 + rng.normal(0, 1e-9, 500))
    assert p_same > 0.01


def test_stack_weights_find_best_combination():
    y = rng.lognormal(-6, 0.5, 2000)
    a = y * (1 + rng.normal(0, 0.2, 2000))
    b = y * (1 + rng.normal(0, 0.2, 2000))
    junk = y * rng.uniform(0.2, 3, 2000)
    w = stack_weights(pd.DataFrame({"a": a, "b": b, "junk": junk}), y)
    assert abs(w.sum() - 1) < 1e-6 and w["junk"] < 0.05
    assert rmspe(y, w["a"] * a + w["b"] * b + w["junk"] * junk) < min(rmspe(y, a), rmspe(y, b))
