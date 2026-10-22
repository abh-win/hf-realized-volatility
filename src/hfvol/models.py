"""Model ladder: econometric baselines -> gradient boosting -> neural nets -> stacking.

All models minimise (or are evaluated by) RMSPE = sqrt(mean(((p - y) / y)^2)),
the competition metric. For tree/linear models that means a weighted squared
loss with weights 1/y^2.
"""

from __future__ import annotations

import math
import time
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ID_COLS = {"stock_id", "time_id", "target", "cluster"}


def rmspe(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean(((p - y) / y) ** 2)))


def feature_columns(df: pd.DataFrame, exclude=()) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS and c not in exclude and
            pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------- baselines

def naive(train, test, **_):
    """Persistence: next 10-min RV = this bucket's RV."""
    return test["rv1_w600"].to_numpy()


def _har_design(df, lag_cols):
    cols = ["rv1_w600", "rv1_w300", "rv1_w100"] + lag_cols
    return cols


def har_ols(train, test, **_):
    """HAR-RV (Corsi 2009) on log-RV: log y = b0 + b_d log RV_600 + b_w log RV_300 + b_m log RV_100
    (+ log lags across buckets when a time order is known), fitted by OLS.

    RMSPE-optimal back-transform: if log y ~ N(mu, s^2) then argmin_p E[((p-y)/y)^2]
    = E[1/y] / E[1/y^2] = exp(mu - 1.5 s^2).
    """
    lag_cols = [c for c in train.columns if c.startswith("lag") and c.endswith("rv1_w600")]
    cols = _har_design(train, lag_cols)
    def X(d):
        return np.log(d[cols].clip(lower=1e-6).fillna(d["rv1_w600"].median()).to_numpy())
    m = LinearRegression().fit(X(train), np.log(train["target"]))
    s2 = np.var(np.log(train["target"]) - m.predict(X(train)))
    return np.exp(m.predict(X(test)) - 1.5 * s2)


def harq(train, test, **_):
    """HARQ (Bollerslev, Patton & Quaedvlieg 2016): the persistence of RV_t is
    attenuated when its measurement error (realized quarticity) is large:
    y = b0 + (b1 + b1Q * sqrt(RQ)) RV_600 + b2 RV_300 + b3 RV_100 (+ lags),
    fitted by WLS with weights 1/y^2 (RMSPE-aware)."""
    lag_cols = [c for c in train.columns if c.startswith("lag") and c.endswith("rv1_w600")]
    cols = _har_design(train, lag_cols)
    rq_scale = np.sqrt(train["rq1"]).median()
    def X(d):
        base = d[cols].fillna(d["rv1_w600"].median()).to_numpy()
        q = (np.sqrt(d["rq1"].to_numpy()) / rq_scale) * d["rv1_w600"].to_numpy()
        return np.column_stack([base, q])
    y = train["target"].to_numpy()
    m = LinearRegression().fit(X(train), y, sample_weight=1 / y ** 2)
    return np.clip(m.predict(X(test)), 1e-5, None)


def ols_all(train, test, feats, **_):
    """Linear OLS on log target with all engineered features (log-transformed
    where positive, standardised), stock fixed effects; the linear counterpart
    of the GBDT models on the same inputs. RMSPE-optimal lognormal back-transform."""
    cols = [f for f in feats if f != "stock_id"]
    tr = train[cols].replace([np.inf, -np.inf], np.nan)
    pos = [c for c in cols if (tr[c].dropna() > 0).all()]
    def X(d):
        x = d[cols].replace([np.inf, -np.inf], np.nan).fillna(tr.median())
        x[pos] = np.log(x[pos].clip(lower=1e-9))
        return x
    Xtr = X(train)
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    dummies = pd.get_dummies(train["stock_id"], prefix="s", drop_first=True, dtype=float)
    def design(d):
        z = np.clip(((X(d) - mu) / sd).to_numpy(), -5, 5)  # winsorise heavy tails
        dd = pd.get_dummies(d["stock_id"], prefix="s", dtype=float).reindex(columns=dummies.columns, fill_value=0.0)
        return np.column_stack([z, dd.to_numpy()])
    m = LinearRegression().fit(design(train), np.log(train["target"]))
    s2 = np.var(np.log(train["target"]) - m.predict(design(train)))
    lo, hi = np.log(train["target"].min()), np.log(train["target"].max())
    return np.exp(np.clip(m.predict(design(test)) - 1.5 * s2, lo, hi))


# --------------------------------------------------------------------- GBDT

def _ratio(df):
    """Target normalised by the bucket's own realized vol. RMSPE is scale-free, so
    fitting y' = y / RV_600 with weights 1/y'^2 optimises the same objective while
    removing cross-stock / cross-regime scale differences the trees would otherwise
    have to learn."""
    return df["target"].to_numpy() / df["rv1_w600"].clip(lower=1e-6).to_numpy()


def _lgb_data(df, feats, weight=True):
    y = _ratio(df)
    return lgb.Dataset(df[feats], y, weight=1 / y ** 2 if weight else None,
                       categorical_feature=["stock_id"] if "stock_id" in feats else "auto", free_raw_data=False)


LGB_DEFAULT = dict(objective="rmse", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   verbose=-1, num_threads=8, seed=0)


def lgbm(train, test, feats, params=None, n_rounds=None, seeds=(0, 1, 2), **_):
    """LightGBM on the ratio target, averaged over bagging seeds."""
    p = {**LGB_DEFAULT, **(params or {})}
    rounds = n_rounds or p.pop("n_rounds", 600)
    p.pop("n_rounds", None)
    scale = test["rv1_w600"].clip(lower=1e-6).to_numpy()
    preds, model = [], None
    for sd in seeds:
        q = {**p, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd}
        model = lgb.train(q, _lgb_data(train, feats), num_boost_round=rounds)
        preds.append(model.predict(test[feats]))
    return np.mean(preds, axis=0) * scale, model


def tune_lgbm(train, feats, folds, n_trials=40, timeout=900, seed=0):
    """Optuna TPE search; objective = mean RMSPE over purged CV folds."""
    def objective(trial):
        p = dict(LGB_DEFAULT)
        p.update(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 800, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.3, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
        )
        scores, iters = [], []
        for tr_t, va_t in folds:
            tr, va = train[train.time_id.isin(tr_t)], train[train.time_id.isin(va_t)]
            m = lgb.train(p, _lgb_data(tr, feats), num_boost_round=3000, valid_sets=[_lgb_data(va, feats)],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            scores.append(rmspe(va["target"], m.predict(va[feats], num_iteration=m.best_iteration) *
                                va["rv1_w600"].clip(lower=1e-6).to_numpy()))
            iters.append(m.best_iteration)
        trial.set_user_attr("n_rounds", int(np.mean(iters)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = dict(study.best_params)
    best["n_rounds"] = study.best_trial.user_attrs["n_rounds"]
    return best, study


XGB_DEFAULT = dict(objective="reg:squarederror", eta=0.05, max_depth=6, min_child_weight=50, subsample=0.8,
                   colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist", nthread=8, seed=0)


def _xgb_matrix(df, feats):
    y = _ratio(df)
    return xgb.DMatrix(df[feats], label=y, weight=1 / y ** 2)


def xgboost_model(train, test, feats, params=None, **_):
    p = {**XGB_DEFAULT, **(params or {})}
    rounds = p.pop("n_rounds", 600)
    scale = test["rv1_w600"].clip(lower=1e-6).to_numpy()
    preds, m = [], None
    for sd in (0, 1, 2):
        m = xgb.train({**p, "seed": sd}, _xgb_matrix(train, feats), num_boost_round=rounds)
        preds.append(m.predict(xgb.DMatrix(test[feats])))
    return np.mean(preds, axis=0) * scale, m


def tune_xgb(train, feats, folds, n_trials=25, timeout=600, seed=0):
    def objective(trial):
        p = dict(XGB_DEFAULT)
        p.update(
            eta=trial.suggest_float("eta", 0.01, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 300, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )
        scores, iters = [], []
        for tr_t, va_t in folds:
            tr, va = train[train.time_id.isin(tr_t)], train[train.time_id.isin(va_t)]
            dva = _xgb_matrix(va, feats)
            m = xgb.train(p, _xgb_matrix(tr, feats), 3000, evals=[(dva, "va")], early_stopping_rounds=100,
                          verbose_eval=False)
            scores.append(rmspe(va["target"], m.predict(dva, iteration_range=(0, m.best_iteration + 1)) *
                                va["rv1_w600"].clip(lower=1e-6).to_numpy()))
            iters.append(m.best_iteration + 1)
        trial.set_user_attr("n_rounds", int(np.mean(iters)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = dict(study.best_params)
    best["n_rounds"] = study.best_trial.user_attrs["n_rounds"]
    return best, study


# --------------------------------------------------------------------- neural nets

def _torch():
    import torch
    torch.set_num_threads(8)
    return torch


class _Prep:
    """Quantile-normalised features (fitted on train), NaN -> 0, plus stock index."""

    def __init__(self, train, feats, stocks):
        self.feats = [f for f in feats if f != "stock_id"]
        self.qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=200_000, random_state=0)
        self.qt.fit(train[self.feats].replace([np.inf, -np.inf], np.nan).fillna(train[self.feats].median()))
        self.stock_index = {s: i for i, s in enumerate(stocks)}

    def __call__(self, df):
        x = self.qt.transform(df[self.feats].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        x = np.nan_to_num(x).astype(np.float32)
        sid = df["stock_id"].map(self.stock_index).fillna(0).astype(int).to_numpy()
        base = np.log(df["rv1_w600"].clip(lower=1e-6).to_numpy()).astype(np.float32)
        return x, sid, base


def _train_torch(make_model, tensors_tr, y_tr, tensors_va, y_va, epochs=60, lr=2e-3, batch=1024, seed=0):
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    yv = torch.tensor(y_va, dtype=torch.float32)
    tr = [torch.tensor(t) for t in tensors_tr]
    va = [torch.tensor(t) for t in tensors_va]
    best, best_state, patience = math.inf, None, 0
    n = len(yt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            pred = model(*[t[b] for t in tr])
            loss = torch.mean(((pred - yt[b]) / yt[b]) ** 2)  # RMSPE^2
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = torch.sqrt(torch.mean(((model(*va) - yv) / yv) ** 2)).item()
        if v < best - 1e-5:
            best, best_state, patience = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 10:
                break
    model.load_state_dict(best_state)
    return model


def _inner_split(train, order):
    """Last 10% of the training period (in time) for early stopping."""
    t = train["time_id"].map(order)
    cut = t.quantile(0.9)
    return train[t <= cut], train[t > cut]


def mlp(train, test, feats, order, seed=0, **_):
    """MLP over normalised features + stock embedding; outputs log(y / RV_600)."""
    torch = _torch()
    nn = torch.nn
    stocks = sorted(train["stock_id"].unique())
    prep = _Prep(train, feats, stocks)
    tr, va = _inner_split(train, order)
    d = len(prep.feats)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(stocks), 8)
            self.body = nn.Sequential(nn.Linear(d + 8, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.SiLU(),
                                      nn.Dropout(0.1), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

        def forward(self, x, sid, base):
            h = torch.cat([x, self.emb(sid)], dim=1)
            return torch.exp(self.body(h).squeeze(1) + base)

    model = _train_torch(Net, prep(tr), tr["target"].to_numpy(), prep(va), va["target"].to_numpy(), seed=seed)
    model.eval()
    with torch.no_grad():
        return model(*[torch.tensor(t) for t in prep(test)]).numpy()


def cnn(train, test, feats, order, seqs, seed=0, **_):
    """1-D CNN over the per-second series (60 x 10s bins) + stock embedding + global features."""
    torch = _torch()
    nn = torch.nn
    stocks = sorted(train["stock_id"].unique())
    glob_feats = [f for f in ["rv1_w600", "rv1_w100", "tsrv1", "spread_mean_w600", "rv1_w600_tmean", "nn20_rv1_w600"]
                  if f in feats]
    prep = _Prep(train, glob_feats, stocks)
    tr, va = _inner_split(train, order)

    def tensors(df):
        x, sid, base = prep(df)
        s = seqs.loc[list(zip(df.stock_id, df.time_id))]["row"].to_numpy()
        return seqs.attrs["array"][s], x, sid, base

    seq_tr = tensors(tr)
    mu = seq_tr[0].mean(axis=(0, 2), keepdims=True)
    sd = seq_tr[0].std(axis=(0, 2), keepdims=True) + 1e-6
    norm = lambda t: ((t[0] - mu) / sd, *t[1:])  # noqa: E731
    C, G = seq_tr[0].shape[1], len(prep.feats)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(stocks), 8)
            self.conv = nn.Sequential(nn.Conv1d(C, 32, 5, padding=2), nn.SiLU(), nn.Conv1d(32, 64, 5, stride=2, padding=2),
                                      nn.SiLU(), nn.Conv1d(64, 64, 3, padding=1), nn.SiLU())
            self.head = nn.Sequential(nn.Linear(128 + G + 8, 64), nn.SiLU(), nn.Linear(64, 1))

        def forward(self, s, x, sid, base):
            h = self.conv(s)
            h = torch.cat([h.mean(dim=2), h.amax(dim=2), x, self.emb(sid)], dim=1)
            return torch.exp(self.head(h).squeeze(1) + base)

    model = _train_torch(Net, norm(seq_tr), tr["target"].to_numpy(), norm(tensors(va)), va["target"].to_numpy(),
                         seed=seed, epochs=40)
    model.eval()
    with torch.no_grad():
        return model(*[torch.tensor(t) for t in norm(tensors(test))]).numpy()


# --------------------------------------------------------------------- stacking

def stack_weights(oof: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Non-negative weights summing to 1 minimising RMSPE of the blended OOF predictions."""
    P = oof.to_numpy()
    m = P.shape[1]
    obj = lambda w: np.mean(((P @ w - y) / y) ** 2)  # noqa: E731
    res = minimize(obj, np.full(m, 1 / m), method="SLSQP", bounds=[(0, 1)] * m,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return pd.Series(res.x, index=oof.columns)
