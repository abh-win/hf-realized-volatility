#!/usr/bin/env python3
"""End-to-end realized-volatility forecasting pipeline on an Optiver-format dataset.

  run_pipeline.py --data data/crypto --out results/crypto [--setting strict|competition]
                  [--trials 40] [--xgb-trials 20] [--folds 5] [--skip-nn] [--skip-ablation]

Steps: features -> time order -> cross-sectional/neighbour/lag features ->
chronological holdout (last 20%) -> purged/embargoed CV on the training period
(Optuna tuning, out-of-fold predictions) -> model ladder -> stacking ->
RMSPE table, Diebold-Mariano tests, Model Confidence Set, SHAP, ablations.

Settings
  strict       only information available at forecast time: neighbour
               time_ids restricted to the past, clusters fit on training data.
  competition  Optiver-competition rules: neighbours from any time_id (the
               test set's book data was fully available).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hfvol import cross_section as X  # noqa: E402
from hfvol import evaluation as E  # noqa: E402
from hfvol import models as M  # noqa: E402
from hfvol.cv import holdout_split, purged_kfold  # noqa: E402
from hfvol.data import load_stock, load_targets, load_time_order, stock_ids  # noqa: E402
from hfvol.features import sequence_features, stock_features  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _one_stock(args):
    data_dir, sid = args
    g = load_stock(data_dir, sid)
    return stock_features(g), sequence_features(g), g.time_ids


def build_features(data_dir: str, cache_dir: str):
    fpath = os.path.join(cache_dir, "features.parquet")
    spath = os.path.join(cache_dir, "sequences.npy")
    if os.path.exists(fpath) and os.path.exists(spath):
        feats = pd.read_parquet(fpath)
        seq = np.load(spath)
        return feats, seq
    sids = stock_ids(data_dir)
    with ProcessPoolExecutor(min(len(sids), 10)) as ex:
        res = list(ex.map(_one_stock, [(data_dir, s) for s in sids]))
    feats = pd.concat([r[0] for r in res], ignore_index=True)
    seq = np.concatenate([r[1] for r in res], axis=0)
    os.makedirs(cache_dir, exist_ok=True)
    feats.to_parquet(fpath, index=False)
    np.save(spath, seq)
    return feats, seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--setting", choices=["strict", "competition"], default="strict")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--xgb-trials", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tune-folds", type=int, default=3)
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    summary: dict = {"data": a.data, "setting": a.setting}

    # ---------------------------------------------------------------- features
    log("computing per-bucket features")
    feats, seq = build_features(a.data, os.path.join(a.out, "..", f"_cache_{os.path.basename(os.path.normpath(a.data))}"))
    df = feats.merge(load_targets(a.data), on=["stock_id", "time_id"], how="inner")
    df = df[df["target"] > 0].reset_index(drop=True)
    seq_index = pd.DataFrame({"row": np.arange(len(feats))},
                             index=pd.MultiIndex.from_arrays([feats.stock_id, feats.time_id]))
    seq_index.attrs["array"] = seq
    log(f"{len(df)} samples, {df.stock_id.nunique()} stocks, {df.time_id.nunique()} time_ids, "
        f"{len(M.feature_columns(df))} raw features")

    # ---------------------------------------------------------------- time order
    true_order = load_time_order(a.data)
    recovered = X.recover_time_order(df)
    truth_path = os.path.join(a.data, "truth_time_order.csv")
    truth = None
    if true_order is not None:
        truth = true_order
    elif os.path.exists(truth_path):
        t = pd.read_csv(truth_path)
        truth = pd.Series(t["order"].to_numpy(), index=t["time_id"].to_numpy())
    if truth is not None:
        summary["order_recovery_spearman"] = X.order_quality(recovered, truth)
        log(f"time-order recovery from tick sizes: |Spearman| = {summary['order_recovery_spearman']:.3f}")
    # Splits use the true chronology when it is known (crypto); otherwise (Optiver,
    # synthetic) the recovered one - exactly what one could do on the real competition data.
    order = true_order if true_order is not None else recovered
    if true_order is None and truth is not None:
        # make the recovered order run forward in time (direction is arbitrary)
        if np.corrcoef(recovered.reindex(truth.index), truth)[0, 1] < 0:
            order = recovered.max() - recovered
    summary["order_source"] = "true" if true_order is not None else "recovered"

    train_t, test_t = holdout_split(order, 0.2)

    # ---------------------------------------------------------------- cross-sectional features
    df = X.add_time_aggregates(df)
    clusters = X.stock_clusters(df[df.time_id.isin(train_t)], k=4)
    df = X.add_cluster_features(df, clusters)
    df = X.add_neighbor_features(df, order=order if a.setting == "strict" else None)
    df = X.add_lag_features(df, order)
    feat_cols = M.feature_columns(df) + ["stock_id"]
    feat_cols = [c for c in feat_cols if c not in ("cluster",)]
    feat_cols = list(dict.fromkeys(feat_cols))
    summary["n_features"] = len(feat_cols)
    log(f"{len(feat_cols)} features after cross-sectional / neighbour / lag features")

    train = df[df.time_id.isin(train_t)].reset_index(drop=True)
    test = df[df.time_id.isin(test_t)].reset_index(drop=True)
    folds = list(purged_kfold(np.unique(train.time_id), order, n_splits=a.folds))
    tune_folds = list(purged_kfold(np.unique(train.time_id), order, n_splits=a.tune_folds))
    summary.update(n_train=len(train), n_test=len(test))

    # ---------------------------------------------------------------- tuning
    log(f"Optuna: LightGBM ({a.trials} trials)")
    lgb_params, lgb_study = M.tune_lgbm(train, feat_cols, tune_folds, n_trials=a.trials)
    log(f"  best CV RMSPE {lgb_study.best_value:.4f} params {lgb_params}")
    log(f"Optuna: XGBoost ({a.xgb_trials} trials)")
    xgb_params, xgb_study = M.tune_xgb(train, feat_cols, tune_folds, n_trials=a.xgb_trials)
    log(f"  best CV RMSPE {xgb_study.best_value:.4f}")
    summary["lgbm_params"], summary["xgb_params"] = lgb_params, xgb_params

    models = {
        "naive (RV persistence)": lambda tr, te: M.naive(tr, te),
        "HAR-RV (OLS, log)": lambda tr, te: M.har_ols(tr, te),
        "HARQ (WLS)": lambda tr, te: M.harq(tr, te),
        "OLS (all features, log)": lambda tr, te: M.ols_all(tr, te, feat_cols),
        "LightGBM": lambda tr, te: M.lgbm(tr, te, feat_cols, lgb_params)[0],
        "XGBoost": lambda tr, te: M.xgboost_model(tr, te, feat_cols, xgb_params)[0],
    }
    if not a.skip_nn:
        models["MLP (stock embedding)"] = lambda tr, te: M.mlp(tr, te, feat_cols, order)
        models["1D-CNN (per-second series)"] = lambda tr, te: M.cnn(tr, te, feat_cols, order, seq_index)

    # ---------------------------------------------------------------- OOF + holdout
    oof = pd.DataFrame(index=train.index, columns=list(models), dtype=float)
    hold = {}
    cv_scores = {}
    for name, fn in models.items():
        t0 = time.time()
        fold_scores = []
        for tr_t, va_t in folds:
            tr_idx = train.time_id.isin(tr_t)
            va_idx = train.time_id.isin(va_t)
            p = fn(train[tr_idx].reset_index(drop=True), train[va_idx].reset_index(drop=True))
            oof.loc[va_idx, name] = p
            fold_scores.append(M.rmspe(train.loc[va_idx, "target"], p))
        hold[name] = fn(train, test)
        cv_scores[name] = (float(np.mean(fold_scores)), float(np.std(fold_scores)))
        log(f"{name:30s} CV {cv_scores[name][0]:.4f} ± {cv_scores[name][1]:.4f}   holdout "
            f"{M.rmspe(test.target, hold[name]):.4f}   ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- stacking
    stack_inputs = [m for m in models if m != "naive (RV persistence)"]
    mask = oof[stack_inputs].notna().all(axis=1)
    w = M.stack_weights(oof.loc[mask, stack_inputs], train.loc[mask, "target"].to_numpy())
    hold["Stacked ensemble"] = sum(w[m] * hold[m] for m in stack_inputs)
    oof_stack = (oof.loc[mask, stack_inputs] * w).sum(axis=1)
    cv_scores["Stacked ensemble"] = (M.rmspe(train.loc[mask, "target"], oof_stack), float("nan"))
    summary["stack_weights"] = w.round(4).to_dict()
    log(f"stack weights {w.round(3).to_dict()}")

    # ---------------------------------------------------------------- evaluation
    y = test["target"].to_numpy()
    base = M.rmspe(y, hold["HAR-RV (OLS, log)"])
    table = pd.DataFrame({
        "holdout_rmspe": {m: M.rmspe(y, p) for m, p in hold.items()},
        "cv_rmspe": {m: cv_scores[m][0] for m in hold},
        "cv_sd": {m: cv_scores[m][1] for m in hold},
    })
    table["vs_HAR_OLS"] = 1 - table["holdout_rmspe"] / base
    table = table.sort_values("holdout_rmspe")
    table.to_csv(os.path.join(a.out, "model_table.csv"))
    summary["models"] = table.round(5).to_dict(orient="index")

    losses = E.loss_series(test, hold, order)
    dm_s, dm_p = E.dm_matrix(losses)
    dm_s.to_csv(os.path.join(a.out, "dm_stat.csv"))
    dm_p.to_csv(os.path.join(a.out, "dm_pvalue.csv"))
    mcs = E.model_confidence_set(losses)
    mcs.to_csv(os.path.join(a.out, "mcs.csv"))
    summary["mcs"] = mcs.round(4).to_dict(orient="index")

    # SHAP on the final LightGBM.
    _, lgb_model = M.lgbm(train, test, feat_cols, lgb_params)
    imp, sv, Xs = E.shap_importance(lgb_model, test[feat_cols])
    imp.to_csv(os.path.join(a.out, "shap_features.csv"), index=False)
    grp = imp.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
    grp.to_csv(os.path.join(a.out, "shap_groups.csv"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        shap.summary_plot(sv, Xs, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, "shap_summary.png"), dpi=130)
        plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 4))
        grp[::-1].plot.barh(ax=ax, color="#4C72B0")
        ax.set_xlabel("sum of mean |SHAP| (LightGBM, holdout)")
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, "shap_groups.png"), dpi=130)
        plt.close("all")
    except Exception as e:  # plotting is optional
        log(f"plotting skipped: {e}")

    # ---------------------------------------------------------------- ablations
    if not a.skip_ablation:
        groups = sorted({E.feature_group(c) for c in feat_cols} - {"other"})
        groups.append("multi-scale windows (last 500/300/200/100 s)")
        rows = []
        full_cv = cv_scores["LightGBM"][0]
        for gname in groups:
            if gname.startswith("multi-scale"):
                drop = set(E.multiscale_columns(feat_cols))
            else:
                drop = {c for c in feat_cols if E.feature_group(c) == gname}
            if not drop:
                continue
            cols = [c for c in feat_cols if c not in drop]
            cvs = []
            for tr_t, va_t in folds:
                tr, va = train[train.time_id.isin(tr_t)], train[train.time_id.isin(va_t)]
                cvs.append(M.rmspe(va.target, M.lgbm(tr, va, cols, lgb_params)[0]))
            ho = M.rmspe(y, M.lgbm(train, test, cols, lgb_params)[0])
            rows.append({"group_removed": gname, "n_features": len(drop), "cv_rmspe": np.mean(cvs),
                         "delta_cv": np.mean(cvs) - full_cv, "holdout_rmspe": ho,
                         "delta_holdout": ho - table.loc["LightGBM", "holdout_rmspe"]})
            log(f"ablation -{gname}: CV {np.mean(cvs):.4f} (Δ {np.mean(cvs) - full_cv:+.4f})  holdout {ho:.4f}")
        abl = pd.DataFrame(rows).sort_values("delta_cv", ascending=False)
        abl.to_csv(os.path.join(a.out, "ablations.csv"), index=False)
        summary["ablations"] = abl.round(5).to_dict(orient="records")

    summary["runtime_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    write_report(a, summary, table, dm_s, dm_p, mcs, grp, imp)
    log(f"done in {summary['runtime_s']}s -> {a.out}")


def write_report(a, summary, table, dm_s, dm_p, mcs, grp, imp):
    L = [f"# Results: `{a.data}` ({a.setting} setting)", ""]
    L.append(f"- samples: {summary['n_train']} train / {summary['n_test']} holdout (last 20% of time)")
    L.append(f"- features: {summary['n_features']}; time order: {summary['order_source']}")
    if "order_recovery_spearman" in summary:
        L.append(f"- time-order recovery from tick sizes: |Spearman ρ| = {summary['order_recovery_spearman']:.3f}")
    L += ["", "## Models (RMSPE, lower is better)", "",
          "| model | holdout RMSPE | CV RMSPE (purged, 5 folds) | improvement vs HAR-RV OLS |", "|---|---|---|---|"]
    for m, r in table.iterrows():
        cv = f"{r.cv_rmspe:.4f}" + (f" ± {r.cv_sd:.4f}" if np.isfinite(r.cv_sd) else "")
        L.append(f"| {m} | {r.holdout_rmspe:.4f} | {cv} | {100 * r.vs_HAR_OLS:+.1f}% |")
    L += ["", f"Stacking weights (chosen on purged out-of-fold predictions): {summary['stack_weights']}", ""]
    L += ["## Model Confidence Set (Hansen et al. 2011, α = 0.10)", "", "| model | MCS p-value | in MCS |", "|---|---|---|"]
    for m, r in mcs.iterrows():
        L.append(f"| {m} | {r.mcs_pvalue:.3f} | {'yes' if r.in_mcs else 'no'} |")
    best = table.index[0]
    L += ["", f"## Diebold–Mariano tests vs best model ({best})", "",
          "Statistic < 0 means the row model has lower loss than the best model; losses are averaged per time_id.", "",
          "| model | DM stat | p-value |", "|---|---|---|"]
    for m in table.index[1:]:
        L.append(f"| {m} | {dm_s.loc[m, best]:+.2f} | {dm_p.loc[m, best]:.4f} |")
    L += ["", "## SHAP attribution by feature group (LightGBM)", "", "| group | Σ mean abs SHAP |", "|---|---|"]
    for g, v in grp.items():
        L.append(f"| {g} | {v:.3g} |")
    L += ["", "Top features: " + ", ".join(imp.feature.head(12)), ""]
    if "ablations" in summary:
        L += ["## Ablations (LightGBM, tuned params, remove one feature group)", "",
              "| group removed | #features | CV RMSPE | Δ CV | holdout | Δ holdout |", "|---|---|---|---|---|---|"]
        for r in summary["ablations"]:
            L.append(f"| {r['group_removed']} | {r['n_features']} | {r['cv_rmspe']:.4f} | {r['delta_cv']:+.4f} | "
                     f"{r['holdout_rmspe']:.4f} | {r['delta_holdout']:+.4f} |")
    L += ["", f"Runtime: {summary['runtime_s']} s"]
    with open(os.path.join(a.out, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
