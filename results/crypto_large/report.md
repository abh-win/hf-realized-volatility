# Results: `data/crypto_large` (strict setting)

- samples: 51750 train / 12960 holdout (last 20% of time)
- features: 112; time order: true
- time-order recovery from tick sizes: |Spearman ρ| = 0.705

## Models (RMSPE, lower is better)

| model | holdout RMSPE | CV RMSPE (purged, 5 folds) | improvement vs HAR-RV OLS |
|---|---|---|---|
| Stacked ensemble | 0.1975 | 0.2298 | +12.9% |
| LightGBM | 0.1987 | 0.2313 ± 0.0196 | +12.3% |
| XGBoost | 0.1993 | 0.2318 ± 0.0194 | +12.0% |
| 1D-CNN (per-second series) | 0.2022 | 0.2400 ± 0.0222 | +10.8% |
| MLP (stock embedding) | 0.2023 | 0.2347 ± 0.0229 | +10.8% |
| HARQ (WLS) | 0.2113 | 0.2489 ± 0.0274 | +6.7% |
| OLS (all features, log) | 0.2126 | 0.2443 ± 0.0268 | +6.2% |
| HAR-RV (OLS, log) | 0.2266 | 0.2588 ± 0.0209 | +0.0% |
| naive (RV persistence) | 0.2531 | 0.3096 ± 0.0345 | -11.7% |

Stacking weights (chosen on purged out-of-fold predictions): {'HAR-RV (OLS, log)': 0.0456, 'HARQ (WLS)': 0.0, 'OLS (all features, log)': 0.0795, 'LightGBM': 0.2627, 'XGBoost': 0.2443, 'MLP (stock embedding)': 0.2629, '1D-CNN (per-second series)': 0.105}

## Model Confidence Set (Hansen et al. 2011, α = 0.10)

| model | MCS p-value | in MCS |
|---|---|---|
| Stacked ensemble | 1.000 | yes |
| LightGBM | 0.015 | no |
| XGBoost | 0.002 | no |
| HAR-RV (OLS, log) | 0.000 | no |
| naive (RV persistence) | 0.000 | no |
| MLP (stock embedding) | 0.000 | no |
| OLS (all features, log) | 0.000 | no |
| HARQ (WLS) | 0.000 | no |
| 1D-CNN (per-second series) | 0.000 | no |

## Diebold–Mariano tests vs best model (Stacked ensemble)

Statistic < 0 means the row model has lower loss than the best model; losses are averaged per time_id.

| model | DM stat | p-value |
|---|---|---|
| LightGBM | +2.18 | 0.0292 |
| XGBoost | +3.84 | 0.0001 |
| 1D-CNN (per-second series) | +4.38 | 0.0000 |
| MLP (stock embedding) | +5.08 | 0.0000 |
| HARQ (WLS) | +6.36 | 0.0000 |
| OLS (all features, log) | +6.85 | 0.0000 |
| HAR-RV (OLS, log) | +8.03 | 0.0000 |
| naive (RV persistence) | +7.35 | 0.0000 |

## SHAP attribution by feature group (LightGBM)

| group | Σ mean abs SHAP |
|---|---|
| realized (RV/BV/jumps/semivariance/RQ) | 0.0911 |
| cross-sectional (market aggregates, clusters) | 0.0663 |
| lags across buckets | 0.0359 |
| microstructure (spread, imbalance, OFI, depth, Kyle, Amihud) | 0.0317 |
| trades | 0.0269 |
| nearest-neighbour time_ids | 0.0209 |
| stock id | 0.0179 |
| noise-robust (TSRV, realized kernel) | 0.00149 |
| price level / tick | 0.000314 |

Top features: rq1, stock_id, lag1_rv1_w600, lag6_rv1_w600, rv1_w100_tstd, lag72_rv1_w600, rv1_w600_tratio, trade_secs_w100, trade_orders_w100, rv1_w100_tmean, ret_sum, rsn1_w600

## Ablations (LightGBM, tuned params, remove one feature group)

| group removed | #features | CV RMSPE | Δ CV | holdout | Δ holdout |
|---|---|---|---|---|---|
| realized (RV/BV/jumps/semivariance/RQ) | 26 | 0.2370 | +0.0057 | 0.2030 | +0.0043 |
| multi-scale windows (last 500/300/200/100 s) | 48 | 0.2330 | +0.0017 | 0.1998 | +0.0011 |
| lags across buckets | 3 | 0.2326 | +0.0013 | 0.2003 | +0.0016 |
| microstructure (spread, imbalance, OFI, depth, Kyle, Amihud) | 29 | 0.2324 | +0.0011 | 0.1998 | +0.0011 |
| trades | 17 | 0.2320 | +0.0007 | 0.1993 | +0.0007 |
| cross-sectional (market aggregates, clusters) | 24 | 0.2319 | +0.0006 | 0.1979 | -0.0008 |
| stock id | 1 | 0.2317 | +0.0004 | 0.1992 | +0.0005 |
| nearest-neighbour time_ids | 8 | 0.2316 | +0.0002 | 0.1981 | -0.0006 |
| price level / tick | 2 | 0.2313 | -0.0001 | 0.1987 | -0.0000 |
| noise-robust (TSRV, realized kernel) | 2 | 0.2312 | -0.0001 | 0.1984 | -0.0003 |

Runtime: 1301.9 s
