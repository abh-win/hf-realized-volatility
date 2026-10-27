# Results: `data/synthetic` (strict setting)

- samples: 28770 train / 7200 holdout (last 20% of time)
- features: 125; time order: recovered
- time-order recovery from tick sizes: |Spearman ρ| = 1.000

## Models (RMSPE, lower is better)

| model | holdout RMSPE | CV RMSPE (purged, 5 folds) | improvement vs HAR-RV OLS |
|---|---|---|---|
| Stacked ensemble | 0.2197 | 0.2200 | +4.6% |
| OLS (all features, log) | 0.2219 | 0.2220 ± 0.0105 | +3.7% |
| XGBoost | 0.2241 | 0.2244 ± 0.0104 | +2.7% |
| LightGBM | 0.2243 | 0.2249 ± 0.0106 | +2.6% |
| MLP (stock embedding) | 0.2265 | 0.2240 ± 0.0104 | +1.6% |
| 1D-CNN (per-second series) | 0.2278 | 0.2289 ± 0.0109 | +1.1% |
| HAR-RV (OLS, log) | 0.2303 | 0.2295 ± 0.0108 | +0.0% |
| HARQ (WLS) | 0.2352 | 0.2321 ± 0.0106 | -2.1% |
| naive (RV persistence) | 0.2926 | 0.2996 ± 0.0171 | -27.1% |

Stacking weights (chosen on purged out-of-fold predictions): {'HAR-RV (OLS, log)': 0.0, 'HARQ (WLS)': 0.0, 'OLS (all features, log)': 0.4886, 'LightGBM': 0.0, 'XGBoost': 0.0775, 'MLP (stock embedding)': 0.2162, '1D-CNN (per-second series)': 0.2178}

## Model Confidence Set (Hansen et al. 2011, α = 0.10)

| model | MCS p-value | in MCS |
|---|---|---|
| Stacked ensemble | 1.000 | yes |
| naive (RV persistence) | 0.000 | no |
| HAR-RV (OLS, log) | 0.000 | no |
| XGBoost | 0.000 | no |
| LightGBM | 0.000 | no |
| HARQ (WLS) | 0.000 | no |
| 1D-CNN (per-second series) | 0.000 | no |
| OLS (all features, log) | 0.000 | no |
| MLP (stock embedding) | 0.000 | no |

## Diebold–Mariano tests vs best model (Stacked ensemble)

Statistic < 0 means the row model has lower loss than the best model; losses are averaged per time_id.

| model | DM stat | p-value |
|---|---|---|
| OLS (all features, log) | +4.21 | 0.0000 |
| XGBoost | +8.12 | 0.0000 |
| LightGBM | +8.55 | 0.0000 |
| MLP (stock embedding) | +4.80 | 0.0000 |
| 1D-CNN (per-second series) | +7.00 | 0.0000 |
| HAR-RV (OLS, log) | +11.40 | 0.0000 |
| HARQ (WLS) | +8.49 | 0.0000 |
| naive (RV persistence) | +17.27 | 0.0000 |

## SHAP attribution by feature group (LightGBM)

| group | Σ mean abs SHAP |
|---|---|
| realized (RV/BV/jumps/semivariance/RQ) | 0.11 |
| microstructure (spread, imbalance, OFI, depth, Kyle, Amihud) | 0.0854 |
| cross-sectional (market aggregates, clusters) | 0.0799 |
| nearest-neighbour time_ids | 0.0495 |
| lags across buckets | 0.0271 |
| trades | 0.0157 |
| level-2 (WAP2, L2 imbalance/spread/depth) | 0.0131 |
| stock id | 0.012 |
| noise-robust (TSRV, realized kernel) | 0.00746 |
| price level / tick | 0.00119 |

Top features: ofi_abs_w100, rq1, nn5_tsrv1, tsrv1_tratio, rv1_w600_tratio, lag1_rv1_w600, stock_id, ofi_abs_w200, lag6_rv1_w600, nn5_rv1_w600, clu_tsrv1_mean, bv1_w100

## Ablations (LightGBM, tuned params, remove one feature group)

| group removed | #features | CV RMSPE | Δ CV | holdout | Δ holdout |
|---|---|---|---|---|---|
| cross-sectional (market aggregates, clusters) | 24 | 0.2271 | +0.0023 | 0.2273 | +0.0030 |
| multi-scale windows (last 500/300/200/100 s) | 56 | 0.2270 | +0.0022 | 0.2262 | +0.0019 |
| realized (RV/BV/jumps/semivariance/RQ) | 26 | 0.2268 | +0.0019 | 0.2266 | +0.0022 |
| lags across buckets | 3 | 0.2254 | +0.0005 | 0.2247 | +0.0004 |
| nearest-neighbour time_ids | 8 | 0.2254 | +0.0005 | 0.2245 | +0.0002 |
| microstructure (spread, imbalance, OFI, depth, Kyle, Amihud) | 29 | 0.2253 | +0.0004 | 0.2242 | -0.0001 |
| price level / tick | 2 | 0.2247 | -0.0001 | 0.2243 | -0.0001 |
| level-2 (WAP2, L2 imbalance/spread/depth) | 13 | 0.2247 | -0.0002 | 0.2240 | -0.0003 |
| trades | 17 | 0.2246 | -0.0002 | 0.2244 | +0.0001 |
| noise-robust (TSRV, realized kernel) | 2 | 0.2246 | -0.0002 | 0.2240 | -0.0003 |
| stock id | 1 | 0.2245 | -0.0004 | 0.2244 | +0.0001 |

Runtime: 1011.3 s
