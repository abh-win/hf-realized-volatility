# Results: `data/crypto_large` (competition setting)

- samples: 51750 train / 12960 holdout (last 20% of time)
- features: 112; time order: true
- time-order recovery from tick sizes: |Spearman ρ| = 0.705

## Models (RMSPE, lower is better)

| model | holdout RMSPE | CV RMSPE (purged, 5 folds) | improvement vs HAR-RV OLS |
|---|---|---|---|
| Stacked ensemble | 0.1945 | 0.2256 | +14.2% |
| LightGBM | 0.1953 | 0.2280 ± 0.0201 | +13.8% |
| XGBoost | 0.1963 | 0.2281 ± 0.0197 | +13.4% |
| MLP (stock embedding) | 0.1985 | 0.2306 ± 0.0222 | +12.4% |
| 1D-CNN (per-second series) | 0.1992 | 0.2375 ± 0.0263 | +12.1% |
| OLS (all features, log) | 0.2098 | 0.2395 ± 0.0269 | +7.4% |
| HARQ (WLS) | 0.2113 | 0.2489 ± 0.0274 | +6.7% |
| HAR-RV (OLS, log) | 0.2266 | 0.2588 ± 0.0209 | +0.0% |
| naive (RV persistence) | 0.2531 | 0.3096 ± 0.0345 | -11.7% |

Stacking weights (chosen on purged out-of-fold predictions): {'HAR-RV (OLS, log)': 0.0, 'HARQ (WLS)': 0.0, 'OLS (all features, log)': 0.181, 'LightGBM': 0.2784, 'XGBoost': 0.2137, 'MLP (stock embedding)': 0.2521, '1D-CNN (per-second series)': 0.0748}

## Model Confidence Set (Hansen et al. 2011, α = 0.10)

| model | MCS p-value | in MCS |
|---|---|---|
| Stacked ensemble | 1.000 | yes |
| LightGBM | 0.341 | yes |
| XGBoost | 0.055 | no |
| 1D-CNN (per-second series) | 0.004 | no |
| HAR-RV (OLS, log) | 0.000 | no |
| MLP (stock embedding) | 0.000 | no |
| OLS (all features, log) | 0.000 | no |
| naive (RV persistence) | 0.000 | no |
| HARQ (WLS) | 0.000 | no |

## Diebold–Mariano tests vs best model (Stacked ensemble)

Statistic < 0 means the row model has lower loss than the best model; losses are averaged per time_id.

| model | DM stat | p-value |
|---|---|---|
| LightGBM | +0.99 | 0.3243 |
| XGBoost | +2.60 | 0.0094 |
| MLP (stock embedding) | +4.43 | 0.0000 |
| 1D-CNN (per-second series) | +3.62 | 0.0003 |
| OLS (all features, log) | +6.58 | 0.0000 |
| HARQ (WLS) | +7.29 | 0.0000 |
| HAR-RV (OLS, log) | +8.84 | 0.0000 |
| naive (RV persistence) | +7.52 | 0.0000 |

## SHAP attribution by feature group (LightGBM)

| group | Σ mean abs SHAP |
|---|---|
| realized (RV/BV/jumps/semivariance/RQ) | 0.0988 |
| cross-sectional (market aggregates, clusters) | 0.0571 |
| nearest-neighbour time_ids | 0.051 |
| microstructure (spread, imbalance, OFI, depth, Kyle, Amihud) | 0.0321 |
| trades | 0.0207 |
| lags across buckets | 0.0118 |
| stock id | 0.00946 |
| noise-robust (TSRV, realized kernel) | 0.00265 |
| price level / tick | 0.000598 |

Top features: rq1, nn5_rv1_w600, nn20_rv1_w600, stock_id, rv1_w600, jump1_w600, rsn1_w600, rv1_w100_tstd, ret_sum, rv1_w100_tmean, rv1_w600_tratio, lag1_rv1_w600


Runtime: 912.6 s
