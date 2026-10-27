# Results: `data/crypto` (strict setting)

- samples: 14472 train / 3624 holdout (last 20% of time)
- features: 112; time order: true
- time-order recovery from tick sizes: |Spearman ρ| = 0.513

## Models (RMSPE, lower is better)

| model | holdout RMSPE | CV RMSPE (purged, 5 folds) | improvement vs HAR-RV OLS |
|---|---|---|---|
| Stacked ensemble | 0.1817 | 0.1966 | +14.4% |
| LightGBM | 0.1843 | 0.2009 ± 0.0163 | +13.2% |
| MLP (stock embedding) | 0.1844 | 0.2026 ± 0.0153 | +13.2% |
| XGBoost | 0.1851 | 0.2004 ± 0.0155 | +12.8% |
| OLS (all features, log) | 0.1898 | 0.2051 ± 0.0072 | +10.7% |
| HARQ (WLS) | 0.1918 | 0.2077 ± 0.0138 | +9.7% |
| 1D-CNN (per-second series) | 0.1921 | 0.2109 ± 0.0228 | +9.5% |
| naive (RV persistence) | 0.2099 | 0.2604 ± 0.0277 | +1.2% |
| HAR-RV (OLS, log) | 0.2124 | 0.2205 ± 0.0035 | +0.0% |

Stacking weights (chosen on purged out-of-fold predictions): {'HAR-RV (OLS, log)': 0.1054, 'HARQ (WLS)': 0.0154, 'OLS (all features, log)': 0.2763, 'LightGBM': 0.1797, 'XGBoost': 0.2168, 'MLP (stock embedding)': 0.1808, '1D-CNN (per-second series)': 0.0257}

## Model Confidence Set (Hansen et al. 2011, α = 0.10)

| model | MCS p-value | in MCS |
|---|---|---|
| Stacked ensemble | 1.000 | yes |
| LightGBM | 0.398 | yes |
| XGBoost | 0.398 | yes |
| MLP (stock embedding) | 0.334 | yes |
| 1D-CNN (per-second series) | 0.085 | no |
| OLS (all features, log) | 0.003 | no |
| HARQ (WLS) | 0.002 | no |
| naive (RV persistence) | 0.000 | no |
| HAR-RV (OLS, log) | 0.000 | no |

## Diebold–Mariano tests vs best model (Stacked ensemble)

Statistic < 0 means the row model has lower loss than the best model; losses are averaged per time_id.

| model | DM stat | p-value |
|---|---|---|
| LightGBM | +0.85 | 0.3980 |
| MLP (stock embedding) | +1.36 | 0.1751 |
| XGBoost | +1.13 | 0.2596 |
| OLS (all features, log) | +4.16 | 0.0000 |
| HARQ (WLS) | +4.13 | 0.0000 |
| 1D-CNN (per-second series) | +2.16 | 0.0319 |
| naive (RV persistence) | +4.58 | 0.0000 |
| HAR-RV (OLS, log) | +4.20 | 0.0000 |

## SHAP attribution by feature group (LightGBM)

| group | Σ mean abs SHAP |
|---|---|
| realized (RV/BV/jumps/semivariance/RQ) | 0.0844 |
| cross-sectional (market aggregates, clusters) | 0.0824 |
| trades | 0.0321 |
| microstructure (spread, imbalance, OFI, depth, Kyle, Amihud) | 0.0318 |
| lags across buckets | 0.0274 |
| nearest-neighbour time_ids | 0.0213 |
| stock id | 0.0106 |
| noise-robust (TSRV, realized kernel) | 0.00163 |
| price level / tick | 0.000146 |

Top features: rq1, tsrv1_tmean, lag1_rv1_w600, trade_orders_w100, rv1_w100_tmean, stock_id, rv1_w600_tstd, ofi_abs_w600_tstd, lag72_rv1_w600, rsn1_w600, tsrv1_tstd, lag6_rv1_w600

## Ablations (LightGBM, tuned params, remove one feature group)

| group removed | #features | CV RMSPE | Δ CV | holdout | Δ holdout |
|---|---|---|---|---|---|
| realized (RV/BV/jumps/semivariance/RQ) | 26 | 0.2073 | +0.0063 | 0.1872 | +0.0029 |
| multi-scale windows (last 500/300/200/100 s) | 48 | 0.2043 | +0.0034 | 0.1851 | +0.0009 |
| lags across buckets | 3 | 0.2024 | +0.0015 | 0.1848 | +0.0006 |
| trades | 17 | 0.2022 | +0.0013 | 0.1854 | +0.0011 |
| microstructure (spread, imbalance, OFI, depth, Kyle, Amihud) | 29 | 0.2015 | +0.0006 | 0.1855 | +0.0012 |
| stock id | 1 | 0.2013 | +0.0004 | 0.1851 | +0.0008 |
| cross-sectional (market aggregates, clusters) | 24 | 0.2013 | +0.0004 | 0.1808 | -0.0034 |
| noise-robust (TSRV, realized kernel) | 2 | 0.2010 | +0.0000 | 0.1843 | +0.0000 |
| price level / tick | 2 | 0.2009 | -0.0001 | 0.1844 | +0.0001 |
| nearest-neighbour time_ids | 8 | 0.2008 | -0.0001 | 0.1836 | -0.0006 |

Runtime: 934.6 s
