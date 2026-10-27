# Results: `data/crypto` (competition setting)

- samples: 14472 train / 3624 holdout (last 20% of time)
- features: 112; time order: true
- time-order recovery from tick sizes: |Spearman ρ| = 0.513

## Models (RMSPE, lower is better)

| model | holdout RMSPE | CV RMSPE (purged, 5 folds) | improvement vs HAR-RV OLS |
|---|---|---|---|
| Stacked ensemble | 0.1792 | 0.1916 | +15.6% |
| MLP (stock embedding) | 0.1830 | 0.1988 ± 0.0144 | +13.8% |
| LightGBM | 0.1832 | 0.1965 ± 0.0157 | +13.7% |
| XGBoost | 0.1833 | 0.1960 ± 0.0147 | +13.7% |
| OLS (all features, log) | 0.1837 | 0.1984 ± 0.0089 | +13.5% |
| HARQ (WLS) | 0.1918 | 0.2077 ± 0.0138 | +9.7% |
| 1D-CNN (per-second series) | 0.1934 | 0.2028 ± 0.0181 | +9.0% |
| naive (RV persistence) | 0.2099 | 0.2604 ± 0.0277 | +1.2% |
| HAR-RV (OLS, log) | 0.2124 | 0.2205 ± 0.0035 | +0.0% |

Stacking weights (chosen on purged out-of-fold predictions): {'HAR-RV (OLS, log)': 0.0709, 'HARQ (WLS)': 0.0, 'OLS (all features, log)': 0.2849, 'LightGBM': 0.1608, 'XGBoost': 0.1975, 'MLP (stock embedding)': 0.1705, '1D-CNN (per-second series)': 0.1154}

## Model Confidence Set (Hansen et al. 2011, α = 0.10)

| model | MCS p-value | in MCS |
|---|---|---|
| Stacked ensemble | 1.000 | yes |
| OLS (all features, log) | 0.117 | yes |
| MLP (stock embedding) | 0.117 | yes |
| LightGBM | 0.117 | yes |
| 1D-CNN (per-second series) | 0.069 | no |
| XGBoost | 0.031 | no |
| naive (RV persistence) | 0.000 | no |
| HARQ (WLS) | 0.000 | no |
| HAR-RV (OLS, log) | 0.000 | no |

## Diebold–Mariano tests vs best model (Stacked ensemble)

Statistic < 0 means the row model has lower loss than the best model; losses are averaged per time_id.

| model | DM stat | p-value |
|---|---|---|
| MLP (stock embedding) | +1.95 | 0.0524 |
| LightGBM | +1.90 | 0.0588 |
| XGBoost | +2.46 | 0.0143 |
| OLS (all features, log) | +2.21 | 0.0276 |
| HARQ (WLS) | +5.04 | 0.0000 |
| 1D-CNN (per-second series) | +2.60 | 0.0097 |
| naive (RV persistence) | +5.21 | 0.0000 |
| HAR-RV (OLS, log) | +4.50 | 0.0000 |

## SHAP attribution by feature group (LightGBM)

| group | Σ mean abs SHAP |
|---|---|
| realized (RV/BV/jumps/semivariance/RQ) | 0.0947 |
| cross-sectional (market aggregates, clusters) | 0.0856 |
| nearest-neighbour time_ids | 0.0521 |
| trades | 0.0362 |
| microstructure (spread, imbalance, OFI, depth, Kyle, Amihud) | 0.036 |
| lags across buckets | 0.0102 |
| stock id | 0.00868 |
| noise-robust (TSRV, realized kernel) | 0.00211 |
| price level / tick | 0.000195 |

Top features: rq1, nn5_rv1_w600, tsrv1_tmean, trade_orders_w100, rv1_w100_tmean, rv1_w600, nn20_rv1_w600, rsn1_w600, stock_id, tsrv1_tstd, lag1_rv1_w600, nn20_tsrv1


Runtime: 614.8 s
