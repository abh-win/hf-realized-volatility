# Spec (from user, 2026-10-17)

CV pointer:
- Engineered high-frequency microstructure features from 10-min tick data, computing WAP, log returns, and net order flow imbalance.
- Benchmarked predictive forecasting models for realized volatility, evaluating baseline OLS against non-linear architectures (XGBoost).
- Optimized model hyperparameters utilizing Optuna and strict time-series CV, achieving a >15% reduction in RMSPE over OLS baselines.

Data: Optiver Realized Volatility Prediction (Kaggle; token at ~/.kaggle/kaggle.json).

## Feature engineering (literature-grounded)
- Realized measures: RV / realized vol from WAP (1- and 2-level), bipower variation (Barndorff-Nielsen & Shephard) -> jump component RV − BV, realized semivariances (good/bad vol), realized quarticity (for HARQ).
- Noise-robust estimators: Two-Scale RV (Zhang, Mykland, Aït-Sahalia 2005), realized kernels (Barndorff-Nielsen et al. 2008).
- Microstructure: order flow imbalance (Cont, Kukanov, Stoikov 2014), L1/L2 depth imbalance, spread dynamics, trade intensity, Kyle's lambda / Amihud illiquidity.
- Multi-scale windows: features over last 100/200/300/500s of each 10-min bucket (HAR inside the bucket).
- Cross-sectional: time-id aggregates across stocks (market vol), stock clusters (KMeans / correlation graph on returns), nearest-neighbour time-ids (recover time ordering from price tick sizes — winning Optiver trick).

## Model ladder
1. HAR-RV (Corsi 2009) and HARQ (Bollerslev, Patton, Quaedvlieg 2016) econometric baselines (OLS on log-RV).
2. LightGBM / XGBoost with RMSPE-aware loss (1/y² weights).
3. Small neural model (MLP or 1D-CNN over per-second series) with stock embedding.
4. Stacking ensemble, weights chosen on purged/embargoed time-series CV (López de Prado).

## Rigor
- SHAP attribution; ablations by feature group on RMSPE.
- Diebold–Mariano tests between models.
- Model Confidence Set (Hansen et al.) rather than a single "best" number.
