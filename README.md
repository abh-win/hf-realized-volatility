# High-Frequency Realized Volatility Prediction

Forecasting the next 10 minutes of realized volatility from 10 minutes of tick-level order-book and
trade data, framed as in the **Optiver Realized Volatility Prediction** competition. The pipeline
covers:

- **Features:** realized measures and noise-robust estimators, microstructure features (WAP, log
  returns, order-flow imbalance, …), multi-scale windows, and cross-sectional features including
  time-order recovery and nearest-neighbour time_ids.
- **Model ladder:** HAR-RV/HARQ, OLS, LightGBM/XGBoost with an RMSPE-aware loss tuned by Optuna on
  purged and embargoed time-series CV, an MLP and a 1D-CNN, and stacking.
- **Rigor:** Diebold–Mariano tests, a Model Confidence Set, SHAP attribution and feature-group
  ablations.

The full specification this implements is in `SPEC.md`.

## Data

| dataset | source | stocks × buckets | used for |
|---|---|---|---|
| `data/optiver` | Kaggle competition (needs an API token: `scripts/download_optiver.sh`) | 112 × 3,830 | drop-in: same format |
| `data/crypto_large` | **real** Binance USD-M futures, 18 perpetuals, 2024-02-09 → 03-29 (50 days) | 18 × 3,600 | main results |
| `data/crypto` | same, 12 perpetuals, 2024-03-01 → 03-21 (21 days) | 12 × 1,512 | smaller replication |
| `data/synthetic` | `hfvol/synth.py` simulator (L1 + L2 book, shuffled time_ids) | 30 × 1,200 | tests the L2 / time-recovery paths |

**Crypto → Optiver format** (`hfvol/binance.py`, `scripts/make_crypto_dataset.py`). Each UTC day is
cut into 72 twenty-minute blocks:

- **Features** come from the first 10 minutes: 1-second best bid/ask snapshots from `bookTicker`
  and per-second trade aggregates from `aggTrades`.
- **Target** is the realized volatility of the 1-second WAP over the next 10 minutes, the
  competition's definition.
- **Prices** are normalised per bucket, as Optiver's are.
- **Level 2:** Binance's archive has no historical L2 depth, so the L2 features are exercised on
  the synthetic data and will be on Optiver's.

**Synthetic generator** (`hfvol/synth.py`). It encodes:

- **Volatility structure:** stochastic vol with market, sector and idiosyncratic AR(1) factors;
  an intra-bucket OU vol path; jumps.
- **Microstructure:** tick discreteness and bid-ask bounce, and an order-book imbalance that
  predicts returns.
- **Order recovery:** real price levels on a 0.01 tick grid, so the tick-size time-order trick
  can be tested.
- **Calibration:** its targets match Optiver's distribution (median ≈ 0.0037, log-sd ≈ 0.63).

## Features (`hfvol/features.py`, `hfvol/cross_section.py`)

| group | contents |
|---|---|
| Realized measures | RV of WAP1 (and WAP2), bipower variation (Barndorff-Nielsen & Shephard), jump variation RV−BV, realized semivariances RS⁺/RS⁻, realized quarticity |
| Noise-robust | two-scale RV (Zhang, Mykland & Aït-Sahalia 2005), Parzen realized kernel with BNHLS optimal bandwidth |
| Microstructure | relative spread (level, std, trend), L1/L2 depth imbalance, Cont–Kukanov–Stoikov order-flow imbalance (signed and absolute), depth, ∣WAP1−WAP2∣, Kyle's λ (WAP returns on signed volume; Lee–Ready signing when no aggressor flag), Amihud illiquidity, range, update count |
| Trades | trade seconds, volume, order count, trade-price RV, signed-volume share |
| Multi-scale | the above over the last 600/500/300/200/100 s of the bucket (HAR inside the bucket) |
| Cross-sectional | leave-one-out market mean/std per time_id and the stock's ratio to it; KMeans clusters of stocks on log-RV correlation (fit on training time only) and cluster means |
| Time order & neighbours | price level from the normalised tick size; **Isomap** 1-D embedding of time_ids in price space recovers chronology (Spearman ρ = 1.000 on synthetic data vs 0.62 for spectral seriation); k-NN time_id features (own and market RV averaged over the 5/20 nearest time_ids) |
| Lags | HAR lags across buckets (previous bucket, last 6, last 72 = one day) in true or recovered order |

All estimators are vectorised over a dense (buckets × 600 s) grid.

## Models (`hfvol/models.py`)

1. **Naive** persistence: RV of the last 10 minutes.
2. **HAR-RV** (Corsi 2009): OLS on log-RV with daily/weekly/monthly analogues (600/300/100 s windows
   plus 1/6/72-bucket lags). The back-transform is RMSPE-optimal: if log y ~ N(μ, s²), then
   argmin E[((p−y)/y)²] = exp(μ − 1.5 s²).
3. **HARQ** (Bollerslev, Patton & Quaedvlieg 2016): RV persistence attenuated by √RQ, fitted by
   WLS with weights 1/y².
4. **OLS on all features** (log, winsorised, stock fixed effects): the linear counterpart of the
   tree models on the same inputs.
5. **LightGBM / XGBoost**:
   - Weighted RMSE with weights 1/y², fitted on the ratio target y/RV₆₀₀. This is the same RMSPE
     objective, scale-free across stocks.
   - Three-seed bagging.
   - **Optuna** TPE (40 / 20 trials), scored by mean RMSPE over purged time-series folds.
6. **MLP** on quantile-normalised features plus a stock embedding. It predicts log(y/RV₆₀₀) and
   trains directly on RMSPE.
7. **1D-CNN** over the per-second series (60 bins × 5 channels: squared returns, spread,
   imbalance, OFI, volume), plus global features and a stock embedding.
8. **Stacking**: non-negative weights summing to 1, chosen on purged out-of-fold predictions.

**Validation** (`hfvol/cv.py`):

- **Holdout:** the last 20% of time.
- **Purged CV:** 5-fold time-series CV with purging and embargo (López de Prado 2018) on the
  remaining 80%, used for Optuna, OOF predictions and stacking weights.

**Strict vs competition setting.** The *strict* setting only uses information available at
forecast time: neighbour time_ids from the past only, clusters fit on training data only. The
*competition* setting lets neighbours come from any time_id, as Optiver's rules effectively
allowed.

**Evaluation** (`hfvol/evaluation.py`):

- **Losses:** squared percentage errors, averaged per time_id to absorb cross-sectional
  dependence.
- **Diebold–Mariano:** Newey–West variance with the Harvey–Leybourne–Newbold correction.
- **Model Confidence Set:** Hansen, Lunde & Nason 2011, block bootstrap via `arch`.
- **SHAP:** TreeExplainer.
- **Ablations:** re-fit LightGBM with each feature group removed.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/make_crypto_dataset.py data/crypto_large --start 2024-02-09 --days 50 \
    --symbols XRPUSDT,LTCUSDT,LINKUSDT,ADAUSDT,AVAXUSDT,DOTUSDT,BCHUSDT,ATOMUSDT,UNIUSDT,FILUSDT,NEARUSDT,ETCUSDT,EOSUSDT,XLMUSDT,TRXUSDT,ALGOUSDT,AAVEUSDT,SANDUSDT
.venv/bin/python scripts/run_pipeline.py --data data/crypto_large --out results/crypto_large --setting strict
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from hfvol.synth import generate; generate('data/synthetic')"
.venv/bin/python scripts/run_pipeline.py --data data/synthetic --out results/synthetic
scripts/download_optiver.sh && .venv/bin/python scripts/run_pipeline.py --data data/optiver --out results/optiver --setting competition
.venv/bin/python -m pytest -q tests      # estimator/CV/DM/stacking unit tests
```

Each run writes `report.md`, `summary.json`, `model_table.csv`, `dm_*.csv`, `mcs.csv`,
`shap_*.csv/png` and `ablations.csv` to its results directory.

## Results

### Main results: 18 crypto perpetuals, 50 days (`results/crypto_large`, strict setting)

There are 64,728 buckets: 51,750 for training and tuning, and a 12,960-bucket holdout covering the
last 20% of time (10 days). Tuning used Optuna with 40 LightGBM and 20 XGBoost trials on 3 purged
folds; out-of-fold predictions use 5 purged folds.

| model | holdout RMSPE | CV RMSPE (purged 5-fold) | vs HAR-RV OLS |
|---|---|---|---|
| **Stacked ensemble** | **0.1975** | 0.2298 | **+12.9%** |
| LightGBM (ratio target, Optuna) | 0.1987 | 0.2313 ± 0.020 | +12.3% |
| XGBoost (ratio target, Optuna) | 0.1993 | 0.2318 ± 0.019 | +12.0% |
| 1D-CNN (per-second series) | 0.2022 | 0.2400 ± 0.022 | +10.8% |
| MLP (stock embedding) | 0.2023 | 0.2347 ± 0.023 | +10.8% |
| HARQ (WLS) | 0.2113 | 0.2489 ± 0.027 | +6.7% |
| OLS (all features, log) | 0.2126 | 0.2443 ± 0.027 | +6.2% |
| HAR-RV (OLS, log) | 0.2266 | 0.2588 ± 0.021 | 0 |
| naive persistence | 0.2531 | 0.3096 ± 0.035 | −11.7% |

- **Headline:** gradient boosting cuts holdout RMSPE by **12.3%** relative to the HAR-RV OLS
  baseline, and the stack by **12.9%**. Relative to naive persistence the cuts are 21.5% and
  22.0%. This falls short of the >15% figure on the CV: on this crypto sample the gain over the
  OLS baseline is about 12–13%.
- **Statistical significance:**
  - **MCS (α = 0.10):** contains only the stacked ensemble. LightGBM is excluded at p = 0.015,
    everything else at p ≤ 0.002.
  - **Diebold–Mariano:** the stack beats LightGBM (DM = 2.18, p = 0.029), XGBoost (3.84), both
    neural nets (4.4–5.1) and HAR-RV (8.03).
- **Stack weights** (from OOF predictions): LightGBM 0.26, MLP 0.26, XGBoost 0.24, CNN 0.11,
  OLS 0.08, HAR 0.05.
- **SHAP by feature group:** realized measures 0.091, cross-sectional market state 0.066, lags
  0.036, microstructure 0.032, trades 0.027, neighbours 0.021. The top features are realized
  quarticity, stock id, the RV lags (1, 6 and 72 buckets), the cross-sectional dispersion of
  100-second RV, the ratio of own RV to the market's, and recent trade activity.
- **Ablations** (change in CV RMSPE when a group is removed):

  | group removed | Δ CV RMSPE |
  |---|---|
  | realized measures | +0.0057 |
  | multi-scale windows | +0.0017 |
  | lags across buckets | +0.0013 |
  | microstructure | +0.0011 |
  | trades | +0.0007 |
  | cross-sectional | +0.0006 |

  Noise-robust estimators and the price level add nothing beyond the rest. At 1-second sampling
  on liquid perpetuals, bid-ask bounce is small relative to the signal.

**Competition setting** (`results/crypto_large_competition`): nearest-neighbour time_ids may come
from any time, as in Optiver's competition, where the whole test period's book data was available.

| | holdout RMSPE | vs HAR-RV OLS |
|---|---|---|
| Stacked ensemble | 0.1945 | +14.2% |
| LightGBM | 0.1953 | +13.8% |
| XGBoost | 0.1963 | +13.4% |

Future neighbours are worth about 1.5% RMSPE here, and 2.5% on the 21-day set. That is a
look-ahead gain a live forecaster cannot use, so the strict numbers above are the honest ones.



### Smaller replication: 12 crypto perpetuals, 21 days (`results/crypto`, strict)

These are holdout RMSPE figures, with the holdout being the last 20% of time (3,624 buckets).

| model | holdout RMSPE | CV RMSPE | vs HAR-RV OLS |
|---|---|---|---|
| Stacked ensemble | **0.1838** | 0.1961 | **+13.5%** |
| LightGBM (ratio target) | 0.1843 | 0.2009 | +13.2% |
| MLP (stock embedding) | 0.1844 | 0.2026 | +13.2% |
| XGBoost (ratio target) | 0.1851 | 0.2004 | +12.9% |
| OLS (all features) | 0.1898 | 0.2051 | +10.6% |
| HARQ | 0.1918 | 0.2077 | +9.7% |
| 1D-CNN | 0.1921 | 0.2109 | +9.5% |
| naive | 0.2099 | 0.2604 | +1.2% |
| HAR-RV OLS | 0.2124 | 0.2205 | 0 |

- **MCS (α = 0.10):** {stack, LightGBM, XGBoost, MLP}. HAR, HARQ, OLS and naive are excluded at
  p ≤ 0.003.
- **DM tests:** the stack beats HAR-RV with DM = 4.2 (p < 1e-4), but is statistically
  indistinguishable from LightGBM, XGBoost and the MLP (p = 0.18–0.40).
- **Competition setting:** allowing neighbour time_ids from the future improves the stack to
  0.1792 (+15.6%). Roughly 2.5% of the competition-style gain is therefore look-ahead that a live
  forecaster cannot use.
- **SHAP:** realized measures and the cross-sectional market state dominate. The top features are
  realized quarticity, market TSRV, the previous bucket's RV, recent trade orders and the market
  mean of 100-second RV.
- **Ablations:** removing the realized-measure group hurts most (+0.0063 CV RMSPE), followed by
  the multi-scale windows (+0.0034), lags, trades and microstructure.

### Synthetic data (`results/synthetic`, recovered time order)

- **Time order:** the true order is recovered exactly (Spearman ρ = 1.000) from tick sizes alone.
- **Models:** the stack improves on HAR-RV by 4.6% (0.2197 vs 0.2303). The best single model is
  the linear OLS, as expected: the simulator's volatility is log-linear by construction, so this
  dataset validates the plumbing (L2 features, recovered order, strict neighbours) rather than
  non-linear signal.

## Layout

```
src/hfvol/   data.py  realized.py  features.py  cross_section.py  cv.py  models.py  evaluation.py
             binance.py (crypto -> Optiver format)  synth.py (synthetic generator)
scripts/     run_pipeline.py  make_crypto_dataset.py  download_optiver.sh
tests/       test_realized.py (RV/BV/jumps/RQ/TSRV/RK properties, OFI, tick recovery, purged CV, DM, stacking)
results/     one directory per dataset/setting
```

## Notes and limitations

- The reported numbers are for crypto perpetuals, not Optiver's equities; the code runs unchanged
  on the Optiver data once a Kaggle token is available.
- Binance's public archive has no historical L2 depth, so L2 features are absent from the crypto
  results.
- On crypto, time-order recovery from tick sizes is weaker (ρ ≈ 0.5). Twelve to eighteen highly
  correlated coins trace a path that crosses itself; the crypto runs therefore use the true
  chronology, which is known.
