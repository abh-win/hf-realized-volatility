#!/usr/bin/env bash
# Downloads the Optiver Realized Volatility Prediction data (Kaggle, 2021).
# Requires: a Kaggle API token in ~/.kaggle/kaggle.json (Kaggle > Settings > API >
# Create New Token) and accepting the competition rules at
# https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/rules
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/optiver
mkdir -p "$OUT"
.venv/bin/kaggle competitions download -c optiver-realized-volatility-prediction -p "$OUT"
cd "$OUT" && unzip -q -o optiver-realized-volatility-prediction.zip && rm optiver-realized-volatility-prediction.zip
echo "done: $(ls)"
echo "run:  .venv/bin/python scripts/run_pipeline.py --data data/optiver --out results/optiver --setting competition"
