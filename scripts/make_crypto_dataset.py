#!/usr/bin/env python3
"""Build an Optiver-format dataset from Binance USD-M futures public data.

Usage: make_crypto_dataset.py OUT_DIR [--start 2024-03-01] [--days 21] [--workers 4]
       [--symbols XRPUSDT,LTCUSDT,...]
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hfvol.binance import process_day, write_day  # noqa: E402

DEFAULT_SYMBOLS = ["XRPUSDT", "LTCUSDT", "LINKUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "BCHUSDT",
                   "ATOMUSDT", "UNIUSDT", "FILUSDT", "NEARUSDT", "ETCUSDT"]


def job(out, sid, sym, date, di):
    for attempt in range(3):
        try:
            res = process_day(sym, date, di)
            write_day(out, sid, date, res)
            return sym, date, None
        except Exception as e:  # network hiccups: retry
            err = e
            time.sleep(5 * (attempt + 1))
    return sym, date, repr(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--start", default="2024-03-01")
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "targets"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "meta"), exist_ok=True)
    symbols = a.symbols.split(",")
    dates = [str(d.date()) for d in pd.date_range(a.start, periods=a.days, freq="D")]
    jobs = []
    for sid, sym in enumerate(symbols):
        for di, date in enumerate(dates):
            if os.path.exists(os.path.join(a.out, "targets", f"{sid}_{date}.csv")):
                continue
            jobs.append((sid, sym, date, di))
    print(f"{len(jobs)} symbol-days to fetch", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, a.out, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            sym, date, err = f.result()
            print(f"[{i}/{len(jobs)} {time.time() - t0:6.0f}s] {sym} {date} {'ERROR ' + err if err else 'ok'}", flush=True)
    tg = pd.concat(map(pd.read_csv, glob.glob(os.path.join(a.out, "targets", "*.csv"))))
    tg.sort_values(["stock_id", "time_id"]).to_csv(os.path.join(a.out, "train.csv"), index=False)
    meta = pd.concat(map(pd.read_csv, glob.glob(os.path.join(a.out, "meta", "*.csv"))))
    meta.sort_values(["stock_id", "time_id"]).to_csv(os.path.join(a.out, "time_ids.csv"), index=False)
    pd.DataFrame({"stock_id": range(len(symbols)), "symbol": symbols}).to_csv(os.path.join(a.out, "stocks.csv"), index=False)
    print(f"wrote {len(tg)} targets", flush=True)


if __name__ == "__main__":
    main()
