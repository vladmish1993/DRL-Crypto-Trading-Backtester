#!/usr/bin/env python3
"""Precompute indicators/features on the FULL dataset, then save to a new CSV.

Why this exists
- You avoid recomputing indicators/normalisation every run.
- Indicators (including ADX) are computed causally (past-only), so computing on the full series is OK.

Usage (PowerShell)
  python scripts\precompute_features.py --in data\SOL_USDT_15m.csv --out data\SOL_USDT_15m_features.csv

Notes
- This script expects your indicators.py:add_indicators() to already create the columns you want
  (e.g. adx + adx_norm if you've added ADX).
- Normalisation is rolling, so it does NOT use future data.
"""

import argparse
import os
import sys

import pandas as pd

# allow imports from scripts/
sys.path.insert(0, os.path.dirname(__file__))

from indicators import add_indicators, normalize_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='Input OHLCV CSV (must include timestamp, open, high, low, close, volume)')
    ap.add_argument('--out', dest='out', required=True, help='Output CSV with indicators/features')
    ap.add_argument('--window', type=int, default=100, help='Rolling window for normalisation (default: 100)')
    ap.add_argument('--keep_timestamp', action='store_true', help='Keep timestamp column (default: true if present)')
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = add_indicators(df)

    # If you use the RL FEATURES list, pass it here; otherwise normalise what exists.
    # This keeps behaviour consistent with the rest of your pipeline.
    feature_cols = [c for c in df.columns if c.endswith('_norm')]
    if not feature_cols:
        # fall back to a common set used elsewhere
        feature_cols = [
            'close_norm', 'open_norm', 'high_norm', 'low_norm',
            'sma_20_norm', 'sma_50_norm',
            'rsi_norm', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
            'bb_width_norm', 'atr_norm', 'volume_ratio_norm', 'returns'
        ]

    df = normalize_features(df, feature_cols, window=args.window)

    # Write
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == '__main__':
    main()
