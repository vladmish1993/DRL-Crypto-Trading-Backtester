#!/usr/bin/env python3
"""
Generate realistic synthetic SOL/USDT 15m data for testing the pipeline
when you don't have the real CSV yet.

    python scripts/generate_sample_data.py

Creates  data/SOL_USDT_15m.csv  with ≈ 40 000 rows  (~6 months of 15 m bars).
"""
import numpy as np
import pandas as pd
import os

def generate(n_bars: int = 40_000, seed: int = 42):
    rng = np.random.default_rng(seed)

    # Start price around SOL's 2024 range
    price = 100.0
    dt = pd.Timestamp('2024-01-01')
    freq = pd.Timedelta(minutes=15)

    rows = []
    for i in range(n_bars):
        # Regime-switching: trending vs mean-reverting
        regime = np.sin(2 * np.pi * i / 5000)  # slow cycle
        drift  = 0.00002 * regime               # slight trend following regime
        vol    = 0.003 + 0.002 * abs(regime)     # volatility varies

        ret = rng.normal(drift, vol)
        # Occasional jumps (news / liquidations)
        if rng.random() < 0.002:
            ret += rng.choice([-1, 1]) * rng.uniform(0.02, 0.06)

        close = price * (1 + ret)
        close = max(close, 1.0)

        high  = close * (1 + abs(rng.normal(0, 0.003)))
        low   = close * (1 - abs(rng.normal(0, 0.003)))
        opn   = low + (high - low) * rng.random()
        vol_usd = rng.lognormal(18, 1.0)   # volume in USD

        rows.append(dict(
            timestamp=dt + i * freq,
            open=round(opn, 4),
            high=round(high, 4),
            low=round(low, 4),
            close=round(close, 4),
            volume=round(vol_usd, 2),
        ))
        price = close

    df = pd.DataFrame(rows)

    os.makedirs('data', exist_ok=True)
    path = 'data/SOL_USDT_15m.csv'
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} bars → {path}")
    print(f"  Range: {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")
    print(f"  Price: ${df['close'].iloc[0]:.2f}  →  ${df['close'].iloc[-1]:.2f}")
    return df


if __name__ == '__main__':
    generate()