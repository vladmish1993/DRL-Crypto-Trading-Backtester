"""
Fetch SOL/USDT 15m futures data from Binance using ccxt.
Run this script once to download historical data to data/ directory.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --symbol SOL/USDT --timeframe 15m --start 2023-06-01
"""
import ccxt
import pandas as pd
import os
import time
import argparse
from datetime import datetime


def fetch_binance_futures(symbol='SOL/USDT', timeframe='15m',
                          start_date='2023-06-01', end_date=None,
                          output_dir='data'):
    """
    Fetch OHLCV candle data from Binance USDT-M Futures.
    Downloads in chunks of 1000 candles, respecting rate limits.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })

    os.makedirs(output_dir, exist_ok=True)

    since = exchange.parse8601(f'{start_date}T00:00:00Z')
    end_ts = exchange.parse8601(f'{end_date}T00:00:00Z') if end_date else exchange.milliseconds()

    all_candles = []
    print(f"Fetching {symbol} {timeframe} futures data from {start_date} ...")

    while since < end_ts:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not candles:
                break

            all_candles.extend(candles)
            since = candles[-1][0] + 1

            latest = datetime.utcfromtimestamp(candles[-1][0] / 1000).strftime('%Y-%m-%d %H:%M')
            print(f"  {len(all_candles):>7} candles  |  up to {latest}")

            time.sleep(exchange.rateLimit / 1000)

        except ccxt.RateLimitExceeded:
            print("  Rate limited, sleeping 10s ...")
            time.sleep(10)
        except Exception as e:
            print(f"  Error: {e}, retrying in 5s ...")
            time.sleep(5)

    # Build DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    # Save
    safe_symbol = symbol.replace('/', '_')
    filename = f"{safe_symbol}_{timeframe}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)

    print(f"\nSaved {len(df)} candles to {filepath}")
    print(f"Range: {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Binance Futures OHLCV data')
    parser.add_argument('--symbol', default='SOL/USDT')
    parser.add_argument('--timeframe', default='15m')
    parser.add_argument('--start', default='2023-06-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--output', default='data')
    args = parser.parse_args()

    fetch_binance_futures(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
    )
