#!/usr/bin/env python3
"""
Rule-based parameter sweep for rule_backtest.py.

Runs a grid over RSI-rule parameters (entry/exit thresholds, trend lookback, SMA period)
and execution parameters (SL/TP, max position fraction, min-hold, cooldown, fees).

Key features
- Fast: metrics are computed online (no heavy equity arrays stored).
- Parallel execution with ProcessPoolExecutor.
- Resume on by default: if a JSON for a config exists, it is skipped.
- Composite scoring copied from param_sweep.py style (Sharpe primary, DD penalty, trades bonus).

Usage
  python scripts/rule_sweep.py --split val --parallel 8 \
      --rsi_entry 20 25 30 35 --rsi_exit 65 70 75 80 \
      --sma_period 0 50 100 200 --trend_lookback 3 5 7 \
      --sl 0 0.005 0.01 0.02 --tp 0 0.01 0.02 0.03 \
      --max_pos 0.05 0.10 0.15 0.25 --min_hold 4 8 16 32 --cooldown 0 2 4

Outputs
- CSV summary (default results/rule_sweep.csv)
- Per-run JSON metrics in results/rule_sweep_runs/
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# allow imports from scripts/
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from indicators import add_indicators, normalize_features
from rule_backtest import FEATURES, StrategyParams, EnvParams, FastBacktester, RsiTrendStrategy


# ---------------------------- scoring (mirrors param_sweep.py)
def composite_score(sharpe: float, max_dd: float, n_trades: int,
                    dd_penalty: float = 0.5,
                    trade_bonus: float = 0.2,
                    min_trades_floor: int = 10) -> float:
    dd_term = dd_penalty * (max_dd / 100.0)
    trade_term = trade_bonus * math.log(max(n_trades, min_trades_floor))
    return sharpe - dd_term + trade_term


def fmt_tag(x) -> str:
    if isinstance(x, int):
        return str(x)
    s = f"{x}"
    return s.replace('-', 'm').replace('.', 'p')


def _fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# ---------------------------- data cache (per process)
_DATA_CACHE: Dict[Tuple[str, float, float, str], pd.DataFrame] = {}


def _load_eval_df(data_path: str, train_ratio: float, val_ratio: float, split: str) -> pd.DataFrame:
    """
    Load, add indicators, normalise (window=100), then return the requested split.
    Cached per-process for speed.
    """
    key = (data_path, train_ratio, val_ratio, split)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]

    df = pd.read_csv(data_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = add_indicators(df)
    df = normalize_features(df, FEATURES, window=100)

    split1 = int(len(df) * train_ratio)
    split2 = int(len(df) * (train_ratio + val_ratio))

    if split == 'val':
        if val_ratio <= 0:
            raise ValueError("val_ratio is 0, cannot evaluate on validation set")
        out = df.iloc[split1:split2].reset_index(drop=True)
    else:
        out = df.iloc[split2:].reset_index(drop=True) if val_ratio > 0 else df.iloc[split1:].reset_index(drop=True)

    _DATA_CACHE[key] = out
    return out


def run_one(job: dict) -> dict:
    """
    Worker entry-point. Returns a dict suitable for CSV row storage.
    Also writes a JSON metrics file for the run.
    """
    t0 = time.time()

    df_eval = _load_eval_df(job['data'], job['train_ratio'], job['val_ratio'], job['split'])

    strat = StrategyParams(
        rsi_entry=job['rsi_entry'],
        rsi_exit=job['rsi_exit'],
        trend_lookback=job['trend_lookback'],
        slope_threshold=job['slope_threshold'],
        sma_period=job['sma_period'],
        allow_short=job['allow_short'],
    )

    env = EnvParams(
        initial_balance=job['initial_balance'],
        leverage=job['leverage'],
        fee_rate=job['fee'],
        max_position_frac=job['max_pos'],
        stop_loss_pct=job['sl'],
        take_profit_pct=job['tp'],
        min_hold_steps=job['min_hold'],
        cooldown_steps=job['cooldown'],
    )

    # fast engine: no trade list, no equity curve list (online metrics only)
    bt = FastBacktester(df_eval, env, capture_equity=False, capture_trades=False)
    st = RsiTrendStrategy(bt.df, strat)
    st.reset()

    done = False
    while not done:
        a = st.decide(bt)
        done = bt.step(a)

    m = bt.get_metrics()

    run_s = time.time() - t0

    score = composite_score(
        float(m.get('sharpe_ratio', 0.0)),
        float(m.get('max_drawdown', 0.0)),
        int(m.get('total_trades', 0)),
        dd_penalty=job['dd_penalty'],
        trade_bonus=job['trade_bonus'],
    )

    passes = (float(m.get('max_drawdown', 0.0)) <= job['max_dd']) and (int(m.get('total_trades', 0)) >= job['min_trades'])
    out_json = os.path.join(job['out_dir'], f"{job['config_key']}.json")
    payload = {
        'algorithm': 'RSI Rule',
        'split': job['split'],
        'params': {
            'rsi_entry': job['rsi_entry'],
            'rsi_exit': job['rsi_exit'],
            'trend_lookback': job['trend_lookback'],
            'slope_threshold': job['slope_threshold'],
            'sma_period': job['sma_period'],
            'allow_short': job['allow_short'],
            'fee_rate': job['fee'],
            'max_position_frac': job['max_pos'],
            'stop_loss_pct': job['sl'],
            'take_profit_pct': job['tp'],
            'min_hold_steps': job['min_hold'],
            'cooldown_steps': job['cooldown'],
            'leverage': job['leverage'],
            'initial_balance': job['initial_balance'],
        },
        'metrics': m,
        'composite_score': round(float(score), 6),
        'passes_constraints': int(passes),
        'run_seconds': round(float(run_s), 4),
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }

    if not job.get('no_json', False):
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=str)
        json_path = out_json
    else:
        json_path = ''

    return {
        'timestamp': payload['timestamp'],
        'config_key': job['config_key'],
        'split': job['split'],
        'rsi_entry': job['rsi_entry'],
        'rsi_exit': job['rsi_exit'],
        'trend_lookback': job['trend_lookback'],
        'slope_threshold': job['slope_threshold'],
        'sma_period': job['sma_period'],
        'allow_short': int(job['allow_short']),
        'sl': job['sl'],
        'tp': job['tp'],
        'max_pos': job['max_pos'],
        'min_hold': job['min_hold'],
        'cooldown': job['cooldown'],
        'fee': job['fee'],
        'val_return': m.get('total_return', 0.0),
        'val_sharpe': m.get('sharpe_ratio', 0.0),
        'val_max_dd': m.get('max_drawdown', 0.0),
        'val_trades': m.get('total_trades', 0),
        'val_win_rate': m.get('win_rate', 0.0),
        'sl_hits': m.get('sl_hits', 0),
        'tp_hits': m.get('tp_hits', 0),
        'composite_score': round(float(score), 6),
        'passes_constraints': int(passes),
        'run_seconds': round(float(run_s), 4),
        'json_path': json_path,
    }


FIELDNAMES = [
    'timestamp',
    'config_key',
    'split',
    'rsi_entry', 'rsi_exit', 'trend_lookback', 'slope_threshold', 'sma_period', 'allow_short',
    'sl', 'tp', 'max_pos', 'min_hold', 'cooldown', 'fee',
    'val_return', 'val_sharpe', 'val_max_dd', 'val_trades', 'val_win_rate',
    'sl_hits', 'tp_hits',
    'composite_score', 'passes_constraints', 'run_seconds',
    'json_path',
]


def _load_existing_csv(path: str) -> Tuple[List[dict], Dict[str, int], set]:
    rows: List[dict] = []
    by_key: Dict[str, int] = {}
    completed: set = set()

    if not os.path.exists(path):
        return rows, by_key, completed

    with open(path, 'r', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            # normalise row keys
            row = {k: row.get(k, '') for k in FIELDNAMES}
            ck = row.get('config_key', '')
            if ck:
                completed.add(ck)
                by_key[ck] = len(rows)
            rows.append(row)

    return rows, by_key, completed


def _write_csv(path: str, rows: List[dict]):
    """Atomic write: write to temp file, then rename. Prevents data loss on crash."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in FIELDNAMES} for r in rows])
    # Atomic rename — old file is only replaced once new file is complete
    if os.name == 'nt':
        # Windows: can't rename over existing file
        if os.path.exists(path):
            os.remove(path)
    os.rename(tmp, path)


def _append_csv(path: str, row: dict):
    """Append a single row. Creates file with header if it doesn't exist."""
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in FIELDNAMES})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join('data', 'SOL_USDT_15m.csv'))
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--train_ratio', type=float, default=0.6)
    ap.add_argument('--val_ratio', type=float, default=0.2)

    ap.add_argument('--out_csv', default=os.path.join('results', 'rule_sweep.csv'))
    ap.add_argument('--out_dir', default=os.path.join('results', 'rule_sweep_runs'))
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--no_json', action='store_true', help='Do not write per-run JSON files, write CSV only')

    # Strategy grids
    ap.add_argument('--rsi_entry', nargs='+', type=float, default=[20, 25, 30, 35])
    ap.add_argument('--rsi_exit', nargs='+', type=float, default=[65, 70, 75, 80])
    ap.add_argument('--trend_lookback', nargs='+', type=int, default=[3, 5, 7])
    ap.add_argument('--sma_period', nargs='+', type=int, default=[0, 50, 100, 200])
    ap.add_argument('--slope_threshold', type=float, default=0.0)
    ap.add_argument('--allow_short', action='store_true')

    # Execution grids
    ap.add_argument('--sl', nargs='+', type=float, default=[0.0, 0.005, 0.01, 0.02, 0.03])
    ap.add_argument('--tp', nargs='+', type=float, default=[0.0, 0.01, 0.02, 0.03, 0.05])
    ap.add_argument('--max_pos', nargs='+', type=float, default=[0.05, 0.10, 0.15, 0.25])
    ap.add_argument('--min_hold', nargs='+', type=int, default=[4, 8, 16, 32])
    ap.add_argument('--cooldown', nargs='+', type=int, default=[0, 2, 4])

    ap.add_argument('--fee', type=float, default=0.0004)
    ap.add_argument('--initial_balance', type=float, default=10_000.0)
    ap.add_argument('--leverage', type=int, default=1)

    # Constraints / scoring
    ap.add_argument('--max_dd', type=float, default=60.0)
    ap.add_argument('--min_trades', type=int, default=10)
    ap.add_argument('--dd_penalty', type=float, default=0.5)
    ap.add_argument('--trade_bonus', type=float, default=0.2)

    ap.add_argument('--no_resume', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows, by_key, completed = _load_existing_csv(args.out_csv)
    resume = not args.no_resume

    if resume:
        print(f"Resume: looking for existing CSV at: {os.path.abspath(args.out_csv)}")
        if completed:
            print(f"Resume: loaded {len(completed)} completed config_keys from CSV")
        else:
            print(f"Resume: CSV {'not found' if not os.path.exists(args.out_csv) else 'found but empty'}")


    # Build grid
    grid = list(itertools.product(
        args.rsi_entry,
        args.rsi_exit,
        args.trend_lookback,
        args.sma_period,
        args.sl,
        args.tp,
        args.max_pos,
        args.min_hold,
        args.cooldown,
    ))

    jobs = []
    skipped = 0
    for (re, rx, lb, sma, sl, tp, mp, mh, cd) in grid:
        if lb < 2:
            continue

        config_key = (
            f"re{fmt_tag(re)}_rx{fmt_tag(rx)}_lb{lb}_sma{sma}_"
            f"sl{fmt_tag(sl)}_tp{fmt_tag(tp)}_mp{fmt_tag(mp)}_"
            f"mh{mh}_cd{cd}_sh{int(args.allow_short)}"
        )

        out_json = os.path.join(args.out_dir, f"{config_key}.json")
        if resume and (config_key in completed) and (args.no_json or os.path.exists(out_json)):
            skipped += 1
            continue

        jobs.append({
            'data': args.data,
            'split': args.split,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'out_dir': args.out_dir,
            'config_key': config_key,

            'rsi_entry': float(re),
            'rsi_exit': float(rx),
            'trend_lookback': int(lb),
            'sma_period': int(sma),
            'slope_threshold': float(args.slope_threshold),
            'allow_short': bool(args.allow_short),

            'sl': float(sl),
            'tp': float(tp),
            'max_pos': float(mp),
            'min_hold': int(mh),
            'cooldown': int(cd),
            'fee': float(args.fee),
            'initial_balance': float(args.initial_balance),
            'leverage': int(args.leverage),

            'max_dd': float(args.max_dd),
            'min_trades': int(args.min_trades),
            'dd_penalty': float(args.dd_penalty),
            'trade_bonus': float(args.trade_bonus),
            'no_json': bool(args.no_json),
        })

    total_runs = len(jobs)
    if total_runs == 0 and skipped > 0:
        print(f"All {skipped} configs already completed. Nothing to do.")
        return
    elif total_runs == 0:
        print("Nothing to do (grid is empty).")
        return

    if skipped:
        print(f"Skipped (already done): {skipped}")

    print(f"Planned runs: {total_runs} (resume={'ON' if resume else 'OFF'})")
    print(f"Split: {args.split}  fee={args.fee}  allow_short={int(args.allow_short)}")
    print(f"Parallel workers: {max(1, args.parallel)}")

    t_all = time.time()
    completed_now = 0
    _SAVE_EVERY = 500  # full rewrite every N runs as safety checkpoint

    def record(result: dict):
        nonlocal completed_now
        completed_now += 1
        ck = result.get('config_key', '')
        if ck and ck in by_key:
            rows[by_key[ck]] = result
        else:
            by_key[ck] = len(rows)
            rows.append(result)

        # Append single row (fast, O(1))
        _append_csv(args.out_csv, result)

        # Full rewrite every N runs as safety checkpoint
        if completed_now % _SAVE_EVERY == 0:
            _write_csv(args.out_csv, rows)

        shp = float(result.get('val_sharpe', 0))
        dd = float(result.get('val_max_dd', 0))
        tr = int(float(result.get('val_trades', 0)))
        sc = float(result.get('composite_score', 0))
        run_s = float(result.get('run_seconds', 0))

        elapsed = time.time() - t_all
        eta = (elapsed / completed_now) * (total_runs - completed_now) if completed_now else 0.0
        # Print every 10 runs to reduce console spam on large sweeps
        if completed_now <= 20 or completed_now % 100 == 0 or completed_now == total_runs:
            print(f"[{completed_now}/{total_runs}] sharpe={shp:+.3f} score={sc:.3f} trades={tr} "
                  f"dd={dd:.1f}% run={run_s:.2f}s ETA={_fmt_time(eta)}")

    # Run
    if args.parallel <= 1:
        for job in jobs:
            try:
                result = run_one(job)
                record(result)
            except KeyboardInterrupt:
                print(f"\nInterrupted! Saving {len(rows)} rows...")
                _write_csv(args.out_csv, rows)
                print(f"Saved to {args.out_csv}")
                return
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(run_one, job): job['config_key'] for job in jobs}
            try:
                for fut in as_completed(futs):
                    try:
                        record(fut.result())
                    except Exception as e:
                        print(f"FAILED: {e}")
            except KeyboardInterrupt:
                print(f"\nInterrupted! Cancelling workers and saving {len(rows)} rows...")
                for f in futs:
                    f.cancel()
                _write_csv(args.out_csv, rows)
                print(f"Saved to {args.out_csv}")
                return

    # Final full write to clean up any append duplicates
    _write_csv(args.out_csv, rows)

    # Summary
    ok = [r for r in rows if str(r.get('passes_constraints', '0')) in ('1', 'True', 'true')]
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE - {len(rows)} total rows in CSV")
    print(f"Passing constraints (dd<={args.max_dd} and trades>={args.min_trades}): {len(ok)}/{len(rows)}")
    print(f"{'='*70}")

    if ok:
        def _key(r):
            try:
                return float(r.get('composite_score', 0))
            except Exception:
                return 0.0

        top = sorted(ok, key=_key, reverse=True)[:10]
        print("\nTOP 10 by composite score:")
        print(f"{'Rank':<5} {'Score':>8} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} {'Trades':>7} {'Config':<}")
        for i, r in enumerate(top, 1):
            print(f"{i:<5} {float(r['composite_score']):>8.3f} {float(r['val_sharpe']):>+8.3f} "
                  f"{float(r['val_return']):>+9.2f} {float(r['val_max_dd']):>8.2f} {int(float(r['val_trades'])):>7} "
                  f"{r['config_key']}")

        best = top[0]['config_key']
        print(f"\nBest config_key: {best}")
        print("To export a full trade list + equity curve for it, run:")
        print(f"  python scripts/rule_backtest.py --split {args.split} "
              f"--rsi_entry <...> --rsi_exit <...> --trend_lookback <...> --sma_period <...> "
              f"--sl <...> --tp <...> --max_pos <...> --min_hold <...> --cooldown <...>")


if __name__ == '__main__':
    main()