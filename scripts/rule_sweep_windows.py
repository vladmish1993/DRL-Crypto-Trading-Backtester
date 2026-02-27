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
import hashlib
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



def _eval_on_df(df_eval: pd.DataFrame, strat: StrategyParams, env: EnvParams) -> dict:
    bt = FastBacktester(df_eval, env, capture_equity=False, capture_trades=False)
    st = RsiTrendStrategy(bt.df, strat)
    st.reset()
    done = False
    while not done:
        a = st.decide(bt)
        done = bt.step(a)
    return bt.get_metrics()


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
        adx_threshold=job.get('adx_threshold', 0.0),
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
    # Evaluate
    if job.get('eval_mode', 'full') == 'random_windows':
        win = int(job.get('window', 2000))
        n_w = int(job.get('n_windows', 50))
        warmup = int(job.get('warmup', 0))
        if win <= 10:
            raise ValueError('window must be > 10')
        if warmup < 0 or warmup >= win - 5:
            raise ValueError('warmup must be in [0, window-5]')

        # deterministic RNG per config so parallel execution stays reproducible
        base_seed = int(job.get('seed', 123))
        h = hashlib.md5(job['config_key'].encode('utf-8')).hexdigest()
        cfg_seed = base_seed ^ int(h[:8], 16)
        rng = np.random.default_rng(cfg_seed)

        if len(df_eval) <= win + 1:
            # fallback to full run if split too small
            m = _eval_on_df(df_eval, strat, env)
            job['_agg'] = {'sharpe_median': m.get('sharpe_ratio', 0.0), 'sharpe_p10': m.get('sharpe_ratio', 0.0),
                           'max_dd_max': m.get('max_drawdown', 0.0), 'trades_mean': m.get('total_trades', 0)}
        else:
            max_start = len(df_eval) - win - 1
            starts = rng.integers(0, max_start + 1, size=n_w)

            sharpes = []
            rets = []
            dds = []
            trades = []
            win_rates = []

            for s in starts:
                wdf = df_eval.iloc[int(s):int(s)+win].reset_index(drop=True)
                if warmup > 0 and len(wdf) > warmup + 10:
                    # run on the full window but ignore first warmup bars for scoring by slicing
                    wdf_scored = wdf.iloc[warmup:].reset_index(drop=True)
                else:
                    wdf_scored = wdf

                mm = _eval_on_df(wdf_scored, strat, env)
                sharpes.append(float(mm.get('sharpe_ratio', 0.0)))
                rets.append(float(mm.get('total_return', 0.0)))
                dds.append(float(mm.get('max_drawdown', 0.0)))
                trades.append(float(mm.get('total_trades', 0.0)))
                win_rates.append(float(mm.get('win_rate', 0.0)))

            # Aggregate. We store mean in the main metric fields for ranking,
            # plus median / p10 / worst dd for robustness inspection.
            m = {
                'total_return': float(np.mean(rets)) if rets else 0.0,
                'sharpe_ratio': float(np.mean(sharpes)) if sharpes else 0.0,
                'max_drawdown': float(np.mean(dds)) if dds else 0.0,
                'win_rate': float(np.mean(win_rates)) if win_rates else 0.0,
                'total_trades': int(round(float(np.mean(trades)))) if trades else 0,
                'avg_trade_pnl': 0.0,
                'final_balance': 0.0,
                'sl_hits': 0,
                'tp_hits': 0,
            }

            job['_agg'] = {
                'sharpe_median': float(np.median(sharpes)) if sharpes else m['sharpe_ratio'],
                'sharpe_p10': float(np.percentile(sharpes, 10)) if sharpes else m['sharpe_ratio'],
                'max_dd_max': float(np.max(dds)) if dds else m['max_drawdown'],
                'trades_mean': float(np.mean(trades)) if trades else float(m['total_trades']),
            }
    else:
        m = _eval_on_df(df_eval, strat, env)
        job['_agg'] = {'sharpe_median': m.get('sharpe_ratio', 0.0), 'sharpe_p10': m.get('sharpe_ratio', 0.0),
                       'max_dd_max': m.get('max_drawdown', 0.0), 'trades_mean': m.get('total_trades', 0)}

    run_s = time.time() - t0
    # --- scoring
    if job.get('eval_mode', 'full') == 'random_windows':
        agg = job.get('_agg', {})
        # Tail-robust Sharpe: blend median + p10 so we optimise for typical AND bad windows.
        s_med = float(agg.get('sharpe_median', float(m.get('sharpe_ratio', 0.0))))
        s_p10 = float(agg.get('sharpe_p10', s_med))
        sharpe_for_score = 0.6 * s_med + 0.4 * s_p10

        max_dd_for_score = float(agg.get('max_dd_max', float(m.get('max_drawdown', 0.0))))
        trades_for_score = float(agg.get('trades_mean', float(m.get('total_trades', 0))))

        score = composite_score(
            float(sharpe_for_score),
            float(max_dd_for_score),
            int(round(trades_for_score)),
            dd_penalty=job['dd_penalty'],
            trade_bonus=job['trade_bonus'],
            min_trades_floor=1,
        )
    else:
        score = composite_score(
            float(m.get('sharpe_ratio', 0.0)),
            float(m.get('max_drawdown', 0.0)),
            int(m.get('total_trades', 0)),
            dd_penalty=job['dd_penalty'],
            trade_bonus=job['trade_bonus'],
        )

    dd_for_pass = job.get('_agg', {}).get('max_dd_max', float(m.get('max_drawdown', 0.0))) if job.get('eval_mode','full')=='random_windows' else float(m.get('max_drawdown', 0.0))
    trades_for_pass = job.get('_agg', {}).get('trades_mean', float(m.get('total_trades', 0))) if job.get('eval_mode','full')=='random_windows' else float(m.get('total_trades', 0))
    passes = (float(dd_for_pass) <= job['max_dd']) and (float(trades_for_pass) >= job['min_trades'])

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
        'robust_metrics': job.get('_agg', {}),
        'composite_score': round(float(score), 6),
        'passes_constraints': int(passes),
        'run_seconds': round(float(run_s), 4),
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }

    if not job.get('no_json', False):
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=str)
    else:
        out_json = ''

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
        'adx_threshold': job.get('adx_threshold', 0.0),
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
        'json_path': out_json,
        'val_sharpe_median': round(float(job.get('_agg', {}).get('sharpe_median', 0.0)), 4),
        'val_sharpe_p10': round(float(job.get('_agg', {}).get('sharpe_p10', 0.0)), 4),
        'val_max_dd_max': round(float(job.get('_agg', {}).get('max_dd_max', 0.0)), 2),
        'val_trades_mean': round(float(job.get('_agg', {}).get('trades_mean', 0.0)), 1),
    }


FIELDNAMES = [
    'timestamp',
    'config_key',
    'split',
    'rsi_entry', 'rsi_exit', 'trend_lookback', 'slope_threshold', 'sma_period', 'allow_short', 'adx_threshold',
    'sl', 'tp', 'max_pos', 'min_hold', 'cooldown', 'fee',
    'val_return', 'val_sharpe', 'val_sharpe_median', 'val_sharpe_p10', 'val_max_dd', 'val_max_dd_max', 'val_trades', 'val_trades_mean', 'val_win_rate',
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
    if os.name == 'nt':
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
    ap.add_argument('--eval_mode', choices=['full', 'random_windows'], default='full', help='Evaluate on full split or on many random windows')
    ap.add_argument('--window', type=int, default=2000, help='Window length in bars for random_windows mode')
    ap.add_argument('--n_windows', type=int, default=50, help='How many random windows per config in random_windows mode')
    ap.add_argument('--seed', type=int, default=123, help='Base seed for window sampling (deterministic)')
    ap.add_argument('--warmup', type=int, default=0, help='Optional warmup bars inside each sampled window that are ignored for scoring')

    ap.add_argument('--from_csv', default='', help='Optional: load candidate configs from an existing rule_sweep CSV (Stage A).')
    ap.add_argument('--top_k', type=int, default=0, help='When using --from_csv, evaluate only the top K rows (by composite_score). 0 = all.')

    # Strategy grids
    ap.add_argument('--rsi_entry', nargs='+', type=float, default=[20, 25, 30, 35])
    ap.add_argument('--rsi_exit', nargs='+', type=float, default=[65, 70, 75, 80])
    ap.add_argument('--trend_lookback', nargs='+', type=int, default=[3, 5, 7])
    ap.add_argument('--sma_period', nargs='+', type=int, default=[0, 50, 100, 200])
    ap.add_argument('--slope_threshold', type=float, default=0.0)
    ap.add_argument('--allow_short', action='store_true')
    ap.add_argument('--adx_threshold', nargs='+', type=float, default=[0.0],
                    help='Entry-only regime filter: only take entries when ADX <= threshold. 0 disables.')

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
    jobs = []
    skipped = 0

    if args.from_csv:
        import pandas as _pd
        cand = _pd.read_csv(args.from_csv)
        if 'passes_constraints' in cand.columns:
            cand = cand[cand['passes_constraints'] == 1]
        if 'composite_score' in cand.columns:
            cand = cand.sort_values('composite_score', ascending=False)
        if args.top_k and args.top_k > 0:
            cand = cand.head(int(args.top_k))

        for _, row in cand.iterrows():
            for adx_thr in args.adx_threshold:
                base_key = str(row.get('config_key', '')).strip()
                if not base_key:
                    continue
                # Keep base key when adx_thr==0 so you can compare directly to old runs / resume.
                config_key = base_key if float(adx_thr) <= 0 else (base_key + f"_adx{fmt_tag(float(adx_thr))}")

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

                    'rsi_entry': float(row.get('rsi_entry', 30.0)),
                    'rsi_exit': float(row.get('rsi_exit', 70.0)),
                    'trend_lookback': int(row.get('trend_lookback', 5)),
                    'sma_period': int(row.get('sma_period', 50)),
                    'slope_threshold': float(row.get('slope_threshold', args.slope_threshold)),
                    'allow_short': bool(int(row.get('allow_short', 0))),
                    'adx_threshold': float(adx_thr),

                    'sl': float(row.get('sl', 0.0)),
                    'tp': float(row.get('tp', 0.0)),
                    'max_pos': float(row.get('max_pos', 0.10)),
                    'min_hold': int(row.get('min_hold', 16)),
                    'cooldown': int(row.get('cooldown', 0)),

                    'fee': float(row.get('fee', args.fee)),
                    'initial_balance': float(args.initial_balance),
                    'leverage': int(args.leverage),

                    'max_dd': float(args.max_dd),
                    'min_trades': int(args.min_trades),
                    'dd_penalty': float(args.dd_penalty),
                    'trade_bonus': float(args.trade_bonus),
                    'no_json': bool(args.no_json),
                    'eval_mode': str(args.eval_mode),
                    'window': int(args.window),
                    'n_windows': int(args.n_windows),
                    'seed': int(args.seed),
                    'warmup': int(args.warmup),
                })

    else:
        grid = list(itertools.product(
            args.rsi_entry,
            args.rsi_exit,
            args.trend_lookback,
            args.sma_period,
            args.adx_threshold,
            args.sl,
            args.tp,
            args.max_pos,
            args.min_hold,
            args.cooldown,
        ))

        for (re, rx, lb, sma, adx_thr, sl, tp, mp, mh, cd) in grid:
            if lb < 2:
                continue

            base_key = (
                f"re{fmt_tag(re)}_rx{fmt_tag(rx)}_lb{lb}_sma{sma}_"
                f"sl{fmt_tag(sl)}_tp{fmt_tag(tp)}_mp{fmt_tag(mp)}_"
                f"mh{mh}_cd{cd}_sh{int(args.allow_short)}"
            )
            config_key = base_key if float(adx_thr) <= 0 else (base_key + f"_adx{fmt_tag(float(adx_thr))}")

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
                'adx_threshold': float(adx_thr),

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
                'eval_mode': str(args.eval_mode),
                'window': int(args.window),
                'n_windows': int(args.n_windows),
                'seed': int(args.seed),
                'warmup': int(args.warmup),
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
    print(f"Split: {args.split}  fee={args.fee}  allow_short={int(args.allow_short)}  eval_mode={args.eval_mode}")
    print(f"Parallel workers: {max(1, args.parallel)}")

    t_all = time.time()
    completed_now = 0
    _SAVE_EVERY = 500

    def record(result: dict):
        nonlocal completed_now
        completed_now += 1
        ck = result.get('config_key', '')
        if ck and ck in by_key:
            rows[by_key[ck]] = result
        else:
            by_key[ck] = len(rows)
            rows.append(result)

        _append_csv(args.out_csv, result)

        if completed_now % _SAVE_EVERY == 0:
            _write_csv(args.out_csv, rows)

        shp = float(result.get('val_sharpe', 0))
        dd = float(result.get('val_max_dd', 0))
        tr = int(float(result.get('val_trades', 0)))
        sc = float(result.get('composite_score', 0))
        run_s = float(result.get('run_seconds', 0))

        elapsed = time.time() - t_all
        eta = (elapsed / completed_now) * (total_runs - completed_now) if completed_now else 0.0
        if completed_now <= 20 or completed_now % max(1, total_runs // 200) == 0 or completed_now == total_runs:
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