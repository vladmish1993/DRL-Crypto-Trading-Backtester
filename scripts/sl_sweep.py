#!/usr/bin/env python3
"""
Stop-Loss / Take-Profit sweep.

Loads trained models and runs each through a grid of SL levels on the
test set.  Outputs a JSON matrix for the dashboard heatmap + line charts.

Usage
-----
    python scripts/sl_sweep.py                              # SL only, 0.5-5%
    python scripts/sl_sweep.py --tp 1.0 2.0 3.0 4.0 5.0    # SL × TP grid
    python scripts/sl_sweep.py --episodes 80 --retrain       # retrain first
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from indicators import add_indicators, normalize_features
from trading_env import CryptoFuturesEnv
from models import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, A2CAgent

FEATURES = [
    'close_norm', 'open_norm', 'high_norm', 'low_norm',
    'sma_20_norm', 'sma_50_norm',
    'rsi_norm', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
    'bb_width_norm', 'atr_norm', 'volume_ratio_norm', 'returns',
]

SL_LEVELS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050]
SL_LABELS = ['0.5%', '1.0%', '1.5%', '2.0%', '2.5%', '3.0%', '3.5%', '4.0%', '4.5%', '5.0%']


def load_data(csv_path, train_ratio=0.7):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = add_indicators(df)
    df = normalize_features(df, FEATURES, window=100)
    split = int(len(df) * train_ratio)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


def backtest_with_sl(agent, test_df, sl_pct, tp_pct=0.0):
    """Run a single backtest with given SL/TP levels."""
    env = CryptoFuturesEnv(
        test_df, FEATURES,
        stop_loss_pct=sl_pct,
        take_profit_pct=tp_pct,
    )
    s, _ = env.reset()
    done = False
    while not done:
        a = agent.select_action(s, training=False)
        s, _, done, _, _ = env.step(a)
    return env.get_metrics()


def run_sweep(agents, test_df, sl_levels, tp_levels=None):
    """
    Run the full sweep grid.

    Returns
    -------
    results : dict
        {
          "sl_levels": ["0.5%", ...],
          "tp_levels": ["None"] or ["1.0%", ...],
          "no_sl_baseline": { "DQN": {...}, ... },
          "grid": {
            "DQN": {
              "0.5%": { "None": { metrics }, "1.0%": { metrics }, ... },
              ...
            },
            ...
          }
        }
    """
    tp_levels = tp_levels or [0.0]
    tp_labels = ['None'] if tp_levels == [0.0] else [f'{t*100:.1f}%' for t in tp_levels]

    results = {
        'sl_levels': SL_LABELS[:len(sl_levels)],
        'tp_levels': tp_labels,
        'no_sl_baseline': {},
        'grid': {},
    }

    # ── Baseline: no SL/TP ──
    print("Running baselines (no SL/TP) ...")
    for ag in agents:
        m = backtest_with_sl(ag, test_df, 0.0, 0.0)
        m['algorithm'] = ag.name
        results['no_sl_baseline'][ag.name] = m
        print(f"  {ag.name:<15}  return={m['total_return']:>+8.2f}%  sharpe={m['sharpe_ratio']:>6.2f}")

    # ── Grid sweep ──
    total_runs = len(agents) * len(sl_levels) * len(tp_levels)
    run = 0
    for ag in agents:
        results['grid'][ag.name] = {}
        for sl, sl_label in zip(sl_levels, SL_LABELS):
            results['grid'][ag.name][sl_label] = {}
            for tp, tp_label in zip(tp_levels, tp_labels):
                run += 1
                m = backtest_with_sl(ag, test_df, sl, tp)
                m['algorithm'] = ag.name
                m['sl_pct'] = sl_label
                m['tp_pct'] = tp_label
                results['grid'][ag.name][sl_label][tp_label] = m

                pct = run / total_runs * 100
                print(f"  [{pct:5.1f}%]  {ag.name:<15}  SL={sl_label}  TP={tp_label}  "
                      f"return={m['total_return']:>+8.2f}%  sharpe={m['sharpe_ratio']:>6.2f}  "
                      f"SL_hits={m['sl_hits']}  TP_hits={m['tp_hits']}")

    # ── Best combos per algorithm ──
    results['best_per_algo'] = {}
    for ag_name in results['grid']:
        best_sharpe = -999
        best_combo = {}
        for sl_label in results['grid'][ag_name]:
            for tp_label in results['grid'][ag_name][sl_label]:
                m = results['grid'][ag_name][sl_label][tp_label]
                if m['sharpe_ratio'] > best_sharpe:
                    best_sharpe = m['sharpe_ratio']
                    best_combo = m.copy()
        results['best_per_algo'][ag_name] = best_combo

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/SOL_USDT_15m.csv')
    ap.add_argument('--output', default='results/sl_sweep_results.json')
    ap.add_argument('--tp', nargs='*', type=float, default=None,
                    help='TP levels as decimals, e.g. --tp 0.01 0.02 0.03')
    args = ap.parse_args()

    _, test_df = load_data(args.data)
    state_dim = len(FEATURES) + 3
    action_dim = 4

    hp = dict(lr=1e-4, gamma=0.99,
              eps_start=0.05, eps_end=0.05, eps_decay=1.0,  # no exploration for eval
              buffer_size=50_000, batch_size=64, target_update=1000)

    agents = [
        DQNAgent(state_dim, action_dim, **hp),
        DoubleDQNAgent(state_dim, action_dim, **hp),
        DuelingDQNAgent(state_dim, action_dim, **hp),
        A2CAgent(state_dim, action_dim, lr=3e-4, gamma=0.99),
    ]

    # Load trained weights
    for ag in agents:
        path = f"models/{ag.name.lower().replace(' ', '_')}.pt"
        if os.path.exists(path):
            ag.load(path)
            print(f"Loaded {ag.name} ← {path}")
        else:
            print(f"WARNING: {path} not found — using random weights!")

    tp_levels = args.tp if args.tp else [0.0]

    print(f"\n{'═'*60}")
    print(f"  STOP-LOSS SWEEP")
    print(f"  SL: {SL_LABELS}")
    if args.tp:
        print(f"  TP: {[f'{t*100:.1f}%' for t in tp_levels]}")
    print(f"  Total runs: {len(agents) * len(SL_LEVELS) * len(tp_levels)}")
    print(f"{'═'*60}\n")

    t0 = time.time()
    results = run_sweep(agents, test_df, SL_LEVELS, tp_levels)
    elapsed = time.time() - t0

    print(f"\n{'═'*60}")
    print(f"  BEST CONFIGURATIONS (by Sharpe)")
    print(f"{'═'*60}")
    for name, m in results['best_per_algo'].items():
        print(f"  {name:<15}  SL={m.get('sl_pct','?')}  TP={m.get('tp_pct','?')}  "
              f"sharpe={m['sharpe_ratio']:>6.2f}  return={m['total_return']:>+8.2f}%")
    print(f"\n  Completed in {elapsed:.1f}s")

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results → {args.output}")

    pub = os.path.join('public', 'sl_sweep_results.json')
    os.makedirs('public', exist_ok=True)
    with open(pub, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results → {pub}")


if __name__ == '__main__':
    main()
