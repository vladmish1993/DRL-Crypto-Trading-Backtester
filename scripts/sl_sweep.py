#!/usr/bin/env python3
"""
Stop-Loss / Take-Profit sweep.

Loads trained models and runs each through a grid of SL levels on a chosen
evaluation split (validation by default). Outputs a JSON matrix for the dashboard
heatmap and line charts.

Important:
- Use this on validation while you are tuning.
- Only run on test once, after you have frozen the parameters.

Usage
-----
    python scripts/sl_sweep.py --split val --model_tag my_tag
    python scripts/sl_sweep.py --split val --model_tag my_tag --tp 0.01 0.02 0.03
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


def load_data(csv_path, train_ratio=0.6, val_ratio=0.2):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = add_indicators(df)
    df = normalize_features(df, FEATURES, window=100)

    split1 = int(len(df) * train_ratio)
    split2 = int(len(df) * (train_ratio + val_ratio))
    train = df.iloc[:split1].reset_index(drop=True)
    val   = df.iloc[split1:split2].reset_index(drop=True) if val_ratio > 0 else None
    test  = df.iloc[split2:].reset_index(drop=True) if val_ratio > 0 else df.iloc[split1:].reset_index(drop=True)
    return train, val, test


def backtest_with_sl(agent, df_eval, sl_pct, tp_pct=0.0, env_base_kwargs=None):
    """Run a single backtest with given SL/TP levels."""
    kw = dict(env_base_kwargs or {})
    kw.update(dict(stop_loss_pct=sl_pct, take_profit_pct=tp_pct))

    env = CryptoFuturesEnv(df_eval, FEATURES, **kw)
    s, _ = env.reset()
    done = False
    while not done:
        a = agent.select_action(s, training=False)
        s, _, done, _, _ = env.step(a)
    return env.get_metrics()


def run_sweep(agents, df_eval, sl_levels, tp_levels=None, env_base_kwargs=None):
    tp_levels = tp_levels or [0.0]
    tp_labels = ['None'] if tp_levels == [0.0] else [f'{t*100:.1f}%' for t in tp_levels]

    results = {
        'sl_levels': SL_LABELS[:len(sl_levels)],
        'tp_levels': tp_labels,
        'no_sl_baseline': {},
        'grid': {},
    }

    # Baseline: no SL/TP
    print("Running baselines (no SL/TP) ...")
    for ag in agents:
        m = backtest_with_sl(ag, df_eval, 0.0, 0.0, env_base_kwargs=env_base_kwargs)
        m['algorithm'] = ag.name
        results['no_sl_baseline'][ag.name] = m
        print(f"  {ag.name:<15}  return={m['total_return']:>+8.2f}%  sharpe={m['sharpe_ratio']:>6.2f}")

    # Grid sweep
    total_runs = len(agents) * len(sl_levels) * len(tp_levels)
    run = 0
    for ag in agents:
        results['grid'][ag.name] = {}
        for sl, sl_label in zip(sl_levels, SL_LABELS):
            results['grid'][ag.name][sl_label] = {}
            for tp, tp_label in zip(tp_levels, tp_labels):
                run += 1
                m = backtest_with_sl(ag, df_eval, sl, tp, env_base_kwargs=env_base_kwargs)
                m['algorithm'] = ag.name
                m['sl_pct'] = sl_label
                m['tp_pct'] = tp_label
                results['grid'][ag.name][sl_label][tp_label] = m

                pct = run / total_runs * 100
                print(f"  [{pct:5.1f}%]  {ag.name:<15}  SL={sl_label}  TP={tp_label}  "
                      f"return={m['total_return']:>+8.2f}%  sharpe={m['sharpe_ratio']:>6.2f}  "
                      f"SL_hits={m['sl_hits']}  TP_hits={m['tp_hits']}")

    # Best combos per algorithm (by Sharpe)
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
    ap.add_argument('--split', choices=['val','test'], default='val')
    ap.add_argument('--train_ratio', type=float, default=0.6)
    ap.add_argument('--val_ratio',   type=float, default=0.2)

    # Must match how the model was trained
    ap.add_argument('--fee',      type=float, default=0.0004)
    ap.add_argument('--max_pos',  type=float, default=0.2)
    ap.add_argument('--min_hold', type=int,   default=16)
    ap.add_argument('--cooldown', type=int,   default=4)
    ap.add_argument('--trade_penalty', type=float, default=0.0002)

    ap.add_argument('--model_tag', default='')
    ap.add_argument('--algo', choices=['all','dqn','double_dqn','dueling_dqn','a2c'], default='all')
    args = ap.parse_args()

    _, val_df, test_df = load_data(args.data, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    if args.split == 'val':
        if val_df is None:
            raise ValueError("val_ratio is 0, cannot sweep on validation set")
        df_eval = val_df
    else:
        df_eval = test_df

    state_dim = len(FEATURES) + 3
    action_dim = 4

    hp = dict(lr=1e-4, gamma=0.99,
              eps_start=0.05, eps_end=0.05, eps_decay=1.0,
              buffer_size=50_000, batch_size=64, target_update=1000)

    agents_all = [
        DQNAgent(state_dim, action_dim, **hp),
        DoubleDQNAgent(state_dim, action_dim, **hp),
        DuelingDQNAgent(state_dim, action_dim, **hp),
        A2CAgent(state_dim, action_dim, lr=3e-4, gamma=0.99),
    ]

    if args.algo == 'all':
        agents = agents_all
    elif args.algo == 'dqn':
        agents = [agents_all[0]]
    elif args.algo == 'double_dqn':
        agents = [agents_all[1]]
    elif args.algo == 'dueling_dqn':
        agents = [agents_all[2]]
    else:
        agents = [agents_all[3]]

    # Load trained weights
    if not args.model_tag.strip():
        raise ValueError("Pass --model_tag to select the trained weights to sweep")

    for ag in agents:
        path = os.path.join('models', f"{ag.name.lower().replace(' ', '_')}_{args.model_tag}.pt")
        if os.path.exists(path):
            ag.load(path)
            print(f"Loaded {ag.name} from {path}")
        else:
            raise FileNotFoundError(f"Model not found: {path}")

    tp_levels = args.tp if args.tp else [0.0]

    env_base_kwargs = dict(
        fee_rate=args.fee,
        max_position_frac=args.max_pos,
        min_hold_steps=args.min_hold,
        cooldown_steps=args.cooldown,
        trade_penalty=args.trade_penalty,
    )

    print(f"\n{'═'*60}")
    print(f"  STOP-LOSS SWEEP ({args.split.upper()})")
    print(f"  SL: {SL_LABELS}")
    if args.tp:
        print(f"  TP: {[f'{t*100:.1f}%' for t in tp_levels]}")
    print(f"  Total runs: {len(agents) * len(SL_LEVELS) * len(tp_levels)}")
    print(f"{'═'*60}\n")

    t0 = time.time()
    results = run_sweep(agents, df_eval, SL_LEVELS, tp_levels, env_base_kwargs=env_base_kwargs)
    elapsed = time.time() - t0

    print(f"\n{'═'*60}")
    print(f"  BEST CONFIGURATIONS (by Sharpe)")
    print(f"{'═'*60}")
    for name, m in results['best_per_algo'].items():
        print(f"  {name:<15}  SL={m.get('sl_pct','?')}  TP={m.get('tp_pct','?')}  "
              f"sharpe={m['sharpe_ratio']:>6.2f}  return={m['total_return']:>+8.2f}%")
    print(f"\n  Completed in {elapsed:.1f}s")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results -> {args.output}")


if __name__ == '__main__':
    main()
