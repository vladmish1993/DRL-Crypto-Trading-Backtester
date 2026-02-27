#!/usr/bin/env python3
"""plot_rl_trades.py

Load a trained RL agent, run it on a data split, and plot price with trade markers.

Reuses the same plotting function from plot_trades.py so the visuals are identical
to rule-based plots — only the decision engine changes (RL agent vs RSI rules).

Usage
-----
    # Full dataset, DQN agent
    python scripts/plot_rl_trades.py \
      --data data/SOL_USDT_15m.csv \
      --split val --train_ratio 0 --val_ratio 1.0 \
      --model_tag rl_sl002_tp007_mh16_cd4_seed42 \
      --algo dqn \
      --fee 0.0004 --max_pos 0.10 \
      --sl 0.02 --tp 0.07 \
      --min_hold 16 --cooldown 4 \
      --trade_penalty 0.0002 \
      --mode detailed --markers price

    # Test split (last 20%), all algos plotted separately
    python scripts/plot_rl_trades.py \
      --data data/SOL_USDT_15m.csv \
      --split test --train_ratio 0.8 --val_ratio 0.0 \
      --model_tag my_tag --algo all \
      --sl 0.02 --tp 0.07 --min_hold 32 --cooldown 4
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

# allow imports from scripts/
sys.path.insert(0, os.path.dirname(__file__))

from indicators import add_indicators, normalize_features
from trading_env import CryptoFuturesEnv
from models import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, A2CAgent

# Import the reusable plotting function + helpers from plot_trades.py
from plot_trades import plot_trades, _figsize_from_candles

# ── Feature columns (must match what the model was trained on) ────
FEATURES = [
    'close_norm', 'open_norm', 'high_norm', 'low_norm',
    'sma_20_norm', 'sma_50_norm',
    'rsi_norm', 'adx_norm',
    'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
    'bb_width_norm', 'atr_norm', 'volume_ratio_norm', 'returns',
]


def load_split(csv_path: str, split: str, train_ratio: float, val_ratio: float) -> pd.DataFrame:
    """Load data, compute indicators, split, return the requested slice."""
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = add_indicators(df)
    df = normalize_features(df, FEATURES, window=100)

    if split == 'full':
        return df.reset_index(drop=True)

    split1 = int(len(df) * train_ratio)
    split2 = int(len(df) * (train_ratio + val_ratio))

    if split == 'train':
        return df.iloc[:split1].reset_index(drop=True)
    if split == 'val':
        if val_ratio <= 0:
            raise ValueError('val_ratio is 0, cannot use split=val')
        return df.iloc[split1:split2].reset_index(drop=True)
    if split == 'test':
        if val_ratio > 0:
            return df.iloc[split2:].reset_index(drop=True)
        return df.iloc[split1:].reset_index(drop=True)

    raise ValueError(f"Unknown split: {split}")


def build_agent(algo: str, state_dim: int, action_dim: int):
    """Create an agent instance (untrained — weights will be loaded from file)."""
    # Minimal hyperparams for inference (no training happens)
    hp = dict(lr=1e-4, gamma=0.99,
              eps_start=0.05, eps_end=0.05, eps_decay=1.0,
              buffer_size=1000, batch_size=64, target_update=1000)

    if algo == 'dqn':
        return DQNAgent(state_dim, action_dim, **hp)
    elif algo == 'double_dqn':
        return DoubleDQNAgent(state_dim, action_dim, **hp)
    elif algo == 'dueling_dqn':
        return DuelingDQNAgent(state_dim, action_dim, **hp)
    elif algo == 'a2c':
        return A2CAgent(state_dim, action_dim, lr=3e-4, gamma=0.99)
    else:
        raise ValueError(f"Unknown algo: {algo}")


def run_agent(agent, df_eval: pd.DataFrame, env_kwargs: dict) -> Tuple[dict, list, list]:
    """Run a trained agent through the environment and return metrics, trades, equity."""
    env = CryptoFuturesEnv(df_eval, FEATURES, **env_kwargs)
    s, _ = env.reset()
    done = False

    while not done:
        a = agent.select_action(s, training=False)
        s, _, done, _, _ = env.step(a)

    metrics = env.get_metrics()
    metrics['algorithm'] = agent.name
    return metrics, env.trades, env.equity_curve


def main():
    ap = argparse.ArgumentParser(description='Plot RL agent trades on price chart')
    ap.add_argument('--data', default=os.path.join('data', 'SOL_USDT_15m.csv'))
    ap.add_argument('--split', choices=['train', 'val', 'test', 'full'], default='val')
    ap.add_argument('--train_ratio', type=float, default=0.6)
    ap.add_argument('--val_ratio', type=float, default=0.2)

    # Model selection
    ap.add_argument('--model_tag', required=True, help='Model tag used during training')
    ap.add_argument('--algo', choices=['dqn', 'double_dqn', 'dueling_dqn', 'a2c', 'all'], default='dqn')
    ap.add_argument('--model_dir', default='models', help='Directory containing .pt files')

    # Env params (must match training)
    ap.add_argument('--fee', type=float, default=0.0004)
    ap.add_argument('--max_pos', type=float, default=0.10)
    ap.add_argument('--sl', type=float, default=0.0)
    ap.add_argument('--tp', type=float, default=0.0)
    ap.add_argument('--min_hold', type=int, default=16)
    ap.add_argument('--cooldown', type=int, default=4)
    ap.add_argument('--trade_penalty', type=float, default=0.0002)
    ap.add_argument('--adx_threshold', type=float, default=0.0)

    # Plot controls (same as plot_trades.py)
    ap.add_argument('--mode', choices=['simple', 'detailed'], default='detailed')
    ap.add_argument('--markers', choices=['top', 'price'], default='price')
    ap.add_argument('--marker_size', type=int, default=70)
    ap.add_argument('--marker_zorder', type=int, default=20)
    ap.add_argument('--line_zorder', type=int, default=1)
    ap.add_argument('--marker_edge', type=str, default='white')
    ap.add_argument('--marker_lw', type=float, default=1.2)
    ap.add_argument('--long_open_color', type=str, default='lime')
    ap.add_argument('--long_close_color', type=str, default='green')
    ap.add_argument('--short_open_color', type=str, default='red')
    ap.add_argument('--short_close_color', type=str, default='darkred')
    ap.add_argument('--buy_color', type=str, default='lime')
    ap.add_argument('--sell_color', type=str, default='red')
    ap.add_argument('--mm_per_candle', type=float, default=1.0)
    ap.add_argument('--min_width_in', type=float, default=12.0)
    ap.add_argument('--max_width_in', type=float, default=30.0)
    ap.add_argument('--height_in', type=float, default=7.0)
    ap.add_argument('--legend_outside', action='store_true')
    ap.add_argument('--legend_inside', action='store_true')
    ap.add_argument('--out_png', type=str, default=None)
    ap.add_argument('--title', type=str, default=None)

    args = ap.parse_args()

    # ── Load data ─────────────────────────────────────────────────
    df_eval = load_split(args.data, args.split, args.train_ratio, args.val_ratio)
    print(f"Loaded {len(df_eval)} bars for split={args.split}")

    # ── Env kwargs ────────────────────────────────────────────────
    env_kwargs = dict(
        fee_rate=args.fee,
        max_position_frac=args.max_pos,
        stop_loss_pct=args.sl,
        take_profit_pct=args.tp,
        min_hold_steps=args.min_hold,
        cooldown_steps=args.cooldown,
        trade_penalty=args.trade_penalty,
        adx_threshold=args.adx_threshold,
    )

    # ── Determine which algos to plot ─────────────────────────────
    state_dim = len(FEATURES) + 3
    action_dim = 4

    if args.algo == 'all':
        algo_list = ['dqn', 'double_dqn', 'dueling_dqn', 'a2c']
    else:
        algo_list = [args.algo]

    # ── Figure sizing ─────────────────────────────────────────────
    fig_w, fig_h = _figsize_from_candles(
        n_candles=len(df_eval),
        mm_per_candle=args.mm_per_candle,
        min_width_in=args.min_width_in,
        max_width_in=args.max_width_in,
        height_in=args.height_in,
    )

    legend_outside = True
    if args.legend_inside:
        legend_outside = False
    if args.legend_outside:
        legend_outside = True

    # ── Run each algo and plot ────────────────────────────────────
    for algo_name in algo_list:
        agent = build_agent(algo_name, state_dim, action_dim)

        # Resolve model file path
        agent_file = f"{agent.name.lower().replace(' ', '_')}_{args.model_tag}.pt"
        model_path = os.path.join(args.model_dir, agent_file)

        if not os.path.exists(model_path):
            print(f"WARNING: {model_path} not found — skipping {agent.name}")
            continue

        agent.load(model_path)
        print(f"Loaded {agent.name} from {model_path}")

        # Run agent
        metrics, trades, equity = run_agent(agent, df_eval, env_kwargs)
        print(f"  {agent.name}: Sharpe={metrics['sharpe_ratio']:+.2f} "
              f"Return={metrics['total_return']:+.2f}% "
              f"DD={metrics['max_drawdown']:.1f}% "
              f"Trades={metrics['total_trades']} "
              f"WR={metrics['win_rate']:.1f}%")

        # Output path
        if args.out_png and len(algo_list) == 1:
            out_png = args.out_png
        else:
            out_dir = os.path.join('results', 'trade_plots')
            out_png = os.path.join(
                out_dir,
                f"rl_{algo_name}_{args.model_tag}_split{args.split}.png"
            )

        # Title
        if args.title and len(algo_list) == 1:
            title = args.title
        else:
            title = (
                f"{agent.name} ({args.split}) | Sharpe {metrics['sharpe_ratio']:+.2f} | "
                f"Ret {metrics['total_return']:+.2f}% | DD {metrics['max_drawdown']:.1f}% | "
                f"Trades {metrics['total_trades']} | WR {metrics['win_rate']:.1f}%"
            )

        plot_trades(
            df_eval,
            trades,
            out_png=out_png,
            title=title,
            mode=args.mode,
            markers=args.markers,
            marker_size=args.marker_size,
            marker_zorder=args.marker_zorder,
            line_zorder=args.line_zorder,
            marker_edge=args.marker_edge,
            marker_lw=args.marker_lw,
            long_open_color=args.long_open_color,
            long_close_color=args.long_close_color,
            short_open_color=args.short_open_color,
            short_close_color=args.short_close_color,
            buy_color=args.buy_color,
            sell_color=args.sell_color,
            figsize=(fig_w, fig_h),
            legend_outside=legend_outside,
        )


if __name__ == '__main__':
    main()