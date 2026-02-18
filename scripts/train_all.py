#!/usr/bin/env python3
"""
Train all four DRL agents on SOL/USDT 15 m futures data,
then backtest on the held-out test set and write results to JSON.

Usage
-----
    python scripts/train_all.py                           # defaults
    python scripts/train_all.py --data data/SOL_USDT_15m.csv --episodes 80
    python scripts/train_all.py --skip-train              # backtest only (models must exist)
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

# allow imports from scripts/
sys.path.insert(0, os.path.dirname(__file__))

from indicators import add_indicators, normalize_features
from trading_env import CryptoFuturesEnv
from models import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, A2CAgent

# ── Feature columns fed to every agent ──────────────────────────────
FEATURES = [
    'close_norm', 'open_norm', 'high_norm', 'low_norm',
    'sma_20_norm', 'sma_50_norm',
    'rsi_norm', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
    'bb_width_norm', 'atr_norm', 'volume_ratio_norm', 'returns',
]


# ═══════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_data(csv_path: str, train_ratio: float = 0.7):
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Raw rows: {len(df)}")

    df = add_indicators(df)
    df = normalize_features(df, FEATURES, window=100)
    print(f"  After indicators + normalization: {len(df)}")

    split = int(len(df) * train_ratio)
    train = df.iloc[:split].reset_index(drop=True)
    test  = df.iloc[split:].reset_index(drop=True)

    print(f"  Train: {len(train)}  ({train['timestamp'].iloc[0].date()} → {train['timestamp'].iloc[-1].date()})")
    print(f"  Test:  {len(test)}   ({test['timestamp'].iloc[0].date()} → {test['timestamp'].iloc[-1].date()})")
    return train, test


# ═══════════════════════════════════════════════════════════════════════
#  Training loops
# ═══════════════════════════════════════════════════════════════════════

def train_dqn_family(agent, train_df, n_episodes, tag=None):
    tag = tag or agent.name
    print(f"\n{'─'*60}\n  Training {tag}  ({n_episodes} episodes)\n{'─'*60}")
    env = CryptoFuturesEnv(train_df, FEATURES)
    rewards = []

    for ep in range(1, n_episodes + 1):
        s, _ = env.reset()
        total_r = 0.0; done = False
        while not done:
            a = agent.select_action(s, training=True)
            s2, r, done, _, info = env.step(a)
            agent.store(s, a, r, s2, done)
            agent.update()
            s = s2; total_r += r
        rewards.append(total_r)

        if ep % max(1, n_episodes // 10) == 0 or ep == n_episodes:
            avg = np.mean(rewards[-10:])
            print(f"  ep {ep:>4}/{n_episodes}  avgR={avg:+.4f}  "
                  f"ε={agent.eps:.3f}  equity=${info['equity']:,.0f}")
    return rewards


def train_a2c(agent, train_df, n_episodes, update_every=32):
    print(f"\n{'─'*60}\n  Training A2C  ({n_episodes} episodes)\n{'─'*60}")
    env = CryptoFuturesEnv(train_df, FEATURES)
    rewards = []

    for ep in range(1, n_episodes + 1):
        s, _ = env.reset()
        total_r = 0.0; done = False; steps = 0
        while not done:
            a = agent.select_action(s, training=True)
            s2, r, done, _, info = env.step(a)
            agent.store(s, a, r, s2, done)
            s = s2; total_r += r; steps += 1
            if steps % update_every == 0 or done:
                agent.update()
        rewards.append(total_r)

        if ep % max(1, n_episodes // 10) == 0 or ep == n_episodes:
            avg = np.mean(rewards[-10:])
            print(f"  ep {ep:>4}/{n_episodes}  avgR={avg:+.4f}  equity=${info['equity']:,.0f}")
    return rewards


# ═══════════════════════════════════════════════════════════════════════
#  Back-testing
# ═══════════════════════════════════════════════════════════════════════

def backtest(agent, test_df) -> dict:
    env = CryptoFuturesEnv(test_df, FEATURES)
    s, _ = env.reset(); done = False
    while not done:
        a = agent.select_action(s, training=False)
        s, _, done, _, _ = env.step(a)

    m = env.get_metrics()
    m['algorithm'] = agent.name

    # Downsample equity curve for JSON (keep ≤ 2000 points)
    eq = env.equity_curve
    step = max(1, len(eq) // 2000)
    m['equity_curve'] = [round(eq[i], 2) for i in range(0, len(eq), step)]

    # Keep only last 50 trades for frontend
    m['trades'] = env.trades[-50:]
    return m


# ═══════════════════════════════════════════════════════════════════════
#  Buy-and-hold baseline
# ═══════════════════════════════════════════════════════════════════════

def buy_and_hold_baseline(test_df, initial=10_000.0) -> dict:
    prices = test_df['close'].values
    equity = initial * prices / prices[0]
    eq = equity.tolist()
    ret = (equity[-1] / initial - 1) * 100
    rets = np.diff(equity) / equity[:-1]
    ann = np.sqrt(4*24*365)
    sharpe = (rets.mean() / rets.std() * ann) if rets.std() > 0 else 0
    peak = np.maximum.accumulate(equity)
    mdd = ((peak - equity) / peak).max() * 100

    step = max(1, len(eq) // 2000)
    return dict(
        algorithm='Buy & Hold',
        total_return=round(ret, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown=round(mdd, 2),
        win_rate=0, total_trades=1, avg_trade_pnl=0,
        final_balance=round(equity[-1], 2),
        equity_curve=[round(eq[i], 2) for i in range(0, len(eq), step)],
        trades=[],
    )


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',     default='data/SOL_USDT_15m.csv')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--output',   default='results/backtest_results.json')
    ap.add_argument('--skip-train', action='store_true')
    args = ap.parse_args()

    train_df, test_df = load_data(args.data)

    state_dim  = len(FEATURES) + 3   # features + position meta
    action_dim = 4                    # hold / long / short / close

    hp = dict(lr=1e-4, gamma=0.99,
              eps_start=1.0, eps_end=0.05, eps_decay=0.998,
              buffer_size=50_000, batch_size=64, target_update=1000)

    agents = [
        DQNAgent(state_dim, action_dim, **hp),
        DoubleDQNAgent(state_dim, action_dim, **hp),
        DuelingDQNAgent(state_dim, action_dim, **hp),
        A2CAgent(state_dim, action_dim, lr=3e-4, gamma=0.99),
    ]

    # ── Train ────────────────────────────────────────────────────
    if not args.skip_train:
        t0 = time.time()
        for ag in agents:
            if isinstance(ag, A2CAgent):
                train_a2c(ag, train_df, args.episodes)
            else:
                train_dqn_family(ag, train_df, args.episodes)
            ag.save(f"models/{ag.name.lower().replace(' ', '_')}.pt")
        print(f"\nTotal training time: {time.time()-t0:.0f}s")
    else:
        for ag in agents:
            path = f"models/{ag.name.lower().replace(' ', '_')}.pt"
            if os.path.exists(path):
                ag.load(path)
                print(f"Loaded {ag.name} from {path}")
            else:
                print(f"WARNING: {path} not found — backtest will use random weights")

    # ── Backtest ─────────────────────────────────────────────────
    print(f"\n{'═'*60}\n  BACKTEST RESULTS\n{'═'*60}")
    results = {}
    for ag in agents:
        m = backtest(ag, test_df)
        results[ag.name] = m
        print(f"\n  {ag.name:<15}  return={m['total_return']:>+8.2f}%  "
              f"sharpe={m['sharpe_ratio']:>6.2f}  "
              f"maxDD={m['max_drawdown']:>6.2f}%  "
              f"winR={m['win_rate']:>5.1f}%  "
              f"trades={m['total_trades']}")

    # Buy & hold baseline
    bh = buy_and_hold_baseline(test_df)
    results['Buy & Hold'] = bh
    print(f"\n  {'Buy & Hold':<15}  return={bh['total_return']:>+8.2f}%  "
          f"sharpe={bh['sharpe_ratio']:>6.2f}  "
          f"maxDD={bh['max_drawdown']:>6.2f}%")

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved → {args.output}")

    # Also copy to public/ for the frontend
    pub = os.path.join('public', 'backtest_results.json')
    os.makedirs('public', exist_ok=True)
    with open(pub, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results copied → {pub}")


if __name__ == '__main__':
    main()
