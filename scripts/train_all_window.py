#!/usr/bin/env python3
"""
Train all four DRL agents on SOL/USDT 15 m futures data,
then backtest on a held-out set and write results to JSON.

This version supports a 3-way split (train, val, test) so you can tune
environment parameters on validation without leaking into the final test.

Usage
-----
    python scripts/train_all_window.py
    python scripts/train_all_window.py --episodes 80 --algo double_dqn --eval val --no_public_copy
    python scripts/train_all_window.py --skip-train --model_tag my_tag --eval test
"""
import argparse, json, os, sys, time, random
import numpy as np
import pandas as pd
import torch

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

def load_data(csv_path: str, train_ratio: float = 0.6, val_ratio: float = 0.2):
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Raw rows: {len(df)}")

    df = add_indicators(df)
    df = normalize_features(df, FEATURES, window=100)
    print(f"  After indicators + normalization: {len(df)}")

    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if val_ratio < 0 or (train_ratio + val_ratio) >= 1:
        raise ValueError("val_ratio must be >= 0 and train_ratio + val_ratio must be < 1")

    split1 = int(len(df) * train_ratio)
    split2 = int(len(df) * (train_ratio + val_ratio))

    train = df.iloc[:split1].reset_index(drop=True)
    val   = df.iloc[split1:split2].reset_index(drop=True) if val_ratio > 0 else None
    test  = df.iloc[split2:].reset_index(drop=True) if val_ratio > 0 else df.iloc[split1:].reset_index(drop=True)

    print(f"  Train: {len(train)}  ({train['timestamp'].iloc[0].date()} → {train['timestamp'].iloc[-1].date()})")
    if val is not None:
        print(f"  Val:   {len(val)}   ({val['timestamp'].iloc[0].date()} → {val['timestamp'].iloc[-1].date()})")
        print(f"  Test:  {len(test)}  ({test['timestamp'].iloc[0].date()} → {test['timestamp'].iloc[-1].date()})")
    else:
        print(f"  Test:  {len(test)}  ({test['timestamp'].iloc[0].date()} → {test['timestamp'].iloc[-1].date()})")

    return train, val, test


# ═══════════════════════════════════════════════════════════════════════
#  Training loops
# ═══════════════════════════════════════════════════════════════════════

def train_dqn_family(agent, train_df, n_episodes, tag=None,
                     window_size=2000, update_every=4, env_kwargs=None, rng=None, log_every: int = 0):
    """
    Train DQN-family agent using random windows of the training data.

    Instead of walking through all bars each episode (slow),
    we sample a random window per episode and update the network
    every few steps instead of every step.
    """
    tag = tag or agent.name
    print(f"\n{'─'*60}\n  Training {tag}  ({n_episodes} eps × {window_size} steps)\n{'─'*60}")

    if rng is None:
        rng = np.random.default_rng()

    max_start = len(train_df) - window_size
    rewards = []

    t_start = time.time()
    le = int(log_every) if int(log_every) > 0 else max(1, n_episodes // 10)
    for ep in range(1, n_episodes + 1):
        start = 0 if max_start <= 0 else int(rng.integers(0, max_start + 1))
        window_df = train_df.iloc[start:start + window_size].reset_index(drop=True)
        env = CryptoFuturesEnv(window_df, FEATURES, **(env_kwargs or {}))

        s, _ = env.reset()
        total_r = 0.0; done = False; steps = 0
        while not done:
            a = agent.select_action(s, training=True)
            s2, r, done, _, info = env.step(a)
            agent.store(s, a, r, s2, done)
            steps += 1
            if steps % update_every == 0:
                agent.update()
            s = s2; total_r += r
        rewards.append(total_r)

        if ep % le == 0 or ep == n_episodes:
            avg = np.mean(rewards[-10:])
            elapsed = time.time() - t_start
            sec_per_ep = elapsed / max(1, ep)
            eta = sec_per_ep * (n_episodes - ep)
            print(f"  ep {ep:>4}/{n_episodes}  avgR={avg:+.4f}  "
                  f"ε={agent.eps:.3f}  equity=${info['equity']:,.0f}  "
                  f"t/ep={sec_per_ep:.2f}s  ETA={_fmt_time(eta)}")
    return rewards


def train_a2c(agent, train_df, n_episodes,
              window_size=2000, env_kwargs=None, rng=None, log_every: int = 0):
    """Train A2C on random windows (one update per window)."""
    print(f"\n{'─'*60}\n  Training A2C  ({n_episodes} eps × {window_size} steps)\n{'─'*60}")

    if rng is None:
        rng = np.random.default_rng()

    max_start = len(train_df) - window_size
    rewards = []

    t_start = time.time()
    le = int(log_every) if int(log_every) > 0 else max(1, n_episodes // 10)
    for ep in range(1, n_episodes + 1):
        start = 0 if max_start <= 0 else int(rng.integers(0, max_start + 1))
        window_df = train_df.iloc[start:start + window_size].reset_index(drop=True)
        env = CryptoFuturesEnv(window_df, FEATURES, **(env_kwargs or {}))

        s, _ = env.reset()
        total_r = 0.0; done = False
        while not done:
            a = agent.select_action(s, training=True)
            s2, r, done, _, info = env.step(a)
            agent.store(s, a, r, s2, done)
            s = s2; total_r += r

        agent.update()
        rewards.append(total_r)

        if ep % le == 0 or ep == n_episodes:
            avg = np.mean(rewards[-10:])
            elapsed = time.time() - t_start
            sec_per_ep = elapsed / max(1, ep)
            eta = sec_per_ep * (n_episodes - ep)
            print(f"  ep {ep:>4}/{n_episodes}  avgR={avg:+.4f}  equity=${info['equity']:,.0f}  "
                  f"t/ep={sec_per_ep:.2f}s  ETA={_fmt_time(eta)}")
    return rewards


# ═══════════════════════════════════════════════════════════════════════
#  Back-testing
# ═══════════════════════════════════════════════════════════════════════

def backtest(agent, df_eval, env_kwargs=None) -> dict:
    env = CryptoFuturesEnv(df_eval, FEATURES, **(env_kwargs or {}))
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

def buy_and_hold_baseline(df_eval, initial=10_000.0) -> dict:
    prices = df_eval['close'].values
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


def _fmt_tag(x):
    if isinstance(x, int):
        return str(x)
    s = f"{x}"
    s = s.replace('-', 'm').replace('.', 'p')
    return s


def _fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['TORCH_NUM_THREADS'] = '1'

    ap = argparse.ArgumentParser()
    ap.add_argument('--data',     default='data/SOL_USDT_15m.csv')
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--log_every', type=int, default=0,
                    help='Print training progress every N episodes (0 = auto)')
    ap.add_argument('--window',   type=int, default=2000,
                    help='Training window size per episode (default 2000 bars)')
    ap.add_argument('--seed',     type=int, default=42,
                    help='Random seed for reproducible training windows')
    ap.add_argument('--train_ratio', type=float, default=0.6)
    ap.add_argument('--val_ratio',   type=float, default=0.2)
    ap.add_argument('--algo', choices=['all','dqn','double_dqn','dueling_dqn','a2c'], default='all')
    ap.add_argument('--eval', choices=['val','test','both'], default='test')
    ap.add_argument('--fee',      type=float, default=0.0004)
    ap.add_argument('--max_pos',  type=float, default=0.2)
    ap.add_argument('--sl',       type=float, default=0.0)
    ap.add_argument('--tp',       type=float, default=0.0)
    ap.add_argument('--min_hold', type=int,   default=16)
    ap.add_argument('--cooldown', type=int,   default=4)
    ap.add_argument('--trade_penalty', type=float, default=0.0002)
    ap.add_argument('--lr',        type=float, default=1e-4,
                    help='Learning rate for all agents')
    ap.add_argument('--eps_decay', type=float, default=0.99997,
                    help='Epsilon decay rate for DQN-family agents')
    ap.add_argument('--model_tag', default='')
    ap.add_argument('--output',   default='results/backtest_results.json')
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--no_public_copy', action='store_true')
    args = ap.parse_args()

    # Make runs repeatable (window sampling + replay buffer sampling + NN init)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    env_kwargs = dict(
        fee_rate=args.fee,
        max_position_frac=args.max_pos,
        stop_loss_pct=args.sl,
        take_profit_pct=args.tp,
        min_hold_steps=args.min_hold,
        cooldown_steps=args.cooldown,
        trade_penalty=args.trade_penalty,
    )

    run_tag = (
        f"seed{args.seed}_fee{_fmt_tag(args.fee)}_mp{_fmt_tag(args.max_pos)}"
        f"_sl{_fmt_tag(args.sl)}_tp{_fmt_tag(args.tp)}"
        f"_mh{_fmt_tag(args.min_hold)}_cd{_fmt_tag(args.cooldown)}_p{_fmt_tag(args.trade_penalty)}"
    )
    model_tag = args.model_tag.strip() or run_tag

    train_df, val_df, test_df = load_data(args.data, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    state_dim  = len(FEATURES) + 3   # features + position meta
    action_dim = 4                    # hold / long / short / close

    hp = dict(lr=args.lr, gamma=0.99,
              eps_start=1.0, eps_end=0.05, eps_decay=args.eps_decay,
              buffer_size=50_000, batch_size=64, target_update=1000)

    # A2C typically benefits from a higher LR; use 3x the base LR (capped at 1e-3)
    a2c_lr = min(args.lr * 3, 1e-3)

    agents_all = [
        DQNAgent(state_dim, action_dim, **hp),
        DoubleDQNAgent(state_dim, action_dim, **hp),
        DuelingDQNAgent(state_dim, action_dim, **hp),
        A2CAgent(state_dim, action_dim, lr=a2c_lr, gamma=0.99),
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

    # ── Train ────────────────────────────────────────────────────
    if not args.skip_train:
        t0 = time.time()
        os.makedirs('models', exist_ok=True)
        for ag in agents:
            if isinstance(ag, A2CAgent):
                train_a2c(ag, train_df, args.episodes, window_size=args.window, env_kwargs=env_kwargs, rng=rng, log_every=args.log_every)
            else:
                train_dqn_family(ag, train_df, args.episodes, window_size=args.window, env_kwargs=env_kwargs, rng=rng, log_every=args.log_every)
            ag.save(f"models/{ag.name.lower().replace(' ', '_')}_{model_tag}.pt")
        print(f"\nTotal training time: {time.time()-t0:.0f}s")
    else:
        for ag in agents:
            path = f"models/{ag.name.lower().replace(' ', '_')}_{model_tag}.pt"
            if os.path.exists(path):
                ag.load(path)
                print(f"Loaded {ag.name} from {path}")
            else:
                print(f"WARNING: {path} not found - backtest will use random weights")

    # ── Backtest ─────────────────────────────────────────────────
    if args.eval == 'test':
        print(f"\n{'═'*60}\n  BACKTEST RESULTS (TEST)\n{'═'*60}")
        eval_sets = [('test', test_df)]
    elif args.eval == 'val':
        if val_df is None:
            raise ValueError("val_ratio is 0, cannot evaluate on validation set")
        print(f"\n{'═'*60}\n  BACKTEST RESULTS (VAL)\n{'═'*60}")
        eval_sets = [('val', val_df)]
    else:
        if val_df is None:
            raise ValueError("val_ratio is 0, cannot evaluate on validation set")
        print(f"\n{'═'*60}\n  BACKTEST RESULTS (VAL + TEST)\n{'═'*60}")
        eval_sets = [('val', val_df), ('test', test_df)]

    results = {}
    for split_name, df_eval in eval_sets:
        split_results = {}
        for ag in agents:
            m = backtest(ag, df_eval, env_kwargs=env_kwargs)
            split_results[ag.name] = m
            print(f"\n  [{split_name}] {ag.name:<11}  return={m['total_return']:>+8.2f}%  "
                  f"sharpe={m['sharpe_ratio']:>6.2f}  "
                  f"maxDD={m['max_drawdown']:>6.2f}%  "
                  f"winR={m['win_rate']:>5.1f}%  "
                  f"trades={m['total_trades']}")

        bh = buy_and_hold_baseline(df_eval)
        split_results['Buy & Hold'] = bh
        print(f"\n  [{split_name}] {'Buy & Hold':<11}  return={bh['total_return']:>+8.2f}%  "
              f"sharpe={bh['sharpe_ratio']:>6.2f}  "
              f"maxDD={bh['max_drawdown']:>6.2f}%")

        results[split_name] = split_results

    # If only one split was evaluated, keep backward compatible shape (flat dict)
    if len(results) == 1:
        results = next(iter(results.values()))

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved -> {args.output}")

    # Also copy to public/ for the frontend
    if not args.no_public_copy:
        pub = os.path.join('public', 'backtest_results.json')
        os.makedirs('public', exist_ok=True)
        with open(pub, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results copied -> {pub}")


if __name__ == '__main__':
    main()