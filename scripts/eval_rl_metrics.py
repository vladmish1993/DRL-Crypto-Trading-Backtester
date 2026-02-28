#!/usr/bin/env python3
"""
eval_rl_metrics.py

Run a trained RL agent on a chosen split (including val_ratio=1.0 "full via loader"),
and print metrics (trades + profit) instead of plotting.

Can also scan a models directory and evaluate ALL models into a CSV.

Examples
--------
# Evaluate one model (like plot_rl_trades, but no chart)
python scripts/eval_rl_metrics.py \
  --data data/SOL_USDT_15m.csv \
  --split val --train_ratio 0 --val_ratio 1 \
  --model_tag sweep_seed42_lr0p0002_ed0p99997_mp0p1_mh24_cd2_p0p0003_sl0p0_tp0p0 \
  --algo double_dqn \
  --fee 0.0004 --max_pos 0.10 \
  --sl 0.02 --tp 0.07 \
  --min_hold 64 --cooldown 4 \
  --trade_penalty 0.0004 \
  --out_csv results/rl_eval.csv

# Evaluate every *.pt in models/ and save a CSV
python scripts/eval_rl_metrics.py --data data/SOL_USDT_15m.csv \
  --split val --train_ratio 0 --val_ratio 1 \
  --scan_models --out_csv results/rl_eval_all.csv
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# allow imports from scripts/
sys.path.insert(0, os.path.dirname(__file__))

from indicators import add_indicators, normalize_features
from trading_env import CryptoFuturesEnv
from models import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, A2CAgent


# IMPORTANT:
# Put optional/newer features (like ADX) at the END, so older checkpoints (feat_dim=14)
# naturally drop the newest columns instead of dropping 'returns' etc.
FEATURES_ALL = [
    'close_norm', 'open_norm', 'high_norm', 'low_norm',
    'sma_20_norm', 'sma_50_norm',
    'rsi_norm',
    'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
    'bb_width_norm', 'atr_norm', 'volume_ratio_norm',
    'returns',
    'adx_norm',
]


def load_split(csv_path: str, split: str, train_ratio: float, val_ratio: float, norm_window: int = 100) -> pd.DataFrame:
    """Same behaviour as plot_rl_trades: read CSV -> indicators -> normalise -> slice."""
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = add_indicators(df)
    df = normalize_features(df, FEATURES_ALL, window=norm_window)

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
    """Create an agent instance for inference (weights loaded from file)."""
    algo = algo.lower()
    # Minimal hyperparams for inference; training params don't matter here.
    hp = dict(lr=1e-4, gamma=0.99,
              eps_start=0.05, eps_end=0.05, eps_decay=1.0,
              buffer_size=1000, batch_size=64, target_update=1000)

    if algo == 'dqn':
        return DQNAgent(state_dim, action_dim, **hp)
    if algo == 'double_dqn':
        return DoubleDQNAgent(state_dim, action_dim, **hp)
    if algo == 'dueling_dqn':
        return DuelingDQNAgent(state_dim, action_dim, **hp)
    if algo == 'a2c':
        return A2CAgent(state_dim, action_dim, lr=3e-4, gamma=0.99)

    raise ValueError(f"Unknown algo: {algo}")


def infer_state_dim_from_checkpoint(model_path: str) -> int | None:
    """
    Infer state_dim by reading the first layer weight shape from the checkpoint.
    Supports DQN-family checkpoints saved as {'q': state_dict, ...}.
    """
    ck = torch.load(model_path, map_location="cpu")
    if not isinstance(ck, dict):
        return None

    # DQN family: ck['q']['net.0.weight'] typically exists
    if 'q' in ck and isinstance(ck['q'], dict):
        sd = ck['q']
        if 'net.0.weight' in sd:
            return int(sd['net.0.weight'].shape[1])
        # fallback: first 2D weight
        for k in sorted(sd.keys()):
            w = sd[k]
            if hasattr(w, 'dim') and w.dim() == 2:
                return int(w.shape[1])

    # A2C or other: try 'net'
    if 'net' in ck and isinstance(ck['net'], dict):
        sd = ck['net']
        for k in sorted(sd.keys()):
            w = sd[k]
            if hasattr(w, 'dim') and w.dim() == 2:
                return int(w.shape[1])

    return None


def filter_env_kwargs(env_kwargs: dict) -> dict:
    """Filter kwargs to match CryptoFuturesEnv.__init__ signature (keeps script compatible across versions)."""
    try:
        sig = inspect.signature(CryptoFuturesEnv.__init__)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in env_kwargs.items() if k in allowed}
    except Exception:
        return env_kwargs


def run_agent(agent, df_eval: pd.DataFrame, feature_cols: List[str], env_kwargs: dict) -> Tuple[dict, list, list, float]:
    """Run agent through env and return metrics, trades, equity, initial_balance."""
    env_kwargs = filter_env_kwargs(env_kwargs)
    env = CryptoFuturesEnv(df_eval, feature_cols, **env_kwargs)
    s, _ = env.reset()
    done = False

    while not done:
        a = agent.select_action(s, training=False)
        s, _, done, _, _ = env.step(a)

    metrics = env.get_metrics()
    metrics['algorithm'] = getattr(agent, 'name', agent.__class__.__name__)
    return metrics, getattr(env, 'trades', []), getattr(env, 'equity_curve', []), getattr(env, 'initial_balance', 0.0)


def parse_model_filename(pt_name: str) -> Tuple[str, str]:
    """
    Expected: <algo>_<model_tag>.pt, e.g. double_dqn_my_tag.pt

    NOTE: Some algos contain underscores (double_dqn, dueling_dqn),
    so we match the *longest* known prefix first.
    Returns (algo, model_tag)
    """
    stem = pt_name[:-3] if pt_name.endswith(".pt") else pt_name

    known = ["dueling_dqn", "double_dqn", "dqn", "a2c"]
    for a in known:
        prefix = a + "_"
        if stem.lower().startswith(prefix):
            return a, stem[len(prefix):]

    if "_" not in stem:
        return ("unknown", stem)
    algo, tag = stem.split("_", 1)
    return algo.lower(), tag


def ensure_out_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Evaluate RL models and print trades/profit (no chart).")

    ap.add_argument('--data', default=os.path.join('data', 'SOL_USDT_15m.csv'))
    ap.add_argument('--split', choices=['train', 'val', 'test', 'full'], default='val')
    ap.add_argument('--train_ratio', type=float, default=0.0)
    ap.add_argument('--val_ratio', type=float, default=1.0)
    ap.add_argument('--norm_window', type=int, default=100)

    # Model selection
    ap.add_argument('--model_tag', default=None, help='Model tag used during training')
    ap.add_argument('--algo', choices=['dqn', 'double_dqn', 'dueling_dqn', 'a2c', 'all'], default='dqn')
    ap.add_argument('--model_dir', default='models', help='Directory containing .pt files')

    # Batch mode
    ap.add_argument('--scan_models', action='store_true', help='Evaluate ALL .pt files in --model_dir (ignores --model_tag/--algo)')
    ap.add_argument('--out_csv', default=None, help='Write/append results to this CSV')

    # Env params (must match training)
    ap.add_argument('--fee', type=float, default=0.0004)
    ap.add_argument('--max_pos', type=float, default=0.10)
    ap.add_argument('--sl', type=float, default=0.0)
    ap.add_argument('--tp', type=float, default=0.0)
    ap.add_argument('--min_hold', type=int, default=16)
    ap.add_argument('--cooldown', type=int, default=4)
    ap.add_argument('--trade_penalty', type=float, default=0.0002)
    ap.add_argument('--adx_threshold', type=float, default=0.0)
    ap.add_argument('--initial_balance', type=float, default=10_000.0)

    args = ap.parse_args()

    # Keep CPU threading sane (important if you run many evals)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # Load data once (fast) — note: split happens after indicators/normalisation.
    df_eval = load_split(args.data, args.split, args.train_ratio, args.val_ratio, norm_window=args.norm_window)
    print(f"Loaded {len(df_eval)} bars for split={args.split} (train_ratio={args.train_ratio}, val_ratio={args.val_ratio})")

    env_kwargs = dict(
        initial_balance=args.initial_balance,
        fee_rate=args.fee,
        max_position_frac=args.max_pos,
        stop_loss_pct=args.sl,
        take_profit_pct=args.tp,
        min_hold_steps=args.min_hold,
        cooldown_steps=args.cooldown,
        trade_penalty=args.trade_penalty,
        adx_threshold=args.adx_threshold,
    )

    action_dim = 4

    def eval_one(model_path: Path, algo: str, tag: str) -> Dict:
        t0 = time.time()
        row = dict(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            model_file=model_path.name,
            algo=algo,
            model_tag=tag,
            split=args.split,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            fee=args.fee,
            max_pos=args.max_pos,
            sl=args.sl,
            tp=args.tp,
            min_hold=args.min_hold,
            cooldown=args.cooldown,
            trade_penalty=args.trade_penalty,
            adx_threshold=args.adx_threshold,
        )

        try:
            state_dim = infer_state_dim_from_checkpoint(str(model_path))
            if state_dim is None:
                # fallback: assume current features list
                state_dim = len(FEATURES_ALL) + 3

            feat_dim = state_dim - 3
            if feat_dim <= 0:
                raise RuntimeError(f"Bad inferred state_dim={state_dim}")

            # choose the first feat_dim features from FEATURES_ALL (newest features are at the end)
            feature_cols = FEATURES_ALL[:feat_dim]

            # sanity check: all features exist in df
            missing = [c for c in feature_cols if c not in df_eval.columns]
            if missing:
                raise RuntimeError(f"Missing feature columns in data: {missing}")

            agent = build_agent(algo, state_dim, action_dim)
            agent.load(str(model_path))

            metrics, trades, equity, init_bal = run_agent(agent, df_eval, feature_cols, env_kwargs)
            profit_abs = float(metrics.get("final_balance", 0.0)) - float(init_bal)

            row.update(dict(
                ok=1,
                elapsed_s=round(time.time() - t0, 2),
                total_trades=metrics.get("total_trades"),
                total_return=metrics.get("total_return"),
                profit_abs=round(profit_abs, 2),
                sharpe_ratio=metrics.get("sharpe_ratio"),
                max_drawdown=metrics.get("max_drawdown"),
                win_rate=metrics.get("win_rate"),
                final_balance=metrics.get("final_balance"),
                sl_hits=metrics.get("sl_hits"),
                tp_hits=metrics.get("tp_hits"),
            ))
        except Exception as e:
            row.update(dict(
                ok=0,
                elapsed_s=round(time.time() - t0, 2),
                error=str(e)[:400],
            ))
        return row

    rows: List[Dict] = []

    if args.scan_models:
        model_dir = Path(args.model_dir)
        pts = sorted(model_dir.glob("*.pt"))
        if not pts:
            print(f"No .pt files found in {model_dir.resolve()}")
            return

        for p in pts:
            algo, tag = parse_model_filename(p.name)
            if algo not in ("dqn", "double_dqn", "dueling_dqn", "a2c"):
                print(f"[SKIP] {p.name} (unknown algo prefix '{algo}')")
                continue
            r = eval_one(p, algo, tag)
            rows.append(r)
            if r.get("ok"):
                print(f"[OK]  {p.name}  trades={r.get('total_trades')}  return={r.get('total_return')}%  profit={r.get('profit_abs')}")
            else:
                print(f"[FAIL] {p.name}  {r.get('error')}")
    else:
        if not args.model_tag:
            raise SystemExit("ERROR: --model_tag is required unless you use --scan_models")

        if args.algo == 'all':
            algo_list = ['dqn', 'double_dqn', 'dueling_dqn', 'a2c']
        else:
            algo_list = [args.algo]

        for algo in algo_list:
            model_path = Path(args.model_dir) / f"{algo}_{args.model_tag}.pt"
            if not model_path.exists():
                print(f"WARNING: {model_path} not found — skipping")
                continue

            r = eval_one(model_path, algo, args.model_tag)
            rows.append(r)
            if r.get("ok"):
                print(f"{algo}: trades={r.get('total_trades')}  return={r.get('total_return')}%  profit={r.get('profit_abs')} "
                      f"(Sharpe={r.get('sharpe_ratio')}, DD={r.get('max_drawdown')}%)")
            else:
                print(f"{algo}: FAIL {r.get('error')}")

    if args.out_csv:
        ensure_out_dir(args.out_csv)
        df_out = pd.DataFrame(rows)
        # append if exists
        out_path = Path(args.out_csv)
        if out_path.exists():
            df_prev = pd.read_csv(out_path)
            df_out = pd.concat([df_prev, df_out], ignore_index=True)
        df_out.to_csv(out_path, index=False)
        print(f"Saved CSV -> {out_path.resolve()}")


if __name__ == '__main__':
    main()
