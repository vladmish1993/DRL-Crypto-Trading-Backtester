#!/usr/bin/env python3
"""
eval_comprehensive.py — Full model evaluation pipeline.

Step 1: Evaluate ALL models on test + full splits with correct per-model params
Step 2: Find models positive on BOTH test and full
Step 3: Sweep inference params (min_hold, cooldown, max_pos, trade_penalty) on top N
Step 4: Print final leaderboard

Usage:
    # Full pipeline
    python scripts/eval_comprehensive.py

    # Just step 1 (eval all models)
    python scripts/eval_comprehensive.py --step 1

    # Just step 3 (sweep params on top models, requires step 1 CSV)
    python scripts/eval_comprehensive.py --step 3

    # Print results only (no eval)
    python scripts/eval_comprehensive.py --step 4

    # Custom settings
    python scripts/eval_comprehensive.py --model_dir models --top_n 10 --min_trades 30
"""

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from indicators import add_indicators, normalize_features
from trading_env import CryptoFuturesEnv
from models import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, A2CAgent

# Features ordered so newest (adx_norm) is last — trimming drops newest first
FEATURES_ALL = [
    'close_norm', 'open_norm', 'high_norm', 'low_norm',
    'sma_20_norm', 'sma_50_norm',
    'rsi_norm',
    'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
    'bb_width_norm', 'atr_norm', 'volume_ratio_norm',
    'returns',
    'adx_norm',
]


# ── Helpers ────────────────────────────────────────────────────────

def load_data(csv_path: str, norm_window: int = 100) -> pd.DataFrame:
    """Load and prepare full dataset (split later)."""
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = add_indicators(df)
    df = normalize_features(df, FEATURES_ALL, window=norm_window)
    return df


def get_split(df: pd.DataFrame, split: str, train_ratio: float) -> pd.DataFrame:
    """Get a split from already-prepared dataframe."""
    if split == 'full':
        return df.reset_index(drop=True)
    split1 = int(len(df) * train_ratio)
    if split == 'train':
        return df.iloc[:split1].reset_index(drop=True)
    if split == 'test':
        return df.iloc[split1:].reset_index(drop=True)
    raise ValueError(f"Unknown split: {split}")


def parse_algo_and_tag(pt_name: str) -> Tuple[str, str]:
    """Parse algo and tag from filename like 'double_dqn_my_tag.pt'."""
    stem = pt_name[:-3] if pt_name.endswith('.pt') else pt_name
    for a in ['dueling_dqn', 'double_dqn', 'dqn', 'a2c']:
        if stem.startswith(a + '_'):
            return a, stem[len(a) + 1:]
    return 'unknown', stem


def parse_tag_params(tag: str) -> Dict:
    """Extract training params encoded in the model tag."""
    d = {}

    m = re.search(r'mh(\d+)', tag)
    d['min_hold'] = int(m.group(1)) if m else 16

    m = re.search(r'cd(\d+)', tag)
    d['cooldown'] = int(m.group(1)) if m else 4

    m = re.search(r'adx(\d+)', tag)
    d['adx_threshold'] = int(m.group(1)) if m else 0

    m = re.search(r'pen(\dp\d+)', tag)
    d['trade_penalty'] = float(m.group(1).replace('p', '.')) if m else 0.0004

    m = re.search(r'_mp(\dp\d+)', tag)
    d['max_pos'] = float(m.group(1).replace('p', '.')) if m else 0.10

    m = re.search(r'_sl(\dp\d+)', tag)
    d['sl'] = float(m.group(1).replace('p', '.')) if m else 0.02

    m = re.search(r'_tp(\dp\d+)', tag)
    d['tp'] = float(m.group(1).replace('p', '.')) if m else 0.07

    return d


def infer_state_dim(model_path: str) -> int:
    """Infer state_dim from checkpoint weights."""
    ck = torch.load(model_path, map_location='cpu')
    if isinstance(ck, dict):
        for key in ['q', 'net']:
            if key in ck and isinstance(ck[key], dict):
                for k in sorted(ck[key].keys()):
                    w = ck[key][k]
                    if hasattr(w, 'dim') and w.dim() == 2:
                        return int(w.shape[1])
    return len(FEATURES_ALL) + 3


def build_agent(algo: str, state_dim: int, action_dim: int = 4):
    """Create agent for inference."""
    hp = dict(lr=1e-4, gamma=0.99, eps_start=0.05, eps_end=0.05,
              eps_decay=1.0, buffer_size=1000, batch_size=64, target_update=1000)
    if algo == 'dqn':
        return DQNAgent(state_dim, action_dim, **hp)
    if algo == 'double_dqn':
        return DoubleDQNAgent(state_dim, action_dim, **hp)
    if algo == 'dueling_dqn':
        return DuelingDQNAgent(state_dim, action_dim, **hp)
    if algo == 'a2c':
        return A2CAgent(state_dim, action_dim, lr=3e-4, gamma=0.99)
    raise ValueError(f"Unknown algo: {algo}")


def filter_env_kwargs(kwargs: dict) -> dict:
    """Filter kwargs to match CryptoFuturesEnv.__init__ signature."""
    try:
        sig = inspect.signature(CryptoFuturesEnv.__init__)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def eval_model(agent, df_split: pd.DataFrame, feature_cols: list, env_kwargs: dict) -> Dict:
    """Run agent on data and return metrics."""
    env_kwargs = filter_env_kwargs(env_kwargs)
    env = CryptoFuturesEnv(df_split, feature_cols, **env_kwargs)
    s, _ = env.reset()
    done = False
    while not done:
        a = agent.select_action(s, training=False)
        s, _, done, _, _ = env.step(a)
    return env.get_metrics()


# ── Step 1: Eval all models on test + full ────────────────────────

def step1_eval_all(model_dir: str, df_full: pd.DataFrame, train_ratio: float,
                   out_csv: str) -> pd.DataFrame:
    """Evaluate every .pt model on both test and full splits."""
    models_dir = Path(model_dir)
    pts = sorted(models_dir.glob('*.pt'))
    print(f"\n{'='*80}")
    print(f"  STEP 1: Evaluating {len(pts)} models on test + full splits")
    print(f"{'='*80}")

    df_test = get_split(df_full, 'test', train_ratio)

    rows = []
    for idx, p in enumerate(pts, 1):
        algo, tag = parse_algo_and_tag(p.name)
        if algo not in ('dqn', 'double_dqn', 'dueling_dqn', 'a2c'):
            continue

        tag_params = parse_tag_params(tag)

        # Infer feature dim from checkpoint
        state_dim = infer_state_dim(str(p))
        feat_dim = state_dim - 3
        feature_cols = FEATURES_ALL[:feat_dim]

        # Build and load agent
        try:
            agent = build_agent(algo, state_dim)
            agent.load(str(p))
        except Exception as e:
            print(f"  [{idx}/{len(pts)}] FAIL load {p.name}: {e}")
            continue

        # Eval on both splits with TRAINING params
        for split_name, df_split in [('test', df_test), ('full', df_full)]:
            env_kwargs = dict(
                fee_rate=0.0004,
                max_position_frac=tag_params.get('max_pos', 0.10),
                stop_loss_pct=tag_params.get('sl', 0.02),
                take_profit_pct=tag_params.get('tp', 0.07),
                min_hold_steps=tag_params['min_hold'],
                cooldown_steps=tag_params['cooldown'],
                trade_penalty=0,  # no penalty at eval
                adx_threshold=tag_params['adx_threshold'],
            )

            try:
                metrics = eval_model(agent, df_split, feature_cols, env_kwargs)
                rows.append({
                    'model_file': p.name,
                    'algo': algo,
                    'model_tag': tag,
                    'split': split_name,
                    'sharpe': metrics.get('sharpe_ratio', 0),
                    'total_return': metrics.get('total_return', 0),
                    'max_dd': metrics.get('max_drawdown', 0),
                    'trades': metrics.get('total_trades', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'final_balance': metrics.get('final_balance', 0),
                    # Training params (for reference)
                    'train_min_hold': tag_params['min_hold'],
                    'train_cooldown': tag_params['cooldown'],
                    'train_max_pos': tag_params.get('max_pos', 0.10),
                    'train_penalty': tag_params['trade_penalty'],
                    'train_adx': tag_params['adx_threshold'],
                    'train_sl': tag_params.get('sl', 0.02),
                    'train_tp': tag_params.get('tp', 0.07),
                })
            except Exception as e:
                print(f"  [{idx}/{len(pts)}] FAIL eval {p.name} on {split_name}: {e}")

        if idx % 20 == 0 or idx == len(pts):
            print(f"  [{idx}/{len(pts)}] evaluated")

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"  Saved {len(df_out)} rows -> {out_csv}")
    return df_out


# ── Step 2: Find models positive on both splits ──────────────────

def step2_find_dual_positive(df_eval: pd.DataFrame, min_trades: int = 30) -> pd.DataFrame:
    """Find models with positive Sharpe on both test and full, with enough trades."""
    print(f"\n{'='*80}")
    print(f"  STEP 2: Finding models positive on both test + full (trades >= {min_trades})")
    print(f"{'='*80}")

    test = df_eval[(df_eval.split == 'test') & (df_eval.trades >= min_trades)].copy()
    full = df_eval[(df_eval.split == 'full') & (df_eval.trades >= min_trades)].copy()

    merged = test.merge(
        full[['model_tag', 'sharpe', 'total_return', 'max_dd', 'trades', 'win_rate']],
        on='model_tag',
        suffixes=('_test', '_full'),
        how='inner',
    )

    # Both positive
    both_pos = merged[(merged.sharpe_test > 0) & (merged.sharpe_full > 0)].copy()

    # Rank by combined score: test Sharpe (primary) + 0.3 * full Sharpe (tiebreaker)
    both_pos['combined_score'] = both_pos['sharpe_test'] + 0.3 * both_pos['sharpe_full']
    both_pos = both_pos.sort_values('combined_score', ascending=False)

    print(f"  Models with {min_trades}+ trades on test: {len(test)}")
    print(f"  Models with {min_trades}+ trades on full: {len(full)}")
    print(f"  Positive Sharpe on test: {(test.sharpe > 0).sum()}")
    print(f"  Positive Sharpe on full: {(full.sharpe > 0).sum()}")
    print(f"  Positive on BOTH: {len(both_pos)}")

    if len(both_pos) > 0:
        print(f"\n  {'Rank':<5} {'Test':>7} {'Full':>7} {'TestRet':>8} {'FullRet':>8} {'TestDD':>6} {'FullDD':>6} {'Trades':>6}  Tag")
        print(f"  {'-'*85}")
        for i, (_, r) in enumerate(both_pos.head(20).iterrows(), 1):
            print(f"  {i:<5} {r.sharpe_test:>+7.2f} {r.sharpe_full:>+7.2f} "
                  f"{r.total_return_test:>+8.2f} {r.total_return_full:>+8.2f} "
                  f"{r.max_dd_test:>5.1f}% {r.max_dd_full:>5.1f}% "
                  f"{int(r.trades_test):>6}  {r.model_tag}")

    return both_pos


# ── Step 3: Inference param sweep on top N ────────────────────────

def step3_inference_sweep(model_dir: str, df_full: pd.DataFrame, train_ratio: float,
                          top_tags: list, out_csv: str) -> pd.DataFrame:
    """Sweep inference params on top models."""
    print(f"\n{'='*80}")
    print(f"  STEP 3: Inference param sweep on {len(top_tags)} top models")
    print(f"{'='*80}")

    # Inference param grid
    sweep_min_holds = [8, 16, 32, 55]
    sweep_cooldowns = [0, 2, 4]
    sweep_max_pos = [0.05, 0.10, 0.15]
    sweep_penalties = [0, 0.0001, 0.0002]
    sweep_sls = [0, 0.01, 0.02, 0.03, 0.05]
    sweep_tps = [0, 0.03, 0.05, 0.07, 0.10, 0.15, 0.2]

    df_test = get_split(df_full, 'test', train_ratio)

    total_combos = (len(sweep_min_holds) * len(sweep_cooldowns) *
                    len(sweep_max_pos) * len(sweep_penalties) *
                    len(sweep_sls) * len(sweep_tps))
    total_runs = len(top_tags) * total_combos
    print(f"  {len(top_tags)} models × {total_combos} param combos = {total_runs} evals")

    rows = []
    run_count = 0

    for tag_info in top_tags:
        tag = tag_info if isinstance(tag_info, str) else tag_info.get('model_tag', tag_info)
        tag_params = parse_tag_params(tag)

        # Find the .pt file
        models_dir = Path(model_dir)
        pt_file = None
        for a in ['dqn', 'double_dqn', 'dueling_dqn', 'a2c']:
            candidate = models_dir / f"{a}_{tag}.pt"
            if candidate.exists():
                pt_file = candidate
                algo = a
                break

        if pt_file is None:
            print(f"  SKIP {tag}: no .pt file found")
            continue

        # Load agent once
        state_dim = infer_state_dim(str(pt_file))
        feat_dim = state_dim - 3
        feature_cols = FEATURES_ALL[:feat_dim]

        agent = build_agent(algo, state_dim)
        agent.load(str(pt_file))

        for mh in sweep_min_holds:
            for cd in sweep_cooldowns:
                for mp in sweep_max_pos:
                    for pen in sweep_penalties:
                        for sl in sweep_sls:
                            for tp in sweep_tps:
                                run_count += 1

                                env_kwargs = dict(
                                    fee_rate=0.0004,
                                    max_position_frac=mp,
                                    stop_loss_pct=sl,
                                    take_profit_pct=tp,
                                    min_hold_steps=mh,
                                    cooldown_steps=cd,
                                    trade_penalty=pen,
                                    adx_threshold=tag_params['adx_threshold'],
                                )

                                try:
                                    metrics = eval_model(agent, df_test, feature_cols, env_kwargs)
                                    rows.append({
                                        'model_tag': tag,
                                        'algo': algo,
                                        'infer_min_hold': mh,
                                        'infer_cooldown': cd,
                                        'infer_max_pos': mp,
                                        'infer_penalty': pen,
                                        'infer_sl': sl,
                                        'infer_tp': tp,
                                        'sharpe': metrics.get('sharpe_ratio', 0),
                                        'total_return': metrics.get('total_return', 0),
                                        'max_dd': metrics.get('max_drawdown', 0),
                                        'trades': metrics.get('total_trades', 0),
                                        'win_rate': metrics.get('win_rate', 0),
                                    })
                                except Exception:
                                    pass

        if run_count % 100 == 0:
            print(f"  [{run_count}/{total_runs}] evaluated")

    print(f"  [{run_count}/{total_runs}] done")

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"  Saved {len(df_out)} rows -> {out_csv}")
    return df_out


# ── Step 4: Print final results ──────────────────────────────────

def step4_print_results(step1_csv: str, sweep_csv: str, min_trades: int = 30):
    """Print final leaderboards."""
    print(f"\n{'='*80}")
    print(f"  STEP 4: FINAL RESULTS")
    print(f"{'='*80}")

    # Step 1 results
    if os.path.exists(step1_csv):
        df1 = pd.read_csv(step1_csv)
        test_df = df1[(df1.split == 'test') & (df1.trades >= min_trades)].sort_values('sharpe', ascending=False)

        print(f"\n  --- TOP 20 TEST (trades >= {min_trades}) ---")
        print(f"  {'Rank':<5} {'Sharpe':>7} {'Ret%':>8} {'DD%':>6} {'Trades':>6} {'WR%':>5}  Tag")
        for i, (_, r) in enumerate(test_df.head(20).iterrows(), 1):
            print(f"  {i:<5} {r.sharpe:>+7.2f} {r.total_return:>+8.2f} {r.max_dd:>5.1f}% {int(r.trades):>6} {r.win_rate:>5.1f}  {r.model_tag}")

        pos_rate = (test_df.sharpe > 0).mean() * 100
        print(f"\n  Positive Sharpe: {(test_df.sharpe > 0).sum()}/{len(test_df)} ({pos_rate:.0f}%)")

    # Sweep results
    if os.path.exists(sweep_csv):
        df3 = pd.read_csv(sweep_csv)
        df3 = df3[df3.trades >= min_trades].sort_values('sharpe', ascending=False)

        print(f"\n  --- TOP 20 INFERENCE SWEEP (test, trades >= {min_trades}) ---")
        print(f"  {'Rank':<5} {'Sharpe':>7} {'Ret%':>8} {'DD%':>6} {'Trades':>6} {'MH':>4} {'CD':>3} {'MP':>5} {'Pen':>6}  Tag")
        for i, (_, r) in enumerate(df3.head(20).iterrows(), 1):
            print(f"  {i:<5} {r.sharpe:>+7.2f} {r.total_return:>+8.2f} {r.max_dd:>5.1f}% "
                  f"{int(r.trades):>6} {int(r.infer_min_hold):>4} {int(r.infer_cooldown):>3} "
                  f"{r.infer_max_pos:>5.2f} {r.infer_penalty:>6.4f}  {r.model_tag}")

        # Best per model
        print(f"\n  --- BEST INFERENCE PARAMS PER MODEL ---")
        best_per_model = df3.loc[df3.groupby('model_tag')['sharpe'].idxmax()]
        best_per_model = best_per_model.sort_values('sharpe', ascending=False)
        for _, r in best_per_model.head(10).iterrows():
            print(f"  Sharpe {r.sharpe:>+7.2f} | mh={int(r.infer_min_hold)} cd={int(r.infer_cooldown)} "
                  f"mp={r.infer_max_pos:.2f} pen={r.infer_penalty:.4f} | {r.model_tag}")

    print(f"\n  Rule baseline (OOS): Sharpe +1.25, Return +1.60%, DD 1.92%, Trades 22")


# ── Main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Comprehensive RL model evaluation pipeline')
    ap.add_argument('--model_dir', default='models')
    ap.add_argument('--data', default=os.path.join('data', 'SOL_USDT_15m.csv'))
    ap.add_argument('--train_ratio', type=float, default=0.8)
    ap.add_argument('--step', type=int, default=0, help='Run specific step (1-4), 0=all')
    ap.add_argument('--top_n', type=int, default=10, help='Number of top models for inference sweep')
    ap.add_argument('--min_trades', type=int, default=30)
    ap.add_argument('--out_dir', default='results')
    args = ap.parse_args()

    step1_csv = os.path.join(args.out_dir, 'eval_step1_test_full.csv')
    sweep_csv = os.path.join(args.out_dir, 'eval_step3_inference_sweep.csv')

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    run_all = args.step == 0

    # Load data once
    if run_all or args.step in (1, 2, 3):
        print("Loading data...")
        df_full = load_data(args.data)
        print(f"Loaded {len(df_full)} bars")

    # Step 1
    if run_all or args.step == 1:
        df_eval = step1_eval_all(args.model_dir, df_full, args.train_ratio, step1_csv)
    elif os.path.exists(step1_csv):
        df_eval = pd.read_csv(step1_csv)
    else:
        df_eval = None

    # Step 2
    if df_eval is not None and (run_all or args.step == 2):
        both_pos = step2_find_dual_positive(df_eval, min_trades=args.min_trades)
    else:
        both_pos = None

    # Step 3
    if run_all or args.step == 3:
        # Get top tags from step 2, or fall back to top test Sharpe
        if both_pos is not None and len(both_pos) > 0:
            top_tags = both_pos.head(args.top_n)['model_tag'].tolist()
            print(f"\n  Using top {len(top_tags)} dual-positive models for sweep")
        elif df_eval is not None:
            test_only = df_eval[(df_eval.split == 'test') & (df_eval.trades >= args.min_trades)]
            test_only = test_only.sort_values('sharpe', ascending=False)
            top_tags = test_only.head(args.top_n)['model_tag'].tolist()
            print(f"\n  No dual-positive models; using top {len(top_tags)} test models for sweep")
        else:
            print("ERROR: Need step 1 results first. Run with --step 1")
            return

        step3_inference_sweep(args.model_dir, df_full, args.train_ratio, top_tags, sweep_csv)

    # Step 4
    if run_all or args.step == 4:
        step4_print_results(step1_csv, sweep_csv, min_trades=args.min_trades)


if __name__ == '__main__':
    main()


