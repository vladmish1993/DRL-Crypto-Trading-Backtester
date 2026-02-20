#!/usr/bin/env python3
"""
Parameter sweep runner for train_all_window.py.

Runs a small grid over environment parameters, trains Double DQN on TRAIN only,
evaluates on VAL, and writes a CSV summary so you can pick the best run.

Notes
- This script calls train_all_window.py as a subprocess so it works the same
  way as your normal training run.
- It uses a unique model_tag per run so weights do not overwrite each other.

Example
    python scripts/param_sweep.py --episodes 120 --window 2000 --seeds 1 2 3 \
        --max_pos 0.15 0.2 --min_hold 8 16 --cooldown 2 4 --trade_penalty 0.0001 0.0002
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime


def fmt_tag(x):
    s = str(x).replace('-', 'm').replace('.', 'p')
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--train_script', default=os.path.join('scripts', 'train_all_window.py'))
    ap.add_argument('--data', default=os.path.join('data', 'SOL_USDT_15m.csv'))

    ap.add_argument('--episodes', type=int, default=120)
    ap.add_argument('--window', type=int, default=2000)
    ap.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43])

    ap.add_argument('--train_ratio', type=float, default=0.6)
    ap.add_argument('--val_ratio', type=float, default=0.2)

    # Grid parameters
    ap.add_argument('--max_pos', nargs='+', type=float, default=[0.15, 0.2])
    ap.add_argument('--min_hold', nargs='+', type=int,   default=[8, 16])
    ap.add_argument('--cooldown', nargs='+', type=int,   default=[2, 4])
    ap.add_argument('--trade_penalty', nargs='+', type=float, default=[0.0001, 0.0002])

    # Fixed env params for the sweep
    ap.add_argument('--fee', type=float, default=0.0004)
    ap.add_argument('--sl',  type=float, default=0.0)
    ap.add_argument('--tp',  type=float, default=0.0)

    # Selection constraints (applied after the run)
    ap.add_argument('--max_dd', type=float, default=80.0)
    ap.add_argument('--min_trades', type=int, default=50)

    ap.add_argument('--out_csv', default=os.path.join('results', 'param_sweep_double_dqn.csv'))
    ap.add_argument('--out_dir', default=os.path.join('results', 'param_sweep_runs'))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    fieldnames = [
        'timestamp',
        'seed',
        'max_pos', 'min_hold', 'cooldown', 'trade_penalty',
        'model_tag',
        'val_return', 'val_sharpe', 'val_max_dd', 'val_trades',
        'json_path',
        'passes_constraints',
    ]

    rows = []
    combos = list(itertools.product(args.max_pos, args.min_hold, args.cooldown, args.trade_penalty))

    print(f"Total combos: {len(combos)} | Seeds: {len(args.seeds)} | Total runs: {len(combos) * len(args.seeds)}")

    for (max_pos, min_hold, cooldown, trade_penalty) in combos:
        for seed in args.seeds:
            model_tag = (
                f"sweep_seed{seed}"
                f"_mp{fmt_tag(max_pos)}_mh{fmt_tag(min_hold)}_cd{fmt_tag(cooldown)}_p{fmt_tag(trade_penalty)}"
                f"_sl{fmt_tag(args.sl)}_tp{fmt_tag(args.tp)}"
            )

            out_json = os.path.join(args.out_dir, f"val_{model_tag}.json")

            cmd = [
                args.python, args.train_script,
                '--data', args.data,
                '--episodes', str(args.episodes),
                '--window', str(args.window),
                '--seed', str(seed),
                '--train_ratio', str(args.train_ratio),
                '--val_ratio', str(args.val_ratio),
                '--algo', 'double_dqn',
                '--eval', 'val',
                '--no_public_copy',
                '--fee', str(args.fee),
                '--max_pos', str(max_pos),
                '--sl', str(args.sl),
                '--tp', str(args.tp),
                '--min_hold', str(min_hold),
                '--cooldown', str(cooldown),
                '--trade_penalty', str(trade_penalty),
                '--model_tag', model_tag,
                '--output', out_json,
            ]

            print("\nRunning:", " ".join(cmd))
            subprocess.run(cmd, check=True)

            with open(out_json, 'r') as f:
                results = json.load(f)

            m = results.get('Double DQN', {})
            val_ret = float(m.get('total_return', 0.0))
            val_shp = float(m.get('sharpe_ratio', 0.0))
            val_dd  = float(m.get('max_drawdown', 0.0))
            val_tr  = int(m.get('total_trades', 0))

            passes = (val_dd <= args.max_dd) and (val_tr >= args.min_trades)

            row = {
                'timestamp': datetime.utcnow().isoformat(timespec='seconds'),
                'seed': seed,
                'max_pos': max_pos,
                'min_hold': min_hold,
                'cooldown': cooldown,
                'trade_penalty': trade_penalty,
                'model_tag': model_tag,
                'val_return': val_ret,
                'val_sharpe': val_shp,
                'val_max_dd': val_dd,
                'val_trades': val_tr,
                'json_path': out_json,
                'passes_constraints': int(passes),
            }
            rows.append(row)

            with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)

    # Print best run under constraints
    ok = [r for r in rows if r['passes_constraints'] == 1]
    if ok:
        best = max(ok, key=lambda r: r['val_sharpe'])
        print("\nBest by VAL Sharpe (with constraints):")
        print(best)
    else:
        best = max(rows, key=lambda r: r['val_sharpe'])
        print("\nNo run passed constraints, best by VAL Sharpe without constraints:")
        print(best)

    print(f"\nCSV written to: {args.out_csv}")


if __name__ == '__main__':
    main()
