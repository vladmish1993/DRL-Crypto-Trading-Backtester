#!/usr/bin/env python3
"""
run_grid.py — Multi-process RL training grid runner.

Spawns N parallel training processes across a full parameter grid.
Resume-friendly: skips runs whose output JSON already exists.

Usage
-----
    # 4 parallel workers (default)
    python scripts/run_grid.py --workers 4

    # 8 workers on beefy AWS instance
    python scripts/run_grid.py --workers 8

    # Dry run (print commands, don't execute)
    python scripts/run_grid.py --dry_run

    # Custom output dir
    python scripts/run_grid.py --workers 6 --out_dir results/rl_full_grid
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
#  GRID DEFINITION — edit these to change your sweep
# ═══════════════════════════════════════════════════════════════════

GRID = dict(
    # Top 3 screen winners (eps_decay, penalty pairs)
    ed_pen=[
        (0.9999,  0.0004),
        (0.99997, 0.0001),
        (0.99997, 0.0004),
    ],
    min_hold=[16, 32, 64],
    cooldown=[0, 4],
    adx_threshold=[0, 28],
    window=[4000, 8000],
    seed=[42, 123, 456],
)

# Fixed params (not swept)
FIXED = dict(
    data="data/SOL_USDT_15m.csv",
    train_ratio=0.8,
    val_ratio=0.0,
    eval="full",
    episodes=300,
    algo="dqn",
    fee=0.0004,
    max_pos=0.10,
    sl=0.02,
    tp=0.07,
    log_every=300,
    no_public_copy=True,
)


def build_jobs(out_dir: str):
    """Generate all (tag, cmd, output_path) tuples from the grid."""
    jobs = []

    ed_pen_list = GRID['ed_pen']
    for (ed, pen), mh, cd, adx, win, seed in itertools.product(
        ed_pen_list,
        GRID['min_hold'],
        GRID['cooldown'],
        GRID['adx_threshold'],
        GRID['window'],
        GRID['seed'],
    ):
        ed_str = str(ed).replace('.', 'p')
        pen_str = str(pen).replace('.', 'p')

        tag = f"g_mh{mh}_cd{cd}_adx{adx}_ed{ed_str}_pen{pen_str}_win{win}_seed{seed}"
        out_path = os.path.join(out_dir, f"{tag}.json")

        cmd = [
            sys.executable, "scripts/train_all_window.py",
            "--data", FIXED['data'],
            "--train_ratio", str(FIXED['train_ratio']),
            "--val_ratio", str(FIXED['val_ratio']),
            "--eval", FIXED['eval'],
            "--episodes", str(FIXED['episodes']),
            "--window", str(win),
            "--seed", str(seed),
            "--algo", FIXED['algo'],
            "--fee", str(FIXED['fee']),
            "--max_pos", str(FIXED['max_pos']),
            "--sl", str(FIXED['sl']),
            "--tp", str(FIXED['tp']),
            "--min_hold", str(mh),
            "--cooldown", str(cd),
            "--trade_penalty", str(pen),
            "--eps_decay", str(ed),
            "--adx_threshold", str(adx),
            "--log_every", str(FIXED['log_every']),
            "--model_tag", tag,
            "--output", out_path,
        ]
        if FIXED.get('no_public_copy'):
            cmd.append("--no_public_copy")

        jobs.append((tag, cmd, out_path))

    return jobs


def run_one(tag, cmd, out_path):
    """Run a single training job. Returns (tag, success, elapsed, error_msg)."""
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per run
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            return (tag, False, elapsed, result.stderr[-500:] if result.stderr else "unknown error")

        return (tag, True, elapsed, "")

    except subprocess.TimeoutExpired:
        return (tag, False, time.time() - t0, "TIMEOUT (2h)")
    except Exception as e:
        return (tag, False, time.time() - t0, str(e))


def main():
    ap = argparse.ArgumentParser(description="Multi-process RL training grid")
    ap.add_argument('--workers', type=int, default=4, help='Number of parallel training processes')
    ap.add_argument('--out_dir', default='results/rl_full_grid', help='Output directory for JSON results')
    ap.add_argument('--dry_run', action='store_true', help='Print jobs without running')
    ap.add_argument('--log', default='results/rl_full_grid/grid_log.txt', help='Log file path')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build all jobs
    all_jobs = build_jobs(args.out_dir)
    total = len(all_jobs)
    print(f"Grid total: {total} runs")

    # Filter out already-completed jobs (resume)
    pending = [(tag, cmd, out) for tag, cmd, out in all_jobs if not os.path.exists(out)]
    skipped = total - len(pending)
    if skipped:
        print(f"Skipping {skipped} completed runs (resume)")
    print(f"Pending: {len(pending)} runs with {args.workers} workers")

    if args.dry_run:
        print("\n--- DRY RUN (first 5 commands) ---")
        for tag, cmd, out in pending[:5]:
            print(f"\n  {tag}")
            print(f"  {' '.join(cmd)}")
        print(f"\n... and {max(0, len(pending)-5)} more")
        return

    if not pending:
        print("Nothing to do — all runs complete!")
        _print_summary(args.out_dir)
        return

    # Open log file
    log_path = args.log
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    log_f = open(log_path, 'a')

    def log(msg):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line)
        log_f.write(line + '\n')
        log_f.flush()

    log(f"Starting grid: {len(pending)} pending / {total} total / {args.workers} workers")
    t_start = time.time()
    done = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for tag, cmd, out in pending:
            f = pool.submit(run_one, tag, cmd, out)
            futures[f] = tag

        for f in as_completed(futures):
            tag, success, elapsed, err = f.result()
            done += 1

            if success:
                log(f"[{done}/{len(pending)}] OK  {tag}  ({elapsed:.0f}s)")
            else:
                failed += 1
                log(f"[{done}/{len(pending)}] FAIL {tag}  ({elapsed:.0f}s)  {err[:200]}")

            # ETA
            avg_per_job = (time.time() - t_start) / done
            remaining = len(pending) - done
            eta_s = avg_per_job * remaining / max(1, args.workers)
            eta_h = eta_s / 3600
            log(f"  ETA: {eta_h:.1f}h remaining ({remaining} jobs)")

    total_time = time.time() - t_start
    log(f"\nGrid complete: {done} runs in {total_time/3600:.1f}h ({failed} failures)")
    log_f.close()

    _print_summary(args.out_dir)


def _print_summary(out_dir: str):
    """Print a quick leaderboard from completed JSONs."""
    import glob

    rows = []
    for fn in sorted(glob.glob(os.path.join(out_dir, 'g_*.json'))):
        tag = os.path.basename(fn).replace('.json', '')
        try:
            with open(fn) as f:
                data = json.load(f)
        except Exception:
            continue

        for split_name in ('test', 'full'):
            d = data.get(split_name, data)
            for algo, m in d.items():
                if algo == 'Buy & Hold':
                    continue
                if not isinstance(m, dict):
                    continue
                rows.append({
                    'tag': tag, 'split': split_name,
                    'sharpe': m.get('sharpe_ratio', 0),
                    'ret': m.get('total_return', 0),
                    'dd': m.get('max_drawdown', 0),
                    'trades': m.get('total_trades', 0),
                    'wr': m.get('win_rate', 0),
                })

    if not rows:
        print("No results found yet.")
        return

    # Top 20 TEST
    test_rows = sorted([r for r in rows if r['split'] == 'test'], key=lambda r: -r['sharpe'])
    print(f"\n{'='*80}")
    print(f"  TOP 20 TEST RESULTS ({len(test_rows)} total)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Sharpe':>7} {'Ret%':>8} {'DD%':>6} {'Trades':>6} {'WR%':>5}  Tag")
    for i, r in enumerate(test_rows[:20], 1):
        print(f"{i:<5} {r['sharpe']:>+7.2f} {r['ret']:>+8.2f} {r['dd']:>5.1f}% {r['trades']:>6} {r['wr']:>5.1f}  {r['tag']}")

    # Top 20 FULL
    full_rows = sorted([r for r in rows if r['split'] == 'full'], key=lambda r: -r['sharpe'])
    print(f"\n{'='*80}")
    print(f"  TOP 20 FULL RESULTS ({len(full_rows)} total)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Sharpe':>7} {'Ret%':>8} {'DD%':>6} {'Trades':>6} {'WR%':>5}  Tag")
    for i, r in enumerate(full_rows[:20], 1):
        print(f"{i:<5} {r['sharpe']:>+7.2f} {r['ret']:>+8.2f} {r['dd']:>5.1f}% {r['trades']:>6} {r['wr']:>5.1f}  {r['tag']}")

    print(f"\nRule baseline (OOS): Sharpe +1.25, Return +1.60%, DD 1.92%, Trades 22")
    print(f"Current best DQN (OOS): Sharpe +1.97, Return +3.67%, DD 2.01%, Trades 125")


if __name__ == '__main__':
    main()