#!/usr/bin/env python3
"""
Parameter sweep runner for train_all_window.py.

Runs a grid over environment + learning parameters, trains Double DQN on TRAIN,
evaluates on VAL, and writes a CSV summary so you can pick the best config.

Features:
- Learning rate sweeping (--lr)
- Epsilon decay sweeping (--eps_decay)
- Multi-metric composite scoring (Sharpe, drawdown, trade count)
- Seed stability analysis (flags configs where seeds disagree)
- Parallel execution (--parallel N) to use all CPU cores
- Resume ON by default: previously completed runs are skipped.

Usage
-----
    # Serial (default)
    python scripts/param_sweep.py --episodes 120 --seeds 42 43 ...

    # Parallel — 8 jobs at once (use N = vCPUs / 2)
    python scripts/param_sweep.py --parallel 8 --episodes 120 --seeds 42 43 ...

    # Fine sweep -- narrow around winners
    python scripts/param_sweep.py --parallel 8 --episodes 200 --seeds 41 42 43 ...
"""
import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone


def fmt_tag(x):
    """Keep tags filesystem-friendly and stable."""
    return str(x).replace('-', 'm').replace('.', 'p')


def _fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# =====================================================================
#  Composite scoring
# =====================================================================

def composite_score(sharpe: float, max_dd: float, n_trades: int,
                    dd_penalty: float = 0.5,
                    trade_bonus: float = 0.2,
                    min_trades_floor: int = 10) -> float:
    """
    Multi-metric ranking score.

    score = sharpe - dd_penalty * (max_dd / 100) + trade_bonus * log(max(trades, floor))

    - Sharpe is the primary driver
    - Drawdown penalises risky equity curves
    - Trade count (log-scaled) rewards statistical significance
    """
    dd_term = dd_penalty * (max_dd / 100.0)
    trade_term = trade_bonus * math.log(max(n_trades, min_trades_floor))
    return sharpe - dd_term + trade_term


# =====================================================================
#  CSV persistence
# =====================================================================

FIELDNAMES = [
    'timestamp',
    'seed',
    'lr', 'eps_decay',
    'max_pos', 'min_hold', 'cooldown', 'trade_penalty',
    'model_tag', 'config_key',
    'val_return', 'val_sharpe', 'val_max_dd', 'val_trades', 'val_win_rate',
    'composite_score',
    'json_path',
    'run_seconds',
    'passes_constraints',
]


def _load_existing_csv(path: str):
    """Load an existing CSV. Returns (rows, by_tag, completed_tags, run_times)."""
    rows: list[dict] = []
    by_tag: dict[str, int] = {}
    completed_tags: set[str] = set()
    run_times: list[float] = []

    if not os.path.exists(path):
        return rows, by_tag, completed_tags, run_times

    with open(path, 'r', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            row = {k: row.get(k, '') for k in FIELDNAMES}

            for k in ('seed', 'min_hold', 'cooldown', 'val_trades', 'passes_constraints'):
                if row.get(k, '') not in ('', None):
                    try:
                        row[k] = int(float(row[k]))
                    except Exception:
                        pass

            for k in ('lr', 'eps_decay', 'max_pos', 'trade_penalty',
                      'val_return', 'val_sharpe', 'val_max_dd', 'val_win_rate',
                      'composite_score', 'run_seconds'):
                if row.get(k, '') not in ('', None):
                    try:
                        row[k] = float(row[k])
                    except Exception:
                        pass

            tag = row.get('model_tag') or ''
            if tag:
                completed_tags.add(tag)
                by_tag[tag] = len(rows)

            try:
                rs = float(row.get('run_seconds', 0.0))
                if rs > 0:
                    run_times.append(rs)
            except Exception:
                pass

            rows.append(row)

    return rows, by_tag, completed_tags, run_times


def _write_csv(path: str, rows: list[dict]):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in FIELDNAMES} for r in rows])


# =====================================================================
#  Single job execution (runs in subprocess, safe for parallel)
# =====================================================================

ALGO_JSON_KEYS = {
    'dqn': 'DQN',
    'double_dqn': 'Double DQN',
    'dueling_dqn': 'Dueling DQN',
    'a2c': 'A2C',
}


def _run_single_job(job: dict) -> dict:
    """
    Execute a single training + backtest run.

    Takes a job dict with all parameters, returns a result dict.
    This function is safe for parallel execution (no shared state).
    """
    cmd = job['cmd']
    model_tag = job['model_tag']
    out_json = job['out_json']
    model_path = job['model_path']
    algo = job['algo']
    resume = job['resume']
    retrain = job['retrain']

    # Determine mode
    mode = 'train'
    if resume and (not retrain):
        if os.path.exists(model_path) and not os.path.exists(out_json):
            mode = 'backtest'
        elif os.path.exists(model_path) and os.path.exists(out_json):
            mode = 'load'

    run_s = 0.0
    if mode == 'load':
        pass  # Just read metrics below
    else:
        run_cmd = cmd[:]
        if mode == 'backtest':
            run_cmd.insert(2, '--skip-train')

        t0 = time.time()
        # Suppress stdout in parallel mode to avoid interleaved output
        subprocess.run(
            run_cmd, check=True,
            stdout=subprocess.DEVNULL if job.get('quiet') else None,
            stderr=subprocess.STDOUT if job.get('quiet') else None,
        )
        run_s = time.time() - t0

    # Read metrics
    with open(out_json, 'r', encoding='utf-8') as f:
        results = json.load(f)

    algo_json_key = ALGO_JSON_KEYS.get(algo, algo)
    m = results.get(algo_json_key, {})

    val_ret = float(m.get('total_return', 0.0))
    val_shp = float(m.get('sharpe_ratio', 0.0))
    val_dd  = float(m.get('max_drawdown', 0.0))
    val_tr  = int(m.get('total_trades', 0))
    val_wr  = float(m.get('win_rate', 0.0))

    score = composite_score(
        val_shp, val_dd, val_tr,
        dd_penalty=job['dd_penalty'],
        trade_bonus=job['trade_bonus'],
    )

    passes = (val_dd <= job['max_dd']) and (val_tr >= job['min_trades'])

    return {
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'seed': job['seed'],
        'lr': job['lr'],
        'eps_decay': job['eps_decay'],
        'max_pos': job['max_pos'],
        'min_hold': job['min_hold'],
        'cooldown': job['cooldown'],
        'trade_penalty': job['trade_penalty'],
        'model_tag': model_tag,
        'config_key': job['config_key'],
        'val_return': round(val_ret, 2),
        'val_sharpe': round(val_shp, 3),
        'val_max_dd': round(val_dd, 2),
        'val_trades': val_tr,
        'val_win_rate': round(val_wr, 2),
        'composite_score': round(score, 4),
        'json_path': out_json,
        'run_seconds': round(run_s, 2),
        'passes_constraints': int(passes),
        'mode': mode,
    }


# =====================================================================
#  Seed stability analysis
# =====================================================================

def analyze_seed_stability(rows: list[dict], min_seeds: int = 2):
    """
    Group runs by config_key, compute mean/std of Sharpe across seeds.
    Returns sorted list of configs with stability metrics.
    """
    configs = defaultdict(list)

    for r in rows:
        ck = r.get('config_key', '')
        if not ck:
            continue
        try:
            sharpe = float(r.get('val_sharpe', 0))
            score = float(r.get('composite_score', 0))
            ret = float(r.get('val_return', 0))
        except (ValueError, TypeError):
            continue
        configs[ck].append(dict(sharpe=sharpe, score=score, ret=ret, row=r))

    results = []
    for ck, runs in configs.items():
        if len(runs) < min_seeds:
            continue
        sharpes = [r['sharpe'] for r in runs]
        scores = [r['score'] for r in runs]
        rets = [r['ret'] for r in runs]

        mean_sharpe = sum(sharpes) / len(sharpes)
        std_sharpe = (sum((s - mean_sharpe)**2 for s in sharpes) / len(sharpes)) ** 0.5
        mean_score = sum(scores) / len(scores)
        mean_ret = sum(rets) / len(rets)
        all_positive = all(s > 0 for s in sharpes)

        stability = mean_sharpe / (std_sharpe + 0.1)

        results.append(dict(
            config_key=ck,
            n_seeds=len(runs),
            mean_sharpe=round(mean_sharpe, 3),
            std_sharpe=round(std_sharpe, 3),
            stability=round(stability, 3),
            mean_score=round(mean_score, 3),
            mean_return=round(mean_ret, 2),
            all_seeds_positive=all_positive,
            sharpes=[round(s, 3) for s in sharpes],
            sample_row=runs[0]['row'],
        ))

    results.sort(key=lambda x: x['stability'], reverse=True)
    return results


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--train_script', default=os.path.join('scripts', 'train_all_window.py'))
    ap.add_argument('--data', default=os.path.join('data', 'SOL_USDT_15m.csv'))

    ap.add_argument('--episodes', type=int, default=120)
    ap.add_argument('--log_every', type=int, default=0)
    ap.add_argument('--window', type=int, default=2000)
    ap.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43])

    ap.add_argument('--train_ratio', type=float, default=0.6)
    ap.add_argument('--val_ratio',   type=float, default=0.2)

    # -- Grid parameters --
    ap.add_argument('--lr',            nargs='+', type=float, default=[5e-5, 1e-4, 3e-4])
    ap.add_argument('--eps_decay',     nargs='+', type=float, default=[0.99997])
    ap.add_argument('--max_pos',       nargs='+', type=float, default=[0.15, 0.2])
    ap.add_argument('--min_hold',      nargs='+', type=int,   default=[8, 16])
    ap.add_argument('--cooldown',      nargs='+', type=int,   default=[2, 4])
    ap.add_argument('--trade_penalty', nargs='+', type=float, default=[0.0001, 0.0002])

    # Fixed env params
    ap.add_argument('--fee', type=float, default=0.0004)
    ap.add_argument('--sl',  type=float, default=0.0)
    ap.add_argument('--tp',  type=float, default=0.0)

    # Selection constraints
    ap.add_argument('--max_dd',     type=float, default=80.0)
    ap.add_argument('--min_trades', type=int,   default=50)

    # Composite score weights
    ap.add_argument('--dd_penalty',    type=float, default=0.5)
    ap.add_argument('--trade_bonus',   type=float, default=0.2)

    ap.add_argument('--out_csv', default=os.path.join('results', 'param_sweep_double_dqn.csv'))
    ap.add_argument('--out_dir', default=os.path.join('results', 'param_sweep_runs'))

    ap.add_argument('--algo', default='double_dqn',
                    choices=['dqn', 'double_dqn', 'dueling_dqn', 'a2c'])

    # Parallelism
    ap.add_argument('--parallel', type=int, default=1,
                    help='Number of parallel jobs (default 1 = serial). '
                         'Recommended: vCPUs / 2 (e.g. 8 for 16-core machine)')

    # Resume / retrain
    ap.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--retrain', '--force', dest='retrain', action='store_true')

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Build the full grid
    combos = list(itertools.product(
        args.lr, args.eps_decay,
        args.max_pos, args.min_hold, args.cooldown, args.trade_penalty
    ))
    total_runs = len(combos) * len(args.seeds)

    rows, by_tag, completed_tags, run_times = ([], {}, set(), [])

    if args.resume and not args.retrain and os.path.exists(args.out_csv):
        rows, by_tag, completed_tags, run_times = _load_existing_csv(args.out_csv)
        if rows:
            print(f"Resume: loaded {len(rows)} completed runs from {args.out_csv}")

    n_params = (f"lr={len(args.lr)} x eps_decay={len(args.eps_decay)} x "
                f"max_pos={len(args.max_pos)} x min_hold={len(args.min_hold)} x "
                f"cooldown={len(args.cooldown)} x penalty={len(args.trade_penalty)}")
    print(f"\nGrid: {n_params}")
    print(f"Combos: {len(combos)} | Seeds: {len(args.seeds)} | Total runs: {total_runs}")

    # ── Build job list ──
    jobs = []
    skipped = 0
    for (lr, eps_decay, max_pos, min_hold, cooldown, trade_penalty) in combos:
        config_key = (
            f"lr{fmt_tag(lr)}_ed{fmt_tag(eps_decay)}"
            f"_mp{fmt_tag(max_pos)}_mh{fmt_tag(min_hold)}"
            f"_cd{fmt_tag(cooldown)}_p{fmt_tag(trade_penalty)}"
        )

        for seed in args.seeds:
            model_tag = f"sweep_seed{seed}_{config_key}_sl{fmt_tag(args.sl)}_tp{fmt_tag(args.tp)}"
            out_json = os.path.join(args.out_dir, f"val_{model_tag}.json")
            model_path = os.path.join('models', f"{args.algo}_{model_tag}.pt")

            # Skip if already done
            if args.resume and (not args.retrain) and (model_tag in completed_tags) and os.path.exists(out_json):
                skipped += 1
                continue

            cmd = [
                args.python, args.train_script,
                '--data', args.data,
                '--episodes', str(args.episodes),
                '--window', str(args.window),
                '--seed', str(seed),
                '--train_ratio', str(args.train_ratio),
                '--val_ratio', str(args.val_ratio),
                '--algo', args.algo,
                '--eval', 'val',
                '--no_public_copy',
                '--lr', str(lr),
                '--eps_decay', str(eps_decay),
                '--fee', str(args.fee),
                '--max_pos', str(max_pos),
                '--sl', str(args.sl),
                '--tp', str(args.tp),
                '--min_hold', str(min_hold),
                '--cooldown', str(cooldown),
                '--trade_penalty', str(trade_penalty),
                '--model_tag', model_tag,
                '--log_every', str(args.log_every),
                '--output', out_json,
            ]

            jobs.append({
                'cmd': cmd,
                'model_tag': model_tag,
                'config_key': config_key,
                'out_json': out_json,
                'model_path': model_path,
                'algo': args.algo,
                'seed': seed,
                'lr': lr,
                'eps_decay': eps_decay,
                'max_pos': max_pos,
                'min_hold': min_hold,
                'cooldown': cooldown,
                'trade_penalty': trade_penalty,
                'resume': args.resume,
                'retrain': args.retrain,
                'dd_penalty': args.dd_penalty,
                'trade_bonus': args.trade_bonus,
                'max_dd': args.max_dd,
                'min_trades': args.min_trades,
                'quiet': args.parallel > 1,  # suppress subprocess output in parallel
            })

    remaining = len(jobs)
    if skipped:
        print(f"Skipped (already done): {skipped}")
    print(f"Jobs to run: {remaining}")
    if args.parallel > 1:
        print(f"Parallel workers: {args.parallel}")
    if run_times and remaining > 0:
        avg_s = sum(run_times) / len(run_times)
        if args.parallel > 1:
            est = avg_s * remaining / args.parallel
        else:
            est = avg_s * remaining
        print(f"Estimated time: {_fmt_time(est)}")
    print()

    if not jobs:
        print("All runs already completed. Printing summary...\n")
    else:
        # ── Execute jobs ──
        completed = len(completed_tags)
        csv_lock = threading.Lock()

        def _record_result(result: dict):
            """Thread-safe: update rows/CSV after a job completes."""
            nonlocal completed
            with csv_lock:
                tag = result['model_tag']
                mode = result.pop('mode', 'train')

                if tag in by_tag:
                    rows[by_tag[tag]] = result
                else:
                    by_tag[tag] = len(rows)
                    rows.append(result)

                completed_tags.add(tag)
                run_s = float(result.get('run_seconds', 0))
                if run_s > 0:
                    run_times.append(run_s)
                completed = len(completed_tags)

                avg_s = (sum(run_times) / len(run_times)) if run_times else 0.0
                rem = total_runs - completed
                if args.parallel > 1:
                    eta_s = avg_s * rem / args.parallel
                else:
                    eta_s = avg_s * rem
                eta_str = _fmt_time(eta_s) if eta_s > 0 else '?'

                val_shp = result.get('val_sharpe', 0)
                score = result.get('composite_score', 0)
                val_tr = result.get('val_trades', 0)
                val_dd = result.get('val_max_dd', 0)

                print(f"  [{completed}/{total_runs}]  {mode:<9} sharpe={val_shp:+.3f}  "
                      f"score={score:.3f}  trades={val_tr}  dd={val_dd:.1f}%  "
                      f"run={run_s:.1f}s  ETA={eta_str}")

                _write_csv(args.out_csv, rows)

        if args.parallel <= 1:
            # ── Serial execution ──
            for job in jobs:
                job['quiet'] = False  # show output in serial mode
                lr = job['lr']
                max_pos = job['max_pos']
                min_hold = job['min_hold']
                cooldown = job['cooldown']
                trade_penalty = job['trade_penalty']
                print(f"\nTRAIN  {job['model_tag']}")
                print(f"  lr={lr}  max_pos={max_pos}  min_hold={min_hold}  "
                      f"cooldown={cooldown}  penalty={trade_penalty}")
                result = _run_single_job(job)
                _record_result(result)
        else:
            # ── Parallel execution ──
            print(f"Starting {args.parallel} parallel workers...\n")
            with ProcessPoolExecutor(max_workers=args.parallel) as executor:
                future_to_tag = {}
                for job in jobs:
                    future = executor.submit(_run_single_job, job)
                    future_to_tag[future] = job['model_tag']

                for future in as_completed(future_to_tag):
                    tag = future_to_tag[future]
                    try:
                        result = future.result()
                        _record_result(result)
                    except Exception as e:
                        print(f"  FAILED  {tag}: {e}")

    # =================================================================
    #  Summary
    # =================================================================

    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE -- {len(rows)} total runs")
    print(f"{'='*70}")

    ok = [r for r in rows if int(r.get('passes_constraints', 0)) == 1]
    print(f"\n  Runs passing constraints (dd<={args.max_dd}%, trades>={args.min_trades}): {len(ok)}/{len(rows)}")

    if ok:
        top = sorted(ok, key=lambda r: float(r.get('composite_score', 0)), reverse=True)[:10]
        print(f"\n  TOP 10 by composite score (Sharpe - {args.dd_penalty}*DD + {args.trade_bonus}*log(trades)):")
        print(f"  {'Rank':<5} {'Score':>7} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} "
              f"{'Trades':>7} {'LR':>10} {'MaxPos':>7} {'MinH':>5} {'CD':>4} {'Penalty':>9} {'Seed':>5}")
        for i, r in enumerate(top, 1):
            print(f"  {i:<5} {float(r['composite_score']):>7.3f} {float(r['val_sharpe']):>+7.3f} "
                  f"{float(r['val_return']):>+8.2f}% {float(r['val_max_dd']):>6.1f}% "
                  f"{int(r['val_trades']):>7} {float(r['lr']):>10.1e} {float(r['max_pos']):>7.2f} "
                  f"{int(r['min_hold']):>5} {int(r['cooldown']):>4} {float(r['trade_penalty']):>9.5f} "
                  f"{int(r['seed']):>5}")

    # -- Seed stability --
    stability = analyze_seed_stability(rows, min_seeds=len(args.seeds))
    if stability:
        print(f"\n  SEED STABILITY (top 10 most stable configs, {len(args.seeds)} seeds each):")
        print(f"  {'Rank':<5} {'Stability':>10} {'MeanSharpe':>11} {'StdSharpe':>10} "
              f"{'AllPos':>7} {'MeanScore':>10} {'Sharpes'}")
        for i, s in enumerate(stability[:10], 1):
            all_pos = 'YES' if s['all_seeds_positive'] else 'NO'
            print(f"  {i:<5} {s['stability']:>10.3f} {s['mean_sharpe']:>+11.3f} "
                  f"{s['std_sharpe']:>10.3f} {all_pos:>7} {s['mean_score']:>10.3f} "
                  f"{s['sharpes']}")

        if ok:
            best_tag = max(ok, key=lambda r: float(r.get('composite_score', 0))).get('config_key', '')
            for s in stability:
                if s['config_key'] == best_tag:
                    if s['std_sharpe'] > 0.5 or not s['all_seeds_positive']:
                        print(f"\n  WARNING: Best config '{best_tag}' has unstable seeds!")
                        print(f"    Sharpes across seeds: {s['sharpes']}")
                        print(f"    Consider picking a more stable alternative.")
                    else:
                        print(f"\n  OK: Best config '{best_tag}' is stable across seeds.")
                    break

    elif not ok:
        best = max(rows, key=lambda r: float(r.get('composite_score', 0))) if rows else None
        if best:
            print(f"\n  No run passed constraints. Best overall:")
            print(f"    {best.get('model_tag')}  score={best.get('composite_score')}  "
                  f"sharpe={best.get('val_sharpe')}  dd={best.get('val_max_dd')}%")

    print(f"\n  CSV: {args.out_csv}")
    print(f"  Run JSONs: {args.out_dir}/")
    print()


if __name__ == '__main__':
    main()