#!/usr/bin/env python3
"""
run_rl_grid.py — Multi-process RL training grid runner (resume-friendly + per-run logs).

Fixes vs previous version:
- Uses absolute paths (no working-directory surprises on servers / systemd).
- Preflight checks for data and train script existence.
- Writes per-run log files (stdout+stderr) so failures are diagnosable.
- Grid log keeps short status lines + points you to the per-run log.
- Resume-friendly: skips runs whose output JSON already exists.
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

# ─────────────────────────────────────────────────────────────────────
#  GRID DEFINITION — edit these to change your sweep
# ─────────────────────────────────────────────────────────────────────
GRID = dict(
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
    eval="test",
    episodes=300,
    algo="dqn",
    fee=0.0004,
    max_pos=0.10,
    sl=0.02,
    tp=0.07,
    log_every=30,
    no_public_copy=True,
)

RUN_TIMEOUT_S = 7200  # 2h per job


def tail_text(path: Path, max_lines: int = 60) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except Exception:
        return ""


def build_jobs(root: Path, out_dir: Path, train_script: Path, data_path: Path):
    jobs = []
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for (ed, pen), mh, cd, adx, win, seed in itertools.product(
        GRID["ed_pen"],
        GRID["min_hold"],
        GRID["cooldown"],
        GRID["adx_threshold"],
        GRID["window"],
        GRID["seed"],
    ):
        ed_str = str(ed).replace(".", "p")
        pen_str = str(pen).replace(".", "p")

        tag = f"g_mh{mh}_cd{cd}_adx{adx}_ed{ed_str}_pen{pen_str}_win{win}_seed{seed}"
        out_path = out_dir / f"{tag}.json"
        run_log = log_dir / f"{tag}.log"

        cmd = [
            sys.executable, str(train_script),
            "--data", str(data_path),
            "--train_ratio", str(FIXED["train_ratio"]),
            "--val_ratio", str(FIXED["val_ratio"]),
            "--eval", FIXED["eval"],
            "--episodes", str(FIXED["episodes"]),
            "--window", str(win),
            "--seed", str(seed),
            "--algo", FIXED["algo"],
            "--fee", str(FIXED["fee"]),
            "--max_pos", str(FIXED["max_pos"]),
            "--sl", str(FIXED["sl"]),
            "--tp", str(FIXED["tp"]),
            "--min_hold", str(mh),
            "--cooldown", str(cd),
            "--trade_penalty", str(pen),
            "--eps_decay", str(ed),
            "--adx_threshold", str(adx),
            "--log_every", str(FIXED["log_every"]),
            "--model_tag", tag,
            "--output", str(out_path),
        ]
        if FIXED.get("no_public_copy"):
            cmd.append("--no_public_copy")

        jobs.append((tag, cmd, out_path, run_log))

    return jobs


def run_one(tag: str, cmd: list[str], out_path: str, run_log_path: str):
    t0 = time.time()
    run_log = Path(run_log_path)
    try:
        run_log.parent.mkdir(parents=True, exist_ok=True)
        with run_log.open("a", encoding="utf-8") as lf:
            lf.write("\n" + "=" * 100 + "\n")
            lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {tag}\n")
            lf.write("CMD: " + " ".join(cmd) + "\n\n")
            lf.flush()

            result = subprocess.run(
                cmd,
                stdout=lf,
                stderr=lf,
                text=True,
                timeout=RUN_TIMEOUT_S,
            )

        elapsed = time.time() - t0

        if result.returncode != 0:
            hint = tail_text(run_log, 40).strip()
            if not hint:
                hint = f"non-zero exit code {result.returncode}"
            return (tag, False, elapsed, hint, run_log_path)

        try:
            if (not os.path.exists(out_path)) or (os.path.getsize(out_path) == 0):
                return (tag, False, elapsed, "job exited 0 but output JSON missing/empty", run_log_path)
        except Exception:
            pass

        return (tag, True, elapsed, "", run_log_path)

    except subprocess.TimeoutExpired:
        return (tag, False, time.time() - t0, f"TIMEOUT ({RUN_TIMEOUT_S}s)", run_log_path)
    except Exception as e:
        return (tag, False, time.time() - t0, str(e), run_log_path)


def _print_summary(out_dir: Path):
    import glob
    rows = []
    for fn in sorted(glob.glob(str(out_dir / "g_*.json"))):
        tag = Path(fn).stem
        try:
            with open(fn, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        for split_name in ("test", "full"):
            d = data.get(split_name, data)
            if not isinstance(d, dict):
                continue
            for algo, m in d.items():
                if algo == "Buy & Hold" or not isinstance(m, dict):
                    continue
                rows.append({
                    "tag": tag,
                    "split": split_name,
                    "algo": algo,
                    "sharpe": m.get("sharpe_ratio", 0),
                    "ret": m.get("total_return", 0),
                    "dd": m.get("max_drawdown", 0),
                    "trades": m.get("total_trades", 0),
                    "wr": m.get("win_rate", 0),
                })

    if not rows:
        print("No results found yet.")
        return

    def show(split: str):
        srows = [r for r in rows if r["split"] == split]
        if not srows:
            return
        srows = sorted(srows, key=lambda r: -float(r["sharpe"]))
        print(f"\n{'='*88}")
        print(f" TOP 20 {split.upper()} RESULTS ({len(srows)} rows incl algos)")
        print(f"{'='*88}")
        print(f"{'Rank':<5} {'Sharpe':>7} {'Ret%':>8} {'DD%':>6} {'Trades':>6} {'WR%':>6}  Algo  Tag")
        for i, r in enumerate(srows[:20], 1):
            print(f"{i:<5} {float(r['sharpe']):>+7.2f} {float(r['ret']):>+8.2f} {float(r['dd']):>5.1f}% "
                  f"{int(r['trades']):>6} {float(r['wr']):>5.1f}%  {r['algo']:<4}  {r['tag']}")

    show("test")
    show("full")


def main():
    ap = argparse.ArgumentParser(description="Multi-process RL training grid (resume-friendly)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_dir", default="results/rl_full_grid")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--log", default=None)
    ap.add_argument("--data", default=None, help="Override data csv path (absolute or relative to repo root)")
    ap.add_argument("--train_script", default=None, help="Override train script path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = (root / args.out_dir).resolve() if not os.path.isabs(args.out_dir) else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log) if args.log else (out_dir / "grid_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    train_script = Path(args.train_script).resolve() if args.train_script else (root / "scripts" / "train_all_window.py")
    data_path = Path(args.data).resolve() if args.data else (root / FIXED["data"])

    if not train_script.exists():
        print(f"ERROR: train script not found: {train_script}")
        sys.exit(2)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}")
        print("Tip: copy your CSV to data/ on the server, or pass --data /absolute/path/to.csv")
        sys.exit(2)

    all_jobs = build_jobs(root, out_dir, train_script, data_path)
    total = len(all_jobs)
    print(f"Grid total: {total} runs")

    pending = [(tag, cmd, out, run_log) for tag, cmd, out, run_log in all_jobs if not out.exists()]
    skipped = total - len(pending)
    if skipped:
        print(f"Skipping {skipped} completed runs (resume)")
    print(f"Pending: {len(pending)} runs with {args.workers} workers")

    if args.dry_run:
        print("\n--- DRY RUN (first 5 commands) ---")
        for tag, cmd, out, run_log in pending[:5]:
            print(f"\n  {tag}")
            print(f"  {' '.join(cmd)}")
            print(f"  out: {out}")
            print(f"  log: {run_log}")
        return

    if not pending:
        print("Nothing to do — all runs complete!")
        _print_summary(out_dir)
        return

    def grid_log(msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    grid_log(f"Starting grid: {len(pending)} pending / {total} total / {args.workers} workers")
    t_start = time.time()
    done = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for tag, cmd, out, run_log in pending:
            f = pool.submit(run_one, tag, cmd, str(out), str(run_log))
            futures[f] = tag

        for f in as_completed(futures):
            tag, success, elapsed, hint, run_log_path = f.result()
            done += 1

            if success:
                grid_log(f"[{done}/{len(pending)}] OK  {tag}  ({elapsed:.0f}s)")
            else:
                failed += 1
                grid_log(f"[{done}/{len(pending)}] FAIL {tag}  ({elapsed:.0f}s)  log={run_log_path}")
                if hint:
                    grid_log(f"  tail: {hint.splitlines()[-1][:240]}")

            avg_per_job = (time.time() - t_start) / done
            remaining = len(pending) - done
            eta_s = avg_per_job * remaining / max(1, args.workers)
            grid_log(f"  ETA: {eta_s/3600:.2f}h remaining ({remaining} jobs)")

    total_time = time.time() - t_start
    grid_log(f"\nGrid complete: {done} runs in {total_time/3600:.2f}h ({failed} failures)")
    _print_summary(out_dir)


if __name__ == "__main__":
    main()
