#!/usr/bin/env python3
"""
screen_rl_to_step1_csv.py

Reads JSON results from a folder and writes a CSV compatible with eval_comprehensive Step 1 output.

Supported JSON shapes:
A) { "test": { "Double DQN": {...metrics...}, "DQN": {...} }, "full": {...} }
B) { "DQN": {...metrics...}, "DoubleDQN": {...} }  (legacy)

Usage:
  python scripts/screen_rl_to_step1_csv.py --in_dir results/rl_screen --out_csv results/eval_step1.csv --flush_every 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULTS = {
    "min_hold": 16,
    "cooldown": 4,
    "adx_threshold": 0,
    "trade_penalty": 0.0004,
    "max_pos": 0.10,
    "sl": 0.02,
    "tp": 0.07,
}


def parse_float_token(tok: str) -> float:
    # "0p0004" -> 0.0004, "0" -> 0.0
    return float(tok.replace("p", "."))


def _tok_or_default(stem: str, key: str, default):
    # allow both "sl0" and "sl0p02"
    m = re.search(rf"(?:^|_)({key})(\d+(?:p\d+)?)", stem)
    if not m:
        return default
    return parse_float_token(m.group(2))


def parse_params_from_stem(stem: str) -> Dict[str, object]:
    d: Dict[str, object] = {}

    m = re.search(r"(?:^|_)mh(\d+)", stem)
    d["min_hold"] = int(m.group(1)) if m else DEFAULTS["min_hold"]

    m = re.search(r"(?:^|_)cd(\d+)", stem)
    d["cooldown"] = int(m.group(1)) if m else DEFAULTS["cooldown"]

    m = re.search(r"(?:^|_)adx(\d+)", stem)
    d["adx_threshold"] = int(m.group(1)) if m else DEFAULTS["adx_threshold"]

    # your tags use p0p0002, p0p0003 etc (not "pen...")
    m = re.search(r"(?:^|_)p(\d+(?:p\d+)?)", stem)
    d["trade_penalty"] = parse_float_token(m.group(1)) if m else DEFAULTS["trade_penalty"]

    d["max_pos"] = _tok_or_default(stem, "mp", DEFAULTS["max_pos"])
    d["sl"] = _tok_or_default(stem, "sl", DEFAULTS["sl"])
    d["tp"] = _tok_or_default(stem, "tp", DEFAULTS["tp"])

    return d


def normalise_algo_name(algo_key: str) -> str:
    a = (algo_key or "").strip().lower().replace(" ", "_").replace("-", "_")
    if a in {"dqn"}:
        return "dqn"
    if a in {"double_dqn", "doubledqn", "double"}:
        return "double_dqn"
    if a in {"dueling_dqn", "duelingdqn", "dueling"}:
        return "dueling_dqn"
    if a in {"a2c"}:
        return "a2c"
    return a


def atomic_append_rows(csv_path: str, rows: List[Dict], columns: List[str]) -> None:
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)

    if out.exists():
        try:
            old_df = pd.read_csv(out)
            df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception:
            df = new_df
    else:
        df = new_df

    for c in columns:
        if c not in df.columns:
            df[c] = ""
    extra = [c for c in df.columns if c not in columns]
    df = df[columns + extra]

    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)


def build_row(file_name: str, stem: str, split: str, algo: str, metrics: Dict, params: Dict[str, object]) -> Dict:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_file": file_name,
        "algo": algo,
        "model_tag": stem,
        "split": split,
        "sharpe": metrics.get("sharpe_ratio", 0),
        "total_return": metrics.get("total_return", 0),
        "max_dd": metrics.get("max_drawdown", 0),
        "trades": metrics.get("total_trades", 0),
        "win_rate": metrics.get("win_rate", 0),
        "final_balance": metrics.get("final_balance", 0),
        "train_min_hold": int(params.get("min_hold", DEFAULTS["min_hold"])),
        "train_cooldown": int(params.get("cooldown", DEFAULTS["cooldown"])),
        "train_max_pos": float(params.get("max_pos", DEFAULTS["max_pos"])),
        "train_penalty": float(params.get("trade_penalty", DEFAULTS["trade_penalty"])),
        "train_adx": float(params.get("adx_threshold", DEFAULTS["adx_threshold"])),
        "train_sl": float(params.get("sl", DEFAULTS["sl"])),
        "train_tp": float(params.get("tp", DEFAULTS["tp"])),
    }


STEP1_COLUMNS = [
    "timestamp",
    "model_file",
    "algo",
    "model_tag",
    "split",
    "sharpe",
    "total_return",
    "max_dd",
    "trades",
    "win_rate",
    "final_balance",
    "train_min_hold",
    "train_cooldown",
    "train_max_pos",
    "train_penalty",
    "train_adx",
    "train_sl",
    "train_tp",
]


def iter_rows_from_json(data: dict) -> List[Tuple[str, str, Dict]]:
    """
    Returns list of (split, algo_key, metrics_dict).
    Supports:
      A) { split: { algo: metrics } }
      B) { algo: metrics }  -> treated as split='test'
    """
    rows: List[Tuple[str, str, Dict]] = []

    # Shape A
    if any(k in data for k in ("test", "full", "val", "train")):
        for split, inner in data.items():
            if not isinstance(inner, dict):
                continue
            for algo_key, metrics in inner.items():
                if isinstance(metrics, dict):
                    rows.append((str(split), str(algo_key), metrics))
        return rows

    # Shape B (legacy)
    for algo_key, metrics in data.items():
        if isinstance(metrics, dict):
            rows.append(("test", str(algo_key), metrics))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=os.path.join("results", "rl_screen"))
    ap.add_argument("--out_csv", default=os.path.join("results", "eval_step1_test_full.csv"))
    ap.add_argument("--flush_every", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="Skip rows already present in out_csv")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No .json files found in {in_dir.resolve()}")

    done: set[Tuple[str, str, str]] = set()
    if args.resume and Path(args.out_csv).exists():
        try:
            df_old = pd.read_csv(args.out_csv)
            for _, r in df_old.iterrows():
                done.add((str(r.get("model_file", "")), str(r.get("split", "")), str(r.get("algo", ""))))
            print(f"Resume enabled: {len(done)} existing rows found in {args.out_csv}")
        except Exception:
            print("Resume enabled, but failed to read existing CSV; will re-write rows.")

    buffer: List[Dict] = []
    processed = 0
    written = 0

    for i, fp in enumerate(files, 1):
        stem = fp.stem
        params = parse_params_from_stem(stem)

        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[{i}/{len(files)}] FAIL read {fp.name}: {e}")
            continue

        if not isinstance(data, dict) or not data:
            print(f"[{i}/{len(files)}] SKIP {fp.name}: empty/invalid JSON")
            continue

        for split, algo_key, metrics in iter_rows_from_json(data):
            algo = normalise_algo_name(algo_key)
            if (fp.name, split, algo) in done:
                continue

            row = build_row(fp.name, stem, split, algo, metrics, params)
            buffer.append(row)
            done.add((fp.name, split, algo))
            processed += 1

            if buffer and len(buffer) >= max(1, args.flush_every):
                atomic_append_rows(args.out_csv, buffer, STEP1_COLUMNS)
                written += len(buffer)
                buffer.clear()

        if i % 50 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] scanned files (rows_written={written})")

    if buffer:
        atomic_append_rows(args.out_csv, buffer, STEP1_COLUMNS)
        written += len(buffer)

    print(f"Done. Processed rows: {processed}. Saved -> {args.out_csv}")


if __name__ == "__main__":
    main()