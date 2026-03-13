"""
backtest_filters.py

Sweeps GMGN-style filter threshold combinations against historical token data
to find which settings best identify tokens that actually graduated.

Target label: reached_graduation (1 = hit bonding curve, 0 = died)

Strategy:
  1. Independent sweep  — find best value per filter in isolation (fast)
  2. Grid search        — sweep top N filters together (thorough)
  3. Output leaderboard — ranked configs by precision @ min recall

Usage:
    python backtest_filters.py --data merged_token_data.csv
    python backtest_filters.py --data merged_token_data.csv --mode grid --top-filters 5
    python backtest_filters.py --data merged_token_data.csv --mode grid --min-recall 0.10

Install:
    pip install pandas numpy itertools tqdm
"""

import argparse
import itertools
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# FILTER DEFINITIONS
#
# Each filter maps a CSV column to a GMGN config field.
# direction: "max" = column value must be BELOW threshold (bad if high)
#            "min" = column value must be ABOVE threshold (good if high)
# sweep:     list of candidate threshold values to test
# ============================================================

FILTERS = [

    # ── TIER 1 — highest ML feature importance ───────────────────────────────

    {
        "name":      "net_flow_excl_top10",
        "label":     "Net flow excl top10",
        "col":       "net_flow_excl_top10",
        "direction": "min",
        # Must be net positive — organic wallets buying > selling
        "sweep":     [0, 0.05, 0.1, 0.2, 0.5, 1.0],
        "gmgn_key":  "net_buy_24h",
    },
    {
        "name":      "volume_per_unique_buyer",
        "label":     "Volume per unique buyer",
        "col":       "volume_per_unique_buyer",
        "direction": "min",
        # Too low = dust bots; too high = whale concentration
        # sweet spot is moderate — sweep both directions
        "sweep":     [0.5, 1.0, 2.0, 5.0, 10.0],
        "gmgn_key":  "volume_24h",  # closest proxy
    },
    {
        "name":      "organic_buyer_pct",
        "label":     "Organic buyer %",
        "col":       "organic_buyer_pct",
        "direction": "min",
        "sweep":     [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "gmgn_key":  "bundler_rate",  # inverse
    },

    # ── TIER 2 — strong single-model signals ─────────────────────────────────

    {
        "name":      "top5_holder_pct",
        "label":     "Top 5 holder %",
        "col":       "top5_holder_pct",
        "direction": "max",
        # High concentration = whale trap = almost never graduates
        "sweep":     [20, 25, 30, 35, 40, 50],
        "gmgn_key":  "top_10_holder_rate",
    },
    {
        "name":      "top10_holder_pct",
        "label":     "Top 10 holder %",
        "col":       "top10_holder_pct",
        "direction": "max",
        "sweep":     [30, 35, 40, 45, 50, 60],
        "gmgn_key":  "top_10_holder_rate",
    },
    {
        "name":      "dev_sell_volume_5m",
        "label":     "Dev sell volume 5m",
        "col":       "dev_sell_volume_5m",
        "direction": "max",
        # Zero = dev hasn't dumped. Any positive = red flag.
        "sweep":     [0, 0.01, 0.1, 0.5, 1.0],
        "gmgn_key":  "creator_balance_rate",
    },
    {
        "name":      "bundler_pct_of_buyers_1m",
        "label":     "Bundler % of buyers 1m",
        "col":       "bundler_pct_of_buyers_1m",
        "direction": "max",
        "sweep":     [0.1, 0.15, 0.20, 0.25, 0.30, 0.40],
        "gmgn_key":  "bundler_rate",
    },
    {
        "name":      "sniper_count",
        "label":     "Sniper count",
        "col":       "sniper_count",
        "direction": "max",
        "sweep":     [5, 10, 15, 20, 30, 50],
        "gmgn_key":  "top70_sniper_hold_rate",
    },

    # ── TIER 3 — supporting signals ──────────────────────────────────────────

    {
        "name":      "fresh_wallet_pct",
        "label":     "Fresh wallet %",
        "col":       "fresh_wallet_pct",
        "direction": "max",
        # High = bot farm of newly created wallets
        "sweep":     [0.2, 0.3, 0.4, 0.5, 0.6],
        "gmgn_key":  "fresh_wallet_rate",
    },
    {
        "name":      "holders_at_5m",
        "label":     "Holders at 5m",
        "col":       "holders_at_5m",
        "direction": "min",
        # Minimum real holder base
        "sweep":     [20, 30, 40, 50, 75, 100],
        "gmgn_key":  "holders",
    },
    {
        "name":      "dev_self_buy_count",
        "label":     "Dev self-buy count",
        "col":       "dev_self_buy_count",
        "direction": "max",
        # Dev buying own token = wash trading / artificial volume
        "sweep":     [0, 1, 2, 3, 5],
        "gmgn_key":  "not_wash_trading",
    },
    {
        "name":      "manipulator_count",
        "label":     "Manipulator count",
        "col":       "manipulator_count",
        "direction": "max",
        # Wallets with >6 buys = price manipulation
        "sweep":     [0, 1, 2, 3, 5, 10],
        "gmgn_key":  "no_suspected_insider",
    },
    {
        "name":      "buyers_5m",
        "label":     "Buyers at 5m",
        "col":       "buyers_5m",
        "direction": "min",
        "sweep":     [10, 20, 30, 40, 50, 75],
        "gmgn_key":  "buys_24h",
    },
    {
        "name":      "liquidity",
        "label":     "Initial liquidity (USD)",
        "col":       "price_at_launch",
        "direction": "min",
        # price_at_launch is a proxy — higher = more liquidity seeded
        # values in SOL-equivalent, not USD, so thresholds are smaller
        "sweep":     [0.00001, 0.0001, 0.001, 0.005, 0.01],
        "gmgn_key":  "liquidity",
    },
]

FILTER_NAMES = [f["name"] for f in FILTERS]


# ============================================================
# SCORING
# ============================================================

def apply_filters(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Apply a threshold config to a dataframe.
    Returns a boolean mask of rows that pass all filters.
    config: {filter_name: threshold_value}  — None = filter disabled
    """
    mask = pd.Series(True, index=df.index)

    for f in FILTERS:
        threshold = config.get(f["name"])
        if threshold is None:
            continue
        col = f["col"]
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")

        if f["direction"] == "max":
            mask &= series <= threshold
        else:
            mask &= series >= threshold

    return mask


def score_config(df: pd.DataFrame, label: str, config: dict) -> dict:
    """Score a single filter config against the ground truth label."""
    mask      = apply_filters(df, config)
    filtered  = df[mask]
    n_total   = len(df)
    n_passed  = mask.sum()

    if n_passed == 0:
        return None

    n_positive    = df[label].sum()
    tp            = filtered[label].sum()
    precision     = tp / n_passed
    recall        = tp / n_positive if n_positive > 0 else 0
    f1            = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)
    pass_rate     = n_passed / n_total

    return {
        "n_passed":   int(n_passed),
        "pass_rate":  round(pass_rate * 100, 2),
        "precision":  round(precision * 100, 2),
        "recall":     round(recall * 100, 2),
        "f1":         round(f1 * 100, 2),
        "tp":         int(tp),
    }


# ============================================================
# MODE 1: INDEPENDENT SWEEP
# Optimise each filter independently, all others disabled.
# Fast — O(filters × values). Good for understanding each filter's impact.
# ============================================================

def independent_sweep(df: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  INDEPENDENT SWEEP")
    print(f"  Each filter tested in isolation — others disabled")
    print(f"{'='*60}")

    baseline_config = {f["name"]: None for f in FILTERS}
    baseline = score_config(df, label, baseline_config)
    print(f"\n  Baseline (no filters): "
          f"{baseline['n_passed']:,} tokens | "
          f"precision: {baseline['precision']:.1f}% | "
          f"recall: {baseline['recall']:.1f}%\n")

    rows = []
    for f in FILTERS:
        best_row = None
        print(f"  {f['label']:<35}", end="")

        for threshold in f["sweep"]:
            config = {f2["name"]: None for f2 in FILTERS}
            config[f["name"]] = threshold

            result = score_config(df, label, config)
            if result is None:
                continue

            result["filter"]    = f["name"]
            result["label"]     = f["label"]
            result["threshold"] = threshold
            result["direction"] = f["direction"]
            result["gmgn_key"]  = f["gmgn_key"]
            rows.append(result)

            if best_row is None or result["precision"] > best_row["precision"]:
                best_row = result

        if best_row:
            print(f"best threshold: {best_row['threshold']} → "
                  f"precision: {best_row['precision']:.1f}% "
                  f"({best_row['n_passed']:,} tokens, "
                  f"recall: {best_row['recall']:.1f}%)")
        else:
            print("no valid results")

    results = pd.DataFrame(rows)
    return results


# ============================================================
# MODE 2: GRID SEARCH
# Sweep combinations of the top N filters simultaneously.
# O(values^N) — keep N <= 6 for reasonable runtime.
# ============================================================

def grid_search(
    df: pd.DataFrame,
    label: str,
    top_n: int = 5,
    min_recall: float = 0.05,
    min_tokens: int = 10,
) -> pd.DataFrame:

    # First run independent sweep to rank filters by precision gain
    print(f"\n[GRID] Running independent sweep to select top {top_n} filters...")
    ind = independent_sweep(df, label)

    # For each filter pick the threshold with highest precision
    best_per_filter = (
        ind.sort_values("precision", ascending=False)
           .groupby("filter")
           .first()
           .reset_index()
    )
    # Rank by precision improvement
    baseline_config = {f["name"]: None for f in FILTERS}
    baseline_precision = score_config(df, label, baseline_config)["precision"]
    best_per_filter["precision_gain"] = best_per_filter["precision"] - baseline_precision
    best_per_filter = best_per_filter.sort_values("precision_gain", ascending=False)

    top_filters = best_per_filter.head(top_n)["filter"].tolist()
    print(f"\n[GRID] Top {top_n} filters selected: {top_filters}")

    # Build sweep space for selected filters only
    selected = [f for f in FILTERS if f["name"] in top_filters]
    sweep_values = []
    for f in selected:
        vals = [None] + f["sweep"]   # None = disabled
        sweep_values.append(vals)

    total_combos = 1
    for v in sweep_values:
        total_combos *= len(v)

    print(f"[GRID] Testing {total_combos:,} combinations...")
    print(f"       min_recall: {min_recall*100:.0f}% | min_tokens: {min_tokens}\n")

    rows     = []
    examined = 0

    for combo in itertools.product(*sweep_values):
        examined += 1
        if examined % 500 == 0:
            print(f"  {examined:>6} / {total_combos:,} tested | "
                  f"results so far: {len(rows):,}", end="\r")

        config = {f["name"]: None for f in FILTERS}
        for f, threshold in zip(selected, combo):
            config[f["name"]] = threshold

        result = score_config(df, label, config)
        if result is None:
            continue
        if result["n_passed"] < min_tokens:
            continue
        if result["recall"] < min_recall * 100:
            continue

        result["config"] = {
            f["name"]: threshold
            for f, threshold in zip(selected, combo)
            if threshold is not None
        }
        rows.append(result)

    print(f"\n  Examined {examined:,} combinations | {len(rows):,} passed filters")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ============================================================
# REPORTING
# ============================================================

def print_leaderboard(results: pd.DataFrame, top_n: int = 20, mode: str = "grid") -> None:
    if results.empty:
        print("[WARN] No results to display")
        return

    results = results.sort_values("precision", ascending=False)

    print(f"\n{'='*70}")
    print(f"  LEADERBOARD — top {min(top_n, len(results))} configs by precision")
    print(f"  (minimum recall and token count filters already applied)")
    print(f"{'='*70}")

    if mode == "grid":
        print(f"\n  {'#':>3}  {'Precision':>10} {'Recall':>8} {'F1':>6} "
              f"{'Tokens':>8} {'Pass%':>7}  Config")
        print(f"  {'-'*75}")

        for i, row in results.head(top_n).iterrows():
            cfg_str = "  |  ".join(
                f"{k}: {'≤' if [f for f in FILTERS if f['name']==k][0]['direction']=='max' else '≥'}{v}"
                for k, v in row["config"].items()
            )
            print(f"  {results.index.get_loc(i)+1:>3}  "
                  f"{row['precision']:>9.1f}%"
                  f"{row['recall']:>8.1f}%"
                  f"{row['f1']:>6.1f}%"
                  f"{row['n_passed']:>8,}"
                  f"{row['pass_rate']:>7.1f}%"
                  f"  {cfg_str}")

    else:  # independent
        print(f"\n  {'Filter':<35} {'Best Threshold':>15} {'Precision':>10} "
              f"{'Recall':>8} {'Tokens':>8}  GMGN Key")
        print(f"  {'-'*95}")

        best = (results.sort_values("precision", ascending=False)
                       .groupby("filter").first().reset_index()
                       .sort_values("precision", ascending=False))

        for _, row in best.iterrows():
            direction = "≤" if row["direction"] == "max" else "≥"
            print(f"  {row['label']:<35} {direction}{row['threshold']:>14} "
                  f"{row['precision']:>9.1f}% "
                  f"{row['recall']:>7.1f}% "
                  f"{row['n_passed']:>7,}  "
                  f"{row['gmgn_key']}")


def build_recommended_config(results: pd.DataFrame, original_config: dict) -> dict:
    """
    Take the top result from grid search and format as a GMGN-ready config dict.
    Merges with the original config so unchanged fields are preserved.
    """
    if results.empty:
        return original_config

    best = results.sort_values("precision", ascending=False).iloc[0]
    config = best["config"]

    # Map back to GMGN keys
    gmgn_map = {f["name"]: (f["gmgn_key"], f["direction"]) for f in FILTERS}

    print(f"\n{'='*60}")
    print(f"  RECOMMENDED GMGN CONFIG")
    print(f"  Precision: {best['precision']:.1f}% | "
          f"Recall: {best['recall']:.1f}% | "
          f"Tokens: {best['n_passed']:,}")
    print(f"{'='*60}\n")

    for filter_name, threshold in config.items():
        gmgn_key, direction = gmgn_map[filter_name]
        side = "max" if direction == "max" else "min"
        print(f"  {gmgn_key:<30} {side}: {threshold}")

    return best["config"]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backtest GMGN filter thresholds against graduation ground truth"
    )
    parser.add_argument("--data",        default="merged_token_data.csv")
    parser.add_argument("--label",       default="reached_graduation",
                        choices=["reached_graduation", "survived_1h",
                                 "survived_24h", "liquidity_withdrawn"])
    parser.add_argument("--mode",        default="both",
                        choices=["independent", "grid", "both"],
                        help="independent=fast single-filter sweep, "
                             "grid=multi-filter combinations, both=run both")
    parser.add_argument("--top-filters", type=int, default=5,
                        help="Number of top filters to include in grid search")
    parser.add_argument("--min-recall",  type=float, default=0.05,
                        help="Minimum recall (0-1) for grid results to be included")
    parser.add_argument("--min-tokens",  type=int, default=10,
                        help="Minimum tokens passing filter to include result")
    parser.add_argument("--output",      default="backtest_results",
                        help="Directory for output CSVs")
    args = parser.parse_args()

    import os
    os.makedirs(args.output, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[DATA] Loading {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")

    df = df.drop_duplicates(subset="token_address", keep="last")
    df = df.dropna(subset=[args.label])
    df[args.label] = df[args.label].astype(int)

    n_pos = df[args.label].sum()
    print(f"  Label: {args.label}")
    print(f"  Positive: {n_pos:,} ({n_pos/len(df)*100:.2f}%)")
    print(f"  Negative: {len(df)-n_pos:,}")

    # ── Use full dataset — no split needed for threshold backtesting ─────────
    # Unlike model training, filter thresholds have no learnable weights,
    # so there is no leakage risk. Using the full dataset gives more stable
    # statistics, especially important since graduations are only ~0.9%
    # of tokens — splitting would leave too few positive examples to measure.
    df["launch_time"] = pd.to_datetime(df["launch_time"], errors="coerce")
    df = df.dropna(subset=["launch_time"]).sort_values("launch_time")
    test_df = df.copy()
    print(f"\n  Using FULL DATASET: {len(test_df):,} tokens")
    print(f"  Period: {test_df['launch_time'].iloc[0].date()} → "
          f"{test_df['launch_time'].iloc[-1].date()}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Independent sweep ─────────────────────────────────────────────────────
    if args.mode in ("independent", "both"):
        ind_results = independent_sweep(test_df, args.label)
        print_leaderboard(ind_results, top_n=15, mode="independent")

        path = f"{args.output}/independent_{args.label}_{timestamp}.csv"
        ind_results.to_csv(path, index=False)
        print(f"\n  Saved → {path}")

    # ── Grid search ───────────────────────────────────────────────────────────
    if args.mode in ("grid", "both"):
        grid_results = grid_search(
            test_df, args.label,
            top_n      = args.top_filters,
            min_recall = args.min_recall,
            min_tokens = args.min_tokens,
        )
        print_leaderboard(grid_results, top_n=20, mode="grid")
        build_recommended_config(grid_results, {})

        if not grid_results.empty:
            # Flatten config dict for CSV export
            export = grid_results.drop(columns=["config"]).copy()
            path = f"{args.output}/grid_{args.label}_{timestamp}.csv"
            export.to_csv(path, index=False)
            print(f"\n  Saved → {path}")


if __name__ == "__main__":
    main()