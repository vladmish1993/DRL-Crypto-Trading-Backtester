"""
merge_token_files.py

Merges pump.fun token stats CSVs by token_address.

Handles:
  - Multiple file parts via glob patterns
  - Duplicate token_address rows (merges by taking first non-null value per column)
  - Overlapping columns between files (stats file takes priority)

Usage:
    python merge_token_files.py
        --stats   "dune_token_stats_part_*.csv"
        --dev     "dune_token_dev_stats_part_*.csv"
        --output  merged_token_data.csv
"""

import argparse
import glob
import sys
import pandas as pd


# Columns that belong exclusively to token stats query (query 1)
STATS_ONLY_COLS = {
    "volume_30s", "volume_1m", "volume_3m", "volume_5m", "volume_30m",
    "volume_after_5m", "buyers_30s", "sellers_30s", "buyers_1m", "sellers_1m",
    "buyers_3m", "sellers_3m", "buyers_5m", "sellers_5m", "buyers_30m",
    "sellers_30m", "unique_wallets_5m", "unique_wallets_30m", "total_unique_wallets",
    "buy_txns_30s", "sell_txns_30s", "buy_txns_1m", "sell_txns_1m",
    "buy_txns_5m", "sell_txns_5m",
    "buy_size_p25_1m", "buy_size_p50_1m", "buy_size_p75_1m", "buy_size_p95_1m",
    "buy_size_p25_5m", "buy_size_p50_5m", "buy_size_p75_5m", "buy_size_p95_5m",
    "buy_sell_ratio_30s", "buy_sell_ratio_1m", "buy_sell_ratio_5m", "buy_sell_ratio_30m",
    "volume_per_unique_buyer", "holders_at_1m", "holders_at_5m", "holders_at_30m",
    "holder_growth_1m_to_5m", "holder_growth_5m_to_30m",
    "top5_holder_pct", "top10_holder_pct", "top20_holder_pct",
    "top10_volume_pct", "net_flow_excl_top10",
    "bundler_wallets_10s", "bundler_wallets_30s", "bundler_wallets_60s",
    "bundler_wallets_5m", "bundler_pct_of_buyers_1m",
    "early_buyers", "late_buyers", "late_to_early_ratio",
    "organic_buyer_pct", "wallet_retention_5m_to_30m",
    "price_at_launch", "peak_price_5m", "price_stddev_5m",
    "net_buy_pressure_5m", "upside_burst_5m",
    "survived_30m", "survived_1h", "survived_24h",
}

# Columns that belong exclusively to dev stats query (query 2)
DEV_ONLY_COLS = {
    "total_supply", "dev_wallet",
    "dev_sold_in_5m", "dev_sell_volume_5m", "dev_sold_in_30m",
    "dev_total_sell_volume", "dev_total_buy_volume", "dev_sell_ratio_pct",
    "dev_self_buy_count", "deployer_transfer_count",
    "sniper_count", "manipulator_count",
    "reached_graduation", "minutes_to_graduation", "seconds_to_graduation", "graduated_at",
    "liquidity_withdrawn", "minutes_to_withdrawal", "seconds_to_withdrawal", "withdrawn_at",
    "graduated_then_rugged",
    "raydium_unique_traders", "raydium_unique_buyers", "raydium_volume", "raydium_trade_count",
    "total_early_buyers", "fresh_wallet_count", "established_wallet_count",
    "fresh_wallet_pct", "established_wallet_pct",
}


def load_files(pattern: str, label: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[ERROR] No files found matching: {pattern}")
        sys.exit(1)

    print(f"[{label}] Loading {len(paths)} file(s):")
    frames = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        print(f"  {p}: {len(df):,} rows, {len(df.columns)} columns")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  → Total before dedup: {len(combined):,} rows")
    return combined


def dedup_by_first_nonnull(df: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    """
    For duplicate token_address rows, merge them by taking the first
    non-null value in each column. This handles cases where the same
    token appears in multiple pages with partial data.
    """
    before = len(df)
    if df[key].duplicated().any():
        # Sort so non-null values come first, then aggregate
        df = df.groupby(key, as_index=False).first()
        after = len(df)
        print(f"  [WARN] {label}: merged {before - after:,} duplicate rows → {after:,} unique tokens")
    else:
        print(f"  No duplicates found in {label}")
    return df


def strip_to_own_cols(df: pd.DataFrame, own_cols: set, key: str, label: str) -> pd.DataFrame:
    """
    Keep only columns that belong to this file's query.
    Drops columns that leaked from the other query.
    """
    all_cols   = set(df.columns)
    keep       = (own_cols & all_cols) | {key}
    dropped    = all_cols - keep
    if dropped:
        print(f"  [{label}] Dropping {len(dropped)} cross-contaminated columns: {sorted(dropped)}")
    return df[[c for c in df.columns if c in keep]]


def merge(stats_pattern: str, dev_pattern: str, output_path: str) -> None:
    key = "token_address"

    # ── Load ──────────────────────────────────────────────────────────────────
    stats = load_files(stats_pattern, "TOKEN STATS")
    dev   = load_files(dev_pattern,   "DEV STATS")

    for name, df in [("token stats", stats), ("dev stats", dev)]:
        if key not in df.columns:
            print(f"[ERROR] '{key}' column not found in {name}.")
            sys.exit(1)

    # ── Strip columns that don't belong to each file ──────────────────────────
    # Handles cases where the combined Dune query exports all columns to both files
    print()
    stats_key_col = stats[[key, "launch_time"]] if "launch_time" in stats.columns else stats[[key]]
    stats = strip_to_own_cols(stats, STATS_ONLY_COLS | {"launch_time"}, key, "TOKEN STATS")
    dev   = strip_to_own_cols(dev,   DEV_ONLY_COLS   | {"launch_time"}, key, "DEV STATS")

    # ── Deduplicate ───────────────────────────────────────────────────────────
    print()
    stats = dedup_by_first_nonnull(stats, key, "token stats")
    dev   = dedup_by_first_nonnull(dev,   key, "dev stats")

    # ── Handle launch_time: prefer stats version, fall back to dev ────────────
    if "launch_time" in stats.columns and "launch_time" in dev.columns:
        dev = dev.rename(columns={"launch_time": "launch_time_dev"})

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = stats.merge(dev, on=key, how="outer")

    # Resolve launch_time if both existed
    if "launch_time_dev" in merged.columns:
        merged["launch_time"] = merged["launch_time"].fillna(merged["launch_time_dev"])
        merged = merged.drop(columns=["launch_time_dev"])

    print(f"\n[MERGE] Result: {len(merged):,} rows, {len(merged.columns)} columns")

    # ── Coverage report ───────────────────────────────────────────────────────
    stats_keys = set(stats[key])
    dev_keys   = set(dev[key])
    in_both    = len(stats_keys & dev_keys)
    only_stats = len(stats_keys - dev_keys)
    only_dev   = len(dev_keys - stats_keys)

    print(f"\n[COVERAGE]")
    print(f"  Tokens in both files : {in_both:,}")
    print(f"  Only in token stats  : {only_stats:,}")
    print(f"  Only in dev stats    : {only_dev:,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    # Reorder: token_address and launch_time first, then stats cols, then dev cols
    col_order = [key]
    if "launch_time" in merged.columns:
        col_order.append("launch_time")
    col_order += [c for c in merged.columns if c in STATS_ONLY_COLS]
    col_order += [c for c in merged.columns if c in DEV_ONLY_COLS]
    col_order += [c for c in merged.columns if c not in col_order]  # any extras
    merged = merged[[c for c in col_order if c in merged.columns]]

    merged.to_csv(output_path, index=False)
    print(f"\n[DONE] Saved → {output_path}")
    print(f"       Rows: {len(merged):,} | Columns: {len(merged.columns)}")
    print(f"\nColumns ({len(merged.columns)}):")
    for i, col in enumerate(merged.columns, 1):
        print(f"  {i:>3}. {col}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge pump.fun token stat CSVs by token_address")
    parser.add_argument("--stats",  default="dune_token_stats_part_*.csv",
                        help="Glob pattern for token stats files")
    parser.add_argument("--dev",    default="dune_token_dev_stats_part_*.csv",
                        help="Glob pattern for dev stats files")
    parser.add_argument("--output", default="merged_token_data.csv",
                        help="Output file path")
    args = parser.parse_args()

    merge(args.stats, args.dev, args.output)