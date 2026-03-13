"""
find_scam_devs.py

Mines your token database to identify dev wallets associated with rugs,
liquidity removals, and serial bad behaviour patterns.

Outputs:
  - scam_devs.csv         : ranked blacklist with full stats per dev
  - scam_devs_gmgn.txt    : wallet addresses ready to paste into GMGN blacklist
  - rug_tokens.csv        : all tokens linked to blacklisted devs

Usage:
    python find_scam_devs.py --data merged_token_data.csv
    python find_scam_devs.py --data merged_token_data.csv --min-tokens 2 --min-rug-rate 0.5
"""

import argparse
import os
from datetime import datetime

import pandas as pd

# ============================================================
# RUG CLASSIFICATION
#
# A token is classified as a rug/scam based on any of:
#   1. liquidity_withdrawn = 1             (dev pulled liquidity after graduation)
#   2. graduated_then_rugged = 1           (explicit rug flag)
#   3. dev_sold_in_5m = 1 AND token died  (dumped immediately and token failed)
#   4. dev_sell_ratio_pct > threshold      (dev sold large % of their holdings)
# ============================================================

RUG_SELL_RATIO_THRESHOLD = 50   # dev sold >50% of their holdings = rug signal
MIN_FAST_DUMP_VOLUME     = 0.1  # min SOL sold in 5m to count as a fast dump (not dust)


def classify_rug(df: pd.DataFrame) -> pd.Series:
    """Returns a boolean Series - True if token is a rug/scam."""
    rug = pd.Series(False, index=df.index)

    # Hard rug: liquidity pulled after graduation
    if "liquidity_withdrawn" in df.columns:
        rug |= df["liquidity_withdrawn"].fillna(0).astype(int) == 1

    # Hard rug: explicit graduated-then-rugged flag
    if "graduated_then_rugged" in df.columns:
        rug |= df["graduated_then_rugged"].fillna(0).astype(int) == 1

    # Soft rug: dev dumped in first 5m AND token didn't survive 1h
    if "dev_sold_in_5m" in df.columns:
        dev_dumped = df["dev_sold_in_5m"].fillna(0).astype(int) == 1
        if "dev_sell_volume_5m" in df.columns:
            # Only count meaningful dumps, not dust sells
            dev_dumped &= df["dev_sell_volume_5m"].fillna(0) >= MIN_FAST_DUMP_VOLUME
        if "survived_1h" in df.columns:
            token_died = df["survived_1h"].fillna(0).astype(int) == 0
            rug |= (dev_dumped & token_died)
        else:
            rug |= dev_dumped

    # Soft rug: dev sold large fraction of holdings
    if "dev_sell_ratio_pct" in df.columns:
        rug |= df["dev_sell_ratio_pct"].fillna(0) >= RUG_SELL_RATIO_THRESHOLD

    return rug


# ============================================================
# DEV SCORING
#
# Each dev wallet gets a danger score based on:
#   - rug_rate:          % of their tokens that were rugs
#   - total_tokens:      how many tokens they launched
#   - fast_dump_rate:    % where they sold in first 5m
#   - avg_dev_sell_pct:  average % of holdings they sold
#   - liq_pull_rate:     % where they pulled liquidity
#   - serial_bonus:      extra weight for devs with 3+ rugs
# ============================================================

def score_dev(row: pd.Series) -> float:
    score = 0.0

    # Base: rug rate (0-50 points)
    score += row.get("rug_rate", 0) * 50

    # Serial rugger bonus: 3+ rugs gets escalating penalty
    n_rugs = row.get("n_rugs", 0)
    if n_rugs >= 3:
        score += min(n_rugs * 3, 30)   # up to 30 extra points

    # Fast dump pattern (0-15 points)
    score += row.get("fast_dump_rate", 0) * 15

    # Liquidity pull rate (0-20 points - worst offence)
    score += row.get("liq_pull_rate", 0) * 20

    # Average sell ratio (0-10 points)
    score += min(row.get("avg_dev_sell_pct", 0) / 100, 1.0) * 10

    return round(score, 2)


def danger_label(score: float) -> str:
    if score >= 70:
        return "🔴 CONFIRMED SERIAL RUGGER"
    elif score >= 50:
        return "🟠 HIGH RISK"
    elif score >= 30:
        return "🟡 SUSPICIOUS"
    else:
        return "⚪ LOW SIGNAL"


# ============================================================
# MAIN ANALYSIS
# ============================================================

def find_scam_devs(
    data_path:      str,
    min_tokens:     int   = 2,
    min_rug_rate:   float = 0.5,
    output_dir:     str   = "scam_devs_output",
) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n[DATA] Loading {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")

    df = df.drop_duplicates(subset="token_address", keep="last")
    print(f"  After dedup: {len(df):,} unique tokens")

    # ── Require dev_wallet column ─────────────────────────────────────────────
    if "dev_wallet" not in df.columns:
        print("[ERROR] 'dev_wallet' column not found - cannot identify devs")
        return pd.DataFrame()

    df = df.dropna(subset=["dev_wallet"])
    df["dev_wallet"] = df["dev_wallet"].astype(str).str.strip()
    print(f"  Tokens with dev wallet: {len(df):,}")

    # ── Classify rugs ─────────────────────────────────────────────────────────
    df["is_rug"] = classify_rug(df)
    n_rugs = df["is_rug"].sum()
    print(f"\n[RUGS] Classified {n_rugs:,} tokens as rugs/scams ({n_rugs/len(df)*100:.1f}%)")
    print("  Breakdown:")

    if "liquidity_withdrawn" in df.columns:
        n = (df["liquidity_withdrawn"].fillna(0).astype(int) == 1).sum()
        print(f"    liquidity_withdrawn:   {n:,}")
    if "graduated_then_rugged" in df.columns:
        n = (df["graduated_then_rugged"].fillna(0).astype(int) == 1).sum()
        print(f"    graduated_then_rugged: {n:,}")
    if "dev_sold_in_5m" in df.columns:
        n = (df["dev_sold_in_5m"].fillna(0).astype(int) == 1).sum()
        print(f"    dev_sold_in_5m:        {n:,}")
    if "dev_sell_ratio_pct" in df.columns:
        n = (df["dev_sell_ratio_pct"].fillna(0) >= RUG_SELL_RATIO_THRESHOLD).sum()
        print(f"    dev_sell_ratio>={RUG_SELL_RATIO_THRESHOLD}%:     {n:,}")

    # ── Aggregate per dev wallet ──────────────────────────────────────────────
    print(f"\n[DEVS] Aggregating stats per dev wallet...")

    agg_spec = {
        "total_tokens": ("token_address", "count"),
        "n_rugs": ("is_rug", "sum"),
    }

    if "dev_sold_in_5m" in df.columns:
        agg_spec["n_fast_dumps"] = ("dev_sold_in_5m", "sum")
    if "dev_sell_volume_5m" in df.columns:
        agg_spec["avg_dump_volume_5m"] = ("dev_sell_volume_5m", "mean")
    if "dev_sell_ratio_pct" in df.columns:
        agg_spec["avg_dev_sell_pct"] = ("dev_sell_ratio_pct", "mean")
    if "dev_total_sell_volume" in df.columns:
        agg_spec["total_sell_volume"] = ("dev_total_sell_volume", "sum")
    if "liquidity_withdrawn" in df.columns:
        agg_spec["n_liq_pulls"] = ("liquidity_withdrawn", "sum")
    if "graduated_then_rugged" in df.columns:
        agg_spec["n_grad_then_rug"] = ("graduated_then_rugged", "sum")
    if "reached_graduation" in df.columns:
        agg_spec["n_graduated"] = ("reached_graduation", "sum")
    if "survived_1h" in df.columns:
        agg_spec["avg_survived_1h"] = ("survived_1h", "mean")
    if "launch_time" in df.columns:
        agg_spec["first_seen"] = ("launch_time", "min")
        agg_spec["last_seen"] = ("launch_time", "max")
    if "sniper_count" in df.columns:
        agg_spec["avg_snipers"] = ("sniper_count", "mean")
    if "manipulator_count" in df.columns:
        agg_spec["avg_manipulators"] = ("manipulator_count", "mean")
    if "dev_self_buy_count" in df.columns:
        agg_spec["total_self_buys"] = ("dev_self_buy_count", "sum")

    dev_stats = df.groupby("dev_wallet", dropna=False).agg(**agg_spec).reset_index()

    # Safety check
    required_cols = ["dev_wallet", "total_tokens", "n_rugs"]
    missing = [c for c in required_cols if c not in dev_stats.columns]
    if missing:
        raise ValueError(f"Missing required aggregated columns: {missing}")

    # Derived rates
    dev_stats["rug_rate"] = dev_stats["n_rugs"] / dev_stats["total_tokens"]
    dev_stats["fast_dump_rate"] = dev_stats.get("n_fast_dumps", 0) / dev_stats["total_tokens"]
    if "n_liq_pulls" in dev_stats.columns:
        dev_stats["liq_pull_rate"] = dev_stats["n_liq_pulls"] / dev_stats["total_tokens"]

    # ── Filter to meaningful devs ─────────────────────────────────────────────
    filtered = dev_stats[
        (dev_stats["total_tokens"] >= min_tokens) &
        (dev_stats["rug_rate"] >= min_rug_rate)
    ].copy()

    print(f"  Total dev wallets found: {len(dev_stats):,}")
    print(f"  After filter (>={min_tokens} tokens, >={min_rug_rate*100:.0f}% rug rate): {len(filtered):,}")

    if filtered.empty:
        print("\n[INFO] No scam devs found with current thresholds.")
        print("  Try: --min-tokens 1 --min-rug-rate 0.3")
        return pd.DataFrame()

    # ── Score and rank ────────────────────────────────────────────────────────
    filtered["danger_score"] = filtered.apply(score_dev, axis=1)
    filtered["danger_label"] = filtered["danger_score"].apply(danger_label)
    filtered = filtered.sort_values("danger_score", ascending=False).reset_index(drop=True)
    filtered.index += 1  # 1-based rank

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SCAM DEV LEADERBOARD")
    print(f"{'='*70}\n")

    for rank, row in filtered.head(30).iterrows():
        rug_pct = row["rug_rate"] * 100
        dump_pct = row.get("fast_dump_rate", 0) * 100
        liq_pct = row.get("liq_pull_rate", 0) * 100
        n_tokens = int(row["total_tokens"])
        n_rugs = int(row["n_rugs"])

        print(f"  #{rank:<3} {row['danger_label']}")
        print(f"       Wallet:     {row['dev_wallet']}")
        print(f"       Tokens:     {n_tokens} launched | {n_rugs} rugs ({rug_pct:.0f}% rug rate)")

        if "n_liq_pulls" in row and row["n_liq_pulls"] > 0:
            print(f"       Liq pulls:  {int(row['n_liq_pulls'])} ({liq_pct:.0f}%)")
        if "n_fast_dumps" in row and row.get("n_fast_dumps", 0) > 0:
            print(f"       Fast dumps: {int(row.get('n_fast_dumps', 0))} ({dump_pct:.0f}%)")
        if "avg_dev_sell_pct" in row and not pd.isna(row["avg_dev_sell_pct"]):
            print(f"       Avg sell%:  {row['avg_dev_sell_pct']:.0f}% of holdings")
        if "total_self_buys" in row and row.get("total_self_buys", 0) > 0:
            print(f"       Self-buys:  {int(row.get('total_self_buys', 0))} wash trades")
        if "first_seen" in row:
            print(f"       Active:     {row['first_seen']} -> {row['last_seen']}")
        print()

    # ── Summary stats ─────────────────────────────────────────────────────────
    confirmed = (filtered["danger_label"].str.contains("CONFIRMED")).sum()
    high_risk = (filtered["danger_label"].str.contains("HIGH")).sum()
    suspicious = (filtered["danger_label"].str.contains("SUSPICIOUS")).sum()

    print("\n[SUMMARY]")
    print(f"  🔴 Confirmed serial ruggers: {confirmed}")
    print(f"  🟠 High risk:                {high_risk}")
    print(f"  🟡 Suspicious:               {suspicious}")
    print(f"  Total blacklist candidates:  {len(filtered)}")

    # ── Save outputs ──────────────────────────────────────────────────────────

    # Full stats CSV
    stats_path = os.path.join(output_dir, f"scam_devs_{timestamp}.csv")
    filtered.to_csv(stats_path, index=False)
    print(f"\n[SAVED] Dev stats    -> {stats_path}")

    # GMGN-ready wallet list (one address per line)
    gmgn_path = os.path.join(output_dir, f"blacklist_wallets_{timestamp}.txt")
    with open(gmgn_path, "w", encoding="utf-8") as f:
        f.write("# Scam dev blacklist - generated from token database\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"# Total: {len(filtered)} wallets\n\n")

        for _, row in filtered.iterrows():
            label = row["danger_label"].replace("🔴 ", "").replace("🟠 ", "").replace("🟡 ", "").replace("⚪ ", "")
            rug_pct = row["rug_rate"] * 100
            f.write(f"{row['dev_wallet']}  # {label} | {int(row['total_tokens'])} tokens | {rug_pct:.0f}% rug rate\n")

    print(f"[SAVED] GMGN wallets -> {gmgn_path}")

    # Rug tokens linked to blacklisted devs
    blacklisted_wallets = set(filtered["dev_wallet"])
    rug_tokens = df[df["dev_wallet"].isin(blacklisted_wallets)].copy()
    rug_tokens_path = os.path.join(output_dir, f"rug_tokens_{timestamp}.csv")
    rug_tokens[
        ["token_address", "dev_wallet", "is_rug",
         "launch_time" if "launch_time" in rug_tokens.columns else "token_address"]
        + [
            c for c in [
                "liquidity_withdrawn",
                "graduated_then_rugged",
                "dev_sold_in_5m",
                "dev_sell_ratio_pct",
                "reached_graduation",
                "survived_1h",
            ] if c in rug_tokens.columns
        ]
    ].to_csv(rug_tokens_path, index=False)
    print(f"[SAVED] Rug tokens   -> {rug_tokens_path}")

    return filtered


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find scam dev wallets in token database")
    parser.add_argument("--data",          default="merged_token_data.csv")
    parser.add_argument("--min-tokens",    type=int,   default=2,
                        help="Min tokens launched by dev to be included (default: 2)")
    parser.add_argument("--min-rug-rate",  type=float, default=0.5,
                        help="Min fraction of rugs to flag dev (default: 0.5 = 50%%)")
    parser.add_argument("--output",        default="scam_devs_output")
    args = parser.parse_args()

    find_scam_devs(
        data_path    = args.data,
        min_tokens   = args.min_tokens,
        min_rug_rate = args.min_rug_rate,
        output_dir   = args.output,
    )