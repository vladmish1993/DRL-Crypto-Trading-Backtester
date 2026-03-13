"""
train_model.py

LightGBM classifier for pump.fun token survival prediction.
Trains on merged token data and evaluates with time-based splits.

Usage:
    python train_model.py --data merged_token_data.csv
    python train_model.py --data merged_token_data.csv --labels "survived_1h,reached_graduation"

Install dependencies:
    pip install lightgbm scikit-learn pandas numpy matplotlib joblib
"""

import argparse
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")


# ============================================================
# FEATURE SETS
#
# LEAKAGE RULE: features must only use data available at the
# moment of prediction. Each label has a decision horizon —
# only features from before that horizon are allowed.
#
#   survived_30m       → decide at 5m  → FEATURES_5M only
#   survived_1h        → decide at 5m  → FEATURES_5M only
#   liquidity_withdrawn → decide at 5m → FEATURES_5M only
#   survived_24h       → decide at 30m → FEATURES_30M ok
#   reached_graduation → decide at 30m → FEATURES_30M ok
#
# Previously survived_1h used FEATURES_30M which included
# volume_after_5m, volume_30m, holder_growth_5m_to_30m etc.
# These are future data relative to the 5m decision point
# and caused inflated ROC-AUC (0.97 → real number ~0.85-0.90)
# ============================================================

# Safe at exactly 5 minutes post-launch — no future data
FEATURES_5M = [
    # Volume windows (all observed within first 5m)
    "volume_30s", "volume_1m", "volume_3m", "volume_5m",
    # Buyer/seller unique wallet counts
    "buyers_30s", "sellers_30s",
    "buyers_1m",  "sellers_1m",
    "buyers_3m",  "sellers_3m",
    "buyers_5m",  "sellers_5m",
    "unique_wallets_5m",
    # Transaction counts (same wallet can tx multiple times)
    "buy_txns_30s", "sell_txns_30s",
    "buy_txns_1m",  "sell_txns_1m",
    "buy_txns_5m",  "sell_txns_5m",
    # Swap size distribution
    "buy_size_p25_1m", "buy_size_p50_1m", "buy_size_p75_1m", "buy_size_p95_1m",
    "buy_size_p25_5m", "buy_size_p50_5m", "buy_size_p75_5m", "buy_size_p95_5m",
    # Flow ratios
    "buy_sell_ratio_30s", "buy_sell_ratio_1m", "buy_sell_ratio_5m",
    "volume_per_unique_buyer",
    # Holder growth (within 5m only)
    "holders_at_1m", "holders_at_5m",
    "holder_growth_1m_to_5m",
    # Concentration (computed from all-time balances but available at launch)
    "top5_holder_pct", "top10_holder_pct", "top20_holder_pct",
    "top10_volume_pct", "net_flow_excl_top10",
    # Bundler signals
    "bundler_wallets_10s", "bundler_wallets_30s", "bundler_wallets_60s",
    "bundler_wallets_5m", "bundler_pct_of_buyers_1m",
    # Organic growth (early_buyers = bought in first 60s)
    "early_buyers", "organic_buyer_pct",
    # Price path (within 5m)
    "price_at_launch", "peak_price_5m", "price_stddev_5m",
    "net_buy_pressure_5m", "upside_burst_5m",
    # Dev behaviour at 5m
    "dev_sold_in_5m", "dev_sell_volume_5m",
    "dev_self_buy_count", "deployer_transfer_count",
    # Sniper/manipulator counts (observable at launch)
    "sniper_count", "manipulator_count",
    # Wallet freshness
    "fresh_wallet_pct", "established_wallet_pct",
]

# Safe at 30 minutes post-launch — adds 30m window data
# Only use for labels where you'd realistically wait 30m before deciding
FEATURES_30M = FEATURES_5M + [
    "volume_30m",
    "buyers_30m",   "sellers_30m",
    "unique_wallets_30m",
    "buy_sell_ratio_30m",
    "holders_at_30m",
    "holder_growth_5m_to_30m",
    "late_buyers", "late_to_early_ratio",
    "wallet_retention_5m_to_30m",
    "dev_sold_in_30m",
    "volume_after_5m",   # volume from 5m-30m window — safe at 30m decision point
]

# ============================================================
# LABEL → FEATURE SET MAPPING
# This is the critical leakage prevention layer.
# Each label maps to the feature set that's safe to use.
# ============================================================
LABEL_FEATURE_MAP = {
    "survived_30m":        FEATURES_5M,   # decide at 5m → strict 5m features only
    "survived_1h":         FEATURES_5M,   # decide at 5m → strict 5m features only
                                          # (was FEATURES_30M — caused leakage)
    "survived_24h":        FEATURES_30M,  # decide at 30m → 30m features safe
    "reached_graduation":  FEATURES_30M,  # decide at 30m → 30m features safe
    "liquidity_withdrawn": FEATURES_5M,   # rug detection → early signals only
}

ALL_LABELS = list(LABEL_FEATURE_MAP.keys())


# ============================================================
# ENGINEERED FEATURES
# All derived from 5m-window data — safe for any label
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Sell pressure acceleration: sells growing relative to buys
    df["sell_accel_1m_to_5m"] = (
        (df.get("sellers_5m", 0) - df.get("sellers_1m", 0)).clip(lower=0)
        / df.get("buyers_5m", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    )

    # Bundler dominance: fraction of early buyers that are suspicious
    df["bundler_dominance"] = (
        df.get("bundler_wallets_5m", 0)
        / df.get("unique_wallets_5m", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    )

    # Buy size uniformity: low IQR = bots buying identical sizes
    if "buy_size_p75_5m" in df.columns and "buy_size_p25_5m" in df.columns:
        df["buy_size_iqr_5m"] = df["buy_size_p75_5m"] - df["buy_size_p25_5m"]

    # Price pump efficiency: how much pump per dollar committed
    if "upside_burst_5m" in df.columns and "volume_5m" in df.columns:
        df["pump_efficiency"] = (
            df["upside_burst_5m"] / df["volume_5m"].replace(0, np.nan)
        )

    # Concentration risk: top holders dominating both supply AND volume
    if "top10_holder_pct" in df.columns and "top10_volume_pct" in df.columns:
        df["concentration_risk"] = df["top10_holder_pct"] * df["top10_volume_pct"] / 100

    # Dev aggression: sold in first 5m AND sold a lot
    if "dev_sold_in_5m" in df.columns and "dev_sell_volume_5m" in df.columns:
        df["dev_aggression"] = df["dev_sold_in_5m"] * df["dev_sell_volume_5m"].fillna(0)

    # Sniper pressure: snipers per unique early buyer
    if "sniper_count" in df.columns and "buyers_1m" in df.columns:
        df["sniper_pressure"] = (
            df["sniper_count"] / df["buyers_1m"].replace(0, np.nan)
        )

    return df


ENGINEERED_FEATURES = [
    "sell_accel_1m_to_5m", "bundler_dominance", "buy_size_iqr_5m",
    "pump_efficiency", "concentration_risk", "dev_aggression", "sniper_pressure",
]


# ============================================================
# DATA LOADING AND CLEANING
# ============================================================

def load_and_clean(path: str, label: str) -> pd.DataFrame:
    print(f"\n[DATA] Loading {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw rows: {len(df):,} | Columns: {len(df.columns)}")

    # Parse and sort by launch time — critical for time-based splits
    df["launch_time"] = pd.to_datetime(df["launch_time"], errors="coerce")
    df = df.dropna(subset=["launch_time"])
    df = df.sort_values("launch_time").reset_index(drop=True)

    # Deduplicate by token_address — keep most recent entry
    before = len(df)
    df = df.drop_duplicates(subset="token_address", keep="last")
    if len(df) < before:
        print(f"  [WARN] Dropped {before - len(df):,} duplicate token rows")

    # Validate and clean label
    if label not in df.columns:
        raise ValueError(
            f"Label '{label}' not found. Available: "
            f"{[c for c in df.columns if c in ALL_LABELS]}"
        )
    df = df.dropna(subset=[label])
    df[label] = df[label].astype(int)

    print(f"  After cleaning: {len(df):,} rows")
    print(f"  Label distribution:")
    vc = df[label].value_counts().sort_index()
    for val, count in vc.items():
        label_name = "survived/graduated" if val == 1 else "died/rugged"
        print(f"    {val} ({label_name}): {count:,} ({count/len(df)*100:.1f}%)")

    return df


# ============================================================
# TRAINING
# ============================================================

def train(data_path: str, label: str, output_dir: str = "model_output") -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # ── Load and engineer features ────────────────────────────────────────────
    df = load_and_clean(data_path, label)
    df = engineer_features(df)

    # Build feature list: base set for this label + engineered features
    base_features  = LABEL_FEATURE_MAP.get(label, FEATURES_5M)
    all_candidates = base_features + ENGINEERED_FEATURES

    # Keep only features that exist in the dataframe
    feature_cols = [f for f in dict.fromkeys(all_candidates) if f in df.columns]

    # Report any missing base features
    missing = [f for f in base_features if f not in df.columns]
    if missing:
        print(f"\n  [WARN] {len(missing)} expected features missing from data: {missing[:8]}{'...' if len(missing) > 8 else ''}")

    print(f"\n[FEATURES] Using {len(feature_cols)} features for label '{label}'")
    print(f"  Decision horizon: {'5 minutes' if LABEL_FEATURE_MAP.get(label) is FEATURES_5M else '30 minutes'} post-launch")

    X = df[feature_cols]
    y = df[label]

    # ── Time-based train/test split — NEVER random on time series ────────────
    split      = int(len(df) * 0.8)
    X_train    = X.iloc[:split]
    X_test     = X.iloc[split:]
    y_train    = y.iloc[:split]
    y_test     = y.iloc[split:]

    print(f"\n[SPLIT] Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Train: {df['launch_time'].iloc[0].date()} → {df['launch_time'].iloc[split-1].date()}")
    print(f"  Test:  {df['launch_time'].iloc[split].date()} → {df['launch_time'].iloc[-1].date()}")

    # ── Class imbalance correction ────────────────────────────────────────────
    pos_rate         = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    print(f"\n[BALANCE] Positive rate: {pos_rate:.1%} → scale_pos_weight: {scale_pos_weight:.2f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = lgb.LGBMClassifier(
        n_estimators      = 500,
        learning_rate     = 0.05,
        num_leaves        = 63,
        max_depth         = -1,
        min_child_samples = 20,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        scale_pos_weight  = scale_pos_weight,
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )

    # Early stopping: last 10% of train as internal validation
    val_split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
    y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

    model.fit(
        X_tr, y_tr,
        eval_set  = [(X_val, y_val)],
        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"\n[MODEL] Best iteration: {model.best_iteration_}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test  = model.predict_proba(X_test)[:, 1]

    roc_train = roc_auc_score(y_train, proba_train)
    roc_test  = roc_auc_score(y_test,  proba_test)
    pr_train  = average_precision_score(y_train, proba_train)
    pr_test   = average_precision_score(y_test,  proba_test)

    print(f"\n{'='*55}")
    print(f"  RESULTS — {label}")
    print(f"{'='*55}")
    print(f"  ROC-AUC  train: {roc_train:.4f}  |  test: {roc_test:.4f}")
    print(f"  PR-AUC   train: {pr_train:.4f}  |  test: {pr_test:.4f}")

    gap = roc_train - roc_test
    if gap > 0.05:
        print(f"\n  [WARN] Overfit detected — train/test ROC gap: {gap:.4f}")
        print(f"         Consider: more data, higher reg_alpha/lambda, or fewer features")

    # Classification report
    y_pred = (proba_test >= 0.5).astype(int)
    print(f"\n  Classification report (threshold=0.5):")
    print(classification_report(y_test, y_pred, target_names=["died/rugged", "survived"]))

    # Feature importance
    importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"  Top 20 features:")
    print(importance.head(20).to_string(index=False))

    importance_path = os.path.join(output_dir, f"feature_importance_{label}.csv")
    importance.to_csv(importance_path, index=False)

    # Score distribution at different thresholds
    print(f"\n  Score distribution on test set:")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        mask      = proba_test >= threshold
        flagged   = mask.sum()
        precision = y_test[mask].mean() if flagged > 0 else 0
        print(f"    ≥ {threshold}: {flagged:>6} tokens | precision: {precision:.1%}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Model: {label}", fontsize=13)

    # Feature importance
    top20 = importance.head(20)
    axes[0].barh(top20["feature"][::-1], top20["importance"][::-1])
    axes[0].set_title("Top 20 Feature Importance")
    axes[0].set_xlabel("Importance")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["died", "survived"]).plot(
        ax=axes[1], colorbar=False
    )
    axes[1].set_title("Confusion Matrix (threshold=0.5)")

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_test, proba_test, n_bins=10)
    axes[2].plot(mean_pred, frac_pos, "s-", label="Model")
    axes[2].plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    axes[2].set_title("Calibration Curve")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Fraction of positives")
    axes[2].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"evaluation_{label}.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Plots  → {plot_path}")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(output_dir, f"model_{label}.pkl")
    joblib.dump({
        "model":        model,
        "feature_cols": feature_cols,
        "label":        label,
        "roc_auc_test": roc_test,
        "pr_auc_test":  pr_test,
    }, model_path)
    print(f"  Model  → {model_path}")

    return {"roc_auc": roc_test, "pr_auc": pr_test}


# ============================================================
# INFERENCE: score a new token from live data
# ============================================================

def score_token(model_path: str, token_data: dict) -> float:
    """
    Score a single token against a saved model.

    Args:
        model_path: path to saved .pkl model file
        token_data: dict of {feature_name: value}

    Returns:
        float: probability 0-1 of the label being 1
    """
    bundle = joblib.load(model_path)
    model  = bundle["model"]
    feats  = bundle["feature_cols"]

    row  = pd.DataFrame([token_data])[feats]
    prob = model.predict_proba(row)[0, 1]
    return prob


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pump.fun token survival models")
    parser.add_argument("--data",   default="merged_token_data.csv",
                        help="Path to merged token CSV")
    parser.add_argument("--labels", default="all",
                        help=f"Comma-separated labels or 'all'. Options: {ALL_LABELS}")
    parser.add_argument("--output", default="model_output",
                        help="Directory to save models, plots, and feature importances")
    args = parser.parse_args()

    labels_to_train = ALL_LABELS if args.labels == "all" else [
        l.strip() for l in args.labels.split(",")
    ]

    # Validate requested labels
    invalid = [l for l in labels_to_train if l not in ALL_LABELS]
    if invalid:
        print(f"[ERROR] Unknown labels: {invalid}. Valid options: {ALL_LABELS}")
        exit(1)

    print(f"\n{'='*55}")
    print(f"  Training {len(labels_to_train)} model(s)")
    print(f"  Labels:  {labels_to_train}")
    print(f"  Data:    {args.data}")
    print(f"  Output:  {args.output}/")
    print(f"{'='*55}")

    results = {}
    for label in labels_to_train:
        try:
            results[label] = train(args.data, label, args.output)
        except Exception as e:
            print(f"\n[ERROR] '{label}' failed: {e}")
            results[label] = None

    # Final summary table
    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Label':<25} {'Horizon':<10} {'ROC-AUC':>8} {'PR-AUC':>8} {'Status':>8}")
    print(f"  {'-'*60}")
    for label, res in results.items():
        horizon = "5m" if LABEL_FEATURE_MAP.get(label) is FEATURES_5M else "30m"
        if res:
            print(f"  {label:<25} {horizon:<10} {res['roc_auc']:>8.4f} {res['pr_auc']:>8.4f} {'OK':>8}")
        else:
            print(f"  {label:<25} {horizon:<10} {'—':>8} {'—':>8} {'FAILED':>8}")