import json
import math
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple

import pandas as pd
from curl_cffi import requests as cf_requests


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = "profit_wallets.csv"

# Safety / speed
MAX_WORKERS_PREFILTER = 6
MAX_WORKERS_DEEP = 3
REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
BASE_RETRY_SLEEP = 1.5
REQUEST_DELAY_BETWEEN_PAGES = 0.20
REQUEST_DELAY_BETWEEN_STATS = 0.10  # delay between 1d/7d/30d stat calls

# Checkpoint saving
SAVE_EVERY_N_RESULTS = 50

# Test mode
LIMIT_WALLETS = 20  # e.g. 50

# After prefilter, deep analyse only top N passing wallets by prefilter score.
# Set to None to deep analyse all passing wallets.
DEEP_ANALYSIS_TOP_N = None  # e.g. 100

# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_XLSX = "gmgn_copytrade_analysis.xlsx"
OUTPUT_JSON = "gmgn_copytrade_analysis.json"
OUTPUT_FINAL_CSV = "gmgn_copytrade_analysis.csv"
OUTPUT_DEEP_CSV = "gmgn_copytrade_analysis_deep.csv"
OUTPUT_TOKEN_CSV = "gmgn_token_level_30d.csv"
SAVE_TOKEN_CSV = True

# Remove noisy columns from wallet exports
EXPORT_DROP_COLUMNS = {
    "solscan",
    "wallet_analysor",
    "breakdown",
}

# ============================================================
# PREFILTER THRESHOLDS
# ============================================================

# Tighter 7d token thresholds for copytrading
TOKEN_NUM_7D_SKIP_GT = 350
TOKEN_NUM_7D_HARD_REJECT_GT = 500
TOKEN_NUM_30D_SKIP_GT = 2000

FAST_TX_RATIO_30D_SKIP_GT = 0.20
FAST_TX_RATIO_30D_HARD_REJECT_GT = 0.25

AVG_HOLDING_PERIOD_30D_SKIP_LT = 60.0

# These come from your source CSV
MIN_MEDIAN_HOLDING_SECONDS = 60
MIN_POSITION_SIZE_USD = 20
MAX_POSITION_SIZE_USD = 500

# Deep-analysis quality filters
SNIPER_RATE_5S_REJECT_GT = 0.20
SNIPER_RATE_5S_WARN_GT = 0.10
TOP_TOKEN_PROFIT_SHARE_REJECT_GT = 0.70
TOP_TOKEN_PROFIT_SHARE_WARN_GT = 0.50

# ── Dev filtering thresholds ──
DEV_CREATED_TOKENS_REJECT_GT = 10      # created >10 tokens → reject
DEV_CREATED_TOKENS_WARN_GT = 3         # created >3 tokens → flag
DEV_OPEN_RATIO_REJECT_LT = 0.05        # <5% of tokens still open → rug pattern

# ── Activity consistency thresholds ──
ACTIVE_DAYS_7D_MIN = 3                  # need at least 3 of 7 days active
ACTIVE_DAYS_30D_MIN = 10                # need at least 10 of 30 days active
DAYS_SINCE_LAST_TRADE_REJECT_GT = 3     # inactive >3 days → reject for copytrade
DAYS_SINCE_LAST_TRADE_WARN_GT = 1       # inactive >1 day → warn

# ── Daily profit % thresholds ──
DAILY_ROI_7D_MIN = 0.005                # at least 0.5% daily ROI over 7d

ILLEGAL_EXCEL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

# ============================================================
# API PARAMS
# ============================================================

BASE_PARAMS = {
    "device_id": "20b797a6-e165-49cc-835b-e0bcc9fa25f7",
    "fp_did": "3fa0ea5ea4368ac14463b9432da366e5",
    "client_id": "gmgn_web_20260313-11689-7b75b83",
    "from_app": "gmgn",
    "app_ver": "20260313-11689-7b75b83",
    "tz_name": "Europe/London",
    "tz_offset": 0,
    "app_lang": "en-US",
    "os": "web",
    "worker": 0,
}

HOLDINGS_BASE_PARAMS = {
    **BASE_PARAMS,
    "limit": 50,
    "order_by": "last_active_timestamp",
    "direction": "desc",
    "hide_airdrop": "false",
    "hide_abnormal": "false",
    "hide_closed": "false",
    "sellout": "true",
    "showsmall": "true",
    "tx30d": "true",
}

BASE_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://gmgn.ai",
}

thread_local = threading.local()
# Lock for shared mutable state across threads
_token_rows_lock = threading.Lock()


# ============================================================
# HELPERS
# ============================================================

def D(value) -> Decimal:
    if value in (None, "", "null"):
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0")


def safe_div(a: Decimal, b: Decimal) -> Decimal:
    if b == 0:
        return Decimal("0")
    return a / b


def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = cf_requests.Session(impersonate="chrome110")
    return thread_local.session


def make_headers(wallet: str) -> Dict[str, str]:
    headers = BASE_HEADERS.copy()
    headers["referer"] = f"https://gmgn.ai/sol/address/{wallet}"
    return headers


def export_wallet_rows(rows: List[dict]) -> List[dict]:
    cleaned = []
    for row in rows:
        cleaned.append({k: v for k, v in row.items() if k not in EXPORT_DROP_COLUMNS})
    return cleaned


def load_existing_json_checkpoint(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        out = {}
        for row in rows:
            wallet = str(row.get("wallet", "")).strip()
            if wallet:
                out[wallet] = row
        print(f"[resume] loaded {len(out)} wallet rows from {path}")
        return out
    except Exception as e:
        print(f"[resume warning] failed to load {path}: {e}")
        return {}


def load_existing_token_rows(path: str) -> List[dict]:
    if not SAVE_TOKEN_CSV or not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
        rows = normalise_records(df)
        print(f"[resume] loaded {len(rows)} token rows from {path}")
        return rows
    except Exception as e:
        print(f"[resume warning] failed to load token CSV {path}: {e}")
        return []


def deduplicate_token_rows(token_rows: List[dict]) -> List[dict]:
    """Deduplicate token rows by (wallet, token_address) keeping the last occurrence."""
    seen = {}
    for row in token_rows:
        key = (row.get("wallet", ""), row.get("token_address", ""))
        seen[key] = row
    return list(seen.values())


def token_key(item: dict, idx: Optional[int] = None) -> str:
    token = item.get("token") or {}
    token_address = token.get("token_address")
    if token_address:
        return token_address
    return f"unknown_{idx if idx is not None else 0}"


def to_float(value, default=0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def to_int(value, default=0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def normalise_records(df: pd.DataFrame) -> List[dict]:
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


def now_ts() -> int:
    return int(time.time())


# ============================================================
# ATOMIC SAVING
# ============================================================

def clean_excel_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return ILLEGAL_EXCEL_CHARS_RE.sub("", value)
    return value


def clean_excel_records(rows: list[dict]) -> list[dict]:
    cleaned = []
    for row in rows:
        cleaned.append({k: clean_excel_value(v) for k, v in row.items()})
    return cleaned


def atomic_write_excel(
    all_wallet_rows: List[dict],
    deep_wallet_rows: List[dict],
    token_rows: List[dict],
    output_path: str,
) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(prefix="gmgn_", suffix=".xlsx", dir=output_dir)
    os.close(fd)

    try:
        all_df = pd.DataFrame(clean_excel_records(all_wallet_rows))
        deep_df = pd.DataFrame(clean_excel_records(deep_wallet_rows))
        token_df = pd.DataFrame(clean_excel_records(token_rows))

        all_df = sort_wallet_df_for_output(all_df)
        deep_df = sort_wallet_df_for_output(deep_df)

        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            all_df.to_excel(writer, sheet_name="wallets_all", index=False)
            deep_df.to_excel(writer, sheet_name="wallets_deep", index=False)

            if not token_df.empty:
                token_df.to_excel(writer, sheet_name="tokens_30d", index=False)

        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def atomic_write_json(data: List[dict], output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(prefix="gmgn_", suffix=".json", dir=output_dir)
    os.close(fd)

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def atomic_write_csv(rows: List[dict], output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(prefix="gmgn_", suffix=".csv", dir=output_dir)
    os.close(fd)

    try:
        pd.DataFrame(rows).to_csv(temp_path, index=False)
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ============================================================
# API
# ============================================================

def fetch_json(url: str, params: dict, headers: dict, wallet: str, tag: str) -> dict:
    session = get_session()
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()

            if payload.get("code") != 0:
                raise RuntimeError(
                    f"GMGN code={payload.get('code')} "
                    f"message={payload.get('message')} "
                    f"reason={payload.get('reason')}"
                )

            return payload

        except Exception as e:
            last_error = e
            sleep_s = BASE_RETRY_SLEEP * attempt
            print(
                f"[retry] wallet={wallet} tag={tag} "
                f"attempt={attempt}/{MAX_RETRIES} error={e}"
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed wallet={wallet} tag={tag}: {last_error}")


def fetch_profit_stat(wallet: str, window: str) -> dict:
    url = f"https://gmgn.ai/pf/api/v1/wallet/sol/{wallet}/profit_stat/{window}"
    payload = fetch_json(
        url=url,
        params=BASE_PARAMS,
        headers=make_headers(wallet),
        wallet=wallet,
        tag=f"profit_stat_{window}",
    )
    return payload.get("data") or {}


def fetch_holdings_page_30d(wallet: str, cursor: Optional[str] = None) -> dict:
    url = f"https://gmgn.ai/pf/api/v1/wallet/sol/{wallet}/holdings"
    params = HOLDINGS_BASE_PARAMS.copy()
    if cursor:
        params["cursor"] = cursor

    return fetch_json(
        url=url,
        params=params,
        headers=make_headers(wallet),
        wallet=wallet,
        tag=f"holdings_30d_cursor_{'yes' if cursor else 'no'}",
    )


def fetch_all_holdings_30d(wallet: str) -> Tuple[List[dict], int]:
    all_items = []
    seen = set()
    cursor = None
    pages = 0

    while True:
        payload = fetch_holdings_page_30d(wallet, cursor=cursor)
        data = payload.get("data") or {}
        items = data.get("list") or []
        next_cursor = data.get("next")

        added = 0
        for idx, item in enumerate(items):
            k = token_key(item, idx)
            if k in seen:
                continue
            seen.add(k)
            all_items.append(item)
            added += 1

        pages += 1
        print(
            f"[page] wallet={wallet} page={pages} "
            f"raw={len(items)} added={added} next={'yes' if next_cursor else 'no'}"
        )

        if not next_cursor or next_cursor == cursor:
            break

        cursor = next_cursor
        time.sleep(REQUEST_DELAY_BETWEEN_PAGES)

    return all_items, pages


def fetch_dev_created_tokens(wallet: str) -> dict:
    """Fetch tokens created by this wallet (dev activity check)."""
    url = f"https://gmgn.ai/api/v1/dev_created_tokens/sol/{wallet}"
    payload = fetch_json(
        url=url,
        params=BASE_PARAMS,
        headers=make_headers(wallet),
        wallet=wallet,
        tag="dev_created_tokens",
    )
    return payload.get("data") or {}


# ============================================================
# DEV ACTIVITY EXTRACTION
# ============================================================

def extract_dev_fields(data: dict) -> Dict[str, object]:
    """Extract dev creation metrics from dev_created_tokens response."""
    inner_count = to_int(data.get("inner_count"))
    open_count = to_int(data.get("open_count"))
    open_ratio = to_float(data.get("open_ratio"))

    tokens = data.get("tokens") or []

    # ATH info from best-performing created token
    ath_info = data.get("creator_ath_info") or {}
    ath_mc = to_float(ath_info.get("ath_mc"))

    # Calculate total creator fees extracted
    total_creator_fees = sum(to_float(t.get("coin_creator_fee")) for t in tokens)
    total_fees_all = sum(to_float(t.get("total_fee")) for t in tokens)

    # Count recently created tokens (last 7 days)
    cutoff_7d = now_ts() - 7 * 86400
    cutoff_30d = now_ts() - 30 * 86400
    recent_7d = sum(1 for t in tokens if to_int(t.get("create_timestamp")) >= cutoff_7d)
    recent_30d = sum(1 for t in tokens if to_int(t.get("create_timestamp")) >= cutoff_30d)

    # Average market cap of created tokens (excluding outliers)
    market_caps = [to_float(t.get("market_cap")) for t in tokens if to_float(t.get("market_cap")) > 0]
    avg_mc = sum(market_caps) / len(market_caps) if market_caps else 0.0

    # Bundler usage across created tokens
    bundler_rates = [to_float(t.get("bundler_rate")) for t in tokens]
    max_bundler_rate = max(bundler_rates) if bundler_rates else 0.0
    avg_bundler_rate = sum(bundler_rates) / len(bundler_rates) if bundler_rates else 0.0

    return {
        "dev_tokens_created": inner_count,
        "dev_tokens_open": open_count,
        "dev_open_ratio": open_ratio,
        "dev_ath_mc": ath_mc,
        "dev_total_creator_fees": round(total_creator_fees, 6),
        "dev_total_fees": round(total_fees_all, 6),
        "dev_tokens_created_7d": recent_7d,
        "dev_tokens_created_30d": recent_30d,
        "dev_avg_market_cap": round(avg_mc, 2),
        "dev_max_bundler_rate": round(max_bundler_rate, 4),
        "dev_avg_bundler_rate": round(avg_bundler_rate, 4),
    }


def empty_dev_fields() -> Dict[str, object]:
    return {
        "dev_tokens_created": None,
        "dev_tokens_open": None,
        "dev_open_ratio": None,
        "dev_ath_mc": None,
        "dev_total_creator_fees": None,
        "dev_total_fees": None,
        "dev_tokens_created_7d": None,
        "dev_tokens_created_30d": None,
        "dev_avg_market_cap": None,
        "dev_max_bundler_rate": None,
        "dev_avg_bundler_rate": None,
    }


# ============================================================
# PROFIT STAT EXTRACTION
# ============================================================

def extract_profit_stat_fields(data: dict, window: str) -> Dict[str, object]:
    pnl_detail = data.get("pnl_detail") or {}
    risk = data.get("risk") or {}

    result = {
        f"native_balance_{window}": to_float(data.get("native_balance")),
        f"realized_profit_{window}": to_float(data.get("realized_profit")),
        f"realized_profit_pnl_{window}": to_float(data.get("realized_profit_pnl")),
        f"unrealized_profit_{window}": to_float(data.get("unrealized_profit")),
        f"unrealized_profit_pnl_{window}": (
            to_float(data.get("unrealized_profit_pnl"))
            if data.get("unrealized_profit_pnl") is not None else None
        ),
        f"total_profit_{window}": to_float(data.get("total_profit")),
        f"total_profit_pnl_{window}": (
            to_float(data.get("total_profit_pnl"))
            if data.get("total_profit_pnl") is not None else None
        ),

        f"buy_{window}": to_int(data.get("buy")),
        f"sell_{window}": to_int(data.get("sell")),
        f"transfer_in_{window}": to_int(data.get("transfer_in")),
        f"transfer_out_{window}": to_int(data.get("transfer_out")),

        f"avg_holding_period_{window}": to_float(data.get("avg_holding_period")),
        f"total_bought_cost_{window}": to_float(data.get("total_bought_cost")),
        f"total_sold_income_{window}": to_float(data.get("total_sold_income")),
        f"total_transfer_in_cost_{window}": to_float(data.get("total_transfer_in_cost")),
        f"total_transfer_out_income_{window}": to_float(data.get("total_transfer_out_income")),
        f"total_fee_{window}": to_float(data.get("total_fee")),
        f"total_fee_usd_{window}": to_float(data.get("total_fee_usd")),
        f"last_active_timestamp_{window}": to_int(data.get("last_active_timestamp")),

        f"winrate_{window}": to_float(pnl_detail.get("winrate")),
        f"token_num_{window}": to_int(pnl_detail.get("token_num")),
        f"pnl_lt_nd5_num_{window}": to_int(pnl_detail.get("pnl_lt_nd5_num")),
        f"pnl_nd5_0x_num_{window}": to_int(pnl_detail.get("pnl_nd5_0x_num")),
        f"pnl_0x_2x_num_{window}": to_int(pnl_detail.get("pnl_0x_2x_num")),
        f"pnl_2x_5x_num_{window}": to_int(pnl_detail.get("pnl_2x_5x_num")),
        f"pnl_gt_5x_num_{window}": to_int(pnl_detail.get("pnl_gt_5x_num")),
        f"first_market_cap_num_{window}": to_int(pnl_detail.get("first_market_cap_num")),
        f"second_market_cap_num_{window}": to_int(pnl_detail.get("second_market_cap_num")),
        f"third_market_cap_num_{window}": to_int(pnl_detail.get("third_market_cap_num")),

        f"risk_token_active_{window}": to_int(risk.get("token_active")),
        f"risk_token_honeypot_{window}": to_int(risk.get("token_honeypot")),
        f"risk_token_honeypot_ratio_{window}": to_float(risk.get("token_honeypot_ratio")),
        f"risk_no_buy_hold_{window}": to_int(risk.get("no_buy_hold")),
        f"risk_no_buy_hold_ratio_{window}": to_float(risk.get("no_buy_hold_ratio")),
        f"risk_sell_pass_buy_{window}": to_int(risk.get("sell_pass_buy")),
        f"risk_sell_pass_buy_ratio_{window}": to_float(risk.get("sell_pass_buy_ratio")),
        f"risk_fast_tx_{window}": to_int(risk.get("fast_tx")),
        f"risk_fast_tx_ratio_{window}": to_float(risk.get("fast_tx_ratio")),
    }
    return result


# ============================================================
# DAILY PROFIT % METRICS
# ============================================================

def compute_daily_roi_fields(row: dict) -> Dict[str, object]:
    """
    Compute daily ROI % for each window.
    daily_roi = realized_profit / total_bought_cost / days_in_window
    This normalises profit relative to capital deployed, not absolute $.
    """
    result = {}

    for window, days in [("1d", 1), ("7d", 7), ("30d", 30)]:
        realized = to_float(row.get(f"realized_profit_{window}"))
        cost = to_float(row.get(f"total_bought_cost_{window}"))

        if cost > 0 and days > 0:
            roi_total = realized / cost           # total ROI over window
            roi_daily = roi_total / days           # average daily ROI
        else:
            roi_total = 0.0
            roi_daily = 0.0

        result[f"roi_total_{window}"] = round(roi_total, 6)
        result[f"roi_daily_{window}"] = round(roi_daily, 6)

    return result


# ============================================================
# ACTIVITY CONSISTENCY METRICS
# ============================================================

def compute_activity_fields(row: dict) -> Dict[str, object]:
    """
    Compute activity recency & consistency from profit_stat timestamps.
    Uses last_active_timestamp from each window + token counts.
    """
    current_ts = now_ts()

    last_active_1d = to_int(row.get("last_active_timestamp_1d"))
    last_active_7d = to_int(row.get("last_active_timestamp_7d"))
    last_active_30d = to_int(row.get("last_active_timestamp_30d"))

    # Most recent activity across all windows
    last_active = max(last_active_1d, last_active_7d, last_active_30d)

    if last_active > 0:
        hours_since_last_trade = round((current_ts - last_active) / 3600, 1)
        days_since_last_trade = round((current_ts - last_active) / 86400, 2)
    else:
        hours_since_last_trade = 9999.0
        days_since_last_trade = 9999.0

    # Token counts per window — proxy for activity volume
    token_num_1d = to_int(row.get("token_num_1d"))
    token_num_7d = to_int(row.get("token_num_7d"))
    token_num_30d = to_int(row.get("token_num_30d"))

    # Daily averages
    avg_tokens_per_day_7d = round(token_num_7d / 7, 2) if token_num_7d else 0.0
    avg_tokens_per_day_30d = round(token_num_30d / 30, 2) if token_num_30d else 0.0

    # Consistency ratio: how much of 7d activity is concentrated vs spread
    # If token_num_7d ≈ 7 * token_num_1d, wallet trades daily at same rate
    # If token_num_7d ≈ token_num_1d, all activity was in one day (burst)
    if token_num_7d > 0 and token_num_1d > 0:
        burst_ratio_7d = round(token_num_1d / avg_tokens_per_day_7d, 2) if avg_tokens_per_day_7d > 0 else 0.0
    else:
        burst_ratio_7d = 0.0

    # Is active today?
    active_today = token_num_1d > 0

    # Estimated active days from token distribution
    # Heuristic: if 7d has N tokens at avg X/day, active days ≈ min(7, N/max(X, 1))
    # Better: use buy+sell counts
    buy_7d = to_int(row.get("buy_7d"))
    sell_7d = to_int(row.get("sell_7d"))
    total_txs_7d = buy_7d + sell_7d

    buy_30d = to_int(row.get("buy_30d"))
    sell_30d = to_int(row.get("sell_30d"))
    total_txs_30d = buy_30d + sell_30d

    return {
        "last_active_ts": last_active,
        "hours_since_last_trade": hours_since_last_trade,
        "days_since_last_trade": days_since_last_trade,
        "active_today": active_today,
        "avg_tokens_per_day_7d": avg_tokens_per_day_7d,
        "avg_tokens_per_day_30d": avg_tokens_per_day_30d,
        "burst_ratio_7d": burst_ratio_7d,
        "total_txs_7d": total_txs_7d,
        "total_txs_30d": total_txs_30d,
    }


def compute_deep_activity_fields(items: List[dict]) -> Dict[str, object]:
    """
    Compute per-day activity distribution from 30d holdings data.
    Each token's start_holding_at gives us the day it was first traded.
    """
    current_ts = now_ts()
    cutoff_1d = current_ts - 1 * 86400
    cutoff_3d = current_ts - 3 * 86400
    cutoff_7d = current_ts - 7 * 86400
    cutoff_30d = current_ts - 30 * 86400

    tokens_1d = 0
    tokens_3d = 0
    tokens_7d = 0
    tokens_30d = 0

    active_days = set()  # set of day offsets (0=today, 1=yesterday, ...)

    for item in items:
        start_ts = to_int(item.get("start_holding_at"))
        last_ts = to_int(item.get("last_active_timestamp"))

        # Use the most recent timestamp for this token
        ts = max(start_ts, last_ts) if last_ts > 0 else start_ts

        if ts <= 0:
            continue

        if ts >= cutoff_1d:
            tokens_1d += 1
        if ts >= cutoff_3d:
            tokens_3d += 1
        if ts >= cutoff_7d:
            tokens_7d += 1
        if ts >= cutoff_30d:
            tokens_30d += 1

        # Track which days had activity
        day_offset = (current_ts - ts) // 86400
        if 0 <= day_offset < 30:
            active_days.add(day_offset)

    active_days_count_7d = len([d for d in active_days if d < 7])
    active_days_count_30d = len(active_days)

    # Consistency score: active_days / total_days (0-1, higher = more consistent)
    consistency_7d = round(active_days_count_7d / 7, 3)
    consistency_30d = round(active_days_count_30d / 30, 3)

    # Longest gap between active days in last 30d
    sorted_days = sorted(active_days) if active_days else []
    max_gap_days = 0
    if len(sorted_days) > 1:
        for i in range(1, len(sorted_days)):
            gap = sorted_days[i] - sorted_days[i - 1]
            if gap > max_gap_days:
                max_gap_days = gap

    return {
        "deep_tokens_active_1d": tokens_1d,
        "deep_tokens_active_3d": tokens_3d,
        "deep_tokens_active_7d": tokens_7d,
        "deep_tokens_active_30d": tokens_30d,
        "deep_active_days_7d": active_days_count_7d,
        "deep_active_days_30d": active_days_count_30d,
        "deep_consistency_7d": consistency_7d,
        "deep_consistency_30d": consistency_30d,
        "deep_max_gap_days_30d": max_gap_days,
    }


# ============================================================
# PREFILTER
# ============================================================

def build_prefilter_skip_reason(row: dict) -> str:
    reasons = []

    token_num_7d = to_int(row.get("token_num_7d"))
    token_num_30d = to_int(row.get("token_num_30d"))
    fast_tx_ratio_30d = to_float(row.get("risk_fast_tx_ratio_30d"))
    avg_holding_period_30d = to_float(row.get("avg_holding_period_30d"))
    realized_profit_pnl_7d = to_float(row.get("realized_profit_pnl_7d"))
    median_holding_seconds = to_float(row.get("median_holding_seconds"))
    median_position_size_usd = to_float(row.get("median_position_size_usd"))
    days_since_last = to_float(row.get("days_since_last_trade"))
    dev_tokens_created = to_int(row.get("dev_tokens_created"))

    if token_num_7d > TOKEN_NUM_7D_HARD_REJECT_GT:
        reasons.append(f"token_num_7d_gt_{TOKEN_NUM_7D_HARD_REJECT_GT}")
    elif token_num_7d > TOKEN_NUM_7D_SKIP_GT:
        reasons.append(f"token_num_7d_gt_{TOKEN_NUM_7D_SKIP_GT}")

    if token_num_30d > TOKEN_NUM_30D_SKIP_GT:
        reasons.append(f"token_num_30d_gt_{TOKEN_NUM_30D_SKIP_GT}")

    if fast_tx_ratio_30d > FAST_TX_RATIO_30D_HARD_REJECT_GT:
        reasons.append(f"fast_tx_ratio_30d_gt_{FAST_TX_RATIO_30D_HARD_REJECT_GT}")
    elif fast_tx_ratio_30d > FAST_TX_RATIO_30D_SKIP_GT:
        reasons.append(f"fast_tx_ratio_30d_gt_{FAST_TX_RATIO_30D_SKIP_GT}")

    if avg_holding_period_30d < AVG_HOLDING_PERIOD_30D_SKIP_LT:
        reasons.append(f"avg_holding_period_30d_lt_{int(AVG_HOLDING_PERIOD_30D_SKIP_LT)}")

    if realized_profit_pnl_7d <= 0:
        reasons.append("realized_profit_pnl_7d_non_positive")

    if median_holding_seconds < MIN_MEDIAN_HOLDING_SECONDS:
        reasons.append(f"median_holding_seconds_lt_{MIN_MEDIAN_HOLDING_SECONDS}")

    if not (MIN_POSITION_SIZE_USD <= median_position_size_usd <= MAX_POSITION_SIZE_USD):
        reasons.append("median_position_size_out_of_range")

    if days_since_last > DAYS_SINCE_LAST_TRADE_REJECT_GT:
        reasons.append(f"inactive_{days_since_last:.0f}d")

    if dev_tokens_created is not None and dev_tokens_created > DEV_CREATED_TOKENS_REJECT_GT:
        reasons.append(f"dev_created_{dev_tokens_created}_tokens")

    return ",".join(reasons)


def should_skip_deep_holdings(row: dict) -> Tuple[bool, str]:
    reason = build_prefilter_skip_reason(row)
    return (len(reason) > 0, reason)


def prefilter_score(row: dict) -> float:
    score = 0.0

    winrate_csv = to_float(row.get("winrate_percentage"))
    closed_positions = to_float(row.get("closed_positions"))
    median_holding_seconds = to_float(row.get("median_holding_seconds"))
    median_position_size_usd = to_float(row.get("median_position_size_usd"))

    realized_profit_1d = to_float(row.get("realized_profit_1d"))
    realized_profit_7d = to_float(row.get("realized_profit_7d"))
    realized_profit_30d = to_float(row.get("realized_profit_30d"))

    realized_profit_pnl_1d = to_float(row.get("realized_profit_pnl_1d"))
    realized_profit_pnl_7d = to_float(row.get("realized_profit_pnl_7d"))
    realized_profit_pnl_30d = to_float(row.get("realized_profit_pnl_30d"))

    token_num_1d = to_int(row.get("token_num_1d"))
    token_num_7d = to_int(row.get("token_num_7d"))
    token_num_30d = to_int(row.get("token_num_30d"))

    avg_holding_period_30d = to_float(row.get("avg_holding_period_30d"))
    fast_tx_ratio_30d = to_float(row.get("risk_fast_tx_ratio_30d"))
    honeypot_ratio_30d = to_float(row.get("risk_token_honeypot_ratio_30d"))
    no_buy_hold_ratio_30d = to_float(row.get("risk_no_buy_hold_ratio_30d"))

    # ── Daily ROI % instead of absolute profit ──
    roi_daily_1d = to_float(row.get("roi_daily_1d"))
    roi_daily_7d = to_float(row.get("roi_daily_7d"))
    roi_daily_30d = to_float(row.get("roi_daily_30d"))

    # ── Activity & consistency ──
    days_since_last = to_float(row.get("days_since_last_trade"))
    burst_ratio_7d = to_float(row.get("burst_ratio_7d"))

    # ── Dev fields ──
    dev_tokens_created = to_int(row.get("dev_tokens_created"))

    # -- Base score components --
    score += min(winrate_csv, 80.0) * 0.65
    score += min(closed_positions, 300.0) * 0.05

    # -- ROI-based scoring (replaces raw profit $ for fairer comparison) --
    score += max(roi_daily_30d, 0.0) * 800.0     # e.g. 0.01 (1%/day) → +8
    score += max(roi_daily_7d, 0.0) * 1200.0     # 7d daily ROI weighted more
    score += max(roi_daily_1d, 0.0) * 400.0

    # -- Still use log(profit) but with less weight (rewards absolute scale) --
    score += math.log1p(max(realized_profit_30d, 0.0)) * 2.0
    score += math.log1p(max(realized_profit_7d, 0.0)) * 3.0
    score += math.log1p(max(realized_profit_1d, 0.0)) * 1.0

    # -- PnL ratio still useful --
    score += max(realized_profit_pnl_30d, 0.0) * 10.0
    score += max(realized_profit_pnl_7d, 0.0) * 15.0
    score += max(realized_profit_pnl_1d, 0.0) * 5.0

    score += min(token_num_1d, 20) * 0.5
    score += min(token_num_7d, 50) * 0.4

    # -- Holding period penalties --
    if median_holding_seconds < 60:
        score -= 35
    elif median_holding_seconds < 90:
        score -= 15

    if avg_holding_period_30d < 60:
        score -= 30
    elif avg_holding_period_30d < 120:
        score -= 10

    # -- Position size penalties --
    if median_position_size_usd < MIN_POSITION_SIZE_USD:
        score -= 25
    if median_position_size_usd > MAX_POSITION_SIZE_USD:
        score -= min(30, (median_position_size_usd - MAX_POSITION_SIZE_USD) / 50.0)

    # -- Risk penalties --
    score -= fast_tx_ratio_30d * 90.0
    score -= honeypot_ratio_30d * 100.0
    score -= no_buy_hold_ratio_30d * 20.0

    if token_num_7d > 150:
        score -= min(50.0, (token_num_7d - 150) / 5.0)
    if token_num_30d > 1000:
        score -= min(40.0, (token_num_30d - 1000) / 25.0)

    if realized_profit_pnl_7d <= 0:
        score -= 20

    # -- Activity recency penalties (critical for copytrade) --
    if days_since_last > DAYS_SINCE_LAST_TRADE_REJECT_GT:
        score -= 50
    elif days_since_last > DAYS_SINCE_LAST_TRADE_WARN_GT:
        score -= days_since_last * 8.0

    # Burst penalty: if all 7d activity is concentrated in 1 day
    if burst_ratio_7d > 5.0 and token_num_7d > 5:
        score -= 15

    # -- Dev penalty --
    if dev_tokens_created is not None:
        if dev_tokens_created > DEV_CREATED_TOKENS_REJECT_GT:
            score -= 100
        elif dev_tokens_created > DEV_CREATED_TOKENS_WARN_GT:
            score -= dev_tokens_created * 5.0
        elif dev_tokens_created > 0:
            score -= dev_tokens_created * 2.0

    return round(score, 3)


def prefilter_label(row: dict) -> str:
    token_num_7d = to_int(row.get("token_num_7d"))
    fast_tx_ratio_30d = to_float(row.get("risk_fast_tx_ratio_30d"))
    honeypot_ratio_30d = to_float(row.get("risk_token_honeypot_ratio_30d"))
    avg_holding_period_30d = to_float(row.get("avg_holding_period_30d"))
    realized_profit_pnl_7d = to_float(row.get("realized_profit_pnl_7d"))
    winrate_csv = to_float(row.get("winrate_percentage"))
    median_holding_seconds = to_float(row.get("median_holding_seconds"))
    median_position_size_usd = to_float(row.get("median_position_size_usd"))
    days_since_last = to_float(row.get("days_since_last_trade"))
    dev_tokens_created = to_int(row.get("dev_tokens_created"))

    hard_reject = (
        token_num_7d > TOKEN_NUM_7D_HARD_REJECT_GT
        or fast_tx_ratio_30d > FAST_TX_RATIO_30D_HARD_REJECT_GT
        or honeypot_ratio_30d > 0
        or avg_holding_period_30d < AVG_HOLDING_PERIOD_30D_SKIP_LT
        or median_holding_seconds < MIN_MEDIAN_HOLDING_SECONDS
        or not (MIN_POSITION_SIZE_USD <= median_position_size_usd <= MAX_POSITION_SIZE_USD)
        or days_since_last > DAYS_SINCE_LAST_TRADE_REJECT_GT
        or (dev_tokens_created is not None and dev_tokens_created > DEV_CREATED_TOKENS_REJECT_GT)
    )
    if hard_reject:
        return "reject"

    strong = (
        realized_profit_pnl_7d > 0
        and winrate_csv >= 45
        and token_num_7d >= 3
        and token_num_7d <= 200
        and fast_tx_ratio_30d <= 0.15
        and avg_holding_period_30d >= 120
        and days_since_last <= DAYS_SINCE_LAST_TRADE_WARN_GT
        and (dev_tokens_created is None or dev_tokens_created <= DEV_CREATED_TOKENS_WARN_GT)
    )
    if strong:
        return "good"

    return "watch"


def build_prefilter_flags(row: dict) -> str:
    flags = []

    token_num_7d = to_int(row.get("token_num_7d"))
    fast_tx_ratio_30d = to_float(row.get("risk_fast_tx_ratio_30d"))
    honeypot_ratio_30d = to_float(row.get("risk_token_honeypot_ratio_30d"))
    avg_holding_period_30d = to_float(row.get("avg_holding_period_30d"))
    median_holding_seconds = to_float(row.get("median_holding_seconds"))
    median_position_size_usd = to_float(row.get("median_position_size_usd"))
    realized_profit_pnl_7d = to_float(row.get("realized_profit_pnl_7d"))
    days_since_last = to_float(row.get("days_since_last_trade"))
    dev_tokens_created = to_int(row.get("dev_tokens_created"))
    burst_ratio_7d = to_float(row.get("burst_ratio_7d"))

    if token_num_7d > 200:
        flags.append("elevated_token_num_7d")
    if token_num_7d > TOKEN_NUM_7D_SKIP_GT:
        flags.append("high_token_num_7d")
    if fast_tx_ratio_30d > FAST_TX_RATIO_30D_SKIP_GT:
        flags.append("high_fast_tx_ratio_30d")
    if honeypot_ratio_30d > 0:
        flags.append("honeypot_exposure_30d")
    if avg_holding_period_30d < AVG_HOLDING_PERIOD_30D_SKIP_LT:
        flags.append("short_avg_holding_period_30d")
    if median_holding_seconds < MIN_MEDIAN_HOLDING_SECONDS:
        flags.append("too_fast_by_source_csv")
    if not (MIN_POSITION_SIZE_USD <= median_position_size_usd <= MAX_POSITION_SIZE_USD):
        flags.append("position_size_out_of_range")
    if realized_profit_pnl_7d <= 0:
        flags.append("non_positive_realized_pnl_7d")
    if days_since_last > DAYS_SINCE_LAST_TRADE_REJECT_GT:
        flags.append(f"inactive_{days_since_last:.0f}d")
    elif days_since_last > DAYS_SINCE_LAST_TRADE_WARN_GT:
        flags.append(f"recently_inactive_{days_since_last:.1f}d")
    if dev_tokens_created is not None and dev_tokens_created > DEV_CREATED_TOKENS_WARN_GT:
        flags.append(f"dev_created_{dev_tokens_created}_tokens")
    if burst_ratio_7d > 5.0 and to_int(row.get("token_num_7d")) > 5:
        flags.append("burst_activity_pattern")

    return ",".join(flags)


# ============================================================
# DEEP 30D HOLDINGS ANALYSIS
# ============================================================

def analyse_holdings_30d(wallet: str, items: List[dict], pages_fetched: int) -> Tuple[Dict[str, object], List[dict]]:
    total_profit = Decimal("0")
    total_realized = Decimal("0")
    total_unrealized = Decimal("0")
    total_cost_basis = Decimal("0")

    sniped_exact_count = 0
    sniped_5s_count = 0
    sniped_exact_profit = Decimal("0")
    sniped_5s_profit = Decimal("0")

    top_token_profit = Decimal("0")
    positive_profit_sum = Decimal("0")

    open_tokens = 0
    closed_tokens = 0
    profitable_tokens = 0
    losing_tokens = 0

    token_rows = []

    for idx, item in enumerate(items):
        token = item.get("token") or {}

        token_address = token.get("token_address") or ""
        symbol = token.get("symbol") or ""
        name = token.get("name") or ""

        creation_ts = to_int(token.get("creation_timestamp"))
        start_holding_at = to_int(item.get("start_holding_at"))
        end_holding_at = to_int(item.get("end_holding_at"))
        last_active_timestamp = to_int(item.get("last_active_timestamp"))

        balance = D(item.get("balance"))
        usd_value = D(item.get("usd_value"))

        realized_profit = D(item.get("realized_profit"))
        unrealized_profit = D(item.get("unrealized_profit"))
        token_total_profit = D(item.get("total_profit"))

        history_bought_cost = D(item.get("history_bought_cost"))
        history_bought_fee = D(item.get("history_bought_fee"))
        history_transfer_in_cost = D(item.get("history_transfer_in_cost"))
        cost_basis = history_bought_cost + history_bought_fee + history_transfer_in_cost

        total_profit += token_total_profit
        total_realized += realized_profit
        total_unrealized += unrealized_profit
        total_cost_basis += cost_basis

        if token_total_profit > 0:
            profitable_tokens += 1
            positive_profit_sum += token_total_profit
            if token_total_profit > top_token_profit:
                top_token_profit = token_total_profit
        elif token_total_profit < 0:
            losing_tokens += 1

        if balance > 0:
            open_tokens += 1
        else:
            closed_tokens += 1

        sniper_delay_seconds = None
        sniped_exact = False
        sniped_5s = False

        if creation_ts > 0 and start_holding_at > 0:
            sniper_delay_seconds = start_holding_at - creation_ts

            if sniper_delay_seconds == 0:
                sniped_exact = True
                sniped_exact_count += 1
                sniped_exact_profit += token_total_profit

            if 0 <= sniper_delay_seconds <= 5:
                sniped_5s = True
                sniped_5s_count += 1
                sniped_5s_profit += token_total_profit

        token_rows.append({
            "wallet": wallet,
            "token_address": token_address,
            "symbol": symbol,
            "name": name,
            "creation_timestamp": creation_ts,
            "start_holding_at": start_holding_at,
            "end_holding_at": end_holding_at,
            "last_active_timestamp": last_active_timestamp,
            "sniper_delay_seconds": sniper_delay_seconds if sniper_delay_seconds is not None else None,
            "sniped_exact": sniped_exact,
            "sniped_5s": sniped_5s,
            "balance": float(balance),
            "usd_value": float(usd_value),
            "realized_profit_30d_token": float(realized_profit),
            "unrealized_profit_30d_token": float(unrealized_profit),
            "total_profit_30d_token": float(token_total_profit),
            "cost_basis_30d_token": float(cost_basis),
        })

    token_count = len(items)
    sniper_rate_exact = float(safe_div(Decimal(sniped_exact_count), Decimal(token_count))) if token_count else 0.0
    sniper_rate_5s = float(safe_div(Decimal(sniped_5s_count), Decimal(token_count))) if token_count else 0.0
    top_token_profit_share = float(safe_div(top_token_profit, positive_profit_sum)) if positive_profit_sum > 0 else 0.0
    holdings_pnl_ratio = float(safe_div(total_profit, total_cost_basis)) if total_cost_basis > 0 else 0.0
    unrealized_share = float(safe_div(total_unrealized, total_profit.copy_abs())) if total_profit != 0 else 0.0

    # ── Activity distribution from holdings ──
    activity_fields = compute_deep_activity_fields(items)

    summary = {
        "deep_done": True,
        "deep_error": None,

        "holdings_pages_fetched_30d": pages_fetched,
        "holdings_token_count_30d": token_count,
        "holdings_open_tokens_30d": open_tokens,
        "holdings_closed_tokens_30d": closed_tokens,
        "holdings_profitable_tokens_30d": profitable_tokens,
        "holdings_losing_tokens_30d": losing_tokens,

        "holdings_realized_profit_30d": float(total_realized),
        "holdings_unrealized_profit_30d": float(total_unrealized),
        "holdings_total_profit_30d": float(total_profit),
        "holdings_cost_basis_30d": float(total_cost_basis),
        "holdings_pnl_ratio_30d": holdings_pnl_ratio,
        "holdings_unrealized_share_30d": unrealized_share,

        "sniped_exact_count_30d": sniped_exact_count,
        "sniped_5s_count_30d": sniped_5s_count,
        "sniped_exact_profit_30d": float(sniped_exact_profit),
        "sniped_5s_profit_30d": float(sniped_5s_profit),
        "sniper_rate_exact_30d": sniper_rate_exact,
        "sniper_rate_5s_30d": sniper_rate_5s,
        "top_token_profit_share_30d": top_token_profit_share,

        # Activity from deep analysis
        **activity_fields,
    }

    return summary, token_rows


def empty_deep_fields() -> Dict[str, object]:
    return {
        "deep_done": False,
        "deep_error": None,
        "holdings_pages_fetched_30d": None,
        "holdings_token_count_30d": None,
        "holdings_open_tokens_30d": None,
        "holdings_closed_tokens_30d": None,
        "holdings_profitable_tokens_30d": None,
        "holdings_losing_tokens_30d": None,
        "holdings_realized_profit_30d": None,
        "holdings_unrealized_profit_30d": None,
        "holdings_total_profit_30d": None,
        "holdings_cost_basis_30d": None,
        "holdings_pnl_ratio_30d": None,
        "holdings_unrealized_share_30d": None,
        "sniped_exact_count_30d": None,
        "sniped_5s_count_30d": None,
        "sniped_exact_profit_30d": None,
        "sniped_5s_profit_30d": None,
        "sniper_rate_exact_30d": None,
        "sniper_rate_5s_30d": None,
        "top_token_profit_share_30d": None,
        "deep_tokens_active_1d": None,
        "deep_tokens_active_3d": None,
        "deep_tokens_active_7d": None,
        "deep_tokens_active_30d": None,
        "deep_active_days_7d": None,
        "deep_active_days_30d": None,
        "deep_consistency_7d": None,
        "deep_consistency_30d": None,
        "deep_max_gap_days_30d": None,
    }


# ============================================================
# FINAL SCORING
# ============================================================

def final_score(row: dict) -> float:
    # Use the stored prefilter_score instead of recomputing
    score = to_float(row.get("prefilter_score"))

    if row.get("deep_done"):
        sniper_rate_5s_30d = to_float(row.get("sniper_rate_5s_30d"))
        top_token_profit_share_30d = to_float(row.get("top_token_profit_share_30d"))
        unrealized_share_30d = to_float(row.get("holdings_unrealized_share_30d"))
        consistency_7d = to_float(row.get("deep_consistency_7d"))
        consistency_30d = to_float(row.get("deep_consistency_30d"))
        max_gap = to_int(row.get("deep_max_gap_days_30d"))
        active_days_7d = to_int(row.get("deep_active_days_7d"))

        # Sniper / one-hit-wonder penalties
        score -= sniper_rate_5s_30d * 90.0
        score -= top_token_profit_share_30d * 45.0
        score -= unrealized_share_30d * 20.0

        if sniper_rate_5s_30d > SNIPER_RATE_5S_WARN_GT:
            score -= 10
        if top_token_profit_share_30d > TOP_TOKEN_PROFIT_SHARE_WARN_GT:
            score -= 10

        # ── Consistency bonuses/penalties from deep data ──
        # Reward consistent daily activity
        score += consistency_7d * 20.0    # max +20 if active every day
        score += consistency_30d * 15.0   # max +15 if active every day

        # Penalise large gaps
        if max_gap > 5:
            score -= (max_gap - 5) * 3.0

        # Penalise low active days
        if active_days_7d < ACTIVE_DAYS_7D_MIN:
            score -= (ACTIVE_DAYS_7D_MIN - active_days_7d) * 8.0

    return round(score, 3)


def final_label(row: dict) -> str:
    pre_label = row.get("prefilter_label")
    if pre_label == "reject":
        return "reject"

    if not row.get("deep_done"):
        return "watch"

    sniper_rate_5s_30d = to_float(row.get("sniper_rate_5s_30d"))
    top_token_profit_share_30d = to_float(row.get("top_token_profit_share_30d"))
    fast_tx_ratio_30d = to_float(row.get("risk_fast_tx_ratio_30d"))
    realized_profit_pnl_7d = to_float(row.get("realized_profit_pnl_7d"))
    token_num_7d = to_int(row.get("token_num_7d"))
    winrate_csv = to_float(row.get("winrate_percentage"))
    consistency_7d = to_float(row.get("deep_consistency_7d"))
    active_days_7d = to_int(row.get("deep_active_days_7d"))

    hard_reject = (
        sniper_rate_5s_30d > SNIPER_RATE_5S_REJECT_GT
        or top_token_profit_share_30d > TOP_TOKEN_PROFIT_SHARE_REJECT_GT
    )
    if hard_reject:
        return "reject"

    strong = (
        realized_profit_pnl_7d > 0
        and token_num_7d >= 3
        and token_num_7d <= 200
        and fast_tx_ratio_30d <= 0.15
        and sniper_rate_5s_30d <= SNIPER_RATE_5S_WARN_GT
        and top_token_profit_share_30d <= TOP_TOKEN_PROFIT_SHARE_WARN_GT
        and winrate_csv >= 45
        and active_days_7d >= ACTIVE_DAYS_7D_MIN
        and consistency_7d >= 0.4
    )
    if strong:
        return "good"

    return "watch"


def build_final_flags(row: dict) -> str:
    flags = []

    # Prefilter flags
    prefilter_flags = row.get("prefilter_flags")
    if prefilter_flags:
        flags.extend([x for x in str(prefilter_flags).split(",") if x])

    # Deep flags
    if row.get("deep_done"):
        sniper_rate_5s_30d = to_float(row.get("sniper_rate_5s_30d"))
        top_token_profit_share_30d = to_float(row.get("top_token_profit_share_30d"))
        unrealized_share_30d = to_float(row.get("holdings_unrealized_share_30d"))
        consistency_7d = to_float(row.get("deep_consistency_7d"))
        max_gap = to_int(row.get("deep_max_gap_days_30d"))
        active_days_7d = to_int(row.get("deep_active_days_7d"))

        if sniper_rate_5s_30d > SNIPER_RATE_5S_WARN_GT:
            flags.append("high_sniper_rate_30d")
        if top_token_profit_share_30d > TOP_TOKEN_PROFIT_SHARE_WARN_GT:
            flags.append("one_hit_wonder_risk")
        if unrealized_share_30d > 0.60:
            flags.append("too_much_unrealized_30d")
        if consistency_7d < 0.3:
            flags.append("low_consistency_7d")
        if max_gap > 5:
            flags.append(f"gap_{max_gap}d_in_30d")
        if active_days_7d < ACTIVE_DAYS_7D_MIN:
            flags.append(f"only_{active_days_7d}_active_days_7d")

    return ",".join(dict.fromkeys(flags))


# ============================================================
# PIPELINE PHASE 1: PREFILTER
# ============================================================

def fetch_prefilter_for_wallet(base_row: dict) -> dict:
    wallet = str(base_row["wallet"]).strip()

    row = dict(base_row)
    row["wallet"] = wallet
    row["prefilter_done"] = False
    row["prefilter_error"] = None

    # Fetch profit stats with delay between calls
    stats_1d = extract_profit_stat_fields(fetch_profit_stat(wallet, "1d"), "1d")
    time.sleep(REQUEST_DELAY_BETWEEN_STATS)

    stats_7d = extract_profit_stat_fields(fetch_profit_stat(wallet, "7d"), "7d")
    time.sleep(REQUEST_DELAY_BETWEEN_STATS)

    stats_30d = extract_profit_stat_fields(fetch_profit_stat(wallet, "30d"), "30d")
    time.sleep(REQUEST_DELAY_BETWEEN_STATS)

    row.update(stats_1d)
    row.update(stats_7d)
    row.update(stats_30d)

    # Fetch dev created tokens
    try:
        dev_data = fetch_dev_created_tokens(wallet)
        dev_fields = extract_dev_fields(dev_data)
        row.update(dev_fields)
    except Exception as e:
        print(f"[dev check warning] wallet={wallet} error={e}")
        row.update(empty_dev_fields())

    # Compute derived metrics
    row.update(compute_daily_roi_fields(row))
    row.update(compute_activity_fields(row))

    skip_deep, skip_reason = should_skip_deep_holdings(row)
    row["skip_holdings_30d"] = skip_deep
    row["skip_holdings_reason"] = skip_reason
    row["prefilter_score"] = prefilter_score(row)
    row["prefilter_label"] = prefilter_label(row)
    row["prefilter_flags"] = build_prefilter_flags(row)
    row["prefilter_done"] = True

    return row


# ============================================================
# PIPELINE PHASE 2: DEEP 30D HOLDINGS
# ============================================================

def fetch_deep_for_wallet(wallet: str) -> Tuple[str, dict, List[dict]]:
    items, pages = fetch_all_holdings_30d(wallet)
    summary, wallet_token_rows = analyse_holdings_30d(wallet, items, pages)
    return wallet, summary, wallet_token_rows


# ============================================================
# BUILD OUTPUT TABLES
# ============================================================

def build_final_wallet_rows(prefilter_rows: List[dict], deep_map: Dict[str, dict]) -> List[dict]:
    final_rows = []

    for row in prefilter_rows:
        wallet = str(row["wallet"]).strip()
        merged = dict(row)

        # Only apply empty defaults for fields not already present
        deep_defaults = empty_deep_fields()
        for k, v in deep_defaults.items():
            if k not in merged:
                merged[k] = v

        if wallet in deep_map:
            merged.update(deep_map[wallet])

        merged["final_score"] = final_score(merged)
        merged["final_label"] = final_label(merged)
        merged["final_flags"] = build_final_flags(merged)
        final_rows.append(merged)

    return final_rows


def build_deep_wallet_rows(final_rows: List[dict]) -> List[dict]:
    rows = [r for r in final_rows if r.get("deep_done")]
    return rows


def sort_wallet_df_for_output(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "final_label" in df.columns:
        df["__final_label_sort"] = df["final_label"].map({
            "good": 0,
            "watch": 1,
            "reject": 2,
        }).fillna(9)
    elif "prefilter_label" in df.columns:
        df["__final_label_sort"] = df["prefilter_label"].map({
            "good": 0,
            "watch": 1,
            "reject": 2,
        }).fillna(9)
    else:
        df["__final_label_sort"] = 9

    sort_cols = []
    ascending = []

    if "__final_label_sort" in df.columns:
        sort_cols.append("__final_label_sort")
        ascending.append(True)

    if "final_score" in df.columns:
        sort_cols.append("final_score")
        ascending.append(False)
    elif "prefilter_score" in df.columns:
        sort_cols.append("prefilter_score")
        ascending.append(False)

    if "realized_profit_pnl_7d" in df.columns:
        sort_cols.append("realized_profit_pnl_7d")
        ascending.append(False)

    if "realized_profit_7d" in df.columns:
        sort_cols.append("realized_profit_7d")
        ascending.append(False)

    df = df.sort_values(by=sort_cols, ascending=ascending)
    return df.drop(columns=["__final_label_sort"], errors="ignore")


def save_checkpoint(
    prefilter_rows: List[dict],
    deep_map: Dict[str, dict],
    token_rows: List[dict],
    note: str,
) -> None:
    final_rows = build_final_wallet_rows(prefilter_rows, deep_map)
    deep_wallet_rows = build_deep_wallet_rows(final_rows)

    export_final_rows = export_wallet_rows(final_rows)
    export_deep_rows = export_wallet_rows(deep_wallet_rows)

    try:
        atomic_write_excel(
            all_wallet_rows=export_final_rows,
            deep_wallet_rows=export_deep_rows,
            token_rows=token_rows,
            output_path=OUTPUT_XLSX,
        )
    except Exception as e:
        print(f"[checkpoint warning] Excel save failed: {e}")

    atomic_write_json(export_final_rows, OUTPUT_JSON)
    atomic_write_csv(export_final_rows, OUTPUT_FINAL_CSV)
    atomic_write_csv(export_deep_rows, OUTPUT_DEEP_CSV)

    if SAVE_TOKEN_CSV:
        atomic_write_csv(token_rows, OUTPUT_TOKEN_CSV)

    print(
        f"[checkpoint] {note} | "
        f"wallets_all={len(final_rows)} deep_wallets={len(deep_wallet_rows)} token_rows={len(token_rows)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    df = pd.read_csv(INPUT_CSV)
    if LIMIT_WALLETS:
        df = df.head(LIMIT_WALLETS)

    base_rows = normalise_records(df)

    # ------------------------
    # Resume state
    # ------------------------
    checkpoint_map = load_existing_json_checkpoint(OUTPUT_JSON)
    token_rows_all = load_existing_token_rows(OUTPUT_TOKEN_CSV)

    # Deduplicate token rows on resume
    token_rows_all = deduplicate_token_rows(token_rows_all)

    prefilter_rows: List[dict] = []
    deep_map: Dict[str, dict] = {}

    for row in checkpoint_map.values():
        wallet = str(row.get("wallet", "")).strip()
        if not wallet:
            continue

        if row.get("prefilter_done") or row.get("prefilter_label") is not None:
            prefilter_rows.append(row)

        if row.get("deep_done"):
            deep_map[wallet] = {
                k: v for k, v in row.items()
                if k in empty_deep_fields() or k in {"deep_done", "deep_error"}
            }

    existing_prefilter_wallets = {str(r["wallet"]).strip() for r in prefilter_rows}
    existing_deep_wallets = set(deep_map.keys())

    print(
        f"[resume] prefilter_done={len(existing_prefilter_wallets)} "
        f"deep_done_or_failed={len(existing_deep_wallets)}"
    )

    # ------------------------
    # Phase 1: prefilter
    # ------------------------
    remaining_prefilter_rows = [
        row for row in base_rows
        if str(row["wallet"]).strip() not in existing_prefilter_wallets
    ]

    print(
        f"Starting prefilter for {len(remaining_prefilter_rows)} wallets "
        f"(skipping {len(existing_prefilter_wallets)} already done)..."
    )

    prefilter_completed = 0

    if remaining_prefilter_rows:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_PREFILTER) as executor:
            future_map = {
                executor.submit(fetch_prefilter_for_wallet, row): str(row["wallet"]).strip()
                for row in remaining_prefilter_rows
            }

            for future in as_completed(future_map):
                wallet = future_map[future]
                try:
                    row = future.result()
                    prefilter_rows.append(row)
                    print(
                        f"[prefilter done] wallet={wallet} "
                        f"score={row['prefilter_score']} "
                        f"label={row['prefilter_label']} "
                        f"skip_holdings={row['skip_holdings_30d']} "
                        f"dev_tokens={row.get('dev_tokens_created', '?')} "
                        f"days_inactive={row.get('days_since_last_trade', '?')} "
                        f"roi_daily_7d={row.get('roi_daily_7d', '?')}"
                    )
                except Exception as e:
                    fallback = {"wallet": wallet}
                    matching = next((r for r in base_rows if str(r["wallet"]).strip() == wallet), fallback)
                    row = dict(matching)
                    row["prefilter_done"] = False
                    row["prefilter_error"] = str(e)
                    row["skip_holdings_30d"] = True
                    row["skip_holdings_reason"] = "prefilter_error"
                    row["prefilter_score"] = -999999.0
                    row["prefilter_label"] = "reject"
                    row["prefilter_flags"] = "prefilter_error"
                    row.update(empty_dev_fields())
                    prefilter_rows.append(row)
                    print(f"[prefilter error] wallet={wallet} error={e}")

                prefilter_completed += 1

                if prefilter_completed % SAVE_EVERY_N_RESULTS == 0:
                    save_checkpoint(
                        prefilter_rows=prefilter_rows,
                        deep_map=deep_map,
                        token_rows=token_rows_all,
                        note=f"prefilter {prefilter_completed}/{len(remaining_prefilter_rows)}",
                    )

    # Deduplicate by wallet, latest row wins
    prefilter_map = {}
    for row in prefilter_rows:
        wallet = str(row["wallet"]).strip()
        prefilter_map[wallet] = row
    prefilter_rows = list(prefilter_map.values())

    save_checkpoint(
        prefilter_rows=prefilter_rows,
        deep_map=deep_map,
        token_rows=token_rows_all,
        note="prefilter complete",
    )

    # ------------------------
    # Select deep-analysis candidates
    # ------------------------
    prefilter_df = pd.DataFrame(prefilter_rows)

    deep_candidates_df = prefilter_df[
        (prefilter_df["skip_holdings_30d"] == False) &
        (prefilter_df["prefilter_label"] != "reject")
    ].copy()

    deep_candidates_df = sort_wallet_df_for_output(deep_candidates_df)

    if DEEP_ANALYSIS_TOP_N is not None:
        deep_candidates_df = deep_candidates_df.head(DEEP_ANALYSIS_TOP_N)

    deep_candidate_wallets = [
        wallet for wallet in deep_candidates_df["wallet"].astype(str).tolist()
        if wallet not in existing_deep_wallets
    ]

    print(
        f"Deep-analysis candidates remaining: {len(deep_candidate_wallets)} "
        f"(already done/failed: {len(existing_deep_wallets)})"
    )

    # ------------------------
    # Phase 2: deep analysis
    # ------------------------
    deep_completed = 0

    if deep_candidate_wallets:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_DEEP) as executor:
            future_map = {
                executor.submit(fetch_deep_for_wallet, wallet): wallet
                for wallet in deep_candidate_wallets
            }

            for future in as_completed(future_map):
                wallet = future_map[future]
                try:
                    wallet_out, deep_summary, wallet_token_rows = future.result()
                    deep_map[wallet_out] = deep_summary

                    with _token_rows_lock:
                        token_rows_all.extend(wallet_token_rows)

                    print(
                        f"[deep done] wallet={wallet_out} "
                        f"pages={deep_summary.get('holdings_pages_fetched_30d')} "
                        f"tokens={deep_summary.get('holdings_token_count_30d')} "
                        f"sniped_5s={deep_summary.get('sniped_5s_count_30d')} "
                        f"active_days_7d={deep_summary.get('deep_active_days_7d')} "
                        f"consistency_7d={deep_summary.get('deep_consistency_7d')}"
                    )

                except Exception as e:
                    deep_map[wallet] = {
                        **empty_deep_fields(),
                        "deep_done": False,
                        "deep_error": str(e),
                    }
                    print(f"[deep error] wallet={wallet} error={e}")

                deep_completed += 1

                if deep_completed % SAVE_EVERY_N_RESULTS == 0:
                    save_checkpoint(
                        prefilter_rows=prefilter_rows,
                        deep_map=deep_map,
                        token_rows=token_rows_all,
                        note=f"deep {deep_completed}/{len(deep_candidate_wallets)}",
                    )

    # Deduplicate token rows before final save
    with _token_rows_lock:
        token_rows_all = deduplicate_token_rows(token_rows_all)

    # ------------------------
    # Final save
    # ------------------------
    save_checkpoint(
        prefilter_rows=prefilter_rows,
        deep_map=deep_map,
        token_rows=token_rows_all,
        note="final",
    )

    final_rows = build_final_wallet_rows(prefilter_rows, deep_map)
    final_df = sort_wallet_df_for_output(pd.DataFrame(export_wallet_rows(final_rows)))

    preview_cols = [
        "wallet",
        "prefilter_label",
        "final_label",
        "prefilter_score",
        "final_score",
        "winrate_percentage",
        "roi_daily_7d",
        "roi_daily_30d",
        "days_since_last_trade",
        "dev_tokens_created",
        "median_holding_seconds",
        "median_position_size_usd",
        "token_num_1d",
        "token_num_7d",
        "token_num_30d",
        "realized_profit_1d",
        "realized_profit_7d",
        "realized_profit_30d",
        "realized_profit_pnl_7d",
        "risk_fast_tx_ratio_30d",
        "deep_active_days_7d",
        "deep_consistency_7d",
        "deep_max_gap_days_30d",
        "sniped_5s_count_30d",
        "sniper_rate_5s_30d",
        "top_token_profit_share_30d",
        "final_flags",
    ]
    preview_cols = [c for c in preview_cols if c in final_df.columns]

    print("\nTop 20 wallets:")
    print(final_df[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()