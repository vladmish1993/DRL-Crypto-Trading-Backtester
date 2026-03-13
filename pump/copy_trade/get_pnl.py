import csv
import json
import os
import time
from decimal import Decimal, InvalidOperation
from curl_cffi import requests as cf_requests

WALLET = "CXdZ6m2PotNq4D7zKK3af426ohGro1xingNuf68SoKmv"
BASE_URL = f"https://gmgn.ai/pf/api/v1/wallet/sol/{WALLET}/holdings"

# If you already saved the first page result, keep this True.
# The script will load gmgn_holdings.json, take data.list from it,
# then continue fetching with data.next.
USE_EXISTING_FIRST_PAGE = True
EXISTING_FIRST_PAGE_FILE = "old/gmgn_holdings.json"

# Optional manual resume cursor.
# Leave as None to use data.next from the saved first page file.
# Set this only if you want to resume from a specific page.
START_CURSOR = None
# START_CURSOR = "Mm1OODlaU2poUjZnNFhEZVNOb2tONWk3ZHZXWnpQVUdkTXVlVnpKb3B1bXA6MTc3MzMyMDQxMg=="

REQUEST_DELAY_SECONDS = 0.15
MAX_PAGES = 1000

params_base = {
    "device_id": "20b797a6-e165-49cc-835b-e0bcc9fa25f7",
    "fp_did": "3fa0ea5ea4368ac14463b9432da366e5",
    "client_id": "gmgn_web_20260312-11621-ec221da",
    "from_app": "gmgn",
    "app_ver": "20260312-11621-ec221da",
    "tz_name": "Europe/London",
    "tz_offset": 0,
    "app_lang": "en-US",
    "os": "web",
    "worker": 0,
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

headers = {
    "accept": "application/json, text/plain, */*",
    "referer": f"https://gmgn.ai/sol/address/{WALLET}",
    "origin": "https://gmgn.ai",
}

session = cf_requests.Session(impersonate="chrome110")


def D(value) -> Decimal:
    if value in (None, "", "null"):
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def fetch_page(cursor=None):
    params = params_base.copy()
    if cursor:
        params["cursor"] = cursor

    response = session.get(
        BASE_URL,
        params=params,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(
            f"API returned code={payload.get('code')}, "
            f"message={payload.get('message')}, reason={payload.get('reason')}"
        )
    return payload


def load_existing_first_page(filepath):
    if not os.path.exists(filepath):
        return [], None

    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    data = payload.get("data") or {}
    items = data.get("list") or []
    next_cursor = data.get("next")
    return items, next_cursor


def token_key(item, fallback_index=None):
    token = item.get("token") or {}
    addr = token.get("token_address")
    if addr:
        return addr
    return f"unknown_{fallback_index}"


def dedupe_items(items):
    seen = set()
    unique = []

    for idx, item in enumerate(items):
        key = token_key(item, idx)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique


def fetch_all_holdings():
    all_items = []
    seen = set()
    cursor = START_CURSOR
    page_count = 0

    # Option 1: load first page from saved file, then continue with its `next`
    if USE_EXISTING_FIRST_PAGE and cursor is None:
        seed_items, saved_next = load_existing_first_page(EXISTING_FIRST_PAGE_FILE)
        if seed_items:
            for idx, item in enumerate(seed_items):
                key = token_key(item, idx)
                if key not in seen:
                    seen.add(key)
                    all_items.append(item)

            cursor = saved_next
            print(
                f"Loaded {len(seed_items)} items from {EXISTING_FIRST_PAGE_FILE}. "
                f"Continuing with saved next cursor: {cursor}"
            )

    # Option 2: no saved first page, so fetch from scratch
    if not all_items and cursor is None:
        first_payload = fetch_page(cursor=None)
        data = first_payload.get("data") or {}
        first_items = data.get("list") or []
        cursor = data.get("next")

        for idx, item in enumerate(first_items):
            key = token_key(item, idx)
            if key not in seen:
                seen.add(key)
                all_items.append(item)

        page_count += 1
        print(f"Fetched page 1: {len(first_items)} items, next={cursor}")

    # Continue with pagination
    while cursor and page_count < MAX_PAGES:
        payload = fetch_page(cursor=cursor)
        data = payload.get("data") or {}
        items = data.get("list") or []
        next_cursor = data.get("next")

        added = 0
        for idx, item in enumerate(items):
            key = token_key(item, f"{page_count}_{idx}")
            if key in seen:
                continue
            seen.add(key)
            all_items.append(item)
            added += 1

        page_count += 1
        print(
            f"Fetched page {page_count + 1}: "
            f"raw_items={len(items)}, added={added}, next={next_cursor}"
        )

        if not next_cursor or next_cursor == cursor:
            break

        cursor = next_cursor
        time.sleep(REQUEST_DELAY_SECONDS)

    return dedupe_items(all_items)


def build_pnl_rows(items):
    rows = []
    total_realized = Decimal("0")
    total_unrealized = Decimal("0")
    total_profit = Decimal("0")
    total_cost_basis = Decimal("0")

    for item in items:
        token = item.get("token") or {}

        symbol = token.get("symbol") or ""
        name = token.get("name") or ""
        token_address = token.get("token_address") or ""

        balance = D(item.get("balance"))
        usd_value = D(item.get("usd_value"))

        history_bought_cost = D(item.get("history_bought_cost"))
        history_bought_fee = D(item.get("history_bought_fee"))
        history_transfer_in_cost = D(item.get("history_transfer_in_cost"))

        realized_profit = D(item.get("realized_profit"))
        unrealized_profit = D(item.get("unrealized_profit"))
        total_profit_token = D(item.get("total_profit"))

        # 30d capital deployed approximation for aggregate PnL:
        # use buy/transfer-in cost in the same 30d window
        cost_basis_30d = (
            history_bought_cost
            + history_bought_fee
            + history_transfer_in_cost
        )

        # Prefer API PnL ratio if present
        api_total_profit_pnl = item.get("total_profit_pnl")
        if api_total_profit_pnl is None:
            total_profit_pnl = (
                (total_profit_token / cost_basis_30d)
                if cost_basis_30d > 0
                else None
            )
        else:
            total_profit_pnl = D(api_total_profit_pnl)

        total_realized += realized_profit
        total_unrealized += unrealized_profit
        total_profit += total_profit_token
        total_cost_basis += cost_basis_30d

        rows.append({
            "token_address": token_address,
            "symbol": symbol,
            "name": name,
            "balance": str(balance),
            "usd_value": str(usd_value),
            "realized_profit_30d": str(realized_profit),
            "unrealized_profit_30d": str(unrealized_profit),
            "total_profit_30d": str(total_profit_token),
            "total_profit_pnl_30d": (
                str(total_profit_pnl) if total_profit_pnl is not None else ""
            ),
            "cost_basis_30d": str(cost_basis_30d),
            "history_total_buys": item.get("history_total_buys", 0),
            "history_total_sells": item.get("history_total_sells", 0),
            "start_holding_at": item.get("start_holding_at", 0),
            "end_holding_at": item.get("end_holding_at", 0),
            "last_active_timestamp": item.get("last_active_timestamp", 0),
            "is_closed": balance == 0,
        })

    aggregate_pnl = (
        (total_profit / total_cost_basis) if total_cost_basis > 0 else None
    )

    summary = {
        "tokens_count": len(rows),
        "total_realized_profit_30d": str(total_realized),
        "total_unrealized_profit_30d": str(total_unrealized),
        "total_profit_30d": str(total_profit),
        "aggregate_cost_basis_30d": str(total_cost_basis),
        "aggregate_total_profit_pnl_30d": (
            str(aggregate_pnl) if aggregate_pnl is not None else ""
        ),
    }

    return rows, summary


def save_outputs(all_items, rows, summary):
    with open("gmgn_holdings_all_pages_30d.json", "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    with open("gmgn_pnl_30d_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "token_address",
        "symbol",
        "name",
        "balance",
        "usd_value",
        "realized_profit_30d",
        "unrealized_profit_30d",
        "total_profit_30d",
        "total_profit_pnl_30d",
        "cost_basis_30d",
        "history_total_buys",
        "history_total_sells",
        "start_holding_at",
        "end_holding_at",
        "last_active_timestamp",
        "is_closed",
    ]

    rows_sorted = sorted(
        rows,
        key=lambda x: D(x["total_profit_30d"]),
        reverse=True,
    )

    with open("gmgn_pnl_30d_by_token.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)


def main():
    all_items = fetch_all_holdings()
    rows, summary = build_pnl_rows(all_items)
    save_outputs(all_items, rows, summary)

    print("\n=== 30d PnL summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    top_winners = sorted(rows, key=lambda x: D(x["total_profit_30d"]), reverse=True)[:10]
    top_losers = sorted(rows, key=lambda x: D(x["total_profit_30d"]))[:10]

    print("\n=== Top 10 winners ===")
    for row in top_winners:
        print(
            f"{row['symbol'] or '(no symbol)'} | "
            f"{row['total_profit_30d']} USD | "
            f"PnL={row['total_profit_pnl_30d']}"
        )

    print("\n=== Top 10 losers ===")
    for row in top_losers:
        print(
            f"{row['symbol'] or '(no symbol)'} | "
            f"{row['total_profit_30d']} USD | "
            f"PnL={row['total_profit_pnl_30d']}"
        )

    print("\nSaved files:")
    print("- gmgn_holdings_all_pages_30d.json")
    print("- gmgn_pnl_30d_summary.json")
    print("- gmgn_pnl_30d_by_token.csv")


if __name__ == "__main__":
    main()