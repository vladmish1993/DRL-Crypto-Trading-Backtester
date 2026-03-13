import json
from curl_cffi import requests as cf_requests

WALLET = "8jVLBkFSjHKgDbkUA5QfSDkV9YbCSiWJANYruKaHXjRx"
BASE_URL = f"https://gmgn.ai/pf/api/v1/wallet/sol/{WALLET}/holdings"

params = {
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

try:
    response = session.get(
        BASE_URL,
        params=params,
        headers=headers,
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()

    print(json.dumps(data, indent=2, ensure_ascii=False))

    with open("old/gmgn_holdings.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\nSaved to gmgn_holdings.json")

except Exception as e:
    print(f"Request failed: {e}")
    try:
        print("Status:", response.status_code)
        print("Body:", response.text[:1000])
    except Exception:
        pass