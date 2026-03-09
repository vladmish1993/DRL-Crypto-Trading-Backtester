from curl_cffi import requests as cf_requests
session = cf_requests.Session(impersonate="chrome110")
url = "https://gmgn.ai/defi/quotation/v1/rank/sol/wallets/7d?device_id=20b797a6-e165-49cc-835b-e0bcc9fa25f7&fp_did=3fa0ea5ea4368ac14463b9432da366e5&client_id=gmgn_web_20260309-11491-7af2de2&from_app=gmgn&app_ver=20260309-11491-7af2de2&tz_name=Europe%2FLondon&tz_offset=0&app_lang=en-US&os=web&worker=0&orderby=winrate_30d&direction=desc"

response = session.get(url, headers={"referer": "https://gmgn.ai/trade/jk4akIak?chain=sol"})
print(response.text)