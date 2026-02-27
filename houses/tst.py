import requests
import pandas as pd
import matplotlib.pyplot as plt

BASE = "https://landregistry.data.gov.uk/data/ukhpi/region/{slug}/month/{ym}.json"

# Use local-authority/city slugs used by Land Registry (examples below).
# You can swap/add slugs to match the "cities" you want.
CITY_SLUGS = {
    "London": "london",
    "Birmingham": "birmingham",
    "Manchester": "manchester",
    "Leeds": "leeds",
    "Liverpool": "liverpool",
    "Sheffield": "sheffield",
    "Bristol": "city-of-bristol",
    "Nottingham": "city-of-nottingham",
    "Leicester": "leicester",
    "Newcastle upon Tyne": "newcastle-upon-tyne",
    "Belfast": "belfast",
}

START = "2024-05"
END   = "2025-12"

def get_avg_price(slug: str, ym: str) -> float:
    url = BASE.format(slug=slug, ym=ym)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    data = r.json()

    # The JSON is a dict with keys like "result"; inside will be the observation.
    # We defensively search for "averagePrice".
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "averagePrice":
                    return v
                out = walk(v)
                if out is not None:
                    return out
        elif isinstance(obj, list):
            for it in obj:
                out = walk(it)
                if out is not None:
                    return out
        return None

    avg = walk(data)
    if avg is None:
        raise ValueError(f"averagePrice not found for {slug} {ym}")
    return float(avg)

rows = []
for city, slug in CITY_SLUGS.items():
    p0 = get_avg_price(slug, START)
    p1 = get_avg_price(slug, END)
    pct = (p1 / p0 - 1) * 100
    rows.append((city, p0, p1, pct))

df = pd.DataFrame(rows, columns=["City", f"AvgPrice {START}", f"AvgPrice {END}", "10yr % change"])
df = df.sort_values("10yr % change", ascending=False)

print(df.to_string(index=False))

print(df[["City", f"AvgPrice {START}", f"AvgPrice {END}", "10yr % change"]].to_string(index=False))

plt.figure()

vals = df["10yr % change"].values
bars = plt.barh(df["City"], vals)
plt.xlabel("House price change (%)")
plt.title(f"Average house price change: {START} → {END}")
plt.gca().invert_yaxis()

# Add labels: £start → £end (and %)
pad = (max(vals) - min(vals)) * 0.01 if len(vals) else 1

for bar, (_, row) in zip(bars, df.iterrows()):
    y = bar.get_y() + bar.get_height() / 2
    x = bar.get_width()

    start_price = row[f"AvgPrice {START}"]
    end_price   = row[f"AvgPrice {END}"]
    pct         = row["10yr % change"]

    label = f"£{start_price:,.0f} → £{end_price:,.0f}  ({pct:.1f}%)"

    # Put the text just to the right of the bar (or left if negative)
    if x >= 0:
        plt.text(x + pad, y, label, va="center", ha="left", fontsize=9)
    else:
        plt.text(x - pad, y, label, va="center", ha="right", fontsize=9)

plt.tight_layout()
plt.show()