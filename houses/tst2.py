import pandas as pd

# --- Input file (use your local path) ---
PATH = "UK-HPI-full-file-2025-12.csv"  # change if needed

# --- “City” proxy: local authority areas in UK HPI ---
LA_PREFIXES = ("E060", "E070", "E080", "E090", "W060", "S120", "N090")

# --- Choose the 10-year window (end month must exist in the file) ---
END_DATE = pd.Timestamp("2025-12-01")
START_DATE = pd.Timestamp("2024-05-01")

# --- Load only what we need ---
df = pd.read_csv(PATH, usecols=["Date", "RegionName", "AreaCode", "AveragePrice"])
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

# Keep local-authority rows (often used as “city” equivalents)
df = df[df["AreaCode"].astype(str).str.startswith(LA_PREFIXES)].copy()

# Pull start/end prices and calculate 10-year % change
start = (
    df[df["Date"] == START_DATE][["AreaCode", "RegionName", "AveragePrice"]]
    .rename(columns={"AveragePrice": "StartPrice", "RegionName": "NameStart"})
)
end = (
    df[df["Date"] == END_DATE][["AreaCode", "RegionName", "AveragePrice"]]
    .rename(columns={"AveragePrice": "EndPrice", "RegionName": "NameEnd"})
)

out = start.merge(end, on="AreaCode", how="inner")
out["RegionName"] = out["NameEnd"].combine_first(out["NameStart"])
out = out[["AreaCode", "RegionName", "StartPrice", "EndPrice"]]
out["PctChange"] = (out["EndPrice"] / out["StartPrice"] - 1) * 100

# Sort for charting
out = out.sort_values("PctChange", ascending=True).reset_index(drop=True)

# Save the underlying data (so you have the prices for every area)
out.to_csv("uk_hpi_10yr_change_local_authorities.csv", index=False)

print(f"Areas charted: {len(out)}")
print(out.tail(10)[["RegionName", "StartPrice", "EndPrice", "PctChange"]])

# --- Chart (interactive, works best for 300+ “cities”) ---
import plotly.express as px

fig = px.bar(
    out,
    x="PctChange",
    y="RegionName",
    orientation="h",
    hover_data={
        "AreaCode": True,
        "StartPrice": ":,.0f",
        "EndPrice": ":,.0f",
        "PctChange": ":.1f",
    },
    title=f"UK HPI: 10-year house price change by local authority ({START_DATE:%b %Y} → {END_DATE:%b %Y})",
    labels={"PctChange": "10-year change (%)", "RegionName": "Local authority (city proxy)"},
)

# Make it tall enough to show everything (you can reduce if you prefer)
fig.update_layout(height=max(800, len(out) * 18), margin=dict(l=10, r=10, t=60, b=10))

fig.show()
fig.write_html("uk_hpi_10yr_change_all_local_authorities.html", include_plotlyjs="cdn")

print("Saved:")
print(" - uk_hpi_10yr_change_local_authorities.csv")
print(" - uk_hpi_10yr_change_all_local_authorities.html")