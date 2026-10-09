import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\skumar\Downloads\CRSS_2016_2024")
m = pd.read_parquet(BASE / "model_data.parquet")
m = m[m["PERIOD"] != "COVID"].copy()
RNAME = {"c1": "Northeast", "c2": "Midwest", "c3": "South", "c4": "West"}
m["REG"] = m["REGION"].map(RNAME)

def wpct(df, col="SEVERE"):
    return 100 * np.average(df[col], weights=df["WEIGHT"])

print("=== 1. Severe % by region and period (weighted), with sample sizes ===")
rows = []
for (r, p), g in m.groupby(["REG", "PERIOD"]):
    rows.append([r, p, round(wpct(g), 1), len(g), g["PSU"].nunique()])
print(pd.DataFrame(rows, columns=["Region", "Period", "Severe%", "Records", "PSUs"]).to_string(index=False))

print("\n=== 2. West severe % by year ===")
w_all = pd.read_parquet(BASE / "model_data.parquet")
w_all = w_all[w_all["REGION"] == "c4"]
print(w_all.groupby("YEAR").apply(lambda g: pd.Series({
    "Severe%": round(wpct(g), 1), "Records": len(g), "PSUs": g["PSU"].nunique()})).to_string())

print("\n=== 3. West vs rest: mix (weighted %) Pre -> Post ===")
m["GROUP"] = np.where(m["REG"] == "West", "West", "Rest")
for col in ["SPEED_LIMIT", "LIGHT", "URBAN", "VEHICLE", "AGE_GRP", "TIME_OF_DAY"]:
    t = (m.groupby(["GROUP", "PERIOD", col])["WEIGHT"].sum()
           / m.groupby(["GROUP", "PERIOD"])["WEIGHT"].sum() * 100).round(1)
    t = t.unstack(["GROUP", "PERIOD"]).reindex(columns=[("West","Pre"),("West","Post"),("Rest","Pre"),("Rest","Post")])
    print(f"\n--- {col} ---"); print(t.to_string())

print("\n=== 4. West: severe % within each category, Pre -> Post ===")
west = m[m["GROUP"] == "West"]
for col in ["SPEED_LIMIT", "LIGHT", "URBAN", "VEHICLE"]:
    t = west.groupby([col, "PERIOD"]).apply(lambda g: pd.Series({
        "Severe%": round(wpct(g), 1), "n": len(g)})).unstack("PERIOD")
    print(f"\n--- {col} ---"); print(t.to_string())

print("\n=== 5. West: severe % by PSU (does one area drive it?) ===")
t = west.groupby(["PSU", "PERIOD"]).apply(lambda g: pd.Series({
    "Severe%": round(wpct(g), 1), "n": len(g)})).unstack("PERIOD")
print(t.to_string())