import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
d = pd.read_parquet(BASE / "pedestrians_2016_2024.parquet")
for c in ["INJ_SEV", "REGION", "WEIGHT", "YEAR", "PSU"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d[d["INJ_SEV"].isin([0, 1, 2, 3, 4])].copy()
d["SEV"] = d["INJ_SEV"].map({0: "O", 1: "C", 2: "B", 3: "A", 4: "K"})
d["GROUP"] = np.where(d["REGION"] == 4, "West", "Rest")
d["PERIOD"] = pd.cut(d["YEAR"], [2015, 2019, 2021, 2024], labels=["Pre", "COVID", "Post"]).astype(str)
ORDER = ["K", "A", "B", "C", "O"]

def share_table(df, by):
    t = df.groupby(by + ["SEV"])["WEIGHT"].sum().unstack("SEV").reindex(columns=ORDER).fillna(0)
    return (t.div(t.sum(axis=1), axis=0) * 100).round(1)

def count_table(df, by):
    t = df.groupby(by + ["SEV"])["WEIGHT"].sum().unstack("SEV").reindex(columns=ORDER).fillna(0)
    t["Total"] = t.sum(axis=1)
    return (t / 1000).round(1)

print("=== 1. KABCO share (weighted %) by year: West ===")
print(share_table(d[d["GROUP"] == "West"], ["YEAR"]).to_string())

print("\n=== 2. KABCO share (weighted %) by year: Rest of US ===")
print(share_table(d[d["GROUP"] == "Rest"], ["YEAR"]).to_string())

print("\n=== 3. Weighted national estimates (thousands) by year: West ===")
print(count_table(d[d["GROUP"] == "West"], ["YEAR"]).to_string())

print("\n=== 4. Weighted national estimates (thousands) by year: Rest of US ===")
print(count_table(d[d["GROUP"] == "Rest"], ["YEAR"]).to_string())

print("\n=== 5. West by PSU: K% and A% Pre vs Post (weighted) ===")
w = d[(d["GROUP"] == "West") & (d["PERIOD"] != "COVID")]
t = share_table(w, ["PSU", "PERIOD"])[["K", "A"]].unstack("PERIOD")
t.columns = [f"{s}_{p}" for s, p in t.columns]
t = t[["K_Pre", "K_Post", "A_Pre", "A_Post"]]
t["n_Pre"] = w[w["PERIOD"] == "Pre"].groupby("PSU").size()
t["n_Post"] = w[w["PERIOD"] == "Post"].groupby("PSU").size()
print(t.to_string())