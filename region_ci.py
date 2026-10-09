import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
d = pd.read_parquet(BASE / "pedestrians_2016_2024.parquet")
STRAT = "PSUSTRAT" if "PSUSTRAT" in d.columns else "STRATUM"
print(f"Variance strata column: {STRAT}")

for c in ["INJ_SEV", "REGION", "WEIGHT", "YEAR", "PSU", STRAT]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d[d["INJ_SEV"].isin([0, 1, 2, 3, 4])].copy()
d["WEST"] = d["REGION"] == 4
d["K"] = (d["INJ_SEV"] == 4).astype(float)
d["KA"] = d["INJ_SEV"].isin([3, 4]).astype(float)
d["KAB"] = d["INJ_SEV"].isin([2, 3, 4]).astype(float)

t = d.groupby(STRAT)["PSU"].nunique()
print(f"Strata: {len(t)}, PSUs: {d['PSU'].nunique()}, strata with 1 PSU: {(t < 2).sum()}\n")

def ratio(y, mask):
    """Weighted proportion in a domain and its linearized values (Taylor series)."""
    w = d["WEIGHT"] * mask
    R = (w * d[y]).sum() / w.sum()
    z = w * (d[y] - R) / w.sum()
    return R, z

def variance(z):
    g = pd.DataFrame({"h": d[STRAT], "psu": d["PSU"], "z": z}).groupby(["h", "psu"])["z"].sum().reset_index()
    v = 0.0
    for _, s in g.groupby("h"):
        n = len(s)
        if n > 1:
            v += n / (n - 1) * ((s["z"] - s["z"].mean()) ** 2).sum()
    return v

def fmt(est, z):
    se = np.sqrt(variance(z))
    return f"{100*est:6.1f}  ({100*(est-1.96*se):5.1f}, {100*(est+1.96*se):5.1f})"

POST = [2022, 2023, 2024]
OUTCOMES = [("K share", "K", None), ("KA share", "KA", None), ("KA among KAB", "KA", "KAB")]

for pre_years, label in [([2016, 2017, 2018, 2019], "Pre = 2016-2019"), ([2017, 2018, 2019], "Pre = 2017-2019 (no 2016)")]:
    print("=" * 70)
    print(f"{label}   |   Post = 2022-2024   |   estimate % (95% CI)")
    print("=" * 70)
    pre = d["YEAR"].isin(pre_years)
    post = d["YEAR"].isin(POST)
    for name, y, sub in OUTCOMES:
        extra = d[sub] if sub else 1.0
        r = {}
        for g, gm in [("West", d["WEST"]), ("Rest", ~d["WEST"])]:
            for p, pm in [("Pre", pre), ("Post", post)]:
                r[(g, p)] = ratio(y, (gm & pm).astype(float) * extra)
        print(f"\n--- {name} ---")
        for g in ["West", "Rest"]:
            print(f"{g:5s} Pre    {fmt(*r[(g,'Pre')])}")
            print(f"{g:5s} Post   {fmt(*r[(g,'Post')])}")
            ch = r[(g, "Post")][0] - r[(g, "Pre")][0]
            zc = r[(g, "Post")][1] - r[(g, "Pre")][1]
            print(f"{g:5s} Change {fmt(ch, zc)}")
        did = (r[("West","Post")][0] - r[("West","Pre")][0]) - (r[("Rest","Post")][0] - r[("Rest","Pre")][0])
        zd = (r[("West","Post")][1] - r[("West","Pre")][1]) - (r[("Rest","Post")][1] - r[("Rest","Pre")][1])
        print(f"West minus Rest change  {fmt(did, zd)}")