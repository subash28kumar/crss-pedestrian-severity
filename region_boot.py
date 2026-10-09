import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

BASE = Path(r"C:\Users\skumar\Downloads\CRSS_2016_2024")
B = 500
rng = np.random.default_rng(2026)

m = pd.read_parquet(BASE / "model_data.parquet")
p = pd.read_parquet(BASE / "pedestrians_2016_2024.parquet", columns=["PSU", "PSUSTRAT"])
p = p.apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
print("Max strata per PSU (should be 1):", p.groupby("PSU")["PSUSTRAT"].nunique().max())
psu2str = p.groupby("PSU")["PSUSTRAT"].first()

m = m[m["PERIOD"] != "COVID"].copy().reset_index(drop=True)
m["PSUSTRAT"] = m["PSU"].map(psu2str)
m["WEST"] = (m["REGION"] == "c4").astype(float)
m["POST"] = (m["PERIOD"] == "Post").astype(float)

COVS = ["AGE_GRP", "SEX", "LIGHT", "WEATHER", "TIME_OF_DAY", "WEEKEND", "URBAN",
        "SPEED_LIMIT", "PEDLOC", "PEDPOS", "RELJCT2", "TYP_INT", "VEHICLE", "VEH_AGE", "HIT_RUN"]
for c in COVS:
    counts = m[c].value_counts()
    m[c] = m[c].where(~m[c].isin(counts[counts < 30].index), "Other").astype(str)

X = pd.get_dummies(m[COVS], drop_first=True, dtype=float)
X.insert(0, "WEST_POST", m["WEST"] * m["POST"])
X.insert(0, "POST", m["POST"])
X.insert(0, "WEST", m["WEST"])
X.insert(0, "const", 1.0)
Xv = X.values
y = m["SEVERE"].values
w0 = (m["WEIGHT"] / m["WEIGHT"].mean()).values
IDX = 3  # WEST_POST column

def fit(w):
    return sm.GLM(y, Xv, family=sm.families.Binomial(), var_weights=w).fit().params

def adj_did(params, w):
    out = {}
    for wv in [0, 1]:
        for pv in [0, 1]:
            Xc = Xv.copy()
            Xc[:, 1], Xc[:, 2], Xc[:, 3] = wv, pv, wv * pv
            pred = 1 / (1 + np.exp(-(Xc @ params)))
            out[(wv, pv)] = 100 * np.average(pred, weights=w)
    return (out[(1, 1)] - out[(1, 0)]) - (out[(0, 1)] - out[(0, 0)])

b0 = fit(w0)
print(f"\nPoint estimate: OR = {np.exp(b0[IDX]):.2f}, adjusted West-minus-Rest = {adj_did(b0, w0):+.1f} pts")

units = m[["PSUSTRAT", "PSU"]].drop_duplicates()
psu_arr = m["PSU"].values
ors, dids = [], []
for b in range(B):
    mult = {}
    for _, g in units.groupby("PSUSTRAT"):
        ids = g["PSU"].values
        n = len(ids)
        if n < 2:
            for i in ids:
                mult[i] = 1.0
            continue
        cnt = pd.Series(rng.choice(ids, n - 1, replace=True)).value_counts()
        for i in ids:
            mult[i] = cnt.get(i, 0) * n / (n - 1)
    wb = w0 * pd.Series(psu_arr).map(mult).values
    keep = wb > 0
    try:
        params = sm.GLM(y[keep], Xv[keep], family=sm.families.Binomial(),
                        var_weights=wb[keep]).fit().params
        ors.append(np.exp(params[IDX]))
        dids.append(adj_did(params, wb))
    except Exception:
        pass
    if (b + 1) % 100 == 0:
        print(f"  {b + 1}/{B} done")

ors, dids = np.array(ors), np.array(dids)
print(f"\nSuccessful replicates: {len(ors)}/{B}")
print(f"Adjusted OR (West x Post): {np.exp(b0[IDX]):.2f}  95% CI {np.percentile(ors, 2.5):.2f}-{np.percentile(ors, 97.5):.2f}")
print(f"Share of replicates with OR <= 1: {100 * np.mean(ors <= 1):.1f}%")
print(f"Adjusted West-minus-Rest (pts): {adj_did(b0, w0):+.1f}  95% CI {np.percentile(dids, 2.5):+.1f} to {np.percentile(dids, 97.5):+.1f}")