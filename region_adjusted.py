import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = Path(__file__).resolve().parent
m = pd.read_parquet(BASE / "model_data.parquet")
m = m[m["PERIOD"] != "COVID"].copy()
m["WEST"] = (m["REGION"] == "c4").astype(int)
m["POST"] = (m["PERIOD"] == "Post").astype(int)
m["w"] = m["WEIGHT"] / m["WEIGHT"].mean()

COVS = ["AGE_GRP", "SEX", "LIGHT", "WEATHER", "TIME_OF_DAY", "WEEKEND", "URBAN",
        "SPEED_LIMIT", "PEDLOC", "PEDPOS", "RELJCT2", "TYP_INT", "VEHICLE", "VEH_AGE", "HIT_RUN"]

# collapse rare categories so the model is stable
for c in COVS:
    counts = m[c].value_counts()
    rare = counts[counts < 30].index
    m[c] = m[c].where(~m[c].isin(rare), "Other").astype(str)

groups = pd.factorize(m["PSU"])[0]

def fit(formula):
    mod = smf.glm(formula, data=m, family=sm.families.Binomial(), var_weights=m["w"])
    return mod.fit(cov_type="cluster", cov_kwds={"groups": groups})

def report(res, label):
    b = res.params["WEST:POST"]
    lo, hi = res.conf_int().loc["WEST:POST"]
    p = res.pvalues["WEST:POST"]
    print(f"{label:28s} OR = {np.exp(b):.2f}  (95% CI {np.exp(lo):.2f}-{np.exp(hi):.2f})  p = {p:.4f}")
    return res

def adjusted_did(res):
    """Standardized predictions: everyone's covariates, region/period switched."""
    out = {}
    for wv in [0, 1]:
        for pv in [0, 1]:
            tmp = m.copy()
            tmp["WEST"], tmp["POST"] = wv, pv
            out[(wv, pv)] = 100 * np.average(res.predict(tmp), weights=m["w"])
    west_ch = out[(1, 1)] - out[(1, 0)]
    rest_ch = out[(0, 1)] - out[(0, 0)]
    print(f"   Adjusted KA%: West {out[(1,0)]:.1f} -> {out[(1,1)]:.1f} (change {west_ch:+.1f}) | "
          f"Rest {out[(0,0)]:.1f} -> {out[(0,1)]:.1f} (change {rest_ch:+.1f}) | "
          f"West minus Rest {west_ch - rest_ch:+.1f} pts")

print("=== West x Post interaction (Pre = 2016-2019, Post = 2022-2024) ===\n")
r1 = report(fit("SEVERE ~ WEST * POST"), "1. Unadjusted")
adjusted_did(r1)
r2 = report(fit("SEVERE ~ WEST * POST + " + " + ".join(f"C({c})" for c in COVS)), "2. Adjusted (all covariates)")
adjusted_did(r2)

print("\n=== Sensitivity: drop 2016 ===\n")
m = m[m["YEAR"] != 2016].copy()
groups = pd.factorize(m["PSU"])[0]
r3 = report(fit("SEVERE ~ WEST * POST + " + " + ".join(f"C({c})" for c in COVS)), "3. Adjusted, no 2016")
adjusted_did(r3)