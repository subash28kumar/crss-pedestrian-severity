import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
full = pd.read_parquet(BASE / "model_data.parquet")
full = full[full["PERIOD"] != "COVID"].copy()
full["WEST"] = (full["REGION"] == "c4").astype(int)
full["POST"] = (full["PERIOD"] == "Post").astype(int)

COVS = ["AGE_GRP", "SEX", "LIGHT", "WEATHER", "TIME_OF_DAY", "WEEKEND", "URBAN",
        "SPEED_LIMIT", "PEDLOC", "PEDPOS", "RELJCT2", "TYP_INT", "VEHICLE", "VEH_AGE", "HIT_RUN"]

def run(m, label):
    m = m.copy()
    m["w"] = m["WEIGHT"] / m["WEIGHT"].mean()
    covs = [c for c in COVS if m[c].nunique() > 1]
    for c in covs:
        counts = m[c].value_counts()
        m[c] = m[c].where(~m[c].isin(counts[counts < 30].index), "Other").astype(str)
    groups = pd.factorize(m["PSU"])[0]
    f = "SEVERE ~ WEST * POST + " + " + ".join(f"C({c})" for c in covs)
    res = smf.glm(f, data=m, family=sm.families.Binomial(), var_weights=m["w"]) \
             .fit(cov_type="cluster", cov_kwds={"groups": groups})
    b = res.params["WEST:POST"]
    lo, hi = res.conf_int().loc["WEST:POST"]
    wpre = 100 * np.average(m.loc[(m.WEST == 1) & (m.POST == 0), "SEVERE"], weights=m.loc[(m.WEST == 1) & (m.POST == 0), "w"])
    wpost = 100 * np.average(m.loc[(m.WEST == 1) & (m.POST == 1), "SEVERE"], weights=m.loc[(m.WEST == 1) & (m.POST == 1), "w"])
    rpre = 100 * np.average(m.loc[(m.WEST == 0) & (m.POST == 0), "SEVERE"], weights=m.loc[(m.WEST == 0) & (m.POST == 0), "w"])
    rpost = 100 * np.average(m.loc[(m.WEST == 0) & (m.POST == 1), "SEVERE"], weights=m.loc[(m.WEST == 0) & (m.POST == 1), "w"])
    print(f"{label:34s} n={len(m):6d} | West {wpre:4.1f}->{wpost:4.1f}  Rest {rpre:4.1f}->{rpost:4.1f} | "
          f"adj OR {np.exp(b):.2f} ({np.exp(lo):.2f}-{np.exp(hi):.2f}) p={res.pvalues['WEST:POST']:.4f}")

print("=== Robustness: West x Post adjusted OR (Pre 2016-19 vs Post 2022-24) ===\n")
run(full, "1. All records (main)")
run(full[full["SPEED_LIMIT"] != "Unknown"], "2. Known speed limit only")
run(full[full["VEH_AGE"] != "Unknown"], "3. Known vehicle age only")
run(full[(full["SPEED_LIMIT"] != "Unknown") & (full["VEH_AGE"] != "Unknown")], "4. Both known")
run(full[full["URBAN"] == "Urban"], "5. Urban only")