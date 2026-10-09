import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import shap
import warnings

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
m = pd.read_parquet(BASE / "model_data.parquet")

FEATURES = ["AGE_GRP", "SEX", "LIGHT", "WEATHER", "TIME_OF_DAY", "WEEKEND",
            "URBAN", "SPEED_LIMIT", "PEDLOC", "PEDPOS", "RELJCT2", "TYP_INT", "REGION",
            "VEHICLE", "VEH_AGE", "HIT_RUN"]
CHECKS = [("LIGHT", "Dark-unlit"), ("LIGHT", "Dark-lit"), ("LIGHT", "Daylight"),
          ("SPEED_LIMIT", "50+"), ("SPEED_LIMIT", "40-45"), ("SPEED_LIMIT", "<=25"),
          ("VEHICLE", "SUV"), ("VEHICLE", "Pickup"), ("VEHICLE", "Car"),
          ("AGE_GRP", "65+"), ("HIT_RUN", "Yes")]
B = 100
rng = np.random.default_rng(42)

for c in FEATURES:
    m[c] = pd.Categorical(m[c], categories=sorted(m[c].unique()))
m["GROUP"] = m["YEAR"].astype(int).astype(str) + "_" + m["CASENUM"].astype(int).astype(str)

params = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=60,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

def fit_and_explain(s, seed):
    mdl = lgb.LGBMClassifier(random_state=seed, **params)
    mdl.fit(s[FEATURES], s["SEVERE"], sample_weight=s["WEIGHT"])
    sv = shap.TreeExplainer(mdl).shap_values(s[FEATURES])
    if isinstance(sv, list):
        sv = sv[1]
    sv = pd.DataFrame(sv, columns=FEATURES, index=s.index)
    out = {f"IMP {f}": sv[f].abs().mean() for f in FEATURES}
    for feat, cat in CHECKS:
        mask = s[feat] == cat
        out[f"{feat} = {cat}"] = sv.loc[mask, feat].mean() if mask.sum() >= 30 else np.nan
    return out

def cluster_resample(s):
    groups = s["GROUP"].unique()
    pick = rng.choice(groups, size=len(groups), replace=True)
    idx = s.groupby("GROUP").indices
    rows = np.concatenate([idx[g] for g in pick])
    out = s.iloc[rows].copy()
    out.index = range(len(out))
    return out

pre = m[m["PERIOD"] == "Pre"].reset_index(drop=True)
post = m[m["PERIOD"] == "Post"].reset_index(drop=True)

diffs = []
for b in range(B):
    r_pre = fit_and_explain(cluster_resample(pre), seed=b)
    r_post = fit_and_explain(cluster_resample(post), seed=b)
    diffs.append({k: r_post[k] - r_pre[k] for k in r_pre})
    if (b + 1) % 10 == 0:
        print(f"  finished {b+1}/{B}")

D = pd.DataFrame(diffs)
base_pre, base_post = fit_and_explain(pre, 0), fit_and_explain(post, 0)

rows = []
for k in D.columns:
    lo, hi = np.nanpercentile(D[k], [2.5, 97.5])
    rows.append({"Measure": k, "Pre": base_pre[k], "Post": base_post[k],
                 "Change": base_post[k] - base_pre[k], "CI low": lo, "CI high": hi,
                 "Real change?": "YES" if (lo > 0 or hi < 0) else "no"})
res = pd.DataFrame(rows).set_index("Measure")

pd.set_option("display.width", 140)
print("\nA) Specific conditions (mean SHAP; + = more severe), Post vs Pre, 95% CI:")
print(res[~res.index.str.startswith("IMP")].round(3).to_string())
print("\nB) Factor importance, Post vs Pre, 95% CI:")
print(res[res.index.str.startswith("IMP")].round(3).sort_values("Change").to_string())

res.to_csv(BASE / "bootstrap_results.csv")
print("\nSaved: bootstrap_results.csv")