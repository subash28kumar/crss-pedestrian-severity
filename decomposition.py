import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")
BASE = Path(r"C:\Users\skumar\Downloads\CRSS_2016_2024")
m = pd.read_parquet(BASE / "model_data.parquet")

FEATURES = ["AGE_GRP", "SEX", "LIGHT", "WEATHER", "TIME_OF_DAY", "WEEKEND",
            "URBAN", "SPEED_LIMIT", "PEDLOC", "PEDPOS", "RELJCT2", "TYP_INT", "REGION",
            "VEHICLE", "VEH_AGE", "HIT_RUN"]
B = 50
rng = np.random.default_rng(7)

for c in FEATURES:
    m[c] = pd.Categorical(m[c], categories=sorted(m[c].unique()))
m["GROUP"] = m["YEAR"].astype(int).astype(str) + "_" + m["CASENUM"].astype(int).astype(str)
pre = m[m["PERIOD"] == "Pre"].reset_index(drop=True)
post = m[m["PERIOD"] == "Post"].reset_index(drop=True)

def wmean(v, w):
    return float(np.sum(v * w) / np.sum(w))

dummy_cols = pd.get_dummies(m[FEATURES], drop_first=True).columns
def dummies(s):
    return pd.get_dummies(s[FEATURES], drop_first=True).reindex(columns=dummy_cols, fill_value=0)

def decompose(pre_s, post_s, kind):
    if kind == "LightGBM":
        mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=60, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, random_state=1, verbose=-1)
        mdl.fit(pre_s[FEATURES], pre_s["SEVERE"], sample_weight=pre_s["WEIGHT"])
        pred = mdl.predict_proba(post_s[FEATURES])[:, 1]
        fit_pre = mdl.predict_proba(pre_s[FEATURES])[:, 1]
    else:
        mdl = LogisticRegression(max_iter=3000)
        mdl.fit(dummies(pre_s), pre_s["SEVERE"], sample_weight=pre_s["WEIGHT"])
        pred = mdl.predict_proba(dummies(post_s))[:, 1]
        fit_pre = mdl.predict_proba(dummies(pre_s))[:, 1]
    a_pre = wmean(pre_s["SEVERE"], pre_s["WEIGHT"])
    a_post = wmean(post_s["SEVERE"], post_s["WEIGHT"])
    p_post = wmean(pred, post_s["WEIGHT"])
    check = wmean(fit_pre, pre_s["WEIGHT"])
    explained = (p_post - a_pre) / (a_post - a_pre) * 100
    return a_pre * 100, p_post * 100, a_post * 100, explained, check * 100

def resample(s):
    groups = s["GROUP"].unique()
    pick = rng.choice(groups, size=len(groups), replace=True)
    idx = s.groupby("GROUP").indices
    return s.iloc[np.concatenate([idx[g] for g in pick])].reset_index(drop=True)

print("Severe share = killed or seriously injured, weighted (%)\n")
for kind in ["Logistic", "LightGBM"]:
    a_pre, p_post, a_post, expl, check = decompose(pre, post, kind)
    boots = [decompose(resample(pre), resample(post), kind)[3] for _ in range(B)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"== {kind} ==")
    print(f"  Pre-COVID actual                         {a_pre:5.1f}%")
    print(f"  (model check: pre model on pre data      {check:5.1f}%)")
    print(f"  Post-COVID predicted with OLD rules      {p_post:5.1f}%   <- effect of changed mix")
    print(f"  Post-COVID actual                        {a_post:5.1f}%")
    print(f"  Share of the rise explained by mix:      {expl:5.1f}%  (95% CI {lo:.1f}% to {hi:.1f}%)")
    print(f"  Share unexplained (new factors):         {100-expl:5.1f}%\n")

# What in the mix changed?
print("What changed in the mix of crashes (weighted % of pedestrians), Pre -> Post:")
rows = []
for feat, cat in [("VEHICLE", "Pickup"), ("VEHICLE", "SUV"), ("VEHICLE", "Car"),
                  ("LIGHT", "Dark-unlit"), ("LIGHT", "Dark-lit"), ("LIGHT", "Daylight"),
                  ("SPEED_LIMIT", "50+"), ("SPEED_LIMIT", "40-45"), ("SPEED_LIMIT", "<=25"),
                  ("SPEED_LIMIT", "Unknown"), ("AGE_GRP", "65+"), ("AGE_GRP", "0-15"),
                  ("TIME_OF_DAY", "Late 20-23"), ("TIME_OF_DAY", "Night 0-5"),
                  ("HIT_RUN", "Yes"), ("URBAN", "Rural")]:
    a = wmean((pre[feat] == cat).astype(float), pre["WEIGHT"]) * 100
    b = wmean((post[feat] == cat).astype(float), post["WEIGHT"]) * 100
    rows.append({"Condition": f"{feat} = {cat}", "Pre %": a, "Post %": b, "Change": b - a})
print(pd.DataFrame(rows).set_index("Condition").round(1).to_string())

print("\nRegion mix (weighted %):")
reg = pd.DataFrame({
    "Pre %": pre.groupby("REGION", observed=True)["WEIGHT"].sum() / pre["WEIGHT"].sum() * 100,
    "Post %": post.groupby("REGION", observed=True)["WEIGHT"].sum() / post["WEIGHT"].sum() * 100})
reg["Severe Pre %"] = pre.groupby("REGION", observed=True).apply(
    lambda g: wmean(g["SEVERE"], g["WEIGHT"]) * 100, include_groups=False)
reg["Severe Post %"] = post.groupby("REGION", observed=True).apply(
    lambda g: wmean(g["SEVERE"], g["WEIGHT"]) * 100, include_groups=False)
print(reg.round(1).to_string())