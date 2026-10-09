import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
m = pd.read_parquet(BASE / "model_data.parquet")

FEATURES = ["AGE_GRP", "SEX", "LIGHT", "WEATHER", "TIME_OF_DAY", "WEEKEND",
            "URBAN", "SPEED_LIMIT", "PEDLOC", "PEDPOS", "RELJCT2", "TYP_INT", "REGION",
            "VEHICLE", "VEH_AGE", "HIT_RUN"]
for c in FEATURES:
    m[c] = pd.Categorical(m[c], categories=sorted(m[c].unique()))
pre = m[m["PERIOD"] == "Pre"].reset_index(drop=True)
post = m[m["PERIOD"] == "Post"].reset_index(drop=True)
rng = np.random.default_rng(11)

def wmean(v, w):
    return float(np.sum(v * w) / np.sum(w))

dummy_cols = pd.get_dummies(m[FEATURES], drop_first=True).columns
def dummies(s):
    return pd.get_dummies(s[FEATURES], drop_first=True).reindex(columns=dummy_cols, fill_value=0)

# Models trained on PRE-COVID crashes ("old rules")
gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                         min_child_samples=60, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=1, verbose=-1)
gbm.fit(pre[FEATURES], pre["SEVERE"], sample_weight=pre["WEIGHT"])
lr = LogisticRegression(max_iter=3000)
lr.fit(dummies(pre), pre["SEVERE"], sample_weight=pre["WEIGHT"])

def predict(kind, s):
    if kind == "LightGBM":
        p = gbm.predict_proba(s[FEATURES])[:, 1]
    else:
        p = lr.predict_proba(dummies(s))[:, 1]
    return wmean(p, s["WEIGHT"]) * 100

pre_weights = pre["WEIGHT"] / pre["WEIGHT"].sum()
results = {}
for kind in ["Logistic", "LightGBM"]:
    full = predict(kind, post)
    contrib = {}
    for f in FEATURES:
        drops = []
        for _ in range(5):                     # average over 5 random swaps
            s = post.copy()
            draw = rng.choice(len(pre), size=len(post), replace=True, p=pre_weights)
            s[f] = pre[f].iloc[draw].values    # swap this factor back to its pre-COVID mix
            s[f] = pd.Categorical(s[f], categories=m[f].cat.categories)
            drops.append(full - predict(kind, s))
        contrib[f] = np.mean(drops)
    results[kind] = contrib
    print(f"{kind}: post-COVID predicted with old rules = {full:.1f}%")

res = pd.DataFrame(results)
res["Average"] = res.mean(axis=1)
res = res.sort_values("Average", ascending=False)
print("\nPercentage points of the predicted rise due to each factor's mix change")
print("(+ = this factor's shift pushed severity UP after 2020):\n")
print(res.round(2).to_string())
print("\nSum of factor contributions:")
print(res[["Logistic", "LightGBM"]].sum().round(2).to_string())