import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import shap
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
BASE = Path(r"C:\Users\skumar\Downloads\CRSS_2016_2024")
m = pd.read_parquet(BASE / "model_data.parquet")

FEATURES = ["AGE_GRP", "SEX", "LIGHT", "WEATHER", "TIME_OF_DAY", "WEEKEND",
            "URBAN", "SPEED_LIMIT", "PEDLOC", "PEDPOS", "RELJCT2", "TYP_INT", "REGION",
            "VEHICLE", "VEH_AGE", "HIT_RUN"]
PERIODS = ["Pre", "COVID", "Post"]

# Same category list in every period so results are comparable
for c in FEATURES:
    m[c] = pd.Categorical(m[c], categories=sorted(m[c].unique()))
m["GROUP"] = m["YEAR"].astype(int).astype(str) + "_" + m["CASENUM"].astype(int).astype(str)

params = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=60,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=42, verbose=-1)

importance = {}
shap_by_period = {}

for p in PERIODS:
    s = m[m["PERIOD"] == p]
    X, y, w, g = s[FEATURES], s["SEVERE"], s["WEIGHT"], s["GROUP"]

    # Fair accuracy check: 5-fold CV keeping crashes together
    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        mdl = lgb.LGBMClassifier(**params)
        mdl.fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], mdl.predict_proba(X.iloc[te])[:, 1]))

    # Final model on the whole period, for explanation
    mdl = lgb.LGBMClassifier(**params)
    mdl.fit(X, y, sample_weight=w)
    sv = shap.TreeExplainer(mdl).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = pd.DataFrame(sv, columns=FEATURES, index=s.index)

    importance[p] = sv.abs().mean()
    shap_by_period[p] = (sv, s)
    print(f"{p:<6} records {len(s):>6,}   severe {y.mean()*100:4.1f}%   "
          f"CV ROC-AUC {np.mean(aucs):.3f} (+/- {np.std(aucs):.3f})")

# ---- 1) Importance by period ----
imp = pd.DataFrame(importance)[PERIODS]
imp["Change Post-Pre"] = imp["Post"] - imp["Pre"]
imp = imp.sort_values("Post", ascending=False)
print("\n1) Factor importance by period (mean |SHAP|):")
print(imp.round(3).to_string())

# ---- 2) Direction of specific conditions by period ----
# Positive = pushes toward severe injury; negative = pushes away
checks = [("LIGHT", "Dark-unlit"), ("LIGHT", "Dark-lit"), ("LIGHT", "Daylight"),
          ("SPEED_LIMIT", "50+"), ("SPEED_LIMIT", "40-45"), ("SPEED_LIMIT", "<=25"),
          ("VEHICLE", "SUV"), ("VEHICLE", "Pickup"), ("VEHICLE", "Car"),
          ("AGE_GRP", "65+"), ("HIT_RUN", "Yes")]
rows = []
for feat, cat in checks:
    row = {"Condition": f"{feat} = {cat}"}
    for p in PERIODS:
        sv, s = shap_by_period[p]
        mask = s[feat] == cat
        row[p] = sv.loc[mask, feat].mean() if mask.sum() >= 30 else np.nan
        row[f"n_{p}"] = int(mask.sum())
    rows.append(row)
eff = pd.DataFrame(rows).set_index("Condition")
print("\n2) Effect of specific conditions (mean SHAP, log-odds; + = more severe):")
print(eff[PERIODS].round(3).to_string())
print("\n   Records per condition:")
print(eff[[f"n_{p}" for p in PERIODS]].to_string())

# ---- Chart ----
top = imp.head(10)[PERIODS].iloc[::-1]
top.plot(kind="barh", figsize=(8, 6))
plt.title("Factor importance by period (SHAP)")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig(BASE / "shap_by_period.png", dpi=150)
print("\nChart saved: shap_by_period.png")