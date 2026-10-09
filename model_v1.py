import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\skumar\Downloads\CRSS_2016_2024")
m = pd.read_parquet(BASE / "model_data.parquet")

FEATURES = ["PERIOD", "AGE_GRP", "SEX", "LIGHT", "WEATHER", "TIME_OF_DAY", "WEEKEND",
            "URBAN", "SPEED_LIMIT", "PEDLOC", "PEDPOS", "RELJCT2", "TYP_INT", "REGION",
            "VEHICLE", "VEH_AGE", "HIT_RUN"]

X = m[FEATURES].astype("category")
y = m["SEVERE"]
w = m["WEIGHT"]

# Keep all pedestrians from the same crash in the same split
groups = m["YEAR"].astype(int).astype(str) + "_" + m["CASENUM"].astype(int).astype(str)
train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                           .split(X, y, groups))
Xtr, Xte, ytr, yte = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
wtr = w.iloc[train_idx]

print(f"Train: {len(Xtr):,}   Test: {len(Xte):,}   Severe share: {y.mean()*100:.1f}%\n")

def report(name, p):
    print(f"{name:<22} ROC-AUC {roc_auc_score(yte, p):.3f}   "
          f"PR-AUC {average_precision_score(yte, p):.3f}   "
          f"Brier {brier_score_loss(yte, p):.3f}")

# ---- Baseline: logistic regression ----
Xtr_d = pd.get_dummies(Xtr, drop_first=True)
Xte_d = pd.get_dummies(Xte, drop_first=True).reindex(columns=Xtr_d.columns, fill_value=0)
lr = LogisticRegression(max_iter=2000)
lr.fit(Xtr_d, ytr, sample_weight=wtr)
report("Logistic regression", lr.predict_proba(Xte_d)[:, 1])

# ---- LightGBM ----
gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                         min_child_samples=50, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=42, verbose=-1)
gbm.fit(Xtr, ytr, sample_weight=wtr)
p_gbm = gbm.predict_proba(Xte)[:, 1]
report("LightGBM", p_gbm)
print(f"{'(no-skill PR-AUC)':<22} {yte.mean():.3f}")

# ---- SHAP: which factors matter most ----
explainer = shap.TreeExplainer(gbm)
sv = explainer.shap_values(Xte)
if isinstance(sv, list):
    sv = sv[1]
imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)

print("\nTop factors (mean |SHAP|, higher = more influence):")
print(imp.round(4).to_string())

plt.figure(figsize=(7, 6))
imp.sort_values().plot(kind="barh")
plt.title("What drives severe pedestrian injury (SHAP)")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig(BASE / "shap_importance_v1.png", dpi=150)
print("\nChart saved: shap_importance_v1.png")