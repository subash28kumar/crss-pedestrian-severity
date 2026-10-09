import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = Path(__file__).resolve().parent
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

FEATURES = ["AGE_GRP", "SEX", "LIGHT", "WEATHER", "TIME_OF_DAY", "WEEKEND", "URBAN",
            "SPEED_LIMIT", "PEDLOC", "PEDPOS", "RELJCT2", "TYP_INT", "REGION",
            "VEHICLE", "VEH_AGE", "HIT_RUN", "PERIOD"]
LABELS = {
    "SPEED_LIMIT": "Posted speed limit", "PEDPOS": "Pedestrian position",
    "LIGHT": "Light condition", "AGE_GRP": "Pedestrian age", "TIME_OF_DAY": "Time of day",
    "TYP_INT": "Intersection type", "RELJCT2": "Relation to junction",
    "PEDLOC": "Pedestrian location", "VEHICLE": "Striking vehicle type",
    "VEH_AGE": "Striking vehicle age", "URBAN": "Urban / rural", "REGION": "Census region",
    "WEATHER": "Weather", "SEX": "Pedestrian sex", "WEEKEND": "Weekday / weekend",
    "HIT_RUN": "Hit-and-run", "PERIOD": "Period (pre / COVID / post)",
}

m = pd.read_parquet(BASE / "model_data.parquet")
m["CRASH_ID"] = m["YEAR"].astype(int).astype(str) + "_" + m["CASENUM"].astype(int).astype(str)
X = m[FEATURES].astype("category")
y = m["SEVERE"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(gss.split(X, y, groups=m["CRASH_ID"]))

model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                           min_child_samples=60, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=42, verbose=-1)
model.fit(X.iloc[tr], y[tr])

p = model.predict_proba(X.iloc[te])[:, 1]
w_te = m["WEIGHT"].values[te]
print(f"Test AUC: {roc_auc_score(y[te], p):.3f} (weighted {roc_auc_score(y[te], p, sample_weight=w_te):.3f})")
print(f"Test PR-AUC: {average_precision_score(y[te], p):.3f} (no-skill {y[te].mean():.3f})")

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X.iloc[te])
if isinstance(sv, list):
    sv = sv[1]
sv = np.asarray(sv)
if sv.ndim == 3:
    sv = sv[:, :, 1]

imp = pd.DataFrame({"Feature": FEATURES, "MeanAbsSHAP": np.abs(sv).mean(axis=0)})
imp["Label"] = imp["Feature"].map(LABELS)
imp = imp.sort_values("MeanAbsSHAP", ascending=True)
imp.sort_values("MeanAbsSHAP", ascending=False).round(4).to_csv(OUT / "fig2_data.csv", index=False)
print("\n", imp.sort_values("MeanAbsSHAP", ascending=False)[["Label", "MeanAbsSHAP"]].round(4).to_string(index=False))

# save SHAP values for later figures
pd.DataFrame(sv, columns=FEATURES).assign(**{f"{c}_val": X.iloc[te][c].astype(str).values for c in FEATURES}) \
  .to_parquet(BASE / "shap_test.parquet", index=False)

# --- plot ---
INK, INK2, MUTED, GRID, BAR = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#2a78d6"
plt.rcParams.update({"font.family": "Arial", "font.size": 10})
fig, ax = plt.subplots(figsize=(6.5, 5.2))
ax.barh(imp["Label"], imp["MeanAbsSHAP"], color=BAR, height=0.65, edgecolor="white", linewidth=1)
xmax = imp["MeanAbsSHAP"].max()
for i, v in enumerate(imp["MeanAbsSHAP"]):
    ax.text(v + xmax * 0.01, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
ax.set_xlim(0, xmax * 1.12)
ax.set_xlabel("Mean |SHAP value| (log-odds of KA injury)", color=INK2)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color(MUTED)
ax.tick_params(colors=INK2)
ax.tick_params(axis="y", length=0, labelcolor=INK)
fig.tight_layout()
fig.savefig(OUT / "fig2_shap.png", dpi=300)
fig.savefig(OUT / "fig2_shap.pdf")
print(f"\nSaved to {OUT}")