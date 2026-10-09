import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

s = pd.read_parquet(BASE / "shap_test.parquet")

PANELS = [
    ("SPEED_LIMIT", "Posted speed limit (mph)", ["<=25", "30-35", "40-45", "50+", "Unknown"],
     {"<=25": "25 or less"}),
    ("PEDPOS", "Pedestrian position", None,
     {"c1": "Intersection area", "c2": "Crosswalk area", "c3": "Travel lane",
      "c4": "Shoulder / bike / parking lane", "c5": "Sidewalk / path / driveway",
      "c6": "Unpaved right-of-way", "c7": "Non-trafficway driveway",
      "c8": "Parking lot / other", "c9": "Other / unknown"}),
    ("LIGHT", "Light condition", None, {}),
    ("AGE_GRP", "Pedestrian age", ["0-15", "16-24", "25-44", "45-64", "65+", "Unknown"], {}),
    ("TIME_OF_DAY", "Time of day", ["Night 0-5", "Morning 6-9", "Midday 10-15",
                                     "Evening 16-19", "Late 20-23", "Unknown"],
     {"Night 0-5": "Night (0-5)", "Morning 6-9": "Morning (6-9)", "Midday 10-15": "Midday (10-15)",
      "Evening 16-19": "Evening (16-19)", "Late 20-23": "Late (20-23)"}),
    ("RELJCT2", "Relation to junction", None,
     {"c1": "Non-junction", "c2": "Intersection", "c3": "Intersection-related",
      "c4": "Driveway access", "c5": "Ramp-related", "c8": "Driveway-related",
      "c18": "Through roadway (interchange)", "c19": "Other interchange area",
      "c20": "Ramp", "c98": "Not reported", "c99": "Unknown"}),
]
MIN_N = 50

POS, NEG, INK, INK2, MUTED, GRID = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
plt.rcParams.update({"font.family": "Arial", "font.size": 9})
fig, axes = plt.subplots(3, 2, figsize=(9.5, 9), sharex=True)

rows = []
for ax, (feat, title, order, names) in zip(axes.flat, PANELS):
    g = s.groupby(f"{feat}_val")[feat].agg(["mean", "count"])
    g = g[g["count"] >= MIN_N]
    if order:
        g = g.reindex([o for o in order if o in g.index])
    else:
        g = g.sort_values("mean")
    labels = [f"{names.get(i, i)}  (n={int(n)})" for i, n in zip(g.index, g["count"])]
    ypos = np.arange(len(g))[::-1] if order else np.arange(len(g))
    colors = [POS if v > 0 else NEG for v in g["mean"]]
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.hlines(ypos, 0, g["mean"], color=colors, linewidth=2)
    ax.scatter(g["mean"], ypos, color=colors, s=40, zorder=3, edgecolor="white", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, color=INK)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK2)
    for i, v, n in zip(g.index, g["mean"], g["count"]):
        rows.append([title, names.get(i, i), round(v, 4), int(n)])

fig.supxlabel("Mean SHAP value (log-odds of KA injury)   |   orange = raises KA risk, blue = lowers KA risk; levels with n < 50 omitted",
              fontsize=8.5, color=INK2)
fig.tight_layout(w_pad=2)
fig.savefig(OUT / "fig3_levels.png", dpi=300)
fig.savefig(OUT / "fig3_levels.pdf")
out = pd.DataFrame(rows, columns=["Factor", "Level", "MeanSHAP", "n"])
out.to_csv(OUT / "fig3_data.csv", index=False)
print(out.to_string(index=False))
print(f"\nSaved to {OUT}")