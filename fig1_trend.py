import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\skumar\Downloads\CRSS_2016_2024")
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

d = pd.read_parquet(BASE / "pedestrians_2016_2024.parquet")
for c in ["INJ_SEV", "REGION", "WEIGHT", "YEAR", "PSU", "PSUSTRAT"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d[d["INJ_SEV"].isin([0, 1, 2, 3, 4])].copy()
d["KA"] = d["INJ_SEV"].isin([3, 4]).astype(float)
d["WEST"] = d["REGION"] == 4

def share_ci(mask):
    w = d["WEIGHT"] * mask
    R = (w * d["KA"]).sum() / w.sum()
    z = w * (d["KA"] - R) / w.sum()
    g = pd.DataFrame({"h": d["PSUSTRAT"], "psu": d["PSU"], "z": z}).groupby(["h", "psu"])["z"].sum().reset_index()
    v = 0.0
    for _, s in g.groupby("h"):
        n = len(s)
        if n > 1:
            v += n / (n - 1) * ((s["z"] - s["z"].mean()) ** 2).sum()
    se = np.sqrt(v)
    return 100 * R, 100 * (R - 1.96 * se), 100 * (R + 1.96 * se)

years = list(range(2016, 2025))
rows = []
for grp, gm in [("West", d["WEST"]), ("Rest of U.S.", ~d["WEST"])]:
    for y in years:
        est, lo, hi = share_ci((gm & (d["YEAR"] == y)).astype(float))
        rows.append([grp, y, est, lo, hi])
res = pd.DataFrame(rows, columns=["Group", "Year", "KA", "Lo", "Hi"])
res.round(1).to_csv(OUT / "fig1_data.csv", index=False)

# --- plot ---
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
STYLE = {"West": ("#eb6834", "o", "-"), "Rest of U.S.": ("#2a78d6", "s", "--")}

plt.rcParams.update({"font.family": "Arial", "font.size": 10})
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.axvspan(2019.5, 2021.5, color="#f0efec", zorder=0)
ax.text(2020.5, 54, "2020–2021", ha="center", va="top", color=MUTED, fontsize=9)

for grp, (col, mk, ls) in STYLE.items():
    s = res[res["Group"] == grp]
    ax.fill_between(s["Year"], s["Lo"], s["Hi"], color=col, alpha=0.15, linewidth=0)
    ax.plot(s["Year"], s["KA"], color=col, marker=mk, markersize=6, linestyle=ls, linewidth=2, label=grp)
    last = s.iloc[-1]
    ax.text(2024.25, last["KA"], f"{grp}\n{last['KA']:.1f}%", color=INK, fontsize=9, va="center")

ax.set_xlim(2015.6, 2025.4)
ax.set_ylim(0, 55)
ax.set_xticks(years)
ax.set_ylabel("Pedestrians killed or seriously injured (KA), %", color=INK2)
ax.set_xlabel("Crash year", color=INK2)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color(MUTED)
ax.tick_params(colors=INK2)
ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig1_trend.png", dpi=300)
fig.savefig(OUT / "fig1_trend.pdf")
print(f"Saved to {OUT}")