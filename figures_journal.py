"""
Redraw Figures 1-3 in Traffic Injury Prevention style:
Times New Roman, no grid lines, no legend boxes, legend not on the right,
speeds in km/h (mph), readable when printed in black and white.
Reads the CSV files already saved in the figures folder (no re-analysis).
Outputs: figures/journal/Figure1.tif, Figure2.tif, Figure3.tif (+ PDF copies).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
SRC = BASE / "figures"
OUT = SRC / "journal"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "legend.frameon": False,
    "savefig.dpi": 600,
})
BLACK, MID = "#000000", "#777777"
WEST_C, REST_C = "#d55e00", "#0072b2"


def clean_axes(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=BLACK, length=3, width=0.8)


def save(fig, name):
    kw = {"bbox_inches": "tight", "pad_inches": 0.05}
    fig.savefig(OUT / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **kw)
    fig.savefig(OUT / f"{name}.pdf", **kw)
    fig.savefig(OUT / f"{name}.png", dpi=200, **kw)
    plt.close(fig)
    print(f"Saved {name}")


# ---------------- Figure 1: KA share by year, West vs rest ----------------
d1 = pd.read_csv(SRC / "fig1_data.csv")
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.axvspan(2019.5, 2021.5, color="#e6e6e6", zorder=0, linewidth=0)
styles = {"West": (WEST_C, "o", "-", "white"), "Rest of U.S.": (REST_C, "s", "--", None)}
for grp, (col, mk, ls, face) in styles.items():
    s = d1[d1["Group"] == grp]
    ax.errorbar(s["Year"], s["KA"], yerr=[s["KA"] - s["Lo"], s["Hi"] - s["KA"]],
                color=col, marker=mk, markersize=4, linestyle=ls, linewidth=1.2,
                elinewidth=0.7, capsize=2, markerfacecolor=face or col, label=grp)
ax.set_xticks(range(2016, 2025))
ax.set_xticklabels([str(y) for y in range(2016, 2025)], rotation=45)
ax.set_xlim(2015.5, 2024.5)
ax.set_ylim(0, 55)
ax.set_ylabel("Pedestrians killed or seriously\ninjured (KA), %")
ax.set_xlabel("Crash year")
clean_axes(ax)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=2.5, borderaxespad=0.2)
fig.tight_layout()
save(fig, "Figure1")

# ---------------- Figure 2: global SHAP importance ----------------
d2 = pd.read_csv(SRC / "fig2_data.csv").sort_values("MeanAbsSHAP")
fig, ax = plt.subplots(figsize=(3.5, 3.7))
ax.barh(d2["Label"], d2["MeanAbsSHAP"], color="#4d4d4d", height=0.65)
xmax = d2["MeanAbsSHAP"].max()
for i, v in enumerate(d2["MeanAbsSHAP"]):
    ax.text(v + xmax * 0.015, i, f"{v:.3f}", va="center", fontsize=7)
ax.set_xlim(0, xmax * 1.18)
ax.set_xlabel("Mean |SHAP value| (log-odds)")
clean_axes(ax)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
save(fig, "Figure2")

# ---------------- Figure 3: level effects for top six predictors ----------------
d3 = pd.read_csv(SRC / "fig3_data.csv")
SPEED_OLD = "Posted speed limit (mph)"
SPEED_NEW = "Posted speed limit, km/h (mph)"
SPEED_LEVELS = {"25 or less": "\u226440 (\u226425)", "30-35": "48\u201356 (30\u201335)",
                "40-45": "64\u201372 (40\u201345)", "50+": "\u226580 (\u226550)", "Unknown": "Unknown"}
d3["Level"] = np.where(d3["Factor"] == SPEED_OLD, d3["Level"].map(SPEED_LEVELS).fillna(d3["Level"]), d3["Level"])
d3["Level"] = d3["Level"].str.replace(r"(\d)-(\d)", "\\1\u2013\\2", regex=True)
d3["Factor"] = d3["Factor"].replace({SPEED_OLD: SPEED_NEW})
ORDERED = {SPEED_NEW, "Pedestrian age", "Time of day"}
factors = list(dict.fromkeys(d3["Factor"]))

fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.4), sharex=True)
letters = "abcdef"
for k, (ax, fac) in enumerate(zip(axes.flat, factors)):
    g = d3[d3["Factor"] == fac].reset_index(drop=True)
    ypos = np.arange(len(g))[::-1] if fac in ORDERED else np.arange(len(g))
    ax.axvline(0, color=MID, linewidth=0.8)
    for y, v in zip(ypos, g["MeanSHAP"]):
        raises = v > 0
        col = WEST_C if raises else REST_C
        ax.hlines(y, 0, v, color=col, linewidth=1.4)
        ax.plot(v, y, marker="o", markersize=5, color=col,
                markerfacecolor=col if raises else "white", markeredgewidth=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{lv} (n={int(n):,})" for lv, n in zip(g["Level"], g["n"])], fontsize=7.5)
    ax.set_title(f"({letters[k]}) {fac}", loc="right", fontsize=8.5, fontweight="bold")
    clean_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
for ax in axes[-1]:
    ax.set_xlabel("Mean SHAP value (log-odds)")
fig.tight_layout(w_pad=0.8, h_pad=1.0)
save(fig, "Figure3")

print(f"\nAll figures saved in: {OUT}")