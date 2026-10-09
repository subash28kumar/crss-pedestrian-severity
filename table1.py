import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

m = pd.read_parquet(BASE / "model_data.parquet")
m["SEVERITY"] = np.where(m["SEVERE"] == 1, "KA (killed/serious)", "BCO (other)")

VARS = [
    ("SEVERITY", "Injury severity"),
    ("AGE_GRP", "Pedestrian age"),
    ("SEX", "Pedestrian sex"),
    ("LIGHT", "Light condition"),
    ("TIME_OF_DAY", "Time of day"),
    ("SPEED_LIMIT", "Posted speed limit (mph)"),
    ("URBAN", "Urban / rural"),
    ("VEHICLE", "Striking vehicle type"),
    ("VEH_AGE", "Striking vehicle age"),
    ("REGION", "Census region"),
]
REGION_NAMES = {"c1": "Northeast", "c2": "Midwest", "c3": "South", "c4": "West"}
PERIODS = ["Pre", "COVID", "Post"]
HEAD = {"Pre": "2016-2019", "COVID": "2020-2021", "Post": "2022-2024"}

rows = []
n_row = ["Records (unweighted n)", ""]
w_row = ["Weighted estimate (thousands)", ""]
for p in PERIODS + ["All"]:
    g = m if p == "All" else m[m["PERIOD"] == p]
    n_row.append(f"{len(g):,}")
    w_row.append(f"{g['WEIGHT'].sum() / 1000:,.0f}")
rows += [n_row, w_row]

for col, label in VARS:
    vals = m[col].replace(REGION_NAMES) if col == "REGION" else m[col]
    levels = sorted(vals.unique(), key=lambda v: (v in ("Unknown", "Other/Unknown"), v))
    for i, lv in enumerate(levels):
        r = [label if i == 0 else "", lv]
        for p in PERIODS + ["All"]:
            mask = np.ones(len(m), bool) if p == "All" else (m["PERIOD"] == p).values
            w = m.loc[mask, "WEIGHT"]
            r.append(f"{100 * w[vals[mask] == lv].sum() / w.sum():.1f}")
        rows.append(r)

t = pd.DataFrame(rows, columns=["Characteristic", "Level"] + [HEAD[p] for p in PERIODS] + ["All years"])
t.to_csv(OUT / "table1.csv", index=False)
print(t.to_string(index=False))
print(f"\nSaved to {OUT / 'table1.csv'}")