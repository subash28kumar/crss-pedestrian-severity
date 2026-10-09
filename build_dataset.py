import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\skumar\Downloads\CRSS_2016_2024")
d = pd.read_parquet(BASE / "pedestrians_2016_2024.parquet")

def num(col):
    return pd.to_numeric(d[col], errors="coerce")

def binned(values, edges, labels):
    """Cut numbers into groups; anything missing becomes 'Unknown'."""
    out = pd.cut(values, edges, labels=labels)
    return out.cat.add_categories("Unknown").fillna("Unknown").astype(str)

# ---- Outcome: killed or seriously injured (KA) vs not ----
d["INJ_SEV"] = num("INJ_SEV")
d = d[d["INJ_SEV"].isin([0, 1, 2, 3, 4])].copy()
m = pd.DataFrame(index=d.index)
m["SEVERE"] = d["INJ_SEV"].isin([3, 4]).astype(int)

# ---- Time period ----
m["YEAR"] = num("YEAR")
m["PERIOD"] = binned(m["YEAR"], [2015, 2019, 2021, 2024], ["Pre", "COVID", "Post"])

# ---- Pedestrian ----
age = num("AGE").where(lambda s: s < 120)
m["AGE_GRP"] = binned(age, [-1, 15, 24, 44, 64, 120], ["0-15", "16-24", "25-44", "45-64", "65+"])
m["SEX"] = num("SEX").map({1: "Male", 2: "Female"}).fillna("Unknown")

# ---- Environment ----
m["LIGHT"] = num("LGT_COND").map({1: "Daylight", 2: "Dark-unlit", 3: "Dark-lit",
                                  4: "Dawn", 5: "Dusk", 6: "Dark-unknown"}).fillna("Unknown")
m["WEATHER"] = num("WEATHER").map({1: "Clear", 10: "Cloudy", 2: "Rain",
                                   3: "Snow/Sleet", 4: "Snow/Sleet", 5: "Fog"}).fillna("Other/Unknown")
hour = num("HOUR").where(lambda s: s <= 23)
m["TIME_OF_DAY"] = binned(hour, [-1, 5, 9, 15, 19, 23],
                          ["Night 0-5", "Morning 6-9", "Midday 10-15", "Evening 16-19", "Late 20-23"])
m["WEEKEND"] = num("DAY_WEEK").isin([1, 7]).map({True: "Weekend", False: "Weekday"})
m["URBAN"] = num("URBANICITY").map({1: "Urban", 2: "Rural"}).fillna("Unknown")

# ---- Road ----
spd = num("SV_VSPD_LIM").where(lambda s: (s > 0) & (s <= 80))
m["SPEED_LIMIT"] = binned(spd, [0, 25, 35, 45, 80], ["<=25", "30-35", "40-45", "50+"])

# Location codes kept as categories (labels added from the manual later)
for c in ["PEDLOC", "PEDPOS", "RELJCT2", "TYP_INT", "REGION"]:
    m[c] = "c" + num(c).fillna(-1).astype(int).astype(str)

# ---- Striking vehicle (BODY_TYP ranges: consistent across all years) ----
b = num("SV_BODY_TYP")
m["VEHICLE"] = np.select(
    [b.between(1, 9), b.between(14, 19), b.between(20, 29), b.between(30, 39),
     b.between(40, 48), b.between(50, 59), b.between(60, 79), b.between(80, 89)],
    ["Car", "SUV", "Van", "Pickup",
     "Other light truck", "Bus", "Heavy truck", "Motorcycle"],
    default="Other/Unknown")

# Vehicle age from the officer-recorded model year (not the aux file)
modyr = num("SV_MOD_YEAR").where(lambda s: (s > 1900) & (s < 2030))
vage = (m["YEAR"] - modyr).where(lambda s: s >= -1)
m["VEH_AGE"] = binned(vage, [-2, 4, 9, 14, 100], ["0-4 yrs", "5-9 yrs", "10-14 yrs", "15+ yrs"])
m["HIT_RUN"] = (num("SV_HIT_RUN") == 1).map({True: "Yes", False: "No"})

# ---- Survey design (needed for correct national estimates) ----
for c in ["WEIGHT", "PSU", "STRATUM", "CASENUM"]:
    m[c] = num(c)

out = BASE / "model_data.parquet"
m.to_parquet(out, index=False)

print("Saved:", out)
print("Rows:", f"{len(m):,}")
print("Severe share (unweighted): ", round(m["SEVERE"].mean() * 100, 1), "%")
print("\nVehicle type by period (weighted %):")
t = m.groupby(["PERIOD", "VEHICLE"])["WEIGHT"].sum().unstack(fill_value=0)
print((t.div(t.sum(axis=1), axis=0) * 100).round(1).to_string())
print("\nVehicle age by period (weighted %):")
t = m.groupby(["PERIOD", "VEH_AGE"])["WEIGHT"].sum().unstack(fill_value=0)
print((t.div(t.sum(axis=1), axis=0) * 100).round(1).to_string())