import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
YEARS = range(2016, 2025)

def find_file(year, name, aux=False):
    folder = BASE / (f"CRSS{year}AuxiliaryCSV" if aux else f"CRSS{year}CSV")
    if not folder.exists():
        print(f"  folder not found: {folder.name}")
        return None
    for f in folder.rglob("*"):
        if f.name.lower() == name.lower():
            return f
    return None

def read(year, name, aux=False):
    f = find_file(year, name, aux)
    if f is None:
        print(f"  missing file: {year} {name}")
        return None
    df = pd.read_csv(f, encoding="latin-1", low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    return df

def keys_in(df, candidates):
    return [k for k in candidates if k in df.columns]

all_years = []
for y in YEARS:
    print(f"Loading {y} ...")
    acc  = read(y, "accident.csv")
    per  = read(y, "person.csv")
    veh  = read(y, "vehicle.csv")
    pbt  = read(y, "pbtype.csv")
    accx = read(y, "acc_aux.csv", aux=True)
    perx = read(y, "per_aux.csv", aux=True)
    vehx = read(y, "veh_aux.csv", aux=True)

    if per is None or acc is None:
        print(f"  skipping {y}: core files missing")
        continue

    # Pedestrians only (person type 5)
    ped = per[per["PER_TYP"] == 5].copy()

    # Crash-level details
    ped = ped.merge(acc, on="CASENUM", how="left", suffixes=("", "_ACC"))
    if accx is not None:
        ped = ped.merge(accx, on="CASENUM", how="left", suffixes=("", "_ACCX"))

    # Person helper variables
    if perx is not None:
        ped = ped.merge(perx, on=keys_in(perx, ["CASENUM", "VEH_NO", "PER_NO"]),
                        how="left", suffixes=("", "_PERX"))

    # Pedestrian crash type and location
    if pbt is not None:
        ped = ped.merge(pbt, on=keys_in(pbt, ["CASENUM", "VEH_NO", "PER_NO"]),
                        how="left", suffixes=("", "_PBT"))

    # The vehicle that struck the pedestrian
    if "STR_VEH" in ped.columns and veh is not None:
        v = veh.copy()
        if vehx is not None:
            v = v.merge(vehx, on=["CASENUM", "VEH_NO"], how="left", suffixes=("", "_VEHX"))
        v = v.add_prefix("SV_").rename(columns={"SV_CASENUM": "CASENUM", "SV_VEH_NO": "STR_VEH"})
        ped = ped.merge(v, on=["CASENUM", "STR_VEH"], how="left")

    ped["YEAR"] = y
    all_years.append(ped)
    print(f"  {y}: {len(ped):,} pedestrian records")

data = pd.concat(all_years, ignore_index=True, sort=False)

# Make mixed-type columns safe for saving
for c in data.columns:
    if data[c].dtype == "object":
        data[c] = data[c].astype("string")

out = BASE / "pedestrians_2016_2024.parquet"
data.to_parquet(out, index=False)

print("\nDONE - saved to", out)
print("Total pedestrian records:", f"{len(data):,}")
print("\nRecords per year:")
print(data.groupby("YEAR").size().to_string())

print("\nKey columns present:")
for c in ["INJ_SEV", "AGE", "SEX", "LGT_COND", "WEATHER", "WEIGHT",
          "PSU", "STRATUM", "STR_VEH", "SV_BODY_TYP", "SV_A_BODY",
          "PEDLOC", "PEDPOS"]:
    print(f"  {c}: {'yes' if c in data.columns else 'NO'}")