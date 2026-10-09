[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23271533.svg)](https://doi.org/10.5281/zenodo.23271533)

# Post-Pandemic Pedestrian Injury Severity in the United States

Code for the study *"Post-pandemic shifts in pedestrian injury severity in the United States: an explainable machine learning and survey-weighted analysis of CRSS 2016–2024"* (manuscript submitted to *Traffic Injury Prevention*, 2026).

**Author:** Subash Kumar ([ORCID 0009-0003-6661-3424](https://orcid.org/0009-0003-6661-3424))

## Key findings

- The weighted share of struck pedestrians killed or seriously injured (KA) rose from **24.5%** (2016–2019) to **30.0%** (2022–2024).
- Posted speed limit, pedestrian position, light condition, age and time of day are the strongest predictors (LightGBM test AUC 0.738).
- Changes in crash characteristics explain about **half** of the national increase.
- The increase was concentrated in the **Western U.S.** (KA 25.1% → 38.0%, vs. 24.3% → 27.7% elsewhere). After adjustment, the Western excess remained: **adjusted OR 1.43 (95% CI 1.14–1.88)**.

## Data

This study uses NHTSA's **Crash Report Sampling System (CRSS)**, 2016–2024. The data are public and are not included in this repository.

1. Download the yearly CSV files from NHTSA: https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system
2. Extract each year into one folder, named `CRSS{YEAR}CSV` and `CRSS{YEAR}AuxiliaryCSV` (for example `CRSS2016CSV`, `CRSS2016AuxiliaryCSV`).
3. Place all scripts in that same folder and run them from there.

## Requirements

Python 3.10+. Install the packages with `pip install pandas pyarrow numpy scikit-learn lightgbm shap statsmodels matplotlib`

## How to run (in order)

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `load_crss.py` | Merges CRSS files, selects pedestrians, links the striking vehicle |
| 2 | `build_dataset.py` | Builds the modeling dataset (outcome, predictors, survey weights) |
| 3 | `model_v1.py`, `model_periods.py` | Logistic regression and LightGBM models, SHAP |
| 4 | `bootstrap.py` | Crash-cluster bootstrap of model effects by period |
| 5 | `decomposition.py`, `factor_contrib.py` | Explained vs. unexplained change, factor contributions |
| 6 | `region.py`, `region_kabco.py` | Regional patterns and KABCO breakdown |
| 7 | `region_ci.py` | Design-based CIs and difference-in-differences |
| 8 | `region_adjusted.py`, `region_boot.py` | Adjusted West × post-pandemic model, stratified PSU bootstrap |
| 9 | `region_robust.py` | Sensitivity analyses |
| 10 | `table1.py`, `fig1_trend.py`, `fig2_shap.py`, `fig3_levels.py` | Table 1 and Figures 1–3 |
| 11 | `figures_journal.py` | Journal-style versions of Figures 1–3 (TIFF/PDF) |

## Methods notes

- Outcome: KA injury (KABCO fatal or suspected serious; `INJ_SEV` 3–4). `MAX_SEV` excluded to avoid leakage.
- Vehicle type from `BODY_TYP` ranges (the `A_BODY` grouping changed in 2020).
- Train/test splits grouped by crash.
- Survey variance: Taylor linearization (`PSUSTRAT`, `PSU`) and Rao–Wu rescaling bootstrap.

## How to cite

If you use this code, please cite:

Kumar S. 2026. crss-pedestrian-severity: Post-pandemic pedestrian injury severity analysis, CRSS 2016–2024 (v1.0.1) [software]. Zenodo. https://doi.org/10.5281/zenodo.23271533

You can also use the **"Cite this repository"** button on the right side of this page.

## Use of AI tools

Claude (Anthropic) was used to help write analysis code and draft documentation. The author designed the study, ran all analyses and verified all results.

## License

MIT. See `LICENSE`.
