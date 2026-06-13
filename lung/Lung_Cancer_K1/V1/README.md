# Lung Cancer — K1 pipeline

Internal-results pipeline for the lung model, in two parts:
1. **Code scoring** (`1_Top_Snomed/`) — rank which clinical codes separate cancer vs non-cancer.
2. **Modeling** (`run_lookback_experiment.py`) — train + evaluate the model on an internal
   test split, across the four lookback windows.

The two parts are independent (you can run either on its own); both use the **same cohort**.

**Prerequisites:** run on a VM with **BigQuery access** (project `prj-cts-ai-dev-sp`) and the
Python env (`lungenv`). All inputs come from BigQuery; all outputs land **inside this folder**.

```
Lung_Cancer_K1/
├── run_lookback_experiment.py   # entry point for modeling (Steps 2+3)
├── _run_pipeline.py             # shared driver: cohort SQL -> events -> FE -> model
├── 1_Top_Snomed/                # Step 1 - feature (code) scoring
├── 2_FE/                        # feature-engineering modules + cohort SQL
├── 3_Modeling/                  # training + metrics
└── 4_Holdout/                   # Step 4 - touch-once held-out eval + Platt recalibration
```

---

## Step 1 — Top-SNOMED feature scoring  (folder: `1_Top_Snomed/`)
Rank which observation/medication codes separate cancer vs non-cancer. Each command does
**both horizons (12mo + 1mo) in one run**, writing to separate `Data/{horizon}/` + `output/{horizon}/`.

```bash
cd 1_Top_Snomed
python build_score_counts.py             # 1a  (builds 12mo + 1mo)
python "Combined Scoring (ML+Stat).py"   # 1b  (scores 12mo + 1mo)
```
- **1a `build_score_counts.py`** — for each horizon, runs `SQL/{horizon}_1to1.sql` on BigQuery and writes, to `1_Top_Snomed/Data/{horizon}/`:
  - count CSVs → `Lung_{positive,negative}_{obs,meds}.csv`
  - patient matrices → `Lung_patient_{obs,meds}_matrix.csv` (real per-patient data for the ML step)
- **1b `Combined Scoring (ML+Stat).py`** — for each horizon, ranks codes (statistical OR/χ² + real-data ML, rank-normalised) and writes, to `1_Top_Snomed/output/{horizon}/Scores_lung/`:
  - `lung_obs_all.csv`, `lung_meds_all.csv`, **`lung_combined_all.csv`** ← ranked deliverable
  - (the matrices are **required**; the scorer stops if they're missing — no synthetic data.)

> Run 1a before 1b. Output of 1a is the input to 1b. Each script does **both horizons in one run**.

---

## Step 2 + 3 — Train + internal results  (entry: `run_lookback_experiment.py`)
One command runs FE (`2_FE/`) → model (`3_Modeling/`) per lookback window. **By default it LOOPS
through all four windows (5yr / 10yr / 20yr / lifetime); pass a single window name to run only that
one** (reuses the cached cohort events).

```bash
GAP=12 NC_RATIO=1 python run_lookback_experiment.py        # default: LOOP all windows (5/10/20yr/lifetime)
GAP=12 NC_RATIO=1 python run_lookback_experiment.py 5yr    # ONLY this window (or 10yr|20yr|lifetime)
```
- Pulls the cohort once (`2_FE/SQL/{GAP}mo_1to{NC_RATIO}.sql`) from BigQuery, builds the feature
  matrix, trains the model (80/10/10 train/val/internal-test), and reports the **free / Youden**
  operating point only.
- Produces, under **`12mo_1to1/lookback/`**:
  - `{5yr,10yr,20yr,lifetime}/` → `train_features.csv`, `model_<window>_1to1.joblib`,
    `metrics.csv` (one `internal_free` row), score arrays (`*.npz`)
  - **`lookback_internal_summary.csv`** ← internal metrics for all windows side by side

> `GAP` = 12 or 1 (12-/1-month horizon). `NC_RATIO` = 1 (controls ratio). Internal-only:
> no real-world held-out is run here.

---

## Feature engineering (`2_FE/`)

The cohort events are turned into a one-row-per-patient matrix by
`transform_features.extract_lung_risk_factors` (per-concept occurrence + value trends, **lifetime**),
then enriched by `enhanced_features.enrich` (always runs — no flags). Per clinical concept:

- **Occurrence dynamics** — count, recency, decay (recency-weighted, all history), `recent_ratio`,
  **acceleration**, frequency, intervals, first/second-half frequency, `is_worsening`.
- **Value / trend** (lab concepts) — first/latest/mean/median/min/max/std, abs & % change, trend
  slope + correlation, **`VALUE_ACCEL`** (recent-half slope − older-half slope = is the change steepening).
- **Level vs population** — `xpoll` cohort + age-band percentiles.
- **Cross-concept** — NLR/PLR, symptom/comorbid clusters, consultation dynamics, problem-list
  (active/significant) flags, smoking dose, interaction terms.

**Time windows (always computed):** per-concept **disjoint bands** `[0–6,6–18,18–36,36–72,72–999]mo`
+ **cumulative** `last-6/12/24/60mo`, each with count/present (+ value mean/latest/slope for lab
concepts). Both give count/value-per-window views *alongside* the lifetime features.

**No trends are truncated.** All trends/intervals/slopes use the **full lifetime**; the bands/
`recent_ratio`/`accel` are additional *windowed views*, not limits. All windowed/recency features
**start at month 0 = the start of available data** (the cohort SQL already applies the gap cutoff).
Windows count back from each patient's most-recent event (gap-agnostic, inference-safe), so they're
the same for every horizon.

---

## Step 4 — Held-out evaluation + Platt recalibration  (folder: `4_Holdout/`)
Touch-once eval of a trained lookback model on the real **500/50k** held-out (0% train overlap), then
**Platt scaling** to calibrate probabilities (the documented approach). Per-patient FE + the **TRAIN**
xpoll reference (leakage-safe), scored with the train-fitted transforms.

```bash
GAP=12 WINDOW=5yr python 4_Holdout/evaluate_heldout.py     # 12mo, 5yr model
GAP=1  WINDOW=5yr python 4_Holdout/evaluate_heldout.py     # 1mo,  5yr model
```
Reports AUROC/AUPRC + Brier/ECE (raw vs Platt vs isotonic) + Sens/Spec/PPV/NPV at free-Youden on the
70% test slice; saves `platt_calib_{window}.joblib` + reliability curve into the window's output dir.
(King-Zeng true-prevalence offset is a separate *future deployment* step — not applied here.)

---

## What to do with the results
- **Step 1** → the ranked code lists (`lung_combined_all.csv`) for clinical review / feature selection.
- **Step 2+3** → pick the best lookback from `lookback_internal_summary.csv`. (Real-world held-out
  evaluation is a separate step, not part of this folder.)

---

## Pipeline at a glance (full flow)

```
STEP 1 — code scoring   (run inside 1_Top_Snomed/, in order; each does BOTH 12mo + 1mo)
  BigQuery (SQL/12mo_1to1.sql + SQL/1mo_1to1.sql)
     │  python build_score_counts.py
     ▼
  Data/{12mo,1mo}/  Lung_{pos,neg}_{obs,meds}.csv  +  Lung_patient_{obs,meds}_matrix.csv
     │  python "Combined Scoring (ML+Stat).py"
     ▼
  output/{12mo,1mo}/Scores_lung/lung_combined_all.csv          ← ranked codes (deliverable)

STEP 2+3 — train + internal results   (run from the folder root)
  BigQuery (2_FE/SQL/12mo_1to1.sql)
     │  GAP=12 NC_RATIO=1 python run_lookback_experiment.py
     ▼  2_FE: events → feature matrix    →    3_Modeling: train 80/10/10 (free/Youden)
  12mo_1to1/lookback/{5yr,10yr,20yr,lifetime}/   (model_*.joblib + metrics.csv)
     │
     ▼
  12mo_1to1/lookback/lookback_internal_summary.csv  ← compare windows, pick best
```

Order: **Step 1 (1a → 1b)** and **Step 2+3** are independent tracks; within Step 1, run
`build_score_counts.py` before the scorer. Inputs come from BigQuery; all outputs stay in this folder.
