# Lung Cancer — K1 pipeline

End-to-end pipeline for the lung model, in **four steps**:

1. **Code scoring** (`1_Top_Snomed/`) — rank which clinical codes separate cancer vs non-cancer.
2. **Feature engineering** (`2_FE/`) — turn cohort events into a one-row-per-patient matrix.
3. **Modeling** (`3_Modeling/`) — train the learner panel, pick the best on an internal test split.
4. **Held-out evaluation** (`4_Holdout/`) — score the chosen model on the touch-once held-out + calibrate.

Step 1 is an **independent** track (clinical review / feature shortlisting). **Steps 2 + 3 run from a
single command** (`run_lookback_experiment.py` — it builds the Step 2 matrix, then trains in Step 3).
Step 4 runs once a Step 3 model exists. All steps use the **same cohort**.

**Prerequisites:** run on a VM with **BigQuery access** (project `prj-cts-ai-dev-sp`) and the Python
env (`lungenv`). All inputs come from BigQuery; all outputs land **inside this folder**.

```
Lung_Cancer_K1/
├── run_lookback_experiment.py   # entry point for Steps 2 + 3 (FE -> model, per lookback window)
├── _run_pipeline.py             # shared driver: cohort SQL -> events -> FE -> model
├── 1_Top_Snomed/                # Step 1 — code (feature) scoring
├── 2_FE/                        # Step 2 — feature-engineering modules + cohort SQL
├── 3_Modeling/                  # Step 3 — training + metrics (learner panel)
└── 4_Holdout/                   # Step 4 — touch-once held-out eval + Platt recalibration
```

---

## Step 1 — Top-SNOMED code scoring  (`1_Top_Snomed/`)
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

> Run 1a before 1b — 1a's output is 1b's input. Each script does **both horizons in one run**.

---

## Step 2 — Feature engineering  (`2_FE/`)
> Not a separate command — FE runs **automatically inside the Step 3 run** (`run_lookback_experiment.py`),
> which pulls the cohort and builds the matrix before training.

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

## Step 3 — Modeling + internal results  (`3_Modeling/`, entry `run_lookback_experiment.py`)
One command runs Step 2 (FE) → Step 3 (model) per lookback window. **By default it LOOPS through all
four windows (5yr / 10yr / 20yr / lifetime); pass a single window name to run only that one** (reuses
the cached cohort events).

```bash
GAP=12 NC_RATIO=1 python run_lookback_experiment.py        # default: LOOP all windows (5/10/20yr/lifetime)
GAP=12 NC_RATIO=1 python run_lookback_experiment.py 5yr    # ONLY this window (or 10yr|20yr|lifetime)
GAP=12 NC_RATIO=5 python run_lookback_experiment.py 5yr    # 1:5 cohort (uses 2_FE/SQL/12mo_1to5.sql)
GAP=1  NC_RATIO=5 python run_lookback_experiment.py 5yr    # 1:5, 1mo horizon
```
- Pulls the cohort once (`2_FE/SQL/{GAP}mo_1to{NC_RATIO}.sql`) from BigQuery, builds the feature
  matrix, trains the panel (80/10/10 train / calib-val / internal-test), and reports the
  **free operating point (Youden-J = max sens+spec−1)** only.
- Produces, under **`{GAP}mo_1to{NC_RATIO}/lookback/`** (e.g. `12mo_1to1/`, `12mo_1to5/`):
  - `{5yr,10yr,20yr,lifetime}/` → `train_features.csv`, `model_<window>_1to<NC_RATIO>.joblib`,
    `metrics.csv` (one `internal_free` row), score arrays (`internal_scores_*.npz`, `per_model_scores_*.npz`)
  - **`lookback_internal_summary.csv`** ← internal metrics for all windows side by side

> `GAP` = 12 or 1 (12-/1-month horizon). `NC_RATIO` = 1 or 5 (non-cancer:cancer ratio; picks
> `2_FE/SQL/{GAP}mo_1to{NC_RATIO}.sql` and writes to `{GAP}mo_1to{NC_RATIO}/`). Internal-only:
> no real-world held-out is run here (that's Step 4).

### Model panel (`lung_training.py`)
All candidates train on the same 80/10/10 split; the **best by internal-test ROC AUC** is saved as
`model_<window>_1to<NC_RATIO>.joblib`. Every learner is **cost-sensitive for the minority (cancer)
class** so imbalance is penalised directly (no SMOTE / resampling — `handle_imbalance("none")`):

| Learner | Imbalance handling |
|---|---|
| CatBoost | `auto_class_weights='Balanced'` |
| LightGBM | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight = n_neg / n_pos` |
| Random Forest | `class_weight='balanced'` |
| Extra Trees | `class_weight='balanced'` |
| Gradient Boosting | `sample_weight='balanced'` (no `class_weight` param, but `.fit` takes weights) |
| AdaBoost | `sample_weight='balanced'` |

`class_weight`/`sample_weight='balanced'` auto-scale with `NC_RATIO`, so the minority penalty grows
with the ratio. Because each learner rebalances internally, **raw probabilities sit at a ~balanced
base rate** — they are recalibrated to the true prevalence by Platt scaling in Step 4.

> **Dropped learners:** KNN + MLP (support neither `class_weight` nor `sample_weight`, so can't be
> cost-weighted), Naive Bayes (far weakest, ~0.75 AUROC — its feature-independence assumption is
> broken by correlated clinical features), Logistic Regression (~0.90, weakest of the weighted set).

---

## Step 4 — Held-out evaluation + Platt recalibration  (`4_Holdout/`)
Touch-once eval of a trained Step 3 model on the real **500 cancer / 50,000 non-cancer** held-out
(0% train overlap), then **Platt scaling** to calibrate probabilities (the documented approach).
Per-patient FE + the **TRAIN** xpoll reference (leakage-safe), scored with the train-fitted transforms.
Unlike Step 3, this runs **ONE window per invocation** (no loop).

```bash
GAP=12 NC_RATIO=1 WINDOW=5yr python 4_Holdout/evaluate_heldout.py   # 12mo, 5yr, 1:1 model
GAP=1  NC_RATIO=1 WINDOW=5yr python 4_Holdout/evaluate_heldout.py   # 1mo,  5yr, 1:1 model
GAP=12 NC_RATIO=5 WINDOW=5yr python 4_Holdout/evaluate_heldout.py   # 12mo, 5yr, 1:5 model
# WINDOW = 5yr | 10yr | 20yr | lifetime   |   NC_RATIO = 1 | 5
```
- The held-out cohort (`SQL/heldout_test_{GAP}mo.sql`) is **fixed regardless of `NC_RATIO`** —
  that only selects which trained model to score. Requires `model_{WINDOW}_1to{NC_RATIO}.joblib`
  + `xpoll_ref_{WINDOW}.json` from the Step 3 run.
- Stratified **30% calib / 70% test** split; fits Platt (sigmoid) + isotonic (cross-check) on calib.
- Reports on the 70% test slice: **AUROC / AUPRC** (threshold-free), **Brier + ECE** (raw vs Platt vs
  isotonic), **Sens/Spec/PPV/NPV at free-Youden**. Saves `platt_calib_{window}.joblib`,
  `heldout_recalib_{window}.txt`, `heldout_reliability_{window}.png` into the window's output dir.

> King-Zeng true-prevalence offset is a separate *future deployment* step (`deploy_calibration.py`) —
> **not** applied here; we calibrate to the held-out's own prevalence for now.

---

## What to do with the results
- **Step 1** → the ranked code lists (`lung_combined_all.csv`) for clinical review / feature selection.
- **Step 3** → pick the best lookback from `lookback_internal_summary.csv`.
- **Step 4** → the held-out metrics (`heldout_recalib_{window}.txt`) + Platt calibrator
  (`platt_calib_{window}.joblib`) — the realistic, calibrated evaluation of the chosen model.

---

## Pipeline at a glance (full flow)

```
STEP 1 — code scoring   (independent track; run inside 1_Top_Snomed/, in order; each does BOTH 12mo + 1mo)
  BigQuery (SQL/12mo_1to1.sql + SQL/1mo_1to1.sql)
     │  python build_score_counts.py
     ▼
  Data/{12mo,1mo}/  Lung_{pos,neg}_{obs,meds}.csv  +  Lung_patient_{obs,meds}_matrix.csv
     │  python "Combined Scoring (ML+Stat).py"
     ▼
  output/{12mo,1mo}/Scores_lung/lung_combined_all.csv          ← ranked codes (deliverable)

STEPS 2 + 3 — feature engineering + train/internal   (one command from the folder root)
  BigQuery (2_FE/SQL/{GAP}mo_1to{NC_RATIO}.sql)
     │  GAP=12 NC_RATIO=1 python run_lookback_experiment.py
     ▼  Step 2 (2_FE): events → feature matrix   →   Step 3 (3_Modeling): train 80/10/10 (free/Youden)
  {GAP}mo_1to{NC_RATIO}/lookback/{5yr,10yr,20yr,lifetime}/   (model_*.joblib + metrics.csv)
     │
     ▼
  {GAP}mo_1to{NC_RATIO}/lookback/lookback_internal_summary.csv  ← compare windows, pick best

STEP 4 — held-out eval + Platt recalibration   (one window per run, after a Step 3 model exists)
  BigQuery (4_Holdout/SQL/heldout_test_{GAP}mo.sql)   [fixed 500/50k cohort]
     │  GAP=12 NC_RATIO=1 WINDOW=5yr python 4_Holdout/evaluate_heldout.py
     ▼  FE (TRAIN xpoll) → score chosen model → 30/70 calib/test → Platt
  {GAP}mo_1to{NC_RATIO}/lookback/{WINDOW}/  heldout_recalib_{window}.txt + platt_calib_{window}.joblib
```

Order: **Step 1** is independent; **Steps 2 + 3** run together via `run_lookback_experiment.py`;
**Step 4** follows once a model exists. Inputs come from BigQuery; all outputs stay in this folder.
