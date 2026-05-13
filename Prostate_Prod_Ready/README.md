# Prostate_Prod_Ready

Production-ready prostate cancer pipeline aligned with the lung team's
`SNOMED_json_v3` interface (JSON-packed feature output → BigQuery
feature store → live API consumption).

## Two-pipeline split — what to use when

There are intentionally two prostate pipelines in this repo. They serve
different roles and produce slightly different feature values:

| Use case | Use this | Why |
|---|---|---|
| **Training / iterating on features / explainability** | `Prostate_2.0_1to1/` | Validated Run 2 (1mo AUC 0.9821). SQL emits `ANCHOR_DATE` + year-stratified sampling. Has holdout test, inference, explainability scripts. |
| **Production deployment / live API / cross-team BQ feature store** | `Prostate_Prod_Ready/` (this dir) | Lung-style SQL (no `anchor_date` column, random sampling). FE derives anchor as `MAX(event_date)` per patient — **matches real-world inference exactly** (no diagnosis date is known at deploy time). JSON-packed BQ output. |

**Important — these pipelines are not interchangeable at the feature-value level.**

Because the anchor is derived differently:
- `2.0_1to1` cancer cohort: anchor = `date_of_diagnosis` (exact)
- `Prod_Ready` cancer cohort: anchor = `MAX(event_date)` (≈ `date_of_diagnosis - months_before`)

Cancer-cohort features computed in the two pipelines will differ slightly
(off by ≤ 1 month of anchor shift). **A model trained on `2.0_1to1` should
not be used directly with `Prod_Ready` data without retraining**, and vice
versa. The intended deployment flow is:

```
Train on 2.0_1to1 (precise anchor) → freeze model artefacts
                                  ↓
                  Deploy via Prod_Ready (lung-style anchor)
                  → retrain on Prod_Ready data before going live
```

Or equivalently: train + deploy entirely within `Prod_Ready` if you want
training and inference to use the identical anchor logic. Either is valid;
pick one and stick with it.

## Why lung-style anchor in Prod_Ready?

At inference time on a new patient, **there is no `date_of_diagnosis`** (the
whole point is that we're predicting whether they'll be diagnosed). The
only sensible anchor is "their latest available data point" = `MAX(event_date)`.

By using the same anchor logic at training (i.e. mirroring inference), the
model never sees a feature distribution it wouldn't also see in production.
This is the lung team's design choice, and we've adopted it for the same
reason in `Prod_Ready`.

The SQL still uses `date_of_diagnosis` internally for the cancer cohort —
but only to filter events to the pre-diagnosis window. The anchor value
itself isn't emitted; the FE derives it from `MAX(event_date)`.

## Layout (step-numbered; runs in numeric order within each dir)

```
Prostate_Prod_Ready/
├── 1_SQL/
│   └── unified_prostate_extraction.sql       # lung SQL + 2 prostate-specific changes
│
├── 2_Feature_Engineering/                    # FE pipeline (run via 4_transform_features_json.py)
│   ├── 0_preprocess.py                       # categorize + TIME_WINDOW + defensive anchor
│   ├── 1_pipeline_blocks.py                  # FE blocks 4a–4g (clinical / med / trend / accel)
│   ├── 2_prostate_features.py                # prostate-specific blocks (PSA emphasis)
│   ├── 3_cleanup.py                          # non-numeric drop, dedup, low-variance, leakage
│   ├── 4_transform_features_json.py          # orchestrator: BQ-in → JSON pack → BQ-out + schema sidecar
│   ├── config.py                             # helper: LAB_CATEGORIES, LAB_BAD_DIRECTION, ranges, etc.
│   ├── io_utils.py                           # helper: read_table / write_table
│   ├── SCHEMA_CONTRACT.md                    # output schema documentation
│   └── codelists/
│       ├── prostate_curated_codes_v2.tsv
│       └── code_category_mapping_v2.json
│
├── 3_Modeling/                               # model training + holdout evaluation
│   ├── 1_training.py                         # train stacked ensemble (XGB+LGB+CB+LR), saves to results/
│   ├── 2_predict_unseen.py                   # evaluate on holdout; loads encoder mappings sidecar
│   ├── _calibrator.py                        # helper
│   └── _load_features.py                     # helper: format-agnostic feature loader
│
├── 4_Inference/                              # live single-patient prediction
│   └── predict_single_patient.py             # entry point for API / on-demand scoring
│
├── 5_Explainability/                         # post-hoc explanation (SHAP)
│   └── 1_explain_predictions.py              # writes per-band / per-patient / summary SHAP outputs
│
└── README.md  (this file)
```

**Execution order** within `2_Feature_Engineering/` when `4_transform_features_json.py` runs:
`0_preprocess` → `1_pipeline_blocks` → `2_prostate_features` → `3_cleanup` → JSON-pack → BQ write + schema sidecar.

**Removed (vs. earlier scaffold):**
- `symptom_occurrence_analysis.py` and `value_trend_analysis.py` — were copied from lung but never called in our generic-SNOMED FE; the value-trend logic in `1_pipeline_blocks.py` (driven by `LAB_CATEGORIES` in `config.py`) covers the same ground for our use case.

## SQL — what was changed from lung base

Base: `lung/sql/unified_cancer_noncancer34_v2.sql`

Modifications (only 2):

1. **`target_cancer_pattern`** `'%lung%'` → `'%prostate%'`
2. **Sex filter `pp.sex = 'M'`** added to BOTH cohorts (prostate is male-only).
   Implementation:
     - `target_cancer_patients` CTE: `JOIN patients pp` + `AND pp.sex = 'M'`
     - `non_cancer_patients` CTE: `JOIN patients pp` + `WHERE pp.sex = 'M'`
   Both marked inline with `-- [PROSTATE]` comments.

**Deliberately kept from lung as-is:**
- No anchor metadata in output (anchor is derived in `preprocess._preprocess_inmem`)
- No year-stratified sampling (lung uses random — see flag below)
- Two-tier ethnicity via `LEFT JOIN Patient_Ethnicity` (LABEL_16, LABEL_6)
- 14-SNOMED admin exclusion list
- `min_obs_events_per_patient = 50`, `min_med_events_per_patient = 10`

## FE — what was preserved from our pipeline

Everything that gives prostate its lift over a generic lung-pattern run:
- All ~1300 prostate features (PSA trend, vitals, comorbidities, symptoms,
  interactions, acceleration, etc.) — ported intact via `pipeline_blocks.py`
  + `prostate_features.py`
- Vital-sign trend categories we just split out (`BMI`, `BODY_WEIGHT`,
  `SYSTOLIC_BP`, `DIASTOLIC_BP`, `HEART_RATE`) — present in `config.py`
- LAB_BAD_DIRECTION + LAB_WORSENING_RULES — preserved
- LAB_RANGES plausibility bounds — preserved
- Defensive anchor fallback (derive `ANCHOR_DATE` from `MAX(EVENT_DATE)` per
  patient if upstream SQL doesn't emit it) — required because the lung
  base SQL omits anchor metadata
- Cleanup step (non-numeric drop, dedup, low-variance filter, leakage check)

## Modeling — what was preserved from our pipeline

- Stacked ensemble (XGBoost + LightGBM + CatBoost + LR meta-learner)
- Per-window models (the directory ships as 1mo by default; pass
  `--window 3mo` / `6mo` / `12mo` for the others)
- Calibration helper (`_calibrator.py`)
- Feature loader (`_load_features.py`) — already JSON-aware via the
  prostate JSON-pilot work; can read either the new packed format or
  the legacy flat parquet

## ⚠ Flags — review BEFORE training/deploying

### 1. Year-stratified non-cancer sampling
Lung's SQL does **random** non-cancer sampling. Our Prostate v4 SQL was
year-stratified. The prod-ready SQL here follows lung (random).

**Risk:** without year-stratification the model can learn anchor-year as a
class signal (cancer anchors cluster in recent years → "looks like 2023" =
high risk). The original prostate v4 results were obtained with
year-stratification.

**Decide:** match lung exactly (current state), or re-add year-stratification?

If re-adding, the unified proposal at
`New_Sql_proposal/SQL_Query_Unified.sql` already has the change in a
clearly-marked `CHANGE [+]` block — port over the `cancer_year_counts` +
`non_cancer_ranked` CTEs.

### 2. Anchor metadata not emitted by SQL
Lung's SQL doesn't output `anchor_date`, `anchor_year`,
`days_before_anchor`, `months_before_anchor`. Our preprocess derives them
in-memory from `MAX(event_date)` per patient — fine for INFERENCE but
**for training the cancer cohort's anchor is supposed to be
date_of_diagnosis**, not max event date. The unified SQL handles this
correctly because it builds the cohort with diagnosis-date anchors.

If we train on this prod-ready SQL output as-is, cancer anchors will be
`MAX(event_date)` — events INCLUDING the cancer diagnosis itself —
which is label leakage.

**Decide:** are we training on this prod SQL? If yes, port the anchor logic
back from `Prostate_2.0_1to1/SQL Queries/v4/*.sql` (the v4 SQLs build the
cancer-cohort anchor as `date_of_diagnosis` correctly). The prod SQL is
**INFERENCE-CORRECT, NOT TRAINING-CORRECT** as it currently stands.

The cleanest split:
- Use `Prostate_2.0_1to1/SQL Queries/v4/*.sql` for TRAINING (anchor metadata + year-stratified)
- Use `Prostate_Prod_Ready/1_SQL/unified_prostate_extraction.sql` for nightly INFERENCE only

### 3. Ethnicity as model feature
Lung's training (`3_Modeling/training_v1.py:175,194-200`) keeps ethnicity
as a feature via `LabelEncoder`. Our prostate experiment dropped it for
deployment (per project memory: "Tier 2 ethnicity tested end-to-end,
dropped for deployment").

Current `training_v1.py` in this directory is **our** code (drops
ethnicity). If we want to align with lung's choice, the LabelEncoder loop
in `training_v1.py` needs adjustment — but the prostate experiment
already settled this. **Recommend: keep ours (drop ethnicity).**

### 4. Window selection
Lung trains one model (single window). We train four (1mo / 3mo / 6mo /
12mo). The orchestrator defaults to 1mo for production live API
(most clinically actionable for the "imminent diagnosis" use case). The
other three windows are available via `--window`.

**Decide:** ship all four models, or just 1mo?

### 5. Cleanup re-attaches demographics
`cleanup.py` step 5a drops all non-numeric columns (sex, ethnicity, dates)
— good for protecting the model from accidentally training on these. But
the JSON-packed output needs `sex`, `patient_ethnicity_16/6`,
`patient_age` as **front (flat) columns** to match lung's contract.

`transform_features_json._preprocess_inmem` re-attaches them from the
preprocessed raw frame BEFORE packing. **Sanity-check this re-attach in
the first end-to-end run** — if the join key drift, demographics
columns will be NULL in the BQ table.

### 6. Per-event vs per-patient ID drift
Our preprocess emits per-event rows. The FE blocks groupby `PATIENT_GUID`
to produce per-patient features. After cleanup the result is per-patient.
JSON packing assumes per-patient. **Verify shape after each step in the
first run** — easy to silently end up with per-event JSON-packed rows if
a groupby is skipped.

### 7. BigQuery dataset/table must be pre-created
`transform_features_json.write_to_bigQ` writes to
`cthesigns-platform-475414-b7.prediction_emis.prostate_features_{window}`.

Pre-create:
- Dataset: `prediction_emis` (US multi-region, to match
  `EMIS_BULK_DATA_PROCESSED`)
- Table partition expiration: 30 days (or per audit policy)

## How to run

### Training (full pipeline, manual / quarterly)
```
cd Prostate_Prod_Ready/v2-ensemble
python training_v1.py --window 1mo   # repeat for 3mo / 6mo / 12mo
```
(Inputs: a fully-trained extraction WITH proper anchor — see flag #2.
Today, point this at `Prostate_2.0_1to1/2_Feature_Engineering/results/`.)

### Nightly feature refresh (cron / scheduled job)
```
cd Prostate_Prod_Ready/SNOMED_json_v3
python transform_features_json.py --window 1mo
# repeat per window OR loop in the cron wrapper
```
Writes to BQ table `prediction_emis.prostate_features_{window}` with
today's `partition_date`.

### Live inference (API server)
```
cd Prostate_Prod_Ready/v2-ensemble
python predict_unseen_input.py --patient_guid <guid> --window 1mo
```
Loads the latest BQ partition for that patient, applies the trained
model, returns risk score.

## Migration checklist

- [ ] Pre-create BQ dataset `prediction_emis` (US region)
- [ ] Set 30-day partition expiration on `prostate_features_{window}` tables
- [ ] Address flag #2 (training-vs-inference anchor split)
- [ ] Decide on flag #1 (year-stratified vs random sampling)
- [ ] Run one end-to-end test on 1mo with `--no-bq` to validate locally
- [ ] Verify cleanup → pack join (flag #5) preserves all rows
- [ ] Confirm prediction latency on a single-patient lookup (target <100ms)
- [ ] Set up nightly cron (Cloud Scheduler → Cloud Run job) calling
      `transform_features_json.py` once per window
