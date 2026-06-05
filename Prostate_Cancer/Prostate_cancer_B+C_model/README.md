<!-- ════════════════════════════════════════════════════════════════════ -->
<!-- PROSTATE — B+C MATURED MODEL  (generated 2026-06-04)            -->
<!-- ════════════════════════════════════════════════════════════════════ -->

# Prostate — B+C Matured Model

Copy of `Prostate_2.0_1to1` (originals untouched) with the **matured B+C cohort/split
design** finalized on breast and rolled out across all cancers. Source: **EMIS**.
Sex: **male-only**. Dual/tri-horizon windows: **12mo + 1mo_12mo + 1mo_5y**.

## What's matured here (vs the parent `Prostate_2.0_1to1`)
- **Gold-layer non-cancer anchor** — each non-cancer patient is anchored on a RANDOM
  date drawn from the **cancer diagnosis-date distribution** (`cancer_anchor_pool` +
  `MOD(ABS(FARM_FINGERPRINT(patient_guid)), N)+1`), so non-cancer and cancer share the
  same anchor-**year** distribution by construction (kills the recency confound).
  Replaces the old own-event-date anchor. Aligned to `Truveta/Git_SQL/cancer_training_truveta_v3_gold.sql`.
- **`non_cancer_ratio = 5`** — oversample non-cancer 5× as a buffer.
- **Per-cohort `min_obs`** — cancer = 1 (keep every positive), non-cancer = 5.
- **Balance-to-1:1 across train / val / test** — the majority class is downsampled to
  the minority count **before** the stratified split, so all three splits are ~1:1 and
  positive cases are never discarded (the 5× buffer absorbs the trim).
- **Balance upgrade**: previously balanced TRAIN only (`BALANCE_TRAIN`); now `build_shared_split` downsamples to 1:1 BEFORE the split so **val + test are also 1:1** (`BALANCE_TRAIN` default → False).
- Trend / worsening (`SEQ_*`/`ACCEL_*`) and Approach B (`__PLACEHOLDER__` rows) are **inherited** from the EMIS template — no change needed.
- Random stratified split (no temporal holdout).

## Re-run (SQL changed → full chain)
1. Run `SQL Queries/v4/*.sql` in BigQuery → cohort tables
2. `2_Feature_Engineering` preprocess + FE
3. `3_Modeling` (watch the split log: if it reports *"dropped surplus cancer"*, the 5× buffer was too small)

---

# Prostate B + C — Dual-Horizon Production Pipeline

Production-ready prostate cancer prediction system with **two horizons** running in parallel:

- **Model B (12mo)** — Early prediction. Trained on data ending 12 months before diagnosis. Catches long-term-risk patients before they present acute symptoms.
- **Model C (1mo)** — Acute alert. Trained on data ending 1 month before diagnosis. Catches "in danger now" patients with recent PSA elevation, urinary symptoms, etc.

The two models run in parallel per patient, OR-combined: if **either** fires → overall result = `Cancer`.

## What's different from `Prostate_2.0_1to1` and `Prostate_Prod_Ready`

| | `Prostate_2.0_1to1` | `Prostate_Prod_Ready` | **`Prostate_B+C` (this folder)** |
|---|---|---|---|
| Purpose | Research / experiments | Production architecture template | **Production-ready dual-horizon deployment** |
| Output format | Flat parquet | JSON-packed parquet | **JSON-packed parquet (native BQ JSON)** |
| Schema sidecar | none | yes | **yes (with all dtypes incl. category)** |
| Windows trained | 4 (1mo, 3mo, 6mo, 12mo) | 1 at a time | **3 (12mo, 1mo_5y, 1mo_12mo)** |
| Cohort | per-window (different patient sets) | per-window | **unified — same patients for all 3 models** |
| Train/val/test split | per-window stratified | per-window | **shared via `shared_split.json`** |
| Min events filter | ≥10 | ≥10 | **≥5 (more permissive — keeps more patients)** |
| Extract cutoff | 2026-04-25 | 2026-01-01 | **2026-05-21** |
| Production lookback experiment | n/a | n/a | **1mo_5y vs 1mo_12mo head-to-head** |

## Folder structure

```
Prostate_B+C/
├── 1_SQL/
│   ├── 12mo.sql           — extracts events ending dx-12mo, 5y lookback
│   ├── 1mo_5y.sql         — extracts events ending dx-1mo,  5y lookback (Model C baseline)
│   └── 1mo_12mo.sql       — extracts events ending dx-1mo,  12mo lookback (experiment)
├── 2_Feature_Engineering/
│   ├── config.py          — 3-window config, ≥5 events filter, shared_split path
│   ├── 0_preprocess.py    — codelist → category mapping
│   ├── 1_pipeline_blocks.py
│   ├── 2_prostate_features.py
│   ├── 3_cleanup.py
│   ├── 4_transform_features_json.py  — packs features into transformed_features JSON
│   ├── codelists/
│   └── data/              — per-window FE inputs/outputs
├── 3_Modeling/
│   ├── _shared_split.py   — builds unified cohort + writes shared_split.json
│   ├── 1_training.py      — trains all 3 models on the shared split
│   ├── 2_predict_unseen.py — batch inference
│   ├── _load_features.py  — loader (handles JSON+sidecar)
│   ├── _calibrator.py     — per-band Platt/isotonic
│   └── shared_split.json  — UNIFIED patient → train/val/test (written ONCE)
├── 4_Inference/
│   ├── predict_dual_horizon.py — single-patient API, OR-combined, threaded
│   └── predict_single_patient.py — single-window scorer
├── 6_Explainability/
│   ├── explain_single_patient.py — per-patient SHAP
│   └── 1_explain_predictions.py  — batch SHAP across windows
└── 7_Holdout_Test/
    └── test_dual_horizon_compare.py — compares (12mo + 1mo_5y) vs (12mo + 1mo_12mo)
```

## Pipeline run order

```bash
cd Prostate_B+C

# ─── 1. SQL extracts (BQ) ───
# Three parallel BQ extracts. Output goes to:
#   2_Feature_Engineering/data/raw/prostate_{12mo,1mo_5y,1mo_12mo}_raw.csv
bash run_extract.sh

# ─── 2. FE pipeline per window (3 × ~30 min) ───
cd 2_Feature_Engineering
for window in 12mo 1mo_5y 1mo_12mo; do
    WINDOW=$window python3 0_preprocess.py
    WINDOW=$window python3 3_cleanup.py
    WINDOW=$window python3 4_transform_features_json.py  # JSON-pack + sidecar
done

# ─── 3. Build shared split ONCE on unified cohort ───
cd ../3_Modeling
python3 _shared_split.py
# → writes shared_split.json — DO NOT regenerate after training starts

# ─── 4. Train all 3 models on the shared split (3 × ~1h) ───
python3 1_training.py
# → saved_models/seed_42/ per window, final_results.csv per window

# ─── 5. Compare dual-horizon setups ───
cd ../7_Holdout_Test
python3 test_dual_horizon_compare.py
# → console: Setup A (12mo + 1mo_5y) vs Setup B (12mo + 1mo_12mo)
# → test_compare_per_patient.csv with per-patient detail
# → tells you which 1mo variant wins
```

## Production deployment

After step 5, the winning 1mo variant is chosen. Point `predict_dual_horizon.py` to it:

```python
# In your serving code:
from predict_dual_horizon import set_oneme_variant, predict_dual_horizon_from_file
set_oneme_variant("1mo_5y")   # or "1mo_12mo" — whichever won the compare step

result = predict_dual_horizon_from_file(
    "patient_features.parquet",
    schema_path="patient_features_schema.json",
)
# {"result": "Cancer", "by_model": {...}, "fired_by": ["1mo_5y"]}
```

## Output schema (per inference call)

```json
{
  "result": "Cancer" | "No Cancer" | "Not enough data",
  "by_model": {
    "1mo_5y":  { "verdict": "...", "risk_proba": 0.91, "decision": 1,
                 "threshold_used": 0.52, "age_band": "65-75",
                 "insufficient_data": false,
                 "top_factors": [{"feature": "LAB_PSA_max", "value": 12.3, "shap": +0.41}, ...] },
    "12mo":    { "verdict": "...", "risk_proba": 0.32, ... "top_factors": [...] }
  },
  "fired_by": ["1mo_5y"]
}
```

## BQ table schema (production)

```
Prostate_B+C deployment table:
  patient_guid           STRING
  sex                    STRING
  patient_ethnicity_16   STRING
  patient_ethnicity_6    STRING
  patient_age            INTEGER
  max_event_date         DATE         ← inference data-freshness marker
  cancer_class           INTEGER      ← null at inference time
  cancer_id              STRING       ← "prostateCancer" / null
  partition_date         DATE         ← daily partition for serving
  transformed_features   JSON         ← native JSON type (NOT string)
```

Schema sidecar (`<table>_schema.json`) emitted alongside parquet exports for downstream typed loading.

## Notes on `shared_split.json`

The shared split is a **hard contract** — write it ONCE before training, never regenerate afterwards. Regenerating invalidates already-trained models.

Format:
```json
{
  "train_guids":   ["{GUID-1}", "{GUID-2}", ...],
  "val_guids":     ["{GUID-101}", ...],
  "test_guids":    ["{GUID-201}", ...],
  "seed": 42,
  "stratify": "cancer_class",
  "ratios": {"train": 0.75, "val": 0.10, "test": 0.15},
  "n_total": 12345, "n_train": 9258, "n_val": 1235, "n_test": 1852,
  "n_cancer": 6173, "n_non_cancer": 6172
}
```

Each training run filters its own feature matrix by these `patient_guid` lists. Patient X ends up in the same split for all 3 models — guaranteed.

## What to do next

After the pipeline runs and you've picked a winning 1mo variant:
1. Pull a **production-prevalence** (~1:1000) sample with the same FE pipeline
2. Re-calibrate the winning models' per-band thresholds on this imbalanced data
3. Document the production thresholds
4. Wire `predict_dual_horizon` into the serving API
5. Set up monitoring on the BQ table (drift, prediction calibration)
