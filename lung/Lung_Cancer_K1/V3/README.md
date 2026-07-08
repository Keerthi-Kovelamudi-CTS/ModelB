# Lung Cancer Prediction Pipeline

A reproducible, leakage-free pipeline that predicts lung cancer from primary-care medical records
(EMIS observations + prescriptions) at two horizons — **12 months** and **1 month** before diagnosis.

It pairs a **curated clinical-concept codelist** with a **focused data-driven code discovery** step,
engineers rich per-patient temporal features, selects the codes that consistently drive predictions,
and trains a calibrated cost-weighted ensemble. A separate touch-once labelled cohort gives an
unbiased real-world estimate of deployment performance.

---

## 1. Design principles

- **Split-first.** One canonical, patient-keyed `train` / `valid` / `test` split is created **once**
  per horizon and saved (to GCS). Every stage loads that single file, so the same patients are in the
  same split everywhere — code scoring, feature engineering, stability selection, and model training.
  The internal **test** split is never seen by any fitting step.
- **Leakage discipline.** Code scoring, value-feature schema, and stability selection are fit on
  **train only**. Model/ensemble/hyperparameter and operating-threshold choices are made on
  **validation**. The internal **test** is scored exactly once, at final evaluation.
- **Honest held-out reporting.** On the labelled held-out cohort, the recalibrator and operating
  threshold are fit on a disjoint **30%** slice and everything is reported on the unseen **70%** —
  so no metric is reported on a patient used to fit the calibrator or pick the threshold.
- **Reproducible & configurable.** All knobs live in `.env` / `config.py`; real env vars override.
  Deterministic seeds throughout.

---

## 2. Pipeline at a glance

```
0_SQL/                     shared cohort SQL (one source of truth per horizon)
        │
        ▼
make_split.py              Step 0 — cohort membership -> ONE canonical split per horizon (-> SPLIT_DIR)
        │
1_Top_Snomed/              Phase 1 — code scoring + codelist construction
   build_score_counts.py     per-code counts + patient matrices  (TRAIN-only)
   Combined Scoring (ML+Stat).py   statistical + ML importance -> combined rank
   build_codelist.py         curated  ∪  top-N data-driven  ->  2_FE/codelist/lung_codelist_{h}.csv
        │
2_FE/                      Phase 2 — feature engineering + selection
   build_features.py         tiered FE -> features_p005_{h}.parquet  (wide per-code matrix)
   stability_select.py       k-fold cumimp99 -> features_p005_{h}_stable.parquet (+ stamps the split)
        │
3_Modeling/                Phase 3 — training
   lung_training.py          7-model cost-weighted ensemble, isotonic calibration -> model_{h}.joblib
        │
4_Heldout/                 Phase 4 — touch-once labelled evaluation
   evaluate_heldout.py       30/70 calib/test -> AUROC/AUPRC + Sens/Spec/PPV/NPV + deployable Platt
```

---

## 3. The codelist (curated ∪ data-driven)

The codelist is the union of two sources, built per **horizon × lookback window** by
`1_Top_Snomed/build_codelist.py` (each lookback re-discovers its data-driven codes on its own train data):

- **Curated** — `1_Top_Snomed/curated_codes.json`, ~217 hand-picked clinical-concept codes
  (symptoms, comorbidities, smoking, relevant meds/labs). Force-included even if sub-threshold. This
  list is **horizon-independent** (one file), since the relevant clinical concepts don't change with
  the prediction horizon. **Toggle with `USE_CURATED`** — off → codelist is the data-driven top-N only
  and every code gets generic per-code FE (no concept tier).
- **Data-driven** — up to `TOP_N` (default **500**) codes that pass three gates on the train split:
  **Bonferroni p<0.05 AND odds-ratio ≥ `OR_MIN` (2.0) AND prevalence ≥ `PREV_MIN` (1%)** of cancer
  patients, ranked by `combined_rank` (statistical + ML importance).

`TOP_N` caps the number of **codes in the codelist** — not model features. Features come later
(Section 4).

---

## 4. Feature engineering (tiered)

`2_FE/build_features.py` engineers features **per patient**, so nothing crosses between patients
(no cross-patient leakage). It is **tiered** — the two code sets are disjoint:

- **Curated codes → concept-level FE** (`concept_features.py`): symptom dynamics, smoking dose,
  problem-list flags, lab level/trend stats, comorbidity/symptom clusters, interaction terms.
- **Data-driven codes → generic per-code FE**: occurrence/dynamics (count, recency, decay,
  acceleration, frequency trend, worsening), problem-list flags, age-at-event, **value/trend**
  (mean/latest/min/max/std/range, abs & % change, within-patient z, slope, correlation, acceleration),
  per-time-band, and cumulative last-N windows. **Value features fire for every per-code code that
  carries any numeric value** (determined train-only); codes with no numeric value simply emit none.

Plus once-per-patient blocks over the full event stream: global volume + demographics, problem-comment
keywords, generic lab-derangement burden, and blood ratios (NLR / PLR / LMR / CRP-albumin).

This produces a **very wide** matrix (~28 features × thousands of codes). The model never trains on it
directly — `stability_select.py` runs k-fold and keeps only features in the cumulative-99%-importance
set in ≥ `MIN_FOLDS` of `N_FOLDS` folds (train-only), writing the compact matrix the model trains on
and **stamping the canonical `split` column** so the split travels with the data.

FE backend is selectable with `FE_ENGINE` (`pandas`, the default; or `polars`). The Polars engine is
gated on `2_FE/fe_parity_check.py` — use it only once that parity check passes.

---

## 5. Modeling

`3_Modeling/lung_training.py` trains seven cost-weighted models (RandomForest, ExtraTrees,
GradientBoosting, AdaBoost, XGBoost, LightGBM, CatBoost), **selects on validation**, optionally tunes
with Optuna (`TUNE=1`), ensembles the top models, and calibrates probabilities with per-age-band
isotonic regression. The operating threshold (Youden) is chosen on validation and applied once to the
internal test at final evaluation.

---

## 6. Held-out evaluation

`4_Heldout/evaluate_heldout.py` runs the trained model on a labelled held-out cohort (~500 cancer /
50k non-cancer, excluded from training at source). It splits the held-out **30% calib / 70% test**
(stratified, fixed seed), fits the Platt recalibrator and picks the threshold on the **calib** slice,
and reports AUROC/AUPRC + Sens/Spec/PPV/NPV + Brier/ECE on the disjoint **test** slice. The calib-fit
Platt (`platt_calib_{h}.joblib`) is the deployable recalibration artifact.

---

## 7. Running it

Prerequisites: a machine with BigQuery access and the env in `requirements.txt`
(`pip install -r requirements.txt`), plus a populated `.env` (see `config.py` for keys).

```bash
python run_v3.py                       # both horizons (12mo then 1mo) × all FE_WINDOWS, GCS caching on
python run_v3.py 12mo --windows 5      # one horizon, single 5-year lookback
python run_v3.py --heldout             # full sweep, then held-out evaluation
python run_v3.py --force                # recompute every stage (ignore cache)
python run_v3.py --no-gcs               # local only (no GCS read/write)
python run_v3.py --engine polars        # use the Polars FE (after fe_parity_check.py passes)
```

`run_v3.py` runs `split → codelist → fe → stable → train` (and optional `--heldout`) for each
**horizon × lookback window**. By default it loops both horizons and every window in `FE_WINDOWS`
(5/10/20/100 yr). The split is built once per horizon (membership is lookback-independent) and shared
across windows; the **codelist is re-discovered per lookback** on that lookback's own train data — no
codelist is reused across windows.

**Each stage is cache-skipped** if its output already exists locally or in GCS — meaning the runner
*reuses* that artifact (pulling it from GCS to local if needed) instead of recomputing it. Because every
artifact is keyed to its exact `(horizon, window)`, this never crosses a window's data with another's.
Use `--force` to recompute from scratch. Stages can also be run individually from their own directories.

---

## 8. Storage & artifacts

The canonical split lives in `SPLIT_DIR` (a `gs://` path). All other artifacts are written under
`V3/` and mirrored to `GCS_ROOT/<relative path>` so reruns and teammates reuse them:

The canonical split is the only per-horizon artifact shared across lookbacks; everything else
(including the codelist) is per-`(horizon, window)`.

| Artifact | Path |
|---|---|
| Canonical split (per horizon) | `SPLIT_DIR/lung_{h}_split.parquet` |
| Codelist (per horizon × window) | `2_FE/codelist/lung_codelist_{h}_{yr}yr.csv` |
| Wide feature matrix | `2_FE/output/{h}/{yr}yr/features_p005_{h}.parquet` |
| Stable matrix (+ split column) | `2_FE/output/{h}/{yr}yr/features_p005_{h}_stable.parquet` |
| Stable feature list | `2_FE/output/{h}/{yr}yr/stable_features_{h}.csv` |
| Trained model | `output/{h}/{yr}yr/model_{h}.joblib` |
| Results plot | `output/{h}/{yr}yr/results_{h}.png` |
| Held-out report + Platt | `output/{h}/{yr}yr/heldout_recalib_{h}.txt`, `platt_calib_{h}.joblib` |

---

## 9. Configuration

All settings load from `.env` via `config.py` (real environment variables win, so any value can be
overridden per run). Key knobs:

| Key | Default | Meaning |
|---|---|---|
| `HORIZONS` | `12mo,1mo` | prediction horizons to build |
| `TOP_N` / `OR_MIN` / `PREV_MIN` | `500` / `2.0` / `0.01` | data-driven codelist gates |
| `USE_CURATED` | `1` | force-include curated codes + run concept FE (0 = data-driven top-N only) |
| `MIN_VALUE_FRAC` | `0.30` | strict value-bearing threshold (lab-derangement family) |
| `N_FOLDS` / `MIN_FOLDS` / `CUM_IMP` | `5` / `3` / `0.99` | stability selection |
| `TEST_SIZE` / `CALIB_SIZE` | `0.10` / `0.10` | canonical split fractions (→ 80/10/10) |
| `TUNE` / `TUNE_TOP_N` | `0` / `5` | Optuna tuning (off by default) |
| `FE_ENGINE` | `pandas` | FE backend (`pandas` \| `polars`) |
| `SPLIT_DIR` / `GCS_ROOT` | `gs://…/Lung_Cancer/{splits,}` | split + artifact storage |

Per-step details are in `1_Top_Snomed/README.md`, `2_FE/README.md`, and `4_Heldout/README.md`.
