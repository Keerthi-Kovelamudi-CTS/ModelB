# Lung Cancer Prediction Pipeline — **CATEGORIZED variant**

A reproducible, leakage-free pipeline that predicts lung cancer from primary-care medical records
(EMIS observations + prescriptions) at two horizons — **12 months** and **1 month** before diagnosis.

> **This is the `V3_categorized` variant — STRICT category-only FE.** Every feature derives from the
> user's hand-curated SNOMED→category map (per `(horizon, lookback)` in
> `2_FE/categorized_codelist/{Astra|Nova}_{yr}yr_categories.csv`, cols `Code, Name, Category`) **plus patient
> demographics** — nothing else. Events are grouped by **category** (not per-SNOMED) and emit `<category>_<family>`:
> - **Per-category temporal**: count, present, recency, decay, accel, recent_ratio, freq, timespan, intervals,
>   freq-trend, halves, time-bands, cumulative-last-N, age-at-event, problem flags.
> - **Category value pooling (unit-robust)**: each SNOMED's values are z-scored vs its own **TRAIN** distribution
>   before pooling, so mixed-unit categories are valid; `<cat>_val_mean/latest/trend/…` + `<cat>_n_extreme`,
>   `<cat>_max_abs_z` (within-category derangement).
> - **Within-category structure**: `<cat>_n_distinct_codes`, `<cat>_distinct_ratio`, `<cat>_count_share`
>   (utilisation-normalised using map events only), `<cat>_first_months` (tenure).
> - **Cross / inter-category**: `cooc_`, `coocn_` (co-occurrence presence+strength) and `seq_..before..`
>   ordering over the top-`CROSS_TOP_K` categories; `INT_<cat>_x_age`; recency-rank; `g_last_category_months`,
>   `g_category_span_months`, `g_n_categories(_6mo/_12mo)`, `g_category_entropy/gini`.
> - **Demographics only**: `age_at_prediction` (+ age-bands), `g_is_male`, `g_eth_*`.
>
> **Deliberately EXCLUDED** (not list-based / skew/leak risk): NLR/PLR/LMR/CRP-alb blood-ratios, generic
> lab-derangement, and the full-stream **utilisation-volume** (`g_total_events`, distinct-code counts,
> consult cadence, problem burden) + **comment** features. There is **no** curated clinical-concept tier.
> Phase-1 per-code scoring is **skipped** (FE reads the hand-made map directly).
>
> **Config:** `CATEGORIZED=True`, `FE_ENGINE=pandas` (Polars unsupported here), `CROSS_TOP_K`, `LAB_Z_EXTREME`.
> No year is hardcoded — `run_v3.py` auto-discovers windows from the category-map files. GCS: `2_FE` +
> `3_Modeling_outputs` mirror under `GCS_ROOT/V3_categorized`; `splits/` + `raw_events/` are shared (common) at
> `GCS_ROOT`. The per-SNOMED sibling pipeline is `V3_snomed`.
>
> The rest of this README describes the shared pipeline mechanics (cohort, split, modeling, held-out eval),
> which are identical to the per-SNOMED variant except for the strict category-only FE above.

It uses a **focused data-driven code discovery** step, engineers rich per-patient temporal features,
selects the codes that consistently drive predictions, and trains a calibrated cost-weighted ensemble.
A separate touch-once labelled cohort gives an unbiased real-world estimate of deployment performance.

---

## TL;DR (the 30-second version)

- **What it does:** flags patients at risk of lung cancer from their GP record, at two lead times —
  **12 months** before diagnosis (codename **Astra**) and **1 month** before (**Nova**).
- **Data-driven codelist:** the data picks the predictive codes (top-N by combined statistical + ML rank).
- **One command:** `python run_v3.py` builds everything for both horizons; add `--heldout` for the
  real-world test. Results land in `3_Modeling_outputs/{Astra|Nova}/{yr}yr_OFF/`.
- **BigQuery is hit once per horizon.** `events.py` pulls each horizon's full event history a single
  time, caches it, and every lookback window (5/10/20/100 yr) is sliced from that cache locally.
- **Why you can trust the numbers:** the model never sees the internal **test** patients while
  learning, and the headline real-world score is measured on a *separate* cohort it was never trained on.

> **Naming:** **Astra** = 12-month horizon, **Nova** = 1-month horizon · **OFF** = the fixed
> data-driven artifact-namespace tag.

---

## 1. Design principles

- **Split-first.** One canonical, patient-keyed `train` / `valid` / `test` split is created **once**
  per horizon and saved (to GCS). Every stage loads that single file, so the same patients are in the
  same split everywhere — code scoring, feature engineering, stability selection, and model training.
  The internal **test** split is never seen by any fitting step.
- **BigQuery once.** `events.py` is the single BigQuery entry point. It pulls each horizon's **lifetime**
  events one time, caches them (GCS parquet under `raw_events/`), and every lookback window (5/10/20/100 yr)
  is sliced from that one cached frame **locally** — using the exact same calendar rule BigQuery would
  (`event_date ≥ anchor − N years`). So all stages, reruns, and teammates share one
  pull — zero per-window re-querying. (Install `google-cloud-bigquery-storage` for the fast Arrow pull.)
- **Leakage discipline.** Code scoring, value-feature schema, and stability selection are fit on
  **train only**. Model/ensemble/hyperparameter and operating-threshold choices are made on
  **validation**. The internal **test** is scored exactly once, at final evaluation.
- **Honest held-out reporting.** On the labelled held-out cohort, the recalibrator and operating
  threshold are fit on a disjoint **30%** slice and everything is reported on the unseen **70%** —
  so no metric is reported on a patient used to fit the calibrator or pick the threshold.
- **Reproducible & configurable.** All knobs live in `config.py` (the single source of settings);
  real environment variables override per run. Deterministic seeds throughout.
- **Meta-selection discipline.** Each run's internal test is honest, but **choosing among
  horizons / lookbacks / arms by the internal-test number is selection-on-test at the meta level.**
  Arbitrate between configurations using the **validation** split or the **external held-out** —
  never the internal test.

---

## 2. Pipeline at a glance

```
0_SQL/                     shared cohort SQL (one source of truth per horizon)
        │
        ▼
events.py                  THE BigQuery gateway — ONE lifetime pull per horizon -> cache (raw_events/),
                           sliced to each lookback window locally. Every stage below reads this cache.
        │
        ▼
make_split.py              Step 0 — cohort membership -> ONE canonical split per horizon (-> SPLIT_DIR)
        │
1_Top_Snomed/              Phase 1 — code scoring + codelist construction
   build_score_counts.py     per-code counts + patient matrices  (TRAIN-only)
   Combined Scoring (ML+Stat).py   statistical + ML importance -> combined rank
   build_codelist.py         top-N data-driven  ->  2_FE/codelist/{Astra|Nova}_{yr}yr_OFF_codelist.csv
        │
2_FE/                      Phase 2 — feature engineering + selection
   build_features.py         per-code/per-category FE -> features_p005_{h}.parquet  (wide matrix)
   stability_select.py       k-fold cumimp99 -> features_p005_{h}_stable.parquet (+ stamps the split)
   fe_keep_parity.py         PARITY GATE — keep-features build == full build, byte-identical (hard-fail)
        │
3_Modeling/                Phase 3 — training
   lung_training.py          7-model cost-weighted ensemble, per-age-band isotonic calibration -> model_{h}.joblib
        │
4_Heldout/                 Phase 4 — touch-once labelled evaluation + explainability
   evaluate_heldout.py       30/70 calib/test -> AUROC/AUPRC + Sens/Spec/PPV/NPV + Platt + per-age-band thresholds
   subgroup_audit.py         fairness audit: Sens/Spec/PPV by sex × ethnicity × age-band × density
   explainability.py         SHAP: per-patient top factors + per-segment drivers (TP/FP/FN/TN)
```

---

## 3. The codelist (data-driven)

The codelist is built per **horizon × lookback window** by `1_Top_Snomed/build_codelist.py` (each
lookback re-discovers its codes on its own train data):

- **Data-driven** — up to `TOP_N` (default **500**) codes that pass three gates on the train split:
  **Bonferroni p<0.05 AND odds-ratio ≥ `OR_MIN` (2.0) AND prevalence ≥ `PREV_MIN` (1%)** of cancer
  patients, ranked by `combined_rank` (statistical + ML importance). There is no curated tier — every
  code gets the generic per-code (or per-category) FE.

`TOP_N` caps the number of **codes in the codelist** — not model features. Features come later
(Section 4).

---

## 4. Feature engineering

`2_FE/build_features.py` engineers features **per patient**, so nothing crosses between patients
(no cross-patient leakage). Every codelist code (or category, in CATEGORIZED mode) gets the generic
per-code FE — there is no curated/concept tier:

- **Generic per-code FE**: occurrence/dynamics (count, recency, decay,
  acceleration, frequency trend, worsening), problem-list flags, age-at-event, **value/trend**
  (mean/latest/min/max/std/range, abs & % change, within-patient z, slope, correlation, acceleration),
  per-time-band, and cumulative last-N windows. **Value features fire for every code that
  carries any numeric value** (determined train-only); codes with no numeric value simply emit none.

Plus once-per-patient blocks. **NOTE — in this STRICT categorized variant only DEMOGRAPHICS are kept**
(`age_at_prediction` + age-bands, sex, ethnicity); the full-stream **utilisation-volume**, **comment**,
**lab-derangement**, and **blood-ratio** blocks are **OFF** (they use non-map codes / skew risk). The
descriptions below are the per-SNOMED (`V3_snomed`) behaviour where those blocks are on.

This produces a **very wide** matrix (~28 features × thousands of codes). The model never trains on it
directly — `stability_select.py` runs k-fold and keeps only features in the cumulative-99%-importance
set in ≥ `MIN_FOLDS` of `N_FOLDS` folds (train-only), writing the compact matrix the model trains on
and **stamping the canonical `split` column** so the split travels with the data.

FE backend: this **CATEGORIZED variant runs the pandas engine only** (`FE_ENGINE=pandas`, the default
here) — `build_features_polars.py` groups per-code and has no category remap, so `run_v3.py` forces
pandas for categorized regardless of the requested engine. (The per-SNOMED `V3_snomed` sibling defaults
to Polars, verified column-for-column against pandas by `2_FE/fe_parity_check.py`.)

---

## 5. Modeling

`3_Modeling/lung_training.py` trains seven cost-weighted models (RandomForest, ExtraTrees,
GradientBoosting, AdaBoost, XGBoost, LightGBM, CatBoost), **selects on validation by AUPRC**
(`SELECTION_METRIC`, the decision-relevant metric at the deployment prevalence — set to `ROC AUC` to
restore the old behaviour; tuning already optimizes AUPRC, so selection now matches it), optionally tunes
with Optuna (`TUNE=1`), ensembles the top models, and calibrates probabilities with per-age-band
isotonic regression. The operating threshold (Youden) is chosen on validation and applied once to the
internal test at final evaluation. It also writes patient-level + TP/FP/FN/TN segment SHAP
explainability on the internal-test split → `explainability_internal/`. **Synthetic balancing
(SMOTE/ADASYN/under-sampling) is research-only** — `handle_imbalance` hard-fails on those unless
`ALLOW_RESAMPLING=1` (the cohort is already 1:1 with cost-weighting; resampling miscalibrates and must
not reach deployment).

---

## 6. Held-out evaluation

`4_Heldout/evaluate_heldout.py` runs the trained model on a labelled held-out cohort (~500 cancer /
50k non-cancer, excluded from training at source). It splits the held-out **30% calib / 70% test**
(stratified, fixed seed), fits the Platt recalibrator and picks the threshold on the **calib** slice,
and reports AUROC/AUPRC + Sens/Spec/PPV/NPV + Brier/ECE on the disjoint **test** slice. The calib-fit
Platt (`platt_calib_{h}.joblib`) is the deployable recalibration artifact.

It then writes **SHAP explainability** (`3_Modeling/explainability.py`) on the test slice → `explainability/`:
**(1)** per-patient top signed-SHAP risk factors + a global summary, and **(2)** the key risk-factor
drivers of each confusion-matrix segment (TP/FP/FN/TN). SHAP is **tree-exact ONLY, in probability
space**: **tree models (CatBoost/XGB/LightGBM) and soft-voting tree ensembles → exact `TreeExplainer`,
ALL patients**. Members are explained with `model_output="probability"` against a small background
sample, so the voting-weighted member sum exactly reconstructs the ensemble's `predict_proba` (verified
by additivity). **Requires `xgboost<3` (pinned in `requirements.txt`):** shap-0.49's `TreeExplainer`
can't parse xgboost 3.x's vector `base_score` (`'[4.85E-1]'` → `ValueError`); xgboost 2.x serializes it
as a scalar, so exact tree SHAP works directly. (The Booster fallback in `_member_tree_shap` is
best-effort and does **not** rescue the 3.x case — the pin is the fix.)
There is **no `KernelExplainer` fallback** — a non-tree model (or a member with no tree route) **raises a
`RuntimeError`** rather than silently switching to a slow/approximate explainer. `EXPLAIN_MAX` is now a
no-op (tree-exact always explains ALL patients). Feature-label terms aren't applied here (features are
already human-readable `<category>_<family>`). The folder is written locally and **mirrored to GCS** at
`{GCS_ARTIFACTS}/3_Modeling_outputs/{config}/explainability/` (`GCS_ARTIFACTS = GCS_ROOT/V3_categorized`).
Set `EXPLAIN=0` to skip.
**SHAP is fast & exact here:** AdaBoost (the only non-tree-exact learner) is demoted off by default
(`config.EXCLUDE_MODELS="AdaBoost"`), so every deployable model — single OR soft-voting ensemble — is
tree-exact and explained on ALL patients in minutes. The same canonical path backs deployment-time SHAP
in `predict_unseen.py` (it delegates to `explainability._shap_values`).

It also writes a **subgroup / fairness audit** (`4_Heldout/subgroup_audit.py` → `subgroup_audit_{h}.csv`)
on the test slice — Sens/Spec/PPV + calibration (Brier/ECE) within **sex × ethnicity (incl. Unknown) ×
age-band × record-density quartile** (groups with n≥30), and flags (`subgroup_disparities_{h}.csv`) any
stratum whose sensitivity spread exceeds 5 points — the equity check the methodology requires.

**Per-age-band operating thresholds** (`operating_threshold_by_age_{h}.json`) — because lung-cancer risk
rises steeply with age, a single global cut over-flags the elderly (most false positives are 60+). Per
band with ≥20 calib positives, a band-specific cut is chosen on the **calib** slice for a target
sensitivity (default 0.90; under-powered bands fall back to the global cut). This is the primary,
**no-retrain** false-positive lever and is the deployment-correct way to neutralise age dominance at the
decision layer (age stays a model feature; only the *threshold* is age-conditioned).

**Robustness (no silent degradation).** The cached held-out matrix is `ensure_local`-pulled from GCS
before any rebuild (so a fresh box reuses the published matrix instead of rebuilding), then validated by a
column-presence + FE-source content-hash staleness check (rebuild on drift; `--force` to force). Both the
chunked and fresh build paths **hard-fail** (no fallback) on a missing model-feature column, duplicate
patients, or a patient-set mismatch — rather than silently median/0-imputing a structurally-absent
feature.

---

## 7. Running it

Prerequisites: a machine with BigQuery access and the env in `requirements.txt`
(`pip install -r requirements.txt`). All settings live in `config.py` — no `.env` needed.

```bash
python run_v3.py                       # both horizons (12mo then 1mo) × all FE_WINDOWS, GCS caching on
python run_v3.py 12mo --windows 5      # one horizon, single 5-year lookback
python run_v3.py --heldout             # full sweep, then held-out evaluation
python run_v3.py --force                # recompute every stage (ignore cache)
python run_v3.py --no-gcs               # local only (no GCS read/write)
GCS_WRITE=0 python run_v3.py --heldout  # compute + read GCS, but NEVER write/overwrite GCS (test/experiment)
```

`run_v3.py` runs `split → codelist → fe → stable → train → parity-gate` (and optional `--heldout`) for
each **horizon × lookback window**. By default it loops both horizons and every window in `FE_WINDOWS`
(5/10/20/100 yr). The split is built once per horizon (membership is lookback-independent) and shared
across windows; the **codelist is re-discovered per lookback** on that lookback's own train data — no
codelist is reused across windows. **FE is single-core** (the validated path; feature engineering builds
the wide matrix in one pass).

**FE parity gate (correctness, hard-fail).** After each train build, `stage_parity_gate`
(`2_FE/fe_keep_parity.py`, gated by `FE_PARITY_GATE`, default on) proves the **keep-features** build used
by held-out / inference is **byte-identical** to the full build for every model feature (`max abs diff
< 1e-9`), aborting the run on any mismatch. It's **non-mutating** (loads persisted TRAIN artifacts; never
re-writes GCS), so a cross-category or restricted family can never silently diverge at serve.

**GCS writes** are gated by one global switch: `GCS_WRITE=0` makes a run non-mutating to GCS (local writes
still happen) — for read-only tooling / experiments so they can never overwrite production artifacts;
default (or `=1`) publishes normally.

**Each stage is cache-skipped** if its output already exists locally or in GCS — meaning the runner
*reuses* that artifact (pulling it from GCS to local if needed) instead of recomputing it. Because every
artifact is keyed to its exact `(horizon, window)`, this never crosses a window's data with another's.
Use `--force` to recompute from scratch. Stages can also be run individually from their own directories.

---

## 8. Storage & artifacts

The canonical split lives in `SPLIT_DIR` (a `gs://` path, COMMON to both pipeline variants). The 2_FE +
3_Modeling_outputs artifacts are written under `V3_categorized/` locally and mirrored to
`GCS_ARTIFACTS/<relative path>` (= `GCS_ROOT/V3_categorized`) so reruns and teammates reuse them; the
common `splits/` and `raw_events/` cache stay at `GCS_ROOT` (shared with the V3_snomed variant):

The canonical split is the only per-horizon artifact shared across lookbacks; everything else
(including the codelist) is per-`(horizon, window)`. **Naming: Astra = 12mo, Nova = 1mo; the `OFF`
suffix is the fixed data-driven artifact-namespace tag.** (`{L}` = Astra|Nova below.)

| Artifact | Path |
|---|---|
| Lifetime events cache — training (per horizon) | `raw_events/{L}_lifetime.parquet` |
| Lifetime events cache — held-out (per horizon) | `raw_events/heldout_{L}_lifetime.parquet` |
| Canonical split (per horizon) | `SPLIT_DIR/lung_{h}_split.parquet` |
| Codelist (ranked, + scores) | `2_FE/codelist/{L}_{yr}yr_OFF_codelist.csv` |
| Wide feature matrix | `2_FE/output/{L}/{yr}yr_OFF/features_p005_{h}.parquet` |
| Zero-fit sidecars (CATEGORIZED; reloaded at held-out/serve — fit on TRAIN, hard-fail if absent) | `2_FE/output/{L}/{yr}yr_OFF/zstats_categorized_{h}.parquet`, `valuecodes_categorized_{h}.parquet`, `moczero_categorized_{h}.parquet`, `topcats_categorized_{h}.parquet` |
| Stable matrix (+ split column) | `2_FE/output/{L}/{yr}yr_OFF/features_p005_{h}_stable.parquet` |
| Stable feature list | `2_FE/output/{L}/{yr}yr_OFF/stable_features_{h}.csv` |
| Trained model | `3_Modeling_outputs/{L}/{yr}yr_OFF/model_{h}.joblib` |
| Results plot | `3_Modeling_outputs/{L}/{yr}yr_OFF/results_{h}.png` |
| Held-out report + Platt | `3_Modeling_outputs/{L}/{yr}yr_OFF/heldout_recalib_{h}.txt`, `platt_calib_{h}.joblib` |
| Operating thresholds (global + per-age-band) | `3_Modeling_outputs/{L}/{yr}yr_OFF/operating_threshold_{h}.json`, `operating_threshold_by_age_{h}.json` |
| Explainability (SHAP, held-out) | `3_Modeling_outputs/{L}/{yr}yr_OFF/explainability/` — `patient_explanations.csv`, `segment_drivers.csv`, `shap_summary.png`, `segment_*.png` |
| Explainability (SHAP, internal test) | `3_Modeling_outputs/{L}/{yr}yr_OFF/explainability_internal/` — same files, on the internal-test split |
| Subgroup / fairness audit | `3_Modeling_outputs/{L}/{yr}yr_OFF/subgroup_audit_{h}.csv` (+ `subgroup_disparities_{h}.csv` if any flagged) |

---

## 9. Configuration

All settings live in `config.py` — the single source (edit the defaults directly). A real environment
variable overrides any setting per run (e.g. `TOP_N=300 python run_v3.py`), and an optional `.env`, if
present, is also loaded — but none is required. Key knobs:

| Key | Default | Meaning |
|---|---|---|
| `HORIZONS` | `12mo,1mo` | prediction horizons to build |
| `TOP_N` / `OR_MIN` / `PREV_MIN` | `500` / `2.0` / `0.01` | data-driven codelist gates |
| `MIN_VALUE_FRAC` | `0.10` | value-bearing threshold — a code is value-bearing if ≥ this fraction of its events carry a number (lowered 0.30→0.10 2026-06-20: more lab values, fewer count-only codes) |
| `N_FOLDS` / `MIN_FOLDS` / `CUM_IMP` | `5` / `3` / `0.99` | stability selection |
| `TEST_SIZE` / `CALIB_SIZE` | `0.10` / `0.10` | canonical split fractions (→ 80/10/10) |
| `TUNE` / `TUNE_TOP_N` | `0` / `5` | Optuna tuning (off by default) |
| `SELECTION_METRIC` | `Avg Precision` | metric driving model selection/ranking/ensemble (AUPRC; set `ROC AUC` for the old behaviour) |
| `FE_PARITY_GATE` | `True` | run the full-vs-keep-features parity gate after each train build (hard-fail on mismatch) |
| `GCS_WRITE` | `1` | global GCS-write switch — set `0` for a non-mutating (read-only-to-GCS) run |
| `FE_ENGINE` | `pandas` | FE backend — CATEGORIZED is **pandas-only** (run_v3 forces it); Polars is the per-SNOMED variant's default. FE is single-core. |
| `GCS_ROOT` | `gs://…/Lung_Cancer` | COMMON root — `splits/` + `raw_events/` (shared by both variants) |
| `GCS_VARIANT` / `GCS_ARTIFACTS` | `V3_categorized` / `GCS_ROOT/V3_categorized` | this variant's `2_FE` + `3_Modeling_outputs` mirror |
| `SPLIT_DIR` | `gs://…/Lung_Cancer/splits` | canonical split (common) |

Per-step details are in `1_Top_Snomed/README.md`, `2_FE/README.md`, and `4_Heldout/README.md`.
