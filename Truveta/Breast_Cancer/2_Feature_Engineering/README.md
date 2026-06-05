# Truveta Breast — Feature Engineering (Category-Based, heavy-FE)

Category-based feature engineering over the Truveta GOLD breast cohort. Codes
are grouped into clinical categories (curated codelist at
`Truveta/Breast_Cancer/codelists/breast_curated_categories.tsv`), and each
(patient × category) is summarised along **6 aggregations**, on top of several
hand-engineered layers. Dual-horizon: built separately for the **1mo** and
**12mo** windows.

## Cohort design (SQL — `SQL/breast_truveta_{1mo,12mo}.sql`)

| Knob | Value | Notes |
|---|---|---|
| Source | `truveta_gold` (`patient` / `clinical_event` / `medication`) | cohort = Breast vs Non-Cancer |
| States | **7** (FL, CT, WI, OR, MS, VT, **WA**) | restricted up front (FIX 1) |
| Cancer def | ICD `^C50` + SNOMED breast-malignancy | first diagnosis per patient = anchor |
| **Non-cancer anchor** | **gold-layer cancer-date pool** | each non-cancer patient is assigned a random date drawn from the **cancer diagnosis-date distribution** (`cancer_anchor_pool` + `MOD(ABS(FARM_FINGERPRINT(patient_id)), N)+1`). Aligned to `Git_SQL/cancer_training_truveta_v3_gold.sql`. Guarantees the two cohorts share the same anchor-**year** distribution by construction. |
| Non-cancer sampling | matched per **(sex, anchor_year)** | keeps both sex mix and year mix aligned (breast ≈99% female → sex-match prevents a "male → non-cancer" confound; the lung gold template does NOT sex-match) |
| `non_cancer_ratio` | **5** | oversample non-cancer 5× as a **buffer** so balancing to 1:1 downstream never has to discard positives |
| Leakage exclusions | breast-specific (Tasks #21/#29) | post-diagnosis Z-codes, mastectomy/reconstruction, abnormal-mammography/BIRADS, biopsy, chemo; deliberately keeps Z12 screening |
| `min_obs_events_per_patient` (cancer) | **1** | keep any cancer patient with ≥1 obs event (protect positives) |
| `min_obs_events_non_cancer` | **5** | non-cancer must have ≥5 obs events in the (foreign-anchor) window |

The 1mo and 12mo SQL files are identical except `months_before` (1 vs 12).

## Approach B — keep sparse patients (`config.KEEP_SPARSE_PATIENTS = True`)

Every cohort patient stays in the feature matrix. Patients with no/few curated
events fall through the left-join as **all-zero / sentinel rows** (the EMIS
`__PLACEHOLDER__` analog), so the model trains on the "no curated signal → no
cancer" boundary and does not go out-of-distribution on sparse patients at
deployment. `1_build_features.py` logs the sparse fraction. (The legacy
`MIN_OBS_EVENTS_PER_PATIENT = 10` drop is only applied if
`KEEP_SPARSE_PATIENTS = False`.)

## Feature schema

**Layer A — category aggregations** (57 curated categories × 6 aggregations ≈ ~340 cols):

| Aggregation | Meaning |
|---|---|
| `{cat}__count`         | total events in the 5yr pre-anchor window |
| `{cat}__present`       | binary (0/1) — ≥1 event in category |
| `{cat}__value_mean`    | mean numeric value (NaN where never measured) |
| `{cat}__duration_sum`  | total medication duration (days) |
| `{cat}__recency_days`  | days from anchor to most-recent event (9999 = never) |
| `{cat}__event_density` | events per active month |

Plus hand-engineered layers (see `breast_features.py`):
- **Layer B** — breast risk factors (`BRCA_HRT_*`, alcohol, age thresholds)
- **Layer C** — engagement / code-type-aware (`ENG_*`)
- **Layer E** — extended composites, has-flags, hereditary load, age × clinical (`BRCA_E_*`)
- **Layer F** — first-event ages, per-category recency, age polynomials
- **Layer D** — cross-feature interactions (`INT_*`). **Built LAST**, because its
  pairs reference Layer B/C/E features — building it earlier silently skips them.
- **Layer G** — lung-style **temporal trend / worsening**, per category, only for
  the **31 STRONG/MEDIUM categories** (`config.TREND_CATEGORIES`). `0_extract.py`
  splits each patient's events into older-half (`count_h1`) and recent-half
  (`count_h2`) at the **midpoint of that patient's OWN event timeline** (patient-
  relative, lung-style — NOT a fixed calendar point), so short-history patients
  still get a real within-history trend. Layer G emits `{cat}__trend_h2_over_h1`,
  `{cat}__is_worsening`, `{cat}__recent_frac` (93 features). `h1==0` (neutral 0)
  now only happens for single-event patients. Plus patient-level **ACCEL** in
  Layer C (`ENG_ACCEL_H2_OVER_H1`, `ENG_ACCEL_H2_MINUS_H1`) for EMIS parity.
  Mirrors lung `SNOMED_json_v3` + EMIS `SEQ_*`/`ACCEL_*`. Skipped on a pre-trend extract.

Total ~560 raw features (~342 Layer A + ~126 B/C/D/E/F + 93 Layer G trend) + 8
spine columns; feature selection in modeling trims to ~372 (1mo) / ~500 (12mo).

## Curated codelist (~362 codes / 57 categories — curated down from the earlier noisy ~71-cat list)

`codelists/breast_curated_categories.tsv` — columns: `code_type`
(loinc/snomed/rxnorm/cvx/cpt/...), `category`, `code_id`, `term`.

## Pipeline

```
truveta_gold.breast_cohort_events_{1mo,12mo}  +  codelists/breast_curated_categories.tsv
(cohort tables built by running SQL/breast_truveta_{1mo,12mo}.sql in BQ Studio)
        │  0_extract.py --window {1mo,12mo}        (BQ aggregate: patient × code, + full spine)
        ▼
   data/raw/{window}/breast_patient_code_aggregates.csv, breast_spine.csv, breast_codelist.parquet
        │  1_build_features.py --window {1mo,12mo}  (categories → 6 aggs + Layers B/C/D/E/F; Approach B)
        ▼
   data/features/{window}/breast_feature_matrix.parquet
        │  ../3_Modeling/_shared_split.py           (random stratified split, balanced to 1:1)
        ▼
   ../3_Modeling/shared_split.json  →  3_Modeling → 4_Inference → 6_Explainability → 7_Holdout_Test
```

## Split + balance (`../3_Modeling/_shared_split.py`)

- Default mode **`random`** (stratified 75/10/15) — matches `Prostate_B+C`. No
  temporal holdout.
- `build_shared_split(balance_to_1to1=True)` downsamples the majority class to
  the minority count **before** the split, so **train, val, and test are all
  ~1:1** regardless of any uneven patient loss at the cross-window intersection
  or the anchor's min-events filter. Positives are never discarded (the 5×
  non-cancer buffer absorbs the trim).

## Run

```bash
gcloud auth login   # if needed
cd Truveta/Breast_Cancer/2_Feature_Engineering
python3 0_extract.py        --window 1mo   && python3 0_extract.py        --window 12mo
python3 1_build_features.py --window 1mo   && python3 1_build_features.py --window 12mo
cd ../3_Modeling && python3 _shared_split.py     # balanced random split
python3 1_training.py
```

## Alignment with the EMIS breast pipeline (`/Breast_Cancer/Breast_2.0_1to1`)

EMIS was aligned to this pipeline (2026-06-04) — same anchor, ratio, split,
balancing, Approach B, and FE design. Remaining differences are inherent to the
data source:

| Dimension | Truveta (this) | EMIS | Match? |
|---|---|---|---|
| Non-cancer anchor | gold cancer-date pool | gold cancer-date pool | ✅ |
| `non_cancer_ratio` | 5 | 5 | ✅ |
| Split + balance-to-1:1 | random + balanced | random + balanced | ✅ |
| Approach B | yes | yes | ✅ |
| FE: 6 aggs + B/C/D/E/F | yes | yes (block-structured) | ✅ |
| Region/state filter | 7 US states | n/a (UK data) | data-source difference |
| Sex matching | sex × year (both sexes) | n/a (female-only) | data-source difference |

## Tweaking

- **Codelist:** edit `codelists/breast_curated_categories.tsv` (picked up next run).
- **Aggregations:** `config.CATEGORY_AGGREGATIONS`.
- **Interactions:** `config.INTERACTION_PAIRS` (resolved in Layer D, built last).
- **Cohort ratio / states / leakage:** `SQL/breast_truveta_*.sql` params block.
- **Keep vs drop sparse patients:** `config.KEEP_SPARSE_PATIENTS`.
