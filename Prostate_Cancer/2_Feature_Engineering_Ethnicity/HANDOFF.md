# FE Pipeline Handoff — Prostate Cancer

The feature-engineering logic for the Prostate Cancer model. Feature formulas are production-ready; orchestration, I/O, and selective-mode computation are research-grade and need to be rewritten for the production data pipeline.

---

## TL;DR — read this first

> The feature formulas in `3_pipeline.py`, `4_cancer_features.py`, `6_text_features.py` and the settings in `config.py` are **production-ready — do not change them, just call them.** Your job is:
>
> 1. **Replace CSV I/O** with BigQuery reads/writes throughout the pipeline.
> 2. **Add selective-mode** — accept a list of top-N feature names and compute/emit only those.
> 3. **Build a job entrypoint** parameterised for prod (BQ tables in, BQ table out, features file, window).
> 4. **Decide a config deployment strategy** so training and prod always use the same `config.py`.
>
> `combine_outputs.py` is your reference for selective mode and how the 4 FE outputs join. The "done" criterion is feature-row parity with the research pipeline on a shared patient. Ping Keerthi on any feature formula that looks odd before changing it.

---

## Data Flow (end-to-end)

```
Raw SQL export  (obs CSV + med CSV, per window)
    │
    ▼  1_sanity_check.py
Cleaned cohort:  deduped, SEX='M' & age≥18, no pos/neg overlap, no med-only
    │
    ▼  2_clean_data.py
Dates parsed, lab outliers → NaN, LEAKAGE GUARD (drops EVENT_DATE >= INDEX_DATE)
    │
    ▼  3_pipeline.py  (generic, 7 blocks)
      + 4_cancer_features.py  (prostate-specific: PSA, urinary, bone)
~900 features per patient, per window
    │
    ▼  5_cleanup.py
Remove near-zero variance, high correlation (>0.98), leakage (label-corr >0.5)
≈ 200–300 features per window retained
    │
    ▼  6_text_features.py
+ 28 keyword features, +15 TF-IDF dims, +15 BERT dims
Saves fitted transformers  (MUST be reused at inference — do not refit in prod)
    │
    ▼  combine_outputs.py   (reference for prod)
One final matrix per window, optionally filtered to top-N feature subset
```

**Prod equivalent**: same flow, but CSV I/O → BigQuery reads/writes, and `combine_outputs.py` replaced by your prod job entrypoint.

---

## What's in this folder

### Production-ready logic (call these — don't modify)

| File | What it contains |
|------|------------------|
| `config.py` | Single source of truth: SNOMED categories, lab physiological ranges, cluster definitions, interaction pairs, text patterns, PREFIX (`PROST_`), windows (`3mo`, `6mo`, `12mo`), modeling params |
| `3_pipeline.py` | 7 generic feature builders — `build_clinical_features`, `build_medication_features`, `build_interaction_features`, `build_advanced_features`, `extract_maximum_features`, `build_new_signal_features`, `build_trend_features`. Parameterised by config — reusable across cancers |
| `4_cancer_features.py` | Prostate-specific block — PSA dynamics, PSA age-band percentile, urinary patterns, bone/metastatic signals, treatment flags, composite risk score |
| `6_text_features.py` | NLP — keyword flags, TF-IDF (15 dims via SVD), BERT embeddings (15 dims via PCA) using PubMedBERT. Auto-detects CUDA |

### Research-grade glue (rewrite for prod, preserve the behaviour)

| File | Why it's research-only |
|------|------------------------|
| `0_run_pipeline.py` | Local driver with `--step` CLI flags; you'll replace this entirely with your prod job entrypoint |
| `1_sanity_check.py` | Overlap / dedup / sex-age filter; currently reads CSVs, should read BQ tables in prod |
| `2_clean_data.py` | Date parsing, lab outlier clamping, **leakage guard** (drops rows where `EVENT_DATE >= INDEX_DATE`) — **keep the guard in prod** |
| `5_cleanup.py` | Feature-cleanup heuristics (near-zero variance, high correlation, leakage > 0.5 label-corr). Logic is fine; I/O is CSV |
| `combine_outputs.py` | **Reference implementation** of the join + filter step you need to productionize |

---

## Input schema

FE expects one obs CSV and one med CSV per window. In prod these become BQ tables.

### Observations (per window, e.g. `prostate_3mo_obs_dropped.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `PATIENT_GUID` | str | Stable patient identifier |
| `INDEX_DATE` | date | Anchor date (diagnosis for positives; synthetic for negatives) |
| `EVENT_DATE` | date | Event date — must be `< INDEX_DATE` (enforced by `2_clean_data.py`) |
| `LABEL` | int (0 \| 1) | 1 = cancer within prediction window, 0 = not |
| `CATEGORY` | str | Mapped SNOMED category (see `config.OBS_CATEGORIES`) |
| `TERM` | str | Human-readable term |
| `CODE_ID` | str/int | Raw SNOMED code (used for code_category_mapping) |
| `VALUE` | float (nullable) | Numeric reading (labs) or NaN |
| `TIME_WINDOW` | str ('A' \| 'B') | Temporal bin: A = earlier, B = later (both before INDEX_DATE) |
| `MONTHS_BEFORE_INDEX` | float | Months between event and index |
| `ASSOCIATED_TEXT` | str (nullable) | Clinical note text (XML/plain) |
| `SEX` | str | 'M' required (filtered in sanity check) |
| `AGE_AT_INDEX` | float | Age at INDEX_DATE |
| `ETHNICITY_GROUP` | str | Patient-level ethnicity (see allowed values below) — required for ethnicity features |

**`ETHNICITY_GROUP` allowed values** (case-sensitive, must match `config.ETHNICITY_CATEGORIES`):

`White`, `Asian`, `Black`, `Chinese`, `Mixed`, `Other`, `Not specified`

Any other value is mapped to `Not specified` by `3_pipeline.py`. `ETHNICITY_GROUP` must be **constant per `PATIENT_GUID`** (enforced by upstream demographics join — pipeline takes the first value per patient if rows disagree).

### Medications (per window)

Same schema as obs, minus `ASSOCIATED_TEXT`. `VALUE` = prescribed quantity.

---

## Output schema

One matrix per window. See `combine_outputs.py` for the reference join of the 4 FE outputs (cleanup + keywords + TF-IDF + BERT).

### Feature naming convention

| Prefix | Source | Example |
|--------|--------|---------|
| `PROST_` | Prostate-specific (`4_cancer_features.py`) | `PROST_LAB_psa_elevated_10` |
| `OBS_<CATEGORY>_` | Generic obs per category | `OBS_PSA_count_A` |
| `MED_<CATEGORY>_` | Generic med per category | `MED_ALPHA_BLOCKERS_count_B` |
| `LAB_<CATEGORY>_` | Lab value stats | `LAB_PSA_mean_B` |
| `LAB_TRAJ_` | Lab trajectory | `LAB_TRAJ_PSA_slope` |
| `LABTERM_` | Per-lab-term (top 30) | `LABTERM_Serum_PSA_last` |
| `CLUSTER_` | Symptom clusters | `CLUSTER_URINARY_count` |
| `INT_` | Obs×med interactions | `INT_luts_plus_psa` |
| `AGEX_`, `AGE_` | Age interactions / bands | `AGEX_PSA_over65` |
| `DECAY_` | Time-decay weighted | `DECAY_PSA_weighted` |
| `RECUR_`, `SEQ_` | Recency / sequence | `RECUR_PSA_days_since_last` |
| `MED_ESC_` | Medication escalation | `MED_ESC_polypharmacy_B` |
| `TEXT_` | Text keyword flags | `TEXT_CLIN_psa_rising` |
| `EMB_text_dim_N` | TF-IDF SVD | `EMB_text_dim_0` |
| `BERT_dim_N` | BERT PCA | `BERT_dim_0` |
| `ETH_<GROUP>` | Ethnicity one-hot | `ETH_BLACK`, `ETH_WHITE`, `ETH_NOT_SPECIFIED` |
| `PROST_RF_black_ethnicity` | Named prostate-specific risk factor | 1 if patient is Black, else 0 |
| `PROST_RF_ethnicity_not_specified` | Missing ethnicity flag | 1 if ethnicity is 'Not specified' |
| `PROST_RF_black_and_elderly` | Interaction: Black × age ≥ 65 | 1 if both conditions met |

Plus `AGE_AT_INDEX`, `AGE_BAND`, `LABEL`, `PATIENT_GUID` (index).

### Output matrix shape
- Rows: one per patient in the cohort
- Columns: ~1,000 features + LABEL
- Dtypes: mostly float64, flag features are int64
- Missing values: filled with 0 at the combine step

---

## What you need to build for prod

### 1. Selective-mode feature computation

**Current behaviour**: `0_run_pipeline.py` computes all ~1,000 features regardless of what's requested.

**Target behaviour**: accept a list of top-N feature names → compute only those → write to BQ.

**Recommended for v1 — filter at end**: run the full pipeline, then use `combine_outputs.py --features features.json` to filter columns before the BQ write. This is the simplest correct implementation.

**v2 optimization (only if compute cost demands it)**: push the whitelist down into each builder in `3_pipeline.py` / `4_cancer_features.py` so they skip computation of unused features. Suggested approach:

- Add a `requested: set[str] | None` arg to each builder.
- Each builder keeps a list of `(feature_name, compute_fn)` tuples and skips any whose name isn't in `requested`.
- When `requested is None`, compute everything (current default).

### 2. BigQuery I/O

Replace the CSV reads in `1_sanity_check.py`, `2_clean_data.py`, etc. with BQ reads. Replace the CSV writes in `5_cleanup.py` and `combine_outputs.py` with BQ writes.

Recommended interface:
```python
# Read
df = pd.read_gbq('SELECT * FROM prj.ds.obs_3mo', project_id='prj')
# Write
features.to_gbq('prj.ds.features_3mo', project_id='prj', if_exists='replace')
```

### 3. Prod job entrypoint

Replace `0_run_pipeline.py` with something like:

```bash
python main.py \
  --input-obs-table prj.ds.obs_3mo \
  --input-med-table prj.ds.med_3mo \
  --output-table prj.ds.features_3mo \
  --features features.json \
  --window 3mo
```

### 4. Config deployment strategy

`config.py` contains category mappings and thresholds that MUST match between training and production — otherwise a feature like `OBS_PSA_count_A` means different things in the two systems and the model breaks silently. Either:
- Pickle the relevant bits of config at training time and load them in prod, or
- Pin `config.py` as the authoritative source and deploy it unchanged.

---

## Acceptance test — the "done" criterion

After your prod pipeline is built, verify parity with the research pipeline:

1. Pick one window (start with `3mo`) and one patient that exists in both the training data and your prod input.
2. Run the research pipeline locally → capture that patient's feature row from the combined matrix.
3. Run your prod pipeline on the same patient → capture their feature row.
4. **Every feature value must match to within floating-point tolerance** (`np.allclose(a, b, rtol=1e-5)`).

If any feature differs, one of these went wrong:
- Config drift between training and prod (`config.py` mismatch)
- Different lab clamping ranges
- Fitted transformers (TF-IDF / SVD / BERT PCA) were refit instead of loaded from training artifacts
- Leakage guard not applied
- `TIME_WINDOW` assigned differently in prod SQL

The AUC on held-out data is the **business** test. Feature parity is the **engineering** test. Both must pass before prod goes live.

---

## Known invariants / assumptions (preserve all of these in prod)

1. Every `EVENT_DATE` is strictly before `INDEX_DATE`. Enforced by the leakage guard in `2_clean_data.py`.
2. Every patient has at least one obs row (med-only patients are dropped in `1_sanity_check.py`).
3. Sex is 'M' (filter applied; prostate cancer model).
4. Age at index ≥ 18 (filter applied).
5. The same patient never appears in both pos and neg (overlap removal).
6. `TIME_WINDOW ∈ {'A', 'B'}` — assigned upstream by the SQL. A = earlier period, B = later period. Both precede index.
7. Lab `VALUE` outliers outside `config.LAB_RANGES` are clamped to NaN (the row stays, only the value is nulled).
8. `code_category_mapping.json` must be consistent between training and prod inference.

---

## Running locally (for reference and debugging)

```bash
# Full pipeline (all features, all windows)
python 0_run_pipeline.py --step all

# Individual steps
python 0_run_pipeline.py --step sanity   # overlap, dedup, filter
python 0_run_pipeline.py --step clean    # date parse, lab clamp, leakage guard
python 0_run_pipeline.py --step fe       # steps 3-4 (generic + prostate-specific)
python 0_run_pipeline.py --step cleanup  # step 5 (variance, correlation, leakage)
python 0_run_pipeline.py --step text     # step 6 (keywords + TF-IDF + BERT)

# Combine outputs into a single matrix
python combine_outputs.py                             # keep all features
python combine_outputs.py --features top_n.json       # selective (filter at end)
python combine_outputs.py --windows 3mo,6mo           # subset of windows
```

---

## Dependencies

```
pandas, numpy, scikit-learn
optuna, joblib
sentence-transformers  (PubMedBERT)
transformers, torch    (torch must match the CUDA driver on the prod box)
```

For a CUDA 12.2 driver: `pip install --index-url https://download.pytorch.org/whl/cu121 torch`.

---

## Point of contact

- FE logic + cancer-specific features: **Keerthi**
- Prod data pipeline: **Alex**
