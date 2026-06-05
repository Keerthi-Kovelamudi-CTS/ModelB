# BigQuery Feature Store — Schema Contract

This document is the **single source of truth** for the schema that
`transform_features_json.py` writes to BigQuery and that downstream
consumers (model server, dashboards, monitoring, lung team) rely on.

The schema is intentionally **identical to lung's** JSON-packed layout
(`prj-cts-ai-dev-sp.prediction_emis.test_prediction1`) so any consumer
that already handles lung's table can also consume ours without
translation.

Any change to this schema **MUST** be coordinated with downstream
consumers and version-bumped. Adding columns is acceptable as an
additive change; renaming or removing columns is a breaking change.

---

## Output table

```
cthesigns-platform-475414-b7.prediction_emis.prostate_features_{window}
```

`{window}` ∈ `{1mo, 3mo, 6mo, 12mo}` — one table per prediction window
(matches the per-window model artefacts).

**Partitioning:** by `partition_date` (DAY). New partition written
every nightly cron run. Set 30-day expiration TTL in BQ console to
bound storage growth.

---

## Columns (7 flat + 1 partition key)

| # | Column | Type | Source | Semantic |
|---|---|---|---|---|
| 1 | `patient_guid` | STRING | EMIS `Admin_Patient.patient_guid` | Patient identifier (stable across runs). |
| 2 | `sex` | STRING | EMIS `Admin_Patient.sex` | `'M'` / `'F'` / NULL. |
| 3 | `patient_ethnicity_16` | STRING | `EMIS_BULK_DATA_temp.Patient_Ethnicity.LABEL_16` | 16-class detailed ethnicity. NULL allowed. |
| 4 | `patient_ethnicity_6` | STRING | `EMIS_BULK_DATA_temp.Patient_Ethnicity.LABEL_6` | 6-class aggregated ethnicity. NULL allowed. |
| 5 | `patient_age` | INT64 | derived in FE preprocess (`AGE_AT_INDEX`) | Age in years at the **prediction anchor date** (not today). For cancer cohort: age at diagnosis; for non-cancer: age at sampled / max-event anchor. |
| 6 | `transformed_features` | STRING (JSON) | FE pipeline output | JSON object containing all ~1300 engineered features (counts, slopes, trends, interactions, acceleration). Schema described below. |
| 7 | `cancer_class` | INT64 | derived in FE | `1` = cancer cohort, `0` = non-cancer cohort. For inference rows on un-diagnosed patients, this is `0` (predicted by the model, not a label). |
| 8 | `partition_date` | DATE | added at BQ write time | UTC date when the nightly FE job ran. Used for partitioning + freshness queries. |

### Notes on flat columns

- **`patient_age`** is named to match lung's contract. Internally in our
  pipeline this is `AGE_AT_INDEX` (semantically clearer). The rename
  happens at the JSON output boundary in `transform_features_json.py`.
- **`sex`, `patient_ethnicity_16/6`** are kept as raw string values
  (not label-encoded). The ENCODED versions live INSIDE
  `transformed_features` as `SEX_MALE`, `ETHNICITY_16_ENC`,
  `ETHNICITY_6_ENC` — see below.

---

## `transformed_features` JSON payload

A JSON object containing all engineered features for one patient.
~1300 keys per row. Schema is feature-set-version-dependent (see
`feature_manifest.json` sidecar — currently `v1.0`).

### Naming conventions inside the JSON

```
{
  "AGE_AT_INDEX": 67,
  "AGE_BAND": 65,
  "SEX_MALE": 1,
  "ETHNICITY_16_ENC": 12,
  "ETHNICITY_6_ENC": 4,

  "OBS_<CATEGORY>_count_A": 3,
  "OBS_<CATEGORY>_count_B": 1,
  "OBS_<CATEGORY>_count_total": 4,
  "OBS_<CATEGORY>_FREQUENCY_PER_YEAR": 0.8,
  "OBS_<CATEGORY>_has_ever": 1,
  ...

  "LAB_<CATEGORY>_mean": 4.5,
  "LAB_<CATEGORY>_slope": -0.02,
  "LAB_<CATEGORY>_worsening": 1,
  "LAB_<CATEGORY>_missing": 0,
  ...

  "MED_<CATEGORY>_count_A": 2,
  ...

  "INT_<feature_name>": 1,            // interaction features
  "AGG_<aggregate_name>": 5,          // aggregate signals
  "TRAJ_<trajectory_name>": 0.3,      // trajectory features
  "ACCEL_<accel_name>": 0.05,         // acceleration features
  "CROSS_<cross_name>": 0,            // cross-domain interactions
  "MEGA_<mega_name>": 1,              // mega-feature combinations
  "<PROSTATE_PREFIX>_<name>": ...     // prostate-specific (see config.PREFIX)
}
```

### Prefix glossary

| Prefix | Block | What it is |
|---|---|---|
| `OBS_` | 4b | Observation event counts/recency per category |
| `LAB_` | 4c | Lab value statistics (mean, slope, min, max, std, worsening) |
| `LABTERM_` | 4c | Per-term lab features (specific test names) |
| `MED_` | 4d | Medication event counts per category |
| `INV_` | (in 4b) | Investigation/imaging counts |
| `INT_` | 4d | Interaction features (pairwise) |
| `AGG_` | 4d | Aggregate symptom-cluster signals |
| `TRAJ_` / `LAB_TRAJ_` | 4f | Trajectory features (slope, persistence) |
| `ACCEL_` | 4g | Acceleration features (recent vs older rate-of-change) |
| `CROSS_` | 4d | Cross-domain interactions |
| `MEGA_` | 4d | Multi-feature mega-aggregates |
| `<PROSTATE_PREFIX>_` | 4d | Prostate-specific (PSA emphasis, see `config.PREFIX`) |

### Time-window suffixes

| Suffix | Meaning |
|---|---|
| `_A` | Earlier half of lookback window (months `[TIME_WINDOW_MID, years_before*12]`) |
| `_B` | Recent half of lookback window (months `[months_before, TIME_WINDOW_MID]`) |
| `_total` | Both A+B combined |
| (no suffix) | Whole lookback or per-patient aggregate |

### Encoded categorical features

| Feature | Encoding | Reference |
|---|---|---|
| `SEX_MALE` | `M`→1, `F`→0, NaN/other→`-1` | hardcoded |
| `ETHNICITY_16_ENC` | stable alphabetical label encoding (0..N-1, unknown→-1) | `encoder_mappings.json` (see below) |
| `ETHNICITY_6_ENC` | stable alphabetical label encoding (0..N-1, unknown→-1) | `encoder_mappings.json` (see below) |

---

## Encoder mappings sidecar

For inference-time consistency, the FE step writes a JSON file:

```
{FE_RESULTS}/{window}/encoder_mappings.json
```

Format:

```json
{
  "sex_male": {"M": 1, "F": 0},
  "ethnicity_16_enc": {
    "Asian or Asian British: Bangladeshi": 0,
    "Asian or Asian British: Indian": 1,
    "...": "..."
  },
  "ethnicity_6_enc": {
    "Asian": 0,
    "Black": 1,
    "Mixed": 2,
    "Other": 3,
    "Unknown": 4,
    "White": 5
  }
}
```

**Inference MUST load this file** and apply the same mapping when
preprocessing new patient data — otherwise integer indices will shift
and the model will see mis-encoded ethnicity values.

Example (in `predict_unseen_input.py`):

```python
import json
with open(f'{FE_RESULTS}/{window}/encoder_mappings.json') as f:
    mappings = json.load(f)

df['SEX_MALE']         = df['SEX'].astype(str).str.upper().map(mappings['sex_male']).fillna(-1).astype(int)
df['ETHNICITY_16_ENC'] = df['PATIENT_ETHNICITY_16'].astype(str).map(mappings['ethnicity_16_enc']).fillna(-1).astype(int)
df['ETHNICITY_6_ENC']  = df['PATIENT_ETHNICITY_6'].astype(str).map(mappings['ethnicity_6_enc']).fillna(-1).astype(int)
```

---

## Versioning policy

| Change type | What to do |
|---|---|
| **Add a new feature key inside `transformed_features`** | Permitted as additive change. Bump `feature_manifest.json` minor version. Notify downstream consumers (no breakage — JSON keys can grow). |
| **Rename or remove a feature key** | Breaking. Bump major version. Coordinate with all consumers BEFORE the FE deploy. |
| **Add a new flat column to the output schema** | Permitted as additive change. Inform consumers (most will ignore new columns; some may want to query them). |
| **Rename or remove a flat column** | Breaking. Must coordinate with the lung team since they own the original contract. |
| **Change the encoding scheme** (e.g. switch ethnicity to one-hot) | Breaking — model artefacts trained on old encoding will mis-predict. Coordinated retrain required. |
| **Change `partition_date` semantics** (e.g. become diagnosis date instead of run date) | Breaking. Affects every freshness query downstream. |

---

## Comparison to lung's schema

| Aspect | Lung | Prostate_Prod_Ready | Compatible? |
|---|---|---|---|
| Flat columns | 7 (patient_guid, sex, eth16, eth6, age, json, class) | Same 7 + `partition_date` | ✅ |
| JSON column name | `transformed_features` | `transformed_features` | ✅ |
| Encoding | Done in modeling (`LabelEncoder` on string columns) | Done in FE (encoded versions packed into JSON, raw strings remain as flat columns) | ✅ both populated — consumer's choice which to use |
| Partition column | `partition_date` (DAY) | `partition_date` (DAY) | ✅ |
| Write disposition | `WRITE_APPEND` | `WRITE_APPEND` | ✅ |

A consumer querying the table never has to know which pipeline
produced the row — the 7-column contract is the same.

---

## Test queries (sanity checks)

```sql
-- 1. How many patients in the latest partition?
SELECT partition_date, COUNT(*) AS rows, COUNT(DISTINCT patient_guid) AS patients
FROM `cthesigns-platform-475414-b7.prediction_emis.prostate_features_1mo`
WHERE partition_date = (SELECT MAX(partition_date) FROM ...)
GROUP BY partition_date;

-- 2. Which features are in transformed_features?
SELECT JSON_EXTRACT_KEYS(transformed_features) AS keys
FROM `...prostate_features_1mo` LIMIT 1;

-- 3. Pull one specific feature for one patient
SELECT
  patient_guid,
  JSON_EXTRACT_SCALAR(transformed_features, '$.LAB_PSA_slope') AS psa_slope,
  JSON_EXTRACT_SCALAR(transformed_features, '$.AGE_AT_INDEX')   AS age,
  cancer_class
FROM `...prostate_features_1mo`
WHERE patient_guid = '<guid>'
  AND partition_date = CURRENT_DATE() - 1;

-- 4. Drift monitoring — median BMI slope over time
SELECT
  partition_date,
  APPROX_QUANTILES(
    SAFE_CAST(JSON_EXTRACT_SCALAR(transformed_features, '$.LAB_BMI_slope') AS FLOAT64),
    100)[OFFSET(50)] AS bmi_slope_median
FROM `...prostate_features_1mo`
GROUP BY partition_date
ORDER BY partition_date DESC
LIMIT 30;
```

---

## Ownership

- **Schema owner:** prostate cancer team (this repo)
- **Producer:** `transform_features_json.py` (this directory)
- **Consumers:** model server, monitoring dashboards, lung team (cross-reference)
- **Last updated:** 2026-05-11
- **Version:** v1.0
