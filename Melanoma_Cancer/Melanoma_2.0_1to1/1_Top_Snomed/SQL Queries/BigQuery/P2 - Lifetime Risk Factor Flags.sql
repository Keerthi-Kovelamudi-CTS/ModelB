/*╔══════════════════════════════════════════════════════════════════╗
  ║  PRODUCTION — MELANOMA  (BigQuery dialect)                     ║
  ║  Phase 1.5: Lifetime Risk Factor Flags                         ║
  ║  Version: 0.1 (SKETCH — codes are clinician-curation placeholders)
  ║  Dates:   1950-01-01 → 2026-04-25                              ║
  ║  Window:  [record start, anchor − 12 months] (full pre-anchor) ║
  ║                                                                 ║
  ║  Purpose:                                                       ║
  ║    Emit one row per cohort patient with binary HAS_* columns    ║
  ║    capturing lifetime "ever-recorded" risk factors that would    ║
  ║    be diluted or missed by a 5-year temporal window.            ║
  ║                                                                 ║
  ║  Anchor convention (matches v4 FE):                             ║
  ║    - Cancer:     date_of_diagnosis                              ║
  ║    - Non-cancer: last_record_date  (pseudo-index)               ║
  ║                                                                 ║
  ║  Output schema:                                                 ║
  ║    PATIENT_GUID, CANCER_CLASS, ANCHOR_DATE,                     ║
  ║    HAS_FAMILY_HX_MELANOMA_CA, HAS_FAMILY_HX_ANY_CA,             ║
  ║    HAS_BRCA_OR_HEREDITARY, HAS_PRIOR_CANCER_NON_MELANOMA,       ║
  ║    HAS_EVER_SMOKER, HAS_DIABETES_EVER, HAS_OBESITY_EVER,        ║
  ║    HAS_PROSTATITIS_EVER, HAS_BPH_EVER, HAS_VASECTOMY_EVER       ║
  ║                                                                 ║
  ║  Downstream:                                                    ║
  ║    Merge into FE feature matrix as static cols (PROST_RF_*).    ║
  ║    LEFT JOIN by PATIENT_GUID; missing → 0.                      ║
  ║                                                                 ║
  ║  TODO before production:                                        ║
  ║    1. Clinical curator validates each flag's SNOMED group.      ║
  ║    2. Decide whether non-cancer anchor should be sampled (to    ║
  ║       match FE's FARM_FINGERPRINT-sampled anchor) vs. fixed at  ║
  ║       last_record_date (current sketch choice).                 ║
  ║    3. Consider adding ETHNICITY-derived risk flag (Black men    ║
  ║       have ~1.7x baseline melanoma cancer risk).                ║
  ╚══════════════════════════════════════════════════════════════════╝*/

WITH params AS (
  SELECT
    DATE '1950-01-01'  AS mh_start,
    DATE '2026-05-27'  AS mh_end,
    12                 AS months_gap_before_anchor,   -- aligned to 12mo-prediction horizon
    '%melanoma%'       AS target_cancer_pattern
),

/* ═══════════════════════════════════════════════════════════════════════════
   Shared reference tables (same conventions as the Top-SNOMED / v4 FE files)
   ═══════════════════════════════════════════════════════════════════════════ */

diagnostic_codes AS (
  SELECT
    SAFE_CAST(code_id AS INT64)               AS code_id,
    source_practice_code,
    SAFE_CAST(snomed_c_t_concept_id AS INT64) AS snomed_c_t_concept_id,
    PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) AS file_date
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Coding_ClinicalCode`
  WHERE snomed_c_t_concept_id IS NOT NULL
    AND SAFE_CAST(code_id AS INT64) IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY SAFE_CAST(code_id AS INT64), source_practice_code
    ORDER BY PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) DESC
  ) = 1
),

patients AS (
  SELECT
    patient_guid,
    sex,
    PARSE_DATE('%Y-%m-%d', date_of_birth) AS date_of_birth
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Admin_Patient`
  WHERE patient_guid IS NOT NULL
    AND date_of_birth IS NOT NULL
    AND (deleted != true OR deleted IS NULL)
),

-- Full pre-anchor medical history (NO lower-bound window — that's the point).
medical_history AS (
  SELECT
    co.patient_guid                                       AS patient_guid,
    PARSE_DATE('%Y-%m-%d', co.effective_date)             AS effective_date,
    dc.snomed_c_t_concept_id
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.CareRecord_Observation` AS co
  INNER JOIN diagnostic_codes AS dc
    ON SAFE_CAST(co.code_id AS INT64) = dc.code_id
   AND co.source_practice_code = dc.source_practice_code
  CROSS JOIN params p
  WHERE PARSE_DATE('%Y-%m-%d', co.effective_date) IS NOT NULL
    AND PARSE_DATE('%Y-%m-%d', co.effective_date) >= p.mh_start
    AND PARSE_DATE('%Y-%m-%d', co.effective_date) <  p.mh_end
    AND dc.snomed_c_t_concept_id IS NOT NULL
),

/* ═══════════════════════════════════════════════════════════════════════════
   Cohort definition (matches v4 12mo logic, minus the 1:1 sampling step —
   downstream FE pipeline applies the sampling; we just need supersets here)
   ═══════════════════════════════════════════════════════════════════════════ */

cancer_snomed_codes AS (
  SELECT DISTINCT SAFE_CAST(SNOMED_CODE AS INT64) AS snomed_code
  FROM `prj-cts-ai-dev-sp.EMIS_MED_CODES.Dim_Cancer_Codes`
  WHERE LOWER(cancer_id) NOT LIKE '%disease%'
),

any_cancer_patient AS (
  SELECT DISTINCT mh.patient_guid
  FROM medical_history mh
  JOIN cancer_snomed_codes csc
    ON mh.snomed_c_t_concept_id = csc.snomed_code
),

target_cancer_patients AS (
  SELECT patient_guid, MIN(effective_date) AS anchor_date
  FROM medical_history mh
  JOIN `prj-cts-ai-dev-sp.EMIS_MED_CODES.Dim_Cancer_Codes` dcc
    ON CAST(mh.snomed_c_t_concept_id AS STRING) = dcc.SNOMED_CODE
  JOIN patients pp USING (patient_guid)
  CROSS JOIN params param
  WHERE TRUE  -- no sex restriction for melanoma
    AND DATE_DIFF(mh.effective_date, pp.date_of_birth, YEAR) >= 18
    AND LOWER(dcc.cancer_id) NOT LIKE '%disease%'
    AND LOWER(dcc.cancer_id) LIKE param.target_cancer_pattern
  GROUP BY 1
),

non_cancer_anchor AS (
  -- Anchor = last_record_date (pseudo-index). Simpler than the FE pipeline's
  -- FARM_FINGERPRINT random anchor — fine here because lifetime flags are
  -- mostly time-stable (family hx, BRCA, etc.). Re-visit if a flag is meant
  -- to be sensitive to a specific point-in-time anchor.
  SELECT
    mh.patient_guid,
    MAX(mh.effective_date) AS anchor_date
  FROM medical_history mh
  JOIN patients pp USING (patient_guid)
  WHERE TRUE  -- no sex restriction for melanoma
    AND NOT EXISTS (
      SELECT 1 FROM any_cancer_patient acp WHERE acp.patient_guid = mh.patient_guid
    )
  GROUP BY 1
),

unified_patients AS (
  SELECT patient_guid, anchor_date, 1 AS cancer_class FROM target_cancer_patients
  UNION ALL
  SELECT patient_guid, anchor_date, 0 AS cancer_class FROM non_cancer_anchor
),

/* ═══════════════════════════════════════════════════════════════════════════
   Flag definitions
   ─────────────────
   Each row attaches one SNOMED code to one flag. EXTEND each UNNEST list
   with the clinician-curated codes before going to production. The
   placeholders below are *illustrative*, not validated.
   ═══════════════════════════════════════════════════════════════════════════ */

flag_codes AS (
  -- ── Family history of melanoma cancer ────────────────────────────────────
  SELECT 'FAMILY_HX_MELANOMA_CA' AS flag_name, snomed_code FROM UNNEST([
    -- TODO: clinician to curate; the following are plausible placeholders.
    160338000,   -- Family history of malignant neoplasm of melanoma
    266702002    -- FH: Carcinoma of melanoma
  ]) AS snomed_code

  UNION ALL
  -- ── Family history of ANY cancer ─────────────────────────────────────────
  SELECT 'FAMILY_HX_ANY_CA', snomed_code FROM UNNEST([
    -- TODO: curate. The "275937001 — Family history of cancer" parent term
    -- and its descendants would normally be expanded via SNOMED hierarchy.
    275937001,   -- Family history of cancer
    160373006,   -- FH: Carcinoma of melanoma
    160405004    -- FH: Carcinoma of colon
  ]) AS snomed_code

  UNION ALL
  -- ── BRCA1 / BRCA2 / hereditary cancer syndromes ──────────────────────────
  SELECT 'BRCA_OR_HEREDITARY', snomed_code FROM UNNEST([
    -- TODO: curate. Include BRCA test recorded, Lynch syndrome, HOXB13.
    412734009,   -- BRCA1 gene mutation positive
    412739004,   -- BRCA2 gene mutation positive
    315058005    -- Hereditary non-polyposis colon cancer syndrome (Lynch)
  ]) AS snomed_code

  UNION ALL
  -- ── Prior cancer of any type (excluding the melanoma target itself) ──────
  -- NB: This is captured below via a NOT-IN join against cancer_snomed_codes,
  -- not via a static list. We still need a marker name. Codes here intentionally
  -- empty (handled by the patient_flag_hits CTE below).
  SELECT 'PRIOR_CANCER_NON_MELANOMA', CAST(NULL AS INT64) FROM UNNEST([]) AS snomed_code

  UNION ALL
  -- ── Ever-smoker ──────────────────────────────────────────────────────────
  SELECT 'EVER_SMOKER', snomed_code FROM UNNEST([
    -- TODO: curate. Includes current smoker + ex-smoker + smoking history codes.
    77176002,    -- Smoker
    8517006,     -- Ex-smoker
    65568007     -- Cigarette smoker
  ]) AS snomed_code

  UNION ALL
  -- ── Diabetes (lifetime ever) ─────────────────────────────────────────────
  SELECT 'DIABETES_EVER', snomed_code FROM UNNEST([
    73211009,    -- Diabetes mellitus
    44054006,    -- Diabetes mellitus type 2
    46635009     -- Diabetes mellitus type 1
  ]) AS snomed_code

  UNION ALL
  -- ── Obesity ever (any BMI≥30 recorded OR obesity diagnosis code) ─────────
  -- For BMI-based detection, the BMI threshold would be applied in clean_data.py
  -- with VALUE filtering. Here we capture explicit diagnosis codes only.
  SELECT 'OBESITY_EVER', snomed_code FROM UNNEST([
    414916001,   -- Obesity
    162864005    -- Body mass index 30+ - obesity
  ]) AS snomed_code

  UNION ALL
  -- ── Prostatitis history ──────────────────────────────────────────────────
  SELECT 'PROSTATITIS_EVER', snomed_code FROM UNNEST([
    9713002,     -- Prostatitis
    27250006     -- Chronic prostatitis
  ]) AS snomed_code

  UNION ALL
  -- ── BPH (benign prostatic hyperplasia) history ───────────────────────────
  SELECT 'BPH_EVER', snomed_code FROM UNNEST([
    266569009    -- Hyperplasia of melanoma (BPH)
  ]) AS snomed_code

  UNION ALL
  -- ── Vasectomy ────────────────────────────────────────────────────────────
  SELECT 'VASECTOMY_EVER', snomed_code FROM UNNEST([
    65487007     -- Vasectomy
  ]) AS snomed_code
),

/* ═══════════════════════════════════════════════════════════════════════════
   Per-patient flag hits — anything before (anchor − 12mo)
   ═══════════════════════════════════════════════════════════════════════════ */

patient_flag_hits AS (
  SELECT
    up.patient_guid,
    up.anchor_date,
    up.cancer_class,
    fc.flag_name
  FROM unified_patients up
  JOIN medical_history mh USING (patient_guid)
  JOIN flag_codes fc ON fc.snomed_code = mh.snomed_c_t_concept_id
  CROSS JOIN params p
  WHERE mh.effective_date < DATE_SUB(up.anchor_date, INTERVAL p.months_gap_before_anchor MONTH)
  GROUP BY 1, 2, 3, 4
),

-- Special case: "prior cancer (non-melanoma)" — driven by any cancer SNOMED
-- in the patient's history that is NOT a melanoma code AND before anchor-12mo.
prior_cancer_non_melanoma_hits AS (
  SELECT
    up.patient_guid,
    up.anchor_date,
    up.cancer_class,
    'PRIOR_CANCER_NON_MELANOMA' AS flag_name
  FROM unified_patients up
  JOIN medical_history mh USING (patient_guid)
  JOIN `prj-cts-ai-dev-sp.EMIS_MED_CODES.Dim_Cancer_Codes` dcc
    ON CAST(mh.snomed_c_t_concept_id AS STRING) = dcc.SNOMED_CODE
  CROSS JOIN params p
  WHERE LOWER(dcc.cancer_id) NOT LIKE '%disease%'
    AND LOWER(dcc.cancer_id) NOT LIKE p.target_cancer_pattern   -- exclude melanoma-itself
    AND mh.effective_date < DATE_SUB(up.anchor_date, INTERVAL p.months_gap_before_anchor MONTH)
  GROUP BY 1, 2, 3
),

all_flag_hits AS (
  SELECT * FROM patient_flag_hits
  UNION ALL
  SELECT * FROM prior_cancer_non_melanoma_hits
)

/* ═══════════════════════════════════════════════════════════════════════════
   Pivot to one row per (patient, anchor) with binary HAS_* columns
   ═══════════════════════════════════════════════════════════════════════════ */

SELECT
  up.patient_guid                                            AS PATIENT_GUID,
  up.cancer_class                                            AS CANCER_CLASS,
  up.anchor_date                                             AS ANCHOR_DATE,
  -- Lifetime flags (1 if ever before anchor-12mo, else 0)
  IF(COUNTIF(h.flag_name = 'FAMILY_HX_MELANOMA_CA')     > 0, 1, 0) AS HAS_FAMILY_HX_MELANOMA_CA,
  IF(COUNTIF(h.flag_name = 'FAMILY_HX_ANY_CA')          > 0, 1, 0) AS HAS_FAMILY_HX_ANY_CA,
  IF(COUNTIF(h.flag_name = 'BRCA_OR_HEREDITARY')        > 0, 1, 0) AS HAS_BRCA_OR_HEREDITARY,
  IF(COUNTIF(h.flag_name = 'PRIOR_CANCER_NON_MELANOMA') > 0, 1, 0) AS HAS_PRIOR_CANCER_NON_MELANOMA,
  IF(COUNTIF(h.flag_name = 'EVER_SMOKER')               > 0, 1, 0) AS HAS_EVER_SMOKER,
  IF(COUNTIF(h.flag_name = 'DIABETES_EVER')             > 0, 1, 0) AS HAS_DIABETES_EVER,
  IF(COUNTIF(h.flag_name = 'OBESITY_EVER')              > 0, 1, 0) AS HAS_OBESITY_EVER,
  IF(COUNTIF(h.flag_name = 'PROSTATITIS_EVER')          > 0, 1, 0) AS HAS_PROSTATITIS_EVER,
  IF(COUNTIF(h.flag_name = 'BPH_EVER')                  > 0, 1, 0) AS HAS_BPH_EVER,
  IF(COUNTIF(h.flag_name = 'VASECTOMY_EVER')            > 0, 1, 0) AS HAS_VASECTOMY_EVER
FROM unified_patients up
LEFT JOIN all_flag_hits h
  USING (patient_guid, anchor_date, cancer_class)
GROUP BY PATIENT_GUID, CANCER_CLASS, ANCHOR_DATE
ORDER BY PATIENT_GUID;
