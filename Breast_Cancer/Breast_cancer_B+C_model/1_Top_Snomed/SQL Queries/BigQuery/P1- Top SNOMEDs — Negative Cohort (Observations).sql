/*╔══════════════════════════════════════════════════════════════════╗
  ║  PRODUCTION — BREAST  (BigQuery dialect)                     ║
  ║  Phase 1.1: Top SNOMEDs (NEGATIVE COHORT — Observations)       ║
  ║  Version: 1.0                                                   ║
  ║  Dates:   1950-01-01 → 2026-03-11                              ║
  ║  Window:  [-15 months, -1 month] before pseudo index date      ║
  ║                                                                 ║
  ║  Negative cohort:                                               ║
  ║    - Female only (breast is female-only)                          ║
  ║    - Adults (>18)                                               ║
  ║    - No cancer diagnosis of ANY type ever                       ║
  ║    - No palliative care code (1403151000000103)                 ║
  ║    - Age-band + sex matched to breast cases (10:1)            ║
  ║    - Pseudo index = last record date                            ║
  ║    - Minimum 48 months GP history required                      ║
  ╚══════════════════════════════════════════════════════════════════╝*/

WITH params AS (
  SELECT
    DATE '1950-01-01'  AS mh_start,
    DATE '2026-05-27'  AS mh_end,
    15                 AS months_lookback_start,
    1                  AS months_lookback_end,
    5                  AS min_patient_threshold
),

diagnostic_codes AS (
  SELECT
    SAFE_CAST(code_id AS INT64)                  AS code_id,
    source_practice_code,
    term                                         AS term,
    SAFE_CAST(snomed_c_t_concept_id AS INT64)    AS snomed_c_t_concept_id,
    snomed_c_t_description_id                    AS snomed_c_t_description_id,
    PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) AS file_date
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Coding_ClinicalCode`
  WHERE snomed_c_t_concept_id IS NOT NULL
    AND term IS NOT NULL
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

medical_history AS (
  SELECT
    co.source_practice_code                                AS practice_id,
    co.patient_guid                                        AS patient_guid,
    PARSE_DATE('%Y-%m-%d', co.effective_date)              AS effective_date,
    co.observation_type                                    AS observation_type,
    co.value                                               AS value,
    co.associated_text                                     AS associated_text,
    dc.term,
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

all_cancer_patients AS (
  SELECT DISTINCT mh.patient_guid
  FROM medical_history AS mh
  INNER JOIN `prj-cts-ai-dev-sp.EMIS_MED_CODES.Dim_Cancer_Codes` AS dcc
    ON mh.snomed_c_t_concept_id = SAFE_CAST(dcc.SNOMED_CODE AS INT64)
  WHERE LOWER(dcc.cancer_id) NOT LIKE '%disease%'
),

cancer_patients AS (
  SELECT mh.patient_guid, MIN(mh.effective_date) AS date_of_diagnosis
  FROM medical_history AS mh
  INNER JOIN `prj-cts-ai-dev-sp.EMIS_MED_CODES.Dim_Cancer_Codes` AS dcc
    ON mh.snomed_c_t_concept_id = SAFE_CAST(dcc.SNOMED_CODE AS INT64)
  WHERE LOWER(dcc.cancer_id) NOT LIKE '%disease%'
    AND LOWER(dcc.cancer_id) LIKE '%breast%'
  GROUP BY 1
  HAVING MIN(mh.effective_date) IS NOT NULL
),

cancer_profile AS (
  SELECT cp.patient_guid, pp.sex,
    CAST(FLOOR(DATE_DIFF(cp.date_of_diagnosis, pp.date_of_birth, DAY) / 365.25 / 5) * 5 AS INT64) AS age_band
  FROM cancer_patients cp
  INNER JOIN patients pp ON pp.patient_guid = cp.patient_guid
  WHERE FLOOR(DATE_DIFF(cp.date_of_diagnosis, pp.date_of_birth, DAY) / 365.25) > 18
),

cancer_age_sex AS (
  SELECT sex, age_band, COUNT(*) AS cancer_count FROM cancer_profile GROUP BY 1, 2
),

non_cancer_last_record AS (
  SELECT patient_guid, MAX(effective_date) AS last_record_date
  FROM medical_history
  GROUP BY 1
),

non_cancer_pool AS (
  SELECT DISTINCT
    p.patient_guid, p.date_of_birth, p.sex,
    lr.last_record_date AS pseudo_index_date,
    CAST(FLOOR(DATE_DIFF(lr.last_record_date, p.date_of_birth, DAY) / 365.25 / 5) * 5 AS INT64) AS age_band
  FROM patients p
  INNER JOIN non_cancer_last_record lr ON lr.patient_guid = p.patient_guid
  CROSS JOIN params param
  WHERE FLOOR(DATE_DIFF(lr.last_record_date, p.date_of_birth, DAY) / 365.25) > 18
    AND NOT EXISTS (SELECT 1 FROM all_cancer_patients cp WHERE cp.patient_guid = p.patient_guid)
    AND NOT EXISTS (
      SELECT 1 FROM medical_history pmh
      WHERE pmh.patient_guid = p.patient_guid
        AND pmh.snomed_c_t_concept_id = 1403151000000103
    )
    AND DATE_SUB(lr.last_record_date, INTERVAL 48 MONTH) >= param.mh_start
    AND lr.last_record_date <= param.mh_end
),

non_cancer_patients AS (
  SELECT ncp.patient_guid, ncp.date_of_birth, ncp.sex, ncp.pseudo_index_date
  FROM non_cancer_pool ncp
  INNER JOIN cancer_age_sex cas ON ncp.sex = cas.sex AND ncp.age_band = cas.age_band
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY ncp.sex, ncp.age_band
    ORDER BY FARM_FINGERPRINT(ncp.patient_guid)
  ) <= cas.cancer_count * 10
),

-- Admin / noise SNOMEDs to drop from the ranking.
excluded_snomeds AS (
  SELECT snomed_code FROM UNNEST([
    1572871000006101, 279991000000102, 428481002,
    979851000000101,  887641000000105, 1958701000006105,
    185317003,        788007007,       283511000000105,
    182836005,        182888003,       184103008,
    498521000006103,  386472008
  ]) AS snomed_code
),

-- LEAKAGE FIX (Task #21): defensive filter — even though non-cancer cohort
-- shouldn't have these treatment/pathway codes, mirror the positive cohort
-- filter for symmetric Scores output.
breast_pathway_codes AS (
  SELECT snomed_code FROM UNNEST([
    -- 874291000000100 (QCancer breast cancer risk) — RE-EXCLUDED 2026-06-01
    907341000006100, 429087003, 874291000000100, 134405005,
    276341000000100, 185270007, 394592004,
    189336000, 109889007, 109888004,
    69031006, 394911000, 442343008, 64368001, 274957008,
    429400009, 384723003, 172043006, 237371007, 392021009,
    428571003, 392023007, 27865001, 79544006,
    116334007, 387736007, 432550005, 122548005, 171176006,
    168531007, 108290001, 168534004, 168750009, 716872004
  ]) AS snomed_code
),

total_negative_cohort AS (
  SELECT COUNT(DISTINCT patient_guid) AS total_n
  FROM non_cancer_patients
),

feature_events AS (
  SELECT
    ncp.patient_guid,
    ncp.sex,
    CAST(FLOOR(DATE_DIFF(ncp.pseudo_index_date, ncp.date_of_birth, DAY) / 365.25) AS INT64)
                                                               AS age_at_index,
    ncp.pseudo_index_date,
    mh.effective_date                                          AS event_date,
    CAST(FLOOR(DATE_DIFF(mh.effective_date, ncp.date_of_birth, DAY) / 365.25) AS INT64)
                                                               AS event_age,
    mh.snomed_c_t_concept_id,
    mh.term,
    mh.observation_type,
    mh.associated_text,
    mh.value
  FROM non_cancer_patients ncp
  INNER JOIN medical_history mh
    ON mh.patient_guid = ncp.patient_guid
  CROSS JOIN params p
  WHERE mh.effective_date >= DATE_SUB(ncp.pseudo_index_date, INTERVAL p.months_lookback_start MONTH)
    AND mh.effective_date <  DATE_SUB(ncp.pseudo_index_date, INTERVAL p.months_lookback_end   MONTH)
    AND NOT EXISTS (
      SELECT 1 FROM `prj-cts-ai-dev-sp.EMIS_MED_CODES.Dim_Cancer_Codes` dcc
      WHERE SAFE_CAST(dcc.SNOMED_CODE AS INT64) = mh.snomed_c_t_concept_id
    )
    -- LEAKAGE FIX (Task #21): also filter cancer-pathway codes
    AND NOT EXISTS (
      SELECT 1 FROM breast_pathway_codes bpc
      WHERE bpc.snomed_code = mh.snomed_c_t_concept_id
    )
    AND NOT EXISTS (
      SELECT 1 FROM excluded_snomeds ex
      WHERE ex.snomed_code = mh.snomed_c_t_concept_id
    )
    AND COALESCE(mh.observation_type, '') NOT IN ('Immunisation')
),

deduped_events AS (
  SELECT *
  FROM feature_events
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY patient_guid, snomed_c_t_concept_id, event_date
    ORDER BY event_date
  ) = 1
)

SELECT
  snomed_c_t_concept_id                                                         AS snomed_id,
  MAX(term)                                                                     AS term,
  APPROX_QUANTILES(age_at_index, 2)[OFFSET(1)]                                  AS median_index_age,
  APPROX_QUANTILES(event_age, 2)[OFFSET(1)]                                     AS median_snomed_age,
  COUNT(DISTINCT patient_guid)                                                  AS n_patient_count,
  (SELECT total_n FROM total_negative_cohort)                                   AS n_patient_count_total,
  ROUND(
    CAST(COUNT(DISTINCT patient_guid) AS FLOAT64)
    / NULLIF((SELECT total_n FROM total_negative_cohort), 0),
    6
  )                                                                             AS prevalence,
  COUNT(*)                                                                      AS total_event_count,
  ROUND(CAST(COUNT(*) AS FLOAT64) / NULLIF(COUNT(DISTINCT patient_guid), 0), 2)
                                                                                AS avg_events_per_patient,
  AVG(SAFE_CAST(IF(TRIM(value) = '' OR value IS NULL, NULL, value) AS FLOAT64))
                                                                                AS avg_value,
  APPROX_QUANTILES(SAFE_CAST(IF(TRIM(value) = '' OR value IS NULL, NULL, value) AS FLOAT64), 2)[OFFSET(1)]
                                                                                AS median_value,
  STDDEV(SAFE_CAST(IF(TRIM(value) = '' OR value IS NULL, NULL, value) AS FLOAT64))
                                                                                AS std_value,
  APPROX_TOP_COUNT(SAFE_CAST(IF(TRIM(value) = '' OR value IS NULL, NULL, value) AS FLOAT64), 1)[OFFSET(0)].value
                                                                                AS freq_value
FROM deduped_events
GROUP BY snomed_c_t_concept_id
HAVING COUNT(DISTINCT patient_guid) >= (SELECT min_patient_threshold FROM params)
ORDER BY n_patient_count DESC;
