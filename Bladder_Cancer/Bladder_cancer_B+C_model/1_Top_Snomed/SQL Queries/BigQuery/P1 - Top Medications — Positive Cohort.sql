/*╔══════════════════════════════════════════════════════════════════╗
  ║  PRODUCTION — BLADDER  (BigQuery dialect)                     ║
  ║  Phase 1.1: Top Medications (POSITIVE COHORT)                  ║
  ║  Version: 1.0                                                   ║
  ║  Dates:   1950-01-01 → 2026-03-11                              ║
  ║  Window:  [-15 months, -1 month] before diagnosis             ║
  ║                                                                 ║
  ║  Cohort:                                                        ║
  ║    - Both sexes (no sex restriction)                          ║
  ║    - Adults (>18 at diagnosis)                                  ║
  ║    - Diagnosed with bladder cancer                             ║
  ║                                                                 ║
  ║  NOTE: Cancer treatment medications are NOT excluded here.      ║
  ║  Leakage filtering is handled during feature selection          ║
  ║  (Phase 1.3 / 1.4 clinical review).                            ║
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
    SAFE_CAST(snomed_c_t_concept_id AS INT64)    AS snomed_c_t_concept_id,
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

medical_history_for_dx AS (
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

cancer_patients AS (
  SELECT mh.patient_guid, MIN(mh.effective_date) AS date_of_diagnosis
  FROM medical_history_for_dx AS mh
  INNER JOIN `prj-cts-ai-dev-sp.EMIS_MED_CODES.Dim_Cancer_Codes` AS dcc
    ON mh.snomed_c_t_concept_id = SAFE_CAST(dcc.SNOMED_CODE AS INT64)
  WHERE LOWER(dcc.cancer_id) NOT LIKE '%disease%'
    AND LOWER(dcc.cancer_id) LIKE '%bladder%'
  GROUP BY 1
  HAVING MIN(mh.effective_date) IS NOT NULL
),

total_positive_cohort AS (
  SELECT COUNT(DISTINCT cp.patient_guid) AS total_n
  FROM cancer_patients cp
  INNER JOIN patients pp ON pp.patient_guid = cp.patient_guid
  WHERE FLOOR(DATE_DIFF(cp.date_of_diagnosis, pp.date_of_birth, DAY) / 365.25) > 18
),

prescribing_drugrecords_emis AS (
  SELECT
    TRIM(drug_record_guid)   AS drug_record_guid,
    TRIM(prescription_type)  AS prescription_type,
    TRIM(patient_guid)       AS patient_guid,
    source_practice_code,
    deleted,
    PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) AS file_date
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Prescribing_DrugRecord`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRIM(drug_record_guid)
    ORDER BY PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) DESC
  ) = 1
),

prescribing_issuerecords_emis AS (
  SELECT
    TRIM(drug_record_guid)                              AS drug_record_guid,
    PARSE_DATE('%Y-%m-%d', TRIM(effective_date))        AS effective_date,
    code_id,                                              -- INT64 in BQ schema; no TRIM/CAST needed
    course_duration_in_days,                              -- INT64 in BQ schema; no TRIM/CAST needed
    deleted,
    PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) AS file_date
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Prescribing_IssueRecord`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRIM(drug_record_guid)
    ORDER BY PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) DESC
  ) = 1
),

codes_emis AS (
  SELECT
    SAFE_CAST(code_id AS INT64)             AS code_id,
    SAFE_CAST(dmd_product_code_id AS INT64) AS dmd_product_code_id,
    term,
    source_practice_code,
    PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) AS file_date
  FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Coding_DrugCode`
  WHERE dmd_product_code_id IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY code_id, source_practice_code
    ORDER BY PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) DESC
  ) = 1
),

final_medication AS (
  SELECT
    dr.drug_record_guid,
    ce.dmd_product_code_id                        AS code_id,
    dr.patient_guid,
    ir.effective_date,
    dr.source_practice_code,
    ir.course_duration_in_days                    AS duration,
    ce.term                                       AS drug_term,
    dr.prescription_type
  FROM prescribing_drugrecords_emis dr
  INNER JOIN prescribing_issuerecords_emis ir
    ON dr.drug_record_guid = ir.drug_record_guid
  INNER JOIN codes_emis ce
    ON ir.code_id = ce.code_id
   AND dr.source_practice_code = ce.source_practice_code
  WHERE COALESCE(dr.deleted, false) = false
    AND COALESCE(ir.deleted, false) = false
    AND ir.effective_date IS NOT NULL
    AND ce.term IS NOT NULL
),

feature_events AS (
  SELECT
    cp.patient_guid,
    pp.sex,
    CAST(FLOOR(DATE_DIFF(cp.date_of_diagnosis, pp.date_of_birth, DAY) / 365.25) AS INT64)
                                                               AS diagnosis_age,
    cp.date_of_diagnosis,
    fm.effective_date                                          AS event_date,
    CAST(FLOOR(DATE_DIFF(fm.effective_date, pp.date_of_birth, DAY) / 365.25) AS INT64)
                                                               AS event_age,
    fm.code_id                                                 AS med_code_id,
    fm.drug_term,
    fm.duration,
    fm.prescription_type
  FROM cancer_patients cp
  INNER JOIN final_medication fm
    ON UPPER(REGEXP_REPLACE(TRIM(fm.patient_guid), r'[{}]', '')) =
       UPPER(REGEXP_REPLACE(TRIM(CAST(cp.patient_guid AS STRING)), r'[{}]', ''))
  INNER JOIN patients pp
    ON pp.patient_guid = cp.patient_guid
  CROSS JOIN params p
  WHERE FLOOR(DATE_DIFF(cp.date_of_diagnosis, pp.date_of_birth, DAY) / 365.25) > 18
    AND fm.effective_date >= DATE_SUB(cp.date_of_diagnosis, INTERVAL p.months_lookback_start MONTH)
    AND fm.effective_date <  DATE_SUB(cp.date_of_diagnosis, INTERVAL p.months_lookback_end   MONTH)
),

deduped_events AS (
  SELECT *
  FROM feature_events
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY patient_guid, med_code_id, event_date
    ORDER BY event_date
  ) = 1
)

SELECT
  med_code_id,
  MAX(drug_term)                                                                AS term,
  APPROX_QUANTILES(diagnosis_age, 2)[OFFSET(1)]                                 AS median_diagnosis_age,
  APPROX_QUANTILES(event_age, 2)[OFFSET(1)]                                     AS median_event_age,
  COUNT(DISTINCT patient_guid)                                                  AS n_patient_count,
  (SELECT total_n FROM total_positive_cohort)                                   AS n_patient_count_total,
  ROUND(
    CAST(COUNT(DISTINCT patient_guid) AS FLOAT64)
    / NULLIF((SELECT total_n FROM total_positive_cohort), 0),
    6
  )                                                                             AS prevalence,
  COUNT(*)                                                                      AS total_prescriptions,
  ROUND(CAST(COUNT(*) AS FLOAT64) / NULLIF(COUNT(DISTINCT patient_guid), 0), 2)
                                                                                AS avg_prescriptions_per_patient,
  AVG(duration)                                                                 AS avg_duration,
  APPROX_QUANTILES(duration, 2)[OFFSET(1)]                                      AS median_duration,
  STDDEV(duration)                                                              AS std_duration
FROM deduped_events
GROUP BY med_code_id
HAVING COUNT(DISTINCT patient_guid) >= (SELECT min_patient_threshold FROM params)
ORDER BY n_patient_count DESC;
