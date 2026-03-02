/*╔═══════════════════════════════��══════════════════════════════════╗
  ║  PRODUCTION — BLADDER CANCER                                    ║
  ║  Phase 1: Top Medications (NEGATIVE COHORT)                    ║
  ║  Version: 1.0                                                   ║
  ║  Dates:   1950-01-01 → 2026-02-25                              ║
  ║  Window:  [-13 months, -1 month] before age 70 (pseudo index)  ║
  ╚══════════════════════════════════════════════════════════════════╝*/

WITH params AS (
  SELECT
    DATE '1950-01-01'  AS mh_start,
    DATE '2026-02-25'  AS mh_end,
    13                 AS months_lookback_start,
    1                  AS months_lookback_end,
    70                 AS pseudo_index_age,
    5                  AS min_patient_threshold
),

diagnostic_codes AS (
  SELECT
    TRY_TO_NUMBER(RAW_RECORDS:code_id::string)                       AS code_id,
    PRACTICE_ID                                                      AS source_practice_code,
    TRY_TO_NUMBER(RAW_RECORDS:snomed_c_t_concept_id::string)         AS snomed_c_t_concept_id
  FROM CTSUK_BULK.RAW.CODING_CLINICALCODE
  WHERE RAW_RECORDS:snomed_c_t_concept_id IS NOT NULL
    AND TRY_TO_NUMBER(RAW_RECORDS:code_id::string) IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRY_TO_NUMBER(RAW_RECORDS:code_id::string), PRACTICE_ID
    ORDER BY TO_DATE(
      REGEXP_SUBSTR(file_name, '/([0-9]{8})/', 1, 1, 'e', 1),
      'YYYYMMDD'
    ) DESC
  ) = 1
),

patients AS (
  SELECT patient_guid, sex, date_of_birth
  FROM CTSUK_BULK.STAGING.PATIENTS_EMIS
  WHERE patient_guid IS NOT NULL
    AND date_of_birth IS NOT NULL
),

medical_history_for_exclusion AS (
  SELECT
    co.RAW_RECORDS:patient_guid::string                          AS patient_guid,
    TRY_TO_DATE(co.RAW_RECORDS:effective_date::string)           AS effective_date,
    dc.snomed_c_t_concept_id
  FROM CTSUK_BULK.RAW.CARERECORD_OBSERVATION AS co
  INNER JOIN diagnostic_codes AS dc
    ON TRY_TO_NUMBER(co.RAW_RECORDS:code_id::string) = dc.code_id
   AND co.practice_id = dc.source_practice_code
  CROSS JOIN params p
  WHERE TRY_TO_DATE(co.RAW_RECORDS:effective_date::string) IS NOT NULL
    AND TRY_TO_DATE(co.RAW_RECORDS:effective_date::string) >= p.mh_start
    AND TRY_TO_DATE(co.RAW_RECORDS:effective_date::string) <  p.mh_end
    AND dc.snomed_c_t_concept_id IS NOT NULL
),

cancer_patients AS (
  SELECT DISTINCT mh.patient_guid
  FROM medical_history_for_exclusion AS mh
  INNER JOIN analytics.cts.dim_cancer_codes AS dcc
    ON mh.snomed_c_t_concept_id = dcc.snomed_code
  WHERE LOWER(dcc.cancer_id) NOT LIKE '%disease%'
),

non_cancer_patients AS (
  SELECT DISTINCT
    p.patient_guid,
    p.date_of_birth
  FROM patients p
  INNER JOIN medical_history_for_exclusion mh
    ON mh.patient_guid = p.patient_guid
  CROSS JOIN params param
  WHERE
    DATEDIFF(year, p.date_of_birth, param.mh_end) > 18
    AND NOT EXISTS (
      SELECT 1 FROM cancer_patients cp
      WHERE cp.patient_guid = p.patient_guid
    )
    AND NOT EXISTS (
      SELECT 1 FROM medical_history_for_exclusion pmh
      WHERE pmh.patient_guid = p.patient_guid
        AND pmh.snomed_c_t_concept_id = 1403151000000103
    )
    AND DATEADD(year, param.pseudo_index_age, p.date_of_birth) <= param.mh_end
    AND DATEADD(year, param.pseudo_index_age, p.date_of_birth) >= param.mh_start
),

/* ── Medication pipeline (deduplicated) ── */
prescribing_drugrecords_emis AS (
  SELECT
    TRIM(raw_records:drug_record_guid)::string  AS drug_record_guid,
    TRIM(raw_records:prescription_type)::string AS prescription_type,
    TRIM(raw_records:patient_guid)::string      AS patient_guid,
    practice_id                                 AS source_practice_code,
    TRIM(raw_records:deleted)::string           AS deleted
  FROM CTSUK_BULK.RAW.PRESCRIBING_DRUGRECORD
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRIM(raw_records:drug_record_guid)
    ORDER BY TO_DATE(
      REGEXP_SUBSTR(file_name, '/([0-9]{8})/', 1, 1, 'e', 1),
      'YYYYMMDD'
    ) DESC
  ) = 1
),

prescribing_issuerecords_emis AS (
  SELECT
    TRIM(raw_records:drug_record_guid)::string         AS drug_record_guid,
    TRY_TO_DATE(TRIM(raw_records:effective_date))      AS effective_date,
    TRY_TO_NUMBER(TRIM(raw_records:code_id))           AS code_id,
    TRY_TO_NUMBER(TRIM(raw_records:course_duration_in_days)) AS course_duration_in_days,
    TRIM(raw_records:deleted)::string                  AS deleted
  FROM CTSUK_BULK.RAW.PRESCRIBING_ISSUERECORD
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRIM(raw_records:drug_record_guid)
    ORDER BY TO_DATE(
      REGEXP_SUBSTR(file_name, '/([0-9]{8})/', 1, 1, 'e', 1),
      'YYYYMMDD'
    ) DESC
  ) = 1
),

codes_emis AS (
  SELECT
    TRY_TO_NUMBER(TRIM(raw_records:code_id))             AS code_id,
    TRY_TO_NUMBER(TRIM(raw_records:dmd_product_code_id)) AS dmd_product_code_id,
    TRIM(raw_records:term)::string                       AS term,
    practice_id                                          AS source_practice_code
  FROM CTSUK_BULK.RAW.CODING_DRUGCODE
  WHERE TRIM(raw_records:dmd_product_code_id) IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRY_TO_NUMBER(TRIM(raw_records:code_id)), practice_id
    ORDER BY TO_DATE(
      REGEXP_SUBSTR(file_name, '/([0-9]{8})/', 1, 1, 'e', 1),
      'YYYYMMDD'
    ) DESC
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
  WHERE dr.deleted = 'false'
    AND ir.deleted = 'false'
    AND ir.effective_date IS NOT NULL
    AND ce.term IS NOT NULL
),

feature_events AS (
  SELECT
    ncp.patient_guid,
    pp.sex,
    DATEDIFF(year, ncp.date_of_birth, fm.effective_date)       AS event_age,
    fm.effective_date                                          AS event_date,
    fm.code_id                                                 AS med_code_id,
    fm.drug_term,
    fm.duration,
    fm.prescription_type
  FROM non_cancer_patients ncp
  INNER JOIN final_medication fm
    ON fm.patient_guid = ncp.patient_guid
  INNER JOIN patients pp
    ON pp.patient_guid = ncp.patient_guid
  CROSS JOIN params p
  WHERE
    fm.effective_date >= DATEADD(month, -p.months_lookback_start,
                           DATEADD(year, p.pseudo_index_age, ncp.date_of_birth))
    AND fm.effective_date < DATEADD(month, -p.months_lookback_end,
                              DATEADD(year, p.pseudo_index_age, ncp.date_of_birth))
),

deduped_events AS (
  SELECT *
  FROM feature_events
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY patient_guid, med_code_id, event_date
    ORDER BY event_date
  ) = 1
)

/* ════════════════════════════════════════════════════════════════ */
SELECT
  drug_term                                                    AS term,
  MAX(med_code_id)                                             AS med_code_id,
  MEDIAN(event_age)                                            AS median_event_age,
  COUNT(DISTINCT patient_guid)                                 AS n_patient_count,
  (SELECT COUNT(DISTINCT patient_guid) FROM deduped_events)    AS n_patient_count_total,
  ROUND(
    COUNT(DISTINCT patient_guid)::FLOAT
    / NULLIF((SELECT COUNT(DISTINCT patient_guid) FROM deduped_events), 0),
    6
  )                                                            AS prevalence,
  COUNT(*)                                                     AS total_prescriptions,
  ROUND(COUNT(*)::FLOAT / NULLIF(COUNT(DISTINCT patient_guid), 0), 2)
                                                               AS avg_prescriptions_per_patient,
  AVG(duration)                                                AS avg_duration,
  MEDIAN(duration)                                             AS median_duration,
  STDDEV(duration)                                             AS std_duration
FROM deduped_events
GROUP BY drug_term
HAVING COUNT(DISTINCT patient_guid) >= (SELECT min_patient_threshold FROM params)
ORDER BY n_patient_count DESC;
