WITH params AS (
  SELECT
    DATE '1950-01-01'  AS mh_start,
    DATE '2026-02-25'  AS mh_end,
    13                 AS months_lookback_start,
    1                  AS months_lookback_end,
    70                 AS pseudo_index_age
),

/* ── Minimal observation pipeline just for cancer identification ── */
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
  WHERE patient_guid IS NOT NULL AND date_of_birth IS NOT NULL
),

medical_history_for_dx AS (
  SELECT
    co.RAW_RECORDS:patient_guid::string        AS patient_guid,
    TRY_TO_DATE(co.RAW_RECORDS:effective_date::string) AS effective_date,
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

/* ── Bladder cancer patients ── */
cancer_patients AS (
  SELECT
    mh.patient_guid,
    MIN(mh.effective_date) AS date_of_diagnosis
  FROM medical_history_for_dx mh
  INNER JOIN analytics.cts.dim_cancer_codes dcc
    ON mh.snomed_c_t_concept_id = dcc.snomed_code
  WHERE LOWER(dcc.cancer_id) NOT LIKE '%disease%'
    AND LOWER(dcc.cancer_id) LIKE '%bladder%'
  GROUP BY 1
  HAVING MIN(mh.effective_date) IS NOT NULL
),

/* ── All cancer patients (for negative exclusion) ── */
all_cancer_patients AS (
  SELECT DISTINCT mh.patient_guid
  FROM medical_history_for_dx mh
  INNER JOIN analytics.cts.dim_cancer_codes dcc
    ON mh.snomed_c_t_concept_id = dcc.snomed_code
  WHERE LOWER(dcc.cancer_id) NOT LIKE '%disease%'
),

/* ── Non-cancer patients ── */
non_cancer_patients AS (
  SELECT DISTINCT
    p.patient_guid,
    p.date_of_birth,
    DATEADD(year, param.pseudo_index_age, p.date_of_birth) AS pseudo_index_date
  FROM patients p
  INNER JOIN medical_history_for_dx mh ON mh.patient_guid = p.patient_guid
  CROSS JOIN params param
  WHERE DATEDIFF(year, p.date_of_birth, param.mh_end) > 18
    AND NOT EXISTS (SELECT 1 FROM all_cancer_patients cp WHERE cp.patient_guid = p.patient_guid)
    AND NOT EXISTS (
      SELECT 1 FROM medical_history_for_dx pmh
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
    ce.term                                       AS drug_term
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

/* ══════════════════════════════════════════════════════════════
   POSITIVE: per-patient medication counts in [-13m, -1m]
   ══════════════════════════════════════════════════════════════ */
positive_patient_meds AS (
  SELECT
    cp.patient_guid,
    1                                                          AS label,
    pp.sex,
    DATEDIFF(year, pp.date_of_birth, cp.date_of_diagnosis)    AS age_at_index,
    fm.drug_term                                               AS feature_name,
    fm.code_id                                                 AS feature_code,
    COUNT(*)                                                   AS event_count
  FROM cancer_patients cp
  INNER JOIN final_medication fm ON fm.patient_guid = cp.patient_guid
  INNER JOIN patients pp ON pp.patient_guid = cp.patient_guid
  CROSS JOIN params p
  WHERE fm.effective_date >= DATEADD(month, -p.months_lookback_start, cp.date_of_diagnosis)
    AND fm.effective_date <  DATEADD(month, -p.months_lookback_end,   cp.date_of_diagnosis)
  GROUP BY 1, 2, 3, 4, 5, 6
),

/* ══════════════════════════════════════════════════════════════
   NEGATIVE: per-patient medication counts in [-13m, -1m]
   ══════════════════════════════════════════════════════════════ */
negative_patient_meds AS (
  SELECT
    ncp.patient_guid,
    0                                                          AS label,
    pp.sex,
    DATEDIFF(year, ncp.date_of_birth, ncp.pseudo_index_date)  AS age_at_index,
    fm.drug_term                                               AS feature_name,
    fm.code_id                                                 AS feature_code,
    COUNT(*)                                                   AS event_count
  FROM non_cancer_patients ncp
  INNER JOIN final_medication fm ON fm.patient_guid = ncp.patient_guid
  INNER JOIN patients pp ON pp.patient_guid = ncp.patient_guid
  CROSS JOIN params p
  WHERE fm.effective_date >= DATEADD(month, -p.months_lookback_start, ncp.pseudo_index_date)
    AND fm.effective_date <  DATEADD(month, -p.months_lookback_end,   ncp.pseudo_index_date)
  GROUP BY 1, 2, 3, 4, 5, 6
)

/* ════════════════════════════════════════════════════════════════
   OUTPUT: Long format medication matrix
   ════════════════════════════════════════════════════════════════ */
SELECT * FROM positive_patient_meds
UNION ALL
SELECT * FROM negative_patient_meds
ORDER BY patient_guid, feature_name;
