WITH params AS (
  SELECT
    DATE '1950-01-01'  AS mh_start,
    DATE '2026-02-25'  AS mh_end,
    13                 AS months_lookback_start,
    1                  AS months_lookback_end,
    70                 AS pseudo_index_age
),

diagnostic_codes AS (
  SELECT
    TRY_TO_NUMBER(RAW_RECORDS:code_id::string)                       AS code_id,
    PRACTICE_ID                                                      AS source_practice_code,
    RAW_RECORDS:term::string                                         AS term,
    TRY_TO_NUMBER(RAW_RECORDS:snomed_c_t_concept_id::string)         AS snomed_c_t_concept_id
  FROM CTSUK_BULK.RAW.CODING_CLINICALCODE
  WHERE RAW_RECORDS:snomed_c_t_concept_id IS NOT NULL
    AND RAW_RECORDS:term IS NOT NULL
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

medical_history AS (
  SELECT
    co.RAW_RECORDS:patient_guid::string                          AS patient_guid,
    TRY_TO_DATE(co.RAW_RECORDS:effective_date::string)           AS effective_date,
    co.RAW_RECORDS:observation_type::string                      AS observation_type,
    dc.term,
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

/* ── Cancer patients (bladder) ── */
cancer_patients AS (
  SELECT
    mh.patient_guid,
    MIN(mh.effective_date) AS date_of_diagnosis
  FROM medical_history mh
  INNER JOIN analytics.cts.dim_cancer_codes dcc
    ON mh.snomed_c_t_concept_id = dcc.snomed_code
  WHERE LOWER(dcc.cancer_id) NOT LIKE '%disease%'
    AND LOWER(dcc.cancer_id) LIKE '%bladder%'
  GROUP BY 1
  HAVING MIN(mh.effective_date) IS NOT NULL
),

/* ── All cancer patients (for negative cohort exclusion) ── */
all_cancer_patients AS (
  SELECT DISTINCT mh.patient_guid
  FROM medical_history mh
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
  INNER JOIN medical_history mh ON mh.patient_guid = p.patient_guid
  CROSS JOIN params param
  WHERE DATEDIFF(year, p.date_of_birth, param.mh_end) > 18
    AND NOT EXISTS (SELECT 1 FROM all_cancer_patients cp WHERE cp.patient_guid = p.patient_guid)
    AND NOT EXISTS (
      SELECT 1 FROM medical_history pmh
      WHERE pmh.patient_guid = p.patient_guid
        AND pmh.snomed_c_t_concept_id = 1403151000000103
    )
    AND DATEADD(year, param.pseudo_index_age, p.date_of_birth) <= param.mh_end
    AND DATEADD(year, param.pseudo_index_age, p.date_of_birth) >= param.mh_start
),

/* ── Excluded SNOMEDs ── */
excluded_snomeds AS (
  SELECT code AS snomed_code FROM EXPERIMENTS.MARTA_S.ETHNICITY_SNOMED_CODES
  UNION ALL
  SELECT column1 FROM (VALUES
    (1572871000006101), (279991000000102), (428481002),
    (979851000000101),  (887641000000105), (1958701000006105),
    (185317003),        (788007007),       (283511000000105),
    (182836005),        (182888003),       (184103008),
    (498521000006103),  (386472008)
  )
),

/* ══════════════════════════════════════════════════════════════
   POSITIVE COHORT: per-patient SNOMED counts in [-13m, -1m]
   ══════════════════════════════════════════════════════════════ */
positive_patient_features AS (
  SELECT
    cp.patient_guid,
    1                                                          AS label,
    pp.sex,
    DATEDIFF(year, pp.date_of_birth, cp.date_of_diagnosis)    AS age_at_index,
    mh.snomed_c_t_concept_id,
    mh.term,
    COUNT(*)                                                   AS event_count
  FROM cancer_patients cp
  INNER JOIN medical_history mh ON mh.patient_guid = cp.patient_guid
  INNER JOIN patients pp ON pp.patient_guid = cp.patient_guid
  CROSS JOIN params p
  WHERE mh.effective_date >= DATEADD(month, -p.months_lookback_start, cp.date_of_diagnosis)
    AND mh.effective_date <  DATEADD(month, -p.months_lookback_end,   cp.date_of_diagnosis)
    AND NOT EXISTS (SELECT 1 FROM analytics.cts.dim_cancer_codes dcc WHERE dcc.snomed_code = mh.snomed_c_t_concept_id)
    AND NOT EXISTS (SELECT 1 FROM excluded_snomeds ex WHERE ex.snomed_code = mh.snomed_c_t_concept_id)
    AND COALESCE(mh.observation_type, '') NOT IN ('Immunisation')
  GROUP BY 1, 2, 3, 4, 5, 6
),

/* ══════════════════════════════════════════════════════════════
   NEGATIVE COHORT: per-patient SNOMED counts in [-13m, -1m]
   ══════════════════════════════════════════════════════════════ */
negative_patient_features AS (
  SELECT
    ncp.patient_guid,
    0                                                          AS label,
    pp.sex,
    DATEDIFF(year, ncp.date_of_birth, ncp.pseudo_index_date)  AS age_at_index,
    mh.snomed_c_t_concept_id,
    mh.term,
    COUNT(*)                                                   AS event_count
  FROM non_cancer_patients ncp
  INNER JOIN medical_history mh ON mh.patient_guid = ncp.patient_guid
  INNER JOIN patients pp ON pp.patient_guid = ncp.patient_guid
  CROSS JOIN params p
  WHERE mh.effective_date >= DATEADD(month, -p.months_lookback_start, ncp.pseudo_index_date)
    AND mh.effective_date <  DATEADD(month, -p.months_lookback_end,   ncp.pseudo_index_date)
    AND NOT EXISTS (SELECT 1 FROM analytics.cts.dim_cancer_codes dcc WHERE dcc.snomed_code = mh.snomed_c_t_concept_id)
    AND NOT EXISTS (SELECT 1 FROM excluded_snomeds ex WHERE ex.snomed_code = mh.snomed_c_t_concept_id)
    AND COALESCE(mh.observation_type, '') NOT IN ('Immunisation')
  GROUP BY 1, 2, 3, 4, 5, 6
),

/* ══════════════════════════════════════════════════════════════
   COMBINED: all patients with their per-SNOMED event counts
   ══════════════════════════════════════════════════════════════ */
all_patient_features AS (
  SELECT * FROM positive_patient_features
  UNION ALL
  SELECT * FROM negative_patient_features
)

/* ════════════════════════════════════════════════════════════════
   OUTPUT: Long format — feed this to Python for pivoting
   (patient_guid, label, sex, age, snomed_term, event_count)
   ════════════════════════════════════════════════════════════════ */
SELECT
  patient_guid,
  label,
  sex,
  age_at_index,
  term                AS feature_name,
  snomed_c_t_concept_id AS feature_code,
  event_count
FROM all_patient_features
ORDER BY patient_guid, feature_name;
