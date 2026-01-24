WITH diagnostic_codes AS (
  SELECT
    TRY_TO_NUMBER(RAW_RECORDS: code_id:: string)          AS code_id,
    PRACTICE_ID                                         AS source_practice_code,
    RAW_RECORDS:term                                    AS term,
    RAW_RECORDS:snomed_c_t_concept_id                   AS snomed_c_t_concept_id,
    RAW_RECORDS:snomed_c_t_description_id               AS snomed_c_t_description_id
  FROM CTSUK_BULK. RAW. CODING_CLINICALCODE
  WHERE snomed_c_t_concept_id IS NOT NULL
    AND term IS NOT NULL
),

patients AS (
    SELECT
        patient_guid,
        sex,
        date_of_birth
    FROM CTSUK_BULK. STAGING.PATIENTS_EMIS
),

medical_history AS (
  SELECT
    co.practice_id,
    co.RAW_RECORDS:code_id:: string                     AS code_id,
    co.RAW_RECORDS:patient_guid                        AS patient_guid,
    co.RAW_RECORDS:consultation_guid                   AS consultation_guid,
    TRY_TO_DATE(co.RAW_RECORDS:effective_date::string) AS effective_date,
    TRY_TO_DATE(co.RAW_RECORDS:entered_date::string)   AS entered_date,
    co.RAW_RECORDS:associated_text                     AS associated_text,
    co.RAW_RECORDS:observation_type                    AS observation_type,
    co.RAW_RECORDS:value                               AS value,
    co.RAW_RECORDS: ENTERED_BY_USER_IN_ROLE_GUID        AS ENTERED_BY_USER_IN_ROLE_GUID,
    dc.term,
    TRY_TO_NUMBER(dc.snomed_c_t_concept_id::string)    AS snomed_c_t_concept_id,
    dc.snomed_c_t_description_id,
    dc.source_practice_code
  FROM CTSUK_BULK.RAW.CARERECORD_OBSERVATION AS co
  LEFT JOIN diagnostic_codes AS dc
    ON TRY_TO_NUMBER(co.RAW_RECORDS:code_id::string) = TRY_TO_NUMBER(dc.code_id)
   AND co.practice_id = dc.source_practice_code
  LEFT JOIN EXPERIMENTS.MARTA_S. ETHNICITY_SNOMED_CODES esc
    ON dc.snomed_c_t_concept_id = esc.code
  WHERE effective_date IS NOT NULL
    AND effective_date >= DATE '2000-01-01'
    AND effective_date <  DATE '2024-06-01'
),

/* Patients with ANY cancer diagnosis (to exclude them from negative cohort) */
cancer_patients AS (
  SELECT
    mh.practice_id,
    mh. patient_guid,
    dcc.cancer_id,
    MIN(mh.effective_date) AS date_of_diagnosis
  FROM medical_history AS mh
  JOIN analytics. cts. dim_cancer_codes AS dcc
    ON mh.snomed_c_t_concept_id = dcc.snomed_code
  WHERE mh.effective_date IS NOT NULL
    AND LOWER(dcc.cancer_id) NOT LIKE '%disease%'
    -- NO FILTER on cancer type - we want to exclude ALL cancer patients
  GROUP BY 1,2,3
),

/* Same trimmed window used for features/filters */
lookup_medical_history AS (
  SELECT *
  FROM medical_history
  WHERE effective_date >= DATE '2000-01-01'
    AND effective_date <  DATE '2024-06-01'
),

/* Males aged 40-95 (as of 2024-06-01), exclude cancer and palliative care */
base_cohort AS (
  SELECT DISTINCT 
    omh.patient_guid,
    p.date_of_birth
  FROM lookup_medical_history AS omh
  LEFT JOIN patients AS p 
    ON p.patient_guid = omh.patient_guid
  WHERE p.sex = 'M'                                                     -- Males only
    AND DATEDIFF(year, p.date_of_birth, DATE '2024-06-01') >= 40        -- Age 40+ as of June 2024
    AND DATEDIFF(year, p.date_of_birth, DATE '2024-06-01') <= 95        -- Age <= 95 (exclude errors)
    AND NOT EXISTS (
      SELECT 1
      FROM cancer_patients AS cp
      WHERE cp.patient_guid = omh.patient_guid
        AND cp.date_of_diagnosis < DATE '2025-06-01'
    )
    AND NOT EXISTS (
      SELECT 1
      FROM lookup_medical_history AS pmh
      WHERE pmh.patient_guid = omh. patient_guid
        AND pmh.snomed_c_t_concept_id = '1403151000000103' -- palliative care
    )
),

/* Final non-cancer set:  base cohort patients who never appear in cancer_patients */
non_cancer_patients AS (
  SELECT 
    bc.patient_guid, 
    bc.date_of_birth
  FROM base_cohort bc
  LEFT JOIN cancer_patients cp 
    ON cp.patient_guid = bc.patient_guid
  WHERE cp.patient_guid IS NULL
),

/* === Medication pipeline CTEs (EMIS prescribing tables) === */
prescribing_drugrecords_emis AS (
  SELECT
    TO_DATE(
      REGEXP_SUBSTR(file_name, '/([0-9]{8})/', 1, 1, 'e', 1),
      'YYYYMMDD'
    )                                   AS file_date,
    TRIM(raw_records: drug_record_guid)  AS drug_record_guid,
    TRIM(raw_records: prescription_type) AS prescription_type,
    TRIM(raw_records:patient_guid)      AS patient_guid,
    practice_id                         AS source_practice_code,
    TRIM(raw_records:deleted)           AS deleted
  FROM CTSUK_BULK.RAW.PRESCRIBING_DRUGRECORD
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY drug_record_guid
    ORDER BY file_date DESC
  ) = 1
),

prescribing_issuerecords_emis AS (
  SELECT
    TO_DATE(
      REGEXP_SUBSTR(file_name, '/([0-9]{8})/', 1, 1, 'e', 1),
      'YYYYMMDD'
    )                                           AS file_date,
    TRIM(raw_records:issue_record_guid)        AS issue_record_guid,
    TRIM(raw_records:drug_record_guid)         AS drug_record_guid,
    TO_DATE(TRIM(raw_records:entered_date))    AS entered_date,
    TO_DATE(TRIM(raw_records:effective_date))  AS effective_date,
    TRY_TO_NUMBER(TRIM(raw_records:code_id))   AS code_id,
    TRY_TO_NUMBER(
      TRIM(raw_records: course_duration_in_days)
    )                                          AS course_duration_in_days,
    TRIM(raw_records:deleted)                  AS deleted
  FROM CTSUK_BULK.RAW. PRESCRIBING_ISSUERECORD
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY drug_record_guid
    ORDER BY file_date DESC
  ) = 1
),

codes_emis AS (
  SELECT
    TO_DATE(
      REGEXP_SUBSTR(file_name, '/([0-9]{8})/', 1, 1, 'e', 1),
      'YYYYMMDD'
    )                                            AS file_date,
    TRY_TO_NUMBER(TRIM(raw_records:code_id))             AS code_id,
    TRY_TO_NUMBER(TRIM(raw_records: dmd_product_code_id)) AS dmd_product_code_id,
    TRIM(raw_records:term)                      AS term,
    practice_id                                 AS source_practice_code
  FROM CTSUK_BULK.RAW.CODING_DRUGCODE
  WHERE dmd_product_code_id IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY code_id, source_practice_code
    ORDER BY file_date DESC
  ) = 1
),

final_medication AS (
  SELECT
    dr.drug_record_guid,
    ir.issue_record_guid,
    ce. dmd_product_code_id                        AS code_id,
    dr.patient_guid,
    ir.effective_date                             AS effective_date,
    ir.entered_date                               AS entered_date,
    dr.source_practice_code,
    ir.course_duration_in_days                    AS duration,
    ce.term                                       AS drug_term,
    dr.prescription_type
  FROM prescribing_drugrecords_emis dr
  LEFT JOIN prescribing_issuerecords_emis ir
    USING (drug_record_guid)
  LEFT JOIN codes_emis ce
    USING (code_id, source_practice_code)
  WHERE dr.deleted = 'false'
    AND ir.deleted = 'false'
)

/* === FINAL OUTPUT:  observations + medications for non-cancer patients === */
SELECT *
FROM (
  /* ===== Observation events ===== */
  SELECT DISTINCT
    ncp.patient_guid:: string                            AS patient_guid,
    pp.sex,
    rc.label_6:: string                                  AS patient_ethnicity,
    CAST(NULL AS STRING)                                AS cancer_id,
    mh.effective_date                                   AS event_date,
    DATEDIFF(year, ncp.date_of_birth, mh.effective_date) AS event_age,
    'observation'                                       AS event_type,
    mh.snomed_c_t_concept_id                            AS snomed_c_t_concept_id,
    TO_VARCHAR(mh.term)                                 AS term,
    TO_VARCHAR(mh.associated_text)                      AS associated_text,
    TO_VARCHAR(mh.value)                                AS value,
    CAST(NULL AS NUMBER)                                AS med_code_id,
    CAST(NULL AS VARCHAR)                               AS drug_term,
    CAST(NULL AS NUMBER)                                AS duration
  FROM non_cancer_patients ncp
  JOIN lookup_medical_history mh
    ON mh.patient_guid = ncp.patient_guid
  JOIN patients pp
    ON pp.patient_guid = ncp.patient_guid
  LEFT JOIN analytics.cts.dim_cancer_codes dcc2
    ON mh.snomed_c_t_concept_id = dcc2.snomed_code
  LEFT JOIN EXPERIMENTS.MARTA_S. ETHNICITY_SNOMED_CODES rc
    ON rc.code = mh.snomed_c_t_concept_id
  WHERE
    mh.snomed_c_t_concept_id NOT IN (
      1572871000006101, 279991000000102, 428481002, 279991000000102,
      979851000000101, 887641000000105, 1958701000006105, 185317003,
      788007007, 283511000000105, 182836005, 182888003, 184103008
    )
    AND mh.snomed_c_t_concept_id NOT IN (498521000006103)  -- Attachment for medical notes
    AND mh.snomed_c_t_concept_id NOT IN (386472008)        -- Telephone consultations
    AND mh.observation_type NOT IN ('Immunisation')

  UNION ALL

  /* ===== Medication events ===== */
  SELECT
    ncp.patient_guid::string                            AS patient_guid,
    pp.sex,
    CAST(NULL AS VARCHAR)                               AS patient_ethnicity,
    CAST(NULL AS STRING)                                AS cancer_id,
    fm.effective_date                                   AS event_date,
    DATEDIFF(year, ncp. date_of_birth, fm. effective_date) AS event_age,
    'medication'                                        AS event_type,
    CAST(NULL AS NUMBER)                                AS snomed_c_t_concept_id,
    CAST(NULL AS VARCHAR)                               AS term,
    CAST(NULL AS VARCHAR)                               AS associated_text,
    CAST(NULL AS VARCHAR)                               AS value,
    fm.code_id                                          AS med_code_id,
    fm.drug_term,
    fm.duration
  FROM non_cancer_patients ncp
  JOIN final_medication fm
    ON fm. patient_guid = ncp.patient_guid
  JOIN patients pp
    ON pp.patient_guid = ncp.patient_guid
  WHERE
    fm.effective_date IS NOT NULL
    AND fm.effective_date >= DATE '2000-01-01'
    AND fm.effective_date <  DATE '2025-06-01'
) combined_events
ORDER BY
  patient_guid,
  event_date
-- LIMIT 50000;  -- Adjust based on your needs (can increase to 10,000,000 if needed)
