WITH diagnostic_codes AS (
    SELECT
        TRY_TO_NUMBER(RAW_RECORDS: code_id:: string)                AS code_id,
        TO_DATE(
            REGEXP_SUBSTR(file_name, '/([0-9]{8})/', 1, 1, 'e', 1),
            'YYYYMMDD'
        )                                                         AS file_date,
        PRACTICE_ID                                               AS source_practice_code,
        RAW_RECORDS: term                                          AS term,
        RAW_RECORDS:snomed_c_t_concept_id                         AS snomed_c_t_concept_id,
        RAW_RECORDS:snomed_c_t_description_id                     AS snomed_c_t_description_id
    FROM CTSUK_BULK. RAW. CODING_CLINICALCODE
    WHERE snomed_c_t_concept_id IS NOT NULL
      AND term IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY code_id, source_practice_code ORDER BY file_date DESC) = 1
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
    co.RAW_RECORDS:code_id:: string                       AS code_id,
    co.RAW_RECORDS:patient_guid                          AS patient_guid,
    co.RAW_RECORDS:gender                                AS patient_gender,
    co.RAW_RECORDS:consultation_guid                     AS consultation_guid,
    TRY_TO_DATE(co.RAW_RECORDS:effective_date:: string)   AS effective_date,
    TRY_TO_DATE(co.RAW_RECORDS:entered_date::string)     AS entered_date,
    co.RAW_RECORDS:associated_text                       AS associated_text,
    co.RAW_RECORDS:observation_type                      AS observation_type,
    co.RAW_RECORDS:value                                 AS value,
    co.RAW_RECORDS: ENTERED_BY_USER_IN_ROLE_GUID          AS ENTERED_BY_USER_IN_ROLE_GUID,
    dc.term,
    TRY_TO_NUMBER(dc.snomed_c_t_concept_id:: string)      AS snomed_c_t_concept_id,
    dc.snomed_c_t_description_id,
    dc.source_practice_code
  FROM CTSUK_BULK.RAW.CARERECORD_OBSERVATION AS co
  LEFT JOIN diagnostic_codes AS dc
    ON TRY_TO_NUMBER(co.RAW_RECORDS:code_id::string) = TRY_TO_NUMBER(dc.code_id)
   AND co.practice_id = dc.source_practice_code
  LEFT JOIN EXPERIMENTS.MARTA_S. ETHNICITY_SNOMED_CODES esc
    ON dc.snomed_c_t_concept_id = esc.code
  WHERE effective_date IS NOT NULL
    AND effective_date >= DATE '1960-01-01'
    AND effective_date <  DATE '2025-06-01'
),

/* Patients with PROSTATE cancer diagnosis (first diagnosis date) */
cancer_patients AS (
  SELECT
    mh.practice_id,
    mh.patient_guid:: string                               AS patient_guid,
    mh.patient_gender,
    dcc.cancer_id,
    MIN(mh.effective_date)                                AS date_of_diagnosis
  FROM medical_history AS mh
  JOIN analytics.cts.dim_cancer_codes AS dcc
    ON mh.snomed_c_t_concept_id = dcc.snomed_code
  JOIN patients pp                                      
    ON pp.patient_guid = mh.patient_guid                  
  WHERE mh.effective_date IS NOT NULL
    AND dcc.cancer_id = 'prostateCancer'                   -- Exact match
    AND pp. sex = 'M'                                      -- Males only
  GROUP BY 1,2,3,4
),

/* === Medication pipeline CTEs (from EMIS prescribing tables) === */
prescribing_drugrecords_emis AS (
  SELECT
    TO_DATE(
      REGEXP_SUBSTR(file_name, '/([0-9]{8})/', 1, 1, 'e', 1),
      'YYYYMMDD'
    )                                   AS file_date,
    TRIM(raw_records:drug_record_guid)  AS drug_record_guid,
    TRIM(raw_records:prescription_type) AS prescription_type,
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
  FROM CTSUK_BULK. RAW.PRESCRIBING_ISSUERECORD
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

/* === FINAL OUTPUT:  observations + medications in one timeline per patient === */
SELECT *
FROM (
  /* ===== Observation events ===== */
  SELECT DISTINCT
    cp.patient_guid,
    pp.sex,
    TO_VARCHAR(rc.ethnicity)                               AS patient_ethnicity,
    cp.cancer_id,
    mh.effective_date                                      AS event_date,
    DATEDIFF(year, pp.date_of_birth, mh.effective_date)    AS event_age,
    'observation'                                          AS event_type,
    mh.snomed_c_t_concept_id,
    TO_VARCHAR(mh.term)                                    AS term,
    TO_VARCHAR(mh.associated_text)                         AS associated_text,
    TO_VARCHAR(mh.value)                                   AS value,
    /* medication fields NULL for observation rows */
    CAST(NULL AS NUMBER)   AS med_code_id,
    CAST(NULL AS VARCHAR)  AS drug_term,
    CAST(NULL AS NUMBER)   AS duration
  FROM cancer_patients cp
  JOIN medical_history mh
    ON mh.patient_guid:: string = cp.patient_guid
  JOIN patients pp
    ON pp.patient_guid = cp.patient_guid
  LEFT JOIN analytics.cts.dim_cancer_codes dcc2
    ON mh.snomed_c_t_concept_id = dcc2.snomed_code
  LEFT JOIN EXPERIMENTS.MARTA_S. ETHNICITY_SNOMED_CODES rc
    ON rc.code = mh.snomed_c_t_concept_id
  WHERE
    cp.date_of_diagnosis IS NOT NULL
    AND mh.effective_date >= DATEADD(year, -21, cp.date_of_diagnosis)
    AND mh.effective_date <  DATEADD(month, -6, cp.date_of_diagnosis)
    AND mh.snomed_c_t_concept_id NOT IN (
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
    cp.patient_guid,                                      
    pp.sex,
    CAST(NULL AS VARCHAR)                                  AS patient_ethnicity, 
    cp.cancer_id,
    fm.effective_date                                      AS event_date,
    DATEDIFF(year, pp.date_of_birth, fm.effective_date)    AS event_age,
    'medication'                                           AS event_type,
    /* observation fields NULL for medication rows */
    CAST(NULL AS NUMBER)   AS snomed_c_t_concept_id,
    CAST(NULL AS VARCHAR)  AS term,
    CAST(NULL AS VARCHAR)  AS associated_text,
    CAST(NULL AS VARCHAR)  AS value,
    /* medication fields */
    fm.code_id          AS med_code_id,
    fm.drug_term,
    fm.duration
  FROM cancer_patients cp
  JOIN final_medication fm
    ON fm. patient_guid = cp.patient_guid
  JOIN patients pp
    ON pp.patient_guid = cp.patient_guid
  WHERE
    cp.date_of_diagnosis IS NOT NULL
    AND fm.effective_date IS NOT NULL
    AND fm.effective_date >= DATEADD(year, -21, cp.date_of_diagnosis)
    AND fm.effective_date <  DATEADD(month, -6, cp.date_of_diagnosis)
) combined_events
ORDER BY
  patient_guid,
  event_date
-- LIMIT 50000; -- Adjust based on your needs (can increase to 10,000,000 if needed)

