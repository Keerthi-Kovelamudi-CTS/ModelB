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
    co.RAW_RECORDS:code_id:: string                      AS code_id,
    co.RAW_RECORDS:patient_guid                         AS patient_guid,
    co.RAW_RECORDS:gender                               AS patient_gender,
    co.RAW_RECORDS:consultation_guid                    AS consultation_guid,
    TRY_TO_DATE(co.RAW_RECORDS:effective_date::string)  AS effective_date,
    TRY_TO_DATE(co.RAW_RECORDS:entered_date::string)    AS entered_date,
    co.RAW_RECORDS:associated_text                      AS associated_text,
    co.RAW_RECORDS:observation_type                     AS observation_type,
    co.RAW_RECORDS:value                                AS value,
    co.RAW_RECORDS: ENTERED_BY_USER_IN_ROLE_GUID         AS ENTERED_BY_USER_IN_ROLE_GUID,
    dc.term,
    TRY_TO_NUMBER(dc.snomed_c_t_concept_id:: string)     AS snomed_c_t_concept_id,
    dc.snomed_c_t_description_id,
    dc.source_practice_code
  FROM CTSUK_BULK.RAW.CARERECORD_OBSERVATION AS co
  LEFT JOIN diagnostic_codes AS dc
    ON TRY_TO_NUMBER(co.RAW_RECORDS:code_id::string) = TRY_TO_NUMBER(dc.code_id)
   AND co.practice_id = dc.source_practice_code
  LEFT JOIN EXPERIMENTS.MARTA_S. ETHNICITY_SNOMED_CODES esc
    ON dc.snomed_c_t_concept_id = esc.code
  WHERE effective_date IS NOT NULL
    AND effective_date >= DATE '1968-01-01'
    AND effective_date <  DATE '2024-06-01'
),

cancer_patients AS (
  SELECT
    mh.practice_id,
    mh. patient_guid,
    mh.patient_gender,
    dcc.cancer_id,
    MIN(mh.effective_date) AS date_of_diagnosis
  FROM medical_history AS mh
  JOIN analytics.cts. dim_cancer_codes AS dcc
    ON mh.snomed_c_t_concept_id = dcc.snomed_code
  JOIN patients pp
    ON pp.patient_guid = mh. patient_guid
  WHERE mh.effective_date IS NOT NULL
    AND dcc. cancer_id = 'prostateCancer'
    AND pp.sex = 'M'
  GROUP BY 1,2,3,4
),

final_table AS (
  SELECT DISTINCT
    cp.practice_id,
    cp.patient_guid,
    pp.sex,
    DATEDIFF(year, pp.date_of_birth, cp.date_of_diagnosis) AS diagnosis_age,
    rc.ethnicity AS patient_ethnicity,
    cp.cancer_id,
    cp.date_of_diagnosis,
    mh.effective_date                                    AS event_date,
    DATEDIFF(year, pp.date_of_birth, mh.effective_date) AS event_age,
    DATEDIFF(day, cp.date_of_diagnosis, mh.effective_date) AS days_from_diagnosis,
    CASE WHEN mh.effective_date < cp.date_of_diagnosis THEN 'pre_dx'
         WHEN mh. effective_date = cp.date_of_diagnosis THEN 'at_dx'
         ELSE 'post_dx'
    END                                                  AS history_phase,
    mh.snomed_c_t_concept_id,
    mh.term,
    mh.observation_type,
    mh. associated_text,
    mh. value,
    CASE WHEN dcc2.snomed_code IS NOT NULL THEN 1 ELSE 0 END AS is_cancer_code
  FROM cancer_patients cp
  JOIN medical_history mh
    ON mh.patient_guid = cp.patient_guid
  JOIN patients pp
    ON pp.patient_guid = cp.patient_guid
  LEFT JOIN analytics.cts.dim_cancer_codes dcc2
    ON mh.snomed_c_t_concept_id = dcc2.snomed_code
  LEFT JOIN EXPERIMENTS.MARTA_S.ETHNICITY_SNOMED_CODES rc
    ON rc.code = mh.snomed_c_t_concept_id
  WHERE 
    date_of_diagnosis IS NOT NULL
    -- 12-MONTH WINDOW (1-12 months before diagnosis)
    AND mh.effective_date >= DATEADD(month, -12, cp.date_of_diagnosis)
    AND mh.effective_date <  DATEADD(month, -1, cp.date_of_diagnosis)
    -- Exclusions
    AND mh.snomed_c_t_concept_id NOT IN (
      1572871000006101, 279991000000102, 428481002, 279991000000102, 
      979851000000101, 887641000000105, 1958701000006105, 185317003, 
      788007007, 283511000000105, 182836005, 182888003, 184103008
    )
    AND mh.snomed_c_t_concept_id NOT IN (498521000006103)
    AND mh.snomed_c_t_concept_id NOT IN (386472008)
    AND mh.observation_type NOT IN ('Immunisation')
  ORDER BY
    cp.patient_guid,
    mh. effective_date
)

SELECT 
  term, 
  MAX(snomed_c_t_concept_id) AS snomed_id, 
  MEDIAN(event_age) AS median_snomed_age,
  COUNT(DISTINCT patient_guid) AS n_patient_count,
  (SELECT COUNT(DISTINCT patient_guid) FROM final_table) AS n_patient_count_total,
  AVG(CAST(IFF(VALUE = '', NULL, VALUE) AS REAL)) AS avg_value,
  MEDIAN(CAST(IFF(VALUE = '', NULL, VALUE) AS REAL)) AS median_value,
  STDDEV(CAST(IFF(VALUE = '', NULL, VALUE) AS REAL)) AS std_value,
  MODE(CAST(IFF(VALUE = '', NULL, VALUE) AS REAL)) AS freq_value
FROM final_table
GROUP BY term
HAVING COUNT(DISTINCT patient_guid) >= 10  -- MINIMUM 10 PATIENTS
ORDER BY n_patient_count DESC;
