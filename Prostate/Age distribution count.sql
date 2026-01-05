--- Age based used for prostate negative to avoid noise


/* Age distribution count ---
WITH diagnostic_codes AS (
    SELECT
        TRY_TO_NUMBER(RAW_RECORDS:code_id::string) AS code_id,
        PRACTICE_ID AS source_practice_code,
        RAW_RECORDS:snomed_c_t_concept_id AS snomed_c_t_concept_id
    FROM CTSUK_BULK.RAW.CODING_CLINICALCODE
    WHERE snomed_c_t_concept_id IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY code_id, source_practice_code ORDER BY PRACTICE_ID) = 1
),

patients AS (
    SELECT
        patient_guid,
        sex,
        date_of_birth
    FROM CTSUK_BULK.STAGING.PATIENTS_EMIS
),

medical_history AS (
  SELECT
    co.RAW_RECORDS:patient_guid AS patient_guid,
    TRY_TO_DATE(co. RAW_RECORDS:effective_date::string) AS effective_date,
    TRY_TO_NUMBER(dc.snomed_c_t_concept_id::string) AS snomed_c_t_concept_id
  FROM CTSUK_BULK.RAW.CARERECORD_OBSERVATION AS co
  LEFT JOIN diagnostic_codes AS dc
    ON TRY_TO_NUMBER(co.RAW_RECORDS:code_id::string) = dc.code_id
   AND co.practice_id = dc.source_practice_code
  WHERE effective_date IS NOT NULL
    AND effective_date >= DATE '1960-01-01'
    AND effective_date < DATE '2025-06-01'
),

cancer_patients AS (
  SELECT
    mh.patient_guid,
    pp.date_of_birth,
    MIN(mh.effective_date) AS date_of_diagnosis,
    DATEDIFF(year, pp.date_of_birth, MIN(mh.effective_date)) AS age_at_diagnosis
  FROM medical_history AS mh
  JOIN analytics.cts.dim_cancer_codes AS dcc
    ON mh.snomed_c_t_concept_id = dcc. snomed_code
  JOIN patients pp 
    ON pp.patient_guid = mh.patient_guid
  WHERE dcc.cancer_id = 'prostateCancer'
    AND pp.sex = 'M'
  GROUP BY 1, 2
)

-- Age distribution
SELECT 
    FLOOR(age_at_diagnosis / 10) * 10 AS age_bucket,
    COUNT(*) AS patient_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM cancer_patients
GROUP BY age_bucket
ORDER BY age_bucket;*/
