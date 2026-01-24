/*
================================================================================
COUNT PROSTATE CANCER PATIENTS
================================================================================
Run this FIRST to know how many cancer patients you have,
then decide the limit for negative cohort
================================================================================
*/

WITH diagnostic_codes AS (
    SELECT
        TRY_TO_NUMBER(RAW_RECORDS: code_id:: string)          AS code_id,
        PRACTICE_ID                                         AS source_practice_code,
        RAW_RECORDS: snomed_c_t_concept_id                   AS snomed_c_t_concept_id
    FROM CTSUK_BULK.RAW.CODING_CLINICALCODE
    WHERE snomed_c_t_concept_id IS NOT NULL
),

medical_history AS (
    SELECT
        co.RAW_RECORDS:patient_guid                        AS patient_guid,
        TRY_TO_DATE(co.RAW_RECORDS:effective_date::string) AS effective_date,
        TRY_TO_NUMBER(dc.snomed_c_t_concept_id::string)    AS snomed_c_t_concept_id
    FROM CTSUK_BULK.RAW.CARERECORD_OBSERVATION AS co
    LEFT JOIN diagnostic_codes AS dc
        ON TRY_TO_NUMBER(co.RAW_RECORDS:code_id::string) = TRY_TO_NUMBER(dc.code_id)
       AND co.practice_id = dc.source_practice_code
    WHERE effective_date IS NOT NULL
),

-- Count prostate cancer patients
prostate_cancer_patients AS (
    SELECT
        mh.patient_guid:: string AS patient_guid,
        MIN(mh.effective_date) AS date_of_diagnosis
    FROM medical_history AS mh
    JOIN analytics. cts.dim_cancer_codes AS dcc
        ON mh.snomed_c_t_concept_id = dcc.snomed_code
    WHERE mh.effective_date IS NOT NULL
      AND dcc.cancer_id = 'prostateCancer'
    GROUP BY 1
)

SELECT 
    COUNT(*) AS total_prostate_cancer_patients,
    MIN(date_of_diagnosis) AS earliest_diagnosis,
    MAX(date_of_diagnosis) AS latest_diagnosis
FROM prostate_cancer_patients;
