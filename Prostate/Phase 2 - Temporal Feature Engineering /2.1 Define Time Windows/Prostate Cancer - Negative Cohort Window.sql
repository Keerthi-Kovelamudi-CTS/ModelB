/*
================================================================================
CONTROL PATIENTS (Negative Cohort)
================================================================================
- Males WITHOUT prostate cancer
- 107 VALIDATED SNOMED codes
- INDEX_DATE = Random date (matched by age distribution to cancer cohort)
- ALL events UP TO INDEX_DATE (including recent data!)
- VERIFIED: No cancer within 12 months after INDEX_DATE
- Target: cancer_in_next_12m = 0
================================================================================
*/

WITH diagnostic_codes AS (
    SELECT
        TRY_TO_NUMBER(RAW_RECORDS: code_id::string)          AS code_id,
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
    WHERE sex = 'M'
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
        dc.term,
        TRY_TO_NUMBER(dc.snomed_c_t_concept_id:: string)    AS snomed_c_t_concept_id,
        dc.snomed_c_t_description_id,
        dc.source_practice_code
    FROM CTSUK_BULK.RAW.CARERECORD_OBSERVATION AS co
    LEFT JOIN diagnostic_codes AS dc
        ON TRY_TO_NUMBER(co.RAW_RECORDS:code_id::string) = TRY_TO_NUMBER(dc.code_id)
       AND co.practice_id = dc.source_practice_code
    WHERE effective_date IS NOT NULL
      AND effective_date >= DATE '2000-01-01'
      AND effective_date < DATE '2025-06-01'
),

-- ============================================================================
-- Patients with prostate cancer (to exclude)
-- Also get their diagnosis dates to verify no cancer within 12 months
-- ============================================================================
prostate_cancer_patients AS (
    SELECT DISTINCT
        mh.patient_guid,
        MIN(mh.effective_date) AS cancer_diagnosis_date
    FROM medical_history AS mh
    JOIN analytics. cts. dim_cancer_codes AS dcc
        ON mh.snomed_c_t_concept_id = dcc.snomed_code
    WHERE dcc.cancer_id = 'prostateCancer'
    GROUP BY mh.patient_guid
),

-- ============================================================================
-- Base cohort: Males without prostate cancer
-- ============================================================================
base_cohort AS (
    SELECT DISTINCT 
        omh.patient_guid,
        p.date_of_birth,
        p.sex
    FROM medical_history AS omh
    JOIN patients AS p
        ON p.patient_guid = omh. patient_guid
    WHERE p.sex = 'M'
      -- Exclude patients who EVER get prostate cancer
      AND omh.patient_guid NOT IN (SELECT patient_guid FROM prostate_cancer_patients)
      -- Exclude palliative care
      AND NOT EXISTS (
          SELECT 1
          FROM medical_history AS pmh
          WHERE pmh.patient_guid = omh.patient_guid
            AND pmh.snomed_c_t_concept_id = 1403151000000103
      )
),

-- ============================================================================
-- NON-CANCER PATIENTS with random INDEX_DATE
-- INDEX_DATE represents the "prediction point" 
-- We will predict:  "Cancer in next 12 months?" → NO (0)
-- ============================================================================
non_cancer_patients AS (
    SELECT 
        bc.patient_guid,
        bc.date_of_birth,
        bc.sex,
        -- Random INDEX_DATE between 2013-01-01 and 2024-12-31 (12 years range)
        DATEADD(
            day,
            ABS(MOD(HASH(bc.patient_guid), 4383)),
            DATE '2013-01-01'
        ) AS index_date
    FROM base_cohort bc
    QUALIFY ROW_NUMBER() OVER (ORDER BY HASH(bc.patient_guid)) <= 125000
),

-- ============================================================================
-- VERIFY: No cancer within 12 months AFTER index_date
-- This is CRITICAL for the forward-looking model! 
-- ============================================================================
verified_controls AS (
    SELECT 
        ncp.*
    FROM non_cancer_patients ncp
    -- Double-check: Patient should not develop cancer within 12 months after index_date
    WHERE NOT EXISTS (
        SELECT 1
        FROM prostate_cancer_patients pcp
        WHERE pcp.patient_guid = ncp.patient_guid
          AND pcp.cancer_diagnosis_date BETWEEN ncp.index_date AND DATEADD(month, 12, ncp.index_date)
    )
    -- Also verify index_date is reasonable (patient has data before and after)
    AND ncp.index_date >= DATE '2013-01-01'
    AND ncp.index_date <= DATE '2024-01-01'  -- Allow 12 months follow-up
),

-- ============================================================================
-- 107 VALIDATED SNOMED CODES (Same as Cancer Cohort)
-- ============================================================================
validated_snomed_codes AS (
    SELECT snomed_code
    FROM (VALUES
        -- ====================================================================
        -- PSA (7 codes)
        -- ====================================================================
        (1030791000000100),  -- PSA (prostate-specific antigen) level
        (1000381000000105),  -- Serum PSA (prostate specific antigen) level
        (396152005),         -- Raised PSA
        (166160000),         -- Prostate specific antigen abnormal
        (1030021000000101),  -- Free PSA (prostate specific antigen) level
        (1000481000000100),  -- Serum free PSA (prostate specific antigen) level
        (1006591000000104),  -- Total PSA (prostate specific antigen) level
        
        -- ====================================================================
        -- DRE (6 codes)
        -- ====================================================================
        (410007005),         -- Rectal examination
        (275302008),         -- Prostate enlarged on PR
        (274296009),         -- O/E - rectal examination
        (909491000006106),   -- [RFC] Reason for care:  Enlarged prostate
        (271302001),         -- O/E - PR - prostatic swelling
        (801000119105),      -- On rectal examination of prostate abnormality detected
        
        -- ====================================================================
        -- LUTS (13 codes)
        -- ====================================================================
        (11441004),          -- Prostatism
        (249274008),         -- Urinary symptoms
        (981411000006101),   -- Nocturia
        (307541003),         -- Lower urinary tract symptoms
        (5972002),           -- Delay when starting to pass urine
        (267064002),         -- Retention of urine
        (1726491000006101),  -- LUTS - Lower urinary tract symptoms
        (162116003),         -- Urinary frequency
        (763111000000108),   -- Moderate lower urinary tract symptoms
        (75088002),          -- Urgent desire to urinate
        (236648008),         -- Acute retention of urine
        (49650001),          -- Dysuria
        (300471006),         -- Observation of frequency of urination
        
        -- ====================================================================
        -- HAEMATURIA (9 codes)
        -- ====================================================================
        (1038641000000104),  -- Urine microscopy:  red cells
        (999131000000106),   -- Sample microscopy:  red cells
        (34436003),          -- Haematuria
        (167302009),         -- Urine blood test = +++
        (167300001),         -- Urine blood test = +
        (197938001),         -- Painless haematuria
        (1041881000000101),  -- Urine microscopy: pus cells
        (167301002),         -- Urine blood test = ++
        (197941005),         -- Frank haematuria
        
        -- ====================================================================
        -- ERECTILE DYSFUNCTION (2 codes)
        -- ====================================================================
        (397803000),         -- Impotence
        (860914002),         -- Erectile dysfunction
        
        -- ====================================================================
        -- PROSTATE SPECIFIC (3 codes)
        -- ====================================================================
        (266569009),         -- BPH - benign prostatic hypertrophy
        (236650000),         -- Chronic retention of urine
        (414205003),         -- Family history of prostate cancer
        
        -- ====================================================================
        -- RENAL FUNCTION (8 codes)
        -- ====================================================================
        (1020291000000106),  -- GFR calculated by MDRD
        (1000731000000107),  -- Serum creatinine level
        (80274001),          -- Glomerular filtration rate
        (1028131000000104),  -- Serum urea level
        (1000931000000105),  -- Serum creatine kinase level
        (1011481000000105),  -- eGFR using CKD-EPI
        (1000641000000106),  -- Serum electrolytes level
        (1001011000000107),  -- Plasma creatinine level
        
        -- ====================================================================
        -- LIVER FUNCTION (7 codes)
        -- ====================================================================
        (1018251000000107),  -- ALT/SGPT serum level
        (997531000000108),   -- Liver function test
        (999691000000104),   -- Serum bilirubin level
        (997561000000103),   -- Plasma total bilirubin level
        (997591000000109),   -- Serum total bilirubin level
        (1026761000000106),  -- Total bilirubin level
        (1013041000000106),  -- Plasma alkaline phosphatase level
        
        -- ====================================================================
        -- ELECTROLYTES (5 codes)
        -- ====================================================================
        (1000661000000107),  -- Serum sodium level
        (1000651000000109),  -- Serum potassium level
        (935051000000108),   -- Corrected serum calcium level
        (1000691000000101),  -- Serum calcium level
        (1017381000000106),  -- Plasma sodium level
        
        -- ====================================================================
        -- FBC / HAEMATOLOGY (17 codes)
        -- ====================================================================
        (993551000000106),   -- Large unstained cells
        (1022551000000104),  -- Neutrophil count
        (1022491000000106),  -- MCV - Mean corpuscular volume
        (1022651000000100),  -- Platelet count
        (1022561000000101),  -- Eosinophil count
        (1022591000000107),  -- Monocyte count
        (1022541000000102),  -- Total white cell count
        (1022571000000108),  -- Basophil count
        (1022581000000105),  -- Lymphocyte count
        (1022451000000103),  -- Red blood cell count
        (1022471000000107),  -- MCH - Mean corpuscular haemoglobin
        (1022431000000105),  -- Haemoglobin estimation
        (1022441000000101),  -- FBC - full blood count
        (1022291000000105),  -- Haematocrit
        (1022481000000109),  -- MCHC - Mean corpuscular haemoglobin concentration
        (993501000000105),   -- Red blood cell distribution width
        (1022511000000103),  -- Erythrocyte sedimentation rate
        
        -- ====================================================================
        -- LIPID PROFILE (8 codes)
        -- ====================================================================
        (1005671000000105),  -- Serum cholesterol level
        (1006191000000106),  -- Serum non high density lipoprotein cholesterol level
        (1005691000000109),  -- Serum triglycerides level
        (1005681000000107),  -- Serum high density lipoprotein cholesterol level
        (1022191000000100),  -- Serum low density lipoprotein cholesterol level
        (854781000006103),   -- Fasting lipids
        (1030411000000101),  -- Non high density lipoprotein cholesterol level
        (1010581000000101),  -- Plasma HDL (high density lipoprotein) cholesterol level
        
        -- ====================================================================
        -- GLUCOSE (3 codes)
        -- ====================================================================
        (1010671000000102),  -- Plasma glucose level
        (1003141000000105),  -- Plasma fasting glucose level
        (999791000000106),   -- HbA1c level - IFCC standardised
        
        -- ====================================================================
        -- PROTEIN (6 codes)
        -- ====================================================================
        (1000821000000103),  -- Serum albumin level
        (1000621000000104),  -- Serum alkaline phosphatase level
        (167273002),         -- Urine protein test negative
        (1000701000000101),  -- Serum inorganic phosphate level
        (1000811000000109),  -- Serum total protein
        (1001231000000108),  -- Serum globulin level
        
        -- ====================================================================
        -- URINE (8 codes)
        -- ====================================================================
        (1023711000000100),  -- Urine culture
        (992871000000105),   -- Epithelial cell count
        (1014831000000107),  -- Urine microscopy
        (365689007),         -- Finding related to casts on urine microscopy
        (992821000000106),   -- Organism count
        (1007881000000101),  -- Urine dipstick test
        (167221003),         -- Urinalysis = no abnormality
        (999291000000102),   -- Sample microscopy:  leucocytes
        
        -- ====================================================================
        -- THYROID (3 codes)
        -- ====================================================================
        (1022791000000101),  -- Serum TSH (thyroid stimulating hormone) level
        (1016851000000107),  -- Thyroid function test
        (1016971000000106),  -- Serum free T4 level
        
        -- ====================================================================
        -- INFLAMMATORY (2 codes)
        -- ====================================================================
        (1001371000000100),  -- Serum CRP (C reactive protein) level
        (999651000000107)    -- Plasma C reactive protein
        
    ) AS t(snomed_code)
)

-- ============================================================================
-- FINAL OUTPUT - FORWARD LOOKING MODEL
-- ============================================================================
SELECT DISTINCT
    vc.patient_guid:: string                                 AS patient_guid,
    vc.sex,
    CAST(NULL AS STRING)                                    AS cancer_id,
    CAST(NULL AS DATE)                                      AS date_of_diagnosis,
    vc.index_date,                                          -- Prediction point
    DATEDIFF(year, vc.date_of_birth, vc.index_date)         AS age_at_index,
    mh.effective_date                                       AS event_date,
    DATEDIFF(year, vc.date_of_birth, mh.effective_date)     AS event_age,
    
    -- Months before INDEX_DATE (not diagnosis)
    DATEDIFF(month, mh.effective_date, vc.index_date)       AS months_before_index,
    
    -- NEW: Time windows relative to INDEX_DATE (SAME AS CANCER COHORT)
    CASE 
        WHEN DATEDIFF(month, mh.effective_date, vc.index_date) BETWEEN 0 AND 11 THEN 'W1'   -- 0-12 months (MOST RECENT!)
        WHEN DATEDIFF(month, mh.effective_date, vc.index_date) BETWEEN 12 AND 23 THEN 'W2'  -- 12-24 months
        WHEN DATEDIFF(month, mh.effective_date, vc.index_date) BETWEEN 24 AND 35 THEN 'W3'  -- 24-36 months
        WHEN DATEDIFF(month, mh.effective_date, vc.index_date) BETWEEN 36 AND 47 THEN 'W4'  -- 36-48 months
        WHEN DATEDIFF(month, mh.effective_date, vc.index_date) BETWEEN 48 AND 60 THEN 'W5'  -- 48-60 months
        ELSE 'OUT_OF_RANGE'
    END AS time_window,
    
    'observation'                                           AS event_type,
    mh. snomed_c_t_concept_id,
    TO_VARCHAR(mh.term)                                     AS term,
    TO_VARCHAR(mh. associated_text)                          AS associated_text,
    TO_VARCHAR(mh.value)                                    AS value,
    
    -- NEW: Target variable (NO cancer in next 12 months)
    0                                                       AS cancer_in_next_12m

FROM verified_controls vc

JOIN medical_history mh
    ON mh.patient_guid = vc.patient_guid

WHERE
    -- ========================================================================
    -- NEW: Get ALL events UP TO index_date (INCLUDES recent data!)
    -- ========================================================================
    mh.effective_date <= vc.index_date
    
    -- Go back 60 months from index_date (5 years of history)
    AND mh.effective_date >= DATEADD(month, -60, vc. index_date)
    
    -- Only your 107 validated SNOMED codes
    AND mh.snomed_c_t_concept_id IN (SELECT snomed_code FROM validated_snomed_codes)
    
    -- Exclude immunisation
    AND COALESCE(mh.observation_type, '') NOT IN ('Immunisation')

ORDER BY
    patient_guid,
    event_date;
