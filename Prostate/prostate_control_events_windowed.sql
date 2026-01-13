-- ============================================================================
-- PROSTATE CONTROL COHORT - WINDOWED EVENTS (FILTERED)
-- Extracts events for controls (NO prostate cancer) with top 136 SNOMED codes
-- Uses latest event date as synthetic "index_date" to match cancer cohort
-- ============================================================================

WITH diagnostic_codes AS (
    SELECT
        TRY_TO_NUMBER(RAW_RECORDS: code_id::string) AS code_id,
        PRACTICE_ID AS source_practice_code,
        RAW_RECORDS:snomed_c_t_concept_id AS snomed_c_t_concept_id,
        RAW_RECORDS:term AS term
    FROM CTSUK_BULK. RAW.CODING_CLINICALCODE
    WHERE snomed_c_t_concept_id IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY code_id, source_practice_code ORDER BY PRACTICE_ID) = 1
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
    co.RAW_RECORDS:patient_guid AS patient_guid,
    TRY_TO_DATE(co.RAW_RECORDS:effective_date:: string) AS effective_date,
    co.RAW_RECORDS:value AS value,
    TRY_TO_NUMBER(dc.snomed_c_t_concept_id:: string) AS snomed_c_t_concept_id,
    dc.term
  FROM CTSUK_BULK.RAW.CARERECORD_OBSERVATION AS co
  LEFT JOIN diagnostic_codes AS dc
    ON TRY_TO_NUMBER(co.RAW_RECORDS:code_id:: string) = dc.code_id
   AND co.practice_id = dc.source_practice_code
  WHERE effective_date IS NOT NULL
    AND effective_date >= DATE '1960-01-01'
    AND effective_date < DATE '2025-06-01'
),

-- ============================================================================
-- CONTROL PATIENTS (NO prostate cancer diagnosis)
-- ============================================================================
control_patients AS (
  SELECT DISTINCT
    mh. practice_id,
    mh.patient_guid::string AS patient_guid,
    pp.sex,
    -- Use latest event date as synthetic "index_date"
    MAX(mh.effective_date) AS index_date
  FROM medical_history AS mh
  JOIN patients pp 
    ON pp.patient_guid = mh.patient_guid
  WHERE pp.sex = 'M'  -- Males only
    -- Exclude anyone who EVER had prostate cancer
    AND mh.patient_guid NOT IN (
      SELECT DISTINCT mh2.patient_guid
      FROM medical_history mh2
      JOIN analytics.cts.dim_cancer_codes dcc
        ON mh2.snomed_c_t_concept_id = dcc.snomed_code
      WHERE dcc.cancer_id = 'prostateCancer'
    )
    -- Optional:  Ensure they have recent data
    AND mh.effective_date >= DATE '2015-01-01'
  GROUP BY 1, 2, 3
  -- Optional: Limit to match cancer cohort size (adjust as needed)
  -- LIMIT 20000  -- Uncomment if you want to limit sample size
),

-- ============================================================================
-- TOP 136 SNOMED CODES (Same as cancer cohort)
-- ============================================================================
top_snomed_codes AS (
    SELECT snomed_code FROM (VALUES
        -- PSA (8 codes)
        (1030791000000100),
        (1000381000000105),
        (396152005),
        (166160000),
        (110061000006102),
        (1030021000000101),
        (1000481000000100),
        (1006591000000104),
        
        -- DRE (6 codes)
        (410007005),
        (275302008),
        (274296009),
        (909491000006106),
        (271302001),
        (801000119105),
        
        -- LUTS (13 codes)
        (11441004),
        (249274008),
        (981411000006101),
        (307541003),
        (5972002),
        (267064002),
        (1726491000006101),
        (162116003),
        (763111000000108),
        (68566005),
        (75088002),
        (236648008),
        (49650001),
        
        -- Haematuria (10 codes)
        (1038641000000104),
        (999131000000106),
        (167297006),
        (34436003),
        (167302009),
        (167300001),
        (197938001),
        (1041881000000101),
        (167301002),
        (197941005),
        
        -- Erectile Dysfunction (2 codes)
        (397803000),
        (860914002),
        
        -- Diagnostic Pathway (11 codes)
        (276431000000102),
        (185220009),
        (831281000006109),
        (25611000000107),
        (183552008),
        (315268008),
        (382841000000109),
        (38661000000100),
        (199251000000107),
        (904551000006100),
        (183444007),
        
        -- Prostate Procedures (6 codes)
        (176178006),
        (22034001),
        (395155002),
        (265593007),
        (266569009),
        (92691004),
        
        -- Renal Function (8 codes)
        (1020291000000106),
        (1000731000000107),
        (80274001),
        (1000971000000107),
        (1011481000000105),
        (1000641000000106),
        (1001011000000107),
        (1000931000000105),
        
        -- Liver Function (7 codes)
        (1018251000000107),
        (997531000000108),
        (999691000000104),
        (997561000000103),
        (997591000000109),
        (1026761000000106),
        (1013041000000106),
        
        -- Electrolytes (5 codes)
        (1000661000000107),
        (1000651000000109),
        (935051000000108),
        (1000691000000101),
        (1017381000000106),
        
        -- FBC/Haematology (17 codes)
        (993551000000106),
        (1022551000000104),
        (1022491000000106),
        (1022651000000100),
        (1022561000000101),
        (1022591000000107),
        (1022541000000102),
        (1022571000000108),
        (1022581000000105),
        (1022451000000103),
        (1022471000000107),
        (1022431000000105),
        (1022441000000101),
        (1022291000000105),
        (1022481000000109),
        (993501000000105),
        (1022511000000103),
        
        -- Lipid Profile (9 codes)
        (1005671000000105),
        (1006191000000106),
        (1005691000000109),
        (1005681000000107),
        (1022191000000100),
        (1028551000000102),
        (854781000006103),
        (1030411000000101),
        (1010581000000101),
        
        -- Glucose/Diabetes (4 codes)
        (1010671000000102),
        (1003141000000105),
        (167261002),
        (401081006),
        
        -- Blood Pressure (6 codes)
        (1091811000000102),
        (72313002),
        (163020007),
        (1085181000000102),
        (413153004),
        (413605002),
        
        -- Protein/Albumin (6 codes)
        (1000821000000103),
        (1000621000000104),
        (167273002),
        (1000701000000101),
        (1000811000000109),
        (1001231000000108),
        
        -- Urine Tests (10 codes)
        (1023711000000100),
        (992871000000105),
        (167287002),
        (1014831000000107),
        (365689007),
        (314138001),
        (992821000000106),
        (1007881000000101),
        (167221003),
        (999291000000102),
        
        -- Thyroid Function (3 codes)
        (1022791000000101),
        (1016851000000107),
        (1016971000000106),
        
        -- Inflammatory Markers (2 codes)
        (1001371000000100),
        (999651000000107),
        
        -- Tumour Markers (1 code)
        (1014601000000100),
        
        -- Imaging/Radiology (2 codes)
        (312250003),
        (174184006)
        
    ) AS codes(snomed_code)
)

-- ============================================================================
-- MAIN QUERY:  Extract filtered events with time windows
-- ============================================================================
SELECT 
    cp.patient_guid,
    cp.sex,
    cp.practice_id,
    cp.index_date AS date_of_diagnosis,  -- Using index_date as proxy
    mh.effective_date AS event_date,
    mh. snomed_c_t_concept_id,
    mh.term,
    mh.value,
    DATEDIFF(day, mh.effective_date, cp.index_date) AS days_before_diagnosis,
    
    -- Time window labels (relative to index_date)
    CASE 
      WHEN DATEDIFF(month, mh.effective_date, cp.index_date) BETWEEN 0 AND 3 
        THEN 'Window_0_3mo'
      WHEN DATEDIFF(month, mh.effective_date, cp.index_date) BETWEEN 3 AND 6 
        THEN 'Window_3_6mo'
      WHEN DATEDIFF(month, mh. effective_date, cp.index_date) BETWEEN 6 AND 12 
        THEN 'Window_6_12mo'
      WHEN DATEDIFF(month, mh.effective_date, cp. index_date) BETWEEN 12 AND 24 
        THEN 'Window_12_24mo'
      WHEN DATEDIFF(month, mh.effective_date, cp. index_date) BETWEEN 24 AND 36 
        THEN 'Window_24_36mo'
      ELSE 'Window_36mo_plus'
    END AS time_window
    
FROM control_patients cp
JOIN medical_history mh
    ON mh.patient_guid:: string = cp.patient_guid
--  KEY FILTER: Only top 136 SNOMED codes! 
JOIN top_snomed_codes tsc
    ON mh.snomed_c_t_concept_id = tsc. snomed_code
WHERE 
    cp.index_date IS NOT NULL
    -- Events from 36 months before to 1 day before index_date
    AND mh.effective_date >= DATEADD(month, -36, cp.index_date)
    AND mh.effective_date < DATEADD(day, -1, cp.index_date)
ORDER BY 
    cp.patient_guid, 
    mh.effective_date;
