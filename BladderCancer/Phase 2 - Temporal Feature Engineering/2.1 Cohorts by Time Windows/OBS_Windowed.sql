WITH params AS (
  SELECT
    DATE '1950-01-01' AS mh_start,
    DATE '2026-02-25' AS mh_end,
    36 AS window_start_months,
    12 AS window_end_months,
    24 AS window_mid_months
),

-- ══════════════════════════════════════════════════════════════
-- APPROVED OBSERVATION CODES (ORIGINAL 55 + SELECTIVE NEW 30)
-- Removed: LEG_OEDEMA, OBESITY_CODE, RAISED_BP, ABDOMINAL_PAIN
-- Kept: cancer-specific symptoms only
-- ═��════════════════════════════════════════════════════════════
approved_obs AS (
  SELECT column1 AS category, column2 AS term, column3 AS snomed_code, 'OBSERVATION' AS event_type
  FROM (VALUES
    -- ── ORIGINAL 55 CODES (unchanged) ──
    ('HAEMATURIA','Haematuria',34436003),
    ('HAEMATURIA','Frank haematuria',197941005),
    ('HAEMATURIA','Blood in urine - haematuria',53298000),
    ('HAEMATURIA','Painless haematuria',197938001),
    ('HAEMATURIA','Painful haematuria',197939009),
    ('HAEMATURIA','History of haematuria',161550001),
    ('HAEMATURIA','Microscopic haematuria',197940006),
    ('URINE LAB ABNORMALITIES','Urine blood test = +++',167302009),
    ('URINE LAB ABNORMALITIES','Urine blood test = ++',167301002),
    ('URINE LAB ABNORMALITIES','Urine: red - blood',167235001),
    ('URINE LAB ABNORMALITIES','Urine specific gravity - 1^030',1856671000006101),
    ('URINE LAB ABNORMALITIES','Urine protein test = +++',167277001),
    ('URINE LAB ABNORMALITIES','Urine protein test = ++++',167278006),
    ('URINE LAB ABNORMALITIES','Urine urobilinogen not detected',167323003),
    ('URINE LAB ABNORMALITIES','Urine looks dark',39575007),
    ('URINE LAB ABNORMALITIES','Urine: dark/concentrated',167232003),
    ('URINE LAB ABNORMALITIES','Sample microscopy: leucocytes',999291000000102),
    ('URINE LAB ABNORMALITIES','Sample microscopy: red cells',999131000000106),
    ('URINE LAB ABNORMALITIES','Urine cytology',310437009),
    ('URINE INVESTIGATIONS','Urine culture',1023711000000100),
    ('URINE INVESTIGATIONS','Urine microscopy',1014831000000107),
    ('URINE INVESTIGATIONS','MSU - Mid-stream urine sample',258574006),
    ('URINE INVESTIGATIONS','Cytology laboratory test',992841000000104),
    ('URINE INVESTIGATIONS','Antibacterial substance screening test',1009861000000101),
    ('LUTS','Bladder pain',15803009),
    ('LUTS','Difficulty passing urine',102835006),
    ('LUTS','Delay when starting to pass urine',5972002),
    ('LUTS','Terminal dribbling of urine',162130008),
    ('LUTS','Micturition frequency and polyuria',274734008),
    ('LUTS','Chronic retention of urine',236650000),
    ('LUTS','Lower urinary tract infectious disease',4009004),
    ('UROLOGICAL CONDITIONS','Disorder of bladder',42643001),
    ('UROLOGICAL CONDITIONS','Bladder trabeculation',79184009),
    ('UROLOGICAL CONDITIONS','Bladder stone',70650003),
    ('UROLOGICAL CONDITIONS','Bladder assessment',268390008),
    ('UROLOGICAL CONDITIONS','Hydronephrosis',43064006),
    ('UROLOGICAL CONDITIONS','Hydroureteronephrosis',40068008),
    ('UROLOGICAL CONDITIONS','Atrophy of kidney',197659005),
    ('UROLOGICAL CONDITIONS','Pyuria',4800001),
    ('CATHETER/PROCEDURES','Insertion of urethral catheter',410021007),
    ('CATHETER/PROCEDURES','Urinary catheter in situ',439053001),
    ('CATHETER/PROCEDURES','Catheter complications',73862001),
    ('CATHETER/PROCEDURES','Successful trial without catheter',864021000000101),
    ('CATHETER/PROCEDURES','Removal of urethral catheter',55449009),
    ('CATHETER/PROCEDURES','Urethroscopic urethral dilatation',176356008),
    ('CATHETER/PROCEDURES','Prostatitis',9713002),
    ('IMAGING','Renal ultrasound abnormal',202541000000104),
    ('IMAGING','Ultrasound scan of bladder',169251004),
    ('IMAGING','Echography of kidney',306005),
    ('IMAGING','US urinary tract',303917008),
    ('GYNAECOLOGICAL/BLEEDING','Intermenstrual bleeding - irregular',64996003),
    ('GYNAECOLOGICAL/BLEEDING','Menorrhagia',386692008),
    ('GYNAECOLOGICAL/BLEEDING','Abnormal vaginal bleeding',301822002),
    ('OTHERS','[D]Abdominal or pelvic symptom NOS',396941000000100),
    ('OTHERS','Acute kidney injury stage 2',851941000000103),

    -- ══════════════════════════════════════════════════════════
    -- NEW: CANCER-SPECIFIC SYMPTOMS ONLY (30 codes)
    -- ══════════════════════════════════════════════════════════

    -- WEIGHT LOSS (78 cancer patients — cancer red flag)
    ('WEIGHT_LOSS','Weight loss',89362005),
    ('WEIGHT_LOSS','Weight decreasing',161832001),
    ('WEIGHT_LOSS','Abnormal weight loss',267024001),
    ('WEIGHT_LOSS','Unintentional weight loss',448765001),
    ('WEIGHT_LOSS','Complaining of weight loss',198511000000103),
    ('WEIGHT_LOSS','Unexplained/progressive weight loss',960561000006106),

    -- FATIGUE (127 cancer patients — cancer symptom)
    ('FATIGUE','Fatigue',84229001),
    ('FATIGUE','Fatigue - symptom',272060000),
    ('FATIGUE','Malaise and fatigue',271795006),
    ('FATIGUE','Lethargy',214264003),
    ('FATIGUE','Exhaustion',60119000),
    ('FATIGUE','C/O - debility - malaise',272036004),
    ('FATIGUE','Feels unwell',367391008),

    -- APPETITE LOSS (11 cancer patients — cancer red flag)
    ('APPETITE_LOSS','Loss of appetite',79890006),
    ('APPETITE_LOSS','Anorexia symptom',249468005),

    -- DYSURIA (122 cancer patients — urinary symptom)
    ('DYSURIA','Dysuria',49650001),

    -- SUPRAPUBIC PAIN (12 cancer — bladder-specific)
    ('SUPRAPUBIC_PAIN','Suprapubic pain',162053006),

    -- BACK/LOIN PAIN (509+33 cancer — metastatic signal)
    ('BACK_PAIN','Low back pain',279039007),
    ('BACK_PAIN','Acute back pain - unspecified',161891005),
    ('BACK_PAIN','C/O - low back pain',161894002),
    ('BACK_PAIN','Chronic low back pain',278860009),
    ('BACK_PAIN','Chronic back pain',134407002),
    ('LOIN_PAIN','Loin pain',271857006),
    ('LOIN_PAIN','C/O - loin pain',272047006),
    ('LOIN_PAIN','Left loin pain',853721000006104),
    ('LOIN_PAIN','Left flank pain',162049009),

    -- DVT/PE (21+26 cancer — paraneoplastic signal)
    ('DVT','Deep vein thrombosis',128053003),
    ('DVT','Deep vein thrombosis of lower limb',404223003),
    ('DVT','DVT leg',266267005),
    ('DVT','Postoperative deep vein thrombosis',213220000),
    ('DVT','Suspected DVT',432805000),
    ('PULMONARY_EMBOLISM','Pulmonary embolism',59282003),
    ('PULMONARY_EMBOLISM','Recurrent pulmonary embolism',438773007),

    -- ANAEMIA DIAGNOSIS (122 cancer — blood loss signal)
    ('ANAEMIA_DX','Anaemia',271737000),
    ('ANAEMIA_DX','Anaemia (variant)',960301000006100),
    ('ANAEMIA_DX','Normocytic anaemia',300980002),
    ('ANAEMIA_DX','Macrocytic anaemia',83414005),
    ('ANAEMIA_DX','Chronic anaemia',191268006),
    ('ANAEMIA_DX','H/O: anaemia',275538002),

    -- NIGHT SWEATS (16 cancer — B-symptom)
    ('NIGHT_SWEATS','Night sweats',42984000),

    -- FRAILTY (874 cancer — appeared in top 40 features)
    ('FRAILTY','Frailty Index score',713636003),
    ('FRAILTY','Mild frailty',925791000000100),
    ('FRAILTY','Moderate frailty',925831000000107),
    ('FRAILTY','Severe frailty',925861000000102)
  )
),

-- ══════════════════════════════════════════════════════════════
-- APPROVED RISK FACTOR CODES (37 — unchanged)
-- ══════════════════════════════════════════════════════════════
approved_risk_factors AS (
  SELECT column1 AS category, column2 AS term, column3 AS snomed_code, 'RISK FACTOR' AS event_type
  FROM (VALUES
    ('CURRENT SMOKER','Current smoker',77176002),
    ('CURRENT SMOKER','Cigarette smoker',65568007),
    ('CURRENT SMOKER','Light cigarette smoker (1-9 cigs/day)',160603005),
    ('CURRENT SMOKER','Moderate cigarette smoker (10-19 cigs/day)',160604004),
    ('CURRENT SMOKER','Heavy cigarette smoker (20-39 cigs/day)',160605003),
    ('CURRENT SMOKER','Very heavy cigarette smoker (40+ cigs/day)',160606002),
    ('CURRENT SMOKER','Tobacco smoking behaviour - finding',365981007),
    ('CURRENT SMOKER','Tobacco smoking consumption',266918002),
    ('CURRENT SMOKER','Cigar consumption',230057008),
    ('CURRENT SMOKER','Current smoker annual review',505651000000103),
    ('CURRENT SMOKER','Passive smoking risk',161080002),
    ('CURRENT SMOKER','Current smoker',854071000006103),
    ('CURRENT SMOKER','Current Smoker NOS',604961000006105),
    ('EX SMOKER','Ex-smoker',8517006),
    ('EX SMOKER','Ex-cigarette smoker',281018007),
    ('EX SMOKER','Ex-light cigarette smoker (1-9/day)',266922007),
    ('EX SMOKER','Ex-moderate cigarette smoker (10-19/day)',266923002),
    ('EX SMOKER','Ex-heavy cigarette smoker (20-39/day)',266924008),
    ('EX SMOKER','Ex-very heavy cigarette smoker (40+/day)',266925009),
    ('EX SMOKER','Ex-trivial cigarette smoker (<1/day)',266921000),
    ('EX SMOKER','Ex roll-up cigarette smoker',492191000000103),
    ('EX SMOKER','Ex-smoker annual review',505761000000105),
    ('EX SMOKER','Stopped smoking',160617001),
    ('EX SMOKER','Date ceased smoking',160625004),
    ('EX SMOKER','Recently stopped smoking',517211000000106),
    ('EX SMOKER','Time since stopped smoking',228486009),
    ('SMOKING CESSATION REFUSED','Not interested in stopping smoking',394873005),
    ('SMOKING CESSATION REFUSED','Smoking cessation programme declined',1087441000000106),
    ('SMOKING CESSATION REFUSED','Smoking cessation advice declined',527151000000107),
    ('SMOKING CESSATION REFUSED','Referral for smoking cessation service declined',871641000000105),
    ('SMOKING CESSATION REFUSED','Smoking cessation drug therapy declined',822591000000108),
    ('SMOKING CESSATION REFUSED','Monitoring of smoking cessation therapy declined',765001003),
    ('ALCOHOL','Alcoholic beverage intake',897148007),
    ('ALCOHOL','Alcohol units consumed per week',1082641000000106),
    ('ALCOHOL','Alcohol consumption',160573003),
    ('ALCOHOL','Alcohol use disorders identification test score',443280005),
    ('ALCOHOL','AUDIT-C score',763256006)
  )
),

-- ══════════════════════════════════════════════════════════════
-- APPROVED COMORBIDITY CODES (24 — unchanged)
-- ══════════════════════════════════════════════════════════════
approved_comorbidities AS (
  SELECT column1 AS category, column2 AS term, column3 AS snomed_code, 'COMORBIDITY' AS event_type
  FROM (VALUES
    ('DIABETES','Type 2 diabetes mellitus',44054006),
    ('DIABETES','Type 1 diabetes mellitus',46635009),
    ('DIABETES','Diabetes mellitus',73211009),
    ('CKD','Chronic kidney disease stage 3',433144002),
    ('CKD','Chronic kidney disease stage 4',431857002),
    ('CKD','Chronic kidney disease stage 5',433146000),
    ('CKD','Chronic kidney disease stage 3A',700378005),
    ('CKD','Chronic kidney disease stage 3B',700379002),
    ('CKD','Chronic kidney disease',709044004),
    ('ANAEMIA','Iron deficiency anaemia',87522002),
    ('ANAEMIA','Anaemia',271737000),
    ('HYPERTENSION','Essential hypertension',59621000),
    ('HYPERTENSION','Hypertension',38341003),
    ('OBESITY','Obesity',414916001),
    ('OBESITY','Body mass index 30+ - obesity',162864005),
    ('COPD','Chronic obstructive pulmonary disease',13645005),
    ('HEART FAILURE','Heart failure',84114007),
    ('HEART FAILURE','Congestive heart failure',42343007),
    ('ATRIAL FIBRILLATION','Atrial fibrillation',49436004),
    ('RECURRENT UTI','Recurrent urinary tract infection',197927001),
    ('BPH','Benign prostatic hyperplasia',266569009),
    ('BPH','Enlarged prostate',236764008),
    ('PREVIOUS CANCER','History of malignant neoplasm',417662000),
    ('PREVIOUS CANCER','Personal history of malignant neoplasm',429667007)
  )
),

-- ══════════════════════════════════════════════════════════════
-- APPROVED LAB CODES (ORIGINAL 13 + SELECTIVE NEW 10 = 23)
-- Removed: ELECTROLYTES, LIPIDS, THYROID, IMMUNOLOGY,
--          COAGULATION, VITALS (BP, pulse, SpO2, height),
--          CARDIAC, most LIVER additions
-- Kept: anaemia workup + ALP + weight/BMI + albumin
-- ══════════════════════════════════════════════════════════════
approved_labs AS (
  SELECT column1 AS category, column2 AS term, column3 AS snomed_code, 'LAB VALUE' AS event_type
  FROM (VALUES
    -- ── ORIGINAL 13 CODES (unchanged) ──
    ('RENAL','eGFR using creatinine',1011481000000105),
    ('RENAL','Serum creatinine level',1000731000000107),
    ('HAEMATOLOGY','Haemoglobin concentration',1022431000000105),
    ('INFLAMMATORY','C-reactive protein level',1015721000000105),
    ('INFLAMMATORY','Erythrocyte sedimentation rate',1018251000000107),
    ('PSA','Prostate specific antigen level',1015681000000101),
    ('LIVER','Alanine aminotransferase level',1018041000000105),
    ('LIVER','Alkaline phosphatase level',1014081000000104),
    ('METABOLIC','Serum albumin level',1017351000000104),
    ('METABOLIC','Plasma glucose level',1010671000000107),
    ('METABOLIC','HbA1c level',1019431000000105),
    ('HAEMATOLOGY','Platelet count',1022651000000102),
    ('HAEMATOLOGY','White blood cell count',1022551000000104),

    -- ══════════════════════════════════════════════════════════
    -- NEW: ANAEMIA WORKUP LABS (iron deficiency = chronic bleed)
    -- ══════════════════════════════════════════════════════════
    ('HAEMATOLOGY','MCV - Mean corpuscular volume',1022491000000106),
    ('HAEMATOLOGY','Serum ferritin level',993381000000106),
    ('HAEMATOLOGY','Serum iron level',1019351000000105),
    ('HAEMATOLOGY','Red blood cell count',1022451000000103),
    ('HAEMATOLOGY','Haematocrit',1022291000000105),

    -- NEW: ALP (bone metastasis marker)
    ('LIVER','Serum alkaline phosphatase level',1000621000000104),

    -- NEW: ALBUMIN (cancer cachexia/malnutrition marker)
    ('METABOLIC','Serum albumin level',1000821000000103),

    -- NEW: WEIGHT + BMI (weight loss trajectory)
    ('VITALS','Body weight',27113001),
    ('VITALS','Body mass index',60621009),

    -- NEW: CALCIUM (hypercalcaemia = paraneoplastic)
    ('ELECTROLYTES','Serum calcium level',1000691000000101),
    ('ELECTROLYTES','Serum adjusted calcium concentration',935051000000108)
  )
),

-- ══════════════════════════════════════════════════════════════
-- BASE PIPELINE (unchanged)
-- ══════════════════════════════════════════════════════════════
diagnostic_codes AS (
  SELECT
    TRY_TO_NUMBER(RAW_RECORDS:code_id::string) AS code_id,
    PRACTICE_ID AS source_practice_code,
    TRY_TO_NUMBER(RAW_RECORDS:snomed_c_t_concept_id::string) AS snomed_c_t_concept_id
  FROM CTSUK_BULK.RAW.CODING_CLINICALCODE
  WHERE RAW_RECORDS:snomed_c_t_concept_id IS NOT NULL
    AND TRY_TO_NUMBER(RAW_RECORDS:code_id::string) IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRY_TO_NUMBER(RAW_RECORDS:code_id::string), PRACTICE_ID
    ORDER BY TO_DATE(REGEXP_SUBSTR(file_name,'/([0-9]{8})/',1,1,'e',1),'YYYYMMDD') DESC
  ) = 1
),

patients AS (
  SELECT patient_guid, sex, date_of_birth
  FROM CTSUK_BULK.STAGING.PATIENTS_EMIS
  WHERE patient_guid IS NOT NULL AND date_of_birth IS NOT NULL
),

medical_history AS (
  SELECT
    co.RAW_RECORDS:patient_guid::string AS patient_guid,
    TRY_TO_DATE(co.RAW_RECORDS:effective_date::string) AS effective_date,
    dc.snomed_c_t_concept_id,
    COALESCE(co.RAW_RECORDS:term::string,'') AS raw_term,
    co.RAW_RECORDS:associated_text::string AS associated_text,
    TRY_TO_DOUBLE(co.RAW_RECORDS:value::string) AS numeric_value
  FROM CTSUK_BULK.RAW.CARERECORD_OBSERVATION AS co
  INNER JOIN diagnostic_codes AS dc
    ON TRY_TO_NUMBER(co.RAW_RECORDS:code_id::string) = dc.code_id
   AND co.practice_id = dc.source_practice_code
  CROSS JOIN params p
  WHERE TRY_TO_DATE(co.RAW_RECORDS:effective_date::string) IS NOT NULL
    AND TRY_TO_DATE(co.RAW_RECORDS:effective_date::string) >= p.mh_start
    AND TRY_TO_DATE(co.RAW_RECORDS:effective_date::string) < p.mh_end
    AND dc.snomed_c_t_concept_id IS NOT NULL
),

cancer_patients AS (
  SELECT mh.patient_guid, MIN(mh.effective_date) AS date_of_diagnosis
  FROM medical_history mh
  INNER JOIN analytics.cts.dim_cancer_codes dcc ON mh.snomed_c_t_concept_id = dcc.snomed_code
  WHERE LOWER(dcc.cancer_id) NOT LIKE '%disease%' AND LOWER(dcc.cancer_id) LIKE '%bladder%'
  GROUP BY 1
  HAVING MIN(mh.effective_date) IS NOT NULL
),

all_cancer_patients AS (
  SELECT DISTINCT mh.patient_guid
  FROM medical_history mh
  INNER JOIN analytics.cts.dim_cancer_codes dcc ON mh.snomed_c_t_concept_id = dcc.snomed_code
  WHERE LOWER(dcc.cancer_id) NOT LIKE '%disease%'
),

cancer_profile AS (
  SELECT cp.patient_guid, pp.sex,
    FLOOR(DATEDIFF(year,pp.date_of_birth,cp.date_of_diagnosis)/5)*5 AS age_band
  FROM cancer_patients cp
  INNER JOIN patients pp ON pp.patient_guid = cp.patient_guid
),

cancer_age_sex AS (
  SELECT sex, age_band, COUNT(*) AS cancer_count FROM cancer_profile GROUP BY 1,2
),

non_cancer_last_record AS (
  SELECT patient_guid, MAX(effective_date) AS last_record_date FROM medical_history GROUP BY 1
),

non_cancer_pool AS (
  SELECT DISTINCT
    p.patient_guid, p.date_of_birth, p.sex,
    lr.last_record_date AS pseudo_index_date,
    FLOOR(DATEDIFF(year,p.date_of_birth,lr.last_record_date)/5)*5 AS age_band
  FROM patients p
  INNER JOIN non_cancer_last_record lr ON lr.patient_guid = p.patient_guid
  CROSS JOIN params param
  WHERE DATEDIFF(year,p.date_of_birth,lr.last_record_date) > 18
    AND NOT EXISTS (SELECT 1 FROM all_cancer_patients cp WHERE cp.patient_guid = p.patient_guid)
    AND NOT EXISTS (SELECT 1 FROM medical_history pmh WHERE pmh.patient_guid = p.patient_guid AND pmh.snomed_c_t_concept_id = 1403151000000103)
    AND DATEADD(month,-36,lr.last_record_date) >= param.mh_start
    AND lr.last_record_date <= param.mh_end
),

non_cancer_patients AS (
  SELECT ncp.patient_guid, ncp.date_of_birth, ncp.sex, ncp.pseudo_index_date
  FROM non_cancer_pool ncp
  INNER JOIN cancer_age_sex cas ON ncp.sex = cas.sex AND ncp.age_band = cas.age_band
  QUALIFY ROW_NUMBER() OVER (PARTITION BY ncp.sex, ncp.age_band ORDER BY MD5(ncp.patient_guid)) <= cas.cancer_count * 10
),

pos_obs_events AS (
  SELECT
    cp.patient_guid, pp.sex, 'BLADDER' AS cancer_id, cp.date_of_diagnosis,
    cp.date_of_diagnosis AS index_date,
    DATEDIFF(year,pp.date_of_birth,cp.date_of_diagnosis) AS age_at_diagnosis,
    DATEDIFF(year,pp.date_of_birth,cp.date_of_diagnosis) AS age_at_index,
    mh.effective_date AS event_date,
    DATEDIFF(year,pp.date_of_birth,mh.effective_date) AS event_age,
    DATEDIFF(month,mh.effective_date,cp.date_of_diagnosis) AS months_before_index,
    CASE
      WHEN DATEDIFF(month,mh.effective_date,cp.date_of_diagnosis) BETWEEN 24 AND 36 THEN 'A'
      WHEN DATEDIFF(month,mh.effective_date,cp.date_of_diagnosis) BETWEEN 12 AND 23 THEN 'B'
    END AS time_window,
    ac.event_type, ac.category, ac.snomed_code AS snomed_id, ac.term,
    mh.associated_text, mh.numeric_value AS value, 1 AS label
  FROM cancer_patients cp
  INNER JOIN medical_history mh ON mh.patient_guid = cp.patient_guid
  INNER JOIN patients pp ON pp.patient_guid = cp.patient_guid
  INNER JOIN (SELECT category,term,snomed_code,event_type FROM approved_obs
              UNION ALL SELECT category,term,snomed_code,event_type FROM approved_labs) ac
    ON mh.snomed_c_t_concept_id = ac.snomed_code
  CROSS JOIN params p
  WHERE mh.effective_date >= DATEADD(month,-p.window_start_months,cp.date_of_diagnosis)
    AND mh.effective_date < DATEADD(month,-p.window_end_months,cp.date_of_diagnosis)
),

pos_rf_events AS (
  SELECT
    cp.patient_guid, pp.sex, 'BLADDER' AS cancer_id, cp.date_of_diagnosis,
    cp.date_of_diagnosis AS index_date,
    DATEDIFF(year,pp.date_of_birth,cp.date_of_diagnosis) AS age_at_diagnosis,
    DATEDIFF(year,pp.date_of_birth,cp.date_of_diagnosis) AS age_at_index,
    mh.effective_date AS event_date,
    DATEDIFF(year,pp.date_of_birth,mh.effective_date) AS event_age,
    DATEDIFF(month,mh.effective_date,cp.date_of_diagnosis) AS months_before_index,
    CASE
      WHEN ac.event_type = 'RISK FACTOR' THEN 'RF'
      WHEN ac.event_type = 'COMORBIDITY' THEN 'COMORB'
    END AS time_window,
    ac.event_type, ac.category, ac.snomed_code AS snomed_id, ac.term,
    mh.associated_text, mh.numeric_value AS value, 1 AS label
  FROM cancer_patients cp
  INNER JOIN medical_history mh ON mh.patient_guid = cp.patient_guid
  INNER JOIN patients pp ON pp.patient_guid = cp.patient_guid
  INNER JOIN (SELECT category,term,snomed_code,event_type FROM approved_risk_factors
              UNION ALL SELECT category,term,snomed_code,event_type FROM approved_comorbidities) ac
    ON mh.snomed_c_t_concept_id = ac.snomed_code
  CROSS JOIN params p
  WHERE mh.effective_date < DATEADD(month,-p.window_end_months,cp.date_of_diagnosis)
),

neg_obs_events AS (
  SELECT
    ncp.patient_guid, ncp.sex, 'NONE' AS cancer_id, NULL::date AS date_of_diagnosis,
    ncp.pseudo_index_date AS index_date, NULL::int AS age_at_diagnosis,
    DATEDIFF(year,ncp.date_of_birth,ncp.pseudo_index_date) AS age_at_index,
    mh.effective_date AS event_date,
    DATEDIFF(year,ncp.date_of_birth,mh.effective_date) AS event_age,
    DATEDIFF(month,mh.effective_date,ncp.pseudo_index_date) AS months_before_index,
    CASE
      WHEN DATEDIFF(month,mh.effective_date,ncp.pseudo_index_date) BETWEEN 24 AND 36 THEN 'A'
      WHEN DATEDIFF(month,mh.effective_date,ncp.pseudo_index_date) BETWEEN 12 AND 23 THEN 'B'
    END AS time_window,
    ac.event_type, ac.category, ac.snomed_code AS snomed_id, ac.term,
    mh.associated_text, mh.numeric_value AS value, 0 AS label
  FROM non_cancer_patients ncp
  INNER JOIN medical_history mh ON mh.patient_guid = ncp.patient_guid
  INNER JOIN (SELECT category,term,snomed_code,event_type FROM approved_obs
              UNION ALL SELECT category,term,snomed_code,event_type FROM approved_labs) ac
    ON mh.snomed_c_t_concept_id = ac.snomed_code
  CROSS JOIN params p
  WHERE mh.effective_date >= DATEADD(month,-p.window_start_months,ncp.pseudo_index_date)
    AND mh.effective_date < DATEADD(month,-p.window_end_months,ncp.pseudo_index_date)
),

neg_rf_events AS (
  SELECT
    ncp.patient_guid, ncp.sex, 'NONE' AS cancer_id, NULL::date AS date_of_diagnosis,
    ncp.pseudo_index_date AS index_date, NULL::int AS age_at_diagnosis,
    DATEDIFF(year,ncp.date_of_birth,ncp.pseudo_index_date) AS age_at_index,
    mh.effective_date AS event_date,
    DATEDIFF(year,ncp.date_of_birth,mh.effective_date) AS event_age,
    DATEDIFF(month,mh.effective_date,ncp.pseudo_index_date) AS months_before_index,
    CASE
      WHEN ac.event_type = 'RISK FACTOR' THEN 'RF'
      WHEN ac.event_type = 'COMORBIDITY' THEN 'COMORB'
    END AS time_window,
    ac.event_type, ac.category, ac.snomed_code AS snomed_id, ac.term,
    mh.associated_text, mh.numeric_value AS value, 0 AS label
  FROM non_cancer_patients ncp
  INNER JOIN medical_history mh ON mh.patient_guid = ncp.patient_guid
  INNER JOIN (SELECT category,term,snomed_code,event_type FROM approved_risk_factors
              UNION ALL SELECT category,term,snomed_code,event_type FROM approved_comorbidities) ac
    ON mh.snomed_c_t_concept_id = ac.snomed_code
  CROSS JOIN params p
  WHERE mh.effective_date < DATEADD(month,-p.window_end_months,ncp.pseudo_index_date)
)

SELECT * FROM pos_obs_events
UNION ALL SELECT * FROM pos_rf_events
UNION ALL SELECT * FROM neg_obs_events
UNION ALL SELECT * FROM neg_rf_events
ORDER BY patient_guid, event_date;
