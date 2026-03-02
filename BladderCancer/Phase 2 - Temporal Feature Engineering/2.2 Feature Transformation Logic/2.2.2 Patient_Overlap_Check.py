WITH params AS (
  SELECT
    DATE '1950-01-01' AS mh_start,
    DATE '2026-02-25' AS mh_end,
    36 AS window_start_months,
    12 AS window_end_months,
    24 AS window_mid_months
),

approved_meds AS (
  SELECT column1 AS category, column2 AS term, column3 AS match_pattern, column4 AS dmd_code
  FROM (VALUES
    -- ── ORIGINAL UTI ANTIBIOTICS ──
    ('UTI ANTIBIOTICS','Nitrofurantoin 100mg MR capsules','%nitrofurantoin%100mg%modified-release%capsule%',39108511000001108),
    ('UTI ANTIBIOTICS','Nitrofurantoin 100mg tablets','%nitrofurantoin%100mg%tablet%',41952911000001104),
    ('UTI ANTIBIOTICS','Nitrofurantoin 50mg capsules','%nitrofurantoin%50mg%capsule%',41953111000001108),
    ('UTI ANTIBIOTICS','Nitrofurantoin 50mg tablets','%nitrofurantoin%50mg%tablet%',41953211000001102),
    ('UTI ANTIBIOTICS','Trimethoprim 200mg tablets','%trimethoprim%200mg%tablet%',41956211000001109),
    ('UTI ANTIBIOTICS','Trimethoprim 100mg tablets','%trimethoprim%100mg%tablet%',41956111000001103),
    ('UTI ANTIBIOTICS','Pivmecillinam 200mg tablets','%pivmecillinam%200mg%tablet%',39702711000001106),
    ('UTI ANTIBIOTICS','Cefalexin 500mg capsules','%cefalexin%500mg%capsule%',39694811000001102),
    ('UTI ANTIBIOTICS','Cefalexin 500mg tablets','%cefalexin%500mg%tablet%',39735911000001106),
    ('UTI ANTIBIOTICS','Cefalexin 250mg capsules','%cefalexin%250mg%capsule%',39735311000001105),
    ('UTI ANTIBIOTICS','Cefradine 500mg capsules','%cefradine%500mg%capsule%',41944611000001104),
    ('UTI ANTIBIOTICS','Ciprofloxacin 500mg tablets','%ciprofloxacin%500mg%tablet%',39687811000001107),
    ('UTI ANTIBIOTICS','Ciprofloxacin 250mg tablets','%ciprofloxacin%250mg%tablet%',39687511000001109),
    ('UTI ANTIBIOTICS','Ciprofloxacin 750mg tablets','%ciprofloxacin%750mg%tablet%',39687011000001104),
    ('UTI ANTIBIOTICS','Ofloxacin 200mg tablets','%ofloxacin%200mg%tablet%',41953411000001104),
    ('UTI ANTIBIOTICS','Fosfomycin 3g granules','%fosfomycin%3g%granule%',18162111000001102),
    ('UTI ANTIBIOTICS','Co-amoxiclav 500mg/125mg tablets','%co-amoxiclav%500mg%125mg%tablet%',39732211000001107),
    ('UTI ANTIBIOTICS','Co-amoxiclav 250mg/125mg tablets','%co-amoxiclav%250mg%125mg%tablet%',39732111000001101),
    ('UTI ANTIBIOTICS','Co-trimoxazole 80mg/400mg tablets','%co-trimoxazole%80mg%400mg%tablet%',41947411000001104),
    ('UTI ANTIBIOTICS','Methenamine hippurate 1g tablets','%methenamine%hippurate%1g%tablet%',41951711000001104),
    ('UTI ANTIBIOTICS','Doxycycline 100mg capsules','%doxycycline%100mg%capsule%',41948311000001104),

    -- ── ORIGINAL CATHETER SUPPLIES - NIGHT BAGS ──
    ('CATHETER SUPPLIES - NIGHT BAGS','Hollister drainable night drainage bag 5550','%hollister%night%drainage%bag%',6771411000001109),
    ('CATHETER SUPPLIES - NIGHT BAGS','Spirit One MT drainable night drainage bag DSP1MT30 2L','%spirit%night%drainage%bag%',40065811000001104),
    ('CATHETER SUPPLIES - NIGHT BAGS','Spirit Bed bag drainable night drainage bag DSBB2000 2L','%spirit%bed%bag%drainage%',34049411000001100),
    ('CATHETER SUPPLIES - NIGHT BAGS','Simpla S2 non-drainable night drainage bag 2L','%simpla%s2%night%drainage%bag%',6754711000001103),
    ('CATHETER SUPPLIES - NIGHT BAGS','Simpla Profile drainable night drainage bag 2L','%simpla%profile%night%drainage%bag%',11368111000001103),
    ('CATHETER SUPPLIES - NIGHT BAGS','Prosys single use non-sterile drainable night bag 2L','%prosys%non-sterile%night%bag%',11432911000001100),
    ('CATHETER SUPPLIES - NIGHT BAGS','Prosys sterile drainable night bag with lever tap 2L','%prosys%sterile%night%bag%lever%',16622011000001104),
    ('CATHETER SUPPLIES - NIGHT BAGS','Uriplan sterile leg bag D5L 500ml','%uriplan%leg%bag%d5l%',6528111000001105),

    -- ── ORIGINAL CATHETER SUPPLIES - LEG BAGS ──
    ('CATHETER SUPPLIES - LEG BAGS','Prosys sterile leg bag 500ml short tube','%prosys%leg%bag%500ml%short%',16619711000001100),
    ('CATHETER SUPPLIES - LEG BAGS','Prosys sterile leg bag 500ml long tube','%prosys%leg%bag%500ml%long%',16619911000001104),
    ('CATHETER SUPPLIES - LEG BAGS','LINC-Flo sterile leg bag 500ml','%linc-flo%leg%bag%500m%',20480911000001100),
    ('CATHETER SUPPLIES - LEG BAGS','Simpla Profile sterile leg bag 500ml 6cm tube','%simpla%profile%leg%bag%500ml%6cm%',11364311000001106),
    ('CATHETER SUPPLIES - LEG BAGS','Simpla Profile sterile leg bag 500ml 25cm tube','%simpla%profile%leg%bag%500ml%25cm%',11364711000001104),

    -- ── ORIGINAL CATHETER SUPPLIES - STRAPS ──
    ('CATHETER SUPPLIES - STRAPS','Ugo fix catheter strap medium 80cm','%ugo%fix%catheter%strap%',27970311000001104),
    ('CATHETER SUPPLIES - STRAPS','Prosys catheter retaining strap adult','%prosys%catheter%retaining%strap%',35623811000001100),
    ('CATHETER SUPPLIES - STRAPS','Elasticated cotton leg strap P10LS','%elasticated%cotton%leg%strap%',6484011000001100),
    ('CATHETER SUPPLIES - STRAPS','Simpla G-Strap adult 50cm','%simpla%g-strap%',6476511000001109),
    ('CATHETER SUPPLIES - STRAPS','Leg bag strap 15LS (Bard)','%leg%bag%strap%15ls%',6456111000001102),
    ('CATHETER SUPPLIES - STRAPS','Elasticated leg bag strap (Coloplast)','%elasticated%leg%bag%strap%',6474511000001103),
    ('CATHETER SUPPLIES - STRAPS','Loc-Strap catheter retaining strap','%loc-strap%catheter%retaining%',30234611000001108),

    -- ── ORIGINAL CATHETER SUPPLIES - CATHETERS ──
    ('CATHETER SUPPLIES - CATHETERS','Brillant AquaFlate catheter male 14Ch','%brillant%aquaflate%14ch%',6406211000001104),
    ('CATHETER SUPPLIES - CATHETERS','Brillant AquaFlate catheter male 12Ch','%brillant%aquaflate%12ch%',6315111000001107),
    ('CATHETER SUPPLIES - CATHETERS','Brillant AquaFlate catheter male 16Ch','%brillant%aquaflate%16ch%',6343111000001100),

    -- ── ORIGINAL CATHETER SUPPLIES - VALVES ──
    ('CATHETER SUPPLIES - VALVES','Prosys sterile catheter valve','%prosys%catheter%valve%',24542811000001108),
    ('CATHETER SUPPLIES - VALVES','Flip-Flo sterile catheter valve','%flip-flo%catheter%valve%',6437311000001103),

    -- ── ORIGINAL CATHETER MAINTENANCE ──
    ('CATHETER MAINTENANCE','OptiLube sterile lubricating jelly','%optilube%lubricating%jelly%',37575511000001104),
    ('CATHETER MAINTENANCE','Cathejell Lidocain CJLL12501','%cathejell%lidocain%',34061711000001104),
    ('CATHETER MAINTENANCE','Cathejell Lidocaine C CJL 08501','%cathejell%lidocaine%',34060611000001100),
    ('CATHETER MAINTENANCE','Instillagel gel (CliniMed)','%instillagel%',8552411000001106),
    ('CATHETER MAINTENANCE','Uro-Tainer sodium chloride 0.9%','%uro-tainer%sodium%chloride%',3494611000001100),
    ('CATHETER MAINTENANCE','OptiFlo S saline 0.9%','%optiflo%saline%',3494911000001106),
    ('CATHETER MAINTENANCE','Uro-Tainer Twin Suby G citric acid 3.23%','%uro-tainer%suby%',7430111000001102),

    -- ── ORIGINAL LUTS MEDICATIONS ──
    ('LUTS MEDICATIONS','Tamsulosin 400mcg MR capsules','%tamsulosin%400%capsule%',38754011000001109),
    ('LUTS MEDICATIONS','Tamsulosin/Dutasteride capsules','%tamsulosin%dutasteride%',17181511000001104),
    ('LUTS MEDICATIONS','Finasteride 5mg tablets','%finasteride%5mg%tablet%',42036011000001106),
    ('LUTS MEDICATIONS','Oxybutynin 2.5mg tablets','%oxybutynin%2.5mg%tablet%',42038911000001109),
    ('LUTS MEDICATIONS','Mirabegron 50mg MR tablets','%mirabegron%50mg%tablet%',38893511000001109),
    ('LUTS MEDICATIONS','Trospium chloride 60mg MR capsules','%trospium%60mg%capsule%',42044211000001104),

    -- ── ORIGINAL HAEMOSTATIC ──
    ('HAEMOSTATIC','Tranexamic acid 500mg tablets','%tranexamic%acid%500mg%tablet%',41989811000001104),

    -- ── ORIGINAL ANTICOAGULANTS ──
    ('ANTICOAGULANTS','Apixaban 2.5mg tablets','%apixaban%2.5mg%tablet%',42206411000001104),
    ('ANTICOAGULANTS','Apixaban 5mg tablets','%apixaban%5mg%tablet%',42206511000001102),
    ('ANTICOAGULANTS','Edoxaban 30mg tablets','%edoxaban%30mg%tablet%',29903311000001108),
    ('ANTICOAGULANTS','Rivaroxaban 15mg tablets','%rivaroxaban%15mg%tablet%',19842111000001101),
    ('ANTICOAGULANTS','Rivaroxaban 20mg tablets','%rivaroxaban%20mg%tablet%',19842211000001107),
    ('ANTICOAGULANTS','Warfarin 3mg tablets','%warfarin%3mg%tablet%',42217611000001104),
    ('ANTICOAGULANTS','Warfarin 500mcg tablets','%warfarin%500mcg%tablet%',42217811000001100),
    ('ANTICOAGULANTS','Warfarin 5mg tablets','%warfarin%5mg%tablet%',42217911000001105),
    ('ANTICOAGULANTS','Warfarin 1mg tablets','%warfarin%1mg%tablet%',42217511000001103),

    -- ── ORIGINAL WOUND/STOMA CARE ──
    ('WOUND/STOMA CARE','Actilite gauze dressing 5cm x 5cm','%actilite%gauze%dressing%5cm%',22232211000001104),
    ('WOUND/STOMA CARE','Povidone-Iodine fabric dressing 9.5cm','%povidone%iodine%dressing%9.5cm%',3440711000001107),
    ('WOUND/STOMA CARE','Aquacel dressing 10cm x 10cm','%aquacel%dressing%10cm%',10230711000001108),
    ('WOUND/STOMA CARE','Brava adhesive remover spray','%brava%adhesive%remover%spray%',17162011000001108),

    -- ── ORIGINAL LAXATIVES ──
    ('LAXATIVES','Laxido Orange oral powder sachets','%laxido%orange%oral%powder%sachet%',17420611000001107),
    ('LAXATIVES','CosmoCol Orange Lemon and Lime oral powder sachets','%cosmocol%orange%lemon%lime%sachet%',23621311000001100),

    -- ── ORIGINAL SMOKING CESSATION MEDS ──
    ('SMOKING CESSATION MEDS','Nicotine 21mg/24h transdermal patches','%nicotine%21mg%patch%',39022411000001104),
    ('SMOKING CESSATION MEDS','Nicotine 14mg/24h transdermal patches','%nicotine%14mg%patch%',36565311000001104),
    ('SMOKING CESSATION MEDS','Varenicline 1mg and 500mcg tablets','%varenicline%',10984311000001108),

    -- ── ORIGINAL COPD/RESPIRATORY ──
    ('COPD/RESPIRATORY','Trelegy Ellipta','%trelegy%ellipta%',34952211000001104),
    ('COPD/RESPIRATORY','Trimbow NEXThaler','%trimbow%nexthaler%',39993311000001104),

    -- ── ORIGINAL NORETHISTERONE ──
    ('NORETHISTERONE','Norethisterone 5mg tablets','%norethisterone%5mg%tablet%',42297111000001104),

    -- ══════════════════════════════════════════════════════════
    -- NEW: IRON SUPPLEMENTS (628 cancer rows — bleeding signal)
    -- ══════════════════════════════════════════════════════════
    ('IRON SUPPLEMENTS','Ferrous fumarate 210mg tablets','%ferrous%fumarate%210mg%tablet%',41984711000001104),
    ('IRON SUPPLEMENTS','Ferrous sulfate 200mg tablets','%ferrous%sulfate%200mg%tablet%',41985211000001107),
    ('IRON SUPPLEMENTS','Ferrous fumarate other','%ferrous%fumarate%',-1),
    ('IRON SUPPLEMENTS','Ferrous sulfate other','%ferrous%sulfate%',-1),
    ('IRON SUPPLEMENTS','Sytron iron','%sytron%',-1),
    ('IRON SUPPLEMENTS','Galfer capsules','%galfer%',-1),
    ('IRON SUPPLEMENTS','Ferric carboxymaltose','%ferinject%',-1),

    -- ══════════════════════════════════════════════════════════
    -- NEW: OPIOID ANALGESICS (1,364 cancer rows — pain signal)
    -- ═══════════════════════════════════════════��══════════════
    ('OPIOID ANALGESICS','Dihydrocodeine 30mg tablets','%dihydrocodeine%30mg%tablet%',39708511000001104),
    ('OPIOID ANALGESICS','Co-codamol 15mg/500mg tablets','%co%codamol%15mg%500mg%tablet%',3805611000001109),
    ('OPIOID ANALGESICS','Codeine 30mg tablets','%codeine%30mg%tablet%',42010411000001105),
    ('OPIOID ANALGESICS','Morphine sulfate oral solution','%morphine%sulfate%oral%solution%',36128311000001101),
    ('OPIOID ANALGESICS','Co-codamol 30mg/500mg','%co%codamol%30mg%500mg%',-1),
    ('OPIOID ANALGESICS','Tramadol capsules','%tramadol%capsule%',-1),
    ('OPIOID ANALGESICS','Tramadol tablets','%tramadol%tablet%',-1),
    ('OPIOID ANALGESICS','Oxycodone','%oxycodone%',-1),
    ('OPIOID ANALGESICS','Fentanyl patches','%fentanyl%patch%',-1),

    -- ══════════════════════════════════════════════════════════
    -- NEW: GI ANTISPASMODICS (248 cancer rows — bladder pain)
    -- ══════════════════════════════════════════════════════════
    ('GI ANTISPASMODICS','Buscopan 10mg tablets','%buscopan%10mg%tablet%',738611000001104),
    ('GI ANTISPASMODICS','Mebeverine 135mg tablets','%mebeverine%135mg%tablet%',42355011000001102),
    ('GI ANTISPASMODICS','Hyoscine butylbromide','%hyoscine%butylbromide%',-1),

    -- ══════════════════════════════════════════════════════════
    -- NEW: BLADDER ANTISPASMODICS (expanded beyond original)
    -- ══════════════════════════════════════════════════════════
    ('BLADDER ANTISPASMODICS','Solifenacin 5mg tablets','%solifenacin%5mg%tablet%',42042911000001105),
    ('BLADDER ANTISPASMODICS','Tolterodine','%tolterodine%',-1),
    ('BLADDER ANTISPASMODICS','Fesoterodine','%fesoterodine%',-1),

    -- ══════════════════════════════════════════════════════════
    -- NEW: URINARY RETENTION DRUGS (144 cancer rows)
    -- ══════════════════════════════════════════════════════════
    ('URINARY RETENTION DRUGS','Doxazosin 4mg MR tablets','%doxazosin%4mg%modified-release%tablet%',39020411000001106),
    ('URINARY RETENTION DRUGS','Doxazosin 4mg tablets','%doxazosin%4mg%tablet%',42209311000001101),
    ('URINARY RETENTION DRUGS','Alfuzosin','%alfuzosin%',-1),

    -- ══════════════════════════════════════════════════════════
    -- NEW: ADDITIONAL LAXATIVES (opioid side effect signal)
    -- ══════════════════════════════════════════════════════════
    ('LAXATIVES','Senna 7.5mg tablets','%senna%7.5mg%tablet%',42358311000001108),
    ('LAXATIVES','Lactulose','%lactulose%',-1),
    ('LAXATIVES','Movicol','%movicol%',-1)
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
    dc.snomed_c_t_concept_id
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
  GROUP BY 1 HAVING MIN(mh.effective_date) IS NOT NULL
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

prescribing_drugrecords_emis AS (
  SELECT
    TRIM(raw_records:drug_record_guid)::string AS drug_record_guid,
    TRIM(raw_records:patient_guid)::string AS patient_guid,
    practice_id AS source_practice_code,
    TRIM(raw_records:deleted)::string AS deleted
  FROM CTSUK_BULK.RAW.PRESCRIBING_DRUGRECORD
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRIM(raw_records:drug_record_guid)
    ORDER BY TO_DATE(REGEXP_SUBSTR(file_name,'/([0-9]{8})/',1,1,'e',1),'YYYYMMDD') DESC
  ) = 1
),

prescribing_issuerecords_emis AS (
  SELECT
    TRIM(raw_records:drug_record_guid)::string AS drug_record_guid,
    TRY_TO_DATE(TRIM(raw_records:effective_date)) AS effective_date,
    TRY_TO_NUMBER(TRIM(raw_records:code_id)) AS code_id,
    TRY_TO_DOUBLE(TRIM(raw_records:quantity)) AS quantity,
    TRIM(raw_records:dosage)::string AS dosage,
    TRY_TO_NUMBER(TRIM(raw_records:duration_in_days)) AS duration_in_days,
    TRIM(raw_records:deleted)::string AS deleted
  FROM CTSUK_BULK.RAW.PRESCRIBING_ISSUERECORD
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRIM(raw_records:drug_record_guid)
    ORDER BY TO_DATE(REGEXP_SUBSTR(file_name,'/([0-9]{8})/',1,1,'e',1),'YYYYMMDD') DESC
  ) = 1
),

codes_emis AS (
  SELECT
    TRY_TO_NUMBER(TRIM(raw_records:code_id)) AS code_id,
    TRY_TO_NUMBER(TRIM(raw_records:dmd_product_code_id)) AS dmd_product_code_id,
    TRIM(raw_records:term)::string AS term,
    practice_id AS source_practice_code
  FROM CTSUK_BULK.RAW.CODING_DRUGCODE
  WHERE TRIM(raw_records:dmd_product_code_id) IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRY_TO_NUMBER(TRIM(raw_records:code_id)), practice_id
    ORDER BY TO_DATE(REGEXP_SUBSTR(file_name,'/([0-9]{8})/',1,1,'e',1),'YYYYMMDD') DESC
  ) = 1
),

final_medication AS (
  SELECT
    dr.patient_guid, ir.effective_date,
    ce.dmd_product_code_id AS dmd_code, ce.term AS drug_term,
    ir.quantity, ir.dosage, ir.duration_in_days
  FROM prescribing_drugrecords_emis dr
  INNER JOIN prescribing_issuerecords_emis ir ON dr.drug_record_guid = ir.drug_record_guid
  INNER JOIN codes_emis ce ON ir.code_id = ce.code_id AND dr.source_practice_code = ce.source_practice_code
  WHERE dr.deleted = 'false' AND ir.deleted = 'false'
    AND ir.effective_date IS NOT NULL AND ce.term IS NOT NULL
),

pos_med_events AS (
  SELECT
    cp.patient_guid, pp.sex, 'BLADDER' AS cancer_id, cp.date_of_diagnosis,
    cp.date_of_diagnosis AS index_date,
    DATEDIFF(year,pp.date_of_birth,cp.date_of_diagnosis) AS age_at_diagnosis,
    DATEDIFF(year,pp.date_of_birth,cp.date_of_diagnosis) AS age_at_index,
    fm.effective_date AS event_date,
    DATEDIFF(year,pp.date_of_birth,fm.effective_date) AS event_age,
    DATEDIFF(month,fm.effective_date,cp.date_of_diagnosis) AS months_before_index,
    CASE
      WHEN DATEDIFF(month,fm.effective_date,cp.date_of_diagnosis) BETWEEN 24 AND 36 THEN 'A'
      WHEN DATEDIFF(month,fm.effective_date,cp.date_of_diagnosis) BETWEEN 12 AND 23 THEN 'B'
    END AS time_window,
    'MEDICATION' AS event_type,
    am.category,
    am.dmd_code AS snomed_id,
    am.term,
    fm.dosage AS associated_text,
    fm.quantity AS value,
    fm.duration_in_days,
    1 AS label
  FROM cancer_patients cp
  INNER JOIN final_medication fm ON fm.patient_guid = cp.patient_guid
  INNER JOIN patients pp ON pp.patient_guid = cp.patient_guid
  INNER JOIN approved_meds am
    ON LOWER(TRIM(fm.drug_term)) LIKE am.match_pattern
    OR (am.dmd_code > 0 AND fm.dmd_code = am.dmd_code)
  CROSS JOIN params p
  WHERE fm.effective_date >= DATEADD(month,-p.window_start_months,cp.date_of_diagnosis)
    AND fm.effective_date < DATEADD(month,-p.window_end_months,cp.date_of_diagnosis)
),

neg_med_events AS (
  SELECT
    ncp.patient_guid, ncp.sex, 'NONE' AS cancer_id, NULL::date AS date_of_diagnosis,
    ncp.pseudo_index_date AS index_date, NULL::int AS age_at_diagnosis,
    DATEDIFF(year,ncp.date_of_birth,ncp.pseudo_index_date) AS age_at_index,
    fm.effective_date AS event_date,
    DATEDIFF(year,ncp.date_of_birth,fm.effective_date) AS event_age,
    DATEDIFF(month,fm.effective_date,ncp.pseudo_index_date) AS months_before_index,
    CASE
      WHEN DATEDIFF(month,fm.effective_date,ncp.pseudo_index_date) BETWEEN 24 AND 36 THEN 'A'
      WHEN DATEDIFF(month,fm.effective_date,ncp.pseudo_index_date) BETWEEN 12 AND 23 THEN 'B'
    END AS time_window,
    'MEDICATION' AS event_type,
    am.category,
    am.dmd_code AS snomed_id,
    am.term,
    fm.dosage AS associated_text,
    fm.quantity AS value,
    fm.duration_in_days,
    0 AS label
  FROM non_cancer_patients ncp
  INNER JOIN final_medication fm ON fm.patient_guid = ncp.patient_guid
  INNER JOIN approved_meds am
    ON LOWER(TRIM(fm.drug_term)) LIKE am.match_pattern
    OR (am.dmd_code > 0 AND fm.dmd_code = am.dmd_code)
  CROSS JOIN params p
  WHERE fm.effective_date >= DATEADD(month,-p.window_start_months,ncp.pseudo_index_date)
    AND fm.effective_date < DATEADD(month,-p.window_end_months,ncp.pseudo_index_date)
)

SELECT * FROM pos_med_events
UNION ALL
SELECT * FROM neg_med_events
ORDER BY patient_guid, event_date;
