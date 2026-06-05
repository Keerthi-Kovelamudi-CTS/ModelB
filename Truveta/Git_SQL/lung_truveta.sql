WITH params AS (
  SELECT
    5    AS years_before,
    1    AS months_before,
    50   AS min_obs_events_per_patient,
    10   AS min_med_events_per_patient,
    1    AS non_cancer_ratio,
    DATE '2026-01-01' AS reference_date
),

lung_cancer_concepts AS (
  SELECT ConceptId, ConceptName
  FROM `prj-cts-ai-dev-sp.truveta_silver_lung.Concept`
  WHERE Domain = 'Condition'
    AND (
      (CodeSystem = 'ICD10CM' AND REGEXP_CONTAINS(ConceptCode, r'^C34'))
      OR (CodeSystem = 'SNOMED CT' AND LOWER(ConceptName) LIKE '%lung%'
          AND (LOWER(ConceptName) LIKE '%cancer%'
            OR LOWER(ConceptName) LIKE '%carcinoma%'
            OR LOWER(ConceptName) LIKE '%malignant neoplasm%'))
    )
),

any_cancer_concepts AS (
    SELECT ConceptId
    FROM `prj-cts-ai-dev-sp.truveta_silver_lung.Concept`
    WHERE
      -- ICD-10-CM: diagnosis + history + therapy-encounter + family-history
      (CodeSystem = 'ICD10CM' AND (
        REGEXP_CONTAINS(ConceptCode, r'^(C[0-9]|D0[0-9]|D3[7-9]|D4[0-8])')
        OR REGEXP_CONTAINS(ConceptCode, r'^Z85')           -- personal history of malignancy
        OR REGEXP_CONTAINS(ConceptCode, r'^Z51\.[01]')     -- encounter for chemo / radiation
        OR REGEXP_CONTAINS(ConceptCode, r'^Z08')           -- follow-up after malignant neoplasm
        OR REGEXP_CONTAINS(ConceptCode, r'^Z80')           -- family history of malignancy
      ))
      -- SNOMED CT: broaden keyword list, drop Domain filter
      OR (CodeSystem = 'SNOMED CT' AND (
        LOWER(ConceptName) LIKE '%cancer%' OR
        LOWER(ConceptName) LIKE '%neoplas%' OR
        LOWER(ConceptName) LIKE '%carcinoma%' OR
        LOWER(ConceptName) LIKE '%lymphoma%' OR
        LOWER(ConceptName) LIKE '%leukaemia%' OR LOWER(ConceptName) LIKE '%leukemia%' OR
        LOWER(ConceptName) LIKE '%melanoma%' OR
        LOWER(ConceptName) LIKE '%sarcoma%' OR
        LOWER(ConceptName) LIKE '%tumor%' OR LOWER(ConceptName) LIKE '%tumour%' OR
        LOWER(ConceptName) LIKE '%metasta%' OR
        LOWER(ConceptName) LIKE '%antineoplastic%' OR
        LOWER(ConceptName) LIKE '%chemotherap%' OR
        LOWER(ConceptName) LIKE '%radiotherap%' OR
        LOWER(ConceptName) LIKE '%radiation therap%' OR
        LOWER(ConceptName) LIKE '%tnm stag%' OR LOWER(ConceptName) LIKE '%cancer stag%'
      ))
      -- CPT / HCPCS: cancer-specific procedures
      OR (CodeSystem IN ('CPT', 'HCPCS', 'ICD10PCS') AND (
        LOWER(ConceptName) LIKE '%antineoplastic%' OR
        LOWER(ConceptName) LIKE '%chemotherap%' OR
        LOWER(ConceptName) LIKE '%radiotherap%' OR
        LOWER(ConceptName) LIKE '%radiation therap%' OR
        LOWER(ConceptName) LIKE '%radiation treatment%' OR
        LOWER(ConceptName) LIKE '%imrt%' OR
        LOWER(ConceptName) LIKE '%brachytherap%' OR
        LOWER(ConceptName) LIKE '%mohs%' OR
        LOWER(ConceptName) LIKE '%tumor%' OR
        LOWER(ConceptName) LIKE '%neoplas%' OR
        LOWER(ConceptName) LIKE '%medical physics consultation%'
      ))
      -- LOINC: cancer-specific assessments
      OR (CodeSystem = 'LOINC' AND (
        LOWER(ConceptName) LIKE '%ecog%' OR
        LOWER(ConceptName) LIKE '%karnofsky%' OR
        LOWER(ConceptName) LIKE '%tumor%' OR
        LOWER(ConceptName) LIKE '%cancer%' OR
        LOWER(ConceptName) LIKE '%neoplas%'
      ))
  ),

patient_location AS (
    SELECT
      'lung' AS cohort,
      pl.PersonId AS patient_id,
      sp.ConceptName AS state_or_province,
      loc.PostalOrZipCode AS postal_code,
      addr_use.ConceptName AS address_use
    FROM `prj-cts-ai-dev-sp.truveta_silver_lung.PersonLocation` pl
    JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.Location` loc        ON pl.LocationId = loc.Id
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.Concept` sp       ON loc.StateOrProvinceConceptId = sp.ConceptId
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.Concept` addr_use ON pl.UseConceptId             = addr_use.ConceptId
    QUALIFY ROW_NUMBER() OVER (
      PARTITION BY pl.PersonId
      ORDER BY pl.EffectiveStartDateTime DESC NULLS LAST
    ) = 1

    UNION ALL
  
    SELECT
      'non_cancer',
      pl.PersonId,
      sp.ConceptName,
      loc.PostalOrZipCode,
      addr_use.ConceptName
    FROM `prj-cts-ai-dev-sp.truveta_silver_non_cancer.PersonLocation` pl
    JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Location` loc        ON pl.LocationId = loc.Id
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Concept` sp       ON loc.StateOrProvinceConceptId = sp.ConceptId
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Concept` addr_use ON pl.UseConceptId             = addr_use.ConceptId
    QUALIFY ROW_NUMBER() OVER (
      PARTITION BY pl.PersonId
      ORDER BY pl.EffectiveStartDateTime DESC NULLS LAST
    ) = 1
  ),

/* ────────────────────────────────────────────────────────────────────────
   Per-cohort medical history. Each event source is deduplicated by picking
   one canonical concept per event Id (vocabulary preference per domain).
   ──────────────────────────────────────────────────────────────────────── */

medical_history_lung AS (
  /* Conditions */
  SELECT
    c.Id AS source_record_id, c.PersonId AS patient_id,
    DATE(COALESCE(c.OnsetDateTime, c.RecordedDateTime)) AS event_date,
    DATE(c.RecordedDateTime) AS recorded_date,
    co.ConceptId AS concept_id, co.ConceptCode AS source_concept_id,
    co.CodeSystem AS source_vocabulary, co.ConceptName AS term,
    CAST(NULL AS NUMERIC) AS value_numeric, CAST(NULL AS INT64) AS value_concept_id,
    'condition' AS domain
  FROM `prj-cts-ai-dev-sp.truveta_silver_lung.Condition` c
  JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.ConditionCodeConceptMap` cm ON c.CodeConceptMapId = cm.Id
  JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.Concept` co ON cm.CodeConceptId = co.ConceptId
  WHERE COALESCE(c.OnsetDateTime, c.RecordedDateTime) IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY c.Id ORDER BY
    CASE co.CodeSystem WHEN 'SNOMED CT' THEN 1 WHEN 'ICD10CM' THEN 2 ELSE 3 END,
    co.ConceptId) = 1

  UNION ALL

  /* Observations */
  SELECT
    o.Id, o.PersonId,
    DATE(o.EffectiveDateTime), DATE(o.RecordedDateTime),
    co.ConceptId, co.ConceptCode, co.CodeSystem, co.ConceptName,
    o.NormalizedValueNumeric, o.NormalizedValueConceptId,
    'observation'
  FROM `prj-cts-ai-dev-sp.truveta_silver_lung.Observation` o
  JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.ObservationCodeConceptMap` cm ON o.CodeConceptMapId = cm.Id
  JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.Concept` co ON cm.CodeConceptId = co.ConceptId
  WHERE o.EffectiveDateTime IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY o.Id ORDER BY
    CASE co.CodeSystem WHEN 'SNOMED CT' THEN 1 WHEN 'LOINC' THEN 2 WHEN 'ICD10CM' THEN 3 ELSE 4 END,
    co.ConceptId) = 1

  UNION ALL

  /* Lab results */
  SELECT
    l.Id, l.PersonId,
    DATE(l.EffectiveDateTime), DATE(l.RecordedDateTime),
    co.ConceptId, co.ConceptCode, co.CodeSystem, co.ConceptName,
    l.NormalizedValueNumeric, l.NormalizedValueConceptId,
    'lab_result'
  FROM `prj-cts-ai-dev-sp.truveta_silver_lung.LabResult` l
  JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.LabResultCodeConceptMap` cm ON l.CodeConceptMapId = cm.Id
  JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.Concept` co ON cm.CodeConceptId = co.ConceptId
  WHERE l.EffectiveDateTime IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY l.Id ORDER BY
    CASE co.CodeSystem WHEN 'LOINC' THEN 1 WHEN 'SNOMED CT' THEN 2 ELSE 3 END,
    co.ConceptId) = 1

  UNION ALL

  /* Procedures */
  SELECT
    p.Id, p.PersonId,
    DATE(p.StartDateTime), DATE(p.RecordedDateTime),
    co.ConceptId, co.ConceptCode, co.CodeSystem, co.ConceptName,
    NULL, NULL,
    'procedure'
  FROM `prj-cts-ai-dev-sp.truveta_silver_lung.Procedure` p
  JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.ProcedureCodeConceptMap` cm ON p.CodeConceptMapId = cm.Id
  JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.Concept` co ON cm.CodeConceptId = co.ConceptId
  WHERE p.StartDateTime IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY p.Id ORDER BY
    CASE co.CodeSystem WHEN 'SNOMED CT' THEN 1 WHEN 'ICD10CM' THEN 2 ELSE 3 END,
    co.ConceptId) = 1
),

medical_history_non_cancer AS (
  /* Conditions */
  SELECT
    c.Id AS source_record_id, c.PersonId AS patient_id,
    DATE(COALESCE(c.OnsetDateTime, c.RecordedDateTime)) AS event_date,
    DATE(c.RecordedDateTime) AS recorded_date,
    co.ConceptId AS concept_id, co.ConceptCode AS source_concept_id,
    co.CodeSystem AS source_vocabulary, co.ConceptName AS term,
    CAST(NULL AS NUMERIC) AS value_numeric, CAST(NULL AS INT64) AS value_concept_id,
    'condition' AS domain
  FROM `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Condition` c
  JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.ConditionCodeConceptMap` cm ON c.CodeConceptMapId = cm.Id
  JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Concept` co ON cm.CodeConceptId = co.ConceptId
  WHERE COALESCE(c.OnsetDateTime, c.RecordedDateTime) IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY c.Id ORDER BY
    CASE co.CodeSystem WHEN 'SNOMED CT' THEN 1 WHEN 'ICD10CM' THEN 2 ELSE 3 END,
    co.ConceptId) = 1

  UNION ALL

  /* Observations */
  SELECT
    o.Id, o.PersonId,
    DATE(o.EffectiveDateTime), DATE(o.RecordedDateTime),
    co.ConceptId, co.ConceptCode, co.CodeSystem, co.ConceptName,
    o.NormalizedValueNumeric, o.NormalizedValueConceptId,
    'observation'
  FROM `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Observation` o
  JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.ObservationCodeConceptMap` cm ON o.CodeConceptMapId = cm.Id
  JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Concept` co ON cm.CodeConceptId = co.ConceptId
  WHERE o.EffectiveDateTime IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY o.Id ORDER BY
    CASE co.CodeSystem WHEN 'SNOMED CT' THEN 1 WHEN 'LOINC' THEN 2 WHEN 'ICD10CM' THEN 3 ELSE 4 END,
    co.ConceptId) = 1

  UNION ALL

  /* Lab results */
  SELECT
    l.Id, l.PersonId,
    DATE(l.EffectiveDateTime), DATE(l.RecordedDateTime),
    co.ConceptId, co.ConceptCode, co.CodeSystem, co.ConceptName,
    l.NormalizedValueNumeric, l.NormalizedValueConceptId,
    'lab_result'
  FROM `prj-cts-ai-dev-sp.truveta_silver_non_cancer.LabResult` l
  JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.LabResultCodeConceptMap` cm ON l.CodeConceptMapId = cm.Id
  JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Concept` co ON cm.CodeConceptId = co.ConceptId
  WHERE l.EffectiveDateTime IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY l.Id ORDER BY
    CASE co.CodeSystem WHEN 'LOINC' THEN 1 WHEN 'SNOMED CT' THEN 2 ELSE 3 END,
    co.ConceptId) = 1

  UNION ALL

  /* Procedures */
  SELECT
    p.Id, p.PersonId,
    DATE(p.StartDateTime), DATE(p.RecordedDateTime),
    co.ConceptId, co.ConceptCode, co.CodeSystem, co.ConceptName,
    NULL, NULL,
    'procedure'
  FROM `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Procedure` p
  JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.ProcedureCodeConceptMap` cm ON p.CodeConceptMapId = cm.Id
  JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Concept` co ON cm.CodeConceptId = co.ConceptId
  WHERE p.StartDateTime IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY p.Id ORDER BY
    CASE co.CodeSystem WHEN 'SNOMED CT' THEN 1 WHEN 'ICD10CM' THEN 2 ELSE 3 END,
    co.ConceptId) = 1
),

medical_history AS (
  SELECT 'lung'       AS cohort, * FROM medical_history_lung
  UNION ALL
  SELECT 'non_cancer' AS cohort, * FROM medical_history_non_cancer
),

target_cancer_patients AS (
  SELECT patient_id, date_of_diagnosis, cancer_term
  FROM (
    SELECT
      mh.patient_id,
      mh.event_date AS date_of_diagnosis,
      mh.term       AS cancer_term,
      ROW_NUMBER() OVER (PARTITION BY mh.patient_id ORDER BY mh.event_date) AS rn
    FROM medical_history mh
    JOIN lung_cancer_concepts lcc ON mh.concept_id = lcc.ConceptId
    WHERE mh.cohort = 'lung'
      AND mh.domain = 'condition'
  )
  WHERE rn = 1
),

  cancer_anchor_pool AS (
    SELECT
      date_of_diagnosis,
      ROW_NUMBER() OVER (ORDER BY FARM_FINGERPRINT(CAST(patient_id AS STRING))) AS pool_rn
    FROM target_cancer_patients
  ),


  non_cancer_with_anchor AS (
    SELECT
      ncp.patient_id,
      cap.date_of_diagnosis AS anchor_date
    FROM (
      SELECT
        patient_id,
        MOD(ABS(FARM_FINGERPRINT(CAST(patient_id AS STRING))),
            (SELECT COUNT(*) FROM cancer_anchor_pool)) + 1 AS assigned_rn
      FROM (
        SELECT DISTINCT patient_id
        FROM medical_history
        WHERE cohort = 'non_cancer'
      )
    ) ncp
    JOIN cancer_anchor_pool cap ON ncp.assigned_rn = cap.pool_rn
  ),

cancer_event_counts AS (
  SELECT patient_id, anchor_date, cancer_term, COUNT(*) AS obs_event_count
  FROM (
    SELECT DISTINCT
      tcp.patient_id,
      tcp.date_of_diagnosis AS anchor_date,
      tcp.cancer_term,
      mh.event_date, mh.concept_id, mh.term, mh.value_numeric
    FROM target_cancer_patients tcp
    JOIN medical_history mh
      ON mh.patient_id = tcp.patient_id AND mh.cohort = 'lung'
    CROSS JOIN params p
    WHERE 
         mh.event_date >= DATE_SUB(tcp.date_of_diagnosis, INTERVAL p.years_before  YEAR)
      AND mh.event_date <  DATE_SUB(tcp.date_of_diagnosis, INTERVAL p.months_before MONTH)
  )
  GROUP BY 1, 2, 3
  HAVING COUNT(*) >= (SELECT min_obs_events_per_patient FROM params)
),

non_cancer_event_counts AS (
  SELECT patient_id, anchor_date, COUNT(*) AS obs_event_count
  FROM (
    SELECT DISTINCT
      nca.patient_id, nca.anchor_date,
      mh.event_date, mh.concept_id, mh.term, mh.value_numeric
    FROM non_cancer_with_anchor nca
    JOIN medical_history mh
      ON mh.patient_id = nca.patient_id AND mh.cohort = 'non_cancer'
    CROSS JOIN params p
    WHERE mh.event_date >= DATE_SUB(nca.anchor_date, INTERVAL p.years_before  YEAR)
      AND mh.event_date <  DATE_SUB(nca.anchor_date, INTERVAL p.months_before MONTH)
  )
  GROUP BY 1, 2
  HAVING COUNT(*) >= (SELECT min_obs_events_per_patient FROM params)
),

non_cancer_sampled AS (
  SELECT patient_id, anchor_date
  FROM (
    SELECT *,
      ROW_NUMBER() OVER (ORDER BY FARM_FINGERPRINT(CAST(patient_id AS STRING))) AS sample_rn
    FROM non_cancer_event_counts
  )
  CROSS JOIN params p
  WHERE sample_rn <= p.non_cancer_ratio * (SELECT COUNT(*) FROM cancer_event_counts)
),

unified_patients AS (
  SELECT patient_id, anchor_date, cancer_term AS cancer_id, 1 AS cancer_class, 'lung' AS cohort
  FROM cancer_event_counts
  UNION ALL
  SELECT patient_id, anchor_date, CAST(NULL AS STRING), 0, 'non_cancer'
  FROM non_cancer_sampled
),

/* ────────────────────────────────────────────────────────────────────────
   Demographics — UNION across both datasets, tagged with cohort.
   ──────────────────────────────────────────────────────────────────────── */

 demographics AS (
    SELECT
      'lung' AS cohort, pe.Id AS patient_id, DATE(pe.BirthDateTime) AS date_of_birth,
      g.ConceptName AS sex, e.ConceptName AS ethnicity, r.ConceptName AS race,
      pl.state_or_province, pl.postal_code,                               
      pdf.DeathDateTime IS NOT NULL AS deceased
    FROM `prj-cts-ai-dev-sp.truveta_silver_lung.Person` pe
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.Concept`         g   ON pe.GenderConceptId    = g.ConceptId
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.Concept`         e   ON pe.EthnicityConceptId = e.ConceptId
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.PersonRace`      pr  ON pr.PersonId           = pe.Id
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.Concept`         r   ON pr.RaceConceptId      = r.ConceptId
    LEFT JOIN patient_location                                        pl  ON pl.patient_id = pe.Id AND pl.cohort = 'lung'      
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.PersonDeathFact` pdf ON pdf.PersonId          = pe.Id

    UNION ALL

    SELECT
      'non_cancer', pe.Id, DATE(pe.BirthDateTime),
      g.ConceptName, e.ConceptName, r.ConceptName,
      pl.state_or_province, pl.postal_code,                                  
      pdf.DeathDateTime IS NOT NULL
    FROM `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Person` pe
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Concept`         g   ON pe.GenderConceptId    = g.ConceptId
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Concept`         e   ON pe.EthnicityConceptId = e.ConceptId
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.PersonRace`      pr  ON pr.PersonId           = pe.Id
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Concept`         r   ON pr.RaceConceptId      = r.ConceptId
    LEFT JOIN patient_location                                              pl  ON pl.patient_id = pe.Id AND pl.cohort = 'non_cancer' 
    LEFT JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.PersonDeathFact` pdf ON pdf.PersonId          = pe.Id
  ),

/* ────────────────────────────────────────────────────────────────────────
   MedicationRequest history — UNION across both datasets, deduplicated.
   ──────────────────────────────────────────────────────────────────────── */

medication_history AS (
  SELECT
    'lung' AS cohort, mr.Id AS source_record_id, mr.PersonId AS patient_id,
    DATE(mr.AuthoredOnDateTime) AS event_date,
    DATE(mr.StartDateTime) AS start_date, DATE(mr.StopDateTime) AS stop_date,
    co.ConceptId AS med_concept_id, co.ConceptName AS drug_term
  FROM `prj-cts-ai-dev-sp.truveta_silver_lung.MedicationRequest` mr
  JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.MedicationCodeConceptMap` cm ON mr.CodeConceptMapId = cm.Id
  JOIN `prj-cts-ai-dev-sp.truveta_silver_lung.Concept` co ON cm.CodeConceptId = co.ConceptId
  WHERE mr.AuthoredOnDateTime IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY mr.Id ORDER BY
    CASE co.CodeSystem WHEN 'RxNorm' THEN 1 WHEN 'SNOMED CT' THEN 2 ELSE 3 END,
    co.ConceptId) = 1

  UNION ALL

  SELECT
    'non_cancer', mr.Id, mr.PersonId,
    DATE(mr.AuthoredOnDateTime),
    DATE(mr.StartDateTime), DATE(mr.StopDateTime),
    co.ConceptId, co.ConceptName
  FROM `prj-cts-ai-dev-sp.truveta_silver_non_cancer.MedicationRequest` mr
  JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.MedicationCodeConceptMap` cm ON mr.CodeConceptMapId = cm.Id
  JOIN `prj-cts-ai-dev-sp.truveta_silver_non_cancer.Concept` co ON cm.CodeConceptId = co.ConceptId
  WHERE mr.AuthoredOnDateTime IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY mr.Id ORDER BY
    CASE co.CodeSystem WHEN 'RxNorm' THEN 1 WHEN 'SNOMED CT' THEN 2 ELSE 3 END,
    co.ConceptId) = 1
),

/* ────────────────────────────────────────────────────────────────────────
   Final event tables.
   ──────────────────────────────────────────────────────────────────────── */

observation_events_raw AS (
  SELECT DISTINCT
    up.patient_id, up.cancer_class, up.cancer_id, up.cohort,
    d.sex, d.ethnicity, d.race,d.state_or_province,
    DATE_DIFF((SELECT reference_date FROM params), d.date_of_birth, YEAR) AS patient_age,
    mh.event_date,
    DATE_DIFF(mh.event_date, d.date_of_birth, YEAR) AS event_age,
    'observation' AS event_type,
    mh.source_record_id, mh.concept_id, mh.source_concept_id, mh.source_vocabulary,
    mh.term, mh.value_numeric, mh.value_concept_id,
    CAST(NULL AS INT64)   AS med_concept_id,
    CAST(NULL AS STRING)  AS drug_term,
    CAST(NULL AS INT64)   AS duration
  FROM unified_patients up
  JOIN medical_history  mh ON mh.patient_id = up.patient_id AND mh.cohort = up.cohort
  JOIN demographics     d  ON d.patient_id  = up.patient_id AND d.cohort  = up.cohort
  CROSS JOIN params p
  WHERE mh.event_date >= DATE_SUB(up.anchor_date, INTERVAL p.years_before  YEAR)
    AND mh.event_date <  DATE_SUB(up.anchor_date, INTERVAL p.months_before MONTH)
    AND NOT EXISTS (SELECT 1 FROM any_cancer_concepts acc WHERE acc.ConceptId = mh.concept_id)

),

observation_events AS (
  SELECT * EXCEPT(patient_obs_count)
  FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY patient_id) AS patient_obs_count
    FROM observation_events_raw
  )
  WHERE patient_obs_count >= (SELECT min_obs_events_per_patient FROM params)
),

medication_events_raw AS (
  SELECT DISTINCT
    up.patient_id, up.cancer_class, up.cancer_id, up.cohort,
    d.sex, d.ethnicity, d.race,d.state_or_province,
    DATE_DIFF((SELECT reference_date FROM params), d.date_of_birth, YEAR) AS patient_age,
    mh.event_date,
    DATE_DIFF(mh.event_date, d.date_of_birth, YEAR) AS event_age,
    'medication' AS event_type,
    mh.source_record_id,
    CAST(NULL AS INT64)   AS concept_id,
    CAST(NULL AS STRING)  AS source_concept_id,
    CAST(NULL AS STRING)  AS source_vocabulary,
    CAST(NULL AS STRING)  AS term,
    CAST(NULL AS NUMERIC) AS value_numeric,
    CAST(NULL AS INT64)   AS value_concept_id,
    mh.med_concept_id, mh.drug_term,
    DATE_DIFF(mh.stop_date, mh.start_date, DAY) AS duration
  FROM unified_patients up
  JOIN medication_history mh ON mh.patient_id = up.patient_id AND mh.cohort = up.cohort
  JOIN demographics       d  ON d.patient_id  = up.patient_id AND d.cohort  = up.cohort
  CROSS JOIN params p
  WHERE mh.event_date >= DATE_SUB(up.anchor_date, INTERVAL p.years_before  YEAR)
    AND mh.event_date <  DATE_SUB(up.anchor_date, INTERVAL p.months_before MONTH)
),

medication_events AS (
  SELECT * EXCEPT(patient_med_count)
  FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY patient_id) AS patient_med_count
    FROM medication_events_raw
  )
  WHERE patient_med_count >= (SELECT min_med_events_per_patient FROM params)
)

SELECT * FROM observation_events
UNION ALL
SELECT * FROM medication_events
ORDER BY patient_id