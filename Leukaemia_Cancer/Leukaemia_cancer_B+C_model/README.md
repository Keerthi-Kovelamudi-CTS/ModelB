<!-- ════════════════════════════════════════════════════════════════════ -->
<!-- LEUKAEMIA — B+C MATURED MODEL  (generated 2026-06-04)            -->
<!-- ════════════════════════════════════════════════════════════════════ -->

# Leukaemia — B+C Matured Model

Copy of `Leukaemia_2.0_1to1` (originals untouched) with the **matured B+C cohort/split
design** finalized on breast and rolled out across all cancers. Source: **EMIS**.
Sex: **both sexes**. Dual/tri-horizon windows: **1mo + 12mo**.

## What's matured here (vs the parent `Leukaemia_2.0_1to1`)
- **Gold-layer non-cancer anchor** — each non-cancer patient is anchored on a RANDOM
  date drawn from the **cancer diagnosis-date distribution** (`cancer_anchor_pool` +
  `MOD(ABS(FARM_FINGERPRINT(patient_guid)), N)+1`), so non-cancer and cancer share the
  same anchor-**year** distribution by construction (kills the recency confound).
  Replaces the old own-event-date anchor. Aligned to `Truveta/Git_SQL/cancer_training_truveta_v3_gold.sql`.
- **`non_cancer_ratio = 5`** — oversample non-cancer 5× as a buffer.
- **Per-cohort `min_obs`** — cancer = 1 (keep every positive), non-cancer = 5.
- **Balance-to-1:1 across train / val / test** — the majority class is downsampled to
  the minority count **before** the stratified split, so all three splits are ~1:1 and
  positive cases are never discarded (the 5× buffer absorbs the trim).
- Trend / worsening (`SEQ_*`/`ACCEL_*`) and Approach B (`__PLACEHOLDER__` rows) are **inherited** from the EMIS template — no change needed.
- Random stratified split (no temporal holdout).

## Re-run (SQL changed → full chain)
1. Run `SQL Queries/v4/*.sql` in BigQuery → cohort tables
2. `2_Feature_Engineering` preprocess + FE
3. `3_Modeling` (watch the split log: if it reports *"dropped surplus cancer"*, the 5× buffer was too small)

---

  ═════════════════════════════════════════════════════════════════════════
  LEUKAEMIA_2.0_1to1 — 1:1 YEAR-STRATIFIED COHORT VARIANT  (SQL v4)
  ─────────────────────────────────────────────────────────────────────────
  Scaffolded from Prostate_2.0_1to1 by copying the pipeline code as-is and
  resetting cancer-specific configuration to TODO placeholders.

  TO BRING THIS ONLINE you need to provide:
    1. SQL Queries/v4/{1mo,3mo,6mo,12mo}_v4.sql
       - target_cancer_pattern    : '%leukaemia%'
       - sex filter (in WHERE)    : both
       - non_cancer_ratio         : 1   (1:1 year-stratified)
       - cancer_snomed_codes      : leukaemia-specific diagnosis codes
    2. codelists/code_category_mapping_v2.json   (leukaemia curated codes)
    3. codelists/leukaemia_curated_codes_v2.tsv
    4. 2_Feature_Engineering/config.py
       - replace prostate OBS/MED/LAB/CLUSTER/INTERACTION configs with
         leukaemia-relevant ones (markers in file).
    5. 2_Feature_Engineering/4_cancer_features.py
       - currently PSA-specific; rewrite with leukaemia markers
         (e.g., leukaemia-specific labs, imaging, symptoms).

  Pipeline code is otherwise identical to Prostate_2.0_1to1 — same FE
  improvements (interval-slope, h1/h2 split, percent-change, trend r-value,
  pluggable LAB_WORSENING_RULES) and same modeling + explainability.
  ═════════════════════════════════════════════════════════════════════════

  [1] SQL EXTRACTION (SQL Queries/v4/{1mo,3mo,6mo,12mo}_v4.sql)  ← TODO: leukaemia cohort─────────────────────────────────────────                                                                                                                  
      Cancer:     ALL prostate cancer patients (no code filter)                                                                                                  
      Non-cancer: ALL matched men (no code filter, ≥10 events of any kind, sampled 1:1 per anchor year)
      Output:     Every event for every patient (raw SNOMEDs, no CATEGORY)                                                                                       
                                                                                                                                                                 
                                                                                                                                                                 
  [2] FEATURE ENGINEERING                                                                                                                                        
      ─────────────────────────────────────────                                                                                                                  
      Map raw SNOMEDs → curated CATEGORIES (via code_category_mapping.json)
      Compute features ONLY for curated codes:                                                                                                                   
        Patient A (rich): 'Haematuria', 'PSA', 'DRE'      → real features                                                                                        
        Patient B (sparse): 'Cold', 'BP check', 'Flu jab' → ALL ZERO features ← KEPT                                                                             
        Patient C (empty): no events at all                → ALL ZERO + placeholder                                                                              
      All patients have a row in the feature matrix.                                                                                                             
                                                                                                                                                                 
                                                                                                                                                                 
  [3] MODEL TRAINING                                                                                                                                             
      ─────────────────────────────────────────
      Trained on ALL patients (including all-zero feature rows).
      Model learns:                                                                                                                                              
        "Patient with PSA + DRE + biopsy + age 70 → cancer (label=1)"
        "Patient with PSA + DRE pattern but no escalation → maybe (label=0)"                                                                                     
        "Patient with all-zero curated features → probably no cancer (label=0)"                                                                                  
                                                        ↑                                                                                                        
                                         this is the missing piece in v1                                                                                         
                                                                                                                                                                 
                                                                                                                                                                 
  [4] PREDICTION (deployment / 300K holdout)                                                                                                                     
      ─────────────────────────────────────────                                                                                                                  
      Real-world clinic patient with 0 curated codes
        → model has trained on this exact pattern → predicts low proba                                                                                           
        → not flagged as FP at sens≥92 threshold                                                                                                                 
        → spec stays high                                                                                                                                        
                                                                                                                                                                 
  One small clarification — "no features (0) could still have cancer"                                                                                            
                                                                                                                                                                 
  You're correct that a patient with no curated GP signal could still develop cancer — we just have no GP-record basis to predict it for them. By including these
   patients in training:
                                                                                                                                                                 
  - Training cancer with all-zero features (rare but exists): model learns cancer can occur without rich GP signal → maintains a small baseline cancer           
  probability for sparse-record patients
  - Training non-cancer with all-zero features (common — ~30-40% of population): model learns "default = no cancer for sparse patient"                           
                                                                                                                                                                 
  The model balances these: default low cancer prob for sparse patients, but not zero. That's the right behavior — clinically and statistically.                 
                                                                                                                                                                 
  This is what v1 missed. Approach A only saw "patients with curated workup" → at deployment, sparse patients were OOD → model defaulted to ~0.16 (moderate      
  proba) → spec collapsed.                
                                                                                                                                                                 
  Why this fixes the spec=34% problem                                                                                                                            
   
  ┌─────────────────────────────────────────────┬──────────────────────────────┬─────────────────────────────────────┐                                           
  │                                             │       v1 (Approach A)        │           v2 (Approach B)           │
  ├─────────────────────────────────────────────┼──────────────────────────────┼─────────────────────────────────────┤                                           
  │ Training included sparse-record non-cancer? │ ❌ No (filtered out)         │ ✅ Yes                              │
  ├─────────────────────────────────────────────┼──────────────────────────────┼─────────────────────────────────────┤
  │ Model has seen "no curated codes" pattern?  │ ❌ Never                     │ ✅ Many examples                    │                                           
  ├─────────────────────────────────────────────┼──────────────────────────────┼─────────────────────────────────────┤                                           
  │ Behavior at deployment on sparse patient    │ OOD → moderate proba (~0.16) │ In-distribution → low proba (~0.02) │                                           
  ├─────────────────────────────────────────────┼──────────────────────────────┼─────────────────────────────────────┤                                           
  │ At sens≥92 threshold (~0.03)                │ Flagged ❌ FP                │ Below threshold ✅ TN               │
  ├─────────────────────────────────────────────┼──────────────────────────────┼─────────────────────────────────────┤                                           
  │ Combined-eval spec                          │ 34%                          │ targeted 60-75%        





Approach B by design. Walk-through:                                                                                                
                                                                                                                                                                 
  Step 1: SQL extracts ALL events per matched patient. Every observation/medication in their 60-month lookback window — no filtering by code. So a patient could 
  have 5,000 events including all sorts of unrelated things.     
                                                                                                                                                                 
  Step 2: Python preprocessor (0_preprocess_to_fe.py):                                                                                                          
  - Loads raw CSV (e.g., 9 GB, 44M rows)                                                                                                                         
  - For each event, looks up its SNOMED/MED code in code_category_mapping_v2.json (179 curated obs + 49 med codes for v3-no-PSA)                                 
  - Event matches a code → CATEGORY assigned (e.g., "PSA", "URINARY", "ALBUMIN_PROTEIN")                                        
  - Event doesn't match → CATEGORY = None → dropped from the FE feature stream                                                                                   
  - Result: ~9.7M rows kept (out of 44M raw — ~22% retention)                                                                                                    
                                                                                                                                                                 
  Step 3: Placeholder rows for sparse patients:                                                                                                                  
  # From 0_preprocess_to_fe.py                                                                                                                                   
  patients: 206,404 total | 124,762 have curated events | 81,642 need placeholder rows                                                                           
  added 81,642 placeholder rows                                                       
  So ~40% of patients have NO curated events at all — they get a single fake row with CATEGORY = __PLACEHOLDER__ so they still appear in the feature matrix.     
                                                                                                                                                                 
  Step 4: FE pipeline builds features from the CATEGORY-tagged events.                                                                                           
  - For a patient with curated events: gets real feature values (counts, trajectories, etc.)                                                                     
  - For a placeholder-only patient: feature values are all 0 (nothing to count/aggregate from)                                                                   
  - BUT they still have:                                                                                                                                         
    - AGE_AT_INDEX (real age)                                                                                                                                    
    - SEX (Male)                                                                                                                                                 
    - LABEL (0 or 1)                                               
    - All ~263 clinical features = 0                                                                                                                             
                                                                                                                                                                 
  Step 5: Modeling sees both kinds.                                                                                                                              
  - Model learns: "all-zero clinical features + age=70 + male" → likely no cancer (strong prior in non-cancer pool)                                              
  - "many features active in PSA/symptoms/labs" → cancer signal                                                                                                  
  - This trains the "no signal = no cancer" boundary you wanted      
















1_sanity_check.py is the FE's first quality gate. It drops rows that fail basic validity checks like:
  - NULL/missing PATIENT_GUID                                                                                                                                    
  - NULL/missing EVENT_DATE                                                                                                                                      
  - NULL/missing CATEGORY                                                                                                                                        
  - Empty TIME_WINDOW                                                                                                                                            
  - Duplicates         































What you're describing                                                                                                                                         
                                                                                                                                                                 
  Current v2 FE:                    Hybrid FE (your idea):
  ─────────────────                  ─────────────────────                                                                                                       
  Curated codes only:                Curated codes (~1000 features):                                                                                             
    LAB_HAEMATOLOGY_*                  LAB_HAEMATOLOGY_*                                                                                                         
    OBS_LUTS_*                         OBS_LUTS_*                                                                                                                
    RECUR_*                            RECUR_*                                                                                                                   
    CLUSTER_*                          CLUSTER_*                                                                                                                 
                                     +                                                                                                                           
                                     Generic activity (~5-10 features):                                                                                          
                                       GENERIC_count_total                                                                                                       
                                       GENERIC_count_A / count_B 
                                       GENERIC_unique_codes                                                                                                      
                                       GENERIC_days_since_last   
                                       GENERIC_acceleration                                                                                                      
                                       GENERIC_unique_categories                                                                                                 
                                             
  Does it hurt? Almost certainly not                                                                                                                             
                                                                                                                                                                 
  ┌──────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │               Concern                │                                                      Reality                                                      │   
  ├──────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ "Generic features dilute cancer      │ No — boosted trees auto-weight by gain. If generic features are noisy, model uses them less. If informative,      │   
  │ signal"                              │ model uses them more. Self-balancing.                                                                             │
  ├──────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ "Could shortcut to 'high activity =  │ Mild risk. Mitigated by year-stratified sampling (already done) + age feature already captures most               │
  │ cancer'"                             │ age-correlated activity.                                                                                          │
  ├──────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ "More features = overfitting"        │ 5-10 extra features in a 1000+ feature space is negligible. Boosted trees handle this fine.                       │
  └──────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘   
                                                                 
  Does it help? Two scenarios:                                                                                                                                   
                                                                                                                                                                 
  Scenario A — patient has curated codes
                                                                                                                                                                 
  - Curated features fire normally (rich signal)                 
  - Generic features add small marginal info                                                                                                                     
  - Model leans on curated — generic gets low importance         
  - Net: small positive or neutral                                                                                                                               
                                             
  Scenario B — patient has only non-curated codes (sparse curated)                                                                                               
                                                                                                                                                                 
  - All curated features are zero                                                                                                                                
  - Generic features now provide signal: "patient has clinical activity, just not prostate-specific"                                                             
  - Model can distinguish "no signal at all" vs "signal but not curated"
  - Net: meaningful positive — exactly what helps spec on holdout                                                                                                
                                             
  Why "give less importance" is automatic                                                                                                                        
                                                                                                                                                                 
  You don't need to manually downweight. Gradient boosting's feature_importance ranks features by their contribution to reducing loss. If generic features       
  dominate cancer-specific ones in real importance, that means they're genuinely better predictors — not a problem.                                              
                                                                                                                                                                 
  If you want explicit downweighting:                                                                                                                            
  - Train with feature_weights in XGBoost (sets importance prior)
  - Use 1.0 for curated, 0.3 for generic                                                                                                                         
  - Forces tree splits to prefer curated when both options are equally informative
                                                                                  
  But this is overengineering — try unweighted first, see if it's needed.                                                                                        
                                             
  Recommended generic features (concrete list)                                                                                                                   
                                                                                                                                                                 
  # In a new 7_generic_features.py (after 6_text_features.py):                                                                                                   
  non_curated = df[df['CATEGORY'].isna()].copy()  # rows with no curated CATEGORY                                                                                
                                                                                                                                                                 
  # 1. Volume features                                           
  GENERIC_count_total = non_curated.groupby('PATIENT_GUID').size()                                                                                               
  GENERIC_count_A     = non_curated[TIME_WINDOW=='A'].groupby(...).size()                                                                                        
  GENERIC_count_B     = non_curated[TIME_WINDOW=='B'].groupby(...).size()                                                                                        
                                                                                                                                                                 
  # 2. Diversity features                                                                                                                                        
  GENERIC_unique_codes    = non_curated.groupby('PATIENT_GUID')['CODE_ID'].nunique()                                                                             
  GENERIC_unique_practices = non_curated.groupby('PATIENT_GUID')['practice_id'].nunique() if you have it                                                         
                                                                                                                                                                 
  # 3. Recency features                                          
  GENERIC_days_since_last = non_curated.groupby('PATIENT_GUID')['DAYS_BEFORE_ANCHOR'].min()                                                                      
  GENERIC_first_event_days = non_curated.groupby('PATIENT_GUID')['DAYS_BEFORE_ANCHOR'].max()                                                                     
                                             
  # 4. Trajectory features                                                                                                                                       
  GENERIC_acceleration = (count_B / count_A_safe) - 1  # is non-curated activity escalating?                                                                     
                                                                                                                                                                 
  That's ~6-8 features added. Negligible compute cost.                                                                                                           
                                                                                                                                                                 
  Do also: a "MEDS_GENERIC" set for non-curated medications                                                                                                      
                                                                                                                                                                 
  Same pattern but for medication events (most prescriptions aren't in your 49 curated meds):                                                                    
  - MED_GENERIC_count_total, MED_GENERIC_unique_drugs, MED_GENERIC_acceleration, etc.                                                                            
                                                                                     
  Should we do it? My recommendation                                                                                                                             
                                                                 
  Yes, but as a SECOND iteration. Don't bundle into v2 launch:                                                                                                   
                                                                 
  1. First: run plain v2 (curated-only FE) end-to-end, get the combined-eval baseline (this is what's running now)                                               
  2. Then: add generic features as v2.1, compare                                                                                                                 
  3. If v2.1 > v2 → keep generic features                                                                                                                        
  4. If v2.1 ≈ v2 → drop them (simpler is better)                                                                                                                
  5. If v2.1 < v2 → unlikely but possible (over-reliance on generic shortcut); investigate
                                                                                                                                                                 
  This way you get a clean A/B on the generic-features question without confounding it with the bigger Approach A vs B test.                                     
                                                                                                                                                                 
  TL;DR                                                                                                                                                          
                                                                                                                                                                 
  Your intuition is right and it's worth doing. Generic activity features add a small "fallback signal" for sparse-curated patients. Boosted trees auto-weight,  
  so no risk of hurting the model. Recommend adding as v2.1 after v2 baseline is validated.                                                                      
                                                                 
 

























 Big insight — age does most of the work.                                      
                                                     
  ┌────────┬────────────────┬──────────────┬───────────────┐          
  │ Window │ Full model AUC │ Age-only AUC │ Lift over age │             
  ├────────┼────────────────┼──────────────┼───────────────┤          
  │ 3mo    │ 0.9501         │ 0.8985       │ +0.052        │                                                                                                     
  ├────────┼────────────────┼──────────────┼───────────────┤          
  │ 6mo    │ 0.9433         │ 0.9009       │ +0.042        │                                                                                                     
  ├────────┼────────────────┼──────────────┼───────────────┤                                                                                                   
  │ 12mo   │ 0.9357         │ 0.8976       │ +0.038        │                                                                                                     
  └────────┴────────────────┴──────────────┴───────────────┘                                                                                                     
                                                                                                                                                                 
  Age alone gets you to ~0.90. The 178 other features (PSA + symptoms + labs etc.) add only ~0.04 AUC.                                                           
                                                                                                                                                                 
  Per-age-band AUC (conditional on age) drops a lot:                                                                                                             
                                                  
  ┌────────┬──────┬───────┬───────┬──────┐                                                                                                                       
  │ Window │ <55  │ 55-65 │ 65-75 │ 75+  │                                                                                                                       
  ├────────┼──────┼───────┼───────┼──────┤                         
  │ 3mo    │ 0.86 │ 0.83  │ 0.82  │ 0.82 │                                                                                                                       
  ├────────┼──────┼───────┼───────┼──────┤                         
  │ 6mo    │ 0.87 │ 0.79  │ 0.79  │ 0.81 │                                                                                                                       
  ├────────┼──────┼───────┼───────┼──────┤                       
  │ 12mo   │ 0.89 │ 0.76  │ 0.76  │ 0.78 │                                                                                                                       
  └────────┴──────┴───────┴───────┴──────┘                       
                                                                                                                                                                 
  Per-age-band SPEC at sens≥0.92 is much worse than overall:       
                                                  
  ┌────────┬─────┬───────┬───────┬─────┬─────────┐                 
  │ Window │ <55 │ 55-65 │ 65-75 │ 75+ │ Overall │                                                                                                               
  ├────────┼─────┼───────┼───────┼─────┼─────────┤                 
  │ 3mo    │ 0%  │ 7%    │ 26%   │ 28% │ 81%     │                                                                                                               
  ├────────┼─────┼───────┼───────┼─────┼─────────┤                 
  │ 6mo    │ 56% │ 18%   │ 32%   │ 39% │ 79%     │                                                                                                               
  ├────────┼─────┼───────┼───────┼─────┼─────────┤               
  │ 12mo   │ 0%  │ 30%   │ 19%   │ 32% │ 79%     │                                                                                                               
  └────────┴─────┴───────┴───────┴─────┴─────────┘               
                                                                                                                                                                 
  This is the key finding — the "overall 78-81% spec" is largely driven by AGE doing the discrimination. Within an age band, spec collapses to 7-39%.
                                                  
  What it means for deployment:                                                                                                                                  
                                                  
  1. Spec depends on deployment population age mix. If you deploy in a younger primary-care population (e.g., <55 dominant), spec will look much worse than the  
  held-out test reports.                                                                                                                                         
  2. Per-age-band thresholds are essential. A single 0.06 threshold won't work across age bands — older men ALL look high-risk to the model. You need separate
  thresholds per band.                                                                                                                                           
  3. The "0.95 AUC" headline is mostly age + PSA together. No-PSA pipeline (running) will reveal what's left when you take PSA out — and the per-age-band check
  after that will tell you what's non-age, non-PSA signal (the truly useful part).                                                                               
                                                                 
  Pre-existing red flags this confirms:                                                                                                                          
  - Top features are AGE + PSA (already known shortcuts)
  - 0-55 band has only 24-30 positives — basically no signal there, so model is "guess no for everyone" → 100% sens, 0% spec                                     
  - Lift over age (0.04) is small; lift over age+PSA might be near zero                                                                                          
                                                                       
  Recommend now: wait for no-PSA run, then re-run eval_at_floor.py --with-age on it. If no-PSA AUC ≈ age-only AUC (~0.90), the model isn't earning its keep      
  beyond age. If no-PSA AUC > age-only AUC by 0.05+, real signal exists.
                                                                                                                                                                 
  Usage:                                                           
  python3 eval_at_floor.py --with-age                      # default floor 0.92                                                                                  
  python3 eval_at_floor.py --floors 0.92 0.95 --with-age   # multiple          
  python3 eval_at_floor.py --age-bands 0-50 50-60 60-70 70+ --with-age   # custom bands                                                                          
        