# Phase 6: Explainability & Enhancement Loop

Ensures every prediction the model makes can be explained to a clinician in plain clinical language — not just "the model says high risk" but **why**.

---

## The Problem

A model might predict high risk because of `EMB_text_dim_3 = 0.42` or `ENTROPY_category = 1.87`. These are statistically valid features, but a GP cannot act on them. They need:

> "This patient is flagged because PSA rose from 3.2 to 11.4 ng/mL over 9 months,
> urinary symptoms are worsening, and new lower back pain appeared recently."

---

## How It Works

### The Three Explainability Classes

Every feature in the model is classified into one of three categories:

| Class | What It Means | Example | Clinician Sees? |
|-------|--------------|---------|----------------|
| **Direct** | Maps directly to something a GP understands | `PROST_LAB_psa_elevated_10` → "PSA above 10 ng/mL" | Yes |
| **Indirect** | Needs a sentence but is clinically meaningful | `OBS_URINARY_SYMPTOMS_acceleration` → "Urinary symptoms worsening" | Yes |
| **Opaque** | Statistical/mathematical — cannot be explained | `EMB_text_dim_3`, `ENTROPY_category`, `GINI_category` | Never shown |

### The Enhancement Loop

```
Train model
    |
    v
[1_explain_predictions.py]
    Compute SHAP values → rank features by impact
    Map each feature to clinical description
    Generate clinician-friendly reports
    |
    v
[2_audit_features.py]
    How many of the top 10 features are opaque?
    What % of model signal comes from opaque features?
    Flag: "3 of your top 10 features can't be explained"
    |
    v
[3_enhancement_loop.py]
    Remove ALL opaque features
    Retrain model
    Check: did AUC drop more than 1.5%?
      YES → selectively add back the most important opaque features
      NO  → great, model is now fully explainable
    Recheck top-10 explainability
    Repeat if needed (up to 3 iterations)
    |
    v
Explainable model + clinician reports
```

---

## Scripts

### `feature_dictionary.py` — The Foundation

Every feature gets two things:
1. An **explainability class** (direct / indirect / opaque)
2. A **clinical description** in plain English

```python
# Class: tells the loop what to remove
'PROST_LAB_psa_elevated_10':  'direct'     # keep always
'OBS_URINARY_acceleration':   'direct'     # keep always
'CLUSTER_BONE_any':           'indirect'   # keep, needs a sentence
'EMB_text_dim_3':             'opaque'     # candidate for removal
'ENTROPY_category':           'opaque'     # candidate for removal

# Description: tells the clinician what it means
'PROST_LAB_psa_elevated_10':  'PSA above 10 ng/mL (significantly elevated)'
'PROST_has_bone_pain':        'Bone pain reported'
'TEXT_SEV_worsening':         'Clinical notes describe worsening symptoms'
'EMB_text_dim_3':             '[Text embedding — not clinically interpretable]'
```

**When adding new features to the pipeline**, add them here too. Run `2_audit_features.py` to find features missing from the dictionary.

### `1_explain_predictions.py` — SHAP + Clinical Reports

**What it does:**
1. Loads the trained model and computes SHAP values for every patient
2. For each patient, ranks features by their SHAP impact (not just value, but how much they pushed the prediction)
3. Maps the top features to clinical descriptions
4. Generates two outputs:
   - `patient_explanations_{w}.json` — structured data for systems
   - `clinician_reports_{w}.txt` — readable reports for GPs

**Usage:**
```bash
python 1_explain_predictions.py --window 12mo
python 1_explain_predictions.py --window 12mo --patient GUID123   # one patient
python 1_explain_predictions.py --window 12mo --top 15            # top 15 factors
```

**Sample clinician report:**
```
Patient: PATIENT_456
Risk Score: 68.40%  |  Threshold: 42.00%  |  Prediction: HIGH RISK

WHY THIS PATIENT IS FLAGGED:
--------------------------------------------------
  1. PSA above 10 ng/mL (significantly elevated)
  2. PSA rising rapidly (>2 ng/mL change)
  3. Urinary symptoms worsening over time
  4. Bone or back pain present (possible metastatic signal)
  5. Clinical notes mention elevated PSA
  6. 3+ symptom systems involved (multi-system presentation)
  7. Pain medication prescriptions increasing

PROTECTIVE FACTORS:
--------------------------------------------------
  1. On combination BPH treatment (alpha blocker + 5-ARI)
  2. No haematuria recorded
```

**Global output** — also produces `global_feature_importance_{w}.csv`:
```
[D]  1. PROST_LAB_psa_elevated_10              SHAP=0.045
[D]  2. PROST_urinary_acceleration              SHAP=0.038
[X]  3. EMB_text_dim_3                          SHAP=0.031   ← opaque
[D]  4. PROST_has_bone_pain                     SHAP=0.028
[X]  5. ENTROPY_category                        SHAP=0.025   ← opaque
     D = direct, I = indirect, X = opaque
```

### `2_audit_features.py` — Find the Problems

**What it does:**
1. Reads the SHAP importance from step 1
2. Cross-references every feature with `feature_dictionary.py`
3. Answers three questions:
   - How many features are opaque?
   - What % of the model's signal comes from opaque features?
   - How many of the **top 10** features are opaque?
4. Generates `audit_recommendations_{w}.csv` with KEEP/REMOVE per feature

**Usage:**
```bash
python 2_audit_features.py --window 12mo
```

**Sample output:**
```
Feature Breakdown:
  Total features:     265
  Direct (D):         142 (54%) — directly explainable
  Indirect (I):        83 (31%) — needs a sentence but OK
  Opaque (X):          40 (15%) — cannot explain to clinician

SHAP Importance Share:
  Direct:     62.3% of total model signal
  Indirect:   28.1%
  Opaque:      9.6%

Opaque Features in Top Rankings:
  Top 10: 3/10 opaque
  Top 20: 5/20 opaque

EXPLAINABILITY SCORES:
  Overall:  85% of features are explainable
  Top 10:   70% of top 10 features are explainable
  → Top-10 score below 80% — run 3_enhancement_loop.py
```

### `3_enhancement_loop.py` — Auto-Fix

**What it does:**
1. Loads the current model and feature set
2. Removes all opaque features
3. Retrains the model (same split, same hyperparameters)
4. Checks: did AUC drop more than the allowed limit?
   - **If within limit** → done, model is now explainable
   - **If too much drop** → tries adding back the most important opaque features one by one until AUC is acceptable
5. Saves the refined feature list and performance comparison

**Usage:**
```bash
python 3_enhancement_loop.py --window 12mo
python 3_enhancement_loop.py --window 12mo --max-auc-drop 0.02    # allow 2% drop
python 3_enhancement_loop.py --window 12mo --max-iterations 5
python 3_enhancement_loop.py --window 12mo --target-top10 0.90    # 90% explainable
```

**Parameters:**
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--max-auc-drop` | 0.015 | Maximum AUC decrease allowed (1.5%). If removing opaque features drops AUC by more than this, some are added back |
| `--max-iterations` | 3 | Maximum removal-retrain cycles |
| `--target-top10` | 0.80 | Stop when 80%+ of top 10 features are explainable |

**Sample output:**
```
BASELINE: 265 features, 40 opaque, AUC=0.847

ITERATION 1:
  Removing 40 opaque features (EMB_*, BERT_*, ENTROPY_*, GINI_*, ...)
  Features: 265 -> 225
  AUC: 0.839 (drop: -0.008) ← within 0.015 limit
  Top-10 explainability: 100%

ENHANCEMENT LOOP COMPLETE
  Features:         265 -> 225
  Opaque removed:   40
  Final AUC:        0.839 (baseline: 0.847, drop: 0.008)
  Final Sensitivity: 0.792
  Final Specificity: 0.714
```

**Interpretation:** we lost 0.8% AUC (negligible) but now every feature the model uses can be explained to a clinician. The tradeoff is almost always worth it.

---

## Results Directory

```
results/
  {3mo,6mo,12mo}/
    global_feature_importance_{w}.csv     Feature + SHAP + class + description
    patient_explanations_{w}.json         Per-patient explanations (structured)
    clinician_reports_{w}.txt             Per-patient reports (readable)
    audit_recommendations_{w}.csv         KEEP/REMOVE per feature
    enhancement_loop_log_{w}.csv          Iteration-by-iteration metrics
    explainable_features_{w}.csv          Final feature set after loop
    enhancement_summary_{w}.json          Summary: AUC drop, features removed
```

---

## Full Workflow

```bash
# 1. After training (Phase 3), run SHAP analysis
python 1_explain_predictions.py --window 12mo

# 2. Check: are the important features explainable?
python 2_audit_features.py --window 12mo

# 3. If top-10 score < 80%, run the enhancement loop
python 3_enhancement_loop.py --window 12mo

# 4. Re-explain with the refined model
python 1_explain_predictions.py --window 12mo

# 5. Re-audit to confirm
python 2_audit_features.py --window 12mo
# → Top-10 score should now be 80%+
```

---

## Porting to Another Cancer

1. Copy `feature_dictionary.py`
2. Keep all generic entries (they're prefix-based, work for any cancer)
3. Replace the `PROST_*` entries with cancer-specific descriptions:
   - Lymphoma: `LYM_has_lymphadenopathy` → "Enlarged lymph nodes present"
   - Ovarian: `OVAR_LAB_ca125_elevated` → "CA-125 above 35 U/mL"
4. Scripts `1_`, `2_`, `3_` are fully generic — no changes needed

---

## Key Decisions

**Q: What if an opaque feature is very important?**
The loop handles this. If removing it drops AUC too much, it's added back and flagged. You then have three options:
1. **Accept the tradeoff** — keep it, accept the AUC drop
2. **Replace it** — find what clinical signal the opaque feature captures (e.g., `EMB_text_dim_3` might correlate with "mentions of biopsy in notes") and add an explicit keyword feature instead
3. **Reclassify it** — if you can write a clinical description, update the dictionary from 'opaque' to 'indirect'

**Q: How much AUC drop is acceptable?**
For cancer screening in primary care:
- < 1% drop: always accept
- 1-2% drop: usually accept (explainability is worth it)
- 2-5% drop: discuss with clinical team
- > 5% drop: opaque features carry real signal — investigate and replace rather than remove

**Q: Why not just use SHAP without the loop?**
SHAP tells you which features matter. But if the top feature is `EMB_text_dim_3`, SHAP can only say "this dimension pushed the prediction up". It cannot tell the clinician **what to do about it**. The loop ensures every feature in the model has a clinical action associated with it.
