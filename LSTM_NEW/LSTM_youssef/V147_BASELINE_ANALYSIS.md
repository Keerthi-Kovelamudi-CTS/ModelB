# v147 Lung Baseline — Run Analysis & Validation Plan

**Companion to**: `README.md` (general onboarding) — this file captures concrete findings from the actual v147 run output Youssef shared.

**Use this for**: framing your first validation deliverable. Every observation here came from one already-completed run, so the experiments below are ready to design without further data collection.

---

## TL;DR

v147 lung achieves **sens 96.09% / spec 70.12% / relevance 29.5%** in a single seed. Sensitivity is on target but specificity is **15 points short** of the 85% goal, and the explainability rating is **50+ points below** the 80% target. The likely root causes (in priority order):

1. **Age confound** (32-year mean-age gap between cancer and non-cancer cohorts)
2. **Admin / frequent-attender codes** flooding top features (`exclude_shortcut_codes=False`)
3. **Within-run val instability** (best epoch chosen on a noisy 1,023-patient val set)
4. **30-event sequence cap** truncating most clinical history

All four are testable with **single-line config changes**.

---

## Table of Contents

1. [LSTM vs your tabular baseline — the structural difference](#1-lstm-vs-tabular)
2. [What's actually in the v147 lung run output](#2-actual-output)
3. [Top model-level features (what the model thinks matters)](#3-model-level-top-features)
4. [Top patient-level features (one example patient)](#4-patient-level-top-features)
5. [Confounders identified](#5-confounders)
6. [What's right vs what's wrong](#6-right-vs-wrong)
7. [Concrete validation checklist (your Day 1)](#7-validation-checklist)
8. [Hyperparameters that matter most](#8-key-hyperparameters)
9. [How to read each metric output](#9-reading-metrics)

---

## 1. LSTM vs tabular

A common point of confusion when joining the LSTM track from the tabular pipeline.

| Aspect | Your tabular (XGB/LGB/CatBoost) | Youssef's LSTM v147 |
|---|---|---|
| **Input filtering** | You hand-pick categories (LUTS, PSA, HAEMATURIA, IPSS…) and engineer per-category features | All codes that appear ≥30× in train. **5,820 unique codes** in lung vocab (incl. PAD + RARE). |
| **Feature engineering** | Hand-engineered (psa_latest, urinary_count_B, RF_over65_with_psa_elevated, etc.) | None. Sequence of integer code IDs → embedding → LSTM → attention pool. |
| **Cancer specificity** | Built into features themselves | **Only at scoring time** via regex/codelist matching, not at training time |
| **Inductive bias toward clinical relevance** | Strong — features are clinical by construction | None. Model learns whatever minimizes loss, including shortcuts. |

**Key consequence**: the LSTM will happily learn that "Housing adequate" and "Right dorsalis pedis pulse palpable" are predictive of lung cancer if the data has that correlation — even though those codes have nothing to do with lung biology. Your job during validation is to spot when this happens.

The "cancer-specific" parts of v147 are **scoring layers, not training layers**:
- `concept_codelist_paths`: lung-cancer SNOMED IDs → used to score top features for relevance ("auth match")
- `relevance_patterns`: regex like `r'\bcough\b'`, `r'haemoptys'` → used to score top features for relevance ("regex match")
- `semantic_reference_phrases`: optional sentence-embedding fallback (off by default)
- `shortcut_text_patterns`: regex like `r'driving.*licen'`, `r'\bdna\b'`, `r'administrative procedure'` → can drop input rows if `exclude_shortcut_codes=True` (currently `False`)

---

## 2. Actual output

From the run output Youssef shared:

### Cohort

| | Count | Mean age | Std |
|---|---|---|---|
| Total patients | 10,230 | — | — |
| Cancer | 5,115 | **74.7** | 10.9 |
| Non-cancer | 5,115 | **42.7** | 21.3 |
| Total observation rows | 5,093,518 | — | — |
| └─ Cancer rows | 3,722,915 | **727 / patient** | — |
| └─ Non-cancer rows | 1,370,603 | **267 / patient** | — |

The mean-age difference is 32 years, and cancer patients have **2.7× more events on record**.

### Vocab

```
Total unique codes:                26,881
Codes appearing ≥30 times (kept):   5,818
Rare codes → <RARE>:               21,063
Final vocab (incl. PAD + RARE):     5,820
```

### Splits

```
Train: 8,184 patients (4,092 cancer / 4,092 non-cancer, 50% positive)
Val:   1,023 patients (  512 cancer /   511 non-cancer)
Test:  1,023 patients (  511 cancer /   512 non-cancer)
```

The val set has **only 1,023 patients**. Early stopping picks "best val score" off this tiny set, which is the source of the within-run instability noted below.

### Training behavior (single run)

| Epoch | val_weighted_sens_spec |
|---|---|
| 7 | **0.8871** ← best, restored |
| 8 | 0.8114 |
| 9 | 0.6118 |
| 10 | 0.3809 |
| 15 | 0.2598 ← early stopping |

The score crashed within 8 epochs. Best weights restored from epoch 7.

### Test results

| Metric | Value | Target | Status |
|---|---|---|---|
| Threshold (auto-picked, sens floor 0.95) | 0.4944 | — | — |
| Sensitivity | **96.09%** | >95% | ✓ |
| Specificity | **70.12%** | >85% | ✗ |
| Accuracy | 83.09% | — | — |
| Precision (PPV) | 76.24% | — | — |
| NPV | 94.72% | — | — |
| F1 | 85.02% | — | — |
| ROC-AUC | 83.10% | — | — |

Confusion matrix:
```
                Predicted-neg  Predicted-pos
Actual-neg          359            153
Actual-pos           20            491
```

20 cancer patients missed (false negatives = ~3.9% — close to the <2% target but not there).

### Explainability rating

| Component | Score | Weight | Comments |
|---|---|---|---|
| **Total rating** | **29.5 / 100** | — | Target 80 |
| Faithfulness (AOPC) | 0.370 | 0.10 | drop@top 0.0997 vs drop@rand 0.0627 — model attributions barely beat random ablation |
| Clinical relevance | 0.321 | **0.80** | 16/50 matched (auth=2, regex=14, sem=0) — the dominant penalty |
| Local↔global consistency | 0.006 | 0.10 | Patient top-10 features almost never appear in model-level top-50 |

The 0.80 weight on relevance means most of the rating is "are the top features clinically meaningful?" Currently 32% — well below 80%.

---

## 3. Model-level top features

The 10 features the model said were most influential (mean signed delta over 1,023 test patients):

| # | Feature | Mean Δ | Cancer-relevant? | Why? |
|---|---|---|---|---|
| 1 | Inhaler technique - poor | 0.084 | ✓ regex match | COPD signal |
| 2 | **Right dorsalis pedis pulse palpable** | 0.068 | ✗ | Foot pulse — diabetic vascular check |
| 3 | **Temperature normal** | 0.063 | ✗ | Routine vital sign |
| 4 | **Seen by consultant** | 0.061 | ✗ | Admin / utilization |
| 5 | **Housing adequate** | 0.060 | ✗ | Social-care indicator |
| 6 | Has COPD care plan | 0.060 | ✓ auth match | Direct lung condition |
| 7 | Asthma review (RCP 3Qs) | 0.059 | ✓ regex match | Lung condition review |
| 8 | **Independent walking** | 0.059 | ✗ | Mobility check |
| 9 | **Physical activity signposted** | 0.058 | ✗ | Public-health admin |
| 10 | **Medication quantities checked** | 0.058 | ✗ | Pharmacy admin |

**Of top-10: only 3 are clinically relevant.** Of top-50: **16/50 (32%)**.

The 16 matched in top-50:
- **Auth-matched (codelist) — 2**:
  - Has COPD care plan
  - Excepted from COPD quality indicators - patient unsuitable
- **Regex-matched (relevance_patterns) — 14**:
  - Inhaler technique - poor
  - Asthma review (RCP 3Qs)
  - Using inhaled steroids - low dose
  - Expected peak flow rate × 50%
  - Education for inhaled therapy
  - +9 more (truncated in run output)

The 34 unmatched top-50 features were dominated by admin codes, generic physical exams (pulse, temperature), social-care indicators, and routine reviews.

---

## 4. Patient-level top features

The 21 unique concepts (after de-dup) that drove the model's prediction for one example test patient:

| # | Code | Δ | Relevant? |
|---|---|---|---|
| 1 | Percentage basophils | 0.037 | ✗ generic blood marker |
| 2 | Serum ferritin level | 0.033 | ✗ iron, not lung |
| 3 | White blood count | 0.031 | ✗ generic |
| 4 | Patient review | 0.027 | ✗ admin |
| 5 | Administrative procedure | 0.024 | ✗ admin |
| 6 | Seen in audiology clinic | 0.021 | ✗ ear, not lung |
| 7 | Letter received | 0.021 | ✗ admin |
| 8 | Image (document) | 0.019 | ✗ admin |
| 9 | Administrative procedure (repeat) | 0.015 | ✗ admin |
| 10 | Patient review (repeat) | 0.014 | ✗ admin |
| 11 | Nonspecific abdominal pain | 0.011 | ✗ |
| 12 | Referral letter | 0.010 | ✗ admin |
| 13 | [D]Abdominal pain | 0.008 | ✗ |
| 14 | [V]Administrative encounters | 0.007 | ✗ admin |
| 15 | [D]Abdominal pain (repeat) | 0.006 | ✗ |
| 16 | US abdominal scan | 0.004 | ✗ |
| 17 | Image (document) (repeat) | 0.003 | ✗ admin |
| 18 | Had a chat to patient | 0.003 | ✗ admin |
| 19 | Seen in ophthalmology clinic | 0.002 | ✗ eye |
| 20 | Seen in audiology clinic (repeat) | (–) | ✗ ear |

**Only 2/21 unique concepts matched relevance for this patient.** The model's reasoning is mostly "this patient gets seen a lot" rather than "this patient has lung symptoms."

---

## 5. Confounders

In priority order (most likely to be polluting predictions first):

### 5.1 Age (the biggest)

- Cancer mean age: **74.7** ± 10.9
- Non-cancer mean age: **42.7** ± 21.3
- Gap: **32 years**

Pure age signal (older = higher risk) likely explains most of the achievable AUC. The model learns "old" through the static-feature `age_at_last_snomed_record` plus indirectly via codes that correlate with age (care plans, mobility checks).

**How to test**: Set `X_other_features[age]` to the median age for all patients. Re-run. If sensitivity stays ≥0.95, the model has cancer signal beyond age. If sensitivity drops to ~0.6-0.7, age was carrying it.

### 5.2 Frequent-attender / care-plan confound

Cancer patients have **2.7× more observation rows** for the same patient count (3.7M vs 1.4M for 5,115 each). Many top features are admin codes (Patient review, Letter received, Administrative procedure) — markers of utilization, not lung biology.

**How to test**: Set `config['exclude_shortcut_codes'] = True`. The lung profile already has 24+ shortcut patterns defined (admin, life-assurance, DNA, fit-for-work, etc.). Re-run and check whether relevance score rises and admin codes drop out of top-50.

### 5.3 Anchor-year matching

- Cancer anchor: year of first cancer diagnosis (`ANCHOR_YEAR`)
- Non-cancer anchor: "matched observation year" (per the notebook)

If matching is loose, you could be comparing 2020-cancer patients with 2015-non-cancer patients — temporal-coverage bias contaminates the comparison.

**How to check**: look at the cohort_counts_per_year plot the notebook prints. Each year should have similar cancer + non-cancer counts. If non-cancer has a long tail in older years that cancer doesn't, that's a problem.

### 5.4 Sequence length truncation

`MAX_SNOMED_SEQ_LENGTH = 30`. Cancer patients have mean **727 events**; non-cancer have **267**. The model only sees the **last 30** events.

For an old patient with 700+ events, the last 30 are mostly recent admin/care-plan codes — exactly the noise we don't want. The diagnostic events (hemoptysis, abnormal CXR, etc.) likely sit much earlier in the record.

**How to test**: Try `MAX_SNOMED_SEQ_LENGTH = 100` (the v116 default). Compare top features.

### 5.5 Small val set + noisy early stopping

Val set is 1,023 patients. With 50% positive class, the val score's standard error per epoch is non-trivial. Early stopping picks "best val epoch" off this noisy trajectory and restores those weights.

In the run shown, val score peaked at epoch 7 (0.8871) and **crashed to 0.2598 by epoch 15**. A different val split (different `SPLIT_SEED`) could pick a different "best" epoch → different final weights → different test metrics. **This is exactly Youssef's 0.20 single-trial variance manifesting.**

**How to test**: Run 5 seeds (`SPLIT_SEED ∈ {42, 1, 13, 99, 2024}`). Measure std-dev of test sens/spec/relevance.

### 5.6 Class-weighting + sensitivity-weight stacking

- `cancer_class_weight = 1.5` (positive class loss multiplied by 1.5)
- `sensitivity_weight = 3.0` (training metric weights sens 3× over spec)
- `use_sensitivity_floor = True`, `sensitivity_floor = 0.95` (threshold picker enforces sens ≥ 0.95)

These three layered together push very hard toward over-flagging — which is intentional (sens > 95% is the hard goal), but it depresses specificity. There may be a sweet spot with smaller `cancer_class_weight` or `sensitivity_weight` that hits sens 95% AND spec >85%.

---

## 6. Right vs wrong

### What's right with v147

- **Architecture is reasonable**: attention pool over LSTM (not just last hidden state), AdamW with weight decay, label smoothing + focal loss option, SpatialDropout1D
- **Sensitivity-floor threshold logic** consistently picks thresholds that satisfy the clinical sens ≥ 0.95 constraint
- **Three-component explainability rating** (faithfulness + clinical-relevance + consistency) gives interpretable diagnostics rather than one opaque number
- **Authoritative codelist matching** for relevance (not just regex) — more rigorous than pure pattern matching
- **Hits the sens target** (96.09% > 95%)
- **Shortcut-pattern infrastructure exists** in the codebase (just turned off in current config)
- **Cache + reproducibility seeding** mostly in place
- **Per-cancer profiles** (CANCER_PROFILES dict) make multi-cancer experimentation clean
- **Per-cancer SQL files** (lung_v45.sql, prostate_v45.sql) ready to swap

### What's wrong with v147

- **Specificity 70% vs target 85%** — 15 percentage points short
- **Explainability rating 29.5 / 100 vs target 80** — 50+ point gap
- **Faithfulness 0.37**: ablating top-K features only drops prediction by 0.0997, vs random-K which drops 0.0627. Model attributions are weak — the model isn't actually depending on the features it claims are most important
- **Consistency 0.006**: patient-level top-10 features almost never overlap with model-level top-50. Different patients are explained by different codes — no consistent reasoning
- **Top features dominated by admin codes** (Housing adequate, Independent walking, Physical activity signposted, Medication quantities checked, Right dorsalis pedis pulse palpable)
- **`exclude_shortcut_codes=False`** — the shortcut-exclusion machinery exists but is currently disabled
- **Age confound unmitigated** — 32-year cohort gap
- **Within-run val instability** — best epoch is fragile
- **`use_med_branch=False`** — medication signal channel disabled
- **MIN_SNOMED_SEQ_LENGTH=10**: includes very short sequences (10 codes), which for old patients is essentially "the last 10 admin codes" — high-noise

---

## 7. Validation checklist

Concrete experiments, ordered by impact-vs-effort. Each can be a single notebook copy with one config change, run end-to-end.

### 7.1 Day 1 trio (highest impact, single-line changes)

```
[ ] R1. Reproduce v147 baseline on lung.
       Config: as-shipped (TARGET_CANCER='lung', SEED=42, SPLIT_SEED=42)
       Confirm: sens ≈ 96%, spec ≈ 70%, relevance ≈ 0.32
       
[ ] R2. Age-blind ablation.
       Code change: in cell 79 OR before split, set X_other_features[:, 0] = median_age
       (the 0th column of static features is age_at_last_snomed_record).
       Re-run training + eval.
       Outcome:
         - If sens ≥ 0.95 still → model has signal beyond age (good)
         - If sens drops to ~0.6 → model was riding age (bad — confound confirmed)

[ ] R3. Shortcut exclusion.
       Config change: config['exclude_shortcut_codes'] = True
       Re-run training + eval.
       Compare: relevance score, top-10 features, spec.
       Outcome:
         - If relevance jumps from 0.32 to 0.5+ → admin codes were the problem
         - If spec rises → frequent-attender confound was hurting decision boundary
```

### 7.2 Reproducibility study (5-seed)

```
[ ] R4. Run R1 baseline 5 times with seeds {42, 1, 13, 99, 2024}.
       Each run: change both SEED and SPLIT_SEED.
       Record: test sens, spec, AUC, relevance per run.
       Compute: mean ± std-dev per metric.
       Outcome: confirm or refute the 0.20 single-trial variance.
```

### 7.3 Architecture & data sweeps

```
[ ] R5. Sequence length sweep: MAX_SNOMED_SEQ_LENGTH ∈ {30, 50, 100, 200}.
       Same seed each time. Record sens, spec, relevance.
       
[ ] R6. Med branch on: config['use_med_branch'] = True.
       Compare to no-med baseline.

[ ] R7. Min sequence length sweep: MIN_SNOMED_SEQ_LENGTH ∈ {10, 30, 50}.
       Higher cutoff = drops short-history patients.
       Watch the get_seq_data filter summary for n_dropped_snomed.
```

### 7.4 HP tuning (longer-running)

```
[ ] R8. config['run_tuning'] = True.
       Default: 50 trials × 1 data combo, ~6-10 hrs on A100.
       Output: tuner_results/lung_lstm_v147_v45_tuning_*/
       Inspect: top-3 trials' sens/spec/relevance.
```

### 7.5 Cross-cancer

```
[ ] R9. Switch TARGET_CANCER='prostate', re-run.
       SQL: sql/prostate_v45.sql is already on the VM.
       Compare prostate vs lung metrics + variance pattern.
```

---

## 8. Key hyperparameters (the ones that swung most in v110 → v147 iterations)

These are the knobs you'll likely be tuning:

| HP | v147 value | Reasonable range | Notes |
|---|---|---|---|
| `embedding_dim` | 16 | 8–32 | smaller is OK; vocab is large |
| `snomed_lstm_units` | 48 | 32–96 | recurrent cell size |
| `learning_rate` | 1.2e-4 | 5e-5 to 5e-4 (log) | AdamW LR |
| `embedding_dropout` | 0.4 | 0.30–0.55 | Spatial dropout on embeddings |
| `dense_dropout` | 0.5 | 0.40–0.65 | classifier head |
| `l2_reg` | 0.0036 | 1e-3 to 1e-2 (log) | L2 on dense layers |
| `weight_decay` | 0.0016 | 5e-4 to 1e-2 (log) | AdamW weight decay |
| `label_smoothing` | 0.15 | 0.0–0.15 | softens binary target |
| `use_focal_loss` | True | bool | down-weights easy examples |
| `mask_rate` | 0.5 | 0.30–0.55 | random `<RARE>` token augmentation |
| `cancer_class_weight` | 1.5 | 1.0–2.0 | extra positive-class loss weight |
| `batch_size` | 128 | 32–128 | smaller = better generalization |
| `MAX_SNOMED_SEQ_LENGTH` | 30 | 30/50/100 | sequence cap |
| `MIN_SNOMED_SEQ_LENGTH` | 10 | 10/30/50 | filter short histories |
| `min_code_frequency` | 30 | 10/30/100 | rare-token threshold |
| `sensitivity_weight` | 3.0 | 1.0–5.0 | weighting in training metric |
| `sensitivity_floor` | 0.95 | 0.90/0.95 | threshold-picker constraint |

---

## 9. Reading metrics

When you run, you'll see these printed:

| Metric | What it means | What to look for |
|---|---|---|
| `Sensitivity` | TP / (TP+FN) — % of cancer patients caught | Should be ≥ 95% |
| `Specificity` | TN / (TN+FP) — % of healthy patients cleared | Should be ≥ 85% |
| `PPV` (precision) | TP / (TP+FP) — % of flagged that are real cancers | Higher = fewer false alarms |
| `NPV` | TN / (TN+FN) — % of cleared that are truly cancer-free | Higher = safer to rule out |
| `AUC (ROC)` | Threshold-free separation | Higher = better, but not the only thing |
| `Threshold` | Probability cutoff for "predict cancer" | Auto-picked when `use_optimal_threshold=True` |
| `Faithfulness` | (drop@top - drop@rand) / drop@top | ≥0.5 means top features really drive predictions |
| `Clinical relevance` | Fraction of top-K matched to codelist+regex+semantic | ≥0.8 means top features are clinical |
| `Consistency` | Patient top-K vs Model top-N overlap | ≥0.5 means model uses same reasoning across patients |
| `Explainability Rating` | 0.1*F + 0.8*R + 0.1*C × 100 | ≥80 is the project goal |

---

## Appendix: Files this analysis is based on

- `lung_lstm_v147_bq_v45.ipynb` — the notebook (84 cells)
- Live run output from Youssef (Apr 27 03:30 UTC) — the cell printouts pasted by the user
- `sql/lung_v45.sql` — the BQ query producing the cohort
- `Vertex_AI_Pipelines_v147.md` — productionization roadmap (not directly relevant to validation)
- `lstm_model_deployment_v147.md` — deployment patterns (the "not ready for clinical use yet" admission is in here)

The single source of truth for the actual run numbers is the **printed output** of cell 83 (`outputs = run(config)`).
