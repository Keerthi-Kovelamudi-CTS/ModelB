# LSTM Cancer Prediction — Onboarding & Validation Guide

**Audience**: you (Keerthie), joining the LSTM track to support Youssef on **model validation and improvement**.

**Purpose of this README**: get you from zero to running the notebook and producing your first validation deliverable, without needing to ask Youssef for help. Read top-to-bottom on first pass, then keep it open as a reference.

---

## Table of Contents

1. [Big picture — what this project is](#1-big-picture)
2. [The team and your specific role](#2-team-and-your-role)
3. [What v147 actually is (the current model)](#3-what-v147-actually-is)
4. [Files in this folder — what you have and why](#4-files-in-this-folder)
5. [The targets vs reality (honest assessment)](#5-targets-vs-reality)
6. [Model architecture — what it's doing under the hood](#6-model-architecture)
7. [Data flow end-to-end](#7-data-flow)
8. [Configuration deep-dive (every knob you can turn)](#8-configuration)
9. [The hyperparameter tuner](#9-hp-tuner)
10. [Your validation playbook (concrete tasks)](#10-validation-playbook)
11. [Setting up on the VM](#11-vm-setup)
12. [How to run the notebook](#12-running-the-notebook)
13. [How to read the metrics](#13-reading-metrics)
14. [Common pitfalls and gotchas](#14-pitfalls)
15. [Glossary (ML terms you'll see)](#15-glossary)

---

## 1. Big picture

**Goal**: Build AI models that predict cancer early from primary-care EHR data (SNOMED codes for observations, medications, labs, clinical text), with **clinical-grade performance** good enough for production deployment.

**Why LSTM**: Patient EHR records are *sequences of events over time*. A patient might have 50 SNOMED codes recorded over 5 years. The order and timing of those codes carry signal (e.g. "haematuria" followed 3 months later by "raised PSA" is more concerning than the same two codes 5 years apart). LSTMs (Long Short-Term Memory networks) are designed to learn from sequences, so they can pick up patterns that flat tabular models like XGBoost can miss.

**Two parallel tracks at C The Signs**:
- **Tabular ML pipeline** (your existing prostate work in `Prostate_Cancer/`) — XGB/LGB/CatBoost ensemble on hand-engineered features.
- **LSTM pipeline** (this project) — sequence model on raw SNOMED code sequences.

The LSTM track aims for higher ceiling and better explainability (which sequences/patterns drove the prediction), but is harder to make stable.

---

## 2. Team and your role

From the project doc Youssef shared:

| Track | Who | What |
|---|---|---|
| **Data for ML Models** | Abbas, Youssef, Alex, Marta | Build leak-free training datasets |
| **Core LSTM Models** | Youssef, Abbas | Develop LSTM architecture, iterate versions |
| **LSTM Explainability** | Nicole, Youssef | Make the model's reasoning intuitive to clinicians |
| **Model Validation & Improvement** | **Keerthie (you)**, Youssef | Pressure-test the model, find issues, run HP tuning, report across cancers |
| **Productization-readiness** | Youssef, ? | Vertex AI deployment |

**Your role in plain English**:
- You are the second pair of eyes that pressure-tests Youssef's models.
- You don't need to redesign the architecture — Youssef owns that.
- You **do** need to: re-run experiments to confirm the numbers, find sources of instability, run HP tuning, and report results across multiple cancer types.
- Final deliverable per the brief: results for each target cancer hitting (or as close as possible to) **sens > 95%, spec > 85%, FN < 2%, model-level explainability relevance > 80%**.

---

## 3. What v147 actually is

`lung_lstm_v147_bq_v45.ipynb` is the **current best-iteration notebook** for lung cancer prediction:
- `v147` = code version (Youssef has been iterating fast — v140 to v147 in just 1-2 days)
- `v45` = SQL/data extraction version (the BigQuery query that builds the training data)

The notebook has **84 cells** organised in this order:
1. **Imports + config helpers** (cells 7-12)
2. **Data loading from BigQuery** (cells 14-19) — runs `sql/lung_v45.sql`, caches result as parquet locally
3. **Pre-processing** (cells 20-26) — cleans raw rows, filters to enabled branches, caps cohort size
4. **Sequence feature engineering** (cells 29-39) — builds vocabularies, encodes SNOMED codes as integer IDs, pads sequences to fixed length
5. **Train/val/test split** (cells 40-44)
6. **Build LSTM model** (cells 45-48) — Keras model definition
7. **Train** (cells 49-51) — model.fit() with callbacks
8. **Save / Load** (cells 52-55)
9. **Evaluate** (cells 56-67) — ROC, threshold tuning, confusion matrix, per-patient inspection
10. **Demographics breakdown** (cells 67-68)
11. **Explainability** (cell 70) — Integrated Gradients on static features + temporal ablation
12. **Hyperparameter tuning** (cell 72) — Keras Tuner with custom Bayesian optimization
13. **Run cells** (cells 73-83) — orchestrates everything

The **cancer profile** (cell 76) defines per-cancer settings: which SNOMED codelists to use, which regex patterns count as "clinically relevant" for the explainability metric, and how many positive/negative patients to sample. Profiles are defined for breast, lung, prostate, and others.

---

## 4. Files in this folder

Pulled from `/home/youssefdrissi_cthesigns_com/gs-y-dev/` on the VM:

```
LSTM_youssef/
├── README.md                              ← you are here
├── lung_lstm_v147_bq_v45.ipynb           ← the actual notebook (latest, Apr 27)
├── Vertex_AI_Pipelines_v147.md           ← Youssef's roadmap for productionizing v147
├── lstm_model_deployment_v147.md         ← Youssef's deployment patterns doc
├── v147_key_cells.txt                    ← extracted key cells (build_model, train, tuner)
├── sql/
│   ├── lung_v45.sql                      ← BigQuery query for lung data
│   └── prostate_v45.sql                  ← BigQuery query for prostate data (already prepared!)
└── reqs/
    ├── nb_reqs.txt                       ← notebook-only deps
    └── reqs_colab_entrprise.txt          ← full conda env deps (15K, the heavy one)
```

The two MD docs (Vertex_AI_Pipelines_v147.md and lstm_model_deployment_v147.md) are about **productionization** — they're Youssef's instructions for how to convert the notebook into a deployable pipeline. They are **not your immediate task**. Skim them for context, but your work is at the model-quality layer beneath.

---

## 5. Targets vs reality

**Targets per project brief**:

| Metric | Target | What it means |
|---|---|---|
| Sensitivity (Sens) | **> 95%** | Of cancer patients, ≥95% are correctly flagged. Low miss rate. |
| Specificity (Spec) | **> 85%** | Of healthy patients, ≥85% are correctly cleared. Low false-alarm rate. |
| False Negative rate | **< 2%** | At most 2% of cancer patients are missed. Very strict. |
| Explainability Relevance | **> 80%** (model-level) | At least 8/10 top-influential features are clinically relevant per the regex patterns. |
| Stable across | data, sources, cancer types, deployments | Numbers don't swing wildly. |

**v147 reality** (per Youssef's deployment doc, line 51):

> *"The clinical-relevance / sensitivity / specificity numbers from v147 aren't ready for clinical decision-making yet — relevance ~0.4 ceiling, single-trial variance ~0.20."*

Translation:
- **Relevance** stuck around 0.4 (target 0.8) — model picks up patterns, but only ~40% of the top influencers are clinically meaningful per the regex check.
- **Single-trial variance ~0.20** — if you train the same model twice with different seeds, the headline metric can swing by 0.20 (i.e. a sensitivity of 0.85 ± 0.20 is unreliable for clinical use).

These two issues are **exactly your job to investigate and reduce.**

---

## 6. Model architecture

High-level flow of one prediction:

```
Patient → SNOMED sequence (last N events) ┐
       → Static features (age, sex, demo) ┤→ LSTM model → P(cancer)
       → MED sequence (currently OFF)    ┘
```

### The LSTM block (per branch)

For the SNOMED branch (and same shape for MED if enabled):

```
SNOMED IDs (length 30, padded)
        ↓
[Embedding(vocab_size → embedding_dim=16)]      ← turns each integer code into a 16-dim vector
        ↓
[SpatialDropout1D(0.30-0.55)]                   ← regularisation, randomly drops whole codes
        ↓
[LSTM(units=32-96, return_sequences=True)]      ← scans sequence, outputs vector per timestep
        ↓
[Attention pool]                                ← weights timesteps by importance, pools to one vector
        ↓
context_snomed (single vector)
```

Then for static features:

```
Other features (age, etc.)
        ↓
[Dense(8-32, relu, l2 reg)]
        ↓
context_other
```

Then fusion + classifier:

```
[Concat(context_snomed, context_med?, context_other)]
        ↓
[Dropout(0.40-0.65)]
        ↓
[Dense(16-64, relu)]
        ↓
[Dense(1, sigmoid)] → probability of cancer (0..1)
```

**Loss**: BinaryCrossentropy with `label_smoothing=0..0.15` (or BinaryFocalCrossentropy if `use_focal_loss=True`).
**Optimizer**: AdamW with weight decay (1e-3 to 1e-2 range).

### Key architectural choices to know

- **Attention pool, not just last LSTM state**: Older LSTM designs took the LSTM's last hidden state. This one weights *all* timesteps by learned importance and sums them. Good for long sequences where the diagnostic event might not be at the end.
- **Embedding dim is small (16-32)**: Vocabulary is large (thousands of unique SNOMED codes), but each code only needs a tiny vector to be useful here.
- **Sequence cap = 30 events per patient**: Most patients have far more in their record. The model uses the **last 30** events before the index date. (Configurable via `MAX_SNOMED_SEQ_LENGTH`.)
- **Single LSTM layer**: No stacking. Simpler models often generalize better on small clinical cohorts (~30K patients).
- **`use_med_branch = False` in v147**: The medication branch exists in the code but is currently **disabled**. Only SNOMED + static features are actually used. (Worth experimenting with re-enabling — see your task list.)

---

## 7. Data flow

```
1. BigQuery → sql/{cancer}_v45.sql
       ↓
2. Parquet cache (in cache/) — so re-runs don't re-query BQ
       ↓
3. pre_process(): clean rows, normalize event types, drop bad rows
       ↓
4. process_data(): symmetric column layout, branch-aware filtering
       ↓
5. get_df_filtered(): drop patients below MIN_SNOMED_SEQ_LENGTH (10), cap at MAX_*_SEQ_LENGTH (30)
       ↓
6. get_df_capped_shuffled(): cap to MAX_CANCER_PATIENTS (30K) and MAX_NO_CANCER_PATIENTS (30K)
       ↓
7. build_vocab(): per-branch vocabulary; rare codes (frequency < 30) → '<RARE>' token
       ↓
8. encode_df(): replace each code string with its integer ID
       ↓
9. get_seq_data(): pad each patient's sequence to length 30
       ↓
10. train_test_split(): 70/15/15 (or similar) using SPLIT_SEED
       ↓
11. build_model() → train() → evaluate() → save()
```

**Key files written during a run**:
- `cache/{hash}.parquet` — BQ result cache (auto-invalidates if SQL changes)
- `saved_models/{cancer}_lstm_v147_v45.keras` — trained model
- `tuner_results/{cancer}_lstm_v147_v45_tuning_*` — HP tuner state + trial logs

---

## 8. Configuration

The active config lives in **cell 79** (Execution → Config). Most-relevant knobs:

### What cancer to run

```python
TARGET_CANCER = 'lung'   # change to 'prostate' to use sql/prostate_v45.sql
```

The cancer profile (defined in cell 76) is auto-applied based on this. It pulls in:
- `MAX_CANCER_PATIENTS` / `MAX_NO_CANCER_PATIENTS` (cohort caps)
- `concept_codelist_paths` (CSV files of SNOMED codes considered relevant)
- `relevance_patterns` (regex patterns for the explainability metric)
- `semantic_reference_phrases` (used in semantic relevance scoring)

### Branch toggles

```python
config['use_snomed_branch'] = True   # SNOMED LSTM ON
config['use_med_branch'] = False     # MED LSTM OFF — try flipping this to True!
```

### Sequence shape

```python
config['MAX_SNOMED_SEQ_LENGTH'] = 30   # cap per-patient sequence at 30 codes
config['MIN_SNOMED_SEQ_LENGTH'] = 10   # require at least 10 codes to include patient
config['min_code_frequency'] = 30      # SNOMED codes seen <30 times → '<RARE>' token
```

### Training-time loss / metric

```python
config['sensitivity_weight'] = 3.0    # weighted_sens_spec metric weights sens 3x
config['specificity_weight'] = 1.0
config['epochs'] = 25
config['early_stopping_patience'] = 8
config['lr_reduction_patience'] = 4
```

### Threshold selection

```python
config['proba_threshold'] = 0.5            # default decision threshold
config['use_optimal_threshold'] = True     # search for best threshold on val ROC
config['use_sensitivity_floor'] = True     # picks threshold s.t. sens >= 0.95, then maximize spec
```

The `use_sensitivity_floor=True` setting is critical — it tells the model: "first ensure sensitivity ≥ 0.95, then optimise specificity." This matches the project's >95% sens target.

### Seeds

```python
config['SEED'] = 42              # numpy/tf/python random seed
config['SPLIT_SEED'] = 42        # train_test_split seed
```

These are the primary knobs for your reproducibility study (vary them, observe variance).

---

## 9. HP tuner

Cell 72 implements a **custom Keras Tuner subclass** called `SensSpecTuner` that wraps `BayesianOptimization`.

**What it does differently from a generic tuner**:
- After each trial, it loads the model, predicts on the validation set, and runs ROC-curve analysis.
- It picks the operating threshold per the same logic the production code will use (sensitivity floor or weighted sens+spec).
- The trial's score is computed at that threshold, not at default 0.5.
- It logs the best-so-far config after each trial.

**Default tuning budget**: `tuner_max_trials = 30` (per data-param combo).

**Hyperparameters searched** (v136 ranges):

| HP | Range | Notes |
|---|---|---|
| `embedding_dim` | [16, 24, 32] | small embedding, vocab is large |
| `snomed_lstm_units` | [32, 48, 64, 96] | LSTM hidden size |
| `med_lstm_units` | [16, 24, 32] | only used if MED branch enabled |
| `other_dense_units` | [8, 16, 32] | static-feature branch size |
| `dense_1_units` | [16, 32, 48, 64] | classifier head size |
| `learning_rate` | 5e-5 to 5e-4 (log) | AdamW LR |
| `embedding_dropout` | 0.30 to 0.55 | regularisation |
| `dense_dropout` | 0.40 to 0.65 | classifier regularisation |
| `l2_reg` | 1e-3 to 1e-2 (log) | dense kernel L2 |
| `weight_decay` | 5e-4 to 1e-2 (log) | AdamW weight decay |
| `label_smoothing` | 0.0 to 0.15 | softens the binary target |
| `mask_rate` | 0.30 to 0.55 | random token-replacement augmentation |
| `cancer_class_weight` | 1.0 to 2.0 step 0.25 | extra weight on positive class |
| `batch_size` | [32, 64] | smaller batches → better generalisation |

**Where results go**: `tuner_results/{cancer}_lstm_v147_v45_tuning_mcf30_snomedseq30_medseq30_sw1.0/`. Trial logs are JSON; the tuner can be resumed or queried.

---

## 10. Validation playbook

Concrete, ordered tasks. Pick the first one and ship before moving on.

### Task 1 — Reproducibility study (start here)

**Question**: Does v147's lung result actually swing by 0.20 single-trial as Youssef claimed?

**Method**:
1. Run the notebook end-to-end on lung **5 times**. Vary only:
   - `SEED ∈ {42, 1, 13, 99, 2024}`
   - `SPLIT_SEED ∈ {42, 1, 13, 99, 2024}`
2. For each run, record: test sens, test spec, test AUC, relevance score (model-level).
3. Compute mean ± std across the 5 runs for each metric.

**Deliverable**: a table like

```
metric        | seed=42 | seed=1 | seed=13 | seed=99 | seed=2024 | mean ± std
sens          | 0.84    | 0.79   | 0.91    | 0.83    | 0.86      | 0.85 ± 0.04
spec          | ...
relevance     | ...
```

**Outcome**: confirms (or refutes) the 0.20 swing. Either way, you have hard numbers to discuss.

### Task 2 — Variance reduction

If Task 1 confirms instability, investigate **why**. Likely culprits ranked by suspicion:

1. **Small validation set** — `early_stopping` uses val score; if val is small, "best epoch" is noisy. Check `len(y_val)`. Consider increasing val ratio.
2. **`SPLIT_SEED` impact on test composition** — with 30K cap and class imbalance, different splits may yield very different cancer-density tests.
3. **High `mask_rate` (0.30-0.55)** — strong augmentation can amplify variance. Try `mask_rate=0.0` and see if variance drops.
4. **`restore_best_weights=True` + noisy val metric** — the "best epoch" can be a fluke peak.
5. **Class weighting variance** — with `compute_class_weight('balanced', ...)`, weights depend on actual fold. Tiny shifts can amplify.

**Deliverable**: 1-2 ablation tables showing which factor moves variance most.

### Task 3 — HP tuning

Once stable, run the tuner cell (cell 72) for each cancer.
- Default `tuner_max_trials=30`. On A100, expect ~6-12 hrs depending on dataset size.
- Set `use_sensitivity_floor=True`, `sensitivity_floor=0.95` to match project target.
- Save tuner output (`tuner_results/...`) and the best config.

**Deliverable**: best HP config per cancer, with sens/spec/relevance achieved.

### Task 4 — Cross-cancer validation

Run on **prostate** (`prostate_v45.sql` is already prepared) using either:
- v147's default config, or
- The HP-tuned config from Task 3 (preferred).

Compare lung vs prostate. Report whether the model generalizes or regresses on prostate.

**Deliverable**: side-by-side metric table (lung vs prostate), each at the operating threshold that hits sens ≥ 0.95 if possible.

### Task 5 — Try `use_med_branch=True`

This is currently OFF in v147. Adding the medication branch could lift performance materially since med history is informative for cancer prediction. Run with it ON and compare to v147 baseline.

**Deliverable**: with-med-branch vs without-med-branch comparison.

---

## 11. VM setup

Same VM you've been using for the prostate XGB pipeline: `cts-ai-dev-gpu-05-a100` (1× A100 GPU). Note that "vm5" in Youssef's docs = "gpu-**05**" — same machine.

### One-time setup

```bash
# SSH in
gcloud compute ssh keerthikovelamudi_cthesigns_com@cts-ai-dev-gpu-05-a100 \
  --zone=us-central1-c

# Create your own conda env (don't share Youssef's)
conda create --name y-dev-keerthie python=3.11 -y
conda activate y-dev-keerthie

# Install deps from Youssef's reqs files
cd ~/work_lstm  # or wherever you set up your working dir
pip install -r reqs_colab_entrprise.txt
pip install -r nb_reqs.txt

# BigQuery auth (one-time)
gcloud auth application-default login --no-launch-browser
```

### Per-session

```bash
gcloud compute ssh keerthikovelamudi_cthesigns_com@cts-ai-dev-gpu-05-a100 \
  --zone=us-central1-c

conda activate y-dev-keerthie

cd ~/work_lstm

# Start Jupyter on a port that doesn't clash with Youssef's (he uses 8085)
jupyter notebook --no-browser --port 8086 \
  --NotebookApp.max_buffer_size=25000000000 \
  --NotebookApp.iopub_data_rate_limit=25000000000 \
  --NotebookApp.iopub_msg_rate_limit=10000 \
  --NotebookApp.rate_limit_window=10.0
```

The big buffer settings let Jupyter handle the large outputs the LSTM cells produce (model summaries, full ROC arrays, etc.).

To open the notebook in your browser, set up port forwarding from a separate terminal on your laptop:

```bash
gcloud compute ssh keerthikovelamudi_cthesigns_com@cts-ai-dev-gpu-05-a100 \
  --zone=us-central1-c -- -L 8086:localhost:8086
```

Then visit `http://localhost:8086` in your browser.

### Working directory layout (suggested)

```
~/work_lstm/
├── lung_lstm_v147_bq_v45.ipynb       ← copy from Youssef's gs-y-dev/
├── sql/
│   ├── lung_v45.sql                  ← copy
│   └── prostate_v45.sql              ← copy
├── data/
│   └── codelists/
│       └── prostate_cancer_relevance/  ← codelists referenced by cancer profile
├── cache/                            ← parquet cache (auto-created)
├── saved_models/                     ← model artifacts (auto-created)
├── tuner_results/                    ← HP tuner state (auto-created)
└── notebooks_output/                 ← your variant notebooks (e.g. v147_seed1.ipynb)
```

**Important**: copy from Youssef's `/home/youssefdrissi_cthesigns_com/gs-y-dev/`, don't symlink — you'll be modifying the notebook with your own seeds/configs and you don't want to overwrite his.

---

## 12. Running the notebook

In the notebook UI:

1. **Run cells 7-12** (config helpers + GPU check) — verify it sees the A100.
2. **Skip down to cell 79** (Execution Config). Confirm `TARGET_CANCER = 'lung'`. Confirm `SEED = 42`.
3. **Run cells 13-44** (data loading + processing). First run will hit BigQuery; later runs read from `cache/`.
4. **Run cells 45-51** (build + train model). Watch the per-epoch sens/spec on val.
5. **Run cells 52-67** (save, evaluate, demographics).
6. **Run cell 70** (explainability) — this is where the relevance score comes from.
7. **Skip cell 72** (HP tuning) for the first pass — it takes hours.
8. **Run cells 73-83** if Youssef has structured the notebook to do an end-to-end run.

For your reproducibility study, you'll be running cells 13 onwards 5 times with different SEED/SPLIT_SEED — recommended to:
- Save each run as a new notebook copy: `lung_v147_seed42.ipynb`, `lung_v147_seed1.ipynb`, etc.
- Or convert to a `.py` script and parameterize from the command line.

---

## 13. Reading metrics

The notebook prints these per run. Key ones:

| Metric | Formula | Target | Why it matters |
|---|---|---|---|
| **Sensitivity (Sens)** | TP / (TP + FN) | > 95% | Of cancer patients, how many we caught. Missing a cancer is the worst error. |
| **Specificity (Spec)** | TN / (TN + FP) | > 85% | Of non-cancer patients, how many we cleared. False alarms cause anxiety + wasted referrals. |
| **AUC (ROC AUC)** | area under ROC curve | n/a | Threshold-free measure of separation. 0.5 = random, 1.0 = perfect. |
| **PPV (Positive Predictive Value)** | TP / (TP + FP) | n/a | Of patients flagged, how many actually have cancer. Low PPV → many false alarms. |
| **NPV (Negative Predictive Value)** | TN / (TN + FN) | n/a | Of patients cleared, how many were truly cancer-free. |
| **Relevance** | top-K influencer regex hit rate | > 80% | Of the top-10 most influential features, how many match clinical relevance regex. |

**Confusion matrix layout** (the notebook prints this):

```
                  Predicted
                  Negative   Positive
Actual Negative   [ TN ]     [ FP ]
       Positive   [ FN ]     [ TP ]
```

**Threshold matters**: A model outputs P(cancer) ∈ [0,1]. The decision threshold (default 0.5) controls the sens/spec tradeoff. The notebook auto-picks an optimal threshold using the val set; check `find_optimal_threshold()` output.

**With `use_sensitivity_floor=True, sensitivity_floor=0.95`**: the threshold is chosen to ensure val sens ≥ 0.95, then to maximize val spec subject to that constraint. This matches the project's target of "catch ≥95% of cancers".

---

## 14. Common pitfalls

1. **Forgetting to switch the cancer profile**: If `TARGET_CANCER='lung'` but you change the SQL to prostate manually, the cancer profile won't match → relevance regexes are wrong → relevance score is meaningless. Always set `TARGET_CANCER` and let the profile auto-load.

2. **BQ cache out of date**: If you change the SQL, the cache hash should change automatically — but if you change e.g. the data version externally, you might still read stale cache. Set `config['force_bq_refresh'] = True` once to force refresh, then back to False.

3. **Sharing Youssef's conda env**: His `y-dev` env may pip-install conflicts with yours. Use a separate env (`y-dev-keerthie`) so neither of you breaks the other.

4. **Running multiple Jupyter kernels on the same GPU**: TF will OOM if Youssef's kernel and yours both grab full GPU memory. Set `tf.config.experimental.set_memory_growth(gpu, True)` (cell 8 already does this).

5. **Comparing across different SQL versions**: A v45 lung run and a v44 prostate run are NOT comparable. Always state which `data_version` (v45 here) was used.

6. **Reading variance into a single run**: 1 run is anecdotal. The whole point of Task 1 is to **never** report a single-seed result without its variance.

7. **`MIN_SNOMED_SEQ_LENGTH=10` filtering**: Patients with <10 SNOMED codes are dropped. If your cohort cap is 30K and many patients are filtered, your effective sample can be much smaller. Check `len(df)` before and after `get_df_filtered()`.

8. **Running on the wrong VM**: The team uses `cts-ai-dev-gpu-05-a100` ("vm5"). If you SSH into a different GPU VM, you won't see Youssef's home dir, codelists, or the existing cache.

---

## 15. Glossary

**LSTM (Long Short-Term Memory)**: A type of recurrent neural network designed to learn from sequences. It maintains a "memory" cell that selectively forgets/remembers information across timesteps. Good for things where order matters.

**Embedding**: A learned mapping from discrete tokens (e.g. SNOMED code IDs) to dense vectors. Instead of one-hot encoding ("PSA test = position 4729 of a 5000-dim vector"), embeddings give each code a small (16-32 dim) vector that the model learns. Similar codes end up near each other in vector space.

**Attention pool**: Instead of just taking the LSTM's last output (which forgets early events) or averaging all outputs (which dilutes), attention learns *weights* per timestep — "this code at this position is more important" — then sums weighted outputs.

**Sequence padding**: All patients have different numbers of events. Padding fills shorter sequences with a special "0" token so they all have length 30. The model learns to ignore the 0s.

**`<RARE>` token**: SNOMED codes seen <30 times in the training set are replaced with this single token. Saves vocab size + prevents overfitting on noise.

**Sequence masking augmentation**: During training, randomly replace some tokens with `<RARE>`. Forces the model to not depend on any single code. (Like dropout but on input tokens.)

**Class weighting**: Cancer is rare (~10% of cohort). Without weighting, the model can get high accuracy by predicting "no cancer" for everyone. Class weighting makes the loss for missing a cancer patient (FN) cost more than a false alarm (FP).

**Label smoothing**: Instead of training labels being exactly 0 or 1, soften to e.g. 0.05 or 0.95. Prevents the model from becoming over-confident → typically improves calibration and generalization.

**Focal loss**: Down-weights "easy" examples (already correctly predicted with high confidence) so the model focuses on the hard ones near the decision boundary.

**Bayesian optimization (in HP tuning)**: Instead of grid or random search, model the relationship "config → score" and pick the next config most likely to improve. Sample-efficient.

**Integrated Gradients (IG)**: Attribution method for explainability. Says "feature X contributed Y to the prediction". Works on differentiable models like neural nets.

**TimeSHAP-style temporal ablation**: Variant of SHAP for sequences. Removes events one at a time, sees how prediction changes, attributes the change to that event.

**Sensitivity floor**: Threshold-selection rule that says "first ensure sens ≥ X, then maximize spec." Used for clinical models where missing a cancer is much worse than a false alarm.

**Weighted Sens+Spec metric**: A single training-time metric `(sw·sens + spw·spec) / (sw + spw)` — Youssef uses sw=3, spw=1 to bias the metric (and hence early stopping) toward high sensitivity.

---

## Final sanity check before you start

Before kicking off Task 1, verify:

- [ ] You can SSH into `cts-ai-dev-gpu-05-a100`
- [ ] You can `conda activate y-dev-keerthie` (your own env)
- [ ] `pip list | grep tensorflow` shows TF installed
- [ ] `nvidia-smi` shows the A100 (and isn't fully occupied by Youssef)
- [ ] You can `cd ~/work_lstm`, find the notebook + `sql/` + reqs locally
- [ ] You can open Jupyter in your browser via port forwarding
- [ ] You can run cells 7-12 (the cheap setup cells) without errors
- [ ] You can read `lung_v45.sql` and recognize it's a `SELECT ... FROM` BQ query

Once those all check, you're ready. Run the lung notebook end-to-end with `SEED=42` first as a sanity baseline — confirm the numbers roughly match what Youssef has been seeing — then start your reproducibility study (Task 1).

If something's broken, ping Youssef with the specific cell number + error message, not "it doesn't work."































What the actual lung v147 run shows

  Test metrics (single seed):

  ┌───────────────────────┬──────────┬────────┬─────────────────────────────────────────────────┐
  │        Metric         │ Achieved │ Target │                     Status                      │
  ├───────────────────────┼──────────┼────────┼─────────────────────────────────────────────────┤
  │ Sensitivity           │ 96.09%   │ >95%   │ ✓                                               │
  ├───────────────────────┼──────────┼────────┼─────────────────────────────────────────────────┤
  │ Specificity           │ 70.12%   │ >85%   │ ✗                                               │
  ├───────────────────────┼──────────┼────────┼─────────────────────────────────────────────────┤
  │ AUC                   │ 83.10%   │ —      │                                                 │
  ├───────────────────────┼──────────┼────────┼─────────────────────────────────────────────────┤
  │ F1                    │ 85.02%   │ —      │                                                 │
  ├───────────────────────┼──────────┼────────┼─────────────────────────────────────────────────┤
  │ Explainability rating │ 29.5/100 │ 80+    │ ✗ (massive gap)                                 │
  ├───────────────────────┼──────────┼────────┼─────────────────────────────────────────────────┤
  │ └─ Faithfulness       │ 0.370    │ —      │ drop@top barely beats drop@rand (0.10 vs 0.06)  │
  ├───────────────────────┼──────────┼────────┼─────────────────────────────────────────────────┤
  │ └─ Clinical relevance │ 0.321    │ 0.80   │ only 16/50 top features matched                 │
  ├───────────────────────┼──────────┼────────┼─────────────────────────────────────────────────┤
  │ └─ Consistency        │ 0.006    │ —      │ patient↔model top features almost never overlap │
  └───────────────────────┴──────────┴────────┴─────────────────────────────────────────────────┘

  Sensitivity-floor logic is working — threshold (0.4944) is auto-picked to satisfy sens ≥ 0.95. The issue is everything around it.

  Three big problems I can see in this single run

  1. The model is gaming an age confound

  Look at the data summary:
  - Cancer mean age: 74.7 ± 10.9
  - Non-cancer mean age: 42.7 ± 21.3

  That's a 32-year gap. The model only needs to learn "old patient = high risk" and it gets ~80% of the signal for free. This is a data-side problem
  (Abbas/Youssef/Alex/Marta own it per the brief) but you'd be the one to catch it during validation.

  2. Top features are mostly "frequent-attender" noise, not cancer

  The top-10 model-level features in this run:
  1. Inhaler technique - poor (relevant)
  2. Right dorsalis pedis pulse palpable (a foot pulse check — irrelevant to lung cancer)
  3. Temperature normal (admin)
  4. Seen by consultant (admin)
  5. Housing adequate (social — irrelevant)
  6. COPD care plan (relevant)
  7. Asthma review (relevant)
  8. Independent walking (mobility check — irrelevant)
  9. Physical activity signposted (admin)
  10. Medication quantities checked (admin)

  Most of the top ten are admin codes that frequently-attending elderly patients accumulate. Hence relevance=0.32 and faithfulness=0.37 (ablating top features barely
  changes prediction). This is exactly the confound the project's "shortcut codes" exclusion was designed for, but with exclude_shortcut_codes=False in the config it's
   off.

  3. Training is unstable within a single run

  Epoch 7  → val_weighted_sens_spec = 0.8871   ← peak
  Epoch 8  → 0.8114
  Epoch 9  → 0.6118
  Epoch 10 → 0.3809
  Epoch 15 → 0.2598   ← early stopping kicks in

  Best weights restored from epoch 7 — but if the val set were a different 1,023 patients (only 10% of cohort), "epoch 7 best" might not be the same place. This is
  Youssef's 0.20 variance issue manifesting within a single run — the val score is noisy enough that early stopping picks a fragile peak.

  Updated validation playbook (now that I've seen real numbers)

  Your README's task list is right but I'd reorder + add specifics:

  Task 1A — confirm age confound is real
  - Quickest win: in the data prep code, mask out age (set everyone to median age or shuffle). Run again. If sens stays ≥0.95, the model has learned cancer signal. If
  sens crashes to ~0.60, it was riding age. One-line config change to test.
  - Same for the gender one-hots.

  Task 1B — reproducibility study (5 seeds)
  - Same as I had in the README. Confirm the 0.20 swing.

  Task 2 — turn on shortcut exclusion
  - Set config['exclude_shortcut_codes'] = True.
  - The lung profile already has 24+ shortcut patterns defined (admin, life-assurance, DNA codes, etc.). Run again and see if relevance jumps from 0.32.

  Task 3 — explainability triage
  - Faithfulness 0.37: try shap_ablation_mode='per_position_mode' (currently marginal_sample) and the 'shap_eg' baseline. The doc says EG diverges most on static
  features.
  - Relevance 0.32: turn on use_semantic_relevance=True (set False now), see if it lifts the matched count.
  - Consistency 0.006: this is shocking — patient explanations almost never match model-level. Likely caused by the small per-patient sequence (30 events) producing
  high-variance per-patient deltas. Could try MAX_SNOMED_SEQ_LENGTH=100 (was v116 default).

  Task 4 — turn on the MED branch
  - Same notebook now has use_med_branch=False. Med codes are clinically informative. Worth comparing.

  Task 5 — HP tuning (longer-running, defer until 1-4 are clean)

  Task 6 — cross-cancer (prostate) as before.

  What I'd write up first as your "Day 1" deliverable

  A 1-page memo to Youssef titled something like "v147 baseline reproduction + 3 immediate concerns":

  1. Confirmed reproduction of v147 lung run (your numbers match within 0.0X)
  2. Flag the age confound (mean 74.7 vs 42.7) — propose age-blinded ablation
  3. Flag that exclude_shortcut_codes=False is letting admin codes pollute the top-10
  4. Flag the within-run val instability (val score crashed 0.88 → 0.26 over 8 epochs)

  That's a strong opening because all three are visible from one already-completed run, and each is directly actionable with a config flag.

  Want me to update the README with these concrete findings + the new task ordering?c