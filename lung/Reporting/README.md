# Lung_Cancer_K1 / Reporting

Shared way to explain **why a model makes false positives / false negatives** as a
**per-patient FP/FN deep-dive** — a single self-contained **local HTML file** (nothing uploaded or
published). Works against any pipeline version (`V1`, `V3_categorized`, `V3_10Kcat_7kcon`, …).

The report is built **by Claude** from the run's outputs and **saved to disk** — no build script to
install, and no data leaves your infrastructure.

**What's in it:** a card for every FN/FP (age/sex/ethnicity, **`patient_guid`** for traceability, on-record
categories, ↓ lowered / ↑ raised SHAP factors), an age strip, error archetypes, aggregate SHAP per segment,
the **internal + held-out threshold trade**, and a "ceiling — what's genuinely hard / what could move it"
section. Shown with **full feature names**, and **every feature, metric, legend chip, confusion tile and
table column carries an instant plain-language hover tooltip** — a clinician can hover any label or number
and be told what it means, in plain English (no ML jargon). Because it carries `patient_guid`s + clinical
detail it is **patient-identifiable → local only, never published**.

| File | What |
|---|---|
| `HOW_TO_GENERATE.md` | **Start here.** The steps + the exact prompt to paste into Claude, which builds the deep-dive and **saves a local `deepdive.html` + `deepdive.pdf`** (does not publish). |
| `explainability.py` | **reference copy** of the SHAP generator so you can see what produces the inputs — but the source of truth is the pipeline's `3_Modeling/explainability.py` (see below). |
| `README.md` | this overview |

**Two-step mental model:**
```
run your model  ─►  explainability outputs  ─►  paste the prompt into Claude  ─►  deepdive.html + deepdive.pdf (local)
   (pipeline produces them)                        (Claude reads, builds, saves locally — no upload)
```
You share the **`.html` or `.pdf`** (internal SharePoint / server / email).
Layout preview: `https://claude.ai/code/artifact/b14ebe22-1513-478d-ab44-4214316e20d1` (a hosted, deliberately
**guid-free** *example* of the look — your real output is a local, patient-identifiable file).

## Where the data comes from

The SHAP outputs are produced by the **pipeline's own** `3_Modeling/explainability.py` — it runs
automatically inside `run_v3.py --heldout` (and `4_Heldout/evaluate_heldout.py`). The `explainability.py`
in this folder is a **read-only reference copy** (so you can see the generator) — it may lag the pipeline's;
**always trust the pipeline's copy**. Every run writes:

- `…/explainability_internal/patient_explanations.csv` — every patient + segment (TP/FP/TN/FN) + risk factors
- `…/fe/features_p005_{h}_stable.parquet` — the stable feature matrix (age / sex / ethnicity / on-record categories)
- `segment_drivers.csv` + per-segment plots

Claude reads those directly and builds the report — no need to run anything from this folder.

## Adjustable

The report defaults are **top-20 factors per aggregate-SHAP table** and **up to 12 factors per patient
card** — both are adjustable: just tell Claude *"use top-N factors"* (per card and/or per segment) when
you paste the prompt. Same for the operating threshold (default 0.50).

## Requirements

Claude access (everyone has it) + one model run's explainability outputs. The pipeline's
`explainability.py` needs `shap` + `xgboost<3` — **tree-exact SHAP** for tree models (LightGBM, RF,
XGBoost, GBM, CatBoost, DecisionTree + soft-voting ensembles), with an automatic **fallback for non-tree
models** (`LinearExplainer` for linear/logistic baselines, `KernelExplainer` for anything else). That's
part of the pipeline env, not this folder. (PDF is rendered from the HTML with headless Chrome.)
