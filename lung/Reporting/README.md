# Lung_Cancer_K1 / Reporting

Shared, Claude-built explainability reports for a cancer model — **local HTML/PDF, nothing uploaded or
published**, works against any pipeline version (`V1`, `V3_categorized`, `V3_10Kcat_7kcon`, …) and **any
cancer** (design fixed; clinical content per-cancer). Two report families:

1. **The FP/FN deep-dive** — a cohort-level "why the model errs" report (confusion tiles, per-patient cards,
   aggregate SHAP, threshold trade, ceiling). → **`HOW_TO_GENERATE.md`**.
2. **The per-patient before-vs-gap sheets** — go deeper, pulling the **source EMIS record around each error**
   to answer *"did the mistake actually matter?"* (FN missed cancers; FP false alarms ±1yr). These need
   BigQuery. → **`FN_before_vs_gap_PROMPT.md`**, **`FP_before_after_PROMPT.md`**, method in
   **`PER_PATIENT_METHOD.md`**.

Everything is built **by Claude** from the run's outputs and **saved to disk** — no data leaves your
infrastructure. All reports carry `patient_guid`s + clinical detail → **local only, never published/hosted.**

> **Cancer-agnostic?** The **layout/design is fixed for every cancer**, and all the numbers/cards/SHAP/sweeps
> come from *your* run. But the **narrative** (the headline error pattern, the archetype emphasis, the
> "ceiling" story, the clinical red-flag examples) must be **derived from your model's own data** — the shipped
> lung preview asserts lung's age-bias findings, which won't hold for every cancer. `HOW_TO_GENERATE.md` marks
> every such spot with **⚙ derive-from-data**; don't copy lung's conclusions to another cancer.

**What's in the deep-dive:** a card for every FN/FP (age/sex/ethnicity, **`patient_guid`**, on-record
categories, ↓ lowered / ↑ raised SHAP factors), an age strip, error archetypes, aggregate SHAP per segment,
the **internal + held-out threshold trade**, and a "ceiling — what's genuinely hard / what could move it"
section — with **full feature names** and an **instant plain-language hover tooltip on every feature, metric,
legend chip, confusion tile and table column** (hover any label → plain English, no ML jargon).

**What's in the per-patient sheets:** one card per FN/FP with `age · sex · anchor · guid`, a plain-English
**"why missed / why flagged"** line, a **cancer-workup** line, and side-by-side code columns — FN: model
window vs the excluded 12-month gap; FP: model window · 12mo-before · 12mo-after — every code coloured
new / recurring / medication, same hover tooltips, plus a filterable Excel.

| File | What |
|---|---|
| `HOW_TO_GENERATE.md` | **Deep-dive — start here.** Steps + the exact prompt to paste into Claude; builds a local `deepdive.html` + `deepdive.pdf` (does not publish). |
| `FN_before_vs_gap_PROMPT.md` | **Per-patient FN** sheet — missed cancers, model-window vs the excluded 12-month gap. |
| `FP_before_after_PROMPT.md` | **Per-patient FP** sheet — false alarms, ±1 year around the flag (model · before · after). |
| `PER_PATIENT_METHOD.md` | Shared method + **per-cancer config for ANY cancer** (§D2/§D3 — detector derived from each cancer's curated codelist; starters for all active cancers) for the two per-patient sheets. Read before the FN/FP prompts. |
| `explainability.py` | **reference copy** of the SHAP generator (produces the deep-dive's inputs); source of truth is the pipeline's `3_Modeling/explainability.py`. |
| `contrast_report/` | **Automation (no Claude).** A deterministic CLI that builds the FN/FP sheets + deep-dive from a run + per-cancer config: `python -m contrast_report --run <dir> --config contrast_report/configs/<cancer>.yaml --cohort all`. Reproduces the sheets exactly (lung: FN 19/3/3, FP 5/29/4); patient data stays local. Use this to run reports from code/CI instead of pasting the prompts. |
| `README.md` | this overview |

*(The prompts are self-contained — they include the SQL, the workup detector and the full build spec, so no
scripts are needed to use them. The byte-exact **lung** reference implementation (`contrast_lib.py`,
`build_fn.py`, `build_fp.py`) is preserved with the lung run at
`…/10k_Cat/results_final/10k7k_wc_ensemble/report_src/`, for anyone wanting to reproduce lung exactly.)*

## Steps to follow

**A) Deep-dive (cohort-level — needs NO BigQuery):**
1. Run your model with explainability on (writes `modeling/explainability_internal/patient_explanations.csv`
   + `fe/features_p005_{h}_stable.parquet`; `--heldout` optionally adds held-out preds). That's all the data it uses.
2. Paste **`HOW_TO_GENERATE.md`** into Claude — swap `<RUN_DIR>`, model display name + internal AUROC; apply
   the **⚙ derive-from-data** notes for your cancer.
3. Claude reads those files and saves `deepdive.html` + `deepdive.pdf` locally. Share internally.

**B) Per-patient FN/FP sheets (deeper — need BigQuery):**
1. Read **`PER_PATIENT_METHOD.md`** and clear its **"Before you start"** checklist (BQ access + ~$6, schema-A
   explainability, **your run's raw-events cache**, **your cancer's §D2/§D3 config**).
2. Paste **`FN_before_vs_gap_PROMPT.md`** (missed cancers) and/or **`FP_before_after_PROMPT.md`** (false
   alarms) into Claude — swap `<RUN>`, apply your cancer config (⚙).
3. Claude identifies the errors, derives each anchor, **runs the BigQuery pulls**, and saves the
   `*.html` + `.pdf` + `.xlsx` locally.

## Inputs at a glance — is it "everything from explainability"?

**The deep-dive is built purely from the explainability outputs — nothing else, no BigQuery.** The
**per-patient sheets need more**: the same explainability outputs (in schema A) **plus** your run's raw-events
cache (for anchors + the model-window column) **plus** a live BigQuery pull of the source EMIS record (for the
excluded-year codes the model never saw).

| Report | Inputs it reads | BigQuery? |
|---|---|---|
| **Deep-dive** | `explainability_internal/patient_explanations.csv` + `fe/…_stable.parquet` (+ optional held-out preds) | **No** |
| **Per-patient FN/FP** | ↑ those (schema A / signed SHAP) **+** your run's raw-events cache **+** `EMIS_BULK_DATA_PROCESSED.CareRecord_Observation` / `…Problem` / `Prescribing_*` / `Coding_*` | **Yes** (~$6/cohort) |

**Sharing:** you share the local **`.html` / `.pdf` / `.xlsx`** internally (SharePoint / shared drive / email) —
never a hosted version. Layout preview (deep-dive look):
`https://claude.ai/code/artifact/b14ebe22-1513-478d-ab44-4214316e20d1` — a hosted, deliberately **guid-free**
*example*; your real output is a local, patient-identifiable file.

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

**Deep-dive** defaults: **top-20 factors per aggregate-SHAP table** and **up to 12 factors per patient card**
— adjustable (tell Claude *"use top-N factors"*). The **per-patient sheets** list **all** codes (no top-N).
Operating **threshold** is **0.50** for both — change it if you report at a different operating point.

## Requirements

Claude access (everyone has it) + one model run's explainability outputs. The pipeline's
`explainability.py` needs `shap` + `xgboost<3` — **tree-exact SHAP** for tree models (LightGBM, RF,
XGBoost, GBM, CatBoost, DecisionTree + soft-voting ensembles), with an automatic **fallback for non-tree
models** (`LinearExplainer` for linear/logistic baselines, `KernelExplainer` for anything else). That's
part of the pipeline env, not this folder. (PDF is rendered from the HTML with headless Chrome.)
The **per-patient sheets** additionally need **BigQuery access to `prj-cts-ai-dev-sp`** (~$6/cohort) — see
`PER_PATIENT_METHOD.md` → "Before you start".
