# contrast_report — automated FN/FP sheets + deep-dive (deterministic, no Claude)

The **code** version of the `../` prompts: one command builds the per-patient FN & FP sheets and the FP/FN
deep-dive from a run + a per-cancer config. **No LLM in the loop → deterministic, auditable, and patient data
never leaves your infrastructure.** (The prompts in `../` remain for when you want Claude's richer narrative;
this is the automation path.)

```
python -m contrast_report --run <RUN_DIR> --config contrast_report/configs/lung.yaml --cohort all --out <OUT_DIR>
```
`--cohort`: `fn` | `fp` | `deepdive` | `all`. Outputs (local): `fn_contrast.html/.pdf` + `fn_before_vs_gap.xlsx`,
`fp_contrast.html/.pdf` + `fp_before_after.xlsx`, `deepdive.html`.

## What it does (all from your run, no external calls except BigQuery)
1. **Identify** FN/FP from `modeling/explainability_internal/patient_explanations.csv` (schema A) + the stable matrix.
2. **Derive** each patient's anchor + model-window from the raw-events cache (`cache_uri`).
3. **BigQuery**: pull the excluded-year record (FN: `[anchor−gap, anchor)`; FP: `±gap`). Cached to
   `<OUT>/bq/` so re-runs are free.
4. **Build** the HTML/PDF/Excel with the same design as the prompt output (hovers, colours, badges, why-lines).

Auto-derived from the run: **horizon/gap** (from the stable-matrix name), **threshold** (default 0.50 — the
internal operating point; *not* the held-out `operating_threshold_*.json`), **FE feature decoding**.

## Per-cancer config (`configs/<cancer>.yaml`)
Everything cancer-specific lives here — nothing is hardcoded in the code:
| field | what |
|---|---|
| `organ` | organ word in labels/narratives |
| `cache_uri` | ⚙ your run's raw-events cache (gs:// or local .parquet) |
| `workup_terms` | organ add-on to the generic oncology core (referrals/imaging/clinics/biopsy) |
| `relevant_terms` | organ-relevant highlighter (orange chips) |
| `lookalike` | the confounding phenotype (FP "why flagged") |
| `threshold`, `gap_months` | `null` = auto-derive; set to override |

Ships with `lung` (validated: reproduces FN 19/3/3, FP 5/29/4 exactly) plus **draft** configs for breast,
prostate, bladder, leukaemia, lymphoma, melanoma, ovarian. **Validate a draft's `workup_terms`/`relevant_terms`
against that cancer's curated codelist before trusting the numbers** — ideally derive the workup detector from
the codelist / FE categories rather than keywords (see `../PER_PATIENT_METHOD.md` §D).

## Requirements
Python (pandas, pyarrow, pyyaml, openpyxl), `bq` + `gcloud` CLIs authenticated for the BigQuery/cache pulls
(~$6/cohort, cached after first run), and headless Chrome for the PDF. **Outputs are patient-identifiable →
local only, never publish/upload.**

## Files
- `__main__.py` — CLI · `pipeline.py` — identify/anchors/BigQuery/FN+FP builders · `deepdive.py` — deep-dive
  builder (parameterized; narrative is templated/derived, not Claude-written) · `render.py` — shared HTML/Excel
  rendering + the workup/relevant detectors (set at runtime from config) · `configs/` — per-cancer configs.
