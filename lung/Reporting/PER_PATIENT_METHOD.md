# Per-patient FN/FP sheets — "what really happened" (method + per-cancer config)

Two **local, per-patient** reports that go one level deeper than the deep-dive (`../Reporting/`):
they pull the **source EMIS record around each error** to answer *"did the model's mistake actually
matter clinically?"*

| Sheet | Question | Prompt |
|---|---|---|
| **FN — missed cancers** | Do the missed cancers get diagnosed anyway in the year the model can't see? | [`FN_before_vs_gap_PROMPT.md`](FN_before_vs_gap_PROMPT.md) |
| **FP — false alarms** | Were the false alarms wasted screens, or did real disease / a workup follow? | [`FP_before_after_PROMPT.md`](FP_before_after_PROMPT.md) |

Each builds a self-contained **HTML + PDF + Excel**, saved locally. **They carry `patient_guid`s and full
clinical detail, so they are patient-identifiable → local only, never published/hosted.**

> **Standardization principle:** the report **design** (layout, colours, sort orders, sentence structure,
> hovers, Excel schema) is **identical for every cancer**, so reports look consistent across the team. The
> **clinical content** (the workup/red-flag code detector, the organ word, the "look-alike" profile) is
> **per-cancer** — set it in **Cancer config (§D2)**. The prompts here use **lung** as the worked example and
> mark every cancer-specific spot with **⚙**. Don't reuse lung's clinical content for another cancer.

---

## Before you start (checklist)

- [ ] **BigQuery access** to `prj-cts-ai-dev-sp` (`bq` CLI authenticated) + ~**$6 budget** (two ~0.5 TB
  source-table scans per cohort). Dry-run first.
- [ ] **Your run's outputs**: `modeling/explainability_internal/patient_explanations.csv` **in schema A**
  (`row, segment, y_true, prob, factor_1..N, shap_1..N` — i.e. *signed per-feature SHAP*), and
  `fe/features_p005_{h}_stable.parquet` (`{h}` = your horizon, e.g. `12mo`). **⚠ Schema A is required** —
  the ↓/↑ split and the "why missed/flagged" line need signed SHAP. If your run only wrote schema B
  (`Top_Factor_N` + `Factor_N_Contribution`, unsigned), re-run explainability to emit signed SHAP first, or
  drop the why-line.
- [ ] **Your run's internal raw-events cache** ⚙ — the withtext/lifetime parquet your FE consumed (for anchors
  + the model-window column). This path is **run/cancer-specific**; lung's is
  `gs://gcs-ai-dev-model-artifacts/keerthi/Lung_Cancer/raw_events/Astra_10yr_7k_withtext_internal.parquet`.
- [ ] **Your cancer's config** (§D2/§D3) — organ word, workup detector, {organ}-relevant terms, look-alike
  profile. **Works for any cancer** — the detector comes from your cancer's curated codelist (§D-preferred);
  §D3 has starters for the active cancers (lung/breast/prostate/bladder/leukaemia/lymphoma/melanoma/ovarian).

## Assumptions & limits — check these for YOUR model/team (⚙ = substitute)

These are the lung defaults; they are **not** universal. Confirm/replace each before you rely on the numbers:

- **⚙ Operating threshold.** FN/FP are defined at **0.50** (lung's internal operating point). **Use your
  model's actual operating threshold** — the error set changes with it. (Edit `prob<0.50` / `prob>=0.50` in step 1.)
- **⚙ Gap / window months.** Lung's gap is **12 months**; the SQL uses `INTERVAL 12 MONTH`. Set these to
  **your model's horizon/gap** (it predicts at `anchor − gap`). Change *every* month-interval in the SQL and the labels.
- **⚙ FE feature naming.** The plain-English decoder + "on record" logic assume the **V3 categorized pipeline**:
  `age_at_prediction`, `ageband_u50`, `g_eth_*`, and `<category>_<family>` features (`_decay_intensity`,
  `_count`, `_recency_rank`, …) with a per-category `<cat>_count`. A different FE (raw-SNOMED, other names) →
  adjust the decoder and the on-record categories.
- **⚙ Categorized model assumed.** The model-window / "on record" column shows **categories**; a raw-SNOMED
  model has none — show top codes instead.
- **⚙ Data source.** The SQL targets **CtheSigns EMIS** (`prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.*` +
  `Prescribing_*` / `Coding_*`). A different project/dataset → swap the table references.
- **⚙ File names / cache schema.** Assumes `features_p005_{h}_stable.parquet` and a withtext raw-events cache
  carrying `days_before_anchor` (used to derive the anchor). Different names/schema → point at yours, or derive
  the anchor another way.
- **Schema A (signed SHAP) required** — see "Before you start".
- **Prompt ≠ byte-identical.** A prompt-built report varies cosmetically run-to-run (LLM); only the lung
  reference scripts are byte-exact. The *data* and *design* are consistent; the exact bytes aren't.

## Why these need BigQuery (the deep-dive doesn't)

The model predicts at **anchor − 12 months**, so the feature/cache data **stops 12 months before the
anchor**. The whole point of these sheets is to look **inside that excluded window** (and, for FPs, the
year after) — data that is **not in any local file**. So each report:

1. identifies the error patients from the run's explainability outputs (local),
2. derives each patient's **anchor date** from the raw-events cache (local),
3. **queries the source EMIS tables** for what was recorded in the target window (BigQuery),
4. builds the HTML/PDF/Excel (local).

You need `bq` / BigQuery access to `prj-cts-ai-dev-sp` for step 3. Each of the two source-table scans is
~0.5–0.6 TB (the tables aren't clustered on patient), so budget ~$3 per query; dry-run first.

## The shared method (both sheets use this)

**A. Identify the error patients** — from `<RUN>/modeling/explainability_internal/patient_explanations.csv`
(schema `row, segment, y_true, y_pred, prob, factor_1..N, shap_1..N`), sorted by `row`. Align to the stable
matrix test slice for the `patient_guid`:
```python
pe = pd.read_csv(".../patient_explanations.csv").sort_values("row").reset_index(drop=True)
st = pd.read_parquet(".../fe/features_p005_12mo_stable.parquet", columns=["split","patient_guid","age_at_prediction"])
test = st[st.split=="test"].reset_index(drop=True)          # same order as pe
# FN = missed cancers; FP = false alarms (threshold 0.50)
fn = pe[(pe.y_true==1)&(pe.prob<0.5)];  fn_guids  = [test.patient_guid.iloc[r] for r in fn.row]
fp = pe[(pe.y_true==0)&(pe.prob>=0.5)]; fp_guids = [test.patient_guid.iloc[r] for r in fp.row]
```
**Verify the alignment** (load-bearing): the age from the stable slice must equal the age from the cache for
every patient — if it doesn't, the row→guid mapping is wrong. (Ours matched for all patients.)

**B. Derive the anchor** from **your run's** internal raw-events cache ⚙ (the withtext/lifetime parquet your FE
consumed — run/cancer-specific; lung's is
`gs://gcs-ai-dev-model-artifacts/keerthi/Lung_Cancer/raw_events/Astra_10yr_7k_withtext_internal.parquet`).
Columns include `patient_guid, event_date, days_before_anchor, snomed_c_t_concept_id, term, event_type,
age_at_anchor, sex, problem_status_description`. **`anchor = event_date + days_before_anchor`** (constant per
patient — check the spread is 0). This is also the **MODEL WINDOW** (left column): group cache events by
`(patient_guid, snomed_c_t_concept_id)` → distinct codes with a count + `event_type` + problem status. It is
the model's *filtered* view: 10-year cap, cohort filters applied, distinct-codes-with-count (not every row).

**C. Query the source EMIS record** (BigQuery) for the target window — see each prompt for the exact window.
Observations come from `CareRecord_Observation` joined to `Coding_ClinicalCode` (latest per
`code_id`+`source_practice_code`) for the SNOMED concept + term, deduped per `observation_guid`, and joined to
`CareRecord_Problem` for `problem_status_description`. Medications come from
`Prescribing_DrugRecord → Prescribing_IssueRecord → Coding_DrugCode`. **"New" = the code's first-ever
occurrence in the whole record falls inside the window** (compute `MIN(effective_date)` over all history, no
date filter, then keep those whose first date is in the window).

**D. The cancer-workup detector** (drives the badge + the "workup" line). Each gap code has a **SNOMED id
(`sct`)** and a **`term`**. It has **two parts** — a generic core (no config) + a small organ add-on:

**D-core — generic oncology pathway (cancer-AGNOSTIC, same for every cancer, needs no config).** Most of the
"workup" is generic: `suspected cancer`, `2-week-wait / fast-track referral`, `referral to oncology`,
`oncology service contact`, `malignant / neoplasm / carcinoma / metastasis`, `biopsy`. This catches the bulk
of the pathway for *any* cancer with zero per-cancer input.

**D-organ — the organ-specific add-on** (the imaging / clinic / procedure for that organ, e.g. chest X-ray +
bronchoscopy for lung; mammogram + breast clinic for breast). This is the *only* per-cancer part, and you can
source it **three ways (not just codelists)** — whichever you have:
1. the cancer's **curated codelist** — flag if the gap code's `sct` is in it (`…/codelist2.0/<cancer>_curated_codes_2.0.tsv`); or
2. the **run's FE categories** — flag by membership of an *oncology / referral / imaging / suspected-cancer* category (`code_category_mapping_2.0.json`); or
3. a short **keyword list** (the lung fallback below).

Because **D-core is generic and D-organ is derivable from data you already have**, this scales to **any cancer**.

**D-fallback — a keyword regex on the `term` text** (quick + human-readable, but **hand-maintained and
brittle** — use only when you can't load a codelist). Generic shape:
`suspected( <organ>)? cancer | fast-track | two week wait | 2ww | refer\w*.*(<organ>|oncolog|cancer|rapid) |
oncolog | <imaging/clinic/procedure/finding words for this cancer> | malignan | neoplasm | carcinoma | metasta`.
**This keyword approach is *why the lung version has a fixed word list*** — it's the fallback, not the design.
The lung example we shipped:
```
suspected( lung)? cancer | fast[- ]track | two week wait | 2 week rule | 2ww |
refer\w*.*(lung|chest|respirat|oncolog|cancer|thorac|medicine|medical service|rapid) |
referral to (respiratory|chest|oncolog) | chest x-?ray | standard chest | chest clinic | seen in chest |
respiratory (physician|medicine|clinic) | seen by respiratory | bronchoscop | lobectomy | thoracoscop |
pleural (asp|effusion|biopsy) | oncolog | lung (nodule|mass|cancer|lesion) | nodule of lung |
solitary (pulmonary )?nodule | malignan | neoplasm | carcinoma | metasta | (ct|computed tomog).*(chest|thorax)
```
Either way, **exclude** `not wanted` / `declined` and any `screening` term not for this organ (screening
declines / other-site screening are false positives — e.g. *"Screening for malignant neoplasm of cervix not
wanted"*, which we hit on lung and fixed). Symptoms (cough, chest pain, …) are deliberately **not** workup.

**D2. Cancer config — set these per cancer (everything else is fixed).** **This works for ANY cancer** — the
configs in §D3 are just **starters, not a whitelist**; any cancer with a curated codelist can be set up the
same way. The DESIGN is standardized across the team; only the **clinical content** changes per cancer. For
your cancer, supply:
| Knob | Lung value (example) | Yours |
|---|---|---|
| **Organ / cancer word** (replaces "lung" in labels & narratives) | lung | breast / prostate / bladder / … |
| **Workup detector** (§D) — a **generic oncology core** (cancer-agnostic, no config) **+** a small **organ add-on** from your codelist **or** FE categories **or** keywords | core + lung imaging/clinic terms | core (free) + your organ terms from codelist/categories (or fallback keywords, §D3) |
| **"<organ>-relevant" highlighter** (the orange chips) — the symptom/test/organ terms to highlight | cough, chest, COPD, haemoptysis, x-ray, nodule, oncolog … | the equivalent organ/symptom terms for this cancer |
| **"Look-alike" profile** (the FP narrative) — the confounding phenotype | older ex-smoker / COPD | e.g. BPH + raised PSA (prostate); benign breast disease (breast) |
The **anchor/cohort** and the SNOMED→category features come from *their* pipeline run — the report just reads
them; nothing else is cancer-specific.

**D3. Per-cancer configs.** The **detector for every cancer should come from that cancer's curated codelist**
(§D-preferred) — so it's *not* a hardcoded word list, and it covers **any** cancer, not just the ones below.
What genuinely needs setting per cancer (judgement calls, can't be derived) is the **organ word** and the
**look-alike profile**; the keyword `Workup`/`relevant` fragments shown are **optional fallbacks** (validate
against the codelist). Starters for the active cancers — **and any other cancer works the same way**:

- **Lung** ✅ (shipped): organ = `lung`; look-alike = *older ex-smoker / COPD*. Workup/relevant as in §D.
- **Leukaemia**: organ = `leukaemia`; look-alike = *reactive cytopenias / infection / ITP (benign blood-count changes)*. Detector: haematology 2ww referral, blood film, bone-marrow aspirate/trephine, flow cytometry/immunophenotyping, leukaemia/myelodysplasia/lymphoproliferative — **from the leukaemia codelist**.
- **Lymphoma**: organ = `lymphoma`; look-alike = *reactive lymphadenopathy / infection*. Detector: haematology 2ww, lymph-node biopsy/excision, CT neck/chest/abdo, PET-CT, bone marrow, lymphadenopathy, Hodgkin/non-Hodgkin — **from the lymphoma codelist**.
- **Melanoma**: organ = `melanoma`; look-alike = *benign naevus / seborrhoeic keratosis*. Detector: skin/dermatology 2ww, dermoscopy, excision/wide-local-excision, sentinel-node biopsy, mole/lesion, melanoma — **from the melanoma codelist**.
- **Ovarian**: organ = `ovarian`; look-alike = *benign ovarian cyst / IBS*. Detector: gynaecology 2ww, CA-125, transvaginal/pelvic ultrasound, CT abdo/pelvis, RMI, ovarian mass/cyst, oophorectomy — **from the ovarian codelist**.

- **Breast**: organ = `breast`; **look-alike** = *benign breast disease / fibroadenoma / cysts in older women*.
  - Workup: `suspected breast cancer | fast[- ]track | two week wait | 2ww | refer\w*.*(breast|oncolog|cancer) | mammogra | breast (clinic|ultrasound|lump|mass|cyst) | one[- ]stop | core biopsy | fine needle | fna | mastectomy | wide local excision | lumpectomy | malignan | neoplasm | carcinoma | metasta | oncolog`
  - {organ}-relevant highlight: `breast | mammogra | lump | nipple | axilla | cyst | ultrasound | biopsy | mastectomy | tamoxifen | malignan | neoplasm | metasta | cancer | oncolog`

- **Prostate**: organ = `prostate`; **look-alike** = *BPH / LUTS with raised PSA in older men*.
  - Workup: `suspected prostate cancer | fast[- ]track | two week wait | 2ww | refer\w*.*(prostate|urolog|oncolog|cancer) | prostate specific antigen | \bpsa\b | trus | transperineal | (mp)?mri.*prostate | prostate (biopsy|mri) | urology (clinic|referral) | prostatectomy | turp | malignan | neoplasm | carcinoma | metasta | oncolog`
  - {organ}-relevant highlight: `prostate | \bpsa\b | urin | urolog | nocturia | hesitanc | luts | biopsy | \bmri\b | catheter | malignan | neoplasm | metasta | cancer | oncolog`

- **Bladder**: organ = `bladder`; **look-alike** = *recurrent UTI / haematuria from BPH or stones (often older smokers)*.
  - Workup: `suspected (bladder|urolog\w*|urinary tract) cancer | fast[- ]track | two week wait | 2ww | haematuria clinic | refer\w*.*(bladder|urolog|haematuria|oncolog|cancer) | cystoscop | ct urogram | (bladder|renal) ultrasound | turbt | transurethral resection of bladder | bladder biopsy | malignan | neoplasm | carcinoma | metasta | oncolog`
  - {organ}-relevant highlight: `bladder | h[ae]maturia | blood in urine | cystoscop | urin | urolog | urogram | turbt | catheter | \buti\b | smok | malignan | neoplasm | metasta | cancer | oncolog`

Always keep the shared exclusions from §D (`not wanted`/`declined`, and `screening` terms not for this organ).

**E. Build the HTML** — one self-contained white page, `<!doctype html><html><head><meta charset="utf-8">…`
(charset required or `↓ ↑ ×` render as mojibake). Design tokens:
`--miss:#b0413e` (red / FN), `--new:#2e6e8e` (blue / first-ever), `--muted` grey (recurring),
`--med:#3f7a52` (green / medication), `--relevant:#a8560f` (orange / organ-relevant), `--workup:#6a4c93`
(purple / cancer-workup line). Per patient a **card** with:
- header: `PID · age · sex · anchor · patient_guid` (guid small/monospace/selectable) + a **badge**;
- a **light-red "Why …" line** — the SHAP story (see each prompt);
- a **purple "Cancer workup / {organ} investigations" line** — the workup codes found (or a grey "none" line);
- the **code columns** (2 for FN, 3 for FP), each a scrollable chipbox; each chip shows the term + a
  `×count`, coloured new/recurring/med/lung, **problem-list diagnoses in bold**.

**F. Hover tooltips on everything** — one instant styled mechanism (a `data-tip="…"` attribute + a tiny
inline script + a fixed `#tip` box that follows the cursor with no delay; `cursor:help` on `[data-tip]`).
Plain clinical English, no ML jargon. Put a tooltip on: every code chip (type + when + new/recurring + any
problem flag), the column headers, the legend/key chips, the badge, and the "Why …" line. Decode
`<category>_<family>` feature names in plain English (`COPD_distinct_ratio` → "how varied the COPD coding
is"; `_decay_intensity` → "recent … activity"; `_recency_rank` → "how recently …"; `_count` → "how often
…"; `age_at_prediction` → "the patient's age"; …).

**G. Outputs** (all local):
- `<name>.html` — the report,
- `<name>.pdf` — `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=<name>.pdf "file://<name>.html"`,
- `<name>.xlsx` — one row per (patient, code): `cohort, patient, patient_guid, age, sex, anchor_date,
  window, code_term, code_type, organ_relevant, count, problem_list_dx, first_seen, status`.

**Do not publish or upload** — these are patient-identifiable. Share the files internally only.

> **The prompts are self-contained** — they carry the SQL, the `WORKUP` detector and the full build spec, so
> you don't need any script to use them. A byte-exact **lung** reference implementation (`contrast_lib.py` with
> the `WORKUP`/`LUNG` detectors + hover mechanism + Excel rows, `build_fn.py`, `build_fp.py`) is preserved
> **with the lung run** at `…/10k_Cat/results_final/10k7k_wc_ensemble/report_src/` — for byte-exact lung
> reproduction only. It is **lung-hardcoded**; other cancers use the prompts + Cancer config (§D2), not that code.
