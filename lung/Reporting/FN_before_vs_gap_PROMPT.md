# FN sheet — missed cancers: what the model saw vs. the excluded year

**Question:** for every cancer the model **missed** (false negative), what was recorded in the **12 months
before the anchor** — the year the model deliberately excludes? Were these patients already being worked up
for cancer (a timing gap, not a lost patient), or did they truly slip through?

**Output:** a local per-patient HTML + PDF + Excel. **Two columns per patient:** *MODEL WINDOW* (what the
model was given) vs. *the COMPLETE excluded 12 months*. Patient-identifiable → **local only, never published.**

Read `PER_PATIENT_METHOD.md` first for the shared method (identify patients, derive anchor, EMIS queries, workup
detector, hovers, colours, outputs). This file gives the FN-specific window, SQL, classification and wording.

> **⚙ This prompt uses LUNG as the worked example. For any other cancer, first apply your PER_PATIENT_METHOD.md §D2 cancer
> config** — swap the **organ word** (replaces "lung"), the **workup detector** (§D), the **"{organ}-relevant"
> highlight terms**, and the **look-alike profile**. The *design* (layout, colours, sort orders, sentence
> structure, hovers, Excel) is **fixed for every cancer**; only the clinical content changes.

---

## Paste this to Claude (swap `<RUN>`)

> **Build a local per-patient FN "before vs the excluded year" sheet for our lung-cancer model. Save HTML +
> PDF + Excel locally — do NOT publish or upload (it carries patient_guids + clinical detail).**
>
> **1. Identify the missed cancers.** From `<RUN>/modeling/explainability_internal/patient_explanations.csv`
> — **must be schema A** (`row, segment, y_true, prob, factor_1..N, shap_1..N`, i.e. *signed* SHAP; if your run
> only has schema B / unsigned `Top_Factor_N`, re-run explainability for signed SHAP or drop the why-line) —
> sorted by `row`, take `y_true==1 & prob<0.50`. Map each to its `patient_guid` via the stable matrix test
> slice (`<RUN>/fe/features_p005_{h}_stable.parquet`, `{h}` = your horizon e.g. `12mo`; `split=='test'`, same
> row order). **Verify** stable age == cache age for every patient.
>
> **2. Anchor + MODEL WINDOW (left column).** From **your run's** internal raw-events cache ⚙
> (run/cancer-specific; lung's is `…/keerthi/Lung_Cancer/raw_events/Astra_10yr_7k_withtext_internal.parquet`):
> `anchor = event_date + days_before_anchor`
> (constant per patient — assert 0 spread). Left column = the patient's cache events grouped to distinct
> `(snomed_c_t_concept_id, term)` with a count + `event_type` + problem status. Sort lung-relevant first, then
> by count.
>
> **3. EXCLUDED-YEAR pull (right column) — BigQuery, window `[anchor − 12mo, anchor)`.** Get **every**
> observation code and **every** prescription in that window (repeats included), plus the **first-ever** flag.
> Run these (dry-run first; ~0.6 TB each):
>
> **3a. New (first-ever in the window):**
> ```sql
> WITH anchors AS (SELECT guid, anchor FROM UNNEST([STRUCT<guid STRING, anchor DATE> ('{..}', DATE '2019-06-24'), … ])),
> dc AS (SELECT SAFE_CAST(code_id AS INT64) code_id, source_practice_code,
>        PARSE_DATE('%Y%m%d', REGEXP_EXTRACT(file_name, r'/([0-9]{8})/')) fd, term,
>        SAFE_CAST(snomed_c_t_concept_id AS INT64) sct
>        FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.Coding_ClinicalCode`
>        WHERE snomed_c_t_concept_id IS NOT NULL AND term IS NOT NULL
>        QUALIFY ROW_NUMBER() OVER (PARTITION BY code_id, source_practice_code ORDER BY fd DESC)=1),
> o AS (SELECT co.patient_guid, PARSE_DATE('%Y-%m-%d', co.effective_date) ed,
>        SAFE_CAST(co.code_id AS INT64) code_id, co.source_practice_code, prob.problem_status_description psd
>       FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.CareRecord_Observation` co
>       JOIN anchors a ON a.guid=co.patient_guid
>       LEFT JOIN `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.CareRecord_Problem` prob
>         ON prob.observation_guid=co.observation_guid AND prob.patient_guid=co.patient_guid AND prob.source_practice_code=co.source_practice_code
>       WHERE co.effective_date IS NOT NULL
>       QUALIFY ROW_NUMBER() OVER (PARTITION BY co.patient_guid, co.source_practice_code, co.observation_guid ORDER BY co.source_date DESC, SAFE_CAST(co.code_id AS INT64))=1),
> firstseen AS (SELECT patient_guid, sct, ANY_VALUE(term) term, MIN(ed) first_ed,
>        ARRAY_TO_STRING(ARRAY_AGG(DISTINCT psd IGNORE NULLS LIMIT 3),'; ') psd
>        FROM o JOIN dc USING(code_id, source_practice_code) GROUP BY patient_guid, sct)
> SELECT f.patient_guid, f.sct, f.term, f.first_ed, f.psd
> FROM firstseen f JOIN anchors a ON a.guid=f.patient_guid
> WHERE f.first_ed >= DATE_SUB(a.anchor, INTERVAL 12 MONTH) AND f.first_ed < a.anchor;
> ```
> **3b. ALL observations in the gap** — same `dc`/`o` CTEs but add the window to the `o` `WHERE`
> (`ed >= DATE_SUB(a.anchor,INTERVAL 12 MONTH) AND ed < a.anchor`), then
> `SELECT patient_guid, sct, term, COUNT(*) n_gap, MIN(ed) first_in_gap, ARRAY_TO_STRING(ARRAY_AGG(DISTINCT psd …),'; ') psd GROUP BY 1,2,3`.
> **3c. ALL medications in the gap:** `Prescribing_DrugRecord dr` (dedupe `drug_record_guid`, latest file) →
> `Prescribing_IssueRecord ir` (dedupe `issue_record_guid`, parse `effective_date`) →
> `Coding_DrugCode ce` (`dmd_product_code_id IS NOT NULL`, latest per `code_id`+practice) for `drug_term`;
> filter `ir.ed ∈ [anchor−12mo, anchor)`, `GROUP BY patient_guid, drug_term → n_gap, MIN(ed)`.
>
> Merge 3b (obs) + 3c (meds, `kind='med'`), mark `is_new` = `(patient, sct) ∈` 3a, **dedupe by display term**
> so nothing shows twice.
>
> **4. Classify each patient** (badge): **On cancer pathway in gap** if any gap term matches the workup
> detector (see PER_PATIENT_METHOD.md §D); else **New diagnoses in gap** if it has a new problem-list diagnosis
> (`problem_status_description` contains "Problem", not a workup term); else **Record went silent**.
>
> **5. "Why missed" line (light red).** From the patient's SHAP factors, take the negative (↓) drivers:
> - if the record is very thin (≤ ~10 coded items) → *"The record held only N coded items in the model's
>   10-year window — too little to find cancer signal, so it defaulted to a low score (p%), driven mainly by
>   the patient being {age}."*
> - else if age (`age_at_prediction`/`ageband_u50`) is the top ↓ driver → *"The model gave this p%. By far the
>   biggest downward pull was the patient's age ({age}, SHAP −0.xx) — it outweighed everything else. Next
>   strongest were {plain-English next drivers}."*
> - else → *"The model gave this p%. The score was held down most by {top ↓ driver} (SHAP −0.xx). … No single
>   lung red-flag was strong enough to lift it over the line."*
>
> **6. Build the HTML** per PER_PATIENT_METHOD.md §E–F: two columns — **MODEL WINDOW — before the gap** (left) and
> **EXCLUDED 12 MONTHS — the full gap** (right, `[anchor−12mo, anchor)`; blue = first-ever, grey = recurring,
> green = medication, {organ}-relevant⚙ orange, diagnoses bold). Add the purple **"Cancer workup / {organ}
> investigations in the excluded year"**⚙ line (the matched workup codes, or a grey "no coded cancer workup"
> line whose hover notes the cancer was likely diagnosed via a route not captured here — e.g. an emergency
> admission or secondary-care workup coded only at diagnosis). Intro **How-to** panel + a **legend** with a
> hover on every chip. Hovers on everything (PER_PATIENT_METHOD.md §F).
>
> **7. Save** `fn_contrast_internal.html`, render `…pdf` (headless Chrome), and `fn_before_vs_gap.xlsx`
> (one row per code; `window` = "model window" / "excluded 12mo (gap)"; include `patient_guid`,
> `gap_status` = new/recurring/medication). Tell me the paths. Do not publish.

---

## What we found (10k+7k ensemble, internal test, 25 missed cancers)

- **24 / 25** had new coded activity in the excluded year (only 1 record was truly empty).
- **19 / 25 were already on a cancer/lung workup** in that year (fast-track referrals, chest X-rays,
  respiratory/oncology clinics, bronchoscopy, a lung nodule, a lobectomy).
- **3** had other new diagnoses; **3** were truly silent (young, thin records; one an A&E fracture).
- *Read-out:* ~three-quarters of "misses" were being correctly worked up inside the year the model holds out —
  a **timing gap, not lost patients**. Dominant miss driver: **age** (age-suppressed).

**Defaults / knobs:** threshold 0.50; window 12 months; left column = the 10-yr cache view. Tell Claude to
change the threshold or window if needed.

---

## PIN THESE VERBATIM (so every teammate's report is the same)

Do **not** paraphrase the **design** — copy it exactly (this is what makes reports consistent across the team;
the exact design is written out below). Items marked **⚙** are **cancer-specific** — substitute
per PER_PATIENT_METHOD.md §D2 (here shown with the lung example). Everything else is identical for every cancer.

**Threshold** 0.50 · **window** `[anchor−12mo, anchor)` · show **all** codes (no top-N truncation) · **dedupe
the right column by display term** (one chip per term).

**Colours (hex):** miss `#b0413e`, new `#2e6e8e`, recurring = muted grey, medication `#3f7a52`, lung
`#a8560f`, workup line `#6a4c93`.

**Sort orders (exact):**
- LEFT (model window): lung-relevant first, then count desc.
- RIGHT (gap): `obs before med`, then lung-relevant, then new-before-recurring, then problem-diagnosis, then count desc.

**Classification → badge (exact predicate, evaluated over ALL gap codes new+recurring):**
- **pathway** → badge **“On cancer pathway in gap”** — *any* gap term matches the workup detector (PER_PATIENT_METHOD.md §D).
- else **other** → badge **“New diagnoses in gap”** — *any NEW obs code* has `problem_status_description`
  containing “Problem” (and isn’t a workup term).
- else **silent** → badge **“Record went silent”**.
- Key chips read exactly: **“{n} on cancer pathway”**, **“{n} other new dx”**, **“{n} silent”**.

**Column headers (exact):** left = **“MODEL WINDOW — before the gap”** with `{n} codes · {n} <organ>-relevant`⚙;
right = **“EXCLUDED 12 MONTHS — the full gap”** with `{n} codes · {n} new · {n} meds · {n} <organ>-relevant`⚙.
**Workup line ⚙:** **“Cancer workup / <organ> investigations in the excluded year:”** + the matched workup terms
as chips; or the grey **“No coded cancer workup in this window — see the full code list below.”** (“<organ>” =
your cancer, e.g. "lung investigations".)

**Chip:** show the term; append `<i>{count}</i>` only when count > 1; classes `chip` + `new|rec|med` +
`lr`(lung) + `dx`(problem diagnosis, bold).

**“Why missed” line (light red, one sentence) — pick ONE template verbatim (fill placeholders):**
1. *data-gap* (≤10 coded items, or prob<3% & <20 items): “The record held only {N} coded items in the
   model's 10-year window — too little for the model to find cancer signal, so it defaulted to a low score
   ({p}%), driven mainly by the patient being {age}.”
2. *age-suppressed* (top ↓ driver is age): “The model gave this {p}%. By far the biggest downward pull was the
   patient's age ({age}, SHAP {−0.xx}) — it outweighed everything else. Next strongest were {next 2 ↓ drivers
   in plain English}.{ + the ↑ clause if any}”
3. *signal-poor* (else): “The model gave this {p}%. The score was held down most by {top ↓ driver in plain
   English} (SHAP {−0.xx}). … No single <organ> red-flag⚙ was strong enough to lift it over the line.”

**Plain-English feature decoder (`describe`, verbatim):** `age_at_prediction`→“age”; `ageband_u50`→“being under
50”; `g_eth_*`→“ethnicity”; else strip the family suffix and map: `_decay_intensity`→“recent {cat} activity”,
`_distinct_ratio`→“how varied the {cat} coding is”, `_recency_rank`/`_recency_months`→“how recently {cat} was
recorded”, `_count_last24`/`_count_last60`→“recent {cat} entries”, `_count`→“how often {cat} was recorded”,
`_present`→“{cat} on record”, `_val_min/max/mean/latest/std/range`→“lowest/highest/average/latest/spread of/range
of {cat} value(s)”, `_max_abs_z`→“how unusual the {cat} values are”, `_first_months`→“how early {cat} appears”,
`_timespan_years`→“how long {cat} spans”, `_interval*`→“gaps between {cat} records”, `_accel`/`_freq*`→“trend in
{cat}”, `_recent_ratio`→“how recent the {cat} is”, `_abs_change`/`_pct_change`→“change in {cat}”.

**Outputs (exact names):** `fn_contrast_internal.html`, `fn_contrast_internal.pdf`,
`fn_before_vs_gap.xlsx`. **Excel columns (exact order):** `cohort, patient, patient_guid, age, sex,
anchor_date, window, code_term, snomed, code_type, lung_relevant, count_in_window, problem_list_dx,
first_seen_date, gap_status` (`window` ∈ {`model window (before gap)`, `excluded 12mo (gap)`};
`gap_status` ∈ {`new (first-ever)`, `recurring`, `medication`}).
