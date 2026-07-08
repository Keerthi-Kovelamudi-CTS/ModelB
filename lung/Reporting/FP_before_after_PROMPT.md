# FP sheet — false alarms: ±1 year around the flag

**Question:** for every patient the model **flagged but who had no cancer** (false positive), was the flag a
**wasted screen**, or did a cancer workup / real active disease follow? Looks at **both the 12 months before
and the 12 months after** the anchor.

**Output:** a local per-patient HTML + PDF + Excel. **Three columns per patient:** *MODEL WINDOW* (what the
model saw) · *12 months BEFORE the anchor* · *12 months AFTER the anchor*. Patient-identifiable →
**local only, never published.**

Read `PER_PATIENT_METHOD.md` first for the shared method. This file gives the FP-specific window, SQL, classification and
wording. It reuses the FN queries but with a **±12-month window** and **before/after counts**.

> **⚙ LUNG is the worked example here. For any other cancer, apply your PER_PATIENT_METHOD.md §D2 cancer config first** —
> organ word, workup detector (§D), "{organ}-relevant" terms, and especially the **look-alike profile** (the
> confounding phenotype differs a lot by cancer). Design is fixed for every cancer; clinical content isn't.

> **Note on interpretation:** the cohort *defines* non-cancer patients as having **no cancer diagnosis ever**
> in the anchor window, so **no FP can "turn out" to be cancer** here — the "after" year vindicates a flag via
> a (negative) **cancer workup** or **other active disease**, not a new cancer. Say this in the report.

---

## Paste this to Claude (swap `<RUN>`)

> **Build a local per-patient FP "±1 year around the flag" sheet for our lung-cancer model. Save HTML + PDF +
> Excel locally — do NOT publish or upload (patient_guids + clinical detail).**
>
> **1. Identify the false alarms.** From `patient_explanations.csv` (**schema A / signed SHAP — see FN §1**)
> take `y_true==0 & prob>=0.50`; map to `patient_guid` via the stable test slice
> (`fe/features_p005_{h}_stable.parquet`); verify stable age == cache age.
>
> **2. Anchor + MODEL WINDOW (left column).** From **your run's** internal raw-events cache ⚙ (run/cancer-specific
> — see FN §2) — anchor from the cache, left column = distinct cache codes + counts. (For FPs `cancer_class==0`
> in the cache — assert it.)
>
> **3. ±1-year pull — BigQuery, window `[anchor − 12mo, anchor + 12mo)`.** Same table joins as the FN prompt
> (§3), but extend the upper bound to `DATE_ADD(a.anchor, INTERVAL 12 MONTH)` and return **before/after
> counts** per code:
> ```sql
> -- anchors + dc CTEs exactly as in FN §3. Carry a.anchor INTO o so the COUNTIFs can compare to it:
> o AS (SELECT co.patient_guid, PARSE_DATE('%Y-%m-%d', co.effective_date) ed, a.anchor,
>        SAFE_CAST(co.code_id AS INT64) code_id, co.source_practice_code, prob.problem_status_description psd
>       FROM `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.CareRecord_Observation` co
>       JOIN anchors a ON a.guid=co.patient_guid
>       LEFT JOIN `prj-cts-ai-dev-sp.EMIS_BULK_DATA_PROCESSED.CareRecord_Problem` prob
>         ON prob.observation_guid=co.observation_guid AND prob.patient_guid=co.patient_guid AND prob.source_practice_code=co.source_practice_code
>       WHERE co.effective_date IS NOT NULL
>         AND PARSE_DATE('%Y-%m-%d', co.effective_date) >= DATE_SUB(a.anchor, INTERVAL 12 MONTH)
>         AND PARSE_DATE('%Y-%m-%d', co.effective_date) <  DATE_ADD(a.anchor, INTERVAL 12 MONTH)
>       QUALIFY ROW_NUMBER() OVER (PARTITION BY co.patient_guid, co.source_practice_code, co.observation_guid ORDER BY co.source_date DESC, SAFE_CAST(co.code_id AS INT64))=1)
> SELECT o.patient_guid, dc.sct, dc.term,
>        COUNTIF(o.ed <  o.anchor) n_before, COUNTIF(o.ed >= o.anchor) n_after,
>        MIN(o.ed) first_ed, ARRAY_TO_STRING(ARRAY_AGG(DISTINCT o.psd IGNORE NULLS LIMIT 3),'; ') psd
> FROM o JOIN dc ON dc.code_id=o.code_id AND dc.source_practice_code=o.source_practice_code
> GROUP BY 1,2,3;
> -- medications: FN §3c but window [anchor-12mo, anchor+12mo); carry a.anchor and return
> --   COUNTIF(ed<anchor) n_before, COUNTIF(ed>=anchor) n_after, MIN(ed) first_ed  GROUP BY patient_guid, drug_term
> -- new (first-ever): FN §3a but upper bound DATE_ADD(a.anchor, INTERVAL 12 MONTH)
> ```
> Merge obs + meds; `is_new` from the first-ever query. A code can appear in **both** the before and after
> columns (place it wherever its count > 0, with that side's count).
>
> **4. Classify each patient** (badge), using all ±1yr codes: **Cancer workup — negative** if any term matches
> the workup detector (PER_PATIENT_METHOD.md §D); else **Other active disease** if there's a new problem-list diagnosis
> (`problem_status_description` ~ "Problem"); else **Genuine false alarm**.
>
> **5. "Why flagged" line (light red — same colour as FN's "why missed").** From the **positive (↑)** SHAP
> drivers:
> - if age is the top ↑ driver → *"The model gave this p% almost entirely on age: being {age} contributed SHAP
>   +0.xx — {N}× the next factor; on top of {plain-English comorbid backdrop, e.g. ex-smoker, COPD,
>   cardiovascular}. This is the over-flagged older-comorbid profile that looks like real lung cancer on
>   structured data."*
> - else → *"The model gave this p% despite no cancer. Pushed up mainly by {top ↑ drivers in plain English} —
>   an older-comorbid picture the model reads as cancer-like."*
>
> **6. Build the HTML** per PER_PATIENT_METHOD.md §E–F, **three columns**:
> - **MODEL WINDOW — what was used** (the 10-yr cache view);
> - **12 MONTHS BEFORE anchor** (red header) — the excluded gap;
> - **12 MONTHS AFTER anchor** (green header) — what happened next.
>
> Per-column "new" = first-ever falls on that side of the anchor (blue); else grey (recurring); meds green;
> lung-relevant orange; diagnoses bold; each chip shows `×count`. Add the purple **"Cancer workup within ±1
> year"** line, tagging each workup code `(before)` / `(after)` (or a grey "no cancer workup within ±1 year"
> line). The intro **lede + How-to** must describe **three columns** (not left/right). Legend chips relabelled:
> **N cancer workup — negative · N other active disease · N genuine false alarm** + new/recurring/med/lung/bold
> keys. Hovers on everything (PER_PATIENT_METHOD.md §F).
>
> **7. Save** `fp_contrast_pm.html`, render `…pdf`, and `fp_before_after.xlsx` (one row per code;
> `window` = "model window" / "12mo BEFORE" / "12mo AFTER"; include `patient_guid`, `count`, `first_seen`,
> `status`). Tell me the paths. Do not publish.

---

## What we found (10k+7k ensemble, internal test, 38 false alarms, ±1yr)

- **5** had a cancer workup within a year that (correctly) found **no cancer** — a defensible flag.
- **29** had **other active disease** recorded around the flag (COPD exacerbations, pneumonia, etc.) — real
  clinical need, not a wasted contact.
- **Only 4** were **genuinely wasted** false alarms (nothing meaningful either side).
- **No FP became a cancer diagnosis** (cohort construction).
- *Why they fire:* almost entirely **age** — for the high-confidence FPs age contributes SHAP ~+0.25
  (≈10–20× the next factor), on an older ex-smoker/COPD backdrop — the "look-alike" profile.

**Defaults / knobs:** threshold 0.50; window ±12 months. (For a *before-only* FP view, use the FN window
`[anchor−12mo, anchor)` instead and drop the third column.)

---

## PIN THESE VERBATIM (so every teammate's report is the same)

Do **not** paraphrase the **design** — copy it exactly (the exact design is written out below).
Items marked **⚙** are **cancer-specific** — substitute per PER_PATIENT_METHOD.md §D2 (shown with the lung example).

**Threshold** 0.50 · **window** `[anchor−12mo, anchor+12mo)` · show **all** codes · a code may appear in both
the before and after column (wherever its side-count > 0).

**Colours (hex):** as FN, plus workup line `#6a4c93`; "why flagged" uses the **same light red as FN**
(`#b0413e`), not blue.

**Per-column newness:** a chip is **new** (blue) if the code's first-ever date falls **on that side** of the
anchor (before-col: first_ed < anchor; after-col: first_ed ≥ anchor); else grey (recurring); meds green.

**Sort within each column:** `obs before med`, then lung-relevant, then column-new, then problem-diagnosis,
then count desc.

**Classification → badge (exact, over ALL ±1yr codes):**
- **“Cancer workup — negative”** (class `b-workup`, purple) — *any* term matches the workup detector (PER_PATIENT_METHOD.md §D).
- else **“Other active disease”** (class `b-disease`, orange) — *any* obs has `problem_status_description` ~ “Problem”.
- else **“Genuine false alarm”** (class `b-silent`, grey).
- Key chips read exactly: **“{n} cancer workup — negative”**, **“{n} other active disease”**, **“{n} genuine false alarm”**.

**Column headers (exact):** **“MODEL WINDOW — what was used”** `{n} codes` · **“12 MONTHS BEFORE anchor”**
(red) `{n} codes · {n} meds · {n} <organ>-relevant`⚙ · **“12 MONTHS AFTER anchor”** (green) same.
**Workup line:** **“Cancer workup / lung investigations in the excluded year:”**… actually for FP use
**“Cancer workup within ±1 year:”** and tag each chip `(before)` or `(after)`; else grey **“No cancer workup
within ±1 year — see the code lists.”**

**“Why flagged” line (light red, one sentence) — pick ONE verbatim:**
1. *age-driven* (top ↑ driver is age): “The model gave this {p}% almost entirely on age: being {age}
   contributed SHAP {+0.xx} — {N}× the next factor; on top of {comorbid backdrop in plain English}. This is
   the over-flagged {look-alike profile}⚙ that looks like real <organ> cancer⚙ on structured data.”
   *(lung look-alike = "older ex-smoker / COPD"; set yours per PER_PATIENT_METHOD.md §D2.)*
2. *else*: “The model gave this {p}% despite no cancer. The score was pushed up mainly by {top ↑ drivers in
   plain English} — an older-comorbid picture the model reads as cancer-like.”
   (`describe()` decoder + colours + chip format: exactly as in the FN prompt's pin section.)

**Cohort caveat (state in the report):** no FP can become a cancer diagnosis (cohort excludes cancer from the
non-cancer class) — the "after" year vindicates via a *negative* workup or other disease, not new cancer.

**Lede / how-to:** must describe **three columns** (model · before · after), not "left/right".

**Outputs (exact names):** `fp_contrast_pm.html`, `fp_contrast_pm.pdf`, `fp_before_after.xlsx`.
**Excel columns (exact order):** `cohort, patient, patient_guid, age, sex, anchor_date, window, code_term,
code_type, lung_relevant, count, problem_list_dx, first_seen, status` (`window` ∈ {`model window`,
`12mo BEFORE`, `12mo AFTER`}).
