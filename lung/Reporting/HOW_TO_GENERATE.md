# Generate the FP/FN patient deep-dive report (local HTML)

The deep-dive is a **self-contained HTML file plus a PDF**, built by Claude from your run's outputs and
**saved locally — nothing is uploaded or published.** What you share is the `.html` or `.pdf` itself: put it
on internal SharePoint / a shared drive / an internal server, or email it. Patient data stays entirely on
your infrastructure.

```
run your model  ─►  explainability outputs (CSV + stable matrix)  ─►  paste the prompt into Claude  ─►  deepdive.html + deepdive.pdf (local files)
```

Layout example (a preview of what the page looks like): `https://claude.ai/code/artifact/b14ebe22-1513-478d-ab44-4214316e20d1`
— your real output is a **local file**, not a hosted link. (That hosted preview is deliberately **guid-free**; your local
report carries `patient_guid`s and therefore must never be hosted/published.)

---

## Step 1 — produce the inputs (a normal run)

The report is built from the **internal-test** explainability, which a **normal run produces** (the
pipeline's `3_Modeling/explainability.py` writes `explainability_internal/` during training).
`--heldout` is **optional** — it only adds the **held-out** threshold sweep. You end up with:

- **`…/explainability_internal/…patient_explanations*.csv`** — per-patient risk factors (either column layout works).
- **`…/fe/features_p005_{h}_stable.parquet`** — the stable matrix (has `split`, `age_at_prediction`, `g_is_male`, `g_eth_*`, `<cat>_count`, often `patient_guid`).
- *(optional)* held-out per-patient predictions (`prob` + label) → adds the held-out sweep.

**Model types:** the SHAP generator handles tree models exactly (LightGBM, RandomForest, XGBoost,
GradientBoosting, CatBoost, DecisionTree + soft-voting ensembles) and now falls back for non-tree models —
**LinearExplainer** for a linear/logistic baseline (fast, all patients) and **KernelExplainer** for any
other `predict_proba` model (slower, capped).

## Step 2 — paste this prompt into Claude (swap the `<...>`)

> **Build a per-patient FP/FN deep-dive report for our lung-cancer model and SAVE it locally as BOTH an HTML file and a PDF (do NOT publish it as an artifact or upload it anywhere — it must stay on our infrastructure).**
>
> **Data** (under `<RUN_DIR>`):
> - `<RUN_DIR>/…/explainability_internal/` — find the `*patient_explanations*.csv`. It's one of two layouts; detect which:
>   - **A:** `row, segment, y_true, prob, factor_1..N, shap_1..N` → join demographics from the stable matrix below by `split=='test'` row order.
>   - **B:** `Confusion_Label, Cancer_Probability, Actual_Cancer, Age, Gender, Ethnicity, Record_Total_Events, Record_Distinct_Codes, Top_Factor_1..N, Factor_N_Contribution, patient_guid` → demographics are inline; join on-record categories from the stable matrix by `patient_guid`.
> - `<RUN_DIR>/fe/features_p005_{h}_stable.parquet` — use `split=='test'`. "On record" = the patient's top non-admin categories where `<cat>_count > 0`.
> - *(optional)* `<held-out predictions>` (`prob` + label) for a held-out threshold sweep.
>
> **Compute:** recompute each patient's confusion segment at threshold **0.50** (cancer if prob ≥ 0.5). Split each patient's features into **↓ lowered** (SHAP < 0) / **↑ raised** (SHAP > 0). Classify each FN/FP into an archetype: *Data-gap* (≤ 8 categories on record), *Age-suppressed* (FN where `age` is the top ↓ driver), *Signal-poor* (other FN), *Look-alike* (FP).
>
> **Build ONE self-contained, white-background HTML page** — a **complete standalone document** starting with `<!doctype html><html><head><meta charset="utf-8">…` (this is a local file, so the charset is required or the `↓ ↑ · —` symbols render as mojibake), no external assets, no matplotlib images — with:
> 1. **Header:** model name · n · operating point 0.50 · Sens/Spec; headline “The model misses the young and over-flags the old”; and an **age strip** — a dot per FN and per FP placed by age, in two rows with the labels in a left gutter (must not overlap the dots or ticks).
> 2. **Confusion tiles** (TP/FP/FN/TN) + Sens/Spec.
> 3. **Archetype buckets** — counts for the FN reasons and the FP reasons.
> 4. **Legend** — how to read a card (risk %, on record, ↓ lowered, ↑ raised; miss = ↓ outweigh ↑, false alarm = ↑ win).
> 5. **A card for every FN and every FP** (youngest first): `age · sex · ethnicity · events · categories · patient_guid` (put the **`patient_guid`** in the header, small/monospace/selectable, so a clinician can trace the patient back in the source data — join it from the stable matrix / layout-B column), a one-sentence plain-English narrative of why it went wrong (tailored to its archetype), then chip rows for **on record**, **↓ lowered**, **↑ raised** (show up to ~12 factors each). *(The guid makes this file patient-identifiable — another reason it stays local and is never published/hosted.)*
> 6. **Aggregate SHAP** — one small table per segment (TP/FP/TN/FN): top ~20 features by mean |SHAP| with direction.
> 7. **Threshold trade** — an **internal** sweep table (Sens/Spec/PPV/Flagged/Missed/False-alarms across thresholds 0.2–0.8, from the internal probs) and, if held-out preds are given, a **held-out** sweep. Highlight the operating-point row.
> 8. A dark **“The ceiling — what is genuinely hard, and what could move it”** section: three floor cards (Young & undocumented, Signal-poor, Clinical look-alikes) + an honest-ceiling paragraph (age inflates the headline AUROC; within-age discrimination is the honest number; single-digit PPV at low prevalence is a discrimination ceiling; real gains need a new signal — imaging — or a higher-prevalence, high-risk-only screen).
> 9. **Footer:** model long name + internal AUROC + confusion counts; “risk % = calibrated probability; chips = the model’s top SHAP contributors.”
>
> **Feature naming:** show the **full feature name exactly as in the data** (e.g. `COPD_distinct_ratio`, `Blood test - neutrophils_val_max`, `age_at_prediction`) in the ↓/↑ chips and the aggregate tables — do **not** shorten to the category. (The **on-record** chips are the exception — those are the patient's clinical *categories*.)
> **Hover help — put a plain-language tooltip on EVERYTHING a reader could wonder about, not just feature chips.** Use one **instant styled tooltip** mechanism throughout (not the native `title`): a `data-tip="…"` attribute on the element + one tiny inline script + a fixed-position `#tip` box that follows the cursor with **no delay**, and `cursor:help` on every hoverable element. Every tooltip is written in **plain clinical English — no ML jargon** (never say "z-score", "tenure", "time-decayed", "distinct/total", "SHAP value" without explaining it). Specifically, give a hover to **all of these**:
> - **Every feature name** (in the ↓/↑ chips *and* the aggregate tables) — decode the `<category>_<family>` name. Examples: `COPD_distinct_ratio` → “How varied the COPD coding is”; `<cat>_decay_intensity` → “How much recent <cat> activity there is (recent events count more)”; `<cat>_recency_rank` → “How recently <cat> was recorded vs the rest of the record”; `<cat>_count` → “How often <cat> was recorded”; `<cat>_present` → “Whether <cat> is on the record at all”; `Blood test - neutrophils_val_max` → “Highest neutrophils value recorded”; `<cat>_max_abs_z` → “How far the most extreme <cat> value is from this patient’s usual level”; `age_at_prediction` → “The patient’s age at the prediction date”; `ageband_u50` → “Whether the patient is under 50”; `g_eth_*` → “The patient’s ethnicity”.
> - **Every legend / key chip** (risk %, on record, ↓ lowered, ↑ raised, colour keys, any “bold = …” note) — spell out exactly what it means in a sentence.
> - **The confusion tiles** (TP/FP/FN/TN) — e.g. FN → “A real cancer the model scored below the alert line — a miss.”
> - **The metrics** (Sensitivity, Specificity, PPV, NPV, prevalence, AUROC) — one-line plain definitions (e.g. Sensitivity → “Of the real cancers, the share the model caught.”).
> - **The archetype buckets** — what qualifies a patient for that bucket.
> - **The threshold-table column headers** (Flagged / Missed / False-alarms / PPV / the operating-point row) — what each column counts.
> - **Section headers / on-record category chips / the ↓ lowered & ↑ raised concepts** — a sentence each.
> - **The `patient_guid`** — e.g. “Patient GUID — for tracing this patient back in the source data (local only).”
> Keep every description short (one sentence), concrete, and readable by a clinician who has never seen the model. A reader should be able to hover *any* label, number, or chip on the page and get told what it means.
>
> Design: clean clinical palette (red = miss, blue = false alarm), tabular numerals, paragraphs span the full column width. Include **print CSS** so the PDF keeps colours and paginates cleanly: `html{print-color-adjust:exact;-webkit-print-color-adjust:exact}`, `@page{margin:12mm}`, and `break-inside:avoid` on `.card/.ac/.tile/section`.
> **Save the finished HTML to `<RUN_DIR>/deepdive.html`, then render `<RUN_DIR>/deepdive.pdf` from it** (headless Chrome, e.g. `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="<RUN_DIR>/deepdive.pdf" "file://<RUN_DIR>/deepdive.html"`). Do **not** publish or upload either — just save the local files and tell me the paths.

**Tweak counts:** the defaults are top-20 per aggregate table and up to 12 factors per card — just tell
Claude *"use top-N …"* to change either (and `--threshold` for a different operating point).

---

## Notes

- **Nothing leaves your systems** — the outputs are a local `deepdive.html` + `deepdive.pdf`; open/host
  them internally or email the file. **No claude.ai publishing** — the report shows per-patient clinical
  detail **and `patient_guid`s**, so it is patient-identifiable and must stay on your infrastructure. (PDF is
  rendered from the HTML with headless Chrome — swap in `wkhtmltopdf`/`weasyprint` if that's what you have.)
- Everything (counts, Sens/Spec, archetypes, the age strip, the internal sweep) is **derived from the
  data** — you only supply the model’s display name + internal AUROC.
- **Held-out sweep** needs held-out per-patient predictions; without them, the page still builds with the
  internal sweep only.
- To share with the team: this `Reporting/` folder (in `CtheSigns/AI-UK`). Each teammate produces the
  inputs (Step 1) and pastes the prompt (Step 2) — Claude saves their `deepdive.html`, which they share
  internally.
