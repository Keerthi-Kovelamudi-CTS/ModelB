# 2_FE — tiered FE (clinical priors + focused discovery) + stability selection

**TIERED** feature engineering over the **curated ∪ data-driven** codelist.
Curated clinical codes get **CONCEPT-level** FE; every *other* codelist code gets the
generic **per-code** families. The two code sets are **disjoint** — no code is featurized twice.
Reads the cohort SQL + curated list from the shared single source.

```
V3/
├── 1_Top_Snomed/
│   ├── build_codelist.py       # builds the codelist: curated ∪ top-500 data-driven
│   └── curated_codes.json      # curated clinical-concept codes (the curated tier)
├── 2_FE/
│   ├── build_features.py       # tiered codelist -> feature matrix (drives the partition)
│   ├── concept_features.py     # CONCEPT-level FE (run on the curated codes only)
│   ├── stability_select.py     # k-fold importance -> the codes that actually drive predictions
│   ├── codelist/   lung_codelist_{12mo,1mo}.csv   (curated ∪ top-500, columns Code,Name,Value)
│   └── output/{horizon}/  ...generated...
├── 0_SQL/             12mo_1to1.sql, 1mo_1to1.sql            (shared cohort SQL)
└── 3_Modeling/        lung_training.py, lung_metrics.py, predict_unseen.py
```

## The codelist (built in `1_Top_Snomed/build_codelist.py`)
**curated ∪ top-500 data-driven**:
- **curated** — hand-curated clinical-concept codes (`curated_codes.json`), force-included even
  if sub-threshold.
- **data-driven** — codes passing 3 gates: **Bonferroni p<0.05 AND OR≥2 AND prevalence≥1%** of
  cancer patients, then ranked by `combined_rank` and cut to the top **TOP_N=500**.

This combines clinical priors with focused, bounded data-driven discovery.

## Run order (VM with BigQuery)
```bash
# end-to-end (both horizons): build -> stability-select -> train on the stable matrix
python ../run_v3.py

# or step-by-step:
cd 2_FE
python build_features.py        # 1) codelist + cohort -> output/{h}/features_p005_{h}.parquet
python stability_select.py      # 2) -> features_p005_{h}_stable.csv  (the driving codes)
# then train 3_Modeling on the *_stable.csv matrix
```
`run_v3.py` trains with `select_features(method="all")` — stability selection already picked the
codes, so the model trains on exactly the stable set (no second cumimp pass). Models + plots land
in `V3/output/{horizon}/`.

## Step 1 — `build_features.py`  (the tiered partition)
`build()` loads the full event stream, restricts to the codelist `keep`, then **partitions**:
```
curated_in = keep & CURATED       # curated codes present in this codelist  -> CONCEPT FE
remaining  = keep - CURATED       # everything else                          -> per-code FE
assert not (curated_in & remaining)   # disjoint — no code featurized twice
```

### A) CONCEPT-level FE on the curated codes (`concept_features.py`)
Hand-curated clinical-concept families, computed on `ev_cur` (curated codes only):
- **symptom dynamics** — per symptom category (cough, breathlessness, chest pain/infection,
  haemoptysis, clubbing): recency / decay-intensity / accel / recent-ratio / presence / first-occ /
  span + `RECENT_SYMPTOM_BURDEN_{6,12}MO` + `SYMPTOM_BURDEN`.
- **problem-list flags** — per category `*_HAS_ACTIVE_PROBLEM` / `*_HAS_SIGNIFICANT_PROBLEM` +
  `NUM_ACTIVE_SIGNIFICANT_PROBLEMS` + `SIGNIFICANT_PROBLEM_BURDEN`.
- **clusters** — `RESP_SYMPTOM` / `RESP_COMORBID` co-occurrence counts (multi-system presentation).
- **interaction terms** — `INT_*` multiplicative products (smoking×age, age×haemoptysis,
  packyears×age, clubbing×smoking, …).
- **smoking dose** — `PACK_YEARS_MAX`, `CIGS_PER_DAY_MAX` (leakage-safe risk dose).
- **lab level-stats** — per analyte `*_LATEST/_VMAX/_VMIN/_VMEAN/_MEASURED/_VALUE_ACCEL`.

### B) Generic PER-CODE FE on the remaining (data-driven) codes
The standard families, applied **per code** on `ev_remaining`:
- **occurrence** (every code): `_count _present _recency_months _decay_intensity _accel
  _recent_ratio _freq_per_year _timespan_years _interval_median _interval_min _interval_max
  _freq_trend_slope _first_half_freq _second_half_freq _is_worsening`
- **flags** (every code): `_has_active _has_significant`
- **age** (every code): `_age_first _age_last _age_median`
- **value/trend** (value-bearing codes only, ≥30% of events numeric): `_val_first _val_latest
  _val_mean _val_median _val_min _val_max _val_std _val_range _val_abs_change _val_pct_change
  _val_latest_z _val_trend_slope _val_trend_corr _val_accel`
- **bands** — disjoint per-time-band windows: every code → `_count/_present_w{lo}_{hi}`; value codes
  → `_val_mean/_val_latest/_val_slope_w{lo}_{hi}`. `TIME_BANDS=[0–6,6–18,18–36,36–72,72–999]mo`
  (the final `72–999` is an open-ended catch-all so no event is dropped).
- **cumulative** — overlapping last-N: every code → `_count_last{n}`; value codes →
  `_val_mean/_val_latest_last{n}`, `CUMULATIVE_WINDOWS=[6,12,24,60]mo`.

### C) Shared once-per-patient blocks (computed once, NOT per tier)
Cross-code per-patient aggregates, fed the **FULL** event stream (true patient utilisation):
- **global** — `g_total_events g_distinct_codes g_distinct_{obs,med}_codes g_value_measured
  g_active_problems g_significant_problems g_consult_total g_consult_recency_months
  g_consult_accel g_consult_recent_rate` + **demographics** `g_age g_is_male g_eth_*` (one-hot).
- **comment** — free-text problem-list comment presence/volume + prodromal-keyword counts (leak-safe
  symptoms/risk only) + symptom-group burden + red-flag recency.
- **derangement** — GENERIC cross-code burden / escalation (**no hardcoded codes**): activity-rate
  trajectory (even 6-mo bands), # extreme labs vs own baseline + mean|z|, # rising / worsening /
  accelerating codes, recent distinct-code burden.
- **blood_ratios** — NLR / PLR / LMR / CRP-albumin (mGPS) from each analyte's latest value + ratio
  trend slopes (the only lightly-hardcoded block — the 4 ratio analytes).

`FEATURE_FAMILIES` toggles: `occurrence flags age value bands cumulative global comment derangement
blood_ratios` = **True**; `percentile` = **False** (OFF — redundant with raw
value for trees, and the only cross-patient-fit family).

**Windows start at 0 and are the same for every horizon** — the SQL already applied the gap cutoff, so
band 0 = the start of available data. `ANCHOR_MODE="patient_last"` (default) counts back from each
patient's most-recent event (gap-agnostic, inference-safe). **No trends are truncated** —
lifetime trends + windowed views coexist (`TREND_MAX_MONTHS=None`).

Value features are emitted **only for value-bearing codes** — a non-numeric code (e.g. "Cough") has
no number to summarize. Fill: count/present/decay/accel/recent_ratio/flags → 0 when absent (genuine
0); everything else → NaN (impute downstream, never fake-0). The same 0-vs-NaN rule applies to the
concept-layer columns (UPPERCASE presence/cluster/flag/burden → 0; month/level/pack-year → NaN).

⚠️ **This is wide** (per-code families × hundreds of data-driven codes). Output is **Parquet** (a
dense CSV this wide would not fit in memory). The model never trains on this raw matrix —
`stability_select.py` reduces it first. **Leak-safe:** stability selection is fit on the SAME 80%
train (seed-42 stratified 80/10/10) the model uses, so the model's internal 10% test never informs
the features.

## Step 2 — `stability_select.py`  (which codes drive predictions)
We **don't** hand-pick a cutoff. Over `N_FOLDS` CV folds, fit a tree model, take each fold's
**cumimp99** set, and keep features selected in **≥ MIN_FOLDS folds** (default 3/5). Stable features
survive every split; one-fold flukes are dropped. Outputs:
- `stable_features_{h}.csv` — `feature, folds_selected, mean_importance`
- `features_p005_{h}_stable.csv` — the reduced matrix to train on

Knobs (top of `stability_select.py`): `N_FOLDS=5`, `MIN_FOLDS=3`, `CUM_IMP=0.99`.

## Design summary
- **codes** — curated ∪ top-500 data-driven
- **FE** — tiered: concept-level on the curated codes, generic per-code on the rest (disjoint)
- **selection** — k-fold stability selection on the TRAIN split
- **purpose** — clinical priors + focused, bounded data-driven discovery
