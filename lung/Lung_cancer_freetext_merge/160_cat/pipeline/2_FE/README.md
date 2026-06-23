# 2_FE — data-driven FE (per-code / per-category) + stability selection

Feature engineering over the per-`(horizon, lookback)` codelist. Every codelist code gets the generic
**per-code** families below (in CATEGORIZED mode the grouping key is the hand-assigned **category**
instead of the individual SNOMED). There is **no** curated clinical-concept tier. Reads the cohort SQL
from the shared single source.

```
V3/
├── 1_Top_Snomed/
│   └── build_codelist.py       # builds the codelist: top-500 data-driven codes
├── 2_FE/
│   ├── build_features.py            # codelist -> feature matrix (pandas; categorized + per-code)
│   ├── build_features_polars.py     # Polars backend for the six per-code families (per-SNOMED only)
│   ├── fe_parity_check.py           # asserts Polars == pandas (the six per-code families) before Polars is trusted
│   ├── fe_keep_parity.py            # PARITY GATE: keep-features build == full build, byte-identical (hard-fail)
│   ├── stability_select.py          # k-fold importance -> the codes that actually drive predictions
│   ├── codelist/   {Astra|Nova}_{yr}yr_OFF_codelist.csv   (Code,Name,Value + selection scores; per horizon×lookback)
│   └── output/{Astra|Nova}/{yr}yr_OFF/  ...generated...
├── 0_SQL/             12mo_1to1.sql, 1mo_1to1.sql            (shared cohort SQL)
└── 3_Modeling/        lung_training.py, lung_metrics.py, predict_unseen.py
```

## The codelist (built in `1_Top_Snomed/build_codelist.py`, per horizon × lookback)
- **data-driven** — codes passing 3 gates: **Bonferroni p<0.05 AND OR≥2 AND prevalence≥1%**
  of cancer patients, ranked by `combined_rank` and cut to the top **TOP_N=500**. Re-discovered for
  **each lookback** on that lookback's own train data.

## Run order (VM with BigQuery)
```bash
# end-to-end: both horizons × all FE_WINDOWS, split -> codelist -> FE -> stability -> train
python ../run_v3.py

# or step-by-step for one (horizon, lookback):
cd 2_FE
python build_features.py 12mo        # codelist + cohort -> output/{Astra|Nova}/{yr}yr_OFF/features_p005_{h}.parquet
python stability_select.py 12mo      # -> features_p005_{h}_stable.parquet (the driving codes, split stamped)
```
The per-SNOMED engine can run in **Polars** (`FE_ENGINE=polars`), verified column-for-column against
pandas by `fe_parity_check.py`. **CATEGORIZED mode (this variant) is pandas-only** — `run_v3.py` forces
`FE_ENGINE=pandas`. `run_v3.py` trains with `select_features(method="all")` — stability selection
already picked the codes.

## Step 1 — `build_features.py`
`build()` loads the full event stream and restricts to the codelist `keep`. In CATEGORIZED mode it
re-keys events to their hand-assigned category; every code (or category) then flows through the generic
per-code families plus the category-structure families.

### A) Generic PER-CODE FE (every code / category)
- **occurrence** (every code): `_count _present _recency_months _decay_intensity _accel
  _recent_ratio _freq_per_year _timespan_years _interval_median _interval_min _interval_max
  _freq_trend_slope _first_half_freq _second_half_freq _is_worsening`
- **flags** (every code): `_has_active _has_significant`
- **age** (every code): `_age_first _age_last _age_median`
- **value/trend** (**every code carrying any numeric value** — determined train-only): `_val_first
  _val_latest _val_mean _val_median _val_min _val_max _val_std _val_range _val_abs_change
  _val_pct_change _val_latest_z _val_trend_slope _val_trend_corr _val_accel`
- **bands** — disjoint per-time-band windows: every code → `_count/_present_w{lo}_{hi}`; value codes
  → `_val_mean/_val_latest/_val_slope_w{lo}_{hi}`. `TIME_BANDS=[0–6,6–12,12–24,24–60,60–999]mo`.
- **cumulative** — overlapping last-N: every code → `_count_last{n}`; value codes →
  `_val_mean/_val_latest_last{n}`, `CUMULATIVE_WINDOWS=[6,12,24,60]mo`.

> **STRICT category-only scope (CATEGORIZED=True, this variant):** of the once-per-patient blocks below
> ONLY **demographics** are kept (`age_at_prediction` + age-bands, `g_is_male`, `g_eth_*`). The
> utilisation-**volume** part of `global`, plus `comment`, `derangement`, and `blood_ratios`, are **OFF**
> (they use non-map codes / are a care-seeking/skew proxy). `AdaBoost` is excluded from the model panel
> (`config.EXCLUDE_MODELS`) so SHAP stays tree-exact/fast. Descriptions below are the per-SNOMED behaviour.

### B) Shared once-per-patient blocks (computed once, on the FULL stream)
- **global** — event/code volume, consult cadence, problem burden + demographics `age_at_prediction g_is_male g_eth_*`.
- **comment** — leak-safe prodromal-keyword presence/volume from problem-list free text.
- **derangement** — GENERIC cross-code burden / escalation (no hardcoded codes): activity-rate
  trajectory, extreme labs vs own baseline, # rising/worsening/accelerating codes.
- **blood_ratios** — NLR / PLR / LMR / CRP-albumin from each analyte's latest value + ratio trend
  slopes (the only lightly-hardcoded block: the 4 ratio analytes; for the lab-derangement family the
  strict ≥`MIN_VALUE_FRAC` value-bearing set is used).

### C) CATEGORIZED-only category-structure families
Within-category diversity (`_n_distinct_codes _distinct_ratio _count_share _first_months _n_extreme
max_abs_z`), patient breadth `g_n_categories`, cross-category co-occurrence / ordering over the top-K
categories (`cooc_ coocn_ seq_`), category×age interactions (`INT_*`), and inter-category temporal
structure (recency rank, months-since-last, `g_n_categories_{6,12}mo`).

`FEATURE_FAMILIES` toggles (per-SNOMED defaults): `occurrence flags age value bands cumulative global
comment derangement blood_ratios` = **True**; CATEGORIZED turns off `comment/derangement/blood_ratios`
+ runs `global` demographics-only + adds the category-structure families.

**Windows start at 0 and are the same for every horizon.** `ANCHOR_MODE="patient_last"` (default)
counts back from each patient's most-recent event (gap-agnostic, inference-safe). No trends truncated
(`TREND_MAX_MONTHS=None`).

Value features are emitted for **every code carrying any numeric value** (a never-numeric code like
"Cough" simply emits none — not a real omission). Fill: count/present/decay/accel/recent_ratio/flags →
0 when absent (genuine 0); everything else → NaN (impute downstream, never fake-0).

**This is wide.** Output is **Parquet** (a dense CSV this wide would not fit in memory). The model
never trains on this raw matrix — `stability_select.py` reduces it first. **Leak-safe & split-first:**
stability selection is fit on the **TRAIN** patients of the one canonical split (see `../splits.py` /
`../make_split.py`), so the internal test never informs the features.

## Step 2 — `stability_select.py`  (which codes drive predictions)
Over `N_FOLDS` CV folds (train patients only) fit a tree model, take each fold's **cumimp99** set, and
keep features selected in **≥ MIN_FOLDS folds** (default 3/5). It also **stamps the canonical `split`
column** onto the reduced matrix so the split travels with the data into training. Outputs:
- `stable_features_{h}.csv` — `feature, folds_selected, mean_importance`
- `features_p005_{h}_stable.parquet` — the reduced matrix (split + stable features + cancer_class)

Knobs (from `config.py`): `N_FOLDS=5`, `MIN_FOLDS=3`, `CUM_IMP=0.99`.

## Step 3 — `fe_keep_parity.py`  (correctness gate, hard-fail)
Proves the **keep-features** build used by held-out / inference is **byte-identical** to the full build
for every model feature (`max abs diff < 1e-9`), on a patient sample — so a cross-category / restricted
family can't silently diverge at serve. Non-mutating (loads persisted TRAIN artifacts; `fit_split=False`).
`run_v3.py` runs it automatically after each train build (`FE_PARITY_GATE`, default on) and **aborts** on
mismatch. (Distinct from `fe_parity_check.py`, which checks the Polars↔pandas engine equivalence.)

## Design summary
- **codes** — top-500 data-driven, per horizon × lookback
- **FE** — generic per-code for every code (+ category-structure in CATEGORIZED); Polars parity-checked
  for the per-SNOMED engine, pandas for CATEGORIZED
- **selection** — k-fold stability selection on the TRAIN split; split stamped onto the matrix
- **purpose** — focused, bounded data-driven discovery
