# Cancer Prediction — Held-out Evaluation & Calibration (2026-06-11)

Held-out evaluation results, calibration, and the full pipeline code for four cancers
(**bladder, breast, lung, prostate**), plus the metaclassifier deliverables for the ensemble owner.

**Start here:** [`CALIBRATION_NOTES.md`](CALIBRATION_NOTES.md) — end-to-end approach, per-cancer
results for both horizons (12-month and 1-month) across all four lookback windows, the calibration
method and why, and the deployment plan.

---

## Layout

```
Model_Results_2026-06-11/
├── README.md                  ← this file
├── CALIBRATION_NOTES.md       ← main results + methodology write-up
├── .gitignore
├── <cancer>/                  bladder | breast | lung | prostate
│   ├── results/
│   │   ├── 12mo_1to1/         12-month horizon (1:1 model)
│   │   │   ├── 5yr/  10yr/  20yr/  lifetime/      ← lookback window
│   │   │   │     ├── metrics.csv            internal-test metrics (all operating points)
│   │   │   │     ├── heldout_metrics.csv    held-out 500/50k metrics @ free/Youden threshold
│   │   │   │     ├── platt_calib_<w>.json   Platt calibration params + ECE (raw→calibrated)
│   │   │   │     ├── platt_calib_<w>.joblib fitted Platt calibrator
│   │   │   │     └── reliability_<w>.png    reliability diagram
│   │   │   └── …
│   │   └── 1mo_1to1/          1-month horizon (same window layout)
│   └── pipeline_code/         code that produced the results (see below)
└── metaclassifier/            deliverables for the ensemble owner (lung b12_v3)
    ├── option_a_blending/     model predicts valid + test  → 2 parquets
    ├── option_b_oof_stack/    k-fold OOF train + test      → 2 parquets
    └── shared/                fold map + test patient ids (do not regenerate)
```

## Naming conventions

| Token | Meaning |
|---|---|
| `12mo` / `1mo` | **prediction horizon** — features must end ≥ 12 (or 1) months before diagnosis |
| `1to1` | **balanced train set** — cancer:non-cancer = 1:1 |
| `5yr / 10yr / 20yr / lifetime` | **lookback window** — how far back before the anchor features are drawn |

## Models are not in the repo

Trained model binaries (`model_*.joblib`, ≈580 MB; one >100 MB) are **git-ignored** — they're
large and fully regeneratable from `pipeline_code/`. They live on the project VMs / GCS. Everything
needed to *read* the results (metrics, calibration, plots) and to *reproduce* the models (code, SQL,
codelists) is in the repo. The small fitted Platt calibrators (`platt_calib_*.joblib`, ~1 KB) **are** kept.

## `pipeline_code/` — how the results were produced

| File / dir | Role |
|---|---|
| `run_lookback_experiment.py` | training sweep — builds features per window, trains the 8-learner panel, picks best single or top-3 soft-vote ensemble, writes `metrics.csv` + model |
| `evaluate_holdout.py` | scores the touch-once 500/50k held-out set at the threshold fixed on the internal test → `heldout_metrics.csv` |
| `_run_pipeline.py` | BigQuery export helper (cohort SQL → events CSV) |
| `FE/` | feature engineering (`FE/SQL/` holds the cohort + held-out queries) |
| `Modeling/` | model training / prediction / explainability |
| `codelist2.0/` | hand-curated SNOMED/DMD codelists |

**Reproduce a run:** from `<cancer>/pipeline_code/`,
`GAP=12 python run_lookback_experiment.py all` then `GAP=12 python evaluate_holdout.py lifetime`
(`GAP` = 12 or 1, for the 12-month / 1-month horizon).

> **Scope.** This archive contains the **1:1** models' results. Each result window dir is the uniform
> set: `metrics.csv`, `heldout_metrics.csv`, `platt_calib_<w>.{json,joblib}`, `reliability_<w>.png`.
> Trained model binaries and deployment prior-correction artifacts are kept local / on GCS (not pushed).
> (Lung carries a few extra pipeline scripts — it's a separate, more-developed pipeline and is
> intentionally not forced to match the other three.)
