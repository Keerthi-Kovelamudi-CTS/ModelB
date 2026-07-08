# 4_Heldout — touch-once held-out evaluation

Evaluates the trained model on the **labelled held-out cohort** (~500 cancer / 50k non-cancer,
from `EMIS_BULK_DATA_temp.lung_cancer_500` / `no_cancer_lung_50000`) — the deliverable metric.
Runs the tiered FE + stable-feature model on the held-out cohort.

## Flow (per horizon)
```
SQL/heldout_test_{GAP}mo.sql  --(BigQuery)-->  events
  -> build_features.build(h, sql_path=heldout SQL, fit_split=False)
       same codelist + tiered families as training; lifetime FE window;
       NO percentile -> nothing is fit on held-out (fully leak-safe)
  -> transform_external (TRAIN-fitted encoders / median-impute / scaler from the saved model)
  -> raw = model["model"].predict_proba(X)        (uncalibrated ensemble's raw scores)
  -> TEST-ON-FULL: score the FULL held-out with RAW scores and report
       AUROC/AUPRC + Sens/Spec/PPV/NPV @ Youden (calibration-invariant; held-out is ~real
       prevalence so PPV is real-world). This is the test — no calib/test split.
  -> THEN fit Platt (sigmoid) [+ isotonic cross-check] on the FULL held-out as a DEPLOYABLE
       calibrator (a deployment artifact, NOT part of the test). Brier/ECE reported raw (honest)
       vs Platt (in-sample).
```

## Run
The model must exist first (`../output/{h}/model_{h}.joblib` from `run_v3.py`). Needs BigQuery auth.
```
python evaluate_heldout.py 12mo      # or 1mo  (default 12mo)
```
Outputs to `../output/{h}/`: `heldout_features_{h}.parquet` (cached), `platt_calib_{h}.joblib`,
`heldout_recalib_{h}.txt`.

## Notes
- **Held-out is never trained on.** It is scored once with the already-trained model; the test
  metrics are reported on RAW scores (calibration-invariant), then a full-set Platt calibrator is
  fit purely as a deployment artifact.
- The held-out SQL uses the same lifetime FE window (`years_before=100`) as training FE.
