# 4_Heldout — touch-once held-out evaluation

Evaluates the trained model on the **labelled held-out cohort** (~500 cancer / 50k non-cancer,
from `EMIS_BULK_DATA_temp.lung_cancer_500` / `no_cancer_lung_50000`) — the deliverable metric.
Evaluated per `(horizon, lookback)`: the model trained at that lookback is scored on held-out FE built
at the SAME lookback, using that lookback's codelist.

## Flow (per horizon × lookback)
```
SQL/heldout_test_{GAP}mo.sql  --(BigQuery)-->  events
  -> build_features.build(h, sql_path=heldout SQL, fit_split=False, years=YEARS)
       same codelist + feature families as training, at the model's lookback; nothing fit on held-out
  -> transform_external (TRAIN-fitted encoders / median-impute / scaler from the saved model)
  -> raw = model["model"].predict_proba(X)        (uncalibrated ensemble's raw scores)

  Honest reporting via a DISJOINT split of the held-out (stratified, fixed seed 42):
  -> CALIB (30%): fit the deployable Platt recalibrator + pick the Youden operating threshold here
  -> TEST  (70%): report AUROC/AUPRC + Sens/Spec/PPV/NPV (+ 95% bootstrap CIs via lung_metrics) and
       Brier/ECE — all on patients NOT used to fit Platt or choose the threshold (so unbiased).
  -> the CALIB-fit Platt is saved as the deployable recalibration artifact.
```

## Run
The model must exist first (`../3_Modeling_outputs/{Astra|Nova}/{yr}yr_OFF/model_{h}.joblib` from `run_v3.py`). Needs BigQuery auth.
```
python evaluate_heldout.py 12mo 5      # horizon [years lookback]; default 12mo + config FE_YEARS_BEFORE
```
(`run_v3.py --heldout` runs this for every horizon × lookback automatically.)
Outputs to `../3_Modeling_outputs/{Astra|Nova}/{yr}yr_OFF/`: `heldout_features_{h}.parquet` (cached), `platt_calib_{h}.joblib`,
`heldout_recalib_{h}.txt`.

## Notes
- **Held-out is never trained on** (excluded from the training cohort SQL at source). It's scored once
  with the already-trained model.
- **Eval-bias fix:** the Platt recalibrator and the operating threshold are fit on the 30% calib slice
  and every metric is reported on the disjoint 70% test slice — so no number is reported on a patient
  used to fit the calibrator or pick the threshold. Brier/ECE for Platt are therefore out-of-sample.
- AUROC/AUPRC + Sens/Spec/PPV/NPV are reported with **95% bootstrap confidence intervals**.
- **Per-age-band operating thresholds** (`operating_threshold_by_age_{h}.json`): a band-specific cut is
  chosen on the calib slice for a target sensitivity (bands with ≥20 calib positives; else the global cut).
  This is the primary no-retrain false-positive lever (the elderly bands generate most FPs) and the
  deployment-correct way to age-condition the *decision* while age stays a model feature.
- **Subgroup / fairness audit** (`subgroup_audit.py` → `subgroup_audit_{h}.csv`, `subgroup_disparities_{h}.csv`):
  Sens/Spec/PPV + Brier/ECE within sex × ethnicity × age-band × density quartile; flags >5-pt sensitivity spread.
- **Robustness:** the cached matrix is `ensure_local`-pulled from GCS before any rebuild, then validated
  (column-presence + FE-source content-hash). Both build paths **hard-fail** (no silent impute) on a
  missing model-feature column, duplicate patients, or a patient-set mismatch. `--force` forces a rebuild.
- **GCS writes** honor the global `GCS_WRITE` switch (`0` = non-mutating run; default publishes).
