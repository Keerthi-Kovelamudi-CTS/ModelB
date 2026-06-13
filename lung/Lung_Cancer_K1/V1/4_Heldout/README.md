# 4_Heldout — held-out evaluation + Platt recalibration

Touch-once evaluation of a trained lookback model on the real **500 cancer / 50,000 non-cancer**
held-out set (0% training overlap), plus **probability calibration via Platt scaling** — the
documented approach (see `Model_Results_.../CALIBRATION_NOTES.md`).

```
4_Heldout/
├── evaluate_heldout.py        # export held-out -> FE -> score -> Platt calib -> metrics
└── SQL/  heldout_test_12mo.sql, heldout_test_1mo.sql   (held-out cohort, per horizon)
```

## What it does
1. Export the held-out cohort from BigQuery (`SQL/heldout_test_{GAP}mo.sql`).
2. Build the FE matrix — **same FE as training**, applying the **TRAIN xpoll reference** (leakage-safe),
   counts→0 / values→NaN.
3. Score with the trained model via `predict_unseen.CancerPredictor` (TRAIN-fitted encoders / medians /
   scaler / feature-selection) → **RAW** best-model probabilities.
4. Stratified **30% calib / 70% test** split (fixed seed).
5. Fit **Platt (sigmoid)** + isotonic (cross-check) on the calib slice, **to the held-out's own
   prevalence**.
6. Report on the disjoint test slice: **AUROC / AUPRC** (threshold-free), **Brier + ECE** (raw / Platt /
   isotonic), and **Sens/Spec/PPV/NPV at free-Youden**. Save `platt_calib_{window}.joblib`,
   `heldout_recalib_{window}.txt`, `heldout_reliability_{window}.png`.

**Why Platt (not isotonic):** equally accurate here but a smooth 1-parameter fit — robust at our small
positive counts, no step artefacts. Monotonic → AUROC/Sens/Spec unchanged; only the probability scale.

> ⚠️ **King-Zeng true-prevalence offset is NOT applied here** — that's a separate *future deployment*
> step (`deploy_calibration.py`). We are calibrating to the held-out's own prevalence for now.

## Run (on the VM, after the matching lookback model exists)
Unlike training, this runs **ONE window per invocation** (no loop) — set `GAP` + `WINDOW` to the
model you want to evaluate, and run it once per horizon/window you care about:
```bash
cd Lung_Cancer_K1/V1
GAP=12 NC_RATIO=1 WINDOW=5yr python 4_Heldout/evaluate_heldout.py   # 12mo, 5yr, 1:1 model
GAP=1  NC_RATIO=1 WINDOW=5yr python 4_Heldout/evaluate_heldout.py   # 1mo,  5yr, 1:1 model
GAP=12 NC_RATIO=5 WINDOW=5yr python 4_Heldout/evaluate_heldout.py   # 12mo, 5yr, 1:5 model
# WINDOW = 5yr | 10yr | 20yr | lifetime   |   NC_RATIO = 1 | 5
```
The held-out cohort itself is fixed (`heldout_test_{GAP}mo.sql`) regardless of `NC_RATIO` — that
only selects which trained model to score. Outputs land in `{GAP}mo_1to{NC_RATIO}/lookback/{WINDOW}/`;
requires `model_{WINDOW}_1to{NC_RATIO}.joblib` + `xpoll_ref_{WINDOW}.json` from the training run.
