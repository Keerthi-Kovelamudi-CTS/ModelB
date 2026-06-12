# Metaclassifier deliverables (lung b12_v3)

Predictions for the ensemble owner, on the shared **b12_v3** split. The prediction parquets are
**patient-level (braced patient GUIDs)** and are therefore **git-ignored** (PHI) — they are delivered
via **GCS**, not committed here. This folder documents what they are and where to find them.

**GCS:** `gs://gcs-ai-dev-model-artifacts/yash/lung_model_artifacts_b12_v3_meta_classifier/`

| Folder | What | Files |
|---|---|---|
| `option_a_blending/` | **Option A (blending)** — one model trained on b12_v3 TRAIN, predicts VALID + TEST. | `xgboost_preds_valid.parquet`, `xgboost_preds_test.parquet` (GCS: `xgb/`) |
| `option_b_oof_stack/` | **Option B (k-fold OOF stack)** — 5-fold out-of-fold preds on TRAIN (4872 rows) + full-train→TEST (610 rows), for stacking. | `xgboost_v<ver>_preds_train_oof.parquet`, `…_preds_test.parquet` (GCS: `xgboost/`) |
| `shared/` | Shared split bundle — **never regenerate.** | `fold_map_b12_v3_k5_seed42.parquet`, `test_patient_ids.parquet` (GCS: `shared/`) |

**Contract (per parquet):** `patient_id` (braced GUID) · `split` · `proba_1` · `proba_0` ·
`model_name` · `model_version` · `abstained` (+ `fold` for Option B).

Base model: XGBoost. Option A selects the best lookback window on VALID; Option B uses the lifetime
window. Self-checks (row counts, no OOF/TEST overlap, fold-map agreement) passed before upload.
