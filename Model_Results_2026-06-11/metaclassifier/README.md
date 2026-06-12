# Metaclassifier deliverables (lung b12_v3)

Predictions for the ensemble owner, on the shared **b12_v3** split. The **code** that produces them is
committed here; the **prediction parquets** are patient-level (braced GUIDs), so they are **git-ignored**
(PHI) and delivered via **GCS**.

**GCS (parquets):** `gs://gcs-ai-dev-model-artifacts/yash/lung_model_artifacts_b12_v3_meta_classifier/`

## Code (committed)
| File | Role |
|---|---|
| `contract.py` | output contract: schema + `validate_df` + `normalise_pid` (braced-GUID handling) — shared by both options |
| `option_a_blending/score_b12v3.py` | **Option A** scorer — builds FE from b12_v3 events via the lung pipeline, selects best window on VALID, emits `xgboost_preds_{valid,test}.parquet` |
| `option_b_oof_stack/option_b_oof.py` | **Option B** — pulls the b12_v3 feature table, k-fold OOF on TRAIN + full-train→TEST, emits `..._preds_train_oof.parquet` + `..._preds_test.parquet` |
| `validate_preds.py` | validates a delivered parquet against `contract.py` |

> The scorers depend on the lung pipeline (`../lung/pipeline_code/`) and BigQuery access; they run on the
> project VMs (cpu-02). Paths/bucket are as configured at delivery time.

| Folder | What | Files |
|---|---|---|
| `option_a_blending/` | **Option A (blending)** — one model trained on b12_v3 TRAIN, predicts VALID + TEST. | `xgboost_preds_valid.parquet`, `xgboost_preds_test.parquet` (GCS: `xgb/`) |
| `option_b_oof_stack/` | **Option B (k-fold OOF stack)** — 5-fold out-of-fold preds on TRAIN (4872 rows) + full-train→TEST (610 rows), for stacking. | `xgboost_v<ver>_preds_train_oof.parquet`, `…_preds_test.parquet` (GCS: `xgboost/`) |
| `shared/` | Shared split bundle — **never regenerate.** | `fold_map_b12_v3_k5_seed42.parquet`, `test_patient_ids.parquet` (GCS: `shared/`) |

**Contract (per parquet):** `patient_id` (braced GUID) · `split` · `proba_1` · `proba_0` ·
`model_name` · `model_version` · `abstained` (+ `fold` for Option B).

Base model: XGBoost. Option A selects the best lookback window on VALID; Option B uses the lifetime
window. Self-checks (row counts, no OOF/TEST overlap, fold-map agreement) passed before upload.
