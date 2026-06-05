# LSTM Model Deployment on GCP — v147

The biggest fork in the road is **inference pattern**: real-time predictions (e.g. a clinician opens a patient record and gets a risk score back in seconds) or batch (score every patient nightly and store the results)? The whole architecture splits on that.

---

## Batch path (simpler — ~80% of healthcare ML deploys go this way)

1. **Refactor the notebook into a Python package** (training script + inference script as separate entry points).

2. **Containerize** with a Dockerfile pinning TF/Keras/numpy versions.

3. **Push the image to Artifact Registry.**

4. **Training**: **Vertex AI Custom Training Jobs** (or Vertex AI Pipelines if you want to orchestrate BQ → preprocess → train → evaluate → register as a DAG with Kubeflow components).

5. **Model artifacts → GCS**, registered in **Vertex AI Model Registry** with metadata (training data version, code version, eval metrics).

6. **Inference**: **Vertex AI Batch Prediction** — point it at a BQ table of patients, get scored output back in BQ. Schedule via Cloud Scheduler + Vertex AI Pipelines.

---

## Real-time path (more moving parts)

Add to the batch path:

7. **Deploy the registered model to a Vertex AI Endpoint** (autoscaling, GPU optional). Call it via gRPC/REST from your application.

8. **Add Vertex AI Model Monitoring** for input-drift and prediction-skew alerts.

9. **Add a feature retrieval layer** — either query BQ live (slow but simple) or **Vertex AI Feature Store** (faster, more infra).

---

## Things that need to happen regardless

In roughly this order before going live:

- Convert the notebook to scripts. Notebooks aren't deployable.
- Pin every dependency (Keras, TF, sentence-transformers if you keep semantic relevance, etc.).
- Move the BigQuery service-account auth out of the notebook and into a workload-identity binding on the runtime.
- Move secrets to Secret Manager.
- Add an evaluation gate: a registered model only gets promoted if test sens ≥ X and clinical-relevance ≥ Y. Otherwise you'll silently ship regressions.
- Decide on retraining cadence (data drifts in primary care — quarterly is a reasonable starting point).
- Logging: every prediction with patient_id (hashed), feature snapshot, score, model version. You'll need this for audit and drift monitoring.

---

## The hard parts that aren't infrastructure

- The clinical-relevance / sensitivity / specificity numbers from v147 aren't ready for clinical decision-making yet (relevance ~0.4 ceiling per earlier conversation, single-trial variance ~0.20). Production deployment usually wants those stable and validated against held-out cohorts before going live.

- **Data governance / compliance**: NHS / EMIS data has its own rules. Your team probably already knows the constraints; just flagging that GCP region (UK vs US — your data is currently in US per `BQ_LOCATION='US'`) may need to change for clinical deployment.

- The model is currently trained per-cancer-type via `CANCER_PROFILES`. Production likely wants either one model per cancer (multiple endpoints) or a unified multi-class model (architecture change).
