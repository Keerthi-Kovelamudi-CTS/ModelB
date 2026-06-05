# Vertex AI Pipelines — Step-by-Step Build Guide (v147)

A concrete walkthrough for converting `lung_lstm_v147_bq_v45.ipynb` into a Vertex AI Pipeline. Each step shows the structure and key files; fill in the body with the existing v147 code.

---

## Step 0 — Prerequisites

Set up locally:

```bash
pip install kfp google-cloud-aiplatform google-cloud-pipeline-components
gcloud auth application-default login
```

Pick GCP resources:

- **Project**: `prj-cts-ai-dev-sp`
- **Region**: `us-central1` (data is currently in US per `BQ_LOCATION='US'`)
- **Pipeline staging bucket**: e.g. `gs://prj-cts-ai-dev-sp-vertex-pipelines/lung-lstm`
- **Artifact Registry for the Docker image**: e.g. `us-central1-docker.pkg.dev/prj-cts-ai-dev-sp/ml-images/lung-lstm`

---

## Step 1 — Refactor the notebook into a Python package

Notebooks aren't pipeline-ready. Convert to:

```
lung-lstm/
├── src/lung_lstm/
│   ├── __init__.py
│   ├── config.py           # CANCER_PROFILES + default_config()
│   ├── data.py             # load_data_bq(), pre_process(), process_data()
│   ├── features.py         # build_vocab(), encode_df(), get_seq_data()
│   ├── model.py            # build_model() (cell 48 verbatim)
│   ├── train.py            # train_model() (cell 51 verbatim)
│   ├── evaluate.py         # evaluate_model() + find_optimal_threshold()
│   └── explainability.py   # run_explainability() + helpers (cell 70)
├── pipelines/
│   ├── components/
│   │   ├── extract.py
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── register.py
│   └── pipeline.py
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

**Practical tip**: use `jupyter nbconvert --to script lung_lstm_v147_bq_v45.ipynb` to dump the cells, then split by section.

---

## Step 2 — Build a base Docker image

`Dockerfile`:

```dockerfile
FROM python:3.11-slim

# CUDA base if you need GPU training
# FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04
# RUN apt-get update && apt-get install -y python3.11 python3-pip

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
ENV PYTHONPATH=/app/src
```

Pin everything in `requirements.txt` — TF, keras, numpy, pandas, scikit-learn, google-cloud-bigquery, google-cloud-aiplatform, kfp, pyarrow, etc.

Build and push:

```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/prj-cts-ai-dev-sp/ml-images/lung-lstm:latest
```

---

## Step 3 — Define KFP components

KFP has two component styles: **Python function components** (lightweight, good for orchestration glue) and **container components** (run your Docker image, good for heavy lifting). Use containers for training/inference, Python for thin coordination.

### `pipelines/components/extract.py`

```python
from kfp.dsl import component, Output, Dataset

@component(
    base_image='us-central1-docker.pkg.dev/prj-cts-ai-dev-sp/ml-images/lung-lstm:latest',
    packages_to_install=[],
)
def extract_data(
    project: str,
    sql_file_gcs_uri: str,
    target_cancer: str,
    data_version: str,
    bq_location: str,
    output_data: Output[Dataset],
) -> None:
    from lung_lstm.data import load_data_bq
    df = load_data_bq(project=project, sql_file_uri=sql_file_gcs_uri,
                      target_cancer=target_cancer, data_version=data_version,
                      bq_location=bq_location)
    df.to_parquet(output_data.path)
```

### `pipelines/components/preprocess.py`

```python
@component(base_image='us-central1-docker.pkg.dev/prj-cts-ai-dev-sp/ml-images/lung-lstm:latest')
def preprocess(
    raw_data: Input[Dataset],
    target_cancer: str,
    output_processed: Output[Dataset],
    output_vocab: Output[Artifact],
) -> None:
    import pandas as pd
    from lung_lstm.config import CANCER_PROFILES, default_config
    from lung_lstm.data import pre_process, process_data, get_df_filtered
    from lung_lstm.features import build_vocab

    config = default_config(); config.update(CANCER_PROFILES[target_cancer])
    df = pd.read_parquet(raw_data.path)
    df = pre_process(df, config)
    df = process_data(df, config)
    df = get_df_filtered(df, config)
    vocab = build_vocab(df, config)

    df.to_parquet(output_processed.path)
    import json
    with open(output_vocab.path, 'w') as f:
        json.dump(vocab, f)
```

### `pipelines/components/train.py`

Heaviest component, needs GPU machine spec at pipeline time:

```python
@component(base_image='us-central1-docker.pkg.dev/prj-cts-ai-dev-sp/ml-images/lung-lstm:latest')
def train(
    processed_data: Input[Dataset],
    vocab: Input[Artifact],
    target_cancer: str,
    code_version: str,
    output_model: Output[Model],
    output_metrics: Output[Metrics],
) -> None:
    import json, pandas as pd
    from lung_lstm.config import CANCER_PROFILES, default_config
    from lung_lstm.features import encode_df, get_seq_data, prepare_split
    from lung_lstm.model import build_model
    from lung_lstm.train import train_model

    config = default_config(); config.update(CANCER_PROFILES[target_cancer])
    config['code_version'] = code_version

    df = pd.read_parquet(processed_data.path)
    with open(vocab.path) as f: vocabs = json.load(f)
    df_enc = encode_df(df, vocabs, config)
    data = prepare_split(df_enc, config)

    model = build_model(config, **data['shapes'])
    history, model = train_model(model, data, config)

    model.save(f'{output_model.path}/model.keras')
    output_metrics.log_metric('best_val_score', float(max(history.history['val_weighted_sens_spec'])))
```

Same idea for `evaluate.py` (loads the saved model, runs `evaluate_model` + `find_optimal_threshold`, writes metrics) and `register.py` (uploads to Vertex AI Model Registry only if metrics pass a gate).

---

## Step 4 — Define the pipeline

`pipelines/pipeline.py`:

```python
from kfp import dsl, compiler
from kfp.dsl import pipeline
from components.extract import extract_data
from components.preprocess import preprocess
from components.train import train
from components.evaluate import evaluate
from components.register import register_model

@pipeline(name='lung-lstm-train-pipeline', description='v147 lung LSTM training')
def lung_lstm_pipeline(
    project: str = 'prj-cts-ai-dev-sp',
    target_cancer: str = 'lung',
    data_version: str = 'v45',
    code_version: str = 'v147',
    sql_file_gcs_uri: str = 'gs://.../sql/lung_v45.sql',
    bq_location: str = 'US',
    sens_floor: float = 0.90,
    relevance_floor: float = 0.30,
):
    extract = extract_data(
        project=project, sql_file_gcs_uri=sql_file_gcs_uri,
        target_cancer=target_cancer, data_version=data_version,
        bq_location=bq_location,
    )

    pp = preprocess(
        raw_data=extract.outputs['output_data'],
        target_cancer=target_cancer,
    )

    tr = train(
        processed_data=pp.outputs['output_processed'],
        vocab=pp.outputs['output_vocab'],
        target_cancer=target_cancer,
        code_version=code_version,
    )
    # GPU machine for training
    tr.set_accelerator_type('NVIDIA_TESLA_T4').set_accelerator_limit(1)
    tr.set_cpu_limit('8').set_memory_limit('32G')

    ev = evaluate(
        model=tr.outputs['output_model'],
        processed_data=pp.outputs['output_processed'],
        vocab=pp.outputs['output_vocab'],
        target_cancer=target_cancer,
    )

    # Gate: only register if both thresholds pass
    with dsl.If((ev.outputs['sensitivity'] >= sens_floor)
                & (ev.outputs['clinical_relevance'] >= relevance_floor)):
        register_model(
            model=tr.outputs['output_model'],
            metrics=ev.outputs['output_metrics'],
            project=project,
            display_name=f'lung-lstm-{code_version}-{data_version}',
        )

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=lung_lstm_pipeline,
        package_path='lung_lstm_pipeline.json',
    )
```

---

## Step 5 — Submit to Vertex AI

`run_pipeline.py`:

```python
from google.cloud import aiplatform

aiplatform.init(
    project='prj-cts-ai-dev-sp',
    location='us-central1',
    staging_bucket='gs://prj-cts-ai-dev-sp-vertex-pipelines/lung-lstm',
)

job = aiplatform.PipelineJob(
    display_name='lung-lstm-v147-run',
    template_path='lung_lstm_pipeline.json',
    pipeline_root='gs://prj-cts-ai-dev-sp-vertex-pipelines/lung-lstm/runs',
    parameter_values={
        'target_cancer': 'lung',
        'data_version': 'v45',
        'code_version': 'v147',
        'sql_file_gcs_uri': 'gs://prj-cts-ai-dev-sp-vertex-pipelines/lung-lstm/sql/lung_v45.sql',
        'sens_floor': 0.90,
        'relevance_floor': 0.30,
    },
    enable_caching=True,
)

job.submit(service_account='sa-cts-ai-compute-dev@prj-cts-ai-dev-sp.iam.gserviceaccount.com')
```

Run:

```bash
python pipelines/pipeline.py    # compile to JSON
python run_pipeline.py          # submit
```

Watch in the GCP console at **Vertex AI → Pipelines**.

---

## Step 6 — Schedule recurring runs

Two options:

1. **Vertex AI Pipelines schedules** (built-in cron):

```python
job.create_schedule(
    cron='0 2 1 * *',   # 2am on the 1st of each month
    display_name='lung-lstm-monthly-retrain',
    max_concurrent_run_count=1,
)
```

2. **Cloud Scheduler → Pub/Sub → Cloud Function**: more flexible if you want to parameterize per run.

---

## Step 7 — Add what production needs but the notebook doesn't have

In rough order:

1. **Service account** with `roles/bigquery.dataViewer`, `roles/aiplatform.user`, `roles/storage.objectAdmin` on the staging bucket.
2. **Tests** under `tests/` — at minimum a smoke test that builds a model with tiny shapes and trains 1 epoch.
3. **Model Registry entries** with metadata: training data version, code version, eval metrics, training cohort size. Lets you roll back.
4. **Logging**: log every prediction at inference time with patient_id (hashed), score, model version → BQ table for drift monitoring.
5. **Drift monitor**: Vertex AI Model Monitoring against a reference dataset (your training data distribution).
6. **Eval gate** (already in the pipeline above): if sens or relevance drops below floor, the new model isn't promoted.
7. **CI**: GitHub Actions workflow that builds the image, runs tests, compiles the pipeline JSON, optionally submits to a dev pipeline-root.

---

## What to build first

Don't try to do all of this at once. The minimum viable progression:

1. Refactor notebook → package, get the existing CSV path running outside Jupyter (1–2 days).
2. Containerize, run the train script inside Docker locally (1 day).
3. Submit a single Vertex AI Custom Training Job (no pipeline yet) just to prove the container works on Vertex (½ day).
4. Wrap into a 2-step pipeline (preprocess + train), ignore eval/register for now (1 day).
5. Add evaluate + register components, add the gate (1 day).
6. Add scheduling and monitoring (1 day each, can be later).

Each step is independently testable. Do not skip steps 2–3; the most common failure mode is debugging a 5-component pipeline because step 1 has an import error.
