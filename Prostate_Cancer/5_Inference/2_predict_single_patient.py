# ═══════════════════════════════════════════════════════════════
# SINGLE PATIENT INFERENCE — SELF-CONTAINED
# No dependency on 2_Feature_Engineering/ or 3_Modeling/.
# Everything needed is in 5_Inference/pipeline/ and artifacts/.
#
# Usage:
#   python 2_predict_single_patient.py --input patient.json --window 12mo
#   python 2_predict_single_patient.py --input patient.csv --window 12mo
#   python 2_predict_single_patient.py --input patient.json --window 12mo --top-factors 15
#
# Input format (JSON): see sample_input/sample_patient.json
# Input format (CSV):  standard windowed CSV with PATIENT_GUID, EVENT_DATE,
#                       CATEGORY, TERM, VALUE, etc.
# ═══════════════════════════════════════════════════════════════

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / 'artifacts'
PIPELINE_DIR = SCRIPT_DIR / 'pipeline'

# Import from bundled pipeline/ (self-contained, no external dependency)
sys.path.insert(0, str(PIPELINE_DIR))
from importlib import import_module

_config = import_module('config')
_pipeline = import_module('3_pipeline')
_cancer = import_module('4_cancer_features')
_text = import_module('6_text_features')


# ═══════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════════

def load_artifacts(window):
    """Load all inference artifacts for a window."""
    art_dir = ARTIFACTS_DIR / window

    if not art_dir.exists():
        raise FileNotFoundError(
            f"Artifacts not found at {art_dir}.\n"
            f"Run: python 1_package_artifacts.py --window {window}"
        )

    artifacts = {}

    # Model config
    artifacts['config'] = joblib.load(art_dir / 'models' / 'config.json')

    # Models
    artifacts['models'] = {}
    for name in artifacts['config']['ensemble_models']:
        artifacts['models'][name] = joblib.load(art_dir / 'models' / f'{name}_model.pkl')

    # Text transformers (optional)
    artifacts['tfidf'] = None
    artifacts['svd'] = None
    artifacts['bert_pca'] = None
    t_dir = art_dir / 'transformers'
    for key, fname in [('tfidf', 'tfidf_vectorizer.pkl'),
                       ('svd', 'svd_transformer.pkl'),
                       ('bert_pca', 'bert_pca.pkl')]:
        path = t_dir / fname
        if path.exists():
            artifacts[key] = joblib.load(path)

    # Code mapping (optional)
    mapping_path = art_dir / 'code_category_mapping.json'
    if mapping_path.exists():
        with open(mapping_path) as f:
            artifacts['mapping'] = json.load(f)
    else:
        artifacts['mapping'] = None

    # Selected features importance (for explanation)
    feat_path = art_dir / 'selected_features.csv'
    if feat_path.exists():
        imp_df = pd.read_csv(feat_path)
        artifacts['feature_importances'] = dict(zip(imp_df['feature'], imp_df['importance']))
    else:
        artifacts['feature_importances'] = {}

    logger.info(f"  Loaded artifacts for {window}")
    logger.info(f"    Models:       {list(artifacts['models'].keys())}")
    logger.info(f"    Ensemble:     {artifacts['config']['ensemble_models']} "
                f"w={[round(w,2) for w in artifacts['config']['ensemble_weights']]}")
    logger.info(f"    Threshold:    {artifacts['config']['threshold']:.4f}")
    logger.info(f"    Features:     {len(artifacts['config']['selected_features'])}")
    logger.info(f"    TF-IDF:       {'yes' if artifacts['tfidf'] else 'no'}")
    logger.info(f"    BERT PCA:     {'yes' if artifacts['bert_pca'] else 'no'}")

    return artifacts


# ═══════════════════════════════════════════════════════════════
# PARSE INPUT
# ═══════════════════════════════════════════════════════════════

def parse_json_input(json_path, window):
    """Convert JSON patient events to obs/med DataFrames."""
    with open(json_path) as f:
        patient = json.load(f)

    patient_guid = patient['patient_guid']
    sex = patient.get('sex', 'M')
    age = patient.get('age_at_index', 65)
    index_date = pd.Timestamp(patient.get('index_date', pd.Timestamp.now().strftime('%Y-%m-%d')))

    events = patient.get('events', [])
    if not events:
        raise ValueError("No events found in patient JSON")

    rows = []
    for evt in events:
        rows.append({
            'PATIENT_GUID': patient_guid,
            'EVENT_DATE': evt.get('event_date'),
            'INDEX_DATE': index_date,
            'CODE_ID': evt.get('code_id'),
            'SNOMED_ID': evt.get('snomed_id', evt.get('code_id')),
            'CATEGORY': evt.get('category', ''),
            'TERM': evt.get('term', ''),
            'VALUE': evt.get('value'),
            'ASSOCIATED_TEXT': evt.get('associated_text', ''),
            'LABEL': 0,
            'SEX': sex,
            'AGE_AT_INDEX': age,
            'EVENT_TYPE': evt.get('event_type', 'observation'),
        })

    df = pd.DataFrame(rows)
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
    df['INDEX_DATE'] = pd.to_datetime(df['INDEX_DATE'], errors='coerce')
    df['MONTHS_BEFORE_INDEX'] = ((df['INDEX_DATE'] - df['EVENT_DATE']).dt.days / 30.44).round(1)

    # Assign time windows
    window_months = {'3mo': 3, '6mo': 6, '12mo': 12}
    w = window_months.get(window, 12)
    df['TIME_WINDOW'] = 'A'
    df.loc[df['MONTHS_BEFORE_INDEX'] <= w, 'TIME_WINDOW'] = 'B'

    obs_df = df[df['EVENT_TYPE'] == 'observation'].copy()
    med_df = df[df['EVENT_TYPE'] == 'medication'].copy()

    if len(med_df) == 0:
        med_df = pd.DataFrame({
            'PATIENT_GUID': [patient_guid], 'LABEL': [0], 'SEX': [sex],
            'AGE_AT_INDEX': [age], 'TIME_WINDOW': ['B'], 'CATEGORY': ['NONE'],
            'EVENT_DATE': [index_date], 'INDEX_DATE': [index_date],
            'CODE_ID': ['0'], 'MONTHS_BEFORE_INDEX': [0],
        })

    logger.info(f"  Patient:  {patient_guid} | Age: {age} | Sex: {sex}")
    logger.info(f"  Events:   {len(obs_df)} observations, {len(med_df)} medications")

    return obs_df, med_df, patient_guid


def parse_csv_input(csv_path):
    """Load pre-formatted CSV."""
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.upper()

    patient_guid = df['PATIENT_GUID'].iloc[0]
    obs_df = df[df['EVENT_TYPE'].str.lower() == 'observation'].copy()
    med_df = df[df['EVENT_TYPE'].str.lower() == 'medication'].copy()

    logger.info(f"  Patient:  {patient_guid}")
    logger.info(f"  Events:   {len(obs_df)} observations, {len(med_df)} medications")

    return obs_df, med_df, patient_guid


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (uses bundled pipeline/)
# ═══════════════════════════════════════════════════════════════

def build_features(obs_df, med_df, window, artifacts):
    """Run the full FE pipeline on single-patient data."""
    cfg = _config

    if 'VALUE' in obs_df.columns:
        obs_df['VALUE'] = pd.to_numeric(obs_df['VALUE'], errors='coerce')

    # Base features
    clin_feat = _pipeline.build_clinical_features(obs_df, cfg)
    med_feat = _pipeline.build_medication_features(med_df, cfg)
    fm = clin_feat.join(med_feat, how='left')
    med_cols = [c for c in fm.columns if c.startswith('MED_')]
    fm[med_cols] = fm[med_cols].fillna(0)
    fm = _pipeline.build_interaction_features(fm, cfg)

    # Advanced
    adv = _pipeline.build_advanced_features(obs_df, med_df, fm, window.upper(), cfg)
    fm = fm.join(adv, how='left').fillna(0).replace([np.inf, -np.inf], 0)
    fm = fm.loc[:, ~fm.columns.duplicated()]

    # Maximum
    mega = _pipeline.extract_maximum_features(obs_df, med_df, fm, window.upper(), cfg)
    fm = fm.join(mega, how='left').fillna(0).replace([np.inf, -np.inf], 0)
    fm = fm.loc[:, ~fm.columns.duplicated()]

    # Cancer-specific
    cancer = _cancer.build_cancer_specific_features(obs_df, med_df, fm, window.upper(), cfg)
    fm = fm.join(cancer, how='left').fillna(0).replace([np.inf, -np.inf], 0)
    fm = fm.loc[:, ~fm.columns.duplicated()]

    # Signal + trend
    sig = _pipeline.build_new_signal_features(obs_df, med_df, fm, window.upper(), cfg)
    fm = fm.join(sig, how='left').fillna(0).replace([np.inf, -np.inf], 0)
    fm = fm.loc[:, ~fm.columns.duplicated()]

    trend = _pipeline.build_trend_features(obs_df, med_df, fm, window.upper(), cfg)
    fm = fm.join(trend, how='left').fillna(0).replace([np.inf, -np.inf], 0)
    fm = fm.loc[:, ~fm.columns.duplicated()]

    logger.info(f"  Clinical features: {fm.shape[1]}")

    # Text keywords
    try:
        text_kw = _text.build_text_keywords(obs_df, window, cfg)
        fm = fm.join(text_kw, how='left')
        logger.info(f"  + {text_kw.shape[1]} text keyword features")
    except Exception as e:
        logger.debug(f"  Text keywords skipped: {e}")

    # TF-IDF embeddings (using saved vectorizer)
    if artifacts['tfidf'] is not None and artifacts['svd'] is not None:
        try:
            patient_docs = _text.build_patient_documents(obs_df, window, cfg, truncate=False)
            patient_list = fm.index.tolist()
            aligned = pd.Series('', index=patient_list)
            for pid in patient_list:
                if pid in patient_docs.index and len(str(patient_docs[pid])) > 10:
                    aligned[pid] = patient_docs[pid]

            tfidf_matrix = artifacts['tfidf'].transform(aligned.values)
            emb = artifacts['svd'].transform(tfidf_matrix)
            emb_df = pd.DataFrame(emb, index=patient_list,
                                  columns=[f'EMB_text_dim_{i}' for i in range(emb.shape[1])])
            emb_df.index.name = 'PATIENT_GUID'
            fm = fm.join(emb_df, how='left')
            logger.info(f"  + {emb.shape[1]} TF-IDF embedding features")
        except Exception as e:
            logger.debug(f"  TF-IDF skipped: {e}")

    # BERT embeddings (using saved PCA)
    if artifacts['bert_pca'] is not None:
        try:
            from sentence_transformers import SentenceTransformer
            patient_docs = _text.build_patient_documents(obs_df, window, cfg, truncate=True)
            patient_list = fm.index.tolist()
            docs = []
            for pid in patient_list:
                if pid in patient_docs.index and len(str(patient_docs[pid])) > 10:
                    docs.append(patient_docs[pid])
                else:
                    docs.append('no clinical text available')

            bert_model = SentenceTransformer(cfg.BERT_MODEL_NAME)
            raw_emb = bert_model.encode(docs, normalize_embeddings=True)
            emb = artifacts['bert_pca'].transform(raw_emb)
            for i, pid in enumerate(patient_list):
                if pid not in patient_docs.index or len(str(patient_docs.get(pid, ''))) <= 10:
                    emb[i] = 0.0

            bert_df = pd.DataFrame(emb, index=patient_list,
                                   columns=[f'BERT_dim_{i}' for i in range(emb.shape[1])])
            bert_df.index.name = 'PATIENT_GUID'
            fm = fm.join(bert_df, how='left')
            logger.info(f"  + {emb.shape[1]} BERT embedding features")
        except Exception as e:
            logger.debug(f"  BERT skipped: {e}")

    fm = fm.fillna(0).replace([np.inf, -np.inf], 0)
    fm = fm.loc[:, ~fm.columns.duplicated()]
    logger.info(f"  Total features: {fm.shape[1]}")
    return fm


# ═══════════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════════

def predict(fm, artifacts):
    """Run ensemble prediction."""
    config = artifacts['config']
    selected = config['selected_features']
    threshold = config['threshold']
    model_names = config['ensemble_models']
    weights = config['ensemble_weights']

    # Align features
    missing = [f for f in selected if f not in fm.columns]
    if missing:
        logger.info(f"  {len(missing)} features not in patient data (filled with 0)")
        for f in missing:
            fm[f] = 0

    X = fm[selected]

    # Individual model predictions
    preds = {}
    for name in model_names:
        preds[name] = float(artifacts['models'][name].predict_proba(X)[:, 1][0])

    # Ensemble
    risk_score = weights[0] * preds[model_names[0]] + weights[1] * preds[model_names[1]]
    prediction = 'HIGH RISK' if risk_score >= threshold else 'LOW RISK'

    return risk_score, prediction, threshold, preds


# ═══════════════════════════════════════════════════════════════
# EXPLAIN
# ═══════════════════════════════════════════════════════════════

def explain_prediction(fm, artifacts, top_n=10):
    """Get top contributing features — features with highest value * importance."""
    selected = artifacts['config']['selected_features']
    imp_dict = artifacts.get('feature_importances', {})

    contributions = []
    for f in selected:
        if f not in fm.columns:
            continue
        val = float(fm[f].iloc[0])
        imp = imp_dict.get(f, 0.0)
        if val != 0 and imp > 0:
            contributions.append({
                'feature': f,
                'value': round(val, 4),
                'importance': round(imp, 6),
                'contribution': round(abs(val * imp), 6),
            })

    contributions.sort(key=lambda x: x['contribution'], reverse=True)
    return contributions[:top_n]


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Single patient cancer risk prediction')
    parser.add_argument('--input', required=True, help='Path to patient JSON or CSV')
    parser.add_argument('--window', default='12mo', choices=['3mo', '6mo', '12mo'])
    parser.add_argument('--top-factors', type=int, default=10)
    parser.add_argument('--output', default=None, help='Output JSON path (default: auto)')
    parser.add_argument('--quiet', action='store_true', help='Suppress FE log output')
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format='%(message)s')

    logger.info(f"\n{'='*60}")
    logger.info(f"  PROSTATE CANCER RISK PREDICTION")
    logger.info(f"  Window: {args.window}")
    logger.info(f"{'='*60}\n")

    # 1. Load artifacts
    artifacts = load_artifacts(args.window)

    # 2. Parse input
    input_path = Path(args.input)
    if input_path.suffix == '.json':
        obs_df, med_df, patient_id = parse_json_input(input_path, args.window)
    elif input_path.suffix == '.csv':
        obs_df, med_df, patient_id = parse_csv_input(input_path)
    else:
        raise ValueError(f"Unsupported: {input_path.suffix} (use .json or .csv)")

    # 3. Feature engineering
    logger.info(f"\n  Building features...")
    fm = build_features(obs_df, med_df, args.window, artifacts)

    # 4. Predict
    risk_score, prediction, threshold, model_preds = predict(fm, artifacts)

    # 5. Explain
    top_factors = explain_prediction(fm, artifacts, top_n=args.top_factors)

    # 6. Display
    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"{'='*60}")
    print(f"  Patient:       {patient_id}")
    print(f"  Window:        {args.window}")
    print(f"  Risk Score:    {risk_score:.4f}")
    print(f"  Threshold:     {threshold:.4f}")
    print(f"  Prediction:    {prediction}")
    print(f"")
    for name, score in model_preds.items():
        print(f"    {name:15s}: {score:.4f}")

    print(f"\n  TOP RISK FACTORS:")
    print(f"  {'─'*55}")
    for i, f in enumerate(top_factors, 1):
        print(f"  {i:2d}. {f['feature']:45s} = {f['value']:.2f}")
    print(f"  {'─'*55}")

    # 7. Save JSON result
    result = {
        'patient_id': patient_id,
        'window': args.window,
        'risk_score': round(risk_score, 4),
        'prediction': prediction,
        'threshold': round(threshold, 4),
        'model_scores': {n: round(s, 4) for n, s in model_preds.items()},
        'top_risk_factors': [
            {'rank': i+1, 'feature': f['feature'], 'value': f['value'], 'importance': f['importance']}
            for i, f in enumerate(top_factors)
        ],
    }

    out_path = Path(args.output) if args.output else SCRIPT_DIR / f'prediction_{patient_id}_{args.window}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
