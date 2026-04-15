# ═══════════════════════════════════════════════════════════════
# TEXT FEATURES — Config-driven (Keywords + TF-IDF + BERT)
# Step 6: Extract text signal from ASSOCIATED_TEXT
# ═══════════════════════════════════════════════════════════════

import logging
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════

def extract_text_value(xml_text):
    """Extract readable text from XML-wrapped ASSOCIATED_TEXT."""
    if pd.isna(xml_text) or str(xml_text).strip() == '':
        return ''
    text = str(xml_text)
    values = re.findall(r"<value[^>]*>([^<]+)</value>", text)
    if values:
        return ' '.join(values).lower()
    return text.lower()


def build_patient_documents(clin_df, window_name, cfg, truncate=False):
    """Build one text document per patient from clinical text."""
    logger.info(f"  Building patient documents for {window_name}...")

    df = clin_df.copy()
    df.columns = df.columns.str.strip().str.upper()

    has_text = df[df['ASSOCIATED_TEXT'].notna() & (df['ASSOCIATED_TEXT'].str.strip() != '')].copy()
    has_text['CLEAN_TEXT'] = has_text['ASSOCIATED_TEXT'].apply(extract_text_value)
    has_text = has_text[has_text['CLEAN_TEXT'].str.len() > 10]

    clinical_text = has_text[has_text['CATEGORY'].isin(cfg.TEXT_CLINICAL_CATEGORIES)].copy()
    cancer_text = has_text[has_text['CLEAN_TEXT'].str.contains(
        cfg.TEXT_SEARCH_PATTERN, na=False, regex=True
    )]
    clinical_text = pd.concat([clinical_text, cancer_text]).drop_duplicates()
    logger.info(f"    Clinical text rows: {len(clinical_text):,}")

    boilerplate_pattern = '|'.join(cfg.TEXT_BOILERPLATE)

    def clean_and_concat(group):
        texts = []
        for text in group['CLEAN_TEXT'].values:
            if re.search(boilerplate_pattern, str(text)):
                cleaned = re.sub(boilerplate_pattern, '', str(text)).strip()
                if len(cleaned) > 20:
                    texts.append(cleaned)
            else:
                texts.append(str(text))
        combined = ' '.join(texts) if texts else ''
        if truncate:
            words = combined.split()
            if len(words) > 400:
                combined = ' '.join(words[:400])
        return combined

    patient_docs = clinical_text.groupby('PATIENT_GUID').apply(clean_and_concat)
    non_empty = patient_docs[patient_docs.str.len() > 10]
    logger.info(f"    Patients with meaningful text: {len(non_empty):,}")
    return patient_docs


# ═══════════════════════════════════════════════════════════════
# FUNCTION 1: TEXT KEYWORDS
# ═══════════════════════════════════════════════════════════════

def build_text_keywords(clin_df, window_name, cfg):
    """Extract clinical keywords from ASSOCIATED_TEXT."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  TEXT KEYWORD EXTRACTION - {window_name}")
    logger.info(f"{'='*70}")

    df = clin_df.copy()
    df.columns = df.columns.str.strip().str.upper()

    has_text = df[df['ASSOCIATED_TEXT'].notna() & (df['ASSOCIATED_TEXT'].str.strip() != '')].copy()
    has_text['CLEAN_TEXT'] = has_text['ASSOCIATED_TEXT'].apply(extract_text_value)
    has_text = has_text[has_text['CLEAN_TEXT'].str.len() > 3]

    patient_list = df['PATIENT_GUID'].unique()
    tf = pd.DataFrame(index=patient_list)
    tf.index.name = 'PATIENT_GUID'

    # BLOCK 1: IMAGING / FINDING FEATURES
    logger.info(f"  BLOCK 1: Imaging/finding features...")
    relevant_cats = cfg.TEXT_CLINICAL_CATEGORIES
    cat_text = has_text[has_text['CATEGORY'].isin(relevant_cats)].copy()
    cancer_text = has_text[has_text['CLEAN_TEXT'].str.contains(
        cfg.TEXT_SEARCH_PATTERN, na=False, regex=True
    )]
    nodal_text = pd.concat([cat_text, cancer_text]).drop_duplicates()

    for feat_name, pattern in cfg.TEXT_IMAGING_KEYWORDS.items():
        matches = nodal_text[nodal_text['CLEAN_TEXT'].str.contains(pattern, na=False, regex=True)]
        patient_match = matches.groupby('PATIENT_GUID').size() > 0
        tf[feat_name] = tf.index.map(patient_match).fillna(False).astype(int)

    has_any = nodal_text.groupby('PATIENT_GUID').size() > 0
    tf['TEXT_IMG_has_report'] = tf.index.map(has_any).fillna(False).astype(int)

    abnormal = nodal_text[~nodal_text['CLEAN_TEXT'].str.contains(
        r'normal|no\s*abnormal|unremark|no\s*significant|within\s*normal', na=False
    )]
    has_abn = abnormal.groupby('PATIENT_GUID').size() > 0
    tf['TEXT_IMG_abnormal_report'] = tf.index.map(has_abn).fillna(False).astype(int)

    # BLOCK 2: REFERRAL — SKIPPED (leakage)
    logger.info(f"  BLOCK 2: Referral features - SKIPPED (leakage)")

    # BLOCK 3: SEVERITY LANGUAGE
    logger.info(f"  BLOCK 3: Symptom severity...")
    severity_cats = cfg.TEXT_SEVERITY_CATEGORIES
    clinical_text = has_text[has_text['CATEGORY'].isin(severity_cats)].copy()

    for feat_name, pattern in cfg.TEXT_SEVERITY_KEYWORDS.items():
        matches = clinical_text[clinical_text['CLEAN_TEXT'].str.contains(pattern, na=False, regex=True)]
        patient_match = matches.groupby('PATIENT_GUID').size() > 0
        tf[feat_name] = tf.index.map(patient_match).fillna(False).astype(int)

    # BLOCK 4: Lab text — SKIPPED
    logger.info(f"  BLOCK 4: Lab text features - SKIPPED (inverted signal)")

    # BLOCK 5: CLINICAL CONTEXT
    logger.info(f"  BLOCK 5: Clinical context...")
    all_text = has_text.copy()
    for feat_name, pattern in cfg.TEXT_CLINICAL_KEYWORDS.items():
        matches = all_text[all_text['CLEAN_TEXT'].str.contains(pattern, na=False, regex=True)]
        patient_match = matches.groupby('PATIENT_GUID').size() > 0
        tf[feat_name] = tf.index.map(patient_match).fillna(False).astype(int)

    # BLOCK 6: COMPOSITE TEXT FEATURES
    logger.info(f"  BLOCK 6: Composite text features...")

    img_finding_cols = [c for c in tf.columns if c.startswith('TEXT_IMG_') and c not in
                        ['TEXT_IMG_has_report', 'TEXT_IMG_abnormal_report', 'TEXT_IMG_normal']]
    tf['TEXT_COMP_imaging_finding_count'] = tf[img_finding_cols].sum(axis=1)

    # Prostate cluster: PSA signal + urinary + bone/prostate finding
    tf['TEXT_COMP_prostate_cluster'] = (
        ((tf.get('TEXT_IMG_psa_elevated', 0) == 1) | (tf.get('TEXT_CLIN_psa_rising', 0) == 1))
        &
        ((tf.get('TEXT_IMG_prostate_enlargement', 0) == 1)
         | (tf.get('TEXT_IMG_bone_lesion', 0) == 1)
         | (tf.get('TEXT_CLIN_urinary', 0) == 1))
    ).astype(int)

    tf['TEXT_COMP_finding_with_severity'] = (
        ((tf.get('TEXT_IMG_psa_elevated', 0) == 1)
         | (tf.get('TEXT_IMG_bone_lesion', 0) == 1)
         | (tf.get('TEXT_IMG_prostate_enlargement', 0) == 1)
         | (tf.get('TEXT_IMG_biopsy_finding', 0) == 1)
         | (tf.get('TEXT_IMG_dre_abnormal', 0) == 1))
        &
        ((tf.get('TEXT_SEV_worsening', 0) == 1)
         | (tf.get('TEXT_SEV_recurrent', 0) == 1)
         | (tf.get('TEXT_SEV_persistent', 0) == 1))
    ).astype(int)

    signal_cols = [c for c in tf.columns if c.startswith('TEXT_') and c not in
                   ['TEXT_IMG_has_report', 'TEXT_IMG_normal']]
    tf['TEXT_COMP_any_signal'] = (tf[signal_cols].sum(axis=1) > 0).astype(int)
    tf['TEXT_COMP_total_signal_count'] = tf[signal_cols].sum(axis=1)

    # Clean up
    tf = tf.fillna(0).replace([np.inf, -np.inf], 0)
    nunique = tf.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        tf = tf.drop(columns=constant)

    logger.info(f"  Text features: {tf.shape[1]} total")
    return tf


# ═══════════════════════════════════════════════════════════════
# FUNCTION 2: TF-IDF EMBEDDINGS
# ═══════════════════════════════════════════════════════════════

def build_tfidf_embeddings(clin_df, existing_fm, window_name, cfg):
    """Build TF-IDF + SVD embedding features."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  TF-IDF EMBEDDINGS - {window_name}")
    logger.info(f"{'='*70}")

    n_comp = cfg.N_EMBEDDING_COMPONENTS
    patient_list = existing_fm.index.tolist()
    patient_docs = build_patient_documents(clin_df, window_name, cfg, truncate=False)

    aligned_docs = pd.Series('', index=patient_list)
    for pid in patient_list:
        if pid in patient_docs.index and len(str(patient_docs[pid])) > 10:
            aligned_docs[pid] = patient_docs[pid]

    non_empty = (aligned_docs.str.len() > 10).sum()
    logger.info(f"  Patients with text: {non_empty:,} / {len(patient_list):,}")

    tfidf = TfidfVectorizer(
        max_features=500, min_df=10, max_df=0.5,
        ngram_range=(1, 2), sublinear_tf=True,
        strip_accents='unicode', token_pattern=r'(?u)\b[a-z][a-z]+\b',
    )
    tfidf_matrix = tfidf.fit_transform(aligned_docs.values)

    actual_comp = min(n_comp, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=actual_comp, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)
    logger.info(f"  Variance explained: {svd.explained_variance_ratio_.sum():.1%}")

    # Save fitted transformers for inference
    transformer_dir = cfg.SCRIPT_DIR / 'results' / '6_text_features' / 'fitted_transformers' / window_name
    transformer_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, transformer_dir / 'tfidf_vectorizer.pkl')
    joblib.dump(svd, transformer_dir / 'svd_transformer.pkl')
    logger.info(f"  Saved fitted TF-IDF + SVD to {transformer_dir}")

    emb_df = pd.DataFrame(
        embeddings, index=patient_list,
        columns=[f'EMB_text_dim_{i}' for i in range(actual_comp)]
    )
    emb_df.index.name = 'PATIENT_GUID'
    return emb_df


# ═══════════════════════════════════════════════════════════════
# FUNCTION 3: BERT EMBEDDINGS
# ═══════════════════════════════════════════════════════════════

def build_bert_embeddings(clin_df, existing_fm, window_name, cfg):
    """Build BERT embedding features."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  BERT EMBEDDINGS - {window_name}")
    logger.info(f"{'='*70}")

    n_comp = cfg.N_EMBEDDING_COMPONENTS
    patient_list = existing_fm.index.tolist()
    patient_docs = build_patient_documents(clin_df, window_name, cfg, truncate=True)

    aligned_docs = []
    for pid in patient_list:
        if pid in patient_docs.index and len(str(patient_docs[pid])) > 10:
            aligned_docs.append(patient_docs[pid])
        else:
            aligned_docs.append('')

    non_empty = sum(1 for d in aligned_docs if len(d) > 10)
    logger.info(f"  Patients with text: {non_empty:,} / {len(patient_list):,}")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(cfg.BERT_MODEL_NAME)
        logger.info(f"  Model loaded: {cfg.BERT_MODEL_NAME}")
    except Exception as e:
        logger.warning(f"  Failed to load {cfg.BERT_MODEL_NAME}: {e}")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

    docs_to_encode = [d if len(d) > 10 else 'no clinical text available' for d in aligned_docs]
    embeddings_raw = model.encode(
        docs_to_encode, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    pca = PCA(n_components=n_comp, random_state=42)
    embeddings = pca.fit_transform(embeddings_raw)
    logger.info(f"  Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    # Save fitted PCA for inference
    transformer_dir = cfg.SCRIPT_DIR / 'results' / '6_text_features' / 'fitted_transformers' / window_name
    transformer_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, transformer_dir / 'bert_pca.pkl')
    logger.info(f"  Saved fitted BERT PCA to {transformer_dir}")

    for i, doc in enumerate(aligned_docs):
        if len(doc) <= 10:
            embeddings[i] = 0.0

    bert_df = pd.DataFrame(
        embeddings, index=patient_list,
        columns=[f'BERT_dim_{i}' for i in range(n_comp)]
    )
    bert_df.index.name = 'PATIENT_GUID'
    return bert_df


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

def run_text_features(cfg=None):
    """Run the full text feature pipeline."""
    if cfg is None:
        cfg = config

    for _dir in [cfg.TEXT_RESULTS, cfg.EMB_RESULTS, cfg.BERT_RESULTS]:
        for _w in cfg.WINDOWS:
            (_dir / _w).mkdir(parents=True, exist_ok=True)

    # Load data
    data = {}
    for window in cfg.WINDOWS:
        path = cfg.BASE_PATH / window / f'{cfg.DATA_PREFIX}_{window}_obs_dropped.csv'
        try:
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.upper()
            data[window] = df
            logger.info(f"Loaded {window}: {df.shape[0]:,} rows")
        except FileNotFoundError as e:
            logger.warning(f"WARNING {window}: {e}")
            data[window] = None

    clean_path = cfg.CLEANUP_RESULTS
    clean_matrices = {}
    for window in cfg.WINDOWS:
        path = clean_path / window / f'feature_matrix_clean_{window}.csv'
        if path.exists():
            clean_matrices[window] = pd.read_csv(path, index_col=0, usecols=['PATIENT_GUID', cfg.LABEL_COL])

    # Process each window
    for window in cfg.WINDOWS:
        if data[window] is None or window not in clean_matrices:
            continue

        # (a) Text keywords
        tf = build_text_keywords(data[window], window, cfg)
        out_path = cfg.TEXT_RESULTS / window / f'text_features_{window}.csv'
        tf.to_csv(out_path)
        logger.info(f"  Saved: {out_path}")

        # (b) TF-IDF embeddings
        emb = build_tfidf_embeddings(data[window], clean_matrices[window], window, cfg)
        out_path = cfg.EMB_RESULTS / window / f'text_embeddings_{window}.csv'
        emb.to_csv(out_path)
        logger.info(f"  Saved: {out_path}")

        # (c) BERT embeddings
        try:
            bert = build_bert_embeddings(data[window], clean_matrices[window], window, cfg)
            out_path = cfg.BERT_RESULTS / window / f'bert_embeddings_{window}.csv'
            bert.to_csv(out_path)
            logger.info(f"  Saved: {out_path}")
        except Exception as e:
            logger.warning(f"  BERT embeddings failed for {window}: {e}")

    logger.info(f"\n  TEXT FEATURE PIPELINE COMPLETE")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_text_features()
