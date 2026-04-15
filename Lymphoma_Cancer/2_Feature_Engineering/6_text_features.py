# ═══════════════════════════════════════════════════════════════
# LYMPHOMA CANCER — TEXT FEATURES (Keywords + TF-IDF + BERT)
# Step 6: Merged text pipeline — keyword extraction, TF-IDF
#         embeddings, and BERT embeddings in a single script
# ═══════════════════════════════════════════════════════════════

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

# ─── Paths & output directories ──────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / 'data'

TEXT_RESULTS = SCRIPT_DIR / 'results' / 'Text_Features'
EMB_RESULTS = SCRIPT_DIR / 'results' / 'Text_Embeddings'
BERT_RESULTS = SCRIPT_DIR / 'results' / 'BERT_Embeddings'

for _dir in [TEXT_RESULTS, EMB_RESULTS, BERT_RESULTS]:
    for _w in ['3mo', '6mo', '12mo']:
        (_dir / _w).mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────
N_COMPONENTS = 15
BERT_MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'

# Lymphoma observation categories with clinically meaningful text
CLINICAL_CATEGORIES = [
    'B_SYMPTOMS', 'LYMPHADENOPATHY', 'SPLENOMEGALY',
    'UNEXPLAINED_MASSES', 'CONSTITUTIONAL_SYMPTOMS',
    'HAEMATOLOGICAL_ABNORMALITIES', 'HAEMATOLOGY_INVESTIGATION_SELECTIVE',
    'LAB_MARKERS', 'PROTEIN_STUDIES', 'ABNORMAL_IMAGING',
    'AUTOIMMUNE_IMMUNE', 'INFECTION_MARKERS', 'RECURRENT_INFECTIONS',
    'SKIN_SYMPTOMS', 'NEUROPATHY',
]

# Lymphoma-oriented keywords: lymph/node/spleen/mass terms, B-symptoms, subtypes
TEXT_SEARCH_PATTERN = (
    r'lymph|node|lymphom|hodgkin|reed[\s-]*sternberg|sternberg'
    r'|splenom|spleen|marrow|biops|follicul|diffuse[\s-]*large|dlbcl'
    r'|b[\s-]*cell|t[\s-]*cell|nodul|mass|haematol|haemato'
    r'|paraprotein|monoclonal|waldenstrom|mycosis|mantle'
)

BOILERPLATE = [
    r'normal[\s,-]*no\s*action',
    r'no\s*significant\s*pathology',
    r'within\s*normal\s*limits',
    r'unremarkable',
    r'please\s*note\s*method\s*changed',
    r'reference\s*range',
    r'adjusting\s*egfr',
    r'ckd\s*guidelines',
    r'efi\s*score',
    r'frailty',
]
BOILERPLATE_PATTERN = '|'.join(BOILERPLATE)


# ═══════════════════════════════════════════════════════════════
# SHARED UTILITY FUNCTIONS
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


def build_patient_documents(clin_df, window_name, truncate=False):
    """
    For each patient, concatenate all clinically meaningful text.
    Returns Series: patient_guid -> text document.

    Parameters
    ----------
    truncate : bool
        If True, cap each patient document at 400 words (used for BERT).
    """
    print(f"\n  Building patient documents for {window_name}...")

    df = clin_df.copy()
    df.columns = df.columns.str.strip().str.upper()

    has_text = df[df['ASSOCIATED_TEXT'].notna() & (df['ASSOCIATED_TEXT'].str.strip() != '')].copy()
    has_text['CLEAN_TEXT'] = has_text['ASSOCIATED_TEXT'].apply(extract_text_value)
    has_text = has_text[has_text['CLEAN_TEXT'].str.len() > 10]

    # Only use clinically meaningful text (not lab boilerplate)
    clinical_text = has_text[has_text['CATEGORY'].isin(CLINICAL_CATEGORIES)].copy()

    # Also include any text mentioning lymphoma-related terms (from any category)
    lym_text = has_text[has_text['CLEAN_TEXT'].str.contains(
        TEXT_SEARCH_PATTERN, na=False, regex=True
    )]
    clinical_text = pd.concat([clinical_text, lym_text]).drop_duplicates()

    print(f"    Clinical text rows: {len(clinical_text):,}")

    def clean_and_concat(group):
        texts = []
        for text in group['CLEAN_TEXT'].values:
            if re.search(BOILERPLATE_PATTERN, str(text)):
                cleaned = re.sub(BOILERPLATE_PATTERN, '', str(text)).strip()
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
    print(f"    Patients with meaningful text: {len(non_empty):,}")

    return patient_docs


# ═══════════════════════════════════════════════════════════════
# FUNCTION 1: TEXT KEYWORDS
# ═══════════════════════════════════════════════════════════════

def build_text_keywords(clin_df, window_name):
    """
    Extract clinical keywords from ASSOCIATED_TEXT.
    Returns one row per patient with binary/count TEXT_ features.
    """

    print(f"\n{'═'*70}")
    print(f"  TEXT KEYWORD EXTRACTION — {window_name}")
    print(f"{'═'*70}")

    df = clin_df.copy()
    df.columns = df.columns.str.strip().str.upper()

    has_text = df[df['ASSOCIATED_TEXT'].notna() & (df['ASSOCIATED_TEXT'].str.strip() != '')].copy()
    print(f"  Rows with text: {len(has_text):,} / {len(df):,} ({len(has_text)/len(df)*100:.1f}%)")

    has_text['CLEAN_TEXT'] = has_text['ASSOCIATED_TEXT'].apply(extract_text_value)
    has_text = has_text[has_text['CLEAN_TEXT'].str.len() > 3]

    patient_list = df['PATIENT_GUID'].unique()
    tf = pd.DataFrame(index=patient_list)
    tf.index.name = 'PATIENT_GUID'

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: NODAL / HAEMATOLOGY / IMAGING FINDINGS
    # ══════════════════════════════════════════════════════════
    print(f"  BLOCK 1: Nodal / haem / imaging findings...")

    nodal_categories = [
        'LYMPHADENOPATHY', 'SPLENOMEGALY', 'UNEXPLAINED_MASSES',
        'ABNORMAL_IMAGING', 'HAEMATOLOGICAL_ABNORMALITIES',
        'HAEMATOLOGY_INVESTIGATION_SELECTIVE', 'LAB_MARKERS',
        'PROTEIN_STUDIES', 'AUTOIMMUNE_IMMUNE',
    ]
    cat_text = has_text[has_text['CATEGORY'].isin(nodal_categories)].copy()

    # Also pick up any text mentioning lymphoma terms
    lym_text = has_text[has_text['CLEAN_TEXT'].str.contains(
        TEXT_SEARCH_PATTERN, na=False, regex=True
    )]
    nodal_text = pd.concat([cat_text, lym_text]).drop_duplicates()

    imaging_keywords = {
        'TEXT_IMG_lymphadenopathy': r'lymph.{0,20}(?:enlarg|node|adenopathy|swollen)|lymphadenopathy',
        'TEXT_IMG_splenomegaly':    r'spleen.{0,15}(?:enlarg|megaly)|splenomegal',
        'TEXT_IMG_hepatomegaly':    r'liver.{0,15}enlarg|hepatomegal',
        'TEXT_IMG_bone_lesion':     r'bone.{0,20}(?:lesion|destruction|lucen|abnormal)',
        'TEXT_IMG_mediastinal':     r'mediastin.{0,15}(?:mass|lymph|widen)',
        'TEXT_IMG_mass_finding':    r'(?:mass|nodule|lump|lesion).{0,20}(?:enlarg|abnormal|suspicious|growing)|suspicious\s*(?:mass|node|lesion)',
        'TEXT_IMG_reed_sternberg':  r'reed[\s-]*sternberg|sternberg\s*cell|hodgkin',
        'TEXT_IMG_biopsy_finding':  r'biops|histolog|cytolog|lymphoma|follicul|diffuse[\s-]*large|dlbcl|mantle|marginal[\s-]*zone',
        'TEXT_IMG_monoclonal':      r'monoclonal|paraprotein|waldenstrom|mycosis|kappa.{0,15}lambda',
        'TEXT_IMG_anaemia_text':    r'anaemi|anemi|low\s*(?:hb|haemoglobin|hemoglobin)',
        'TEXT_IMG_abnormal_blood':  r'abnormal.{0,20}(?:blood|cell|count|film)|atypical\s*(?:cell|lymph)',
        'TEXT_IMG_normal':          r'(?:normal|no\s*abnormal|unremark|no\s*significant|within\s*normal)',
    }

    for feat_name, pattern in imaging_keywords.items():
        matches = nodal_text[nodal_text['CLEAN_TEXT'].str.contains(pattern, na=False, regex=True)]
        patient_match = matches.groupby('PATIENT_GUID').size() > 0
        tf[feat_name] = tf.index.map(patient_match).fillna(False).astype(int)

    # Has any haem/imaging text at all
    has_any = nodal_text.groupby('PATIENT_GUID').size() > 0
    tf['TEXT_IMG_has_report'] = tf.index.map(has_any).fillna(False).astype(int)

    # Abnormal report (has text and NOT normal-phrased)
    abnormal = nodal_text[~nodal_text['CLEAN_TEXT'].str.contains(
        r'normal|no\s*abnormal|unremark|no\s*significant|within\s*normal', na=False
    )]
    has_abn = abnormal.groupby('PATIENT_GUID').size() > 0
    tf['TEXT_IMG_abnormal_report'] = tf.index.map(has_abn).fillna(False).astype(int)

    img_count = len([c for c in tf.columns if c.startswith('TEXT_IMG_')])
    print(f"    Added {img_count} imaging/finding features")

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: REFERRAL & URGENCY LANGUAGE — SKIPPED (leakage)
    # ══════════════════════════════════════════════════════════
    print(f"  BLOCK 2: Referral features — SKIPPED (leakage/inverted signal)")

    # ══════════════════════════════════════════════════════════
    # BLOCK 3: SYMPTOM SEVERITY LANGUAGE
    # ══════════════════════════════════════════════════════════
    print(f"  BLOCK 3: Symptom severity (non-lab text only)...")

    severity_categories = [
        'B_SYMPTOMS', 'LYMPHADENOPATHY', 'SPLENOMEGALY',
        'UNEXPLAINED_MASSES', 'CONSTITUTIONAL_SYMPTOMS',
        'ABNORMAL_IMAGING', 'RECURRENT_INFECTIONS',
        'INFECTION_MARKERS', 'SKIN_SYMPTOMS', 'NEUROPATHY',
        'PAIN_SYMPTOMS', 'ADDITIONAL_SYMPTOMS',
    ]
    clinical_text = has_text[has_text['CATEGORY'].isin(severity_categories)].copy()
    print(f"    Clinical text rows (non-lab): {len(clinical_text):,}")

    severity_keywords = {
        'TEXT_SEV_worsening':  r'worsen|getting\s*worse|deteriorat|increas.{0,10}(pain|symptom|lump|swell|sweat|fatigue)',
        'TEXT_SEV_persistent': r'persistent\s*(pain|lump|symptom|infect|fever|sweat|fatigue|cough|itch)',
        'TEXT_SEV_severe':     r'severe\s*(pain|anaemi|infect|symptom|fatigue|sweat|swelling)',
        'TEXT_SEV_recurrent':  r'recurrent\s*(pain|infection|episode|presentation|fever|swelling)',
        'TEXT_SEV_new':        r'new\s*(symptom|lump|swelling|pain|mass|node)|recently\s*(develop|start|notic|enlarg)',
    }

    for feat_name, pattern in severity_keywords.items():
        matches = clinical_text[clinical_text['CLEAN_TEXT'].str.contains(pattern, na=False, regex=True)]
        patient_match = matches.groupby('PATIENT_GUID').size() > 0
        tf[feat_name] = tf.index.map(patient_match).fillna(False).astype(int)

    sev_count = len([c for c in tf.columns if c.startswith('TEXT_SEV_')])
    print(f"    Added {sev_count} severity features")

    # BLOCK 4: Lab text features REMOVED (inverted signal, boilerplate noise)
    print(f"  BLOCK 4: Lab text features — SKIPPED (inverted signal, boilerplate noise)")

    # ══════════════════════════════════════════════════════════
    # BLOCK 5: CLINICAL CONTEXT
    # ══════════════════════════════════════════════════════════
    print(f"  BLOCK 5: Clinical context...")

    all_text = has_text.copy()

    clinical_keywords = {
        'TEXT_CLIN_biopsy':         r'biops|histolog|cytolog|fine\s*needle|core\s*needle|aspirat',
        'TEXT_CLIN_lymphoma_term':  r'lymphom|hodgkin|follicul|diffuse[\s-]*large|dlbcl|mantle|marginal[\s-]*zone|burkitt|t[\s-]*cell\s*lymph|b[\s-]*cell\s*lymph',
        'TEXT_CLIN_bone_marrow':   r'bone\s*marrow|marrow\s*biops|marrow\s*aspir|trephine',
        'TEXT_CLIN_staging':        r'staging|ann\s*arbor|ipi\s*score|prognostic\s*index',
        'TEXT_CLIN_night_sweats':   r'night\s*sweat|drenching\s*sweat|soaked\s*sheets',
        'TEXT_CLIN_weight_loss':    r'weight\s*(loss|lost|losing|reduc)|lost\s*weight|appetite\s*(loss|poor|reduc)|cachex',
        'TEXT_CLIN_fever':          r'(?:fever|pyrexia|febrile|high\s*temperature)(?!\s*range)',
        'TEXT_CLIN_fatigue':        r'fatigu|tir|lethar|malaise|exhausti',
        'TEXT_CLIN_itch_pruritus':  r'itch|pruritus|pruritic',  # Hodgkin lymphoma clue
        'TEXT_CLIN_family_history': r'family\s*histor',
    }

    for feat_name, pattern in clinical_keywords.items():
        matches = all_text[all_text['CLEAN_TEXT'].str.contains(pattern, na=False, regex=True)]
        patient_match = matches.groupby('PATIENT_GUID').size() > 0
        tf[feat_name] = tf.index.map(patient_match).fillna(False).astype(int)

    clin_count = len([c for c in tf.columns if c.startswith('TEXT_CLIN_')])
    print(f"    Added {clin_count} clinical context features")

    # ══════════════════════════════════════════════════════════
    # BLOCK 6: COMPOSITE TEXT FEATURES
    # ══════════════════════════════════════════════════════════
    print(f"  BLOCK 6: Composite text features...")

    img_finding_cols = [c for c in tf.columns if c.startswith('TEXT_IMG_') and c not in
                        ['TEXT_IMG_has_report', 'TEXT_IMG_abnormal_report', 'TEXT_IMG_normal']]
    tf['TEXT_COMP_imaging_finding_count'] = tf[img_finding_cols].sum(axis=1)

    # Strong lymphoma hint: nodal/mass finding + severity language
    tf['TEXT_COMP_finding_with_severity'] = (
        ((tf.get('TEXT_IMG_lymphadenopathy', 0) == 1)
         | (tf.get('TEXT_IMG_mass_finding', 0) == 1)
         | (tf.get('TEXT_IMG_splenomegaly', 0) == 1)
         | (tf.get('TEXT_IMG_reed_sternberg', 0) == 1)
         | (tf.get('TEXT_IMG_biopsy_finding', 0) == 1))
        &
        ((tf.get('TEXT_SEV_worsening', 0) == 1)
         | (tf.get('TEXT_SEV_recurrent', 0) == 1)
         | (tf.get('TEXT_SEV_persistent', 0) == 1))
    ).astype(int)

    tf['TEXT_COMP_abnormal_img_with_severity'] = (
        (tf.get('TEXT_IMG_abnormal_report', 0) == 1)
        &
        ((tf.get('TEXT_SEV_worsening', 0) == 1)
         | (tf.get('TEXT_SEV_recurrent', 0) == 1)
         | (tf.get('TEXT_SEV_persistent', 0) == 1))
    ).astype(int)

    # Hodgkin-ish cluster (B symptoms + itch + nodal)
    tf['TEXT_COMP_hodgkin_cluster'] = (
        (tf.get('TEXT_CLIN_night_sweats', 0) == 1)
        & ((tf.get('TEXT_CLIN_itch_pruritus', 0) == 1) | (tf.get('TEXT_CLIN_weight_loss', 0) == 1))
        & (tf.get('TEXT_IMG_lymphadenopathy', 0) == 1)
    ).astype(int)

    signal_cols = [c for c in tf.columns if c.startswith('TEXT_') and c not in
                   ['TEXT_IMG_has_report', 'TEXT_IMG_normal']]
    tf['TEXT_COMP_any_signal'] = (tf[signal_cols].sum(axis=1) > 0).astype(int)
    tf['TEXT_COMP_total_signal_count'] = tf[signal_cols].sum(axis=1)

    comp_count = len([c for c in tf.columns if c.startswith('TEXT_COMP_')])
    print(f"    Added {comp_count} composite features")

    # Clean up
    tf = tf.fillna(0).replace([np.inf, -np.inf], 0)
    nunique = tf.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        tf = tf.drop(columns=constant)
        print(f"\n  Removed {len(constant)} constant columns")

    print(f"\n  Text features: {tf.shape[1]} total")
    for prefix, name in [
        ('TEXT_IMG_', 'Haem/imaging findings'), ('TEXT_SEV_', 'Severity'),
        ('TEXT_CLIN_', 'Clinical context'), ('TEXT_COMP_', 'Composite')
    ]:
        count = len([c for c in tf.columns if c.startswith(prefix)])
        if count > 0:
            print(f"    {name:25s}: {count}")

    return tf


# ═══════════════════════════════════════════════════════════════
# FUNCTION 2: TF-IDF EMBEDDINGS
# ═══════════════════════════════════════════════════════════════

def build_tfidf_embeddings(clin_df, existing_fm, window_name):
    """Build TF-IDF + SVD embedding features for each patient."""
    print(f"\n{'═'*70}")
    print(f"  TF-IDF EMBEDDINGS — {window_name}")
    print(f"{'═'*70}")

    patient_list = existing_fm.index.tolist()
    patient_docs = build_patient_documents(clin_df, window_name, truncate=False)

    aligned_docs = pd.Series('', index=patient_list)
    for pid in patient_list:
        if pid in patient_docs.index and len(str(patient_docs[pid])) > 10:
            aligned_docs[pid] = patient_docs[pid]

    non_empty_count = (aligned_docs.str.len() > 10).sum()
    print(f"  Patients with text: {non_empty_count:,} / {len(patient_list):,} "
          f"({non_empty_count/len(patient_list)*100:.1f}%)")

    print(f"  Building TF-IDF matrix...")
    tfidf = TfidfVectorizer(
        max_features=500,
        min_df=10,
        max_df=0.5,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents='unicode',
        token_pattern=r'(?u)\b[a-z][a-z]+\b',
    )

    tfidf_matrix = tfidf.fit_transform(aligned_docs.values)
    print(f"    TF-IDF shape: {tfidf_matrix.shape}")
    print(f"    Top terms: {tfidf.get_feature_names_out()[:20].tolist()}")

    n_comp = min(N_COMPONENTS, tfidf_matrix.shape[1] - 1)
    print(f"  Reducing to {n_comp} dimensions via SVD...")
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)

    variance_explained = svd.explained_variance_ratio_.sum()
    print(f"    Variance explained: {variance_explained:.1%}")

    emb_df = pd.DataFrame(
        embeddings,
        index=patient_list,
        columns=[f'EMB_text_dim_{i}' for i in range(n_comp)]
    )
    emb_df.index.name = 'PATIENT_GUID'

    merged = emb_df.join(existing_fm[['LABEL']], how='inner')
    corr = merged.drop(columns=['LABEL']).corrwith(merged['LABEL']).abs().sort_values(ascending=False)
    print(f"\n  Embedding dimensions by label correlation:")
    for i, (feat, c) in enumerate(corr.head(10).items()):
        print(f"    {i+1:2d}. {feat}: {c:.4f}")

    print(f"\n  Top TF-IDF terms enriched in cancer:")
    pos_idx = [i for i, pid in enumerate(patient_list)
               if pid in existing_fm.index and existing_fm.loc[pid, 'LABEL'] == 1]
    neg_idx = [i for i, pid in enumerate(patient_list)
               if pid in existing_fm.index and existing_fm.loc[pid, 'LABEL'] == 0]

    if len(pos_idx) > 0 and len(neg_idx) > 0:
        pos_mean = np.asarray(tfidf_matrix[pos_idx].mean(axis=0)).flatten()
        neg_mean = np.asarray(tfidf_matrix[neg_idx].mean(axis=0)).flatten()
        ratio = np.where(neg_mean > 0, pos_mean / neg_mean, 0)
        terms = tfidf.get_feature_names_out()
        top_terms = np.argsort(ratio)[-15:][::-1]
        for idx in top_terms:
            if ratio[idx] > 1.2:
                print(f"    {terms[idx]:<30s} cancer={pos_mean[idx]:.4f} "
                      f"non-ca={neg_mean[idx]:.4f} ratio={ratio[idx]:.1f}x")

    print(f"\n  Embedding features: {emb_df.shape[1]} dimensions")
    return emb_df


# ═══════════════════════════════════════════════════════════════
# FUNCTION 3: BERT EMBEDDINGS
# ═══════════════════════════════════════════════════════════════

def build_bert_embeddings(clin_df, existing_fm, window_name):
    """Build BERT embedding features using a clinical sentence-transformer."""
    print(f"\n{'═'*70}")
    print(f"  BERT EMBEDDINGS — {window_name}")
    print(f"{'═'*70}")

    patient_list = existing_fm.index.tolist()
    patient_docs = build_patient_documents(clin_df, window_name, truncate=True)

    aligned_docs = []
    for pid in patient_list:
        if pid in patient_docs.index and len(str(patient_docs[pid])) > 10:
            aligned_docs.append(patient_docs[pid])
        else:
            aligned_docs.append('')

    non_empty_count = sum(1 for d in aligned_docs if len(d) > 10)
    print(f"  Patients with text: {non_empty_count:,} / {len(patient_list):,}")

    print(f"  Loading model: {BERT_MODEL_NAME}...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(BERT_MODEL_NAME)
        print(f"  Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"  Failed to load {BERT_MODEL_NAME}: {e}")
        print(f"  Falling back to all-MiniLM-L6-v2...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"  Encoding {len(aligned_docs):,} patient documents...")
    docs_to_encode = [d if len(d) > 10 else 'no clinical text available' for d in aligned_docs]

    embeddings_raw = model.encode(
        docs_to_encode,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print(f"  Raw embedding shape: {embeddings_raw.shape}")

    print(f"  Reducing to {N_COMPONENTS} dimensions via PCA...")
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    embeddings = pca.fit_transform(embeddings_raw)
    variance = pca.explained_variance_ratio_.sum()
    print(f"  Variance explained: {variance:.1%}")

    for i, doc in enumerate(aligned_docs):
        if len(doc) <= 10:
            embeddings[i] = 0.0

    bert_df = pd.DataFrame(
        embeddings,
        index=patient_list,
        columns=[f'BERT_dim_{i}' for i in range(N_COMPONENTS)]
    )
    bert_df.index.name = 'PATIENT_GUID'

    merged = bert_df.join(existing_fm[['LABEL']], how='inner')
    corr = merged.drop(columns=['LABEL']).corrwith(merged['LABEL']).abs().sort_values(ascending=False)
    print(f"\n  BERT dimensions by label correlation:")
    for i, (feat, c) in enumerate(corr.head(10).items()):
        print(f"    {i+1:2d}. {feat}: {c:.4f}")

    print(f"\n  BERT features: {bert_df.shape[1]} dimensions")
    return bert_df


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Load obs data for all 3 windows ──────────────────────
    data = {}
    for window in ['3mo', '6mo', '12mo']:
        path = BASE_PATH / window / f"lymphoma_{window}_obs_dropped.csv"
        try:
            data[window] = pd.read_csv(path, low_memory=False)
            # Normalize column names (lymphoma data ships lowercase originally)
            data[window].columns = data[window].columns.str.upper()
            print(f"Loaded {window}: {data[window].shape[0]:,} rows")
        except FileNotFoundError as e:
            print(f"WARNING {window}: {e}")
            data[window] = None

    # ── Load clean matrices (patient list + labels) ───────────
    clean_path = SCRIPT_DIR / 'results' / 'Cleanup_Finalresults'
    clean_matrices = {}
    for window in ['3mo', '6mo', '12mo']:
        path = clean_path / window / f"feature_matrix_clean_{window}.csv"
        clean_matrices[window] = pd.read_csv(path, index_col=0, usecols=['PATIENT_GUID', 'LABEL'])

    # ── Process each window ───────────────────────────────────
    text_features = {}
    emb_features = {}
    bert_features = {}

    for window in ['3mo', '6mo', '12mo']:
        if data[window] is None:
            continue

        # (a) Text keywords
        tf = build_text_keywords(data[window], window)
        text_features[window] = tf

        out_path = TEXT_RESULTS / window / f"text_features_{window}.csv"
        tf.to_csv(out_path)
        print(f"  Saved: {out_path}")

        clean_fm = clean_matrices[window]
        merged = tf.join(clean_fm[['LABEL']], how='inner')
        if len(merged) > 0 and 'LABEL' in merged.columns:
            corr = merged.drop(columns=['LABEL']).corrwith(merged['LABEL']).abs().sort_values(ascending=False)
            print(f"\n  Top text features by label correlation ({window}):")
            for i, (feat, c) in enumerate(corr.head(15).items()):
                print(f"    {i+1:2d}. {feat}: {c:.4f}")

        # (b) TF-IDF embeddings
        emb = build_tfidf_embeddings(data[window], clean_matrices[window], window)
        emb_features[window] = emb

        out_path = EMB_RESULTS / window / f"text_embeddings_{window}.csv"
        emb.to_csv(out_path)
        print(f"  Saved: {out_path}")

        # (c) BERT embeddings (try/except so failure does not kill pipeline)
        try:
            bert = build_bert_embeddings(data[window], clean_matrices[window], window)
            bert_features[window] = bert

            out_path = BERT_RESULTS / window / f"bert_embeddings_{window}.csv"
            bert.to_csv(out_path)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"\n  WARNING: BERT embeddings failed for {window}: {e}")
            print(f"  Continuing without BERT for this window.\n")

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  LYMPHOMA TEXT FEATURE PIPELINE COMPLETE")
    print(f"{'═'*70}")

    for window in ['3mo', '6mo', '12mo']:
        parts = []
        if window in text_features:
            parts.append(f"{text_features[window].shape[1]} keywords")
        if window in emb_features:
            parts.append(f"{emb_features[window].shape[1]} TF-IDF dims")
        if window in bert_features:
            parts.append(f"{bert_features[window].shape[1]} BERT dims")
        if parts:
            n_patients = text_features.get(window, emb_features.get(window, bert_features.get(window))).shape[0]
            print(f"  {window}: {n_patients:,} patients  |  {' + '.join(parts)}")

    # TF-IDF vs BERT comparison (3mo)
    tfidf_path = EMB_RESULTS / '3mo' / 'text_embeddings_3mo.csv'
    bert_path = BERT_RESULTS / '3mo' / 'bert_embeddings_3mo.csv'
    if tfidf_path.exists() and bert_path.exists():
        tfidf_df = pd.read_csv(tfidf_path, index_col=0)
        bert_df = pd.read_csv(bert_path, index_col=0)
        labels = clean_matrices['3mo']
        tfidf_corr = tfidf_df.join(labels).drop(columns=['LABEL']).corrwith(labels['LABEL']).abs().max()
        bert_corr = bert_df.join(labels).drop(columns=['LABEL']).corrwith(labels['LABEL']).abs().max()
        print(f"\n  TF-IDF vs BERT correlation comparison (3mo):")
        print(f"    TF-IDF best dim correlation: {tfidf_corr:.4f}")
        print(f"    BERT best dim correlation:   {bert_corr:.4f}")
        print(f"    Winner: {'BERT' if bert_corr > tfidf_corr else 'TF-IDF'}")

    print(f"\n  Output directories:")
    print(f"    Keywords:   {TEXT_RESULTS}")
    print(f"    TF-IDF:     {EMB_RESULTS}")
    print(f"    BERT:       {BERT_RESULTS}")
    print(f"  Next: Merge with clean matrices and run modeling.")
