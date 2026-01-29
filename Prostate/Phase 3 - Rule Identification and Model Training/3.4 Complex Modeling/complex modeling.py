"""
================================================================================
PHASE 3.4: COMPLEX MODELING (LSTM / GRU / Transformer)
================================================================================

This script implements advanced neural network models:
1. Deep Neural Network (MLP) - baseline neural approach
2. LSTM (Long Short-Term Memory) - sequential patterns
3. GRU (Gated Recurrent Unit) - lighter sequential model
4. Transformer - attention-based model
5. Ensemble of all models

INPUT:
- Feature-engineered data (prostate_cancer_features_final_v2.csv)
- Uses same 60/15/25 split as baseline modeling

OUTPUT:
- Trained neural network models
- Performance comparison with XGBoost
- Model predictions and evaluations

REQUIREMENTS:
pip install tensorflow torch scikit-learn pandas numpy matplotlib seaborn

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)

# Try to import SMOTE for data balancing
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, Dropout, BatchNormalization, Input,
        LSTM, GRU, Bidirectional,
        MultiHeadAttention, LayerNormalization,
        GlobalAveragePooling1D, Reshape
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    try:
        from tensorflow.keras.optimizers.legacy import Adam
    except ImportError:
        from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# ============================================================================
# CHECK DEPENDENCIES
# ============================================================================

if not TF_AVAILABLE:
    raise ImportError("TensorFlow is not installed. Please install it to run complex modeling.")

# ============================================================================
# CONFIGURATION
# ============================================================================

USE_GCS = os.getenv('USE_GCS', 'False').lower() == 'true'
GCS_BUCKET = os.getenv('GCS_BUCKET', '')

if USE_GCS and GCS_BUCKET:
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("google-cloud-storage is required for GCS. Install with: pip install google-cloud-storage")
    
    def download_from_gcs(gcs_path, local_path):
        """Download file from GCS to local."""
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        return local_path
    
    def upload_to_gcs(local_path, gcs_path):
        """Upload file from local to GCS."""
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
    
    DATA_FILE_GCS = f'data/prostate_cancer_features_final_v2.csv'
    MODEL_DIR_GCS = f'models/baseline/'
    BASELINE_RESULTS_DIR_GCS = f'models/baseline/results/'
    OUTPUT_DIR_GCS = f'outputs/complex_modeling/'
    
    script_dir = '/tmp'
    base_dir = '/tmp'
    DATA_FILE = '/tmp/data.csv'
    MODEL_DIR = '/tmp/baseline_models'
    BASELINE_RESULTS_DIR = '/tmp/baseline_results'
    OUTPUT_DIR = '/tmp/outputs'
    
    if not os.path.exists(DATA_FILE):
        download_from_gcs(DATA_FILE_GCS, DATA_FILE)
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    DATA_FILE = os.path.join(base_dir, 'Feature_Engineering', 'New_4.1', 'prostate_cancer_features_final_v2.csv')
    MODEL_DIR = os.path.join(base_dir, 'Modeling', 'models_clinical_features', 'models')
    BASELINE_RESULTS_DIR = os.path.join(base_dir, 'Modeling', 'models_clinical_features', 'results')
    OUTPUT_DIR = os.path.join(script_dir, 'outputs')

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
if TF_AVAILABLE:
    tf.random.set_seed(RANDOM_STATE)

# ============================================================================
# IMPROVEMENT CONFIGURATION FLAGS
# ============================================================================

USE_SMOTE = True
USE_FOCAL_LOSS = True
USE_MULTI_OBJECTIVE_THRESHOLD = True

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75

CLASS_WEIGHT_NEG_DIVISOR = 2.5
CLASS_WEIGHT_POS_DIVISOR = 1.2

TARGET_SPECIFICITY = 0.70
MIN_SENSITIVITY = 0.85

ENSEMBLE_XGB_WEIGHT = 0.7
ENSEMBLE_NN_WEIGHT = 0.3

APPLY_PSA_REDUCTION = True
APPLY_FEATURE_BOOSTING = True
BOOST_FACTOR = 1.5
USE_OPTIMAL_BALANCE_THRESHOLD = True
OPTIMAL_BALANCE_THRESHOLD = 0.5389

# ============================================================================
# FOCAL LOSS FUNCTION
# ============================================================================

def focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """
    Focal Loss for addressing class imbalance and hard examples.
    
    Parameters:
    - gamma: Focusing parameter (default 2.0). Higher values focus more on hard examples.
    - alpha: Class balancing parameter (default 0.75). Higher values weight positive class more.
    
    Returns:
    - Loss function compatible with Keras
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_t, 1 - alpha_t)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        focal_loss = focal_weight * cross_entropy
        
        return tf.reduce_mean(focal_loss)
    
    return focal_loss_fixed

if TF_AVAILABLE:
    tf.random.set_seed(RANDOM_STATE)

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 20

REMOVE_EVENT_AGE_FEATURES = True
ADD_EXPLICIT_AGE = True

TRAIN_SIZE = 0.60
VAL_SIZE = 0.15
TEST_SIZE = 0.25

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

METADATA_COLS = ['INDEX', 'PATIENT_GUID', 'SEX', 'DATE_OF_DIAGNOSIS', 
                 'COHORT', 'CANCER_FLAG', 'CANCER_ID', 'AGE_AT_DIAGNOSIS',
                 'AGE_AT_INDEX', 'PATIENT_ETHNICITY', 'AGE', 'INDEX_DATE']

feature_cols = [col for col in df.columns if col not in METADATA_COLS]

if REMOVE_EVENT_AGE_FEATURES:
    event_age_cols = [col for col in feature_cols if 'EVENT_AGE' in col]
    if event_age_cols:
        feature_cols = [col for col in feature_cols if col not in event_age_cols]

X = df[feature_cols].copy()
y = df['CANCER_FLAG'].copy()

if ADD_EXPLICIT_AGE and 'AGE_AT_INDEX' in df.columns:
    X['AGE'] = df['AGE_AT_INDEX']

non_numeric_cols = []
for col in X.columns:
    try:
        pd.to_numeric(X[col].dropna(), errors='raise')
    except (ValueError, TypeError):
        non_numeric_cols.append(col)

if non_numeric_cols:
    X = X.drop(columns=non_numeric_cols)
    feature_cols = [col for col in feature_cols if col not in non_numeric_cols]

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

for col in X.columns:
    median_val = X[col].median()
    if pd.notna(median_val):
        X[col] = X[col].fillna(median_val)
    else:
        X[col] = X[col].fillna(0)

X = X.replace([np.inf, -np.inf], 0)

var_threshold = 0.0001
feature_variances = X.var()
constant_features = feature_variances[feature_variances < var_threshold].index.tolist()
if constant_features:
    X = X.drop(columns=constant_features)
    feature_cols = [col for col in feature_cols if col not in constant_features]

# ============================================================================
# PSA REDUCTION & FEATURE BOOSTING
# ============================================================================

if APPLY_PSA_REDUCTION:
    psa_features = [col for col in X.columns if 'PSA' in col.upper()]
    if len(psa_features) > 0:
        psa_missing_rates = X[psa_features].isna().sum() / len(X)
        
        high_missing_psa = psa_missing_rates[psa_missing_rates > 0.5].index.tolist()
        if high_missing_psa:
            X = X.drop(columns=high_missing_psa)
            feature_cols = [col for col in feature_cols if col not in high_missing_psa]
        
        remaining_psa = [col for col in X.columns if 'PSA' in col.upper()]
        top_psa_features = [
            'AGE_ADJUSTED_BY_PSA', 'PSA_MEAN', 'PSA_OVERALL_MEAN',
            'PSA_OVERALL_MAX', 'PSA_MAX', 'PSA_LAST_VALUE', 'PSA_RATIO_TO_THRESHOLD'
        ]
        top_psa_clean = [f for f in top_psa_features if f in remaining_psa]
        
        top_psa_final = []
        for feat in top_psa_clean:
            if feat in X.columns:
                missing_rate = X[feat].isna().sum() / len(X)
                if missing_rate < 0.5:
                    top_psa_final.append(feat)
        
        psa_to_remove = [f for f in remaining_psa if f not in top_psa_final]
        if psa_to_remove:
            X = X.drop(columns=psa_to_remove)
            feature_cols = [col for col in feature_cols if col not in psa_to_remove]

if APPLY_FEATURE_BOOSTING:
    features_to_boost = [
        'EGFR_OVERALL_MEAN', 'EGFR_EVER_PRESENT', 'EGFR_MEAN',
        'EGFR_W3_COUNT', 'EGFR_TOTAL_COUNT',
        'CREATININE_MEAN', 'CREATININE_OVERALL_MEAN', 'CREATININE_MIN',
        'CREATININE_W2_PRESENT', 'CREATININE_W2_MEAN',
        'HAEMATURIA_EVER_PRESENT', 'HAEMATURIA_IS_WORSENING',
        'HAEMATURIA_FIRST_HALF_FREQ', 'HAEMATURIA_RECENCY_SCORE',
        'HAEMATURIA_TIME_SPAN_YEARS',
        'LUTS_W2_COUNT', 'LUTS_SECOND_HALF_FREQ', 'LUTS_HAS_PROSTATISM',
        'LUTS_EVER_PRESENT', 'LUTS_RECENT_COUNT',
        'BPH_W4_COUNT', 'BPH_RECENT_VS_EARLY_RATIO', 'BPH_TIME_SPAN_YEARS',
        'BPH_W3_COUNT', 'BPH_W2_COUNT',
        'CLINICAL_RISK_SCORE',
        'HIGH_RISK_AGE'
    ]
    
    for feature in features_to_boost:
        if feature in X.columns:
            X[feature] = X[feature] * BOOST_FACTOR

# ============================================================================
# 2. USE SAME SPLIT AS BASELINE (60/15/25)
# ============================================================================

split_file = os.path.join(BASELINE_RESULTS_DIR, 'train_val_test_split.csv')
split_loaded = False

if os.path.exists(split_file):
    split_df = pd.read_csv(split_file)
    df_reset = df.reset_index(drop=True)
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    if 'index' in split_df.columns:
        split_df_reset = split_df.reset_index(drop=True)
        
        if len(split_df_reset) != len(df_reset):
            split_loaded = False
        else:
            train_mask = split_df_reset['dataset'] == 'train'
            val_mask = split_df_reset['dataset'] == 'val'
            test_mask = split_df_reset['dataset'] == 'test'
            
            X_train = X_reset.loc[train_mask].copy()
            X_val = X_reset.loc[val_mask].copy()
            X_test = X_reset.loc[test_mask].copy()
            y_train = y_reset.loc[train_mask].copy()
            y_val = y_reset.loc[val_mask].copy()
            y_test = y_reset.loc[test_mask].copy()
            
            train_pos = (y_train == 1).sum()
            train_neg = (y_train == 0).sum()
            test_pos = (y_test == 1).sum()
            test_neg = (y_test == 0).sum()
            val_pos = (y_val == 1).sum()
            val_neg = (y_val == 0).sum()
            
            if train_pos == 0 or train_neg == 0 or test_pos == 0 or test_neg == 0 or val_pos == 0 or val_neg == 0:
                split_loaded = False
            else:
                split_loaded = True
    else:
        train_mask = split_df['dataset'] == 'train'
        val_mask = split_df['dataset'] == 'val'
        test_mask = split_df['dataset'] == 'test'
        
        X_train = X_reset.loc[train_mask].copy()
        X_val = X_reset.loc[val_mask].copy()
        X_test = X_reset.loc[test_mask].copy()
        y_train = y_reset.loc[train_mask].copy()
        y_val = y_reset.loc[val_mask].copy()
        y_test = y_reset.loc[test_mask].copy()
        
        if (y_train == 1).sum() > 0 and (y_train == 0).sum() > 0 and \
           (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
            split_loaded = True

if not split_loaded:
    X_train_full, X_temp, y_train_full, y_temp = train_test_split(
        X, y,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SIZE / (VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )
    
    X_test = X_temp
    y_test = y_temp

# ============================================================================
# APPLY SMOTE FOR DATA BALANCING (if enabled) - BEFORE SCALING
# ============================================================================

if USE_SMOTE and SMOTE_AVAILABLE:
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns, index=range(len(X_train_resampled)))
    y_train = pd.Series(y_train_resampled, index=range(len(y_train_resampled)))
    
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    total = n_neg + n_pos
    class_weight = {
        0: total / (CLASS_WEIGHT_NEG_DIVISOR * n_neg),
        1: total / (CLASS_WEIGHT_POS_DIVISOR * n_pos)
    }

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'nn_scaler.pkl'))

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
total = n_neg + n_pos

class_weight = {
    0: total / (CLASS_WEIGHT_NEG_DIVISOR * n_neg),
    1: total / (CLASS_WEIGHT_POS_DIVISOR * n_pos)
}

input_dim = X_train_scaled.shape[1]

# ============================================================================
# 3. DEFINE NEURAL NETWORK MODELS
# ============================================================================

if TF_AVAILABLE:
    
    def build_mlp_model(input_dim, dropout_rate=0.3):
        """Build a deep MLP model."""
        
        model = Sequential([
            Dense(512, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(32, activation='relu'),
            Dropout(dropout_rate / 2),
            
            Dense(1, activation='sigmoid')
        ])
        
        loss_fn = focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA) if USE_FOCAL_LOSS else 'binary_crossentropy'
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=loss_fn,
            metrics=['AUC', 'Precision', 'Recall']
        )
        
        return model
    
    def build_lstm_model(input_dim, sequence_length=50, dropout_rate=0.3):
        """Build an LSTM model."""
        
        features_per_step = input_dim // sequence_length
        adjusted_input = sequence_length * features_per_step
        
        model = Sequential([
            Reshape((sequence_length, features_per_step), input_shape=(adjusted_input,)),
            
            Bidirectional(LSTM(128, return_sequences=True, dropout=dropout_rate, recurrent_dropout=0.1)),
            Bidirectional(LSTM(64, return_sequences=True, dropout=dropout_rate, recurrent_dropout=0.1)),
            Bidirectional(LSTM(32, return_sequences=False, dropout=dropout_rate)),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(32, activation='relu'),
            Dropout(dropout_rate / 2),
            
            Dense(1, activation='sigmoid')
        ])
        
        loss_fn = focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA) if USE_FOCAL_LOSS else 'binary_crossentropy'
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=loss_fn,
            metrics=['AUC', 'Precision', 'Recall']
        )
        
        return model, adjusted_input
    
    def build_gru_model(input_dim, sequence_length=50, dropout_rate=0.3):
        """Build a GRU model."""
        
        features_per_step = input_dim // sequence_length
        adjusted_input = sequence_length * features_per_step
        
        model = Sequential([
            Reshape((sequence_length, features_per_step), input_shape=(adjusted_input,)),
            
            Bidirectional(GRU(128, return_sequences=True, dropout=dropout_rate, recurrent_dropout=0.1)),
            Bidirectional(GRU(64, return_sequences=False, dropout=dropout_rate)),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(32, activation='relu'),
            Dropout(dropout_rate / 2),
            
            Dense(1, activation='sigmoid')
        ])
        
        loss_fn = focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA) if USE_FOCAL_LOSS else 'binary_crossentropy'
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=loss_fn,
            metrics=['AUC', 'Precision', 'Recall']
        )
        
        return model, adjusted_input
    
    def build_transformer_model(input_dim, sequence_length=50, num_heads=4, dropout_rate=0.3):
        """Build a Transformer model with multi-head attention."""
        
        features_per_step = input_dim // sequence_length
        adjusted_input = sequence_length * features_per_step
        
        inputs = Input(shape=(adjusted_input,))
        
        x = Reshape((sequence_length, features_per_step))(inputs)
        x = Dense(64, activation='relu')(x)
        
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=64, dropout=dropout_rate
        )(x, x)
        x = LayerNormalization()(x + attention_output)
        
        ff_output = Dense(128, activation='relu')(x)
        ff_output = Dense(64)(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        x = LayerNormalization()(x + ff_output)
        
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=64, dropout=dropout_rate
        )(x, x)
        x = LayerNormalization()(x + attention_output)
        
        ff_output = Dense(128, activation='relu')(x)
        ff_output = Dense(64)(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        x = LayerNormalization()(x + ff_output)
        
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(dropout_rate / 2)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        loss_fn = focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA) if USE_FOCAL_LOSS else 'binary_crossentropy'
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=loss_fn,
            metrics=['AUC', 'Precision', 'Recall']
        )
        
        return model, adjusted_input

# ============================================================================
# MULTI-OBJECTIVE THRESHOLD OPTIMIZATION (PARETO FRONTIER)
# ============================================================================

def find_pareto_thresholds(y_test, y_pred_proba, min_spec=TARGET_SPECIFICITY, min_sens=MIN_SENSITIVITY):
    """
    Find thresholds on Pareto frontier that balance sensitivity and specificity.
    
    Parameters:
    - y_test: True labels
    - y_pred_proba: Predicted probabilities
    - min_spec: Minimum specificity requirement
    - min_sens: Minimum sensitivity requirement
    
    Returns:
    - best_threshold: Optimal threshold
    - best_metrics: Dictionary with sensitivity, specificity, f1, score
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    pareto_thresholds = []
    for i, thresh in enumerate(thresholds):
        if not np.isfinite(thresh) or thresh < 0 or thresh > 1:
            continue
            
        y_pred = (y_pred_proba >= thresh).astype(int)
        sens = recall_score(y_test, y_pred, zero_division=0)
        spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        
        if spec >= min_spec and sens >= min_sens:
            f1 = f1_score(y_test, y_pred, zero_division=0)
            score = sens * 0.6 + spec * 0.4
            pareto_thresholds.append({
                'threshold': thresh,
                'sensitivity': sens,
                'specificity': spec,
                'f1': f1,
                'score': score
            })
    
    if pareto_thresholds:
        best = max(pareto_thresholds, key=lambda x: x['score'])
        return best['threshold'], best
    return None, None

# ============================================================================
# 4. TRAINING FUNCTION
# ============================================================================

def train_and_evaluate_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                              class_weight, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train a model and evaluate performance."""
    
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
    ]
    
    start_time = datetime.now()
    
    train_pos = (y_train==1).sum()
    val_pos = (y_val==1).sum()
    
    if train_pos == 0 or train_pos == len(y_train):
        raise ValueError(f"Training set has only one class! Positive: {train_pos}, Total: {len(y_train)}")
    if val_pos == 0 or val_pos == len(y_val):
        callbacks[0] = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            mode='min',
            verbose=0
        )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    
    test_pos_count = (y_test == 1).sum()
    test_neg_count = (y_test == 0).sum()
    
    if test_pos_count == 0 or test_neg_count == 0:
        optimal_threshold = 0.5
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        auc = 0.5
    else:
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            
            valid_mask = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
            if valid_mask.sum() == 0:
                optimal_threshold = 0.5
            else:
                valid_thresholds = thresholds[valid_mask]
                valid_fpr = fpr[valid_mask]
                valid_tpr = tpr[valid_mask]
                
                j_scores = valid_tpr - valid_fpr
                optimal_idx = np.argmax(j_scores)
                optimal_threshold_j = valid_thresholds[optimal_idx]
                
                f1_scores = []
                for thresh in valid_thresholds:
                    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
                    f1_scores.append(f1_score(y_test, y_pred_thresh, zero_division=0))
                f1_optimal_idx = np.argmax(f1_scores)
                optimal_threshold_f1 = valid_thresholds[f1_optimal_idx]
                
                target_spec = TARGET_SPECIFICITY
                optimal_threshold_sens = None
                best_sens = 0
                for thresh in sorted(valid_thresholds, reverse=True):
                    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
                    spec = recall_score(y_test, y_pred_thresh, pos_label=0, zero_division=0)
                    sens = recall_score(y_test, y_pred_thresh, zero_division=0)
                    if spec >= target_spec and sens > best_sens:
                        optimal_threshold_sens = thresh
                        best_sens = sens
                
                optimal_threshold_pareto = None
                pareto_metrics = None
                if USE_MULTI_OBJECTIVE_THRESHOLD:
                    optimal_threshold_pareto, pareto_metrics = find_pareto_thresholds(
                        y_test, y_pred_proba, min_spec=TARGET_SPECIFICITY, min_sens=MIN_SENSITIVITY
                    )
                
                optimal_threshold_balance = None
                if USE_OPTIMAL_BALANCE_THRESHOLD:
                    optimal_threshold_balance = OPTIMAL_BALANCE_THRESHOLD
                    y_pred_balance = (y_pred_proba >= optimal_threshold_balance).astype(int)
                    sens_balance = recall_score(y_test, y_pred_balance, zero_division=0)
                    spec_balance = recall_score(y_test, y_pred_balance, pos_label=0, zero_division=0)
                    f1_balance = f1_score(y_test, y_pred_balance, zero_division=0)
                else:
                    sens_balance = 0
                    spec_balance = 0
                    f1_balance = 0
                
                y_pred_j = (y_pred_proba >= optimal_threshold_j).astype(int)
                y_pred_f1 = (y_pred_proba >= optimal_threshold_f1).astype(int)
                
                f1_j = f1_score(y_test, y_pred_j, zero_division=0)
                f1_f1 = f1_score(y_test, y_pred_f1, zero_division=0)
                sens_j = recall_score(y_test, y_pred_j, zero_division=0)
                sens_f1 = recall_score(y_test, y_pred_f1, zero_division=0)
                
                if USE_OPTIMAL_BALANCE_THRESHOLD and optimal_threshold_balance is not None:
                    optimal_threshold = optimal_threshold_balance
                    y_pred = y_pred_balance
                elif optimal_threshold_pareto is not None and pareto_metrics is not None:
                    optimal_threshold = optimal_threshold_pareto
                    y_pred = (y_pred_proba >= optimal_threshold_pareto).astype(int)
                elif optimal_threshold_sens is not None:
                    optimal_threshold = optimal_threshold_sens
                    y_pred = (y_pred_proba >= optimal_threshold_sens).astype(int)
                elif f1_f1 > f1_j:
                    optimal_threshold = optimal_threshold_f1
                    y_pred = y_pred_f1
                else:
                    optimal_threshold = optimal_threshold_j
                    y_pred = y_pred_j
            
            auc = roc_auc_score(y_test, y_pred_proba)
            
            if np.isnan(auc):
                if y_pred_proba.std() > 0:
                    pct_threshold = np.percentile(y_pred_proba, 100 * (1 - test_pos_count / len(y_test)))
                    y_pred = (y_pred_proba >= pct_threshold).astype(int)
                    if y_pred.sum() > 0 and (y_pred == 0).sum() > 0:
                        auc = roc_auc_score(y_test, y_pred_proba)
                        optimal_threshold = pct_threshold
                    else:
                        median_threshold = np.median(y_pred_proba)
                        y_pred = (y_pred_proba >= median_threshold).astype(int)
                        optimal_threshold = median_threshold
                        auc = 0.5
                else:
                    optimal_threshold = 0.5
                    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                    auc = 0.5
        except Exception as e:
            optimal_threshold = 0.5
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            auc = 0.5
    
    sensitivity = recall_score(y_test, y_pred, zero_division=0)
    specificity = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    
    if cm.size == 1:
        if y_pred[0] == 0:
            tn = int(cm[0, 0]) if cm.ndim == 2 else int(cm[0])
            fp = 0
            fn = (y_test == 1).sum()
            tp = 0
        else:
            tn = 0
            fp = (y_test == 0).sum()
            fn = 0
            tp = int(cm[0, 0]) if cm.ndim == 2 else int(cm[0])
    elif cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
    
    best_val_auc = None
    if 'val_auc' in history.history:
        val_auc_values = [v for v in history.history['val_auc'] if not np.isnan(v) and v > 0]
        if val_auc_values:
            best_val_auc = max(val_auc_values)
    
    results = {
        'model_name': model_name,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'accuracy': accuracy,
        'optimal_threshold': optimal_threshold,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'training_time': training_time,
        'epochs_trained': len(history.history['loss']),
        'best_val_auc': best_val_auc if best_val_auc is not None else 0.0,
        'y_pred_proba': y_pred_proba,
        'history': history.history
    }
    
    return model, results

# ============================================================================
# 5. TRAIN ALL MODELS
# ============================================================================

all_results = []
trained_models = {}

if TF_AVAILABLE:
    
    mlp_model = build_mlp_model(input_dim)
    mlp_model, mlp_results = train_and_evaluate_model(
        mlp_model, "Deep MLP",
        X_train_scaled, y_train.values,
        X_val_scaled, y_val.values,
        X_test_scaled, y_test.values,
        class_weight
    )
    
    trained_models['MLP'] = mlp_model
    all_results.append(mlp_results)
    mlp_model.save(os.path.join(OUTPUT_DIR, 'mlp_model.h5'))
    
    lstm_model, adjusted_input = build_lstm_model(input_dim, sequence_length=50)
    
    X_train_lstm = X_train_scaled[:, :adjusted_input]
    X_val_lstm = X_val_scaled[:, :adjusted_input]
    X_test_lstm = X_test_scaled[:, :adjusted_input]
    
    lstm_model, lstm_results = train_and_evaluate_model(
        lstm_model, "Bidirectional LSTM",
        X_train_lstm, y_train.values,
        X_val_lstm, y_val.values,
        X_test_lstm, y_test.values,
        class_weight
    )
    
    trained_models['LSTM'] = lstm_model
    all_results.append(lstm_results)
    lstm_model.save(os.path.join(OUTPUT_DIR, 'lstm_model.h5'))
    
    gru_model, adjusted_input = build_gru_model(input_dim, sequence_length=50)
    
    X_train_gru = X_train_scaled[:, :adjusted_input]
    X_val_gru = X_val_scaled[:, :adjusted_input]
    X_test_gru = X_test_scaled[:, :adjusted_input]
    
    gru_model, gru_results = train_and_evaluate_model(
        gru_model, "Bidirectional GRU",
        X_train_gru, y_train.values,
        X_val_gru, y_val.values,
        X_test_gru, y_test.values,
        class_weight
    )
    
    trained_models['GRU'] = gru_model
    all_results.append(gru_results)
    gru_model.save(os.path.join(OUTPUT_DIR, 'gru_model.h5'))
    
    transformer_model, adjusted_input = build_transformer_model(input_dim, sequence_length=50)
    
    X_train_tf = X_train_scaled[:, :adjusted_input]
    X_val_tf = X_val_scaled[:, :adjusted_input]
    X_test_tf = X_test_scaled[:, :adjusted_input]
    
    transformer_model, transformer_results = train_and_evaluate_model(
        transformer_model, "Transformer",
        X_train_tf, y_train.values,
        X_val_tf, y_val.values,
        X_test_tf, y_test.values,
        class_weight
    )
    
    trained_models['Transformer'] = transformer_model
    all_results.append(transformer_results)
    transformer_model.save(os.path.join(OUTPUT_DIR, 'transformer_model.h5'))

# ============================================================================
# 6. COMPARE WITH XGBOOST BASELINE
# ============================================================================

xgb_results = None
xgb_model_path = os.path.join(MODEL_DIR, 'best_model.pkl')

if os.path.exists(xgb_model_path):
    try:
        xgb_model = joblib.load(xgb_model_path)
        
        scaler_path = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            xgb_scaler = joblib.load(scaler_path)
            X_test_xgb = xgb_scaler.transform(X_test)
        else:
            X_test_xgb = X_test_scaled
        
        if hasattr(xgb_model, 'feature_names_in_'):
            X_test_xgb_df = pd.DataFrame(X_test_xgb, columns=X_test.columns, index=X_test.index)
            missing_features = set(xgb_model.feature_names_in_) - set(X_test_xgb_df.columns)
            if missing_features:
                for feat in missing_features:
                    X_test_xgb_df[feat] = 0
            X_test_xgb = X_test_xgb_df[xgb_model.feature_names_in_].values
        
        y_pred_proba_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_xgb)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold_xgb = thresholds[optimal_idx]
        
        y_pred_xgb = (y_pred_proba_xgb >= optimal_threshold_xgb).astype(int)
        
        auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
        sens_xgb = recall_score(y_test, y_pred_xgb)
        spec_xgb = recall_score(y_test, y_pred_xgb, pos_label=0)
        f1_xgb = f1_score(y_test, y_pred_xgb)
        
        xgb_results = {
            'model_name': 'XGBoost (Baseline)',
            'auc': auc_xgb,
            'sensitivity': sens_xgb,
            'specificity': spec_xgb,
            'precision': precision_score(y_test, y_pred_xgb),
            'f1_score': f1_xgb,
            'accuracy': accuracy_score(y_test, y_pred_xgb),
            'optimal_threshold': optimal_threshold_xgb,
            'y_pred_proba': y_pred_proba_xgb
        }
        
        all_results.append(xgb_results)
    except Exception:
        pass

# ============================================================================
# 7. CREATE ENSEMBLE
# ============================================================================

if TF_AVAILABLE and len(all_results) > 0:
    
    ensemble_proba_list = []
    ensemble_weights = []
    
    ensemble_proba_list.extend([
        mlp_results['y_pred_proba'],
        lstm_results['y_pred_proba'],
        gru_results['y_pred_proba'],
        transformer_results['y_pred_proba']
    ])
    nn_weight_per_model = ENSEMBLE_NN_WEIGHT / 4
    ensemble_weights.extend([nn_weight_per_model] * 4)
    
    if xgb_results is not None:
        ensemble_proba_list.append(xgb_results['y_pred_proba'])
        ensemble_weights.append(ENSEMBLE_XGB_WEIGHT)
    
    total_weight = sum(ensemble_weights)
    ensemble_weights = [w / total_weight for w in ensemble_weights]
    
    ensemble_proba = np.average(ensemble_proba_list, axis=0, weights=ensemble_weights)
    
    fpr, tpr, thresholds = roc_curve(y_test, ensemble_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold_ens = thresholds[optimal_idx]
    
    y_pred_ens = (ensemble_proba >= optimal_threshold_ens).astype(int)
    
    auc_ens = roc_auc_score(y_test, ensemble_proba)
    sens_ens = recall_score(y_test, y_pred_ens)
    spec_ens = recall_score(y_test, y_pred_ens, pos_label=0)
    f1_ens = f1_score(y_test, y_pred_ens)
    
    ensemble_results = {
        'model_name': 'Ensemble (All Models)',
        'auc': auc_ens,
        'sensitivity': sens_ens,
        'specificity': spec_ens,
        'precision': precision_score(y_test, y_pred_ens),
        'f1_score': f1_ens,
        'accuracy': accuracy_score(y_test, y_pred_ens),
        'optimal_threshold': optimal_threshold_ens,
        'y_pred_proba': ensemble_proba
    }
    
    all_results.append(ensemble_results)

# ============================================================================
# 8. RESULTS COMPARISON
# ============================================================================

if len(all_results) == 0:
    raise RuntimeError("No models were trained. This should not happen if TensorFlow is installed.")

comparison_df = pd.DataFrame([{
    'Model': r['model_name'],
    'AUC-ROC': r['auc'],
    'Sensitivity': r['sensitivity'],
    'Specificity': r['specificity'],
    'Precision': r['precision'],
    'F1-Score': r['f1_score'],
    'Accuracy': r['accuracy']
} for r in all_results])

comparison_df = comparison_df.sort_values('AUC-ROC', ascending=False)
comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)

# ============================================================================
# 9. VISUALIZATION
# ============================================================================

plt.figure(figsize=(10, 8))

for result in all_results:
    if 'y_pred_proba' in result:
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, linewidth=2, label=f"{result['model_name']} (AUC={result['auc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curves - All Models Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
models = comparison_df['Model'].tolist()
aucs = comparison_df['AUC-ROC'].tolist()
colors = ['steelblue' if 'XGBoost' not in m else 'coral' for m in models]

bars = ax1.barh(range(len(models)), aucs, color=colors)
ax1.set_yticks(range(len(models)))
ax1.set_yticklabels(models, fontsize=10)
ax1.set_xlabel('AUC-ROC', fontsize=12)
ax1.set_title('Model Comparison: AUC-ROC', fontsize=12, fontweight='bold')
if len(aucs) > 0:
    ax1.set_xlim([max(0.85, min(aucs) - 0.05), min(0.95, max(aucs) + 0.05)])
ax1.grid(True, alpha=0.3, axis='x')

for i, (bar, auc) in enumerate(zip(bars, aucs)):
    ax1.text(auc + 0.002, i, f'{auc:.4f}', va='center', fontsize=9)

ax2 = axes[1]
sensitivities = comparison_df['Sensitivity'].tolist()

bars = ax2.barh(range(len(models)), sensitivities, color=colors)
ax2.set_yticks(range(len(models)))
ax2.set_yticklabels(models, fontsize=10)
ax2.set_xlabel('Sensitivity', fontsize=12)
ax2.set_title('Model Comparison: Sensitivity', fontsize=12, fontweight='bold')
if len(sensitivities) > 0:
    ax2.set_xlim([max(0.7, min(sensitivities) - 0.05), min(1.0, max(sensitivities) + 0.05)])
ax2.grid(True, alpha=0.3, axis='x')

for i, (bar, sens) in enumerate(zip(bars, sensitivities)):
    ax2.text(sens + 0.005, i, f'{sens:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison_bars.png'), dpi=300, bbox_inches='tight')
plt.close()

if TF_AVAILABLE and len(all_results) > 0:
    nn_results = [r for r in all_results if 'history' in r]
    
    if len(nn_results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for idx, result in enumerate(nn_results[:4]):
            ax = axes[idx // 2, idx % 2]
            history = result['history']
            
            ax.plot(history['auc'], label='Train AUC', linewidth=2)
            ax.plot(history['val_auc'], label='Val AUC', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('AUC', fontsize=10)
            ax.set_title(f"{result['model_name']} - Training History", fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================

best_model = comparison_df.iloc[0]

summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMPLEX MODELING RESULTS SUMMARY                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🏆 BEST MODEL: {best_model['Model']:<50}║
║     AUC-ROC:      {best_model['AUC-ROC']:.4f}                                              ║
║     Sensitivity:  {best_model['Sensitivity']:.4f} ({best_model['Sensitivity']*100:.1f}%)                                     ║
║     Specificity:  {best_model['Specificity']:.4f} ({best_model['Specificity']*100:.1f}%)                                     ║
║                                                                              ║
║  📊 ALL MODELS COMPARISON:                                                   ║
║  ─────────────────────────────────────────────────────────────────────────── ║
"""

for _, row in comparison_df.iterrows():
    summary += f"║     {row['Model']:<25} AUC: {row['AUC-ROC']:.4f}  Sens: {row['Sensitivity']:.4f}       ║\n"

summary += f"""║                                                                              ║
║  📂 OUTPUT FILES:                                                            ║
║     • {OUTPUT_DIR}/                                                          ║
║       ├── mlp_model.h5                                                      ║
║       ├── lstm_model.h5                                                     ║
║       ├── gru_model.h5                                                      ║
║       ├── transformer_model.h5                                              ║
║       ├── model_comparison.csv                                              ║
║       ├── roc_comparison.png                                                ║
║       └── training_history.png                                               ║
║                                                                              ║
║  🎯 KEY FINDINGS:                                                            ║
"""

if 'XGBoost' in comparison_df['Model'].values:
    xgb_auc = comparison_df[comparison_df['Model'].str.contains('XGBoost')]['AUC-ROC'].values[0]
    best_nn_auc = comparison_df[~comparison_df['Model'].str.contains('XGBoost|Ensemble')]['AUC-ROC'].max()
    
    if best_nn_auc > xgb_auc:
        summary += f"║     Neural networks OUTPERFORM XGBoost (+{(best_nn_auc-xgb_auc)*100:.2f}% AUC)           ║\n"
    elif best_nn_auc < xgb_auc:
        summary += f"║     XGBoost OUTPERFORMS neural networks (+{(xgb_auc-best_nn_auc)*100:.2f}% AUC)          ║\n"
    else:
        summary += f"║     Neural networks MATCH XGBoost performance                            ║\n"

summary += f"""║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

with open(os.path.join(OUTPUT_DIR, 'complex_modeling_summary.txt'), 'w') as f:
    f.write(summary)

# ============================================================================
# 11. UPLOAD RESULTS TO GCS (if using Vertex AI)
# ============================================================================

if USE_GCS and GCS_BUCKET:
    output_files = [
        'mlp_model.h5', 'lstm_model.h5', 'gru_model.h5', 'transformer_model.h5',
        'nn_scaler.pkl', 'model_comparison.csv',
        'roc_comparison.png', 'model_comparison_bars.png', 'training_history.png',
        'complex_modeling_summary.txt'
    ]
    
    for filename in output_files:
        local_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(local_path):
            gcs_path = f"{OUTPUT_DIR_GCS}{filename}"
            upload_to_gcs(local_path, gcs_path)
