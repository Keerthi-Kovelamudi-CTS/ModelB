"""
PROSTATE CANCER PREDICTION MODEL TRAINING
===========================================================================

This is a modified version that removes EVENT_AGE features (age proxies) to focus
on clinically relevant features like PSA values, lab values, and symptoms.

MODIFICATIONS:
- Removes all *_MEAN_EVENT_AGE, *_FIRST_EVENT_AGE, *_LAST_EVENT_AGE features
- Optionally adds explicit AGE feature if available
- Forces model to use actual test values and clinical markers
- Should result in PSA features becoming more important
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
from datetime import datetime
import joblib

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, f1_score, matthews_corrcoef, 
    precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths - use script directory as base
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
DATA_FILE = os.path.join(base_dir, 'Feature_Engineering', 'New_4.1', 'prostate_cancer_features_final.csv')
OUTPUT_DIR = os.path.join(script_dir, 'models_clinical_features')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

# Create output directories
for dir_path in [OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model parameters - 3-WAY SPLIT (Train/Validation/Test)
TRAIN_SIZE = 0.60
VAL_SIZE = 0.15
TEST_SIZE = 0.25
RANDOM_STATE = 42
CV_FOLDS = 5

# Feature filtering options
REMOVE_EVENT_AGE_FEATURES = True
ADD_EXPLICIT_AGE = True

# Verify split adds to 100%
assert abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) < 0.001, \
    "Split sizes must sum to 1.0"

# Metadata columns to exclude from modeling
METADATA_COLS = ['INDEX', 'PATIENT_GUID', 'SEX', 'DATE_OF_DIAGNOSIS', 
                 'COHORT', 'CANCER_FLAG', 'CANCER_ID', 'AGE_AT_DIAGNOSIS',
                 'AGE_AT_INDEX', 'PATIENT_ETHNICITY', 'INDEX_DATE']

# ============================================================================
# 1. LOAD DATA
# ============================================================================

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"ERROR: File not found: {DATA_FILE}")
    sys.exit(1)

# Check required columns
if 'CANCER_FLAG' not in df.columns:
    print("ERROR: CANCER_FLAG column not found!")
    sys.exit(1)

# ============================================================================
# 2. PREPARE FEATURES (WITH EVENT_AGE REMOVAL)
# ============================================================================

# Separate features and target
feature_cols = [col for col in df.columns if col not in METADATA_COLS]
X = df[feature_cols].copy()
y = df['CANCER_FLAG'].copy()

# ============================================================================
# REMOVE EVENT_AGE FEATURES (AGE PROXIES)
# ============================================================================

if REMOVE_EVENT_AGE_FEATURES:
    # Find all EVENT_AGE features
    event_age_features = [col for col in feature_cols if 
                         'MEAN_EVENT_AGE' in col or 
                         'FIRST_EVENT_AGE' in col or 
                         'LAST_EVENT_AGE' in col]
    
    # Remove them
    X = X.drop(columns=event_age_features, errors='ignore')
    feature_cols = [col for col in feature_cols if col not in event_age_features]

# ============================================================================
# ADD EXPLICIT AGE FEATURE (IF AVAILABLE)
# ============================================================================

if ADD_EXPLICIT_AGE:
    if 'AGE' in df.columns and 'AGE' not in METADATA_COLS:
        X['AGE'] = df['AGE'].copy()
        feature_cols.append('AGE')
    elif 'AGE_AT_INDEX' in df.columns:
        X['AGE'] = df['AGE_AT_INDEX'].copy()
        feature_cols.append('AGE')

# ============================================================================
# LEAKAGE VALIDATION
# ============================================================================

LEAKAGE_KEYWORDS = ['CANCER_ID', 'DATE_OF_DIAGNOSIS', 'AGE_AT_DIAGNOSIS', 
                     'INDEX_DATE', 'COHORT', 'DIAGNOSIS']
leakage_found = []
for col in feature_cols:
    for keyword in LEAKAGE_KEYWORDS:
        if keyword in col.upper():
            leakage_found.append(col)
            break

if leakage_found:
    print(f"ERROR: Found potential leakage columns in features:")
    for col in leakage_found:
        print(f"  - {col}")
    sys.exit(1)

# ============================================================================
# HANDLE MISSING VALUES AND DATA CLEANING
# ============================================================================

# Identify numeric vs non-numeric columns
numeric_cols = []
non_numeric_cols = []

for col in X.columns:
    try:
        pd.to_numeric(X[col].dropna(), errors='raise')
        numeric_cols.append(col)
    except (ValueError, TypeError):
        non_numeric_cols.append(col)

if non_numeric_cols:
    X = X.drop(columns=non_numeric_cols)
    feature_cols = [col for col in feature_cols if col not in non_numeric_cols]

# Convert all remaining columns to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Fill missing with median
X_filled = X.copy()
for col in X_filled.columns:
    median_val = X_filled[col].median()
    if pd.notna(median_val):
        X_filled[col] = X_filled[col].fillna(median_val)
    else:
        X_filled[col] = X_filled[col].fillna(0)

# Replace infinite values
inf_count = np.isinf(X_filled.select_dtypes(include=[np.number]).values).sum()
if inf_count > 0:
    X_filled = X_filled.replace([np.inf, -np.inf], np.nan)
    for col in X_filled.columns:
        median_val = X_filled[col].median()
        if pd.notna(median_val):
            X_filled[col] = X_filled[col].fillna(median_val)
        else:
            X_filled[col] = X_filled[col].fillna(0)

# Remove constant features
var_threshold = 0.0001
feature_variances = X_filled.select_dtypes(include=[np.number]).var()
constant_features = feature_variances[feature_variances < var_threshold].index.tolist()

if constant_features:
    X_filled = X_filled.drop(columns=constant_features)
    feature_cols = [col for col in feature_cols if col not in constant_features]

# ============================================================================
# 3. TRAIN/VALIDATION/TEST SPLIT
# ============================================================================

# Step 1: Split into Train (60%) and Temp (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_filled, y,
    test_size=(VAL_SIZE + TEST_SIZE),
    random_state=RANDOM_STATE,
    stratify=y
)

# Step 2: Split Temp into Val (15%) and Test (25%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
    random_state=RANDOM_STATE,
    stratify=y_temp
)

# Save split indices
split_info = pd.DataFrame({
    'index': df.index,
    'dataset': ['train'] * len(X_train) + ['val'] * len(X_val) + ['test'] * len(X_test)
})
split_info = split_info.iloc[:len(df)]
split_info.to_csv(os.path.join(RESULTS_DIR, 'train_val_test_split.csv'), index=False)

# ============================================================================
# 4. FEATURE SCALING
# ============================================================================

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=X_val.columns,
    index=X_val.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# ============================================================================
# 5. TRAIN MODELS
# ============================================================================

models = {}
feature_names = X_train.columns.tolist()

# Calculate class weights
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Logistic Regression
lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
lr.fit(X_train_scaled, y_train)
lr_train_pred = lr.predict_proba(X_train_scaled)[:, 1]
lr_val_pred = lr.predict_proba(X_val_scaled)[:, 1]
lr_test_pred = lr.predict_proba(X_test_scaled)[:, 1]

lr_train_auc = roc_auc_score(y_train, lr_train_pred)
lr_val_auc = roc_auc_score(y_val, lr_val_pred)
lr_test_auc = roc_auc_score(y_test, lr_test_pred)

cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), scoring='roc_auc')

models['Logistic Regression'] = {
    'model': lr,
    'train_auc': lr_train_auc,
    'val_auc': lr_val_auc,
    'test_auc': lr_test_auc,
    'test_pred': lr_test_pred,
    'cv_auc_mean': cv_scores.mean(),
    'cv_auc_std': cv_scores.std()
}

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_train_pred = rf.predict_proba(X_train)[:, 1]
rf_val_pred = rf.predict_proba(X_val)[:, 1]
rf_test_pred = rf.predict_proba(X_test)[:, 1]

rf_train_auc = roc_auc_score(y_train, rf_train_pred)
rf_val_auc = roc_auc_score(y_val, rf_val_pred)
rf_test_auc = roc_auc_score(y_test, rf_test_pred)

cv_scores = cross_val_score(rf, X_train, y_train, cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), scoring='roc_auc')

models['Random Forest'] = {
    'model': rf,
    'train_auc': rf_train_auc,
    'val_auc': rf_val_auc,
    'test_auc': rf_test_auc,
    'test_pred': rf_test_pred,
    'cv_auc_mean': cv_scores.mean(),
    'cv_auc_std': cv_scores.std()
}

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=20
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
xgb_train_pred = xgb_model.predict_proba(X_train)[:, 1]
xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]

xgb_train_auc = roc_auc_score(y_train, xgb_train_pred)
xgb_val_auc = roc_auc_score(y_val, xgb_val_pred)
xgb_test_auc = roc_auc_score(y_test, xgb_test_pred)

# Cross-validation model (without early stopping)
xgb_model_cv = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)
cv_scores = cross_val_score(
    xgb_model_cv, 
    X_train, y_train, 
    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), 
    scoring='roc_auc',
    n_jobs=-1
)

models['XGBoost'] = {
    'model': xgb_model,
    'train_auc': xgb_train_auc,
    'val_auc': xgb_val_auc,
    'test_auc': xgb_test_auc,
    'test_pred': xgb_test_pred,
    'cv_auc_mean': cv_scores.mean(),
    'cv_auc_std': cv_scores.std()
}

# CatBoost
cat_model = CatBoostClassifier(
    iterations=200,
    depth=6,
    class_weights=[1.0, scale_pos_weight],
    random_state=RANDOM_STATE,
    verbose=False,
    early_stopping_rounds=20
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
cat_train_pred = cat_model.predict_proba(X_train)[:, 1]
cat_val_pred = cat_model.predict_proba(X_val)[:, 1]
cat_test_pred = cat_model.predict_proba(X_test)[:, 1]

cat_train_auc = roc_auc_score(y_train, cat_train_pred)
cat_val_auc = roc_auc_score(y_val, cat_val_pred)
cat_test_auc = roc_auc_score(y_test, cat_test_pred)

cv_scores = cross_val_score(cat_model, X_train, y_train, cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), scoring='roc_auc')

models['CatBoost'] = {
    'model': cat_model,
    'train_auc': cat_train_auc,
    'val_auc': cat_val_auc,
    'test_auc': cat_test_auc,
    'test_pred': cat_test_pred,
    'cv_auc_mean': cv_scores.mean(),
    'cv_auc_std': cv_scores.std()
}

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================

comparison_df = pd.DataFrame({
    'Model': list(models.keys()),
    'Train_AUROC': [models[m]['train_auc'] for m in models.keys()],
    'Val_AUROC': [models[m]['val_auc'] for m in models.keys()],
    'Test_AUROC': [models[m]['test_auc'] for m in models.keys()],
    'CV_AUROC_Mean': [models[m]['cv_auc_mean'] for m in models.keys()],
    'CV_AUROC_Std': [models[m]['cv_auc_std'] for m in models.keys()],
    'Overfit_Gap': [models[m]['train_auc'] - models[m]['test_auc'] for m in models.keys()]
})

comparison_df = comparison_df.sort_values('Test_AUROC', ascending=False)
comparison_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)

best_model_name = comparison_df.iloc[0]['Model']
best_model_info = models[best_model_name]

# ============================================================================
# 7. DETAILED EVALUATION (BEST MODEL) - THRESHOLD SELECTION
# ============================================================================

best_model = best_model_info['model']
y_pred_proba = best_model_info['test_pred']

# ============================================================================
# MULTIPLE THRESHOLD SELECTION METHODS - COMPARISON
# ============================================================================

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Method 1: Youden's J
j_scores = tpr - fpr
optimal_idx_youden = np.argmax(j_scores)
optimal_threshold_youden = thresholds[optimal_idx_youden]

# Method 2: F1-optimized
f1_scores_list = []
for t in thresholds:
    y_pred_t = (y_pred_proba >= t).astype(int)
    f1_t = f1_score(y_test, y_pred_t)
    f1_scores_list.append(f1_t)
optimal_idx_f1 = np.argmax(f1_scores_list)
optimal_threshold_f1 = thresholds[optimal_idx_f1]

# Method 3: High Specificity (≥90% specificity target)
specificities_list = []
for t in thresholds:
    y_pred_t = (y_pred_proba >= t).astype(int)
    tn_t = ((y_test == 0) & (y_pred_t == 0)).sum()
    fp_t = ((y_test == 0) & (y_pred_t == 1)).sum()
    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
    specificities_list.append(spec_t)

high_spec_indices = [i for i, s in enumerate(specificities_list) if s >= 0.90]
if high_spec_indices:
    optimal_idx_high_spec = max(high_spec_indices)
    optimal_threshold_high_spec = thresholds[optimal_idx_high_spec]
    high_spec_achievable = True
else:
    optimal_idx_high_spec = optimal_idx_youden
    optimal_threshold_high_spec = optimal_threshold_youden
    high_spec_achievable = False

threshold_methods = {
    "Youden's J": {
        'threshold': optimal_threshold_youden,
        'description': "Balances Sensitivity and Specificity (maximizes J = TPR - FPR)"
    },
    "F1-Optimized": {
        'threshold': optimal_threshold_f1,
        'description': "Maximizes F1-Score (balances Precision and Recall)"
    },
    "High Specificity (≥90%)": {
        'threshold': optimal_threshold_high_spec,
        'description': "Prioritizes Specificity ≥90% (minimizes False Positives)" if high_spec_achievable else "Target not achievable, using Youden's J"
    }
}

# Evaluate all threshold methods
threshold_results = []

for method_name, method_info in threshold_methods.items():
    threshold = method_info['threshold']
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_val = f1_score(y_test, y_pred_binary)
    mcc_val = matthews_corrcoef(y_test, y_pred_binary)
    youden_j = sensitivity + specificity_val - 1
    
    threshold_results.append({
        'Method': method_name,
        'Threshold': threshold,
        'Sensitivity_TPR': sensitivity,
        'Specificity_TNR': specificity_val,
        'PPV_Precision': ppv,
        'NPV': npv,
        'Accuracy': accuracy,
        'F1_Score': f1_val,
        'MCC': mcc_val,
        'Youden_J': youden_j,
        'True_Positives': int(tp),
        'False_Positives': int(fp),
        'True_Negatives': int(tn),
        'False_Negatives': int(fn)
    })

threshold_comparison_df = pd.DataFrame(threshold_results)
threshold_comparison_df.to_csv(
    os.path.join(RESULTS_DIR, 'threshold_methods_comparison.csv'),
    index=False
)

# Use Youden's J as recommended threshold
best_overall = threshold_comparison_df[threshold_comparison_df['Method'] == "Youden's J"].iloc[0]
optimal_threshold = best_overall['Threshold']
y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_binary)
tn, fp, fn, tp = cm.ravel()
sensitivity = best_overall['Sensitivity_TPR']
specificity = best_overall['Specificity_TNR']
ppv = best_overall['PPV_Precision']
npv = best_overall['NPV']
accuracy = best_overall['Accuracy']
f1 = best_overall['F1_Score']
mcc = best_overall['MCC']

# Save detailed metrics
detailed_metrics = {
    'Model': best_model_name,
    'Threshold_Method': best_overall['Method'],
    'AUROC': best_model_info['test_auc'],
    'CV_AUROC_Mean': best_model_info['cv_auc_mean'],
    'CV_AUROC_Std':  best_model_info['cv_auc_std'],
    'Optimal_Threshold': optimal_threshold,
    'Sensitivity_TPR': sensitivity,
    'Specificity_TNR': specificity,
    'PPV_Precision': ppv,
    'NPV':  npv,
    'Accuracy': accuracy,
    'F1_Score': f1,
    'MCC': mcc,
    'True_Negatives': int(tn),
    'False_Positives': int(fp),
    'False_Negatives': int(fn),
    'True_Positives': int(tp)
}

pd.DataFrame([detailed_metrics]).to_csv(
    os.path.join(RESULTS_DIR, 'detailed_metrics.csv'),
    index=False
)

# ============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

best_model = models[best_model_name]['model']

if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    feature_importance_df['Importance_Normalized'] = (
        feature_importance_df['Importance'] / feature_importance_df['Importance'].max() * 100
    )
    
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    feature_importance_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False)
else:
    feature_importance = np.abs(best_model.coef_[0])
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    feature_importance_df['Importance_Normalized'] = (
        feature_importance_df['Importance'] / feature_importance_df['Importance'].max() * 100
    )
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    feature_importance_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False)

# ============================================================================
# 9. GENERATE VISUALIZATIONS
# ============================================================================

# Plot 1: ROC Curves (All Models)
plt.figure(figsize=(10, 8))
for model_name in models.keys():
    model_info = models[model_name]
    y_pred = model_info['test_pred']
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = model_info['test_auc']
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'roc_curves_all_models.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Confusion Matrix (Best Model)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', cbar=False,
            xticklabels=['No Cancer', 'Cancer'],
            yticklabels=['No Cancer', 'Cancer'])
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Feature Importance (Top 20)
plt.figure(figsize=(12, 10))
top_features = feature_importance_df.head(20)
plt.barh(range(len(top_features)), top_features['Importance_Normalized'])
plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=10)
plt.xlabel('Relative Importance (%)', fontsize=12)
plt.title(f'Top 20 Most Important Features - {best_model_name}', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance_top20.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

if 'Overfit_Gap' not in comparison_df.columns:
    comparison_df['Overfit_Gap'] = comparison_df['Train_AUROC'] - comparison_df['Test_AUROC']

models_sorted = comparison_df.sort_values('Test_AUROC')

ax1 = axes[0]
ax1.barh(range(len(models_sorted)), models_sorted['Test_AUROC'], color='steelblue')
ax1.set_yticks(range(len(models_sorted)))
ax1.set_yticklabels(models_sorted['Model'])
ax1.set_xlabel('Test AUROC', fontsize=12)
ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax1.set_xlim([0.5, 1.0])
ax1.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(models_sorted.iterrows()):
    ax1.text(row['Test_AUROC'] + 0.01, i, f"{row['Test_AUROC']:.3f}", 
             va='center', fontsize=10)

ax2 = axes[1]
ax2.barh(range(len(models_sorted)), models_sorted['Overfit_Gap'], color='coral')
ax2.set_yticks(range(len(models_sorted)))
ax2.set_yticklabels(models_sorted['Model'])
ax2.set_xlabel('Overfitting Gap (Train - Test AUROC)', fontsize=12)
ax2.set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Precision-Recall Curve (Best Model)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, linewidth=2, color='steelblue')
plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision (PPV)', fontsize=12)
plt.title(f'Precision-Recall Curve - {best_model_name}', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 10. SAVE MODELS
# ============================================================================

for model_name, model_info in models.items():
    model_filename = f"{model_name.lower().replace(' ', '_')}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model_info['model'], model_path)

# Save best model
best_model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
joblib.dump(best_model, best_model_path)

# Save model metadata
model_metadata = {
    'best_model': best_model_name,
    'best_test_auroc': best_model_info['test_auc'],
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'num_features': X_train.shape[1],
    'random_state': RANDOM_STATE,
    'test_size': TEST_SIZE,
    'cv_folds': CV_FOLDS,
    'remove_event_age_features': REMOVE_EVENT_AGE_FEATURES,
    'add_explicit_age': ADD_EXPLICIT_AGE
}

pd.DataFrame([model_metadata]).to_csv(
    os.path.join(MODELS_DIR, 'model_metadata.csv'),
    index=False
)

# ============================================================================
# 11. GENERATE CLINICAL RULES
# ============================================================================

top_10_features = feature_importance_df.head(10)['Feature'].tolist()

rules = []

# High-risk patients
high_risk_mask = y_pred_proba > 0.7
high_risk_actual = y_test[high_risk_mask]
high_risk_correct = (high_risk_actual == 1).sum()
high_risk_total = len(high_risk_actual)

if high_risk_total > 0:
    rules.append({
        'Rule': 'High Risk',
        'Condition': 'Predicted Probability > 0.7',
        'Patients': high_risk_total,
        'Cancers_Detected': high_risk_correct,
        'Precision': high_risk_correct / high_risk_total if high_risk_total > 0 else 0,
        'Description': 'Patients with >70% predicted cancer probability'
    })

# Low-risk patients
low_risk_mask = y_pred_proba < 0.3
low_risk_actual = y_test[low_risk_mask]
low_risk_correct = (low_risk_actual == 0).sum()
low_risk_total = len(low_risk_actual)

if low_risk_total > 0:
    rules.append({
        'Rule':  'Low Risk',
        'Condition': 'Predicted Probability < 0.3',
        'Patients': low_risk_total,
        'Cancers_Detected': low_risk_total - low_risk_correct,
        'Precision': low_risk_correct / low_risk_total if low_risk_total > 0 else 0,
        'Description': 'Patients with <30% predicted cancer probability'
    })

# PSA-based rules
psa_features = [f for f in top_10_features if 'PSA' in f.upper()]
if psa_features:
    psa_value_cols = [col for col in X_test.columns if 'PSA' in col.upper() and ('MEAN' in col.upper() or 'LAST' in col.upper() or 'MAX' in col.upper())]
    if psa_value_cols:
        psa_col = psa_value_cols[0]
        
        if psa_col in X_test.columns:
            high_psa_mask = X_test[psa_col] > 10
            high_psa_actual = y_test[high_psa_mask]
            high_psa_cancers = (high_psa_actual == 1).sum()
            high_psa_total = len(high_psa_actual)
            
            if high_psa_total > 0:
                rules.append({
                    'Rule': 'High PSA',
                    'Condition':  f'{psa_col} > 10 ng/mL',
                    'Patients': high_psa_total,
                    'Cancers_Detected': high_psa_cancers,
                    'Precision': high_psa_cancers / high_psa_total,
                    'Description': 'Significantly elevated PSA level'
                })

if rules:
    rules_df = pd.DataFrame(rules)
    rules_df.to_csv(os.path.join(RESULTS_DIR, 'clinical_rules.csv'), index=False)

# ============================================================================
# 11.5. GENERATE RISK SCORES WITH INTERPRETATION
# ============================================================================

def interpret_risk_score(probability):
    """Convert probability (0-1) to risk level and description."""
    if probability < 0.1:
        return "LOW", "Very Low Risk", "Continue routine screening"
    elif probability < 0.3:
        return "LOW", "Low Risk", "Continue routine screening"
    elif probability < 0.5:
        return "MODERATE", "Moderate Risk", "Consider closer monitoring"
    elif probability < 0.7:
        return "MODERATE-HIGH", "Moderate-High Risk", "Recommend further investigation"
    elif probability < 0.9:
        return "HIGH", "High Risk", "Urgent investigation recommended"
    else:
        return "VERY HIGH", "Very High Risk", "Immediate investigation required"

# Generate risk scores for test set
test_indices = X_test.index
if 'PATIENT_GUID' in df.columns:
    patient_guids = df.loc[test_indices, 'PATIENT_GUID'].values
else:
    patient_guids = test_indices

test_risk_scores = pd.DataFrame({
    'PATIENT_GUID': patient_guids,
    'PREDICTED_PROBABILITY': y_pred_proba,
    'ACTUAL_CANCER': y_test.values,
    'PREDICTED_CANCER': y_pred_binary
})

risk_interpretations = test_risk_scores['PREDICTED_PROBABILITY'].apply(interpret_risk_score)
test_risk_scores['RISK_LEVEL'] = [r[0] for r in risk_interpretations]
test_risk_scores['RISK_DESCRIPTION'] = [r[1] for r in risk_interpretations]
test_risk_scores['RECOMMENDATION'] = [r[2] for r in risk_interpretations]
test_risk_scores['RISK_SCORE_PERCENT'] = (test_risk_scores['PREDICTED_PROBABILITY'] * 100).round(1)

test_risk_scores = test_risk_scores.sort_values('PREDICTED_PROBABILITY', ascending=False)
test_risk_scores.to_csv(os.path.join(RESULTS_DIR, 'patient_risk_scores.csv'), index=False)
