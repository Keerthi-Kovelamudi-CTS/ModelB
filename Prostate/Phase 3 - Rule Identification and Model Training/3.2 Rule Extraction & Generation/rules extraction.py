"""
RULE EXTRACTION & MODEL EXPLAINABILITY
==================================================

PREDICTION TASK:
"Based on a patient's data up to INDEX_DATE (TODAY), what is their risk 
of being diagnosed with prostate cancer in the NEXT 12 months?"

This script:
1. Loads trained model and optimal threshold from Phase 3.1
2. Comprehensive SHAP analysis (global + local + visualizations)
3. Decision Tree surrogate model with visualization
4. LIME for local patient explanations (optional)
5. RuleFit for advanced rule extraction (optional)
6. Generates patient-level explanations with risk factors
7. Creates clinical prediction rules and summary document
8. Exports results in multiple formats (CSV + JSON)
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from rulefit import RuleFit
    RULEFIT_AVAILABLE = True
except ImportError:
    RULEFIT_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths - use script directory as base
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# Input paths (from Phase 3.1)
DATA_FILE = os.path.join(base_dir, 'Feature_Engineering', 'New_4.1', 'prostate_cancer_features_final.csv')
MODEL_DIR = os.path.join(script_dir, 'models_clinical_features', 'models')
RESULTS_DIR = os.path.join(script_dir, 'models_clinical_features', 'results')

# Output paths
OUTPUT_DIR = os.path.join(script_dir, 'models_clinical_features', 'rules_output')
SHAP_DIR = os.path.join(script_dir, 'models_clinical_features', 'shap_analysis')
RULES_DIR = os.path.join(script_dir, 'models_clinical_features', 'rules')

# Create output directories
for dir_path in [OUTPUT_DIR, SHAP_DIR, RULES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Configuration
RANDOM_STATE = 42
N_SHAP_SAMPLES = 1000
np.random.seed(RANDOM_STATE)

# Feature filtering options (must match training script)
REMOVE_EVENT_AGE_FEATURES = True
ADD_EXPLICIT_AGE = True

# ============================================================================
# 1. LOAD MODEL AND DATA
# ============================================================================

def load_optimal_threshold():
    """Load optimal threshold from Phase 3.1 results."""
    threshold_file = os.path.join(RESULTS_DIR, 'detailed_metrics.csv')
    if os.path.exists(threshold_file):
        metrics_df = pd.read_csv(threshold_file)
        if 'Optimal_Threshold' in metrics_df.columns:
            return metrics_df['Optimal_Threshold'].iloc[0]
    return 0.5

def load_model_and_data():
    """Load trained model and feature-engineered data."""
    
    # Load model metadata to determine model type
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.csv")
    model_type = None
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        if 'best_model' in metadata_df.columns:
            model_type = metadata_df['best_model'].iloc[0]
    
    # Load best model
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Check if model needs scaling (Logistic Regression)
    needs_scaling = (model_type == 'Logistic Regression')
    
    # Load scaler if needed
    scaler = None
    if needs_scaling:
        scaler_path = os.path.join(MODEL_DIR, "feature_scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
    
    # Load data
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    
    df = pd.read_csv(DATA_FILE)
    
    # Get feature columns (same as training)
    METADATA_COLS = ['INDEX', 'PATIENT_GUID', 'SEX', 'DATE_OF_DIAGNOSIS', 
                     'COHORT', 'CANCER_FLAG', 'CANCER_ID', 'AGE_AT_DIAGNOSIS',
                     'AGE_AT_INDEX', 'PATIENT_ETHNICITY', 'INDEX_DATE']
    
    feature_cols = [col for col in df.columns if col not in METADATA_COLS]
    X = df[feature_cols].copy()
    y = df['CANCER_FLAG'].copy()
    
    # Remove EVENT_AGE features
    if REMOVE_EVENT_AGE_FEATURES:
        event_age_features = [col for col in feature_cols if 
                             'MEAN_EVENT_AGE' in col or 
                             'FIRST_EVENT_AGE' in col or 
                             'LAST_EVENT_AGE' in col]
        
        X = X.drop(columns=event_age_features, errors='ignore')
        feature_cols = [col for col in feature_cols if col not in event_age_features]
    
    # Add explicit AGE feature
    if ADD_EXPLICIT_AGE:
        if 'AGE' in df.columns and 'AGE' not in METADATA_COLS:
            X['AGE'] = df['AGE'].copy()
            if 'AGE' not in feature_cols:
                feature_cols.append('AGE')
        elif 'AGE_AT_INDEX' in df.columns:
            X['AGE'] = df['AGE_AT_INDEX'].copy()
            if 'AGE' not in feature_cols:
                feature_cols.append('AGE')
    
    # Preprocess (same as training script)
    # Handle non-numeric columns
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
    
    return model, X_filled, y, feature_cols, scaler, model_type

# ============================================================================
# 2. SHAP-BASED RULE EXTRACTION
# ============================================================================

def extract_shap_rules(model, X, y, feature_names, model_type=None, scaler=None, n_samples=1000):
    """Extract rules based on SHAP feature importance and generate visualizations."""
    
    if not SHAP_AVAILABLE:
        return None, None, None
    
    # Prepare data for SHAP
    X_processed = X.copy()
    if scaler is not None:
        X_processed = pd.DataFrame(
            scaler.transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
    
    # Sample for faster computation
    if len(X_processed) > n_samples:
        sample_idx = np.random.choice(len(X_processed), min(n_samples, len(X_processed)), replace=False)
        X_shap = X_processed.iloc[sample_idx]
    else:
        X_shap = X_processed
    
    try:
        # Choose appropriate SHAP explainer based on model type
        if model_type == 'Logistic Regression':
            try:
                explainer = shap.LinearExplainer(model, X_shap)
                shap_values = explainer.shap_values(X_shap)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            except:
                explainer = shap.KernelExplainer(model.predict_proba, X_shap.sample(min(100, len(X_shap))))
                shap_values = explainer.shap_values(X_shap.iloc[:min(100, len(X_shap))])[1]
                X_shap = X_shap.iloc[:min(100, len(X_shap))]
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        # Ensure shap_values is 2D
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(-1, 1)
        
        # SHAP Visualizations
        plt.figure(figsize=(14, 12))
        shap.summary_plot(
            shap_values,
            X_shap,
            feature_names=feature_names,
            show=False,
            max_display=30,
            plot_size=(14, 12)
        )
        plt.title('SHAP Summary Plot - Feature Impact on Cancer Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, 'shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values,
            X_shap,
            feature_names=feature_names,
            plot_type='bar',
            show=False,
            max_display=30
        )
        plt.title('Mean |SHAP Value| - Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, 'shap_summary_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Dependence plots for top 6 features
        shap_importance_prelim = pd.DataFrame({
            'feature': feature_names[:len(shap_values[0])],
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        top_features_for_dependence = shap_importance_prelim.head(6)['feature'].tolist()
        
        if len(top_features_for_dependence) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for idx, feature in enumerate(top_features_for_dependence):
                if feature in X_shap.columns:
                    try:
                        feature_idx = list(X_shap.columns).index(feature)
                        shap.dependence_plot(
                            feature_idx,
                            shap_values,
                            X_shap,
                            feature_names=feature_names,
                            ax=axes[idx],
                            show=False
                        )
                        axes[idx].set_title(f'{feature}', fontsize=11)
                    except:
                        axes[idx].text(0.5, 0.5, f'Could not plot\n{feature}', 
                                      ha='center', va='center', transform=axes[idx].transAxes)
            
            plt.suptitle('SHAP Dependence Plots - Top 6 Features', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(SHAP_DIR, 'shap_dependence_plots.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Get mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        if len(mean_shap) != len(feature_names):
            min_len = min(len(mean_shap), len(feature_names))
            mean_shap = mean_shap[:min_len]
            feature_names_subset = feature_names[:min_len]
        else:
            feature_names_subset = feature_names
        
        shap_importance = pd.DataFrame({
            'Feature': feature_names_subset,
            'Mean_Abs_SHAP': mean_shap,
            'Mean_SHAP': shap_values.mean(axis=0),
            'Std_SHAP': shap_values.std(axis=0)
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        
        shap_importance['Direction'] = shap_importance['Mean_SHAP'].apply(
            lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk'
        )
        
        shap_importance['Importance_Pct'] = (
            shap_importance['Mean_Abs_SHAP'] / shap_importance['Mean_Abs_SHAP'].sum() * 100
        )
        
        shap_importance.to_csv(os.path.join(SHAP_DIR, 'shap_feature_importance.csv'), index=False)
        
        # Save SHAP values
        np.save(os.path.join(SHAP_DIR, 'shap_values.npy'), shap_values)
        X_shap.to_csv(os.path.join(SHAP_DIR, 'shap_samples.csv'), index=False)
        
        # Extract top features and their typical thresholds
        rules = []
        top_features = shap_importance.head(20)
        
        for _, row in top_features.iterrows():
            feat_name = row['Feature']
            
            if feat_name not in X.columns:
                continue
                
            feat_values = X[feat_name].dropna()
            
            if len(feat_values) > 0:
                q75 = feat_values.quantile(0.75)
                high_risk_mask = X[feat_name] > q75
                if high_risk_mask.sum() > 0:
                    high_risk_rate = y[high_risk_mask].mean()
                    if high_risk_rate > 0.3:
                        rules.append({
                            'rule_type': 'SHAP',
                            'rule': f"IF {feat_name} > {q75:.2f} THEN High Risk",
                            'feature': feat_name,
                            'threshold': q75,
                            'direction': '>',
                            'risk_score': high_risk_rate,
                            'support': high_risk_mask.sum(),
                            'shap_importance': row['Mean_Abs_SHAP']
                        })
        
        return pd.DataFrame(rules), shap_values, X_shap
        
    except Exception as e:
        return None, None, None

# ============================================================================
# 3. DECISION TREE RULE EXTRACTION
# ============================================================================

def extract_decision_tree_rules(X, y, feature_names, max_depth=5, min_samples_split=50):
    """Extract rules from decision tree with visualization."""
    
    # Train decision tree
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42
    )
    dt.fit(X.fillna(0), y)
    
    # Extract rules from tree
    tree_rules = export_text(dt, feature_names=feature_names, max_depth=max_depth)
    
    # Save text rules
    with open(os.path.join(RULES_DIR, 'decision_tree_rules.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("PROSTATE CANCER PREDICTION - DECISION TREE RULES\n")
        f.write("="*80 + "\n\n")
        f.write(f"Tree Depth: {max_depth}\n")
        f.write(f"Number of Leaves (Rules): {dt.get_n_leaves()}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("="*80 + "\n")
        f.write("RULES:\n")
        f.write("="*80 + "\n\n")
        f.write(tree_rules)
    
    # Visualize decision tree
    plt.figure(figsize=(30, 15))
    plot_tree(
        dt,
        feature_names=feature_names,
        class_names=['Low Risk', 'High Risk'],
        filled=True,
        rounded=True,
        fontsize=8,
        proportion=True
    )
    plt.title(f'Decision Tree Model (Depth={max_depth})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RULES_DIR, 'decision_tree_visualization.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Convert to structured rules
    rules = []
    tree = dt.tree_
    
    def extract_node_rules(node_id, path="", depth=0):
        if depth > max_depth:
            return
        
        if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf
            samples = tree.n_node_samples[node_id]
            values = tree.value[node_id][0]
            prob_cancer = values[1] / values.sum() if values.sum() > 0 else 0
            
            if prob_cancer > 0.3:
                rules.append({
                    'rule_type': 'Decision_Tree',
                    'rule': path,
                    'risk_score': prob_cancer,
                    'support': samples,
                    'precision': prob_cancer
                })
        else:
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature_{feature_idx}"
            
            left_path = f"{path} AND {feature_name} <= {threshold:.2f}" if path else f"{feature_name} <= {threshold:.2f}"
            extract_node_rules(tree.children_left[node_id], left_path, depth+1)
            
            right_path = f"{path} AND {feature_name} > {threshold:.2f}" if path else f"{feature_name} > {threshold:.2f}"
            extract_node_rules(tree.children_right[node_id], right_path, depth+1)
    
    extract_node_rules(0)
    
    return pd.DataFrame(rules), dt

# ============================================================================
# 4. RULEFIT RULE EXTRACTION
# ============================================================================

def extract_rulefit_rules(X, y, feature_names):
    """Extract rules using RuleFit."""
    
    if not RULEFIT_AVAILABLE:
        return None
    
    try:
        rulefit = RuleFit(
            tree_size=4,
            sample_fract='default',
            max_rules=2000,
            memory_par=0.01,
            tree_generator=None,
            rfmode='classify',
            lin_standardise=True,
            lin_trim_quantile=0.025,
            model_type='r'
        )
        
        rulefit.fit(X.fillna(0).values, y.values, feature_names=feature_names)
        
        rules_df = rulefit.get_rules()
        rules_df = rules_df[rules_df['coef'] != 0].sort_values('importance', ascending=False)
        
        high_risk_rules = rules_df[rules_df['coef'] > 0].head(50).copy()
        
        if len(high_risk_rules) == 0:
            return None
        
        evaluated_rules = []
        for _, row in high_risk_rules.iterrows():
            rule_str = row['rule']
            
            evaluated_rules.append({
                'rule_type': 'RuleFit',
                'rule': rule_str,
                'coefficient': row['coef'],
                'importance': row['importance'],
                'support': row.get('support', 0),
                'risk_score': 0.5
            })
        
        return pd.DataFrame(evaluated_rules)
        
    except Exception as e:
        return None

# ============================================================================
# 5. LIME-BASED LOCAL RULES
# ============================================================================

def extract_lime_rules(model, X, y, feature_names, scaler=None, n_examples=100):
    """Extract local rules using LIME."""
    
    if not LIME_AVAILABLE:
        return None
    
    X_processed = X.fillna(0).copy()
    
    if scaler is not None:
        def predict_proba_wrapper(X_input):
            X_scaled = scaler.transform(X_input)
            return model.predict_proba(X_scaled)
        predict_fn = predict_proba_wrapper
    else:
        predict_fn = model.predict_proba
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_processed.values,
        feature_names=feature_names,
        class_names=['No Cancer', 'Cancer'],
        mode='classification'
    )
    
    try:
        y_pred_proba = predict_fn(X_processed.values)[:, 1]
        high_risk_idx = np.where(y_pred_proba > 0.7)[0]
        
        if len(high_risk_idx) > n_examples:
            sample_idx = np.random.choice(high_risk_idx, n_examples, replace=False)
        else:
            sample_idx = high_risk_idx if len(high_risk_idx) > 0 else []
        
        if len(sample_idx) == 0:
            return None
        
        rules = []
        for idx in sample_idx:
            try:
                exp = explainer.explain_instance(
                    X_processed.iloc[idx].values,
                    predict_fn,
                    num_features=10,
                    top_labels=1
                )
                
                explanation_list = exp.as_list(label=1)
                
                rule_parts = []
                for feat, weight in explanation_list[:5]:
                    if weight > 0:
                        rule_parts.append(feat)
                
                if rule_parts:
                    rules.append({
                        'rule_type': 'LIME_Local',
                        'rule': ' AND '.join(rule_parts),
                        'patient_idx': idx,
                        'predicted_risk': y_pred_proba[idx],
                        'actual': y.iloc[idx]
                    })
            except Exception:
                continue
        
        return pd.DataFrame(rules) if len(rules) > 0 else None
        
    except Exception as e:
        return None

# ============================================================================
# 6. RULE EVALUATION AND PRIORITIZATION
# ============================================================================

def evaluate_and_prioritize_rules(all_rules, X, y, feature_names):
    """Evaluate rules and prioritize by performance."""
    
    prioritized_rules = []
    
    for _, rule_row in all_rules.iterrows():
        rule_str = str(rule_row['rule'])
        rule_type = rule_row['rule_type']
        
        try:
            mask = None
            
            if 'IF' in rule_str and '>' in rule_str and 'THEN' in rule_str:
                parts = rule_str.replace('IF', '').replace('THEN', '').split('>')
                if len(parts) == 2:
                    feat_name = parts[0].strip()
                    threshold_val = float(parts[1].strip().split()[0])
                    
                    if feat_name in X.columns:
                        mask = X[feat_name] > threshold_val
            
            elif 'AND' in rule_str or ('<=' in rule_str and '>' in rule_str):
                conditions = rule_str.split('AND')
                mask = pd.Series([True] * len(X), index=X.index)
                
                for condition in conditions:
                    condition = condition.strip()
                    
                    if '<=' in condition:
                        parts = condition.split('<=')
                        if len(parts) == 2:
                            feat_name = parts[0].strip()
                            threshold_val = float(parts[1].strip())
                            if feat_name in X.columns:
                                mask = mask & (X[feat_name] <= threshold_val)
                    elif '>' in condition:
                        parts = condition.split('>')
                        if len(parts) == 2:
                            feat_name = parts[0].strip()
                            threshold_val = float(parts[1].strip())
                            if feat_name in X.columns:
                                mask = mask & (X[feat_name] > threshold_val)
            
            elif '>' in rule_str or '<=' in rule_str:
                if '>' in rule_str and '<=' not in rule_str:
                    parts = rule_str.split('>')
                    if len(parts) == 2:
                        feat_name = parts[0].strip()
                        threshold_val = float(parts[1].strip())
                        if feat_name in X.columns:
                            mask = X[feat_name] > threshold_val
                elif '<=' in rule_str:
                    parts = rule_str.split('<=')
                    if len(parts) == 2:
                        feat_name = parts[0].strip()
                        threshold_val = float(parts[1].strip())
                        if feat_name in X.columns:
                            mask = X[feat_name] <= threshold_val
            
            if mask is not None and mask.sum() > 0:
                precision = y[mask].mean()
                recall = (y[mask] & (y == 1)).sum() / y.sum() if y.sum() > 0 else 0
                support = mask.sum()
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                feat_name = rule_row.get('feature', 'Multiple')
                
                prioritized_rules.append({
                    'rule': rule_str,
                    'rule_type': rule_type,
                    'feature': feat_name,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': support,
                    'risk_score': precision,
                    'priority_score': f1 * precision * (support / len(X))
                })
                
        except Exception:
            continue
    
    if len(prioritized_rules) > 0:
        prioritized_df = pd.DataFrame(prioritized_rules)
        prioritized_df = prioritized_df.sort_values('priority_score', ascending=False)
        return prioritized_df
    else:
        return pd.DataFrame()

# ============================================================================
# 7. GENERATE HUMAN-READABLE RULES
# ============================================================================

def generate_patient_explanations(model, X_test, y_test, shap_values, X_shap, feature_names, optimal_threshold):
    """Generate patient-level explanations using SHAP."""
    
    if shap_values is None or X_shap is None:
        return None
    
    # Get predictions for SHAP samples
    y_pred_proba = model.predict_proba(X_shap)[:, 1]
    y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Find example patients at different risk levels
    high_risk_idx = np.where(y_pred_proba > 0.8)[0]
    medium_risk_idx = np.where((y_pred_proba >= 0.4) & (y_pred_proba <= 0.6))[0]
    low_risk_idx = np.where(y_pred_proba < 0.2)[0]
    
    shap_indices = X_shap.index.tolist()
    
    patient_explanations = []
    
    def explain_patient(idx, shap_idx, prediction, actual):
        shap_vals = shap_values[shap_idx]
        
        feature_contributions = pd.DataFrame({
            'Feature': feature_names[:len(shap_vals)],
            'SHAP_Value': shap_vals,
            'Feature_Value': X_shap.iloc[shap_idx].values[:len(shap_vals)]
        })
        
        feature_contributions['Abs_SHAP'] = np.abs(feature_contributions['SHAP_Value'])
        feature_contributions = feature_contributions.sort_values('Abs_SHAP', ascending=False)
        
        top_positive = feature_contributions[feature_contributions['SHAP_Value'] > 0].head(5)
        top_negative = feature_contributions[feature_contributions['SHAP_Value'] < 0].head(5)
        
        return {
            'patient_idx': int(idx),
            'prediction': float(prediction),
            'actual': int(actual),
            'risk_level': 'HIGH' if prediction > 0.7 else 'MODERATE' if prediction > 0.3 else 'LOW',
            'top_risk_factors': top_positive[['Feature', 'SHAP_Value', 'Feature_Value']].to_dict('records'),
            'top_protective_factors': top_negative[['Feature', 'SHAP_Value', 'Feature_Value']].to_dict('records')
        }
    
    for category, indices in [('HIGH_RISK', high_risk_idx), 
                               ('MEDIUM_RISK', medium_risk_idx), 
                               ('LOW_RISK', low_risk_idx)]:
        if len(indices) > 0:
            sample_idx = indices[:min(3, len(indices))]
            for shap_idx in sample_idx:
                original_idx = shap_indices[shap_idx]
                explanation = explain_patient(
                    original_idx, shap_idx,
                    y_pred_proba[shap_idx],
                    y_test.iloc[original_idx] if original_idx in y_test.index else 0
                )
                explanation['category'] = category
                patient_explanations.append(explanation)
    
    if patient_explanations:
        explanations_df = pd.DataFrame(patient_explanations)
        
        explanations_df.to_csv(os.path.join(RULES_DIR, 'patient_explanations.csv'), index=False)
        
        with open(os.path.join(RULES_DIR, 'patient_explanations.json'), 'w') as f:
            json.dump(patient_explanations, f, indent=2, default=str)
        
        return explanations_df
    
    return None

def generate_clinical_summary(model_metrics, shap_importance, top_rules):
    """Generate clinical summary document."""
    
    metrics_file = os.path.join(RESULTS_DIR, 'detailed_metrics.csv')
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
        auroc = metrics_df['AUROC'].iloc[0] if 'AUROC' in metrics_df.columns else 0.0
        sensitivity = metrics_df['Sensitivity_TPR'].iloc[0] if 'Sensitivity_TPR' in metrics_df.columns else 0.0
        specificity = metrics_df['Specificity_TNR'].iloc[0] if 'Specificity_TNR' in metrics_df.columns else 0.0
    else:
        auroc = 0.0
        sensitivity = 0.0
        specificity = 0.0
    
    if shap_importance is not None and len(shap_importance) > 0:
        top_features = shap_importance.head(10)
    else:
        top_features = pd.DataFrame()
    
    clinical_summary = f"""
================================================================================
PROSTATE CANCER PREDICTION - CLINICAL RULES SUMMARY
================================================================================

PREDICTION TASK: 
Based on a patient's data up to TODAY, predict the probability of being
diagnosed with prostate cancer in the NEXT 12 months.

MODEL PERFORMANCE:
- AUC-ROC: {auroc:.4f}
- Sensitivity: {sensitivity:.4f} ({sensitivity*100:.1f}% - catches {sensitivity*100:.1f}% of cancers)
- Specificity: {specificity:.4f} ({specificity*100:.1f}%)
- NPV: ~{specificity*100:.1f}% (low risk predictions are reliable)

RISK STRATIFICATION:
- VERY HIGH (>90%): Immediate investigation required
- HIGH (70-90%): Urgent investigation recommended
- MODERATE-HIGH (50-70%): Further investigation recommended
- MODERATE (30-50%): Closer monitoring advised
- LOW (<30%): Continue routine screening

KEY PREDICTIVE FACTORS:
"""
    
    if len(top_features) > 0:
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            feature = row['Feature'] if 'Feature' in row else row['feature']
            direction = row.get('Direction', 'Unknown')
            clinical_summary += f"\n{i:2d}. {feature} ({direction})"
    else:
        clinical_summary += "\n   (Feature importance not available)"
    
    clinical_summary += f"""

TOP CLINICAL PREDICTION RULES:
"""
    
    if top_rules is not None and len(top_rules) > 0:
        for i, (_, row) in enumerate(top_rules.head(10).iterrows(), 1):
            rule = row['rule'] if 'rule' in row else str(row.get('Rule', ''))
            risk_score = row.get('risk_score', 'N/A')
            if isinstance(risk_score, str):
                clinical_summary += f"\n{i:2d}. {rule} (Risk: {risk_score})"
            else:
                clinical_summary += f"\n{i:2d}. {rule} (Risk: {risk_score*100:.1f}%)"
    else:
        clinical_summary += "\n   (Rules not available)"
    
    clinical_summary += f"""

CLINICAL INTERPRETATION:
- Features capture patient age, PSA levels, laboratory values, and symptoms
- Higher values in most features indicate increased cancer risk
- Model learns patterns from historical data

RECOMMENDATIONS:
1. HIGH/VERY HIGH risk patients should receive urgent PSA/DRE and urology referral
2. MODERATE-HIGH risk patients should have enhanced monitoring and follow-up within 3-6 months
3. MODERATE risk patients should have closer monitoring and review of risk factors
4. LOW risk patients can continue routine screening as per guidelines
5. Model should be used as decision support tool, not replacement for clinical judgment
6. Always combine model predictions with clinical expertise and patient history

GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
"""
    
    summary_file = os.path.join(RULES_DIR, 'clinical_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(clinical_summary)

def generate_human_readable_rules(prioritized_rules, X):
    """Generate final human-readable rules with risk scores."""
    
    if len(prioritized_rules) == 0:
        return pd.DataFrame()
    
    final_rules = []
    
    for i, (_, row) in enumerate(prioritized_rules.head(50).iterrows(), 1):
        risk_pct = row['risk_score'] * 100
        support_pct = (row['support'] / len(X)) * 100
        
        if risk_pct < 30:
            risk_level = "LOW"
        elif risk_pct < 50:
            risk_level = "MODERATE"
        elif risk_pct < 70:
            risk_level = "MODERATE-HIGH"
        elif risk_pct < 90:
            risk_level = "HIGH"
        else:
            risk_level = "VERY HIGH"
        
        final_rules.append({
            'priority_rank': i,
            'rule': row['rule'],
            'rule_type': row['rule_type'],
            'risk_score': f"{risk_pct:.1f}%",
            'risk_level': risk_level,
            'precision': f"{row['precision']:.3f}",
            'recall': f"{row['recall']:.3f}",
            'f1_score': f"{row['f1_score']:.3f}",
            'support': f"{row['support']:,} ({support_pct:.1f}%)",
            'priority_score': f"{row['priority_score']:.4f}"
        })
    
    return pd.DataFrame(final_rules)

# ============================================================================
# 8. EXTRACT FEATURE THRESHOLDS AND INTERACTIONS
# ============================================================================

def extract_feature_thresholds_from_model(model, X_test, y_test, y_pred_proba, feature_names, top_n=20):
    """Extract feature thresholds that lead to high/low risk predictions."""
    
    very_high_mask = y_pred_proba > 0.9
    high_mask = (y_pred_proba > 0.7) & (y_pred_proba <= 0.9)
    moderate_high_mask = (y_pred_proba > 0.5) & (y_pred_proba <= 0.7)
    moderate_mask = (y_pred_proba > 0.3) & (y_pred_proba <= 0.5)
    low_mask = y_pred_proba <= 0.3
    
    feature_importance_file = os.path.join(RESULTS_DIR, 'feature_importance.csv')
    if os.path.exists(feature_importance_file):
        importance_df = pd.read_csv(feature_importance_file)
        top_features = importance_df.head(top_n)['Feature'].tolist()
    else:
        top_features = feature_names[:top_n]
    
    threshold_rules = []
    
    for feat in top_features:
        if feat not in X_test.columns:
            continue
        
        feat_values = X_test[feat].dropna()
        if len(feat_values) == 0:
            continue
        
        very_high_values = X_test.loc[very_high_mask, feat].dropna()
        high_values = X_test.loc[high_mask, feat].dropna()
        moderate_values = X_test.loc[moderate_mask, feat].dropna()
        low_values = X_test.loc[low_mask, feat].dropna()
        
        if len(very_high_values) > 0 and len(low_values) > 0:
            very_high_median = very_high_values.median()
            low_median = low_values.median()
            
            if very_high_median > low_median:
                direction = "Higher values increase risk"
                
                thresholds = {}
                
                if len(very_high_values) > 10:
                    thresholds['VERY_HIGH'] = very_high_values.quantile(0.25)
                    thresholds['HIGH'] = high_values.quantile(0.25) if len(high_values) > 10 else very_high_values.quantile(0.5)
                    thresholds['MODERATE'] = moderate_values.quantile(0.5) if len(moderate_values) > 10 else feat_values.quantile(0.5)
                    thresholds['LOW'] = low_values.quantile(0.75) if len(low_values) > 10 else feat_values.quantile(0.25)
                else:
                    thresholds['VERY_HIGH'] = very_high_values.quantile(0.5)
                    thresholds['HIGH'] = high_values.quantile(0.5) if len(high_values) > 10 else feat_values.quantile(0.75)
                    thresholds['MODERATE'] = feat_values.quantile(0.5)
                    thresholds['LOW'] = feat_values.quantile(0.25)
                
                for risk_level, threshold in thresholds.items():
                    if pd.notna(threshold):
                        mask = X_test[feat] > threshold
                        if mask.sum() > 10:
                            precision = y_test[mask].mean()
                            recall = (y_test[mask] & (y_test == 1)).sum() / y_test.sum() if y_test.sum() > 0 else 0
                            support = mask.sum()
                            
                            threshold_rules.append({
                                'Feature': feat,
                                'Direction': direction,
                                'Risk_Level': risk_level,
                                'Condition': f'{feat} > {threshold:.2f}',
                                'Threshold_Value': threshold,
                                'Precision': precision,
                                'Recall': recall,
                                'Support': support,
                                'Support_Pct': (support / len(X_test)) * 100,
                                'High_Risk_Median': very_high_median,
                                'Low_Risk_Median': low_median
                            })
            
            else:
                direction = "Lower values increase risk"
                
                thresholds = {}
                if len(very_high_values) > 10:
                    thresholds['VERY_HIGH'] = very_high_values.quantile(0.75)
                    thresholds['HIGH'] = high_values.quantile(0.75) if len(high_values) > 10 else very_high_values.quantile(0.5)
                    thresholds['MODERATE'] = moderate_values.quantile(0.5) if len(moderate_values) > 10 else feat_values.quantile(0.5)
                    thresholds['LOW'] = low_values.quantile(0.25) if len(low_values) > 10 else feat_values.quantile(0.75)
                else:
                    thresholds['VERY_HIGH'] = very_high_values.quantile(0.5)
                    thresholds['HIGH'] = high_values.quantile(0.5) if len(high_values) > 10 else feat_values.quantile(0.25)
                    thresholds['MODERATE'] = feat_values.quantile(0.5)
                    thresholds['LOW'] = feat_values.quantile(0.75)
                
                for risk_level, threshold in thresholds.items():
                    if pd.notna(threshold):
                        mask = X_test[feat] < threshold
                        if mask.sum() > 10:
                            precision = y_test[mask].mean()
                            recall = (y_test[mask] & (y_test == 1)).sum() / y_test.sum() if y_test.sum() > 0 else 0
                            support = mask.sum()
                            
                            threshold_rules.append({
                                'Feature': feat,
                                'Direction': direction,
                                'Risk_Level': risk_level,
                                'Condition': f'{feat} < {threshold:.2f}',
                                'Threshold_Value': threshold,
                                'Precision': precision,
                                'Recall': recall,
                                'Support': support,
                                'Support_Pct': (support / len(X_test)) * 100,
                                'High_Risk_Median': very_high_median,
                                'Low_Risk_Median': low_median
                            })
    
    if len(threshold_rules) > 0:
        threshold_df = pd.DataFrame(threshold_rules)
        
        risk_order = {'VERY_HIGH': 1, 'HIGH': 2, 'MODERATE': 3, 'LOW': 4}
        threshold_df['Risk_Order'] = threshold_df['Risk_Level'].map(risk_order)
        threshold_df = threshold_df.sort_values(['Risk_Order', 'Precision'], ascending=[True, False])
        threshold_df = threshold_df.drop('Risk_Order', axis=1)
        
        threshold_file = os.path.join(OUTPUT_DIR, 'feature_thresholds_easy_to_understand.csv')
        threshold_df.to_csv(threshold_file, index=False)
        
        return threshold_df
    else:
        return pd.DataFrame()

def extract_feature_interactions_easy(model, X_test, y_test, y_pred_proba, feature_names, top_features=10):
    """Extract feature interactions (combinations) that lead to high risk."""
    
    feature_importance_file = os.path.join(RESULTS_DIR, 'feature_importance.csv')
    if os.path.exists(feature_importance_file):
        importance_df = pd.read_csv(feature_importance_file)
        top_features_list = importance_df.head(top_features)['Feature'].tolist()
    else:
        top_features_list = feature_names[:top_features]
    
    top_features_list = [f for f in top_features_list if f in X_test.columns]
    
    high_risk_mask = y_pred_proba > 0.7
    low_risk_mask = y_pred_proba < 0.3
    
    interactions = []
    
    for i, feat1 in enumerate(top_features_list[:8]):
        for feat2 in top_features_list[i+1:min(i+4, len(top_features_list))]:
            
            high_risk_feat1 = X_test.loc[high_risk_mask, feat1].dropna()
            low_risk_feat1 = X_test.loc[low_risk_mask, feat1].dropna()
            high_risk_feat2 = X_test.loc[high_risk_mask, feat2].dropna()
            low_risk_feat2 = X_test.loc[low_risk_mask, feat2].dropna()
            
            if len(high_risk_feat1) > 10 and len(low_risk_feat1) > 10 and \
               len(high_risk_feat2) > 10 and len(low_risk_feat2) > 10:
                
                threshold1 = high_risk_feat1.quantile(0.25)
                threshold2 = high_risk_feat2.quantile(0.25)
                
                if high_risk_feat1.median() > low_risk_feat1.median():
                    condition1 = f"{feat1} > {threshold1:.2f}"
                    mask1 = X_test[feat1] > threshold1
                else:
                    condition1 = f"{feat1} < {threshold1:.2f}"
                    mask1 = X_test[feat1] < threshold1
                
                if high_risk_feat2.median() > low_risk_feat2.median():
                    condition2 = f"{feat2} > {threshold2:.2f}"
                    mask2 = X_test[feat2] > threshold2
                else:
                    condition2 = f"{feat2} < {threshold2:.2f}"
                    mask2 = X_test[feat2] < threshold2
                
                combined_mask = mask1 & mask2
                
                if combined_mask.sum() > 20:
                    precision = y_test[combined_mask].mean()
                    recall = (y_test[combined_mask] & (y_test == 1)).sum() / y_test.sum() if y_test.sum() > 0 else 0
                    support = combined_mask.sum()
                    
                    if precision > 0.5:
                        interactions.append({
                            'Feature1': feat1,
                            'Feature2': feat2,
                            'Condition': f"{condition1} AND {condition2}",
                            'Precision': precision,
                            'Recall': recall,
                            'Support': support,
                            'Support_Pct': (support / len(X_test)) * 100,
                            'Risk_Level': 'VERY_HIGH' if precision > 0.8 else 'HIGH' if precision > 0.6 else 'MODERATE-HIGH'
                        })
    
    if len(interactions) > 0:
        interactions_df = pd.DataFrame(interactions)
        interactions_df = interactions_df.sort_values('Precision', ascending=False)
        
        interactions_file = os.path.join(OUTPUT_DIR, 'feature_interactions_easy_to_understand.csv')
        interactions_df.to_csv(interactions_file, index=False)
        
        return interactions_df
    else:
        return pd.DataFrame()

def generate_simple_rules_summary(threshold_df, interactions_df, output_file=None):
    """Generate a simple, easy-to-read summary of all rules."""
    
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, 'simple_rules_summary.txt')
    
    summary = []
    summary.append("="*80)
    summary.append("PROSTATE CANCER PREDICTION - SIMPLE RULES SUMMARY")
    summary.append("="*80)
    summary.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("\n" + "="*80)
    summary.append("HOW TO USE THESE RULES:")
    summary.append("="*80)
    summary.append("\nThese rules show what feature values lead to different risk levels.")
    summary.append("Use them to understand which patient characteristics increase cancer risk.")
    summary.append("\n" + "-"*80)
    
    if len(threshold_df) > 0:
        summary.append("\nSECTION 1: FEATURE THRESHOLDS")
        summary.append("-"*80)
        summary.append("\nThese rules show individual feature values that indicate risk levels.")
        summary.append("\nVERY HIGH RISK (>90% probability):")
        summary.append("-"*80)
        
        very_high = threshold_df[threshold_df['Risk_Level'] == 'VERY_HIGH'].head(10)
        for _, row in very_high.iterrows():
            summary.append(f"  • {row['Condition']:50s} → {row['Precision']*100:5.1f}% cancer rate ({row['Support']:,} patients)")
        
        summary.append("\nHIGH RISK (70-90% probability):")
        summary.append("-"*80)
        high = threshold_df[threshold_df['Risk_Level'] == 'HIGH'].head(10)
        for _, row in high.iterrows():
            summary.append(f"  • {row['Condition']:50s} → {row['Precision']*100:5.1f}% cancer rate ({row['Support']:,} patients)")
        
        summary.append("\nMODERATE RISK (30-50% probability):")
        summary.append("-"*80)
        moderate = threshold_df[threshold_df['Risk_Level'] == 'MODERATE'].head(10)
        for _, row in moderate.iterrows():
            summary.append(f"  • {row['Condition']:50s} → {row['Precision']*100:5.1f}% cancer rate ({row['Support']:,} patients)")
        
        summary.append("\nLOW RISK (<30% probability):")
        summary.append("-"*80)
        low = threshold_df[threshold_df['Risk_Level'] == 'LOW'].head(10)
        for _, row in low.iterrows():
            summary.append(f"  • {row['Condition']:50s} → {row['Precision']*100:5.1f}% cancer rate ({row['Support']:,} patients)")
    
    if len(interactions_df) > 0:
        summary.append("\n\n" + "="*80)
        summary.append("SECTION 2: FEATURE INTERACTIONS")
        summary.append("-"*80)
        summary.append("\nThese rules show combinations of features that lead to high risk.")
        summary.append("When BOTH conditions are true, the risk is higher.")
        summary.append("\nTop Feature Combinations:")
        summary.append("-"*80)
        
        for _, row in interactions_df.head(15).iterrows():
            summary.append(f"  • {row['Condition']:60s} → {row['Precision']*100:5.1f}% cancer rate ({row['Support']:,} patients)")
    
    summary.append("\n\n" + "="*80)
    summary.append("CLINICAL INTERPRETATION")
    summary.append("="*80)
    summary.append("\n• Higher values in most features (AGE, PSA, etc.) generally increase risk")
    summary.append("• Multiple risk factors together increase risk more than individual factors")
    summary.append("• These rules are derived from the trained model and reflect patterns in the data")
    summary.append("• Always combine model predictions with clinical judgment")
    
    summary.append("\n\n" + "="*80)
    
    summary_text = "\n".join(summary)
    with open(output_file, 'w') as f:
        f.write(summary_text)
    
    return summary_text

def main():
    """Main function."""
    
    try:
        # Load optimal threshold from Phase 3.1
        optimal_threshold = load_optimal_threshold()
        
        # Load model and data
        model, X, y, feature_names, scaler, model_type = load_model_and_data()
        
        # Get train/test split
        split_file_new = os.path.join(RESULTS_DIR, 'train_val_test_split.csv')
        split_file_old = os.path.join(RESULTS_DIR, 'train_test_split.csv')
        
        if os.path.exists(split_file_new):
            split_df = pd.read_csv(split_file_new)
            test_indices = split_df[split_df['dataset'] == 'test']['index'].tolist()
            X_test = X.loc[test_indices]
            y_test = y.loc[test_indices]
        elif os.path.exists(split_file_old):
            split_df = pd.read_csv(split_file_old)
            test_indices = split_df[split_df['dataset'] == 'test']['index'].tolist()
            X_test = X.loc[test_indices]
            y_test = y.loc[test_indices]
        else:
            X_test = X
            y_test = y
        
        # Extract rules from different methods
        all_rules_list = []
        shap_values = None
        X_shap = None
        shap_importance = None
        
        # SHAP rules
        shap_rules, shap_values, X_shap = extract_shap_rules(model, X, y, feature_names, model_type, scaler)
        if shap_rules is not None and len(shap_rules) > 0:
            all_rules_list.append(shap_rules)
            shap_importance_file = os.path.join(SHAP_DIR, 'shap_feature_importance.csv')
            if os.path.exists(shap_importance_file):
                shap_importance = pd.read_csv(shap_importance_file)
        
        # Decision Tree rules
        dt_rules, dt_model = extract_decision_tree_rules(X, y, feature_names)
        if dt_rules is not None and len(dt_rules) > 0:
            all_rules_list.append(dt_rules)
        
        # RuleFit rules
        rulefit_rules = extract_rulefit_rules(X, y, feature_names)
        if rulefit_rules is not None and len(rulefit_rules) > 0:
            all_rules_list.append(rulefit_rules)
        
        # LIME rules
        lime_rules = extract_lime_rules(model, X, y, feature_names, scaler)
        if lime_rules is not None and len(lime_rules) > 0:
            all_rules_list.append(lime_rules)
        
        # Combine all rules
        if all_rules_list:
            all_rules = pd.concat(all_rules_list, ignore_index=True)
            
            # Evaluate and prioritize
            prioritized_rules = evaluate_and_prioritize_rules(all_rules, X, y, feature_names)
            
            if len(prioritized_rules) > 0:
                # Generate human-readable format
                final_rules = generate_human_readable_rules(prioritized_rules, X)
                
                # Save in multiple formats
                csv_file = os.path.join(OUTPUT_DIR, 'prioritized_prediction_rules.csv')
                final_rules.to_csv(csv_file, index=False)
                
                json_file = os.path.join(OUTPUT_DIR, 'prioritized_prediction_rules.json')
                final_rules.to_json(json_file, orient='records', indent=2)
                
                # Generate patient-level explanations
                if shap_values is not None and X_shap is not None:
                    generate_patient_explanations(model, X_test, y_test, shap_values, X_shap, 
                                                 feature_names, optimal_threshold)
                
                # Generate clinical summary
                generate_clinical_summary(None, shap_importance, final_rules)
        
        # Extract feature thresholds and interactions
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
            y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        
        threshold_df = extract_feature_thresholds_from_model(
            model, X_test, y_test, y_pred_proba_test, feature_names, top_n=20
        )
        
        interactions_df = extract_feature_interactions_easy(
            model, X_test, y_test, y_pred_proba_test, feature_names, top_features=10
        )
        
        # Generate simple summary document
        if len(threshold_df) > 0 or len(interactions_df) > 0:
            generate_simple_rules_summary(threshold_df, interactions_df)
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
