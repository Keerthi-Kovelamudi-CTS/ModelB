"""
Lung Cancer Prediction - Inference Script
==========================================
Use a trained model to predict lung cancer on new/unseen data.
If CANCER_FLAG labels are present, calculates comprehensive evaluation metrics.
Includes explainability - shows WHY the model predicted cancer for each patient.

Usage:
    python predict_cancer.py --data <path_to_csv> --model <path_to_model.joblib>
    
Example:
    python3 predict_cancer.py --data 13-all_patients_lungCancer_trend_medication_observation_5M_25M_20yrs-1yr_NoEthnicity.csv
    python3 predict_cancer.py --data test_data.csv --model lung_cancer_best_model.joblib
    python3 predict_cancer.py --data test_data.csv --plot
    python3 predict_cancer.py --data test_data.csv --save-plot evaluation_results.png
    python3 predict_cancer.py --data test_data.csv --output my_predictions.csv
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import os
import sys
from datetime import datetime

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ SHAP not available. Install with: pip install shap")
    print("  Explainability features will be limited.")

import warnings
warnings.filterwarnings('ignore')


class CancerPredictor:
    """
    Class to load a trained model and make predictions on new data.
    Preprocessing matches exactly what was done in lung_cancer_prediction.py
    Includes explainability features to understand WHY a prediction was made.
    """
    
    def __init__(self, model_path):
        """
        Initialize the predictor by loading the trained model.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved joblib model file.
        """
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.selected_features = None
        self.feature_selector = None
        self.explainer = None
        self.X_processed = None  # Store for explanations
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and associated artifacts."""
        print("=" * 60)
        print("LOADING MODEL")
        print("=" * 60)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model_data = joblib.load(self.model_path)
        
        self.model = self.model_data['model']
        self.model_name = self.model_data.get('model_name', 'Unknown')
        self.scaler = self.model_data.get('scaler', None)
        self.feature_names = self.model_data.get('feature_names', None)
        self.selected_features = self.model_data.get('selected_features', None)
        self.feature_selector = self.model_data.get('feature_selector', None)
        
        print(f"✓ Model loaded: {self.model_name}")
        print(f"  Model type: {type(self.model).__name__}")
        if self.feature_names:
            print(f"  Features after preprocessing: {len(self.feature_names)}")
        if self.selected_features:
            print(f"  Features after selection: {len(self.selected_features)}")
        if self.feature_selector:
            print(f"  Feature selector: Available")
        else:
            print(f"  ⚠ Feature selector: Not available (will use selected_features)")
        
    def preprocess_data(self, df, drop_threshold=0.90, variance_threshold=0.01):
        """
        Preprocess the input data to match the training data format.
        This mirrors the preprocessing in lung_cancer_prediction.py exactly.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with patient data.
            
        Returns:
        --------
        np.ndarray : Preprocessed feature matrix
        np.ndarray or None : True labels if CANCER_FLAG present
        """
        print("\n" + "=" * 60)
        print("PREPROCESSING DATA")
        print("=" * 60)
        
        df_processed = df.copy()
        
        # Check for CANCER_FLAG and store if present
        has_labels = 'CANCER_FLAG' in df_processed.columns
        y_true = None
        if has_labels:
            y_true = df_processed['CANCER_FLAG'].values
            df_processed = df_processed.drop(columns=['CANCER_FLAG'])
            print("✓ CANCER_FLAG labels found - will calculate evaluation metrics")
        else:
            print("ℹ CANCER_FLAG labels not found - will only generate predictions")
        
        # Step 1: Remove identifier columns (same as training)
        cols_to_drop = ['index', 'PATIENT_GUID']
        cols_to_drop = [c for c in cols_to_drop if c in df_processed.columns]
        if cols_to_drop:
            df_processed = df_processed.drop(columns=cols_to_drop)
            print(f"✓ Dropped identifier columns: {cols_to_drop}")
        
        # Step 2: Drop date columns and text/code columns (same as training)
        date_cols = [col for col in df_processed.columns if 'DATE' in col.upper()]
        text_cols = [col for col in df_processed.columns if 'TERMS' in col.upper() or 'CODES' in col.upper()]
        df_processed = df_processed.drop(columns=date_cols + text_cols, errors='ignore')
        print(f"✓ Dropped {len(date_cols)} date columns and {len(text_cols)} text/code columns")
        
        # Step 3: Encode categorical columns (same as training)
        cat_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            df_processed[col] = df_processed[col].fillna('Unknown')
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        print(f"✓ Encoded {len(cat_cols)} categorical columns")
        
        # Step 4: Handle missing values (same as training)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        # Drop columns with too many missing values
        missing_pct = df_processed[numeric_cols].isnull().sum() / len(df_processed)
        cols_high_missing = missing_pct[missing_pct > drop_threshold].index.tolist()
        if cols_high_missing:
            df_processed = df_processed.drop(columns=cols_high_missing)
            print(f"✓ Dropped {len(cols_high_missing)} columns with >{drop_threshold*100}% missing values")
        
        # Impute remaining missing values with median
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
        print(f"✓ Imputed missing values with median")
        
        # Step 5: Remove low variance features
        variances = df_processed.var()
        low_var_cols = variances[variances < variance_threshold].index.tolist()
        if low_var_cols:
            df_processed = df_processed.drop(columns=low_var_cols)
            print(f"✓ Dropped {len(low_var_cols)} low-variance features")
        
        # Step 6: Remove highly correlated features (same logic as training)
        corr_matrix = df_processed.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
        high_corr_cols = high_corr_cols[:len(high_corr_cols)//2]
        if high_corr_cols:
            df_processed = df_processed.drop(columns=high_corr_cols)
            print(f"✓ Dropped {len(high_corr_cols)} highly correlated features")
        
        print(f"\nAfter preprocessing: {df_processed.shape[1]} features")
        
        # Step 7: Align columns with training features
        if self.feature_names:
            # Add missing columns with 0s
            missing_cols = set(self.feature_names) - set(df_processed.columns)
            for col in missing_cols:
                df_processed[col] = 0
            
            if missing_cols:
                print(f"✓ Added {len(missing_cols)} missing columns (filled with 0)")
            
            # Remove extra columns not in training
            extra_cols = set(df_processed.columns) - set(self.feature_names)
            if extra_cols:
                df_processed = df_processed.drop(columns=list(extra_cols), errors='ignore')
                print(f"✓ Removed {len(extra_cols)} extra columns not in training")
            
            # Reorder columns to match training exactly
            df_processed = df_processed[self.feature_names]
            print(f"✓ Aligned {len(self.feature_names)} features with training data")
        
        # Step 8: Scale features using training scaler
        X = df_processed.values
        if self.scaler:
            X = self.scaler.transform(X)
            print("✓ Features scaled using training scaler")
        
        # Step 9: Apply feature selection (same as training)
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
            print(f"✓ Applied feature selection: {X.shape[1]} features selected")
        elif self.selected_features is not None:
            # Fallback: use selected feature names if selector not available
            # This might not work perfectly if column order changed
            print(f"⚠ Using selected feature names (feature selector not saved)")
            selected_indices = [self.feature_names.index(f) for f in self.selected_features if f in self.feature_names]
            X = X[:, selected_indices]
            print(f"  Selected {X.shape[1]} features by name matching")
        
        print(f"\n✓ Final preprocessed shape: {X.shape}")
        
        return X, y_true
    
    def predict(self, data_path):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing patient data.
            
        Returns:
        --------
        dict : Dictionary containing predictions and optionally evaluation metrics
        """
        print("\n" + "=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"✓ Loaded data: {df.shape[0]} samples, {df.shape[1]} columns")
        
        # Preprocess
        X, y_true = self.preprocess_data(df)
        
        # Make predictions
        print("\n" + "=" * 60)
        print("MAKING PREDICTIONS")
        print("=" * 60)
        
        y_pred = self.model.predict(X)
        
        # Get prediction probabilities if available
        y_pred_proba = None
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        results = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'n_samples': len(y_pred),
            'n_positive': int(np.sum(y_pred == 1)),
            'n_negative': int(np.sum(y_pred == 0)),
            'has_labels': y_true is not None
        }
        
        print(f"\n📊 Prediction Summary:")
        print(f"  Total samples: {results['n_samples']}")
        print(f"  Predicted Cancer (1): {results['n_positive']} ({100*results['n_positive']/results['n_samples']:.1f}%)")
        print(f"  Predicted No Cancer (0): {results['n_negative']} ({100*results['n_negative']/results['n_samples']:.1f}%)")
        
        # Calculate evaluation metrics if labels are available
        if y_true is not None:
            results['y_true'] = y_true
            results['metrics'] = self._calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Generate explanations for predictions (especially cancer cases)
        # Store X for explanations
        self.X_processed = X
        results['explanations'] = self.explain_predictions(X, y_pred, y_pred_proba)
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive evaluation metrics.
        """
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)  # PPV
        recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Specificity: TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # NPV: TN / (TN + FN)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # PPV (same as precision): TP / (TP + FP)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        metrics = {
            'confusion_matrix': cm,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'accuracy': accuracy,
            'sensitivity': recall,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'f1_score': f1,
            'roc_auc': None,
            'avg_precision': None
        }
        
        # Calculate AUC if probabilities are available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
            metrics['y_pred_proba'] = y_pred_proba
        
        # Print metrics
        print("\n📊 Confusion Matrix:")
        print(f"                    Predicted")
        print(f"                 No Cancer  Cancer")
        print(f"Actual No Cancer    {tn:5d}    {fp:5d}")
        print(f"Actual Cancer       {fn:5d}    {tp:5d}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"  Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Sensitivity:       {recall:.4f} ({recall*100:.2f}%) - Ability to detect cancer")
        print(f"  Specificity:       {specificity:.4f} ({specificity*100:.2f}%) - Ability to identify non-cancer")
        print(f"  PPV (Precision):   {ppv:.4f} ({ppv*100:.2f}%) - Probability of cancer given positive test")
        print(f"  NPV:               {npv:.4f} ({npv*100:.2f}%) - Probability of no cancer given negative test")
        print(f"  F1 Score:          {f1:.4f}")
        
        if metrics['roc_auc'] is not None:
            print(f"  ROC AUC:           {metrics['roc_auc']:.4f}")
            print(f"  Average Precision: {metrics['avg_precision']:.4f}")
        
        # Detailed classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['No Cancer', 'Cancer']))
        
        return metrics
    
    def explain_predictions(self, X, y_pred, y_pred_proba, top_n_features=10):
        """
        Generate explanations for predictions, especially for cancer cases.
        Uses SHAP values to identify top contributing factors.
        
        Parameters:
        -----------
        X : np.ndarray
            Preprocessed feature matrix
        y_pred : np.ndarray
            Predicted labels
        y_pred_proba : np.ndarray
            Prediction probabilities
        top_n_features : int
            Number of top features to include in explanation
            
        Returns:
        --------
        list : List of explanation dictionaries for each sample
        """
        print("\n" + "=" * 60)
        print("GENERATING EXPLANATIONS")
        print("=" * 60)
        
        # Get feature names for the selected features
        if self.selected_features is not None:
            feature_names = self.selected_features
        elif self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        explanations = []
        shap_values = None
        
        # Try SHAP-based explanation first
        if SHAP_AVAILABLE:
            try:
                print("Computing SHAP values for explainability...")
                
                # Create SHAP explainer based on model type
                model_type = type(self.model).__name__
                
                if model_type in ['LGBMClassifier', 'XGBClassifier', 'CatBoostClassifier', 
                                  'RandomForestClassifier', 'GradientBoostingClassifier', 
                                  'ExtraTreesClassifier']:
                    # Tree-based models - use TreeExplainer (fast)
                    self.explainer = shap.TreeExplainer(self.model)
                    shap_values = self.explainer.shap_values(X)
                    
                    # Handle different output formats
                    if isinstance(shap_values, list):
                        # For multi-class, take the positive class (index 1)
                        shap_values = shap_values[1]
                    
                elif model_type == 'VotingClassifier':
                    # For ensemble, use KernelExplainer with a sample of data
                    print("  Using KernelExplainer for ensemble model (may take longer)...")
                    background = shap.sample(X, min(100, len(X)))
                    self.explainer = shap.KernelExplainer(
                        lambda x: self.model.predict_proba(x)[:, 1], 
                        background
                    )
                    shap_values = self.explainer.shap_values(X, nsamples=100)
                    
                else:
                    # Generic approach for other models
                    print(f"  Using KernelExplainer for {model_type}...")
                    background = shap.sample(X, min(100, len(X)))
                    self.explainer = shap.KernelExplainer(
                        lambda x: self.model.predict_proba(x)[:, 1], 
                        background
                    )
                    shap_values = self.explainer.shap_values(X, nsamples=100)
                
                print(f"✓ SHAP values computed for {len(X)} samples")
                
            except Exception as e:
                print(f"⚠ SHAP computation failed: {str(e)}")
                print("  Falling back to feature importance based explanation")
                shap_values = None
        
        # Generate explanations for each sample
        for i in range(len(X)):
            explanation = {
                'prediction': int(y_pred[i]),
                'probability': float(y_pred_proba[i]) if y_pred_proba is not None else None,
                'risk_level': self._get_risk_level(y_pred_proba[i] if y_pred_proba is not None else y_pred[i]),
                'top_factors': [],
                'explanation_text': ''
            }
            
            if y_pred[i] == 1:  # Cancer predicted
                if shap_values is not None:
                    # Get SHAP values for this sample
                    sample_shap = shap_values[i]
                    
                    # Get indices sorted by absolute SHAP value (most important first)
                    sorted_indices = np.argsort(np.abs(sample_shap))[::-1]
                    
                    # Get top contributing factors
                    top_factors = []
                    for j in sorted_indices[:top_n_features]:
                        feature_name = feature_names[j] if j < len(feature_names) else f'Feature_{j}'
                        shap_value = sample_shap[j]
                        feature_value = X[i, j]
                        
                        # Determine direction of contribution
                        if shap_value > 0:
                            direction = "increases"
                        else:
                            direction = "decreases"
                        
                        factor = {
                            'feature': feature_name,
                            'value': float(feature_value),
                            'shap_value': float(shap_value),
                            'contribution': direction,
                            'importance': abs(float(shap_value))
                        }
                        top_factors.append(factor)
                    
                    explanation['top_factors'] = top_factors
                    
                    # Generate human-readable explanation
                    explanation['explanation_text'] = self._generate_explanation_text(top_factors)
                    
                else:
                    # Fallback: Use feature importance if available
                    if hasattr(self.model, 'feature_importances_'):
                        importances = self.model.feature_importances_
                        sorted_indices = np.argsort(importances)[::-1]
                        
                        top_factors = []
                        for j in sorted_indices[:top_n_features]:
                            feature_name = feature_names[j] if j < len(feature_names) else f'Feature_{j}'
                            feature_value = X[i, j]
                            
                            factor = {
                                'feature': feature_name,
                                'value': float(feature_value),
                                'importance': float(importances[j])
                            }
                            top_factors.append(factor)
                        
                        explanation['top_factors'] = top_factors
                        explanation['explanation_text'] = self._generate_simple_explanation(top_factors, X[i])
            else:
                explanation['explanation_text'] = "Low cancer risk based on patient features"
            
            explanations.append(explanation)
        
        # Count how many cancer predictions have explanations
        cancer_predictions = sum(1 for e in explanations if e['prediction'] == 1)
        print(f"✓ Generated explanations for {cancer_predictions} cancer predictions")
        
        return explanations
    
    def _get_risk_level(self, prob_or_pred):
        """Convert probability to risk level category."""
        if isinstance(prob_or_pred, (int, np.integer)):
            return "High" if prob_or_pred == 1 else "Low"
        
        if prob_or_pred >= 0.8:
            return "Very High"
        elif prob_or_pred >= 0.5:
            return "High"
        elif prob_or_pred >= 0.2:
            return "Moderate"
        else:
            return "Low"
    
    def _generate_explanation_text(self, top_factors):
        """Generate human-readable explanation from SHAP-based factors."""
        if not top_factors:
            return "Unable to determine contributing factors"
        
        explanations = []
        for factor in top_factors[:3]:  # Top 3 for text
            feature = factor['feature']
            # Clean up feature name for readability
            feature_readable = feature.replace('_', ' ').replace('NUM ', 'Number of ')
            
            if factor['shap_value'] > 0:
                explanations.append(f"{feature_readable} (contributes to higher risk)")
            else:
                explanations.append(f"{feature_readable} (partially protective)")
        
        return "Key factors: " + "; ".join(explanations)
    
    def _generate_simple_explanation(self, top_factors, sample_values):
        """Generate explanation when SHAP is not available."""
        if not top_factors:
            return "Unable to determine contributing factors"
        
        explanations = []
        for factor in top_factors[:3]:
            feature = factor['feature']
            feature_readable = feature.replace('_', ' ')
            explanations.append(f"{feature_readable}")
        
        return "Important features: " + "; ".join(explanations)
    
    def plot_results(self, results, save_path=None):
        """
        Generate visualization plots for the results.
        """
        if not results['has_labels']:
            print("\n⚠ Cannot plot evaluation metrics without true labels.")
            return
        
        metrics = results['metrics']
        y_true = results['y_true']
        y_pred_proba = metrics.get('y_pred_proba', None)
        
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)
        
        # Determine number of subplots based on available data
        if y_pred_proba is not None:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes = axes.reshape(1, -1)
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0] if y_pred_proba is not None else axes[0, 0]
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['No Cancer', 'Cancer'],
                   yticklabels=['No Cancer', 'Cancer'],
                   annot_kws={'size': 14})
        ax1.set_xlabel('Predicted', fontsize=12)
        ax1.set_ylabel('Actual', fontsize=12)
        ax1.set_title(f'Confusion Matrix\n(Model: {self.model_name})', fontsize=14)
        
        # 2. Metrics Bar Chart
        ax2 = axes[0, 1] if y_pred_proba is not None else axes[0, 1]
        metric_names = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1 Score']
        metric_values = [
            metrics['accuracy'],
            metrics['sensitivity'],
            metrics['specificity'],
            metrics['ppv'],
            metrics['npv'],
            metrics['f1_score']
        ]
        
        colors = ['#2ecc71' if v >= 0.9 else '#f39c12' if v >= 0.8 else '#e74c3c' for v in metric_values]
        bars = ax2.barh(metric_names, metric_values, color=colors)
        ax2.set_xlim(0, 1.1)
        ax2.set_xlabel('Score', fontsize=12)
        ax2.set_title('Performance Metrics', fontsize=14)
        
        # Add value labels
        for bar, val in zip(bars, metric_values):
            ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=11)
        
        # 3. ROC Curve (if probabilities available)
        if y_pred_proba is not None:
            ax3 = axes[1, 0]
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            roc_auc = metrics['roc_auc']
            
            ax3.plot(fpr, tpr, color='#3498db', lw=2, 
                    label=f'ROC Curve (AUC = {roc_auc:.4f})')
            ax3.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
            ax3.fill_between(fpr, tpr, alpha=0.3, color='#3498db')
            
            # Mark optimal threshold point
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            ax3.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, zorder=5,
                       label=f'Optimal Threshold = {optimal_threshold:.3f}')
            
            ax3.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
            ax3.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
            ax3.set_title('ROC Curve', fontsize=14)
            ax3.legend(loc='lower right')
            ax3.grid(True, alpha=0.3)
            
            # 4. Precision-Recall Curve
            ax4 = axes[1, 1]
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = metrics['avg_precision']
            
            ax4.plot(recall_curve, precision_curve, color='#9b59b6', lw=2,
                    label=f'PR Curve (AP = {avg_precision:.4f})')
            ax4.fill_between(recall_curve, precision_curve, alpha=0.3, color='#9b59b6')
            
            # Baseline (proportion of positive class)
            baseline = np.sum(y_true) / len(y_true)
            ax4.axhline(y=baseline, color='gray', linestyle='--', 
                       label=f'Baseline (Prevalence = {baseline:.3f})')
            
            ax4.set_xlabel('Recall (Sensitivity)', fontsize=12)
            ax4.set_ylabel('Precision (PPV)', fontsize=12)
            ax4.set_title('Precision-Recall Curve', fontsize=14)
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to: {save_path}")
        
        plt.show()
    
    def save_predictions(self, results, data_path, output_path=None):
        """
        Save predictions to a CSV file with explanations for cancer predictions.
        """
        print("\n" + "=" * 60)
        print("SAVING PREDICTIONS WITH EXPLANATIONS")
        print("=" * 60)
        
        # Load original data to get patient IDs if available
        df_original = pd.read_csv(data_path)
        
        # Create output dataframe
        output_df = pd.DataFrame()
        
        # Add patient identifier if available
        if 'PATIENT_GUID' in df_original.columns:
            output_df['PATIENT_GUID'] = df_original['PATIENT_GUID']
        elif 'index' in df_original.columns:
            output_df['Patient_Index'] = df_original['index']
        else:
            output_df['Sample_Index'] = range(len(results['predictions']))
        
        # Add predictions
        output_df['Predicted_Cancer'] = results['predictions']
        
        # Add probabilities if available
        if results['probabilities'] is not None:
            output_df['Cancer_Probability'] = np.round(results['probabilities'], 4)
            output_df['Risk_Category'] = pd.cut(
                results['probabilities'],
                bins=[0, 0.2, 0.5, 0.8, 1.0],
                labels=['Low', 'Moderate', 'High', 'Very High']
            )
        
        # Add true labels if available
        if results['has_labels']:
            output_df['Actual_Cancer'] = results['y_true']
            output_df['Correct_Prediction'] = (
                results['predictions'] == results['y_true']
            ).astype(int)
        
        # Add explanations for each prediction
        if 'explanations' in results and results['explanations']:
            explanations = results['explanations']
            
            # Add explanation text column
            output_df['Explanation'] = [exp.get('explanation_text', '') for exp in explanations]
            
            # Add detailed factor columns for cancer predictions
            # Factor 1 (most important)
            output_df['Top_Factor_1'] = [
                exp['top_factors'][0]['feature'] if exp['prediction'] == 1 and exp['top_factors'] else ''
                for exp in explanations
            ]
            output_df['Factor_1_Contribution'] = [
                f"{exp['top_factors'][0]['shap_value']:.4f}" if exp['prediction'] == 1 and exp['top_factors'] and 'shap_value' in exp['top_factors'][0] else ''
                for exp in explanations
            ]
            
            # Factor 2
            output_df['Top_Factor_2'] = [
                exp['top_factors'][1]['feature'] if exp['prediction'] == 1 and len(exp['top_factors']) > 1 else ''
                for exp in explanations
            ]
            output_df['Factor_2_Contribution'] = [
                f"{exp['top_factors'][1]['shap_value']:.4f}" if exp['prediction'] == 1 and len(exp['top_factors']) > 1 and 'shap_value' in exp['top_factors'][1] else ''
                for exp in explanations
            ]
            
            # Factor 3
            output_df['Top_Factor_3'] = [
                exp['top_factors'][2]['feature'] if exp['prediction'] == 1 and len(exp['top_factors']) > 2 else ''
                for exp in explanations
            ]
            output_df['Factor_3_Contribution'] = [
                f"{exp['top_factors'][2]['shap_value']:.4f}" if exp['prediction'] == 1 and len(exp['top_factors']) > 2 and 'shap_value' in exp['top_factors'][2] else ''
                for exp in explanations
            ]
            
            # Factor 4
            output_df['Top_Factor_4'] = [
                exp['top_factors'][3]['feature'] if exp['prediction'] == 1 and len(exp['top_factors']) > 3 else ''
                for exp in explanations
            ]
            output_df['Factor_4_Contribution'] = [
                f"{exp['top_factors'][3]['shap_value']:.4f}" if exp['prediction'] == 1 and len(exp['top_factors']) > 3 and 'shap_value' in exp['top_factors'][3] else ''
                for exp in explanations
            ]
            
            # Factor 5
            output_df['Top_Factor_5'] = [
                exp['top_factors'][4]['feature'] if exp['prediction'] == 1 and len(exp['top_factors']) > 4 else ''
                for exp in explanations
            ]
            output_df['Factor_5_Contribution'] = [
                f"{exp['top_factors'][4]['shap_value']:.4f}" if exp['prediction'] == 1 and len(exp['top_factors']) > 4 and 'shap_value' in exp['top_factors'][4] else ''
                for exp in explanations
            ]
            
            # Factor 6
            output_df['Top_Factor_6'] = [
                exp['top_factors'][5]['feature'] if exp['prediction'] == 1 and len(exp['top_factors']) > 5 else ''
                for exp in explanations
            ]
            output_df['Factor_6_Contribution'] = [
                f"{exp['top_factors'][5]['shap_value']:.4f}" if exp['prediction'] == 1 and len(exp['top_factors']) > 5 and 'shap_value' in exp['top_factors'][5] else ''
                for exp in explanations
            ]
            
            # Factor 7
            output_df['Top_Factor_7'] = [
                exp['top_factors'][6]['feature'] if exp['prediction'] == 1 and len(exp['top_factors']) > 6 else ''
                for exp in explanations
            ]
            output_df['Factor_7_Contribution'] = [
                f"{exp['top_factors'][6]['shap_value']:.4f}" if exp['prediction'] == 1 and len(exp['top_factors']) > 6 and 'shap_value' in exp['top_factors'][6] else ''
                for exp in explanations
            ]
            
            # Factor 8
            output_df['Top_Factor_8'] = [
                exp['top_factors'][7]['feature'] if exp['prediction'] == 1 and len(exp['top_factors']) > 7 else ''
                for exp in explanations
            ]
            output_df['Factor_8_Contribution'] = [
                f"{exp['top_factors'][7]['shap_value']:.4f}" if exp['prediction'] == 1 and len(exp['top_factors']) > 7 and 'shap_value' in exp['top_factors'][7] else ''
                for exp in explanations
            ]
            
            # Factor 9
            output_df['Top_Factor_9'] = [
                exp['top_factors'][8]['feature'] if exp['prediction'] == 1 and len(exp['top_factors']) > 8 else ''
                for exp in explanations
            ]
            output_df['Factor_9_Contribution'] = [
                f"{exp['top_factors'][8]['shap_value']:.4f}" if exp['prediction'] == 1 and len(exp['top_factors']) > 8 and 'shap_value' in exp['top_factors'][8] else ''
                for exp in explanations
            ]
            
            # Factor 10
            output_df['Top_Factor_10'] = [
                exp['top_factors'][9]['feature'] if exp['prediction'] == 1 and len(exp['top_factors']) > 9 else ''
                for exp in explanations
            ]
            output_df['Factor_10_Contribution'] = [
                f"{exp['top_factors'][9]['shap_value']:.4f}" if exp['prediction'] == 1 and len(exp['top_factors']) > 9 and 'shap_value' in exp['top_factors'][9] else ''
                for exp in explanations
            ]
            
            # Add a combined top factors column for easy reading
            output_df['All_Top_Factors'] = [
                ' | '.join([f['feature'] for f in exp['top_factors']]) if exp['prediction'] == 1 and exp['top_factors'] else ''
                for exp in explanations
            ]
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(data_path))[0]
            output_path = f"predictions_{base_name}_{timestamp}.csv"
        
        output_df.to_csv(output_path, index=False)
        print(f"✓ Predictions saved to: {output_path}")
        print(f"  Total records: {len(output_df)}")
        
        # Print summary of explanation columns
        cancer_with_explanations = sum(1 for exp in results.get('explanations', []) 
                                        if exp['prediction'] == 1 and exp['top_factors'])
        print(f"  Cancer predictions with explanations: {cancer_with_explanations}")
        
        return output_path


def main():
    """Main function to run predictions from command line."""
    parser = argparse.ArgumentParser(
        description='Lung Cancer Prediction - Make predictions on new data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python predict_cancer.py --data patients.csv
                python predict_cancer.py --data patients.csv --model my_model.joblib
                python predict_cancer.py --data patients.csv --output predictions.csv --plot
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to the input CSV file (same format as training data)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='lung_cancer_best_model.joblib',
        help='Path to the trained model file (default: lung_cancer_best_model.joblib)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path for output predictions CSV (default: auto-generated)'
    )
    
    parser.add_argument(
        '--plot', '-p',
        action='store_true',
        help='Generate and display evaluation plots (only if labels are present)'
    )
    
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help='Path to save the evaluation plot'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("LUNG CANCER PREDICTION - INFERENCE")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data file: {args.data}")
    print(f"Model file: {args.model}")
    
    # Initialize predictor
    predictor = CancerPredictor(args.model)
    
    # Make predictions
    results = predictor.predict(args.data)
    
    # Generate plots if requested and labels are available
    if args.plot or args.save_plot:
        predictor.plot_results(results, save_path=args.save_plot)
    
    # Save predictions
    predictor.save_predictions(results, args.data, args.output)
    
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()
