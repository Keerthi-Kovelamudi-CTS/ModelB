"""
Lung Cancer Prediction Model
============================
A comprehensive machine learning pipeline to predict lung cancer (CANCER_FLAG)
from patient medical records with high accuracy.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import joblib
import os

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Class Imbalance Handling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

# XGBoost and LightGBM (if available)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class LungCancerPredictor:
    """
    A comprehensive class for lung cancer prediction using machine learning.
    """
    
    def __init__(self, data_path):
        """Initialize the predictor with data path."""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_data(self):
        """Load and perform initial data exploration."""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Total Samples: {self.df.shape[0]}")
        print(f"Total Features: {self.df.shape[1]}")
        
        # Target distribution
        print(f"\nTarget Distribution (CANCER_FLAG):")
        print(self.df['CANCER_FLAG'].value_counts())
        print(f"\nClass Imbalance Ratio: {self.df['CANCER_FLAG'].value_counts()[0] / self.df['CANCER_FLAG'].value_counts()[1]:.2f}:1")
        
        return self
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Data types
        print(f"\nData Types:")
        print(self.df.dtypes.value_counts())
        
        # Missing values
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        }).sort_values('Missing %', ascending=False)
        
        print(f"\nTop 20 Features with Most Missing Values:")
        print(missing_df[missing_df['Missing Count'] > 0].head(20))
        
        # Numeric columns statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"\nNumeric Features: {len(numeric_cols)}")
        
        # Categorical columns
        cat_cols = self.df.select_dtypes(include=['object']).columns
        print(f"Categorical Features: {len(cat_cols)}")
        print(f"Categorical columns: {list(cat_cols)}")
        
        return self
    
    def preprocess_data(self, drop_threshold=0.95, variance_threshold=0.01):
        """
        Preprocess the data:
        - Remove identifier columns
        - Handle missing values
        - Encode categorical variables
        - Remove low-variance features
        - Scale features
        """
        print("\n" + "=" * 60)
        print("PREPROCESSING DATA")
        print("=" * 60)
        
        df_processed = self.df.copy()
        
        # Remove identifier and non-informative columns
        cols_to_drop = ['index', 'PATIENT_GUID']
        cols_to_drop = [c for c in cols_to_drop if c in df_processed.columns]
        if cols_to_drop:
            df_processed = df_processed.drop(columns=cols_to_drop)
            print(f"Dropped identifier columns: {cols_to_drop}")
        
        # Separate target
        self.y = df_processed['CANCER_FLAG'].values
        df_processed = df_processed.drop(columns=['CANCER_FLAG'])
        
        # Identify column types
        date_cols = [col for col in df_processed.columns if 'DATE' in col.upper()]
        text_cols = [col for col in df_processed.columns if 'TERMS' in col.upper() or 'CODES' in col.upper()]
        
        # Drop date columns (or extract features from them)
        print(f"Dropping {len(date_cols)} date columns and {len(text_cols)} text/code columns")
        df_processed = df_processed.drop(columns=date_cols + text_cols, errors='ignore')
        
        # Handle categorical columns
        cat_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        print(f"\nEncoding {len(cat_cols)} categorical columns: {cat_cols}")
        
        for col in cat_cols:
            le = LabelEncoder()
            df_processed[col] = df_processed[col].fillna('Unknown')
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        
        # Handle missing values in numeric columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        # Drop columns with too many missing values
        missing_pct = df_processed[numeric_cols].isnull().sum() / len(df_processed)
        cols_high_missing = missing_pct[missing_pct > drop_threshold].index.tolist()
        if cols_high_missing:
            df_processed = df_processed.drop(columns=cols_high_missing)
            print(f"Dropped {len(cols_high_missing)} columns with >{drop_threshold*100}% missing values")
        
        # Impute remaining missing values with median
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
        
        # Remove low variance features
        variances = df_processed.var()
        low_var_cols = variances[variances < variance_threshold].index.tolist()
        if low_var_cols:
            df_processed = df_processed.drop(columns=low_var_cols)
            print(f"Dropped {len(low_var_cols)} low-variance features")
        
        # Remove highly correlated features (correlation > 0.95)
        corr_matrix = df_processed.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
        # Only remove up to half to avoid losing too much information
        high_corr_cols = high_corr_cols[:len(high_corr_cols)//2]
        if high_corr_cols:
            df_processed = df_processed.drop(columns=high_corr_cols)
            print(f"Dropped {len(high_corr_cols)} highly correlated features")
        
        self.X = df_processed.values
        self.feature_names = df_processed.columns.tolist()
        
        print(f"\nFinal dataset shape: {self.X.shape}")
        print(f"Features: {len(self.feature_names)}")
        
        return self
    
    def split_data(self, test_size=0.2):
        """Split data into training and testing sets."""
        print("\n" + "=" * 60)
        print("SPLITTING DATA")
        print("=" * 60)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=RANDOM_STATE,
            stratify=self.y
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        print(f"Training class distribution: {np.bincount(self.y_train)}")
        print(f"Testing class distribution: {np.bincount(self.y_test)}")
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return self
    
    def handle_imbalance(self, method='smote'):
        """Handle class imbalance using various techniques."""
        print("\n" + "=" * 60)
        print(f"HANDLING CLASS IMBALANCE ({method.upper()})")
        print("=" * 60)
        
        print(f"Before resampling: {np.bincount(self.y_train)}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=RANDOM_STATE)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=RANDOM_STATE)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=RANDOM_STATE)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=RANDOM_STATE)
        else:
            print("No resampling applied")
            return self
        
        self.X_train_resampled, self.y_train_resampled = sampler.fit_resample(
            self.X_train, self.y_train
        )
        
        print(f"After resampling: {np.bincount(self.y_train_resampled)}")
        
        return self
    
    def select_features(self, n_features=100, method='mutual_info'):
        """Select top features using various methods."""
        print("\n" + "=" * 60)
        print(f"FEATURE SELECTION ({method.upper()})")
        print("=" * 60)
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(n_features, self.X_train.shape[1]))
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=min(n_features, self.X_train.shape[1]))
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
            selector = RFE(estimator, n_features_to_select=min(n_features, self.X_train.shape[1]), step=10)
        else:
            print("No feature selection applied")
            return self
        
        # Use original training data for feature selection (not resampled)
        selector.fit(self.X_train, self.y_train)
        
        # Store the selector for later use in prediction
        self.feature_selector = selector
        
        # Get selected feature indices
        if hasattr(selector, 'get_support'):
            selected_mask = selector.get_support()
            self.selected_features = [self.feature_names[i] for i, selected in enumerate(selected_mask) if selected]
        
        # Transform data
        self.X_train = selector.transform(self.X_train)
        self.X_test = selector.transform(self.X_test)
        
        if hasattr(self, 'X_train_resampled'):
            self.X_train_resampled = selector.transform(self.X_train_resampled)
        
        print(f"Selected {self.X_train.shape[1]} features")
        print(f"Top 100 selected features: {self.selected_features[:100] if hasattr(self, 'selected_features') else 'N/A'}")
        
        return self
    
    def get_models(self):
        """Get dictionary of models to train."""
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                random_state=RANDOM_STATE
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE
            ),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500,
                random_state=RANDOM_STATE, early_stopping=True
            ),
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                scale_pos_weight=len(self.y_train[self.y_train==0]) / len(self.y_train[self.y_train==1]),
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            )
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.1,
                auto_class_weights='Balanced', random_state=RANDOM_STATE, verbose=0
            )
        
        return models
    
    def train_and_evaluate(self, use_resampled=True):
        """Train multiple models and evaluate their performance."""
        print("\n" + "=" * 60)
        print("TRAINING AND EVALUATING MODELS")
        print("=" * 60)
        
        models = self.get_models()
        
        # Use resampled data if available
        if use_resampled and hasattr(self, 'X_train_resampled'):
            X_train_use = self.X_train_resampled
            y_train_use = self.y_train_resampled
            print("Using RESAMPLED training data")
        else:
            X_train_use = self.X_train
            y_train_use = self.y_train
            print("Using ORIGINAL training data")
        
        results = []
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = datetime.now()
            
            try:
                # Train model
                model.fit(X_train_use, y_train_use)
                
                # Predict
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, zero_division=0)
                recall = recall_score(self.y_test, y_pred, zero_division=0)
                f1 = f1_score(self.y_test, y_pred, zero_division=0)
                
                if y_pred_proba is not None:
                    roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                    avg_precision = average_precision_score(self.y_test, y_pred_proba)
                else:
                    roc_auc = None
                    avg_precision = None
                
                # Confusion matrix
                cm = confusion_matrix(self.y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp)
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                result = {
                    'Model': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall (Sensitivity)': recall,
                    'Specificity': specificity,
                    'F1 Score': f1,
                    'ROC AUC': roc_auc,
                    'Avg Precision': avg_precision,
                    'True Positives': tp,
                    'False Positives': fp,
                    'True Negatives': tn,
                    'False Negatives': fn,
                    'Training Time (s)': training_time
                }
                results.append(result)
                
                # Store model
                self.results[name] = {
                    'model': model,
                    'metrics': result,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
                print(f"  Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC AUC: {roc_auc_str}")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
        
        # Create results DataFrame
        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values('ROC AUC', ascending=False)
        
        print("\n" + "=" * 60)
        print("MODEL COMPARISON RESULTS")
        print("=" * 60)
        print(self.results_df.to_string(index=False))
        
        # Find best model
        best_idx = self.results_df['ROC AUC'].idxmax()
        self.best_model_name = self.results_df.loc[best_idx, 'Model']
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\n🏆 Best Model: {self.best_model_name} (ROC AUC: {self.results_df.loc[best_idx, 'ROC AUC']:.4f})")
        
        return self
    
    def hyperparameter_tuning(self, model_name=None):
        """Perform hyperparameter tuning on the best model."""
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING")
        print("=" * 60)
        
        if model_name is None:
            model_name = self.best_model_name
        
        print(f"Tuning: {model_name}")
        
        # Use resampled data if available
        if hasattr(self, 'X_train_resampled'):
            X_train_use = self.X_train_resampled
            y_train_use = self.y_train_resampled
        else:
            X_train_use = self.X_train
            y_train_use = self.y_train
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 100]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 150, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_samples_split': [2, 5]
            },
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['saga']
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}. Skipping tuning.")
            return self
        
        # Get base model
        models = self.get_models()
        base_model = models[model_name]
        
        # Perform RandomizedSearchCV for faster tuning
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        search = RandomizedSearchCV(
            base_model,
            param_grids[model_name],
            n_iter=20,
            cv=cv,
            scoring='roc_auc',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train_use, y_train_use)
        
        print(f"\nBest Parameters: {search.best_params_}")
        print(f"Best CV ROC AUC: {search.best_score_:.4f}")
        
        # Evaluate tuned model on test set
        y_pred = search.best_estimator_.predict(self.X_test)
        y_pred_proba = search.best_estimator_.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\nTuned Model Test Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        
        # Update best model if improved
        if roc_auc > self.results_df['ROC AUC'].max():
            self.best_model = search.best_estimator_
            self.best_model_name = f"{model_name} (Tuned)"
            print(f"\n✅ Tuned model is now the best model!")
        
        self.tuned_model = search.best_estimator_
        self.tuned_params = search.best_params_
        
        return self
    
    def create_ensemble(self):
        """Create an ensemble of top performing models."""
        print("\n" + "=" * 60)
        print("CREATING ENSEMBLE MODEL")
        print("=" * 60)
        
        # Get top 3 models by ROC AUC
        top_models = self.results_df.head(3)['Model'].tolist()
        print(f"Top models for ensemble: {top_models}")
        
        estimators = [(name, self.results[name]['model']) for name in top_models]
        
        # Voting classifier (soft voting for probability averaging)
        ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        
        # Use resampled data if available
        if hasattr(self, 'X_train_resampled'):
            X_train_use = self.X_train_resampled
            y_train_use = self.y_train_resampled
        else:
            X_train_use = self.X_train
            y_train_use = self.y_train
        
        # Train ensemble
        ensemble.fit(X_train_use, y_train_use)
        
        # Evaluate
        y_pred = ensemble.predict(self.X_test)
        y_pred_proba = ensemble.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\nEnsemble Model Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        
        # Update best model if improved
        if roc_auc > self.results_df['ROC AUC'].max():
            self.best_model = ensemble
            self.best_model_name = "Ensemble"
            print(f"\n✅ Ensemble is now the best model!")
        
        self.ensemble_model = ensemble
        
        return self
    
    def final_evaluation(self):
        """Perform final evaluation and generate detailed report."""
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        
        print(f"\n🏆 Final Best Model: {self.best_model_name}")
        
        # Make predictions
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Cancer', 'Cancer']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\n📊 Confusion Matrix:")
        print(cm)
        
        # Detailed metrics
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n📈 Detailed Metrics:")
        print(f"  True Positives (TP): {tp} - Cancer patients correctly identified")
        print(f"  True Negatives (TN): {tn} - Non-cancer patients correctly identified")
        print(f"  False Positives (FP): {fp} - Non-cancer patients incorrectly flagged")
        print(f"  False Negatives (FN): {fn} - Cancer patients missed")
        
        print(f"\n  Sensitivity (Recall): {tp/(tp+fn):.4f} - Ability to detect cancer")
        print(f"  Specificity: {tn/(tn+fp):.4f} - Ability to identify non-cancer")
        print(f"  Positive Predictive Value: {tp/(tp+fp):.4f} - Probability of cancer given positive test")
        print(f"  Negative Predictive Value: {tn/(tn+fn):.4f} - Probability of no cancer given negative test")
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"\n  Optimal Threshold: {optimal_threshold:.4f}")
        
        # Predictions with optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        cm_optimal = confusion_matrix(self.y_test, y_pred_optimal)
        tn_opt, fp_opt, fn_opt, tp_opt = cm_optimal.ravel()
        
        print(f"\n📊 Metrics at Optimal Threshold ({optimal_threshold:.4f}):")
        print(f"  Sensitivity (Recall): {tp_opt/(tp_opt+fn_opt):.4f}")
        print(f"  Specificity: {tn_opt/(tn_opt+fp_opt):.4f}")
        print(f"  F1 Score: {f1_score(self.y_test, y_pred_optimal):.4f}")
        
        return self
    
    def plot_results(self, save_path=None):
        """Generate visualization plots."""
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Model Comparison - ROC AUC
        ax1 = axes[0, 0]
        models = self.results_df['Model'].values
        roc_aucs = self.results_df['ROC AUC'].values
        colors = ['green' if m == self.best_model_name else 'steelblue' for m in models]
        bars = ax1.barh(models, roc_aucs, color=colors)
        ax1.set_xlabel('ROC AUC Score')
        ax1.set_title('Model Comparison (ROC AUC)')
        ax1.set_xlim(0.5, 1.0)
        for bar, val in zip(bars, roc_aucs):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center')
        
        # 2. ROC Curves
        ax2 = axes[0, 1]
        for name, data in self.results.items():
            if data['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, data['y_pred_proba'])
                roc_auc = roc_auc_score(self.y_test, data['y_pred_proba'])
                ax2.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves')
        ax2.legend(loc='lower right', fontsize=8)
        
        # 3. Precision-Recall Curves
        ax3 = axes[0, 2]
        for name, data in self.results.items():
            if data['y_pred_proba'] is not None:
                precision, recall, _ = precision_recall_curve(self.y_test, data['y_pred_proba'])
                avg_precision = average_precision_score(self.y_test, data['y_pred_proba'])
                ax3.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curves')
        ax3.legend(loc='upper right', fontsize=8)
        
        # 4. Confusion Matrix - Best Model
        ax4 = axes[1, 0]
        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                   xticklabels=['No Cancer', 'Cancer'],
                   yticklabels=['No Cancer', 'Cancer'])
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        ax4.set_title(f'Confusion Matrix - {self.best_model_name}')
        
        # 5. Metric Comparison
        ax5 = axes[1, 1]
        metrics = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1 Score']
        x = np.arange(len(metrics))
        width = 0.35
        
        # Top 2 models
        top2 = self.results_df.head(2)['Model'].tolist()
        for i, name in enumerate(top2):
            values = [self.results[name]['metrics'][m] for m in metrics]
            ax5.bar(x + i*width, values, width, label=name)
        
        ax5.set_ylabel('Score')
        ax5.set_title('Metrics Comparison (Top 2 Models)')
        ax5.set_xticks(x + width/2)
        ax5.set_xticklabels(metrics, rotation=45, ha='right')
        ax5.legend()
        ax5.set_ylim(0, 1.1)
        
        # 6. Feature Importance (if available)
        ax6 = axes[1, 2]
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            # Get top 20 features
            if hasattr(self, 'selected_features'):
                feature_names = self.selected_features
            else:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            indices = np.argsort(importances)[-20:]
            ax6.barh(range(len(indices)), importances[indices], color='steelblue')
            ax6.set_yticks(range(len(indices)))
            ax6.set_yticklabels([feature_names[i][:30] for i in indices], fontsize=8)
            ax6.set_xlabel('Importance')
            ax6.set_title(f'Top 20 Feature Importances - {self.best_model_name}')
        else:
            ax6.text(0.5, 0.5, 'Feature importance not available\nfor this model type',
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Feature Importances')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {save_path}")
        
        plt.show()
        
        return self
    
    def save_model(self, filepath=None):
        """Save the best model to disk."""
        if filepath is None:
            filepath = 'lung_cancer_best_model.joblib'
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'selected_features': getattr(self, 'selected_features', None),
            'feature_selector': getattr(self, 'feature_selector', None)
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to: {filepath}")
        
        return self


def main():
    """Main function to run the lung cancer prediction pipeline."""
    
    # Data path
    data_path = "13-all_patients_lungCancer_trend_medication_observation_5M_25M_20yrs-1yr_NoEthnicity.csv"
    
    # Initialize predictor
    predictor = LungCancerPredictor(data_path)
    
    # Run pipeline
    (predictor
     .load_data()
     .explore_data()
     .preprocess_data(drop_threshold=0.90)
     .split_data(test_size=0.2)
     .handle_imbalance(method='smote')
     .select_features(n_features=150, method='mutual_info')
     .train_and_evaluate(use_resampled=True)
     .hyperparameter_tuning()
     .create_ensemble()
     .final_evaluation()
     .plot_results(save_path='lung_cancer_model_results.png')
     .save_model('lung_cancer_best_model.joblib'))
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return predictor


if __name__ == "__main__":
    predictor = main()
