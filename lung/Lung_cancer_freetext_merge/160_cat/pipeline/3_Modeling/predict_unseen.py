"""
Lung Cancer Prediction - Inference Script
==========================================
Use a trained model to predict lung cancer on new/unseen data.
If cancer_class labels are present, calculates comprehensive evaluation metrics.
Includes explainability - shows WHY the model predicted cancer for each patient.

Two input modes (exactly one):
    --data <csv>        : a PRE-BUILT FE matrix (same columns as training) — applies saved transforms + scores.
    --events-sql <sql>  : RAW lifetime events — runs the SAME FE as the held-out path (build_features.build,
                          fit_split=False, restricted to the model's features, reusing the persisted train
                          z-stats/top_cats/value_codes) then scores. One call: raw events -> label.
                          Requires --window (12mo|1mo) and --years (model lookback).

Usage:
    python predict_unseen.py --data <path_to_csv> --model <model.joblib>
    python predict_unseen.py --events-sql <events.sql> --window 12mo --years 10 --model <model.joblib>

Example:
    python3 predict_unseen_input_explainability_unified_v3.py --data 16-all_patients_trend_lungCancer_noCancer_20yrs_1_1_v2.csv  --output predictions_16.csv --model lung_cancer_best_mode_unified_v2.joblib
    python3 predict_cancer.py --data test_data.csv --model lung_cancer_best_model.joblib
    python3 predict_cancer.py --data test_data.csv --plot
    python3 predict_cancer.py --data test_data.csv --save-plot evaluation_results.png
    python3 predict_cancer.py --data test_data.csv --output my_predictions.csv
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import json
import os
import sys
from datetime import datetime

# Visualization
import matplotlib
matplotlib.use("Agg")          # headless-safe: inference/explainability runs as a batch job (no display)
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing is applied via the TRAIN-fitted transforms loaded from the model
# (encoders + impute medians + scaler) — no fit/refit happens here.

# Metrics
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[warn] SHAP not available. Install with: pip install shap")
    print("  Explainability features will be limited.")

# NOTE: warnings are intentionally NOT blanket-suppressed — they surface real issues (deprecations,
# NaN/divide-by-zero). Scope any specific suppression narrowly at its call site instead.


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
        self.calibrated_model = self.model_data.get('calibrated_model')   # training per-age-band isotonic, if present
        self.model_name = self.model_data.get('model_name', 'Unknown')
        self.scaler = self.model_data.get('scaler', None)
        self.feature_names = self.model_data.get('feature_names', None)
        self.selected_features = self.model_data.get('selected_features', None)
        self.feature_selector = self.model_data.get('feature_selector', None)
        # TRAIN-fitted transforms (new models) -> apply identically to held-out (no refit/leakage).
        self.encoders = self.model_data.get('encoders', None)
        self.impute_medians = self.model_data.get('impute_medians', None)
        self.value_cols = self.model_data.get('value_cols', None)   # fill w/ median; counts -> 0

        # C1 guard: re-build FE with the SAME flags this model was TRAINED under. Without this, a model
        # trained with an arm ON (e.g. CLINICAL_FE=1) but served under the default-OFF config would silently
        # NaN-impute its trained features. Align config.* from the model's stamped fe_config (warn on change).
        _fec = self.model_data.get('fe_config')
        if _fec:
            try:
                import config as _C
                for _k, _v in _fec.items():
                    if _v is not None and getattr(_C, _k, None) != _v:
                        print(f"  [serve] aligning FE flag {_k}: config={getattr(_C, _k, None)} -> model-trained={_v}")
                        setattr(_C, _k, _v)
            except Exception as _fe:
                print(f"  [serve][warn] could not align fe_config ({_fe}) — verify FE flags match training")

        print(f"Model loaded: {self.model_name}")
        print(f"  Model type: {type(self.model).__name__}")
        if self.feature_names:
            print(f"  Features after preprocessing: {len(self.feature_names)}")
        if self.selected_features:
            print(f"  Features after selection: {len(self.selected_features)}")
        if self.feature_selector:
            print(f"  Feature selector: Available")
        else:
            print(f"  [warn] Feature selector: Not available (will use selected_features)")
        
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
        np.ndarray or None : True labels if cancer_class present
        """
        print("\n" + "=" * 60)
        print("PREPROCESSING DATA")
        print("=" * 60)
        
        df_processed = df.copy()
        
        # Check for cancer_class and store if present
        has_labels = 'cancer_class' in df_processed.columns
        y_true = None
        if has_labels:
            y_true = df_processed['cancer_class'].values
            df_processed = df_processed.drop(columns=['cancer_class'])
            print("cancer_class labels found - will calculate evaluation metrics")
        else:
            print("ℹ cancer_class labels not found - will only generate predictions")
        
        # Step 1: Remove identifier columns (same as training)
        cols_to_drop = ['index', 'patient_guid']
        cols_to_drop = [c for c in cols_to_drop if c in df_processed.columns]
        if cols_to_drop:
            df_processed = df_processed.drop(columns=cols_to_drop)
            print(f"Dropped identifier columns: {cols_to_drop}")
        
        # Step 2: Drop RAW date / free-text / code-string columns (legacy CSV inputs), same as training.
        # Guard on NON-NUMERIC dtype so numeric ENGINEERED features whose names merely contain these
        # substrings (e.g. '<cat>_n_distinct_codes', 'g_distinct_*_codes') are NEVER dropped.
        _nonnum = set(df_processed.select_dtypes(exclude=[np.number]).columns)
        date_cols = [c for c in df_processed.columns if 'DATE' in c.upper() and c in _nonnum]
        text_cols = [c for c in df_processed.columns if ('TERMS' in c.upper() or 'CODES' in c.upper()) and c in _nonnum]
        df_processed = df_processed.drop(columns=date_cols + text_cols, errors='ignore')
        print(f"Dropped {len(date_cols)} raw date + {len(text_cols)} raw text/code columns (non-numeric only)")
        
        # Steps 3-8: apply the TRAIN-fitted preprocessing so the held-out set is processed
        # IDENTICALLY to training (no refit -> no leakage / distribution shift). The trainer saves
        # the encoders, impute medians, feature order and scaler with the model; we just apply them.
        if self.encoders is None or self.impute_medians is None or not self.feature_names:
            raise ValueError(
                "Model artifact lacks stored train transforms (encoders / impute_medians / "
                "feature_names). Retrain with the current lung_training.py so held-out scoring uses "
                "the SAME train-fitted preprocessing (no leakage).")
        for col, mp in self.encoders.items():                    # categorical -> train code map
            if col in df_processed.columns:
                df_processed[col] = (df_processed[col].fillna('Unknown').astype(str)
                                     .map(mp).fillna(-1))
        # schema-coverage guard. predict_unseen does NOT run feature engineering — it assumes the input
        # IS the categorized FE matrix and just applies the saved transforms. If fed raw events / a wrong
        # matrix, almost none of feature_names are present, the reindex below fills everything with median/0,
        # and the model returns a meaningless ~baseline score with NO error. Fail loud: require a minimum
        # fraction of the model's features to be present (non-null) in the input. (Override via env if a
        # legitimately sparse cohort trips it.) The correct upstream is build_features.build(fit_split=False,
        # keep_features=feature_names) — the same FE the held-out path runs (see 4_Heldout/evaluate_heldout.py).
        _min_cov = float(os.environ.get("PREDICT_MIN_SCHEMA_COVERAGE", "0.5"))
        _present = [c for c in self.feature_names if c in df_processed.columns and df_processed[c].notna().any()]
        _cov = len(_present) / max(1, len(self.feature_names))
        if _cov < _min_cov:
            raise ValueError(
                f"[predict] input has only {_cov:.0%} of the {len(self.feature_names)} model features present "
                f"(non-null) — looks like raw events or a wrong/mismatched matrix, NOT the categorized FE matrix "
                f"the model expects, so every feature would silently become median/0 and the score is meaningless. "
                f"Build features first via build_features.build(fit_split=False, keep_features=<feature_names>) "
                f"(the same FE the held-out path runs), or set PREDICT_MIN_SCHEMA_COVERAGE to override.")
        df_processed = df_processed.reindex(columns=self.feature_names)   # align to TRAIN cols/order
        df_processed = df_processed.apply(pd.to_numeric, errors='coerce')
        # VALUE cols -> TRAIN median; COUNT cols -> 0 (absent code/category = 0, not a 'typical' value)
        val = [c for c in self.feature_names if self.value_cols and c in set(self.value_cols)]
        if val:
            df_processed[val] = df_processed[val].fillna(self.impute_medians)
        # inf -> NaN before the final 0-fill (scaler-safe; parity with evaluate_heldout._transform_external)
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X = df_processed.values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        # fail loud: a non-finite scored matrix means a broken impute/scale (NaN >= thr -> silently label 0)
        assert np.isfinite(X).all(), "[predict] non-finite values in preprocessed matrix X (impute/scale bug)"
        print(f"Applied TRAIN-fitted transforms (encoders + median impute + scaler) "
              f"-> {X.shape[1]} features")
        
        # Step 9: Apply feature selection (same as training)
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
            print(f"Applied feature selection: {X.shape[1]} features selected")
        elif self.selected_features is not None:
            # Fallback: use selected feature names if selector not available
            # This might not work perfectly if column order changed
            print(f"[warn] Using selected feature names (feature selector not saved)")
            selected_indices = [self.feature_names.index(f) for f in self.selected_features if f in self.feature_names]
            X = X[:, selected_indices]
            print(f"  Selected {X.shape[1]} features by name matching")
        
        print(f"\nFinal preprocessed shape: {X.shape}")
        
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
        
        df = pd.read_parquet(data_path) if str(data_path).endswith(".parquet") else pd.read_csv(data_path)
        print(f"Loaded data: {df.shape[0]} samples, {df.shape[1]} columns")
        
        # Preprocess
        X, y_true = self.preprocess_data(df)
        
        # Make predictions
        print("\n" + "=" * 60)
        print("MAKING PREDICTIONS")
        print("=" * 60)
        
        y_pred = self.model.predict(X)
        
        # Get prediction probabilities if available (RAW = balanced-scale, ~50% prior)
        y_pred_proba = None
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X)[:, 1]

        # ---- Deployment calibration ----
        # Raw scores are on the balanced training scale (~50% prior) and far too high at real prevalence.
        # Deployment contract (pick ONE; never stack): (1) the held-out-fitted Platt recalibrator
        # `platt_calib_{h}.joblib` from 4_Heldout — prevalence-correct, the validated artifact; else
        # (2) the training per-age-band isotonic `calibrated_model` in the joblib. (The old code imported
        # a non-existent `deploy_calibration` module and silently shipped RAW balanced-prior scores.)
        y_pred_proba_calibrated = None
        _calib_source = "raw"     # which calibrator produced y_pred_proba_calibrated -> gates the threshold scale
        if y_pred_proba is not None:
            _platt_path = self.model_path.replace("model_", "platt_calib_")
            if os.path.exists(_platt_path):
                _platt = joblib.load(_platt_path)
                y_pred_proba_calibrated = _platt.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
                _calib_source = "platt_calibrated"
                print(f"Applied held-out Platt recalibration ({os.path.basename(_platt_path)}) — prevalence-correct.")
            elif self.calibrated_model is not None:
                _agecol = next((c for c in ("age_at_prediction", "g_age", "patient_age") if c in df.columns), None)
                _age = pd.to_numeric(df[_agecol], errors="coerce").values if _agecol else None
                y_pred_proba_calibrated = self.calibrated_model.predict_proba(X, _age)[:, 1]
                _calib_source = "isotonic"
                print("Applied training per-age-band isotonic (calibrated_model)"
                      f"{' [global only: no age column]' if _age is None else ''}.")
            else:
                print("[calib] no platt_calib or calibrated_model found -> RAW (balanced-prior) probabilities. "
                      "For deployment, fit/apply platt_calib via 4_Heldout/evaluate_heldout.py.")

        # OPERATING POINT: label at the held-out-chosen threshold on the CALIBRATED probability — NOT the
        # raw model's internal 0.5 cut (which is on the balanced-prior scale and wrong at real prevalence).
        # The threshold is the deployment artifact operating_threshold_{h}.json written by 4_Heldout. CRITICAL:
        # that threshold was chosen on the Platt-calibrated scale (the JSON records "scale"), so it is ONLY
        # valid against the Platt probability. Applying it to the isotonic-fallback or raw probability is a
        # silent SCALE MISMATCH that would mislabel patients — so we GATE on the calibrator that produced the
        # probability and refuse to cross scales (keep the raw labels + warn loud) rather than mis-score.
        if y_pred_proba_calibrated is not None:
            _thr_path = self.model_path.replace("model_", "operating_threshold_").replace(".joblib", ".json")
            if os.path.exists(_thr_path):
                with open(_thr_path) as _tf:
                    _meta = json.load(_tf)
                _thr = _meta.get("threshold", 0.5)
                _thr_scale = str(_meta.get("scale", "platt_calibrated"))
                if _calib_source == _thr_scale:
                    y_pred = (y_pred_proba_calibrated >= _thr).astype(int)
                    print(f"Labels at held-out operating threshold {_thr:.4f} ({_thr_scale} scale, {os.path.basename(_thr_path)}).")
                    # Operating-point refinement (env THRESHOLD_REFINE: 'age' default | 'density' | 'none').
                    # Both confounds (age, record-density) skew the single global cut. These are ALTERNATIVES,
                    # NOT stacked — combining them needs joint calibration (max-conservative is wrong for the
                    # sparse/young who need a LOWER cut). 'none' keeps the global cut.
                    _refine = os.environ.get("THRESHOLD_REFINE", "age").strip().lower()
                    if _refine == "age":
                        # Per-age-band: relabel each patient at its age band's threshold (recovers old-band
                        # specificity); bands without a band-specific cut keep the global. Band from ageband_* one-hots.
                        _age_path = self.model_path.replace("model_", "operating_threshold_by_age_").replace(".joblib", ".json")
                        _abcols = [c for c in df.columns if c.startswith("ageband_")]
                        if os.path.exists(_age_path) and _abcols:
                            with open(_age_path) as _af:
                                _am = json.load(_af)
                            if str(_am.get("scale", "platt_calibrated")) == _calib_source:
                                _bb = _am.get("by_band", {}); _gc = float(_am.get("global", _thr))
                                _A = df[_abcols].fillna(0.0).to_numpy()
                                _lab = np.array([c[len("ageband_"):] for c in _abcols])
                                _band = np.where(_A.sum(1) > 0, _lab[_A.argmax(1)], "Unknown")
                                _thr_vec = np.array([float(_bb.get(b, _gc)) for b in _band], dtype=float)
                                y_pred = (np.asarray(y_pred_proba_calibrated) >= _thr_vec).astype(int)
                                print(f"Labels REFINED by age band ({os.path.basename(_age_path)}): "
                                      f"{len(_bb)} band-specific cut(s), other bands at global {_gc:.4f}.")
                    elif _refine == "density":
                        # Per-record-density-quartile: density = sum of *_count cols; quartile via the PERSISTED
                        # calib edges; relabel at each quartile's cut (lifts sparse-patient sens / rich-patient
                        # spec). Quartiles without a band-specific cut keep the global.
                        _den_path = self.model_path.replace("model_", "operating_threshold_by_density_").replace(".joblib", ".json")
                        _ccols = [c for c in df.columns if c.endswith("_count")]
                        if os.path.exists(_den_path) and _ccols:
                            with open(_den_path) as _df2:
                                _dm = json.load(_df2)
                            if str(_dm.get("scale", "platt_calibrated")) == _calib_source:
                                _bq = _dm.get("by_quartile", {}); _edges = _dm.get("edges", []); _gc = float(_dm.get("global", _thr))
                                _dens = np.nan_to_num(df[_ccols].to_numpy(dtype=float)).sum(1)
                                _q = np.digitize(_dens, _edges)
                                _thr_vec = np.array([float(_bq.get(str(int(qq)), _gc)) for qq in _q], dtype=float)
                                y_pred = (np.asarray(y_pred_proba_calibrated) >= _thr_vec).astype(int)
                                print(f"Labels REFINED by density quartile ({os.path.basename(_den_path)}): "
                                      f"{len(_bq)} quartile cut(s), others at global {_gc:.4f}.")
                else:
                    print(f"[predict] SCALE MISMATCH: operating_threshold is on '{_thr_scale}' scale but the "
                          f"calibrated probability is '{_calib_source}' — NOT applying it (would mislabel). "
                          f"Keeping the raw model's 0.5-cut labels. Run 4_Heldout/evaluate_heldout.py to produce "
                          f"a matching {os.path.basename(_platt_path)}.")
            else:
                print("[predict] no operating_threshold_*.json -> labels at the raw model's 0.5 cut "
                      "(run 4_Heldout/evaluate_heldout.py to fit the deployable operating point).")

        # Headline 'probabilities' = the calibrated (deployment) probability when available; raw kept too.
        results = {
            'predictions': y_pred,
            'probabilities': y_pred_proba_calibrated if y_pred_proba_calibrated is not None else y_pred_proba,
            'probabilities_raw': y_pred_proba,
            'probabilities_calibrated': y_pred_proba_calibrated,
            'n_samples': len(y_pred),
            'n_positive': int(np.sum(y_pred == 1)),
            'n_negative': int(np.sum(y_pred == 0)),
            'has_labels': y_true is not None
        }
        
        print(f"\nPrediction Summary:")
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

    def predict_from_events(self, events_sql, window, years):
        """ONE-CALL inference from RAW EVENTS. Runs the SAME feature engineering as training and
        the held-out eval — build_features.build(fit_split=False, keep_features=<model feature_names>), which
        reloads the persisted TRAIN zero-fit artifacts (z-stats / top_cats / value_codes), so there is no
        refit and no train/serve skew — then scores via the normal matrix path (predict()). This closes the
        gap where predict_unseen assumed its input was already the wide FE matrix: deployment now goes raw
        events -> label in a single call, identical FE+transform to 4_Heldout/evaluate_heldout.py.

        `events_sql` = path to the SQL that returns the patient(s) lifetime events (as the held-out SQL does);
        `window` = '12mo'|'1mo'; `years` = the model's lookback. Requires the model's feature_names + the
        persisted train artifacts to exist (build raises loud if they don't)."""
        import importlib.util
        if not self.feature_names:
            raise ValueError("[predict] model has no feature_names — cannot restrict FE to the model's features")
        here = os.path.dirname(os.path.abspath(__file__))
        fe_dir = os.path.join(os.path.dirname(here), "2_FE")
        if os.path.dirname(here) not in sys.path:
            sys.path.insert(0, os.path.dirname(here))           # V3 root: build_features needs config/artifacts/splits

        def _load(path, name):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
            spec.loader.exec_module(mod); return mod
        # register lung_training first so any pickled CalibratedLungModel unpickles cleanly
        _load(os.path.join(here, "lung_training.py"), "lung_training")
        fe = _load(os.path.join(fe_dir, "build_features.py"), "fe_engine")
        print(f"[predict] FE from raw events: build(fit_split=False, keep_features) {window} {years}yr "
              f"(same FE as held-out; reuses persisted train z-stats/top_cats/value_codes) ...")
        mat, _ = fe.build(window, sql_path=events_sql, fit_split=False, years=years,
                          keep_features=self.feature_names)
        # route the FE matrix through the normal scoring path (transform + Platt + operating threshold).
        # PARQUET not CSV: a CSV round-trip lossily re-formats the z-scored value features (float drift vs the
        # parity-gated parquet path). predict() detects the extension and reads parquet exactly.
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), f"predict_fe_{window}_{os.getpid()}.parquet")
        mat.to_parquet(tmp, index=False)
        try:
            return self.predict(tmp)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive evaluation metrics.
        """
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        
        # Confusion Matrix — labels=[0,1] forces a 2x2 even on a single-class batch (e.g. an all-negative
        # labelled cohort), so `cm.ravel()` always yields 4 values instead of crashing on the unpack.
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity (PPV is computed below as tp/(tp+fp))
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
        print("\nConfusion Matrix:")
        print(f"                    Predicted")
        print(f"                 No Cancer  Cancer")
        print(f"Actual No Cancer    {tn:5d}    {fp:5d}")
        print(f"Actual Cancer       {fn:5d}    {tp:5d}")
        
        print(f"\nPerformance Metrics:")
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
        print("\nClassification Report:")
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
                
                # SHAP — TREE-EXACT ONLY, via the repo's single canonical path
                # (explainability._shap_values): a tree model uses TreeExplainer; a soft-voting ensemble
                # of trees uses exact per-tree weighted SHAP. Requires xgboost<3 (pinned in
                # requirements.txt — shap-0.49 can't parse xgboost 3.x's vector base_score). There is NO
                # KernelExplainer — a non-tree model raises, which is caught below and degrades to the
                # feature-importance fallback. Output is (n, n_features) for the positive class.
                here = os.path.dirname(os.path.abspath(__file__))
                if here not in sys.path:
                    sys.path.insert(0, here)
                import explainability as _xai
                model_type = type(self.model).__name__
                shap_values, _ = _xai._shap_values(self.model, X)
                self.explainer = None                       # canonical path holds no explainer object
                print(f"SHAP values computed for {len(X)} samples ({model_type}, tree-exact)")
                
            except Exception as e:
                # honor explainability._shap_values' fail-loud contract — it raises on a NON-tree model
                # precisely so we never silently ship an approximate explainer. Re-raise that case instead of
                # masking it with the feature-importance fallback; only genuine SHAP runtime hiccups degrade.
                if "not a tree model" in str(e) or "KernelExplainer has been removed" in str(e):
                    raise
                print(f"[warn] SHAP computation failed: {str(e)}")
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
        print(f"Generated explanations for {cancer_predictions} cancer predictions")
        
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
            print("\n[warn] Cannot plot evaluation metrics without true labels.")
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
            print(f"Plot saved to: {save_path}")
        
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
        if 'patient_guid' in df_original.columns:
            output_df['patient_guid'] = df_original['patient_guid']
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
                labels=['Low', 'Moderate', 'High', 'Very High'],
                include_lowest=True   # a probability of exactly 0.0 -> 'Low' (not NaN)
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
        print(f"Predictions saved to: {output_path}")
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
        default=None,
        help='Path to a pre-built FE-matrix CSV (same columns as training). Either --data OR --events-sql.'
    )
    # raw-events -> label in one call. --events-sql points at the SQL returning the patient(s)
    # lifetime events; predict_from_events runs the SAME FE as the held-out path then scores.
    parser.add_argument('--events-sql', type=str, default=None,
                        help='Path to a SQL that returns raw lifetime events; runs FE then scores (one call).')
    parser.add_argument('--window', type=str, default=None, choices=['12mo', '1mo'],
                        help='Horizon for --events-sql FE (required with --events-sql).')
    parser.add_argument('--years', type=int, default=None,
                        help='Lookback years for --events-sql FE (required with --events-sql; matches the model).')
    
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
    print(f"Input: {args.data or args.events_sql}")   # the chosen source (FE-matrix CSV or raw-events SQL)
    print(f"Model file: {args.model}")
    
    # Exactly one input mode: a pre-built FE matrix (--data) OR raw events (--events-sql, runs FE first).
    if bool(args.data) == bool(args.events_sql):
        parser.error("provide exactly one of --data (pre-built FE matrix) or --events-sql (raw events -> FE -> score)")

    # Initialize predictor
    predictor = CancerPredictor(args.model)

    # Make predictions
    if args.events_sql:
        if not args.window or not args.years:
            parser.error("--events-sql requires --window (12mo|1mo) and --years (model lookback)")
        results = predictor.predict_from_events(args.events_sql, args.window, args.years)
        _src = args.events_sql
    else:
        results = predictor.predict(args.data)
        _src = args.data

    # Generate plots if requested and labels are available
    if args.plot or args.save_plot:
        predictor.plot_results(results, save_path=args.save_plot)

    # Save predictions
    predictor.save_predictions(results, _src, args.output)
    
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()
