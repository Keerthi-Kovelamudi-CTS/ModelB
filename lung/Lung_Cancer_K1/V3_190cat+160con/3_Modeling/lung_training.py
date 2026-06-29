"""
Lung Cancer Prediction Model
============================
A comprehensive machine learning pipeline to predict lung cancer (cancer_class)
from patient medical records with high accuracy.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so `import lung_metrics` works under importlib
import lung_metrics                                              # AUROC/AUPRC/Sens/Spec/PPV/NPV with 95% bootstrap CIs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # V3 root
import artifacts                                                 # atomic model write (no partial joblib on interruption)
try:
    import config as C                                           # SINGLE source of tuning knobs (seed/TUNE/EXCLUDE_MODELS/…)
except Exception:                                                # standalone use without config on path -> getattr defaults
    C = None

# Visualization
import matplotlib
matplotlib.use("Agg")          # headless-safe: training runs as a batch job on VMs/containers (no display)
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE

# Models
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
# NOTE: imbalanced-learn (SMOTE/ADASYN/…) is imported LAZILY inside handle_imbalance() only when a
# resampling method is requested — the default handle_imbalance="none" (cost-weighting) needs no
# imblearn, so it stays an OPTIONAL dependency (see requirements.txt).

# Hyperparameter Tuning
from sklearn.base import clone
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss
)

# Calibration (isotonic, per-age-band)
from sklearn.isotonic import IsotonicRegression

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

# NOTE: do NOT blanket-suppress warnings here — sklearn deprecations, divide-by-zero/NaN RuntimeWarnings
# and dtype warnings are real signal. (This module is also imported by evaluate_heldout via importlib, so
# a global filter would silence warnings for that process too.) Scope any needed suppression narrowly.

# Tuning knobs sourced from the central config.py (single source; logged in the run manifest). getattr
# falls back to the literal only when config didn't import (true standalone use, C is None).
RANDOM_STATE = getattr(C, "RANDOM_STATE", 42)
N_TUNING_TRIALS = getattr(C, "N_TUNING_TRIALS", 100)   # Optuna trials per hyperparameter search
TUNING_CV_FOLDS = getattr(C, "TUNING_CV_FOLDS", 5)     # inner CV folds used during tuning (honest check stays on held-out)
TUNE_TOP_N = getattr(C, "TUNE_TOP_N", 5)               # tune the top-N models AND ensemble them (so tuning feeds the deployed ensemble)
CORR_MAX_FEATURES = 10000  # skip the high-correlation drop above this width (keeps the F×F corr matrix memory-safe)
# the metric that drives MODEL SELECTION / ranking / ensemble membership. AUPRC ('Avg Precision') is
# the decision-relevant metric at the deployment prevalence (~1:100) — far more sensitive to ranking the
# rare positives than ROC AUC. Tuning already optimizes average_precision (cross_val_score below), so this
# aligns selection with tuning. Config-overridable; set to 'ROC AUC' to restore the old behaviour.
SELECTION_METRIC = getattr(C, "SELECTION_METRIC", "Avg Precision")
np.random.seed(RANDOM_STATE)


class _MaskSelector:
    """Lightweight column-mask selector, API-compatible with sklearn selectors
    (transform / get_support) so it drops into save_model + the prediction script."""

    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=bool)

    def transform(self, X):
        return X[:, self.mask]

    def get_support(self):
        return self.mask


class CalibratedLungModel:
    """Wraps a fitted base estimator with isotonic calibration.

    - ``global_iso``  : IsotonicRegression fitted on all calibration rows.
    - ``band_isos``   : {band_index -> IsotonicRegression} fitted within an
                        age band; thin bands fall back to ``global_iso``.
    Calibration is monotonic, so within-band ranking is preserved; it maps raw
    scores onto trustworthy probabilities (what a clinical threshold needs).
    """

    def __init__(self, base_model, global_iso, band_edges, band_isos):
        self.base_model = base_model
        self.global_iso = global_iso
        self.band_edges = list(band_edges)
        self.band_isos = dict(band_isos or {})

    def _raw(self, X):
        return self.base_model.predict_proba(X)[:, 1]

    def _apply(self, raw, age):
        if self.global_iso is None:
            return raw
        if age is None or not self.band_isos:
            return self.global_iso.predict(raw)
        out = np.empty_like(raw, dtype=float)
        bidx = np.digitize(np.asarray(age, dtype=float), self.band_edges[1:-1])
        for b in np.unique(bidx):
            m = bidx == b
            iso = self.band_isos.get(int(b), self.global_iso)
            out[m] = iso.predict(raw[m])
        return out

    def predict_proba(self, X, age=None):
        p = self._apply(self._raw(X), age)
        return np.column_stack([1.0 - p, p])

    def predict(self, X, age=None, threshold=0.5):
        return (self.predict_proba(X, age)[:, 1] >= threshold).astype(int)


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
        # calibration support
        self.age = None              # patient_age aligned to rows of self.X
        self.X_calib = None
        self.y_calib = None
        self.age_train = self.age_test = self.age_calib = None
        self.calibrated_model = None
        self._split = None            # per-row 'train'/'valid'/'test' from the stamped split column (set in preprocess_data)
        
    def load_data(self):
        """Load and perform initial data exploration."""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        self.df = (pd.read_parquet(self.data_path) if str(self.data_path).endswith(".parquet")
                   else pd.read_csv(self.data_path))           # stable matrix is parquet; csv still accepted
        self.df = self.df.replace([np.inf, -np.inf], np.nan)   # inf -> NaN so impute/scale never sees infinity

        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Total Samples: {self.df.shape[0]}")
        print(f"Total Features: {self.df.shape[1]}")
        
        # Target distribution
        print(f"\nTarget Distribution (cancer_class):")
        print(self.df['cancer_class'].value_counts())
        print(f"\nClass Imbalance Ratio: {self.df['cancer_class'].value_counts()[0] / self.df['cancer_class'].value_counts()[1]:.2f}:1")
        
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
    
    def preprocess_data(self, drop_threshold=0.95, variance_threshold=0.01, horizon=None):
        """LEAKAGE-FREE structural preprocessing only: drop identifier/date/text columns and
        separate the target. Every DATA-DEPENDENT step (categorical encoding, high-missing /
        low-variance / high-correlation dropping, median imputation, scaling) is deferred to
        split_data and fit on the TRAINING split only (then applied to test/calib and saved for
        held-out). This removes the train->test leakage those fitted transforms had when run on the
        full dataset before the split."""
        print("\n" + "=" * 60)
        print("PREPROCESSING (structural only; data-dependent fits happen on TRAIN in split_data)")
        print("=" * 60)

        df = self.df.copy()
        # DRIFT GUARD: verify the stamped split column matches the canonical split file per-patient before
        # we trust it (catches a cached/stale matrix whose stamp diverged from the canonical split — the
        # silent 'test patient in train' failure mode). The horizon is passed EXPLICITLY by run_v3
        # (it always knows it), so the guard is never skipped in the pipeline; filename inference is only a
        # last-resort fallback for a standalone run with a non-standard name.
        try:
            import splits as _splits
            _fname = os.path.basename(str(self.data_path))
            _h = horizon or next((h for h in (getattr(C, "HORIZONS", None) or ["12mo", "1mo"]) if f"_{h}_" in _fname), None)
            if _h is not None:
                _splits.verify_stamp(df, _h)
            else:
                print(f"[split] verify_stamp skipped: no horizon passed and could not infer from {_fname!r}")
        except Exception as _ve:
            if isinstance(_ve, ValueError) and "verify_stamp" in str(_ve):
                raise                                        # a genuine drift mismatch MUST hard-fail
            print(f"[split] verify_stamp skipped ({type(_ve).__name__}: {_ve})")
        # capture the PREDEFINED split (stamped by stability_select) before dropping id columns, so
        # split_data partitions by it — the split travels WITH the data (no re-derivation, no leakage).
        self._split = df['split'].astype(str).values if 'split' in df.columns else None
        df = df.drop(columns=[c for c in ('index', 'patient_guid', 'split') if c in df.columns], errors='ignore')
        self.y = df['cancer_class'].astype(int).values     # label is 0/1; force int (np.bincount etc.)
        df = df.drop(columns=['cancer_class'])
        # Drop RAW date / free-text / code-string columns (legacy CSV-matrix inputs). Guard on NON-NUMERIC
        # dtype so we NEVER drop a numeric ENGINEERED feature whose NAME merely contains these substrings —
        # e.g. category-diversity '<cat>_n_distinct_codes' / 'g_distinct_*_codes' (all float64). The bare
        # substring match previously silently dropped those engineered features at train time.
        _nonnum = set(df.select_dtypes(exclude=[np.number]).columns)
        date_cols = [c for c in df.columns if 'DATE' in c.upper() and c in _nonnum]
        text_cols = [c for c in df.columns if ('TERMS' in c.upper() or 'CODES' in c.upper()) and c in _nonnum]
        df = df.drop(columns=date_cols + text_cols, errors='ignore')
        print(f"Dropped {len(date_cols)} raw date + {len(text_cols)} raw text/code columns (non-numeric only)")

        self.cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        self._drop_threshold = drop_threshold
        self._variance_threshold = variance_threshold
        # Age kept aligned for per-age-band calibration. The categorized FE emits `age_at_prediction`
        # (= age at anchor); accept legacy `g_age` / `patient_age` too for back-compat with older matrices.
        _age_col = next((c for c in ("age_at_prediction", "g_age", "patient_age") if c in df.columns), None)
        self.age = pd.to_numeric(df[_age_col], errors='coerce').values if _age_col else None
        if _age_col is None:
            print("[calib] no age_at_prediction/g_age/patient_age column in the matrix -> per-age-band "
                  "calibration unavailable; global isotonic only.")
        self._pre_df = df.reset_index(drop=True)
        print(f"Rows: {len(self._pre_df)}, raw feature columns: {self._pre_df.shape[1]} "
              f"({len(self.cat_cols)} categorical). Transforms fit on train next.")
        return self

    def split_data(self, test_size=0.2, calib_size=0.0):
        """Split into train / (optional calibration) / test by index, then FIT every data-dependent
        transform on the TRAIN split ONLY and apply to all splits (no leakage). The fitted
        transformers (encoders, impute medians, kept-column order, scaler) are stored on self and
        saved with the model so held-out scoring uses the SAME train-derived transforms."""
        print("\n" + "=" * 60)
        print("SPLITTING DATA (transforms then fit on TRAIN only)")
        print("=" * 60)

        df = self._pre_df
        idx = np.arange(len(df))
        # Honor the PREDEFINED canonical split — the 'split' column stamped onto the stable matrix by
        # stability_select (the single source of train/valid/test; see splits.py / make_split.py). No
        # re-derivation here, so the internal test stays unseen for selection. test_size/calib_size are
        # accepted for back-compat but ignored (the split fractions live in make_split).
        if self._split is None:
            raise ValueError(
                "split_data: no 'split' column on the matrix. Run stability_select (which stamps the "
                "canonical split) — or make_split.py to (re)create it. See splits.py.")
        sp = np.asarray(self._split)
        tr_idx = idx[sp == "train"]; ca_idx = idx[sp == "valid"]; te_idx = idx[sp == "test"]
        print(f"[split] split_data using the stamped CANONICAL split: "
              f"{len(tr_idx)} train / {len(ca_idx)} valid / {len(te_idx)} test")

        self.y_train, self.y_test, self.y_calib = self.y[tr_idx], self.y[te_idx], self.y[ca_idx]
        tr, te, ca = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy(), df.iloc[ca_idx].copy()

        # 1) categorical encoding — fit code map on TRAIN; unseen categories -> -1
        self.encoders = {}
        for col in self.cat_cols:
            mp = {c: i for i, c in enumerate(sorted(tr[col].fillna('Unknown').astype(str).unique()))}
            self.encoders[col] = mp
            for part in (tr, te, ca):
                part[col] = part[col].fillna('Unknown').astype(str).map(mp).fillna(-1)
        tr = tr.apply(pd.to_numeric, errors='coerce')       # all numeric now
        te = te.apply(pd.to_numeric, errors='coerce')
        ca = ca.apply(pd.to_numeric, errors='coerce')

        # 2) drop high-missing columns (TRAIN missing fraction)
        keep = tr.columns[tr.isnull().mean() <= self._drop_threshold].tolist()
        tr, te, ca = tr[keep], te[keep], ca[keep]
        # Value-type columns = those still carrying NaN in train (counts were 0-filled upstream by
        # build_matrix). Held-out then fills value cols with the TRAIN median and count cols with 0.
        nan_cols = set(tr.columns[tr.isnull().any()])
        # 3) median imputation — TRAIN medians (train-all-NaN cols -> 0)
        self._impute_medians = tr.median(numeric_only=True)
        tr = tr.fillna(self._impute_medians).fillna(0.0)
        te = te.fillna(self._impute_medians).fillna(0.0)
        ca = ca.fillna(self._impute_medians).fillna(0.0)
        # 4) low-variance drop (TRAIN)
        keep = tr.columns[tr.var() >= self._variance_threshold].tolist()
        tr, te, ca = tr[keep], te[keep], ca[keep]
        # 5) high-correlation drop (TRAIN), up to half to avoid over-pruning. Use numpy corrcoef on the
        #    values (no extra F×F ones-matrix, no DataFrame copy) and guard very wide matrices so the
        #    F×F correlation can never blow up memory. Behaviour-identical to the prior pandas version.
        if 1 < tr.shape[1] <= CORR_MAX_FEATURES:
            cm = np.abs(np.corrcoef(tr.values, rowvar=False))            # one F×F float64 matrix
            hi_mask = (np.triu(cm, k=1) > 0.95).any(axis=0)              # col correlated >0.95 with an earlier col
            hi = [tr.columns[j] for j in np.flatnonzero(hi_mask)]
            hi = set(hi[:len(hi) // 2])
            keep = [c for c in tr.columns if c not in hi]
            tr, te, ca = tr[keep], te[keep], ca[keep]
        elif tr.shape[1] > CORR_MAX_FEATURES:
            keep = list(tr.columns)
            print(f"  [corr] {tr.shape[1]} features > {CORR_MAX_FEATURES}: skipping high-corr drop (memory-safe)")
        else:
            keep = list(tr.columns)
        self.feature_names = keep
        self._value_cols = [c for c in keep if c in nan_cols]   # NaN->median at held-out; rest->0
        print(f"Dropped to {len(keep)} features (high-missing/low-var/high-corr fit on train); "
              f"{len(self._value_cols)} value-type")

        # 6) scale — fit on TRAIN
        self.X_train = self.scaler.fit_transform(tr.values)
        self.X_test = self.scaler.transform(te.values) if len(te) else np.empty((0, len(keep)))
        self.X_calib = self.scaler.transform(ca.values) if len(ca) else np.empty((0, len(keep)))

        if self.age is not None:
            self.age_train, self.age_test, self.age_calib = self.age[tr_idx], self.age[te_idx], self.age[ca_idx]

        print(f"Train {self.X_train.shape}, test {self.X_test.shape}, calib {self.X_calib.shape}")
        print(f"Train class dist {np.bincount(self.y_train)}; test {np.bincount(self.y_test)}")
        return self

    def transform_external(self, df_raw):
        """Apply the TRAIN-fitted pipeline (encoders -> kept cols -> fill -> scale) to an external/
        held-out frame, identically to training (no refit). VALUE cols fill with the TRAIN median;
        COUNT cols fill with 0 (absent code/category = 0, not a 'typical' nonzero). Returns a scaled array
        aligned to self.feature_names."""
        d = df_raw.copy()
        for col, mp in self.encoders.items():
            d[col] = (d[col].fillna('Unknown').astype(str).map(mp).fillna(-1) if col in d.columns else -1)
        d = d.reindex(columns=self.feature_names)            # same columns/order as train; missing -> NaN
        d = d.apply(pd.to_numeric, errors='coerce')
        val = [c for c in self.feature_names if c in set(self._value_cols)]
        d[val] = d[val].fillna(self._impute_medians)         # value -> TRAIN median
        return self.scaler.transform(d.fillna(0.0).values)   # remaining (count) -> 0
    
    def handle_imbalance(self, method='none'):
        """Handle class imbalance. Default 'none' = no resampling: the pipeline relies on
        per-learner cost weighting (class_weight / scale_pos_weight / sample_weight) instead
        of SMOTE. Other options: 'smote' / 'adasyn' / 'undersample' / 'smote_tomek'."""
        print("\n" + "=" * 60)
        print(f"HANDLING CLASS IMBALANCE ({method.upper()})")
        print("=" * 60)

        # synthetic/under-sampling (SMOTE/ADASYN/undersample/smote_tomek) is RESEARCH-ONLY. Methodology
        # §4.1: the cohort is already 1:1 and balanced via cost-weighting; SMOTE-type resampling miscalibrates
        # the headline probabilities and must NOT be used for a deployable model. Default 'none' is the only
        # path that reaches deployment; any other method hard-fails unless explicitly opted into for research.
        if method != 'none' and os.environ.get("ALLOW_RESAMPLING", "0") != "1":
            raise SystemExit(
                f"handle_imbalance(method={method!r}) is RESEARCH-ONLY (methodology §4.1 — synthetic "
                f"resampling harms calibration; the cohort is already 1:1 with cost-weighting). "
                f"Set ALLOW_RESAMPLING=1 to override for a research run — do NOT deploy the resulting model.")

        if method == 'none':
            print("No resampling applied")
            return self

        print(f"Before resampling: {np.bincount(self.y_train)}")
        # LAZY import — only needed for an explicit resampling method (keeps imbalanced-learn optional).
        try:
            from imblearn.over_sampling import SMOTE, ADASYN
            from imblearn.under_sampling import RandomUnderSampler
            from imblearn.combine import SMOTETomek
        except ImportError as e:
            raise ImportError(f"handle_imbalance(method={method!r}) needs imbalanced-learn — "
                              f"`pip install imbalanced-learn`, or use method='none' (cost-weighting).") from e

        if method == 'smote':
            sampler = SMOTE(random_state=RANDOM_STATE)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=RANDOM_STATE)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=RANDOM_STATE)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=RANDOM_STATE)
        else:
            print(f"Unknown method {method!r} — no resampling applied")
            return self
        
        self.X_train_resampled, self.y_train_resampled = sampler.fit_resample(
            self.X_train, self.y_train
        )
        
        print(f"After resampling: {np.bincount(self.y_train_resampled)}")
        
        return self
    
    def select_features(self, n_features=100, method='mutual_info', threshold=0.99):
        """Select features.

        Methods:
          'all'         — keep every feature (no further selection).
          'cumimp'      — keep the features covering ``threshold`` (default 0.99) of
                          cumulative RandomForest importance ("cumimp99").
          'mutual_info' / 'f_classif' / 'rfe' — top-``n_features`` (legacy baseline).
        """
        print("\n" + "=" * 60)
        print(f"FEATURE SELECTION ({method.upper()})")
        print("=" * 60)

        def _has_calib():
            return getattr(self, 'X_calib', None) is not None and len(self.X_calib) > 0

        # ── all: keep everything ──────────────────────────────────────────────
        if method == 'all':
            self.feature_selector = None
            self.selected_features = list(self.feature_names)
            print(f"Using ALL {self.X_train.shape[1]} features (no selection).")
            return self

        # ── cumimp99: cumulative RandomForest importance ─────────────────────
        if method == 'cumimp':
            rf = RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
                class_weight='balanced'
            )
            rf.fit(self.X_train, self.y_train)
            imp = rf.feature_importances_
            order = np.argsort(imp)[::-1]
            cum = np.cumsum(imp[order])
            k = int(np.searchsorted(cum, threshold)) + 1
            k = max(1, min(k, len(order)))
            mask = np.zeros(len(imp), dtype=bool)
            mask[order[:k]] = True
            self.feature_selector = _MaskSelector(mask)
            self.selected_features = [self.feature_names[i] for i in range(len(mask)) if mask[i]]
            self.X_train = self.X_train[:, mask]
            self.X_test = self.X_test[:, mask]
            if _has_calib():
                self.X_calib = self.X_calib[:, mask]
            if hasattr(self, 'X_train_resampled'):
                self.X_train_resampled = self.X_train_resampled[:, mask]
            print(f"cumimp{int(threshold*100)}: kept {k}/{len(imp)} features "
                  f"covering {threshold:.0%} of importance.")
            return self

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
        if _has_calib():
            self.X_calib = selector.transform(self.X_calib)

        if hasattr(self, 'X_train_resampled'):
            self.X_train_resampled = selector.transform(self.X_train_resampled)
        
        print(f"Selected {self.X_train.shape[1]} features")
        print(f"Top 100 selected features: {self.selected_features[:100] if hasattr(self, 'selected_features') else 'N/A'}")
        
        return self
    
    def get_models(self):
        """Get dictionary of models to train."""
        models = {
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
        }
        # KNN, MLP, Naive Bayes, Logistic Regression removed: KNN/MLP can't be cost-weighted
        # (no class_weight/sample_weight); Naive Bayes far-weakest (~0.75 AUROC, broken independence
        # assumption); Logistic Regression the weakest of the weighted learners (~0.90 vs ~0.94).
        # Panel is now an all-tree/boosting ensemble, every member cost-sensitive for the minority.
        
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

        # Drop excluded candidates (config.EXCLUDE_MODELS, via env; default 'AdaBoost'). AdaBoost is the
        # only non-tree-exact learner -> excluding it keeps every deployable model (single or soft-voting
        # ensemble) tree-exact. This matters because SHAP is now TREE-EXACT ONLY (KernelExplainer removed):
        # a non-tree model would RAISE in explainability._shap_values, so AdaBoost must stay demoted off.
        _excl = {m.strip() for m in getattr(C, "EXCLUDE_MODELS", ["AdaBoost"]) if m.strip()}
        if _excl:
            dropped = [m for m in models if m in _excl]
            models = {k: v for k, v in models.items() if k not in _excl}
            if dropped:
                print(f"[models] excluded (config.EXCLUDE_MODELS): {dropped} -> {len(models)} candidates")
        return models

    def _valid_set(self):
        """Validation split used for model/threshold SELECTION = the calib split, so the TEST split
        stays completely unseen until final_evaluation. A non-empty calib split is a PRECONDITION of
        the leak-safe design — selecting on TEST would be leakage, so we hard-fail rather than fall back."""
        if getattr(self, 'X_calib', None) is not None and len(self.X_calib) > 0:
            return self.X_calib, np.asarray(self.y_calib), getattr(self, 'age_calib', None)
        raise ValueError(
            "_valid_set: no validation (calib) split — model/threshold selection would fall on TEST "
            "(leakage). Ensure CALIB_SIZE>0 and the stamped split has 'valid' rows.")

    def _age_balance_weights(self):
        """AGE-OFFSET de-confounding (config.AGE_OFFSET_FE). Inverse-propensity TRAIN weights that make
        AGE-BAND ⊥ class: within each class the band distribution is reweighted to the marginal P(band),
        so age carries ~no marginal class information and EVERY learner is forced onto the within-age
        clinical signal (instead of leaning on age, which inflates global AUROC but doesn't help WITHIN
        age). Age's legitimate prior is re-added by the per-age-band isotonic calibration + per-age
        operating threshold, so calibrated probabilities stay prevalence/age-correct. The ensemble-
        compatible equivalent of a base_margin age offset. Returns None when OFF / no train age.

        weight(c,b) = P(band=b) / P(band=b | y=c)  -> within class c the band dist becomes P(b) (==across
        classes), folded with a 'balanced' class weight (no-op on the ~1:1 cohort; robust otherwise),
        mean-normalised to 1. Caches on self._sw_train."""
        if getattr(self, "_sw_train", "unset") != "unset":
            return self._sw_train
        self._sw_train = None
        self._age_offset_active = False
        if not getattr(C, "AGE_OFFSET_FE", False):
            return None
        if getattr(self, "age_train", None) is None:
            print("[age-offset] AGE_OFFSET_FE=1 but the matrix has no age column -> CANNOT de-confound; "
                  "no reweight applied.")
            return None
        a = np.asarray(self.age_train, dtype=float)
        y = np.asarray(self.y_train).astype(int)
        edges = [0, 55, 65, 70, 75, 80, 200]
        band = np.digitize(a, edges[1:-1])
        N = len(y); w = np.ones(N, dtype=float)
        for b in np.unique(band):
            mb = band == b; p_b = mb.sum() / N
            for c in (0, 1):
                mc = mb & (y == c); ncb = int(mc.sum()); nc = int((y == c).sum())
                if ncb and nc:
                    w[mc] = p_b / (ncb / nc)            # P(band) / P(band|class)
        from sklearn.utils.class_weight import compute_sample_weight
        w = w * compute_sample_weight("balanced", y)     # robust if not exactly 1:1 (no-op when it is)
        w = w * (N / w.sum())                            # mean weight -> 1 (keep regularisation scale sane)
        # report the de-confounding effect: how strongly age predicted class before vs after reweighting
        from sklearn.metrics import roc_auc_score as _au
        au_before = _au(y, a) if len(np.unique(y)) > 1 else float("nan")
        print(f"[age-offset] AGE_OFFSET_FE=1: age-band⊥class IPW weights over {len(np.unique(band))} bands "
              f"(weight range [{w.min():.2f}, {w.max():.2f}], mean {w.mean():.2f}). "
              f"age-alone AUROC on train was {au_before:.3f} -> reweighted so age is ~uninformative; "
              f"per-age isotonic re-adds the prior.")
        self._sw_train = w; self._age_offset_active = True
        return w

    def train_and_evaluate(self, use_resampled=False):
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

        # AGE-OFFSET de-confound weights (config.AGE_OFFSET_FE). Only valid when the row order matches
        # y_train/age_train (i.e. NOT the resampled path) — resampling re-orders/synthesises rows so the
        # weight vector can't align; skip with a warning there (resampling is research-only anyway).
        sw = self._age_balance_weights()
        if sw is not None and (use_resampled and hasattr(self, 'X_train_resampled')):
            print("[age-offset] resampled training data in use -> IPW weights don't align; NOT applied "
                  "(use handle_imbalance('none'), the deployment default).")
            sw = None
        if sw is not None and len(sw) != len(y_train_use):
            print(f"[age-offset] weight length {len(sw)} != train rows {len(y_train_use)} -> NOT applied.")
            sw = None
        self._sw_applied = sw
        
        results = []
        # Model SELECTION is scored on the VALIDATION (calib) split — the TEST split is never seen
        # here; it's only touched in final_evaluation, at the validation-chosen threshold.
        Xv, yv, _ = self._valid_set()
        self.y_select = yv                 # labels matching the stored per-model preds (used by plots)
        print(f"Selecting models on the VALIDATION split (n={len(yv)}, pos={int(np.sum(yv))}); test unseen.")

        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = datetime.now()
            
            try:
                # Train model. RF/ExtraTrees/LightGBM use class_weight='balanced',
                # XGBoost scale_pos_weight, CatBoost auto_class_weights. GradientBoosting
                # and AdaBoost have no class-weight param but DO accept sample_weight in
                # .fit() -> pass a 'balanced' weight so the minority (cancer) class is
                # penalized too.
                if sw is not None:
                    # AGE-OFFSET on: sw already folds in class balancing, so pass it to EVERY learner
                    # (every tree learner's .fit accepts sample_weight; it composes with their class_weight,
                    # which is ~no-op on the 1:1 cohort) — this is what forces them onto within-age signal.
                    model.fit(X_train_use, y_train_use, sample_weight=sw)
                elif name in ('Gradient Boosting', 'AdaBoost'):
                    from sklearn.utils.class_weight import compute_sample_weight
                    _sw = compute_sample_weight('balanced', y_train_use)
                    model.fit(X_train_use, y_train_use, sample_weight=_sw)
                else:
                    model.fit(X_train_use, y_train_use)
                
                # Predict on the VALIDATION split (selection must not see the test split)
                y_pred = model.predict(Xv)
                y_pred_proba = model.predict_proba(Xv)[:, 1] if hasattr(model, 'predict_proba') else None

                # Calculate metrics (on validation)
                accuracy = accuracy_score(yv, y_pred)
                precision = precision_score(yv, y_pred, zero_division=0)
                recall = recall_score(yv, y_pred, zero_division=0)
                f1 = f1_score(yv, y_pred, zero_division=0)

                if y_pred_proba is not None:
                    roc_auc = roc_auc_score(yv, y_pred_proba)
                    avg_precision = average_precision_score(yv, y_pred_proba)
                else:
                    roc_auc = None
                    avg_precision = None

                # Confusion matrix (validation)
                cm = confusion_matrix(yv, y_pred, labels=[0, 1])
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
        self.results_df = self.results_df.sort_values(SELECTION_METRIC, ascending=False)   # rank by selection metric

        print("\n" + "=" * 60)
        print("MODEL COMPARISON RESULTS")
        print("=" * 60)
        print(self.results_df.to_string(index=False))

        # Guard: if EVERY candidate failed to train, results_df is empty and idxmax() would raise a
        # cryptic error — fail with a legible message instead (catastrophic-env signal).
        if self.results_df.empty:
            raise RuntimeError("no model trained successfully — every candidate raised during fit "
                               "(see the per-model errors above). Cannot select a best model.")
        # Find best model (by the selection metric — AUPRC by default, decision-relevant at low prevalence)
        best_idx = self.results_df[SELECTION_METRIC].idxmax()
        self.best_model_name = self.results_df.loc[best_idx, 'Model']
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\nBest Model: {self.best_model_name} ({SELECTION_METRIC}: {self.results_df.loc[best_idx, SELECTION_METRIC]:.4f}, "
              f"ROC AUC: {self.results_df.loc[best_idx, 'ROC AUC']:.4f})")
        
        return self
    
    def hyperparameter_tuning(self, model_name=None, n_trials=N_TUNING_TRIALS, cv_folds=TUNING_CV_FOLDS,
                              top_n=TUNE_TOP_N):
        """Optuna (TPE) tuning with k-fold CV on the TRAIN split only (internal test + held-out
        untouched). Tunes the TOP-`top_n` models by internal ROC AUC and REPLACES each in self.results
        with its tuned version, so create_ensemble (which uses the top models) ensembles the TUNED
        learners — otherwise the tuning is discarded when the untuned ensemble wins. Only the tunable
        learners (RF/XGB/LGBM/GB) have search spaces; others are skipped."""
        if not getattr(C, "TUNE", False):          # OFF by default; set TUNE=1 (config reads env) to enable (slow, top-5)
            print("[tuning] TUNE not set -> skipping Optuna (ensemble uses untuned top-5).")
            return self
        print("\n" + "=" * 60)
        print(f"HYPERPARAMETER TUNING (Optuna: {n_trials} trials x {cv_folds}-fold, top-{top_n} models)")
        print("=" * 60)

        X_train_use = getattr(self, 'X_train_resampled', self.X_train)
        y_train_use = getattr(self, 'y_train_resampled', self.y_train)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        tunable = {'Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting'}

        def _suggest(trial, mname):
            if mname == 'Random Forest':
                return dict(n_estimators=trial.suggest_int('n_estimators', 100, 400, step=50),
                            max_depth=trial.suggest_categorical('max_depth', [10, 15, 20, None]),
                            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 6))
            if mname == 'XGBoost':
                return dict(n_estimators=trial.suggest_int('n_estimators', 100, 400, step=50),
                            max_depth=trial.suggest_int('max_depth', 3, 8),
                            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                            subsample=trial.suggest_float('subsample', 0.7, 1.0),
                            colsample_bytree=trial.suggest_float('colsample_bytree', 0.7, 1.0))
            if mname == 'LightGBM':
                return dict(n_estimators=trial.suggest_int('n_estimators', 100, 400, step=50),
                            max_depth=trial.suggest_int('max_depth', 3, 8),
                            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                            num_leaves=trial.suggest_int('num_leaves', 20, 150))
            if mname == 'Gradient Boosting':
                return dict(n_estimators=trial.suggest_int('n_estimators', 100, 250, step=50),
                            max_depth=trial.suggest_int('max_depth', 3, 7),
                            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                            min_samples_split=trial.suggest_int('min_samples_split', 2, 10))
            return None

        ranked = self.results_df.sort_values(SELECTION_METRIC, ascending=False)['Model'].tolist()
        to_tune = [m for m in ranked if m in tunable][:top_n]
        print(f"Top-{top_n} tunable models to tune: {to_tune}")
        best_overall = self.results_df[SELECTION_METRIC].max()

        for mname in to_tune:
            base = self.get_models()[mname]

            def objective(trial, _m=mname, _b=base):
                m = clone(_b).set_params(**_suggest(trial, _m))
                if 'n_jobs' in m.get_params():
                    m.set_params(n_jobs=1)            # let cross_val_score own the parallelism
                return cross_val_score(m, X_train_use, y_train_use, cv=cv,
                                       scoring='average_precision', n_jobs=-1).mean()

            study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            est = clone(base).set_params(**study.best_params)
            if 'n_jobs' in est.get_params():
                est.set_params(n_jobs=-1)
            # AGE-OFFSET: refit the tuned learner with the same de-confound weights (the inner Optuna CV
            # tuned unweighted — acceptable since TUNE is off by default; the DEPLOYED fit is weighted).
            _sw = getattr(self, "_sw_applied", None)
            if _sw is not None and len(_sw) == len(y_train_use):
                est.fit(X_train_use, y_train_use, sample_weight=_sw)
            else:
                est.fit(X_train_use, y_train_use)
            Xv, yv, _ = self._valid_set()
            _pv = est.predict_proba(Xv)[:, 1]
            auc = roc_auc_score(yv, _pv)                            # validation ROC AUC (test stays unseen)
            ap = average_precision_score(yv, _pv)                  # keep the AUPRC column current post-tuning
            prev = float(self.results_df.loc[self.results_df['Model'] == mname, SELECTION_METRIC].iloc[0])
            print(f"  {mname}: CV AUPRC {study.best_value:.4f} -> validation AUPRC {ap:.4f} / ROC AUC {auc:.4f} "
                  f"({SELECTION_METRIC} was {prev:.4f})")
            # REPLACE the trained model so create_ensemble uses the TUNED version
            if mname in self.results:
                self.results[mname]['model'] = est
            self.results_df.loc[self.results_df['Model'] == mname, 'ROC AUC'] = auc
            self.results_df.loc[self.results_df['Model'] == mname, 'Avg Precision'] = ap
            _sel = ap if SELECTION_METRIC == 'Avg Precision' else auc   # the leader is tracked on the selection metric
            if _sel > best_overall:
                best_overall = _sel
                self.best_model, self.best_model_name = est, f"{mname} (Tuned)"

        # re-rank so create_ensemble's top-N picks the tuned leaders (by the selection metric)
        self.results_df = self.results_df.sort_values(SELECTION_METRIC, ascending=False).reset_index(drop=True)
        self.tuned_top_n = top_n
        return self
    
    def create_ensemble(self):
        """Create an ensemble of top performing models."""
        print("\n" + "=" * 60)
        print("CREATING ENSEMBLE MODEL")
        print("=" * 60)
        
        # Get the top models by the SELECTION metric (AUPRC by default — results_df is kept sorted by it)
        top_models = self.results_df.head(TUNE_TOP_N)['Model'].tolist()   # tuned top-N (matches tuning scope)
        print(f"Top models for ensemble: {top_models}")
        
        estimators = [(name, self.results[name]['model']) for name in top_models]
        
        # Voting classifier (soft voting for probability averaging)
        # n_jobs=1: fit the 3 sub-models serially (each keeps its own threads) — avoids the
        # VotingClassifier x sub-estimator nested-parallelism oversubscription.
        ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)
        
        # Use resampled data if available
        if hasattr(self, 'X_train_resampled'):
            X_train_use = self.X_train_resampled
            y_train_use = self.y_train_resampled
        else:
            X_train_use = self.X_train
            y_train_use = self.y_train

        # Train ensemble — carry the AGE-OFFSET weights (VotingClassifier.fit forwards sample_weight to
        # every sub-estimator that supports it). _sw_applied was vetted in train_and_evaluate (None unless
        # AGE_OFFSET_FE on + non-resampled + aligned).
        _sw = getattr(self, "_sw_applied", None)
        if _sw is not None and len(_sw) == len(y_train_use):
            ensemble.fit(X_train_use, y_train_use, sample_weight=_sw)
        else:
            ensemble.fit(X_train_use, y_train_use)
        
        # Evaluate on the VALIDATION split (selection only; test stays unseen)
        Xv, yv, _ = self._valid_set()
        y_pred = ensemble.predict(Xv)
        y_pred_proba = ensemble.predict_proba(Xv)[:, 1]

        accuracy = accuracy_score(yv, y_pred)
        precision = precision_score(yv, y_pred, zero_division=0)
        recall = recall_score(yv, y_pred, zero_division=0)
        f1 = f1_score(yv, y_pred, zero_division=0)
        roc_auc = roc_auc_score(yv, y_pred_proba)
        avg_precision = average_precision_score(yv, y_pred_proba)   # compare on the selection metric

        print(f"\nEnsemble Model Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  Avg Precision (AUPRC): {avg_precision:.4f}")

        # Update best model if improved — on the SELECTION metric, consistent with single-model selection
        _ens_sel = avg_precision if SELECTION_METRIC == 'Avg Precision' else roc_auc
        if _ens_sel > self.results_df[SELECTION_METRIC].max():
            self.best_model = ensemble
            self.best_model_name = "Ensemble"
            print(f"\nEnsemble is now the best model!")
        
        self.ensemble_model = ensemble

        return self

    def calibrate(self, method='isotonic', by_age_band=True,
                  band_edges=(0, 55, 65, 70, 75, 80, 200),
                  min_band=50, min_band_pos=5):
        """Isotonic calibration of the best model on the held-out calibration set.

        Fits a global isotonic map and, when ``by_age_band`` and ``patient_age`` are
        available, per-age-band maps (thin bands fall back to the global map). Reports
        Brier score + AUC before/after on the test set. Requires a calibration split
        (``split_data(..., calib_size>0)``); otherwise it is skipped cleanly.
        """
        print("\n" + "=" * 60)
        print("CALIBRATION (ISOTONIC" + (", PER-AGE-BAND" if by_age_band else "") + ")")
        print("=" * 60)

        if getattr(self, 'X_calib', None) is None or len(self.X_calib) == 0:
            print("No calibration split (calib_size=0) — skipping calibration.")
            return self
        if method != 'isotonic':
            print(f"Only 'isotonic' supported here; got {method!r} — skipping.")
            return self

        base = self.best_model
        raw_cal = base.predict_proba(self.X_calib)[:, 1]
        yb = np.asarray(self.y_calib)

        global_iso = IsotonicRegression(out_of_bounds='clip').fit(raw_cal, yb)

        band_isos = {}
        if by_age_band and getattr(self, 'age_calib', None) is not None:
            edges = list(band_edges)
            bidx = np.digitize(np.asarray(self.age_calib, dtype=float), edges[1:-1])
            for b in np.unique(bidx):
                m = bidx == b
                pos = int(yb[m].sum()); neg = int((yb[m] == 0).sum())
                if m.sum() >= min_band and pos >= min_band_pos and neg >= min_band_pos:
                    band_isos[int(b)] = IsotonicRegression(out_of_bounds='clip').fit(
                        raw_cal[m], yb[m])
            print(f"Per-age-band isotonic maps fitted for bands {sorted(band_isos.keys())} "
                  f"(edges {edges}); global fallback elsewhere.")
        else:
            edges = list(band_edges)
            print("Global isotonic only (no usable patient_age).")

        self.calibrated_model = CalibratedLungModel(base, global_iso, edges, band_isos)

        # Report calibration effect on the CALIB split (in-sample — the isotonic map was fit here, so
        # indicative only). The TEST split is deliberately left untouched until final_evaluation.
        cal_cv = self.calibrated_model.predict_proba(self.X_calib, getattr(self, 'age_calib', None))[:, 1]
        print(f"\n  [calib, in-sample] Brier  raw={brier_score_loss(yb, raw_cal):.4f} "
              f"-> calibrated={brier_score_loss(yb, cal_cv):.4f}  (lower=better)")
        print(f"  [calib, in-sample] ROC AUC raw={roc_auc_score(yb, raw_cal):.4f} "
              f"-> calibrated={roc_auc_score(yb, cal_cv):.4f}")
        return self

    def final_evaluation(self):
        """Perform final evaluation and generate detailed report."""
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        
        print(f"\nFinal Best Model: {self.best_model_name}")

        # Make predictions — use the calibrated model when available so reported
        # probabilities / thresholds are the deployed ones.
        if getattr(self, 'calibrated_model', None) is not None:
            print("Using ISOTONIC-CALIBRATED probabilities.")
            y_pred_proba = self.calibrated_model.predict_proba(self.X_test, self.age_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            y_pred = self.best_model.predict(self.X_test)
            y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Cancer', 'Cancer']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
        print("\nConfusion Matrix:")
        print(cm)
        
        # Detailed metrics
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nDetailed Metrics:")
        print(f"  True Positives (TP): {tp} - Cancer patients correctly identified")
        print(f"  True Negatives (TN): {tn} - Non-cancer patients correctly identified")
        print(f"  False Positives (FP): {fp} - Non-cancer patients incorrectly flagged")
        print(f"  False Negatives (FN): {fn} - Cancer patients missed")
        
        print(f"\n  Sensitivity (Recall): {100*tp/(tp+fn):.1f}% - Ability to detect cancer")
        print(f"  Specificity: {100*tn/(tn+fp):.1f}% - Ability to identify non-cancer")
        print(f"  Positive Predictive Value: {100*tp/(tp+fp):.1f}% - Probability of cancer given positive test")
        print(f"  Negative Predictive Value: {100*tn/(tn+fn):.1f}% - Probability of no cancer given negative test")
        
        # Find the optimal (Youden-J) threshold on the VALIDATION split, then APPLY it to the test
        # scores below — the test split never participates in threshold selection.
        Xv, yv, agev = self._valid_set()
        if getattr(self, 'calibrated_model', None) is not None:
            valid_proba = self.calibrated_model.predict_proba(Xv, agev)[:, 1]
        else:
            valid_proba = self.best_model.predict_proba(Xv)[:, 1]
        fpr, tpr, thresholds = roc_curve(yv, valid_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        print(f"\n  Optimal Threshold (chosen on validation, applied to test): {optimal_threshold:.4f}")

        # Predictions with the validation-chosen threshold, on the TEST scores
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        cm_optimal = confusion_matrix(self.y_test, y_pred_optimal, labels=[0, 1])
        tn_opt, fp_opt, fn_opt, tp_opt = cm_optimal.ravel()
        
        print(f"\nMetrics at Optimal Threshold ({optimal_threshold:.4f}):")
        print(f"  Sensitivity (Recall): {100*tp_opt/(tp_opt+fn_opt):.1f}%")
        print(f"  Specificity: {100*tn_opt/(tn_opt+fp_opt):.1f}%")
        print(f"  F1 Score: {100*f1_score(self.y_test, y_pred_optimal):.1f}%")

        # AGGREGATE operating-point comparison (mirror of held-out): overall Sens/Spec/FP at the single global
        # (validation-Youden) cut vs at per-age-band cuts (each patient judged at their band's threshold; bands
        # with <20 validation positives fall back to global). Internal test is small so bands often fall back.
        try:
            yt = np.asarray(self.y_test); _edges = (getattr(getattr(self, 'calibrated_model', None), 'band_edges', None)
                                                    or [0, 55, 65, 70, 75, 80, 200])
            _bt = np.digitize(np.asarray(self.age_test, dtype=float), _edges[1:-1])
            _bv = np.digitize(np.asarray(agev, dtype=float), _edges[1:-1])
            _by_band = {}
            for b in sorted(set(_bv.tolist())):
                mb = _bv == b
                if int(np.asarray(yv)[mb].sum()) >= 20:
                    _by_band[int(b)] = float(lung_metrics.threshold_at_sensitivity(np.asarray(yv)[mb], valid_proba[mb], 0.90))
            _thr = np.array([_by_band.get(int(b), optimal_threshold) for b in _bt])
            def _ovr(pred):
                tp = int(((pred == 1) & (yt == 1)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
                tn = int(((pred == 0) & (yt == 0)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
                return (tp/(tp+fn) if tp+fn else float('nan'), tn/(tn+fp) if tn+fp else float('nan'), fp)
            _seg, _spg, _fpg = _ovr((y_pred_proba >= optimal_threshold).astype(int))
            _seb, _spb, _fpb = _ovr((y_pred_proba >= _thr).astype(int))
            print(f"\nOVERALL operating-point comparison (internal TEST, n={len(yt)}):")
            print(f"    @ global threshold        : Sens {_seg*100:.1f}  Spec {_spg*100:.1f}  FP {_fpg}")
            print(f"    @ per-age-band thresholds : Sens {_seb*100:.1f}  Spec {_spb*100:.1f}  FP {_fpb}   "
                  f"(dFP {_fpb-_fpg:+d}, dSpec {(_spb-_spg)*100:+.1f}pp)")
        except Exception as _e:
            print(f"  [internal per-age-band comparison skipped: {_e}]")

        # Full clinical suite with 95% bootstrap CIs on the internal TEST split, at the
        # validation-chosen threshold (stored for save_model -> internal_{h}.csv).
        self.internal_metrics_ci = lung_metrics.eval_with_ci(
            self.y_test, y_pred_proba, threshold=optimal_threshold, label="internal test")
        print("\nInternal test — metrics with 95% bootstrap CIs:")
        print(lung_metrics.format_result(self.internal_metrics_ci))

        # WITHIN-AGE-BAND discrimination — the honest "does the model help BEYOND age?" number (global
        # AUROC is inflated by easy young-vs-old separation). This is the headline metric for the
        # AGE_OFFSET work: it should hold/improve when age is de-confounded, even if global AUROC dips.
        _wamacro_au = _wamacro_ap = float("nan")
        if getattr(self, "age_test", None) is not None:
            try:
                _warows, _wamacro_au, _wamacro_ap, _waglob = lung_metrics.within_age_metrics(
                    self.y_test, y_pred_proba, self.age_test)
                print("\n" + lung_metrics.format_within_age(_warows, _wamacro_au, _wamacro_ap, _waglob,
                                                            label="internal test"))
            except Exception as _wae:
                print(f"[within-age] internal within-age metrics skipped: {_wae}")
        else:
            print("[within-age] no age_test -> within-age-band metrics unavailable.")

        # Store internal-test metrics so save_model can write internal_{h}.csv.
        from sklearn.metrics import roc_auc_score as _auroc, average_precision_score as _auprc
        self.internal_metrics = {
            "best_model": self.best_model_name,
            "n_test": int(len(self.y_test)), "n_pos": int(np.sum(self.y_test)),
            "auroc": float(_auroc(self.y_test, y_pred_proba)),
            "auprc": float(_auprc(self.y_test, y_pred_proba)),
            # point metrics at BOTH operating points, named explicitly to avoid ambiguity: the 0.5 cut
            # (diagnostic only) vs the validation-chosen Youden threshold (the meaningful operating point).
            "sensitivity_at_0p5": tp / (tp + fn) if (tp + fn) else 0.0,
            "specificity_at_0p5": tn / (tn + fp) if (tn + fp) else 0.0,
            "ppv_at_0p5": tp / (tp + fp) if (tp + fp) else 0.0,
            "npv_at_0p5": tn / (tn + fn) if (tn + fn) else 0.0,
            "optimal_threshold": float(optimal_threshold),
            "sens_at_opt": tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) else 0.0,
            "spec_at_opt": tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) else 0.0,
            # within-age (de-confounded) discrimination + whether the AGE-OFFSET reweighting was active
            "within_age_macro_auroc": float(_wamacro_au),
            "within_age_macro_auprc": float(_wamacro_ap),
            "age_offset_active": bool(getattr(self, "_age_offset_active", False)),
        }
        # Stash the internal-TEST predictions (at the validation-chosen threshold) + calibrated probs so
        # save_model can emit patient-level + TP/FP/TN/FN segment explainability on the internal test
        # (same SHAP machinery as the held-out path, on the model that gets saved = self.best_model).
        self._internal_pred = y_pred_optimal
        self._internal_proba = y_pred_proba
        return self
    
    def plot_results(self, save_path=None, best_model_only=True):
        """
        Generate visualization plots.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        best_model_only : bool, default=True
            If True, ROC curve only shows best model + random baseline.
            If False, shows all models.
        """
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
        if best_model_only:
            best_name_label_text = self.best_model_name   # was hardcoded 'Model B' (mislabelled the ROC legend)
            # Only plot best model and random baseline
            best_name = self.best_model_name.replace(' (Tuned)', '').replace('Ensemble', self.best_model_name)
            if best_name in self.results and self.results[best_name]['y_pred_proba'] is not None:
                data = self.results[best_name]
                fpr, tpr, _ = roc_curve(self.y_select, data['y_pred_proba'])
                roc_auc = roc_auc_score(self.y_select, data['y_pred_proba'])
                ax2.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'{best_name_label_text} (AUC = {roc_auc:.3f})')
            elif hasattr(self, 'best_model') and hasattr(self.best_model, 'predict_proba'):
                y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                ax2.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'{self.best_model_name} (AUC = {roc_auc:.3f})')
        else:
            # Plot all models
            for name, data in self.results.items():
                if data['y_pred_proba'] is not None:
                    fpr, tpr, _ = roc_curve(self.y_select, data['y_pred_proba'])
                    roc_auc = roc_auc_score(self.y_select, data['y_pred_proba'])
                    ax2.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.500)')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve' if best_model_only else 'ROC Curves (All Models)')
        ax2.legend(loc='lower right', fontsize=10 if best_model_only else 8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curves
        ax3 = axes[0, 2]
        for name, data in self.results.items():
            if data['y_pred_proba'] is not None:
                precision, recall, _ = precision_recall_curve(self.y_select, data['y_pred_proba'])
                avg_precision = average_precision_score(self.y_select, data['y_pred_proba'])
                ax3.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curves')
        ax3.legend(loc='upper right', fontsize=8)
        
        # 4. Confusion Matrix - Best Model
        ax4 = axes[1, 0]
        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
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
            'feature_selector': getattr(self, 'feature_selector', None),
            'calibrated_model': getattr(self, 'calibrated_model', None),
            # TRAIN-fitted preprocessing transforms (for leakage-free held-out scoring via
            # transform_external): categorical code maps + median impute values.
            'encoders': getattr(self, 'encoders', {}),
            'impute_medians': getattr(self, '_impute_medians', None),
            'value_cols': getattr(self, '_value_cols', None),   # fill these w/ median at held-out; rest -> 0
            'cat_cols': getattr(self, 'cat_cols', []),
            # FE-config provenance (C1 guard): the feature-engineering flags this model was TRAINED under.
            # Held-out/serve MUST rebuild FE with the SAME flags or the trained features go silently missing
            # (NaN-imputed). evaluate_heldout / predict_unseen read this and align config before build().
            'fe_config': {k: getattr(C, k, None) for k in
                          ("AGE_PROXY_FE", "DENSITY_PROXY_FE", "ADMIN_NOISE_FE", "BLOOD_RATIOS_FE",
                           "CLINICAL_FE", "CHRONICITY_FE", "REDFLAG_FE", "CATEGORY_PRUNE_FE", "MIN_VALUE_FRAC",
                           "CATEGORIZED", "CROSS_TOP_K")},
            # MODELING-only provenance (NOT an FE flag — does not change feature columns, so held-out/serve
            # do NOT realign FE on it; recorded so the run manifest documents whether age de-confounding
            # was active for this model and its within-age discrimination).
            'model_config': {"AGE_OFFSET_FE": getattr(C, "AGE_OFFSET_FE", False),
                             "age_offset_active": bool(getattr(self, "_age_offset_active", False))},
        }
        
        artifacts.atomic_write(filepath, lambda t: joblib.dump(model_data, t))   # .tmp -> os.replace
        print(f"\nModel saved to: {filepath}")

        # Also write internal-test metrics CSV next to the model.
        if getattr(self, "internal_metrics", None):
            import csv as _csv, io as _io
            tag = os.path.splitext(os.path.basename(filepath))[0].replace("model_", "")   # e.g. 12mo
            ipath = os.path.join(os.path.dirname(filepath) or ".", f"internal_{tag}.csv")
            _buf = _io.StringIO()
            _w = _csv.writer(_buf)
            _w.writerow(list(self.internal_metrics.keys()))
            _w.writerow(list(self.internal_metrics.values()))
            artifacts.atomic_write_text(ipath, _buf.getvalue())   # atomic: no partial CSV that publish() could mirror
            print(f"Internal metrics saved to: {ipath}")

        # Patient-level + TP/FP/TN/FN segment explainability on the INTERNAL TEST split (same SHAP machinery
        # as the held-out path, on the saved model = self.best_model). Writes explainability_internal/:
        # patient_explanations.csv (per-patient y_true/y_pred/prob + signed-SHAP top factors), segment_drivers.csv
        # (+ per-segment beeswarms/bars), shap_summary.png. Gated by EXPLAIN env (=0 to skip); non-fatal —
        # a SHAP hiccup must not lose the saved model/metrics, but it prints the full traceback (loud).
        if os.environ.get("EXPLAIN", "1").strip().lower() not in ("0", "false", "no") \
                and getattr(self, "_internal_pred", None) is not None and getattr(self, "feature_names", None):
            try:
                here = os.path.dirname(os.path.abspath(__file__))
                if here not in sys.path:
                    sys.path.insert(0, here)
                import explainability as _xai
                tag = os.path.splitext(os.path.basename(filepath))[0].replace("model_", "")
                xdir = os.path.join(os.path.dirname(filepath) or ".", "explainability_internal")
                _xai.explain(self.best_model, self.X_test, self.feature_names,
                             np.asarray(self.y_test), self._internal_pred, self._internal_proba, xdir,
                             title=f"{tag} internal-test", code_terms=None)   # CATEGORIZED feats are human-readable
                print(f"Internal-test explainability saved to: {xdir}")
            except Exception as _xe:
                import traceback
                print(f"[train] internal-test explainability FAILED (model + metrics above are saved):\n"
                      f"{traceback.format_exc()}")
        return self


def main():
    """Main function to run the lung cancer prediction pipeline."""
    
    # Data path
    data_path = "18-unified_B1_all_patients_trend_lungCancer_noCancer_15yrs_10Obs_0Med_1_1_v2_Min50.csv"
    
    # Initialize predictor
    predictor = LungCancerPredictor(data_path)
    
    # Standalone demo only. The REAL entry point is run_v3.py (build_features -> stability_select -> lung_training),
    # which uses an 80/10/10 split, handle_imbalance("none") (cost-weighting, no SMOTE) and the stable codelist selection. Args below mirror that design so the demo doesn't contradict it.
    (predictor
     .load_data()
     .explore_data()
     .preprocess_data(drop_threshold=0.90)
     .split_data(test_size=0.10, calib_size=0.10)
     .handle_imbalance(method='none')
     .select_features(method='cumimp', threshold=0.99)
     .train_and_evaluate(use_resampled=False)
     .hyperparameter_tuning()
     .create_ensemble()
     .final_evaluation()
     .plot_results(save_path='lung_cancer_model_results.png')
     .save_model('lung_cancer_best_mode_unified_v2.joblib'))
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return predictor


if __name__ == "__main__":
    predictor = main()
