"""Shared calibrator wrapper — must be importable by both training and inference
so joblib can deserialize CalibratorWrapper instances regardless of entry point.
"""
import numpy as np


class CalibratorWrapper:
    """Unifies IsotonicRegression and Platt (LogisticRegression) under .transform()."""

    def __init__(self, calibrator, kind):
        self.calibrator = calibrator
        self.kind = kind  # 'isotonic' or 'platt'

    def transform(self, x):
        x = np.asarray(x)
        if self.kind == 'isotonic':
            return self.calibrator.transform(x)
        return self.calibrator.predict_proba(x.reshape(-1, 1))[:, 1]
