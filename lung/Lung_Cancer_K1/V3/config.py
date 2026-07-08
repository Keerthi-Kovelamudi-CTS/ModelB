"""Central configuration for the V3 lung pipeline.

Loads `.env` (via python-dotenv) once, then exposes typed, validated settings. Real environment
variables always win over `.env`, so any value can be overridden per-run, e.g.:
    TUNE=1 TOP_N=500 python run_v3.py

Usage (scripts add the V3 root to sys.path, then):
    import config as C
    C.TOP_N, C.RANDOM_STATE, C.FE_YEARS_BEFORE, ...

Defaults below match the values previously hard-coded across the V3 scripts, so behaviour is
unchanged if `.env` is absent.
"""
import os
from pathlib import Path

# Load .env next to this file (no-op if python-dotenv isn't installed or the file is missing).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass


def _s(key, default):
    return os.environ.get(key, default)


def _i(key, default):
    return int(os.environ.get(key, default))


def _f(key, default):
    return float(os.environ.get(key, default))


def _b(key, default=False):
    return str(os.environ.get(key, str(default))).strip().lower() in ("1", "true", "yes", "on")


def _opt_i(key, default):
    """int or None (the literal 'none'/'' -> None). For e.g. TREND_MAX_MONTHS."""
    v = str(os.environ.get(key, default)).strip().lower()
    return None if v in ("none", "", "null") else int(v)


# --- GCP / data ---
GCP_PROJECT          = _s("GCP_PROJECT", "prj-cts-ai-dev-sp")
HORIZONS             = [h.strip() for h in _s("HORIZONS", "12mo,1mo").split(",") if h.strip()]
FE_YEARS_BEFORE      = _i("FE_YEARS_BEFORE", 5)        # FE lookback window (years)
SCORING_YEARS_BEFORE = _i("SCORING_YEARS_BEFORE", 5)  # Phase-1 scoring lookback (years)
FE_WINDOWS           = [int(x) for x in _s("FE_WINDOWS", "5,10,20,100").split(",") if x.strip()]  # all lookback windows to run (yrs; 100=lifetime)

# --- Phase 1b: codelist construction ---
TOP_N          = _i("TOP_N", 500)
OR_MIN         = _f("OR_MIN", 2.0)
PREV_MIN       = _f("PREV_MIN", 0.01)
MIN_VALUE_FRAC = _f("MIN_VALUE_FRAC", 0.30)
# Force-include the curated clinical-concept codes (curated_codes.json) in the codelist AND give them
# concept-level FE. OFF -> codelist is the data-driven top-N only and every code gets generic per-code FE.
USE_CURATED    = _b("USE_CURATED", True)

# --- Phase 2b: stability selection ---
N_FOLDS   = _i("N_FOLDS", 5)
MIN_FOLDS = _i("MIN_FOLDS", 3)
CUM_IMP   = _f("CUM_IMP", 0.99)

# --- Phase 2: feature engineering ---
FE_ENGINE        = _s("FE_ENGINE", "pandas")   # FE backend: "pandas" (trustworthy default) | "polars" (accelerated per-code families; use only after fe_parity_check.py passes)
ANCHOR_MODE      = _s("ANCHOR_MODE", "patient_last")
TREND_MAX_MONTHS = _opt_i("TREND_MAX_MONTHS", "none")
DECAY_TAU_MONTHS = _f("DECAY_TAU_MONTHS", 12.0)
LAB_Z_EXTREME    = _f("LAB_Z_EXTREME", 2.0)

# --- Phase 3: modeling ---
RANDOM_STATE    = _i("RANDOM_STATE", 42)
TEST_SIZE       = _f("TEST_SIZE", 0.10)
CALIB_SIZE      = _f("CALIB_SIZE", 0.10)
DROP_THRESHOLD  = _f("DROP_THRESHOLD", 0.90)
TUNE            = _b("TUNE", False)
TUNE_TOP_N      = _i("TUNE_TOP_N", 5)
N_TUNING_TRIALS = _i("N_TUNING_TRIALS", 100)
TUNING_CV_FOLDS = _i("TUNING_CV_FOLDS", 5)

# --- Phase 4: held-out evaluation ---
HELDOUT_CHUNK = _i("HELDOUT_CHUNK", 0)

# --- canonical split (split-first; created once by make_split.py, loaded by every step) ---
SPLIT_DIR       = _s("SPLIT_DIR", "splits")     # local dir or gs:// bucket for the train/valid/test split
USE_SAVED_SPLIT = _b("USE_SAVED_SPLIT", True)   # True = load the saved canonical split everywhere
GCS_ROOT        = _s("GCS_ROOT", "")          # common GCS root for shared feature/model artifacts


def summary():
    """One-line-per-setting dump (handy at the top of a run for the log)."""
    keys = [k for k in globals() if k.isupper()]
    return "\n".join(f"  {k} = {globals()[k]!r}" for k in keys)


if __name__ == "__main__":
    print("V3 config:\n" + summary())
