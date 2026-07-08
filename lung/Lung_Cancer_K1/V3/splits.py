"""Canonical, saved train/valid/test split — the single source of truth for the whole V3 pipeline.

The split is keyed by `patient_guid` (NOT by row order), created ONCE per horizon, and saved (local
or GCS). Every step (code scoring, codelist, FE, stability-selection, model training) loads THIS file
and filters to its assigned patients — so:
  * train/valid/test are the SAME patients in every step and every experiment (reproducible),
  * the test split is never seen by any fitting step (no leakage by construction),
  * results from different lookback windows / arms are directly comparable.

Determinism: patients are sorted by CLEANED `patient_guid` (canonical order, independent of how the
cohort SQL or feature matrix happens to be ordered), then a seed-stratified split is applied:
    test = TEST_SIZE (default 0.10), valid = CALIB_SIZE (default 0.10), train = remainder (0.80).

Usage:
    import splits
    df = splits.load_or_make(horizon="12mo", guids=guids, labels=y)   # patient_guid, split
    train = splits.guids_for(df, "train")        # set of patient_guids
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import config as C
    SEED        = C.RANDOM_STATE
    TEST_SIZE   = C.TEST_SIZE
    VALID_SIZE  = C.CALIB_SIZE
    SPLIT_DIR   = getattr(C, "SPLIT_DIR", "splits")
    FE_YEARS    = getattr(C, "FE_YEARS_BEFORE", 5)
except Exception:                       # standalone / config unavailable
    SEED, TEST_SIZE, VALID_SIZE, SPLIT_DIR, FE_YEARS = 42, 0.10, 0.10, "splits", 5


def clean_guid(s):
    """Match the GUID cleaning used across the pipeline (strip triple-quote/brace chars + whitespace)."""
    return (pd.Series(s).astype(str).str.replace('"""', "", regex=False)
            .str.replace("{", "", regex=False).str.replace("}", "", regex=False).str.strip())


def split_path(horizon):
    """Canonical path for a horizon's split file (local dir or gs:// bucket via SPLIT_DIR).
    Per-HORIZON (membership-based): the patient set is identical across lookback windows — the window
    only changes which EVENTS are pulled — so ALL windows of a horizon share this one split."""
    name = f"lung_{horizon}_split.parquet"
    return SPLIT_DIR.rstrip("/") + "/" + name


def make_split(guids, labels, seed=SEED, test_size=TEST_SIZE, valid_size=VALID_SIZE):
    """Build the canonical split DataFrame (columns: patient_guid, split) — deterministic, guid-keyed.
    `guids`/`labels` are per-PATIENT (one row per patient). Sorted by cleaned guid for canonical order."""
    df = pd.DataFrame({"patient_guid": list(guids), "cancer_class": np.asarray(labels).astype(int)})
    df["_cg"] = clean_guid(df["patient_guid"])
    df = df.drop_duplicates("_cg").sort_values("_cg").reset_index(drop=True)
    y = df["cancer_class"].to_numpy()
    idx = np.arange(len(df))
    trva, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    rel = valid_size / (1.0 - test_size)        # valid as a fraction of the train+valid remainder
    tr, va = train_test_split(trva, test_size=rel, random_state=seed, stratify=y[trva])
    split = np.empty(len(df), dtype=object)
    split[tr] = "train"; split[va] = "valid"; split[test] = "test"
    df["split"] = split
    return df[["patient_guid", "split", "cancer_class"]]


def save_split(df, path):
    if not (path.startswith("gs://")) :
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)        # pandas + pyarrow; gs:// handled via gcsfs if installed
    return path


def load_split(path):
    return pd.read_parquet(path)


def exists(path):
    if path.startswith("gs://"):
        try:
            import gcsfs
            return gcsfs.GCSFileSystem().exists(path)
        except Exception:
            return False
    return os.path.exists(path)


def load_or_make(horizon, guids, labels, path=None):
    """Load the saved split for `horizon` if present; else build it from (guids, labels) and save.
    Prints which path was taken so runs are auditable. Returns the split DataFrame."""
    path = path or split_path(horizon)
    if exists(path):
        df = load_split(path)
        print(f"[split] loaded canonical {horizon} split: {path}  "
              f"({(df['split']=='train').sum()} train / {(df['split']=='valid').sum()} valid / "
              f"{(df['split']=='test').sum()} test)")
        return df
    df = make_split(guids, labels)
    save_split(df, path)
    print(f"[split] CREATED canonical {horizon} split -> {path}  "
          f"({(df['split']=='train').sum()} train / {(df['split']=='valid').sum()} valid / "
          f"{(df['split']=='test').sum()} test)")
    return df


def guids_for(df, which):
    """Set of CLEANED patient_guids in the given split ('train'|'valid'|'test'). Match callers' guids
    after cleaning them with clean_guid() for a robust join."""
    return set(clean_guid(df.loc[df["split"] == which, "patient_guid"]).tolist())


def split_map(df):
    """{cleaned_guid -> 'train'|'valid'|'test'} from a canonical split DataFrame — for stamping a
    `split` column onto a feature matrix so the split travels WITH the data downstream."""
    return dict(zip(clean_guid(df["patient_guid"]), df["split"]))
