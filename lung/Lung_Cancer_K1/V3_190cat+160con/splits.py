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
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import config as C
    # HARD requirements (no getattr defaults): a config that imports but is MISSING any of these raises
    # AttributeError here rather than silently using a wrong value — e.g. a missing SPLIT_DIR must NOT
    # quietly write the canonical split to a local 'splits/' instead of the shared GCS path.
    SEED        = C.SPLIT_SEED      # the SPLIT seed — decoupled from RANDOM_STATE so model-seed changes never reshuffle the cohort
    TEST_SIZE   = C.TEST_SIZE
    VALID_SIZE  = C.CALIB_SIZE
    SPLIT_DIR   = C.SPLIT_DIR
    FE_YEARS    = C.FE_YEARS_BEFORE
except ModuleNotFoundError:             # config GENUINELY absent (true standalone use) -> defaults, loudly
    import warnings
    warnings.warn("splits.py: config.py not importable — falling back to hardcoded defaults "
                  "(SEED=42, TEST=0.10, VALID=0.10, SPLIT_DIR='splits', FE_YEARS=5)", stacklevel=2)
    SEED, TEST_SIZE, VALID_SIZE, SPLIT_DIR, FE_YEARS = 42, 0.10, 0.10, "splits", 5
# A config.py that imports but is BROKEN (renamed/typo'd constant) now RAISES instead of silently
# using wrong constants — we must never train on the wrong split fractions/seed.


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
    # ATOMIC: local write goes to .tmp then os.replace so an interrupted run never leaves a partial
    # canonical split that downstream stages would then trust. gs:// is a single-object write (atomic on close).
    if path.startswith("gs://"):
        df.to_parquet(path, index=False)    # gs:// handled via gcsfs if installed
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    return path


def load_split(path):
    return pd.read_parquet(path)


def backup_existing(path):
    """Copy an existing canonical split to a timestamped sibling ({path}.bak_YYYYmmdd_HHMMSS) so any
    re-creation is reversible. Works for local + gs:// (re-writes the tiny parquet). Returns the backup path."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{path}.bak_{ts}"
    save_split(load_split(path), bak)
    return bak


def exists(path):
    if path.startswith("gs://"):
        try:
            import gcsfs
            return gcsfs.GCSFileSystem().exists(path)
        except Exception:
            return False
    return os.path.exists(path)


def load_or_make(horizon, guids, labels, path=None, force=False):
    """Load the saved split for `horizon` if present; else build it from (guids, labels) and save.
    `force=True` REBUILDS even if a split exists (e.g. the cohort changed) — overwrites the canonical
    artifact. Prints which path was taken so runs are auditable. Returns the split DataFrame."""
    path = path or split_path(horizon)
    if exists(path) and not force:
        df = load_split(path)
        print(f"[split] loaded canonical {horizon} split: {path}  "
              f"({(df['split']=='train').sum()} train / {(df['split']=='valid').sum()} valid / "
              f"{(df['split']=='test').sum()} test)")
        return df
    # OVERWRITE GUARD: re-creating an EXISTING canonical split silently changes train/valid/test
    # membership for the WHOLE team and breaks alignment with the shared (free-text) pipeline that
    # consumes the same split. So `--force` ALONE is NOT enough to clobber it — require an explicit
    # ALLOW_SPLIT_OVERWRITE=1 env IN ADDITION, and back the current split up (timestamped) first, so a
    # casual/accidental `make_split.py --force` cannot nuke the shared artifact. (Creating it the FIRST
    # time, when none exists, is unguarded.)
    if exists(path):                                  # implies force here (the not-force branch returned)
        if os.environ.get("ALLOW_SPLIT_OVERWRITE", "").strip().lower() not in ("1", "true", "yes"):
            raise SystemExit(
                f"[split] REFUSING to overwrite the existing canonical {horizon} split at {path}.\n"
                f"        --force alone won't do it: re-creating changes train/valid/test for EVERYONE "
                f"and breaks the shared-cohort alignment.\n"
                f"        If you truly intend to re-create it, re-run with  ALLOW_SPLIT_OVERWRITE=1  "
                f"(the current split is backed up first).")
        bak = backup_existing(path)
        print(f"[split] ALLOW_SPLIT_OVERWRITE=1: backed up existing {horizon} split -> {bak} before re-creating.")
    df = make_split(guids, labels)
    save_split(df, path)
    print(f"[split] {'RE-CREATED (--force)' if force else 'CREATED'} canonical {horizon} split -> {path}  "
          f"({(df['split']=='train').sum()} train / {(df['split']=='valid').sum()} valid / "
          f"{(df['split']=='test').sum()} test)")
    return df


def load_required(horizon, path=None):
    """Load the canonical split for `horizon`; HARD-ERROR if it's missing. Unlike load_or_make this
    never silently creates a split from a caller's cohort subset (which could overwrite the shared
    canonical artifact with a window-specific subset). Consumers (scoring/FE/stability) use this;
    make_split.py remains the sole creator via load_or_make."""
    path = path or split_path(horizon)
    if not exists(path):
        raise FileNotFoundError(
            f"canonical split for {horizon} not found at {path} — run make_split.py first. "
            f"Downstream stages must not create the split themselves.")
    df = load_split(path)
    print(f"[split] loaded canonical {horizon} split: {path}  "
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


def verify_stamp(df, horizon, guid_col="patient_guid", split_col="split"):
    """Belt-and-suspenders drift guard: assert the `split` column STAMPED on a matrix matches the
    canonical split file per-patient. Catches the 'cached stable matrix + changed canonical split'
    case — i.e. the silent 'test patient ends up in train' failure mode the manifest is meant to
    prevent. No-ops if the matrix lacks the guid/split columns (older matrices). Raises on any
    missing-from-canonical or train/valid/test mismatch."""
    if split_col not in df.columns or guid_col not in df.columns:
        return
    canon = split_map(load_required(horizon))
    stamped = dict(zip(clean_guid(df[guid_col]), df[split_col].astype(str)))
    missing = [g for g in stamped if g not in canon]
    mismatch = [(g, stamped[g], canon[g]) for g in stamped if g in canon and stamped[g] != canon[g]]
    if missing:
        raise ValueError(f"verify_stamp[{horizon}]: {len(missing)} stamped patients absent from the canonical "
                         f"split (e.g. {missing[:3]}) — stale/mismatched matrix; rebuild FE+stable.")
    if mismatch:
        raise ValueError(f"verify_stamp[{horizon}]: {len(mismatch)} patients whose stamped split != canonical "
                         f"(e.g. {mismatch[:3]}) — STALE stamp ('test in train' risk); rebuild FE+stable.")
    print(f"[split] verify_stamp[{horizon}]: stamped split matches canonical for {len(stamped):,} patients")


def filter_to_train(df, horizon, guid_col="patient_guid"):
    """Restrict a row-frame to the canonical TRAIN patients for `horizon` — the SINGLE leak-safe
    filter every fitting consumer (code scoring, FE value-schema, stability selection) must use, so
    the internal valid/test never informs anything fit on train. Centralized here (one source) rather
    than re-implemented per module. Held-out (no internal split) does NOT call this. HARD-ERRORS if the
    canonical split is missing (via load_required)."""
    tg = guids_for(load_required(horizon), "train")
    return df[clean_guid(df[guid_col]).isin(tg)].copy()
