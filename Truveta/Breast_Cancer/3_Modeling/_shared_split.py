"""
Shared train/val/test split for the dual-horizon B+C system.

WHY THIS EXISTS
===============
Prostate_B+C/ trains three models on the SAME patient cohort:
  - 12mo (Model B)
  - 1mo_5y (Model C₁)
  - 1mo_12mo (Model C₂)

For the dual-horizon evaluation to be apples-to-apples, Patient X must end up
in the SAME split (train/val/test) for ALL three models. Otherwise the
performance gap between 1mo_5y and 1mo_12mo could be a cohort artefact, not
a true lookback-window effect.

HOW IT WORKS
============
1. Compute the unified cohort = intersection of patient_guids that pass the
   ≥5-events filter in all 3 windows' lookback periods.
2. Run ONE stratified split (75/10/15, stratified by cancer_class, seed=42).
3. Save patient_guid → split assignment to shared_split.json.
4. Each model's training step reads this file and filters its feature matrix
   accordingly. Patient X is in TRAIN for all 3 models OR VAL for all 3 OR
   TEST for all 3 — never mixed.

The shared_split.json file becomes a hard contract — once written, it
shouldn't be regenerated unless the whole experiment is restarted.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_RATIO = 0.75
DEFAULT_VAL_RATIO   = 0.10
DEFAULT_TEST_RATIO  = 0.15
DEFAULT_SEED        = 42
DEFAULT_TEMPORAL_TEST_YEAR = 2025  # test = anchor_year >= this; trainval = < this


def _extract_anchor_year_per_patient(feature_matrix_path: Path, spine_csv_path: Optional[Path] = None) -> pd.Series:
    """Resolve patient_id → anchor_year by trying feature matrix → spine CSV.

    Returns a Series indexed by patient_id (str), values are integer years.
    Patients without resolvable anchor year are dropped from the returned Series.
    """
    df = pd.read_parquet(feature_matrix_path)
    if "patient_id" not in df.columns:
        df = df.reset_index()
    df["patient_id"] = df["patient_id"].astype(str)

    for col in ("anchor_year", "ANCHOR_YEAR"):
        if col in df.columns:
            ay = (df[["patient_id", col]].dropna()
                    .drop_duplicates("patient_id")
                    .set_index("patient_id")[col]
                    .astype(int))
            logger.info(f"    anchor_year resolved from feature matrix column '{col}': {len(ay):,} patients")
            return ay

    for col in ("anchor_date", "ANCHOR_DATE"):
        if col in df.columns:
            ay = (df[["patient_id", col]].dropna()
                    .drop_duplicates("patient_id")
                    .assign(_y=lambda d: pd.to_datetime(d[col], errors="coerce").dt.year)
                    .dropna(subset=["_y"])
                    .set_index("patient_id")["_y"]
                    .astype(int))
            logger.info(f"    anchor_year resolved from feature matrix date column '{col}': {len(ay):,} patients")
            return ay

    # Fallback: spine CSV (always has anchor_date per Truveta extract)
    if spine_csv_path is not None and spine_csv_path.exists():
        spine = pd.read_csv(spine_csv_path)
        if "anchor_date" in spine.columns:
            spine["patient_id"] = spine["patient_id"].astype(str)
            ay = (spine[["patient_id", "anchor_date"]].dropna()
                      .drop_duplicates("patient_id")
                      .assign(_y=lambda d: pd.to_datetime(d["anchor_date"], errors="coerce").dt.year)
                      .dropna(subset=["_y"])
                      .set_index("patient_id")["_y"]
                      .astype(int))
            logger.info(f"    anchor_year resolved from spine CSV: {len(ay):,} patients")
            return ay

    raise FileNotFoundError(
        f"Could not find anchor_year/anchor_date in {feature_matrix_path} or spine CSV. "
        f"Cannot do temporal split. Pass --mode random to skip."
    )


def build_unified_cohort(per_window_cleanup_paths: dict, min_events: int = 5) -> pd.DataFrame:
    """Find patients eligible for ALL windows (intersection).

    Parameters
    ----------
    per_window_cleanup_paths : dict[str, Path]
        Mapping of window name → cleanup parquet path. Each parquet must have
        patient_id as a column (or index) and a label column.
    min_events : int
        Minimum events filter (already applied by the SQL, but we double-check).

    Returns
    -------
    pd.DataFrame with columns:
        - patient_guid
        - cancer_class  (LABEL from the cleanup parquets; consistent across windows)
    """
    common_guids = None
    labels_by_window = {}
    for window, path in per_window_cleanup_paths.items():
        df = pd.read_parquet(path, columns=None)  # need patient_id + label — read all to be safe
        if "patient_id" not in df.columns:
            df = df.reset_index()
        if "patient_id" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{path}: missing patient_id or label column")

        guids = set(df["patient_id"].astype(str).tolist())
        labels_by_window[window] = (
            df[["patient_id", "label"]].astype({"patient_id": str}).set_index("patient_id")["label"]
        )
        if common_guids is None:
            common_guids = guids
        else:
            common_guids &= guids
        logger.info(f"  {window}: {len(guids):,} patients (running intersection: {len(common_guids):,})")

    # Sanity: labels should agree across windows for the same patient
    common_list = sorted(common_guids)
    label_arrays = [labels_by_window[w].reindex(common_list) for w in per_window_cleanup_paths.keys()]
    base = label_arrays[0]
    for w, arr in zip(list(per_window_cleanup_paths.keys())[1:], label_arrays[1:]):
        mismatch = (base != arr).sum()
        if mismatch:
            logger.warning(f"  ⚠ {mismatch} patients have label mismatch between first window and {w}")
    logger.info(f"  Unified cohort (intersection): {len(common_list):,} patients")

    return pd.DataFrame({"patient_guid": common_list, "cancer_class": base.astype(int).values})


def build_shared_split(
    unified_cohort: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio:   float = DEFAULT_VAL_RATIO,
    test_ratio:  float = DEFAULT_TEST_RATIO,
    seed:        int   = DEFAULT_SEED,
    balance_to_1to1: bool = True,
) -> dict:
    """Compute a stratified 3-way split on the unified cohort.

    If balance_to_1to1 is True (default), the majority class is randomly
    downsampled to the minority count BEFORE splitting, so TRAIN, VAL, and TEST
    all come out ~1:1. This is important because the SQL emits a ~1:1 cohort but
    downstream filters (the cross-window intersection in build_unified_cohort,
    and the gold-layer foreign-anchor min-events HAVING) can drop cancer and
    non-cancer at different rates — leaving the surviving cohort off 1:1. A
    stratified split only preserves whatever ratio it is handed, so we re-balance
    here to guarantee balanced val/test, not just train.

    Returns a dict ready to serialise as shared_split.json:
        {
          "train_guids": [...],
          "val_guids":   [...],
          "test_guids":  [...],
          "seed": 42,
          "stratify": "cancer_class",
          "ratios": {"train": 0.75, "val": 0.10, "test": 0.15},
          "n_total": N, "n_train": ..., "n_val": ..., "n_test": ...,
          "n_cancer": ..., "n_non_cancer": ...
        }
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
        raise ValueError(
            f"Ratios must sum to 1.0; got {train_ratio} + {val_ratio} + {test_ratio} = "
            f"{train_ratio + val_ratio + test_ratio}"
        )
    guids = np.array(unified_cohort["patient_guid"].tolist())
    y = unified_cohort["cancer_class"].astype(int).to_numpy()

    # ── Balance the surviving cohort to exactly 1:1 before splitting ──────────
    if balance_to_1to1:
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_keep = min(len(pos_idx), len(neg_idx))
        rng = np.random.RandomState(seed)
        pos_keep = rng.choice(pos_idx, size=n_keep, replace=False) if len(pos_idx) > n_keep else pos_idx
        neg_keep = rng.choice(neg_idx, size=n_keep, replace=False) if len(neg_idx) > n_keep else neg_idx
        keep = np.sort(np.concatenate([pos_keep, neg_keep]))
        n_dropped = len(y) - len(keep)
        if n_dropped:
            majority = "cancer" if len(pos_idx) > len(neg_idx) else "non-cancer"
            logger.info(
                f"  Balanced cohort to 1:1 — kept {n_keep:,} cancer + {n_keep:,} non-cancer; "
                f"dropped {n_dropped:,} surplus {majority} patients "
                f"(was {len(pos_idx):,} cancer / {len(neg_idx):,} non-cancer)"
            )
        else:
            logger.info(f"  Cohort already 1:1 ({len(pos_idx):,} cancer / {len(neg_idx):,} non-cancer) — no downsampling")
        guids = guids[keep]
        y = y[keep]

    guids = guids.tolist()

    # 85% trainval / 15% test
    trainval_guids, test_guids, y_trainval, _ = train_test_split(
        guids, y, test_size=test_ratio, stratify=y, random_state=seed,
    )
    # 75/10 of trainval = 0.882 train / 0.118 val (after sanity)
    val_frac = val_ratio / (train_ratio + val_ratio)
    train_guids, val_guids = train_test_split(
        trainval_guids, test_size=val_frac, stratify=y_trainval, random_state=seed,
    )

    return {
        "train_guids": sorted(train_guids),
        "val_guids":   sorted(val_guids),
        "test_guids":  sorted(test_guids),
        "seed": int(seed),
        "stratify": "cancer_class",
        "balanced_to_1to1": bool(balance_to_1to1),
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "n_total":      int(len(guids)),
        "n_train":      int(len(train_guids)),
        "n_val":        int(len(val_guids)),
        "n_test":       int(len(test_guids)),
        "n_cancer":     int((y == 1).sum()),
        "n_non_cancer": int((y == 0).sum()),
    }


def build_temporal_split(
    unified_cohort: pd.DataFrame,
    anchor_year_by_guid: pd.Series,
    temporal_test_year: int = DEFAULT_TEMPORAL_TEST_YEAR,
    temporal_test_year_max: Optional[int] = None,
    val_frac_of_trainval: float = DEFAULT_VAL_RATIO / (DEFAULT_TRAIN_RATIO + DEFAULT_VAL_RATIO),
    seed: int = DEFAULT_SEED,
) -> dict:
    """Temporal-holdout split: test = anchor_year >= temporal_test_year, trainval = earlier.

    If temporal_test_year_max is set, also caps test at <= that year (drops any patients
    with anchor_year > max — useful when newer years are data-lag-affected and unreliable).

    Within trainval, val is a random stratified subset (default ~12% of trainval).
    """
    guids = unified_cohort["patient_guid"].astype(str).tolist()
    y = unified_cohort.set_index(unified_cohort["patient_guid"].astype(str))["cancer_class"].astype(int)

    ay_aligned = anchor_year_by_guid.reindex(guids)
    n_missing = ay_aligned.isna().sum()
    if n_missing:
        logger.warning(f"  ⚠ {n_missing:,} patients have no anchor_year — excluded from temporal split")
        keep = ay_aligned.notna()
        guids_kept = [g for g, k in zip(guids, keep) if k]
        ay_aligned = ay_aligned.loc[guids_kept]
        y = y.loc[guids_kept]
    else:
        guids_kept = guids

    # Drop patients beyond max test year (data-lag artifact filter)
    n_dropped_max = 0
    if temporal_test_year_max is not None:
        beyond_max = ay_aligned > temporal_test_year_max
        n_dropped_max = int(beyond_max.sum())
        if n_dropped_max:
            logger.info(f"  Dropping {n_dropped_max:,} patients with anchor_year > {temporal_test_year_max} (data-lag filter)")
        ay_aligned = ay_aligned[~beyond_max]
        y = y.loc[ay_aligned.index]
        guids_kept = ay_aligned.index.tolist()

    is_test = ay_aligned >= temporal_test_year
    test_guids = sorted(ay_aligned.index[is_test].tolist())
    trainval_guids = sorted(ay_aligned.index[~is_test].tolist())

    y_trainval = y.loc[trainval_guids].to_numpy()
    train_guids, val_guids = train_test_split(
        trainval_guids, test_size=val_frac_of_trainval,
        stratify=y_trainval, random_state=seed,
    )

    y_all = y.to_numpy()
    return {
        "mode":                "temporal",
        "temporal_test_year":  int(temporal_test_year),
        "train_guids":         sorted(train_guids),
        "val_guids":           sorted(val_guids),
        "test_guids":          test_guids,
        "seed":                int(seed),
        "stratify":            "cancer_class (within trainval only)",
        "n_total":             int(len(guids_kept)),
        "n_train":             int(len(train_guids)),
        "n_val":               int(len(val_guids)),
        "n_test":              int(len(test_guids)),
        "n_cancer":            int((y_all == 1).sum()),
        "n_non_cancer":        int((y_all == 0).sum()),
        "n_dropped_no_anchor": int(n_missing),
        "test_year_range":     [int(ay_aligned[is_test].min()) if is_test.any() else None,
                                int(ay_aligned[is_test].max()) if is_test.any() else None],
        "trainval_year_range": [int(ay_aligned[~is_test].min()) if (~is_test).any() else None,
                                int(ay_aligned[~is_test].max()) if (~is_test).any() else None],
    }


def write_shared_split(split: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
    logger.info(
        f"  wrote shared split → {path.name}: "
        f"train {split['n_train']:,} | val {split['n_val']:,} | test {split['n_test']:,} "
        f"({split['n_cancer']:,} cancer / {split['n_non_cancer']:,} non-cancer)"
    )


def load_shared_split(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def apply_shared_split(feature_matrix: pd.DataFrame, shared_split: dict, which: str):
    """Filter feature_matrix to the patients in a given split.

    Parameters
    ----------
    feature_matrix : DataFrame with patient_id column or index
    shared_split   : dict loaded from shared_split.json
    which          : 'train', 'val', or 'test'

    Returns
    -------
    Filtered DataFrame in the same order as shared_split[which + '_guids'].
    """
    key = which + "_guids"
    if key not in shared_split:
        raise KeyError(f"shared_split missing '{key}' (expected one of train/val/test)")
    guids = shared_split[key]
    fm = feature_matrix.copy()
    if "patient_id" not in fm.columns:
        fm = fm.reset_index()
    fm["patient_id"] = fm["patient_id"].astype(str)
    fm = fm[fm["patient_id"].isin(guids)].set_index("patient_id")
    return fm.loc[[g for g in guids if g in fm.index]]


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2_Feature_Engineering"))
    import config as fe_cfg

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(description="Build the unified cohort + shared train/val/test split.")
    p.add_argument("--features-root", default=str(fe_cfg.FEATURES_DIR),
                   help="Root containing {window}/breast_feature_matrix.parquet")
    p.add_argument("--out", default=str(fe_cfg.SHARED_SPLIT_PATH))
    p.add_argument("--min-events", type=int, default=fe_cfg.MIN_EVENTS_PER_WINDOW)
    p.add_argument("--mode", choices=["random", "temporal"], default="random",
                   help="random = stratified 75/10/15 (default, matches Prostate_B+C — balanced "
                        "train/val/test, no temporal holdout); temporal = test >= --test-year, train+val before")
    p.add_argument("--test-year", type=int, default=DEFAULT_TEMPORAL_TEST_YEAR,
                   help="For --mode temporal: test set = patients with anchor_year >= this year")
    p.add_argument("--test-year-max", type=int, default=None,
                   help="Optional: cap test set at anchor_year <= this year (drops data-lag-affected newer years)")
    p.add_argument("--raw-root", default=str(fe_cfg.RAW_DIR),
                   help="Where to find Truveta spine CSVs if feature matrix lacks anchor info")
    args = p.parse_args()

    features_root = Path(args.features_root)
    paths = {
        w: features_root / w / "breast_feature_matrix.parquet"
        for w in fe_cfg.WINDOWS
    }
    for w, pth in paths.items():
        if not pth.exists():
            raise FileNotFoundError(f"Missing feature matrix for {w}: {pth}")

    logger.info("Building unified cohort across windows: " + ", ".join(fe_cfg.WINDOWS))
    cohort = build_unified_cohort(paths, min_events=args.min_events)

    if args.mode == "temporal":
        logger.info(f"Mode: TEMPORAL holdout (test = anchor_year >= {args.test_year})")
        any_window = next(iter(fe_cfg.WINDOWS))
        any_path = paths[any_window]
        spine_csv = Path(args.raw_root) / any_window / "breast_spine.csv"
        anchor_by_guid = _extract_anchor_year_per_patient(any_path, spine_csv_path=spine_csv)
        split = build_temporal_split(cohort, anchor_by_guid, temporal_test_year=args.test_year,
                                       temporal_test_year_max=args.test_year_max)
        logger.info(f"  Train years: {split['trainval_year_range']}")
        logger.info(f"  Test  years: {split['test_year_range']}")
    else:
        logger.info("Mode: RANDOM stratified 75/10/15")
        split = build_shared_split(cohort)

    write_shared_split(split, args.out)
    logger.info("  ✅ done")
