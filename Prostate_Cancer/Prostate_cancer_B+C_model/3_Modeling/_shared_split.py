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


def build_unified_cohort(per_window_cleanup_paths: dict, min_events: int = 5) -> pd.DataFrame:
    """Find patients eligible for ALL windows (intersection).

    Parameters
    ----------
    per_window_cleanup_paths : dict[str, Path]
        Mapping of window name → cleanup parquet path. Each parquet must have
        PATIENT_GUID as a column (or index) and a LABEL column.
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
        df = pd.read_parquet(path, columns=None)  # need PATIENT_GUID + LABEL — read all to be safe
        if "PATIENT_GUID" not in df.columns:
            df = df.reset_index()
        if "PATIENT_GUID" not in df.columns or "LABEL" not in df.columns:
            raise ValueError(f"{path}: missing PATIENT_GUID or LABEL column")

        guids = set(df["PATIENT_GUID"].astype(str).tolist())
        labels_by_window[window] = (
            df[["PATIENT_GUID", "LABEL"]].astype({"PATIENT_GUID": str}).set_index("PATIENT_GUID")["LABEL"]
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
) -> dict:
    """Compute a stratified 3-way split on the unified cohort.

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

    # ── Balance the surviving cohort to exactly 1:1 BEFORE splitting ──────────
    # SQL oversamples non-cancer 5x; downsample the majority to the minority
    # count here so TRAIN, VAL, and TEST all come out ~1:1 (not just train).
    _pos = np.where(y == 1)[0]; _neg = np.where(y == 0)[0]
    _n = min(len(_pos), len(_neg))
    _rng = np.random.RandomState(seed)
    _pk = _rng.choice(_pos, _n, replace=False) if len(_pos) > _n else _pos
    _nk = _rng.choice(_neg, _n, replace=False) if len(_neg) > _n else _neg
    _keep = np.sort(np.concatenate([_pk, _nk]))
    if len(_keep) < len(y):
        _maj = 'cancer' if len(_pos) > len(_neg) else 'non-cancer'
        logger.info(f"  Balanced cohort to 1:1 — kept {_n:,} cancer + {_n:,} non-cancer; "
                    f"dropped {len(y)-len(_keep):,} surplus {_maj}")
    guids = guids[_keep].tolist(); y = y[_keep]

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
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "n_total":      int(len(guids)),
        "n_train":      int(len(train_guids)),
        "n_val":        int(len(val_guids)),
        "n_test":       int(len(test_guids)),
        "n_cancer":     int((y == 1).sum()),
        "n_non_cancer": int((y == 0).sum()),
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
    feature_matrix : DataFrame with PATIENT_GUID column or index
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
    if "PATIENT_GUID" not in fm.columns:
        fm = fm.reset_index()
    fm["PATIENT_GUID"] = fm["PATIENT_GUID"].astype(str)
    fm = fm[fm["PATIENT_GUID"].isin(guids)].set_index("PATIENT_GUID")
    return fm.loc[[g for g in guids if g in fm.index]]


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2_Feature_Engineering"))
    import config as fe_cfg

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(description="Build the unified cohort + shared train/val/test split.")
    p.add_argument("--cleanup-root", default=str(fe_cfg.CLEANUP_RESULTS),
                   help="Root containing {window}/feature_matrix_clean_{window}.parquet")
    p.add_argument("--out", default=str(fe_cfg.SHARED_SPLIT_PATH))
    p.add_argument("--min-events", type=int, default=fe_cfg.MIN_EVENTS_PER_WINDOW)
    args = p.parse_args()

    cleanup_root = Path(args.cleanup_root)
    paths = {
        w: cleanup_root / w / f"feature_matrix_clean_{w}.parquet"
        for w in fe_cfg.WINDOWS
    }
    for w, pth in paths.items():
        if not pth.exists():
            raise FileNotFoundError(f"Missing cleanup parquet for {w}: {pth}")

    logger.info("Building unified cohort across windows: " + ", ".join(fe_cfg.WINDOWS))
    cohort = build_unified_cohort(paths, min_events=args.min_events)
    split = build_shared_split(cohort)
    write_shared_split(split, args.out)
    logger.info("  ✅ done")
