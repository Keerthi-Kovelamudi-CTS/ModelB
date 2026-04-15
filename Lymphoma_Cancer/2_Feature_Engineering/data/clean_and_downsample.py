"""
Lymphoma Phase 2 — Clean & Downsample
  Step 1: Drop full-row duplicates
  Step 2: Downsample negatives to 1:10 at the patient level,
          using the SAME sampled negative set across obs + meds per window.

Writes back to the original file paths (overwrites). Set DRY_RUN=True to skip save.
Run: python clean_and_downsample.py   (from any cwd)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DRY_RUN = False
TARGET_RATIO = 10  # negatives per positive

DATA_DIR = Path(__file__).resolve().parent

# Deterministic per-window seed offsets (avoid Python's randomized hash())
WINDOW_OFFSETS = {'3mo': 0, '6mo': 1, '12mo': 2}

windows = {
    '12mo': {
        'obs':  DATA_DIR / '12mo' / 'lymphoma_12mo_obs.csv',
        'meds': DATA_DIR / '12mo' / 'lymphoma_12mo_med.csv',
    },
    '6mo': {
        'obs':  DATA_DIR / '6mo' / 'lymphoma_6mo_obs.csv',
        'meds': DATA_DIR / '6mo' / 'lymphoma_6mo_med.csv',
    },
    '3mo': {
        'obs':  DATA_DIR / '3mo' / 'lymphoma_3mo_obs.csv',
        'meds': DATA_DIR / '3mo' / 'lymphoma_3mo_med.csv',
    },
}


def run_window(window: str, paths: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  WINDOW: {window}")
    print(f"{'='*70}")

    for key, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing {key} file: {p}")

    obs = pd.read_csv(paths['obs'], low_memory=False)
    meds = pd.read_csv(paths['meds'], low_memory=False)

    # ─── STEP 1: Drop full-row duplicates ───
    obs_before, meds_before = len(obs), len(meds)
    obs = obs.drop_duplicates().reset_index(drop=True)
    meds = meds.drop_duplicates().reset_index(drop=True)
    print(f"  OBS:  {obs_before:,} → {len(obs):,}  (dropped {obs_before - len(obs):,} full-row dups)")
    print(f"  MEDS: {meds_before:,} → {len(meds):,}  (dropped {meds_before - len(meds):,} full-row dups)")

    # ─── STEP 2: Downsample negatives to 1:TARGET_RATIO ───
    # Cohort is defined by obs (SQL filters patients on obs codes; meds patients ⊆ obs patients).
    pos_patients = set(obs.loc[obs['label'] == 1, 'patient_guid'].unique())
    neg_all = set(obs.loc[obs['label'] == 0, 'patient_guid'].unique())
    n_pos = len(pos_patients)
    target_neg = n_pos * TARGET_RATIO

    print(f"\n  Positive patients: {n_pos:,}")
    print(f"  Negative patients: {len(neg_all):,}")
    print(f"  Target negatives at 1:{TARGET_RATIO}: {target_neg:,}")

    # Sanity: no patient should be both labels
    conflict = pos_patients & neg_all
    if conflict:
        print(f"  ⚠️  {len(conflict)} patients appear as BOTH pos and neg — dropping them from neg pool")
        neg_all = neg_all - pos_patients

    rng = np.random.RandomState(SEED + WINDOW_OFFSETS[window])
    neg_list = sorted(neg_all)  # sort → deterministic regardless of set hashing
    if len(neg_list) > target_neg:
        sampled_neg = set(rng.choice(neg_list, size=target_neg, replace=False))
    else:
        sampled_neg = set(neg_list)
        print(f"  ⚠️  Only {len(neg_list):,} negatives available — keeping all (ratio < 1:{TARGET_RATIO})")

    keep_patients = pos_patients | sampled_neg

    obs_clean = obs[obs['patient_guid'].isin(keep_patients)].copy()
    meds_clean = meds[meds['patient_guid'].isin(keep_patients)].copy()

    # ─── Results ───
    obs_pos_n = obs_clean.loc[obs_clean['label'] == 1, 'patient_guid'].nunique()
    obs_neg_n = obs_clean.loc[obs_clean['label'] == 0, 'patient_guid'].nunique()
    med_pos_n = meds_clean.loc[meds_clean['label'] == 1, 'patient_guid'].nunique()
    med_neg_n = meds_clean.loc[meds_clean['label'] == 0, 'patient_guid'].nunique()
    print(f"\n  After downsample:")
    print(f"  OBS:  {len(obs_clean):,} rows | pos={obs_pos_n:,} neg={obs_neg_n:,}"
          f"  ratio=1:{(obs_neg_n/obs_pos_n):.1f}" if obs_pos_n else "")
    print(f"  MEDS: {len(meds_clean):,} rows | pos={med_pos_n:,} neg={med_neg_n:,}"
          f"  ratio=1:{(med_neg_n/med_pos_n):.1f}" if med_pos_n else "")

    # Label leak re-check
    leak = (set(obs_clean.loc[obs_clean['label'] == 1, 'patient_guid'])
            & set(obs_clean.loc[obs_clean['label'] == 0, 'patient_guid']))
    print(f"  Label leakage (should be 0): {len(leak)}")

    # ─── Save (overwrite originals) ───
    if DRY_RUN:
        print(f"  DRY_RUN=True → skipped write")
        return

    obs_clean.to_csv(paths['obs'], index=False)
    meds_clean.to_csv(paths['meds'], index=False)
    print(f"  Saved: {paths['obs'].name} ({len(obs_clean):,} rows)")
    print(f"  Saved: {paths['meds'].name} ({len(meds_clean):,} rows)")


if __name__ == '__main__':
    for window in ('3mo', '6mo', '12mo'):
        run_window(window, windows[window])

    print(f"\n{'='*70}")
    print(f"  DONE — 6 files overwritten at 1:{TARGET_RATIO} negative:positive")
    print(f"{'='*70}")
