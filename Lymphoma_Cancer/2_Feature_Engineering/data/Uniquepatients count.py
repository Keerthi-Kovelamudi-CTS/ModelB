"""
Lymphoma Phase 2 — Sanity Checks (all windows)
Resolves CSV paths next to this script (data/3mo, data/6mo, data/12mo).
Run: python sanitycheck.py   (from any cwd)
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

# Filenames under data/{window}/ — adjust if your exports differ
files = {
    "12mo": {
        "obs": DATA_DIR / "12mo" / "lymphoma_12mo_obs.csv",
        "meds": DATA_DIR / "12mo" / "lymphoma_12mo_med.csv",
    },
    "6mo": {
        "obs": DATA_DIR / "6mo" / "lymphoma_6mo_obs.csv",
        "meds": DATA_DIR / "6mo" / "lymphoma_6mo_med.csv",
    },
    "3mo": {
        "obs": DATA_DIR / "3mo" / "lymphoma_3mo_obs.csv",
        "meds": DATA_DIR / "3mo" / "lymphoma_3mo_med.csv",
    },
}

# ═══════════════════════════════════════════════════════
# RUN CHECKS
# ═══════════════════════════════════════════════════════
for window, paths in files.items():
    print(f"\n{'='*70}")
    print(f"  WINDOW: {window}")
    print(f"{'='*70}")

    for key, p in paths.items():
        if not Path(p).is_file():
            raise FileNotFoundError(f"Missing {key} file: {p}")

    obs = pd.read_csv(paths["obs"])
    meds = pd.read_csv(paths["meds"])

    for name, df in [('OBS', obs), ('MEDS', meds)]:
        print(f"\n  --- {name} ---")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {df.shape[1]}")
        print(f"  Unique patients: {df['patient_guid'].nunique():,}")

        pos = df[df['label']==1]['patient_guid'].nunique()
        neg = df[df['label']==0]['patient_guid'].nunique()
        ratio = neg / pos if pos > 0 else 0
        print(f"  Positive (label=1): {pos:,} patients")
        print(f"  Negative (label=0): {neg:,} patients")
        print(f"  Ratio: 1:{ratio:.1f}")

        print(f"  Events per label: label=1 → {len(df[df['label']==1]):,} rows, label=0 → {len(df[df['label']==0]):,} rows")

        if 'time_window' in df.columns:
            tw = df['time_window'].value_counts().to_dict()
            print(f"  Window A: {tw.get('A', 0):,} rows, Window B: {tw.get('B', 0):,} rows")

        print(f"  Unique codes: {df['code_id'].nunique()}")

    # Patient overlap
    obs_pos = set(obs[obs['label']==1]['patient_guid'].unique())
    obs_neg = set(obs[obs['label']==0]['patient_guid'].unique())
    med_pos = set(meds[meds['label']==1]['patient_guid'].unique())
    med_neg = set(meds[meds['label']==0]['patient_guid'].unique())
    obs_all = set(obs['patient_guid'].unique())
    med_all = set(meds['patient_guid'].unique())

    print(f"\n  --- PATIENT OVERLAP ---")
    print(f"  Positive: obs={len(obs_pos):,}, meds={len(med_pos):,}, shared={len(obs_pos & med_pos):,}, obs-only={len(obs_pos - med_pos):,}, meds-only={len(med_pos - obs_pos):,}")
    print(f"  Negative: obs={len(obs_neg):,}, meds={len(med_neg):,}, shared={len(obs_neg & med_neg):,}, obs-only={len(obs_neg - med_neg):,}, meds-only={len(med_neg - obs_neg):,}")
    print(f"  Total:    obs={len(obs_all):,}, meds={len(med_all):,}, shared={len(obs_all & med_all):,}")

    # Any label overlap (patient in both positive and negative)?
    label_leak = obs_pos & obs_neg
    if label_leak:
        print(f"  ⚠️  LABEL LEAK: {len(label_leak)} patients appear in BOTH pos and neg!")
    else:
        print(f"  ✅ No label leakage")

print(f"\n{'='*70}")
print(f"  DONE")
print(f"{'='*70}")
