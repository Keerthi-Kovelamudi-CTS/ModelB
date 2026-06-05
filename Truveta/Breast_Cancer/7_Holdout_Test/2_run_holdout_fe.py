"""
Truveta breast — run FE on the holdout cohort.

Truveta's FE happens via 0_extract.py + 1_build_features.py against a
`truveta_gold.breast_*_cohort_events_{window}` table. For holdout, we just
point those scripts at the `breast_holdout_cohort_events_{window}` tables
built by 1_preprocess_holdout.py.

WORKFLOW
========
For each window:
  1. Call 0_extract.py --window holdout_<w> --cohort-table <holdout_table>
     → writes data/raw/holdout_<w>/breast_patient_code_aggregates.csv (and spine)
  2. Call 1_build_features.py --window holdout_<w>
     → writes data/features/holdout_<w>/breast_feature_matrix.parquet

No config patching needed — both scripts accept arbitrary window labels and
0_extract.py honors a --cohort-table override.

Usage:
    python 2_run_holdout_fe.py
    python 2_run_holdout_fe.py --windows 1mo 12mo
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent
FE_DIR = BASE / "2_Feature_Engineering"

sys.path.insert(0, str(FE_DIR))
import config as fe_config


def main(windows):
    for w in windows:
        holdout_table = f"prj-cts-ai-dev-sp.truveta_gold.breast_holdout_cohort_events_{w}"
        holdout_window_label = f"holdout_{w}"
        print(f"\n[{w}] running FE on {holdout_table}")

        r1 = subprocess.run(
            ["python3", "0_extract.py",
             "--window", holdout_window_label,
             "--cohort-table", holdout_table],
            cwd=str(FE_DIR), capture_output=True, text=True,
        )
        if r1.returncode != 0:
            print(r1.stdout); print(r1.stderr, file=sys.stderr)
            sys.exit(f"[{w}] 0_extract.py failed")
        print(r1.stdout.strip().splitlines()[-1] if r1.stdout else "")

        r2 = subprocess.run(
            ["python3", "1_build_features.py", "--window", holdout_window_label],
            cwd=str(FE_DIR), capture_output=True, text=True,
        )
        if r2.returncode != 0:
            print(r2.stdout); print(r2.stderr, file=sys.stderr)
            sys.exit(f"[{w}] 1_build_features.py failed")
        print(r2.stdout.strip().splitlines()[-1] if r2.stdout else "")

        print(f"[{w}] ✓ holdout FE done — feature matrix at "
              f"data/features/{holdout_window_label}/breast_feature_matrix.parquet")

    print("\nDone. Next: run 3_Modeling/2_predict_unseen.py for each holdout window, "
          "then 3_combined_evaluation.py.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", default=fe_config.WINDOWS,
                    choices=fe_config.WINDOWS,
                    help="Which windows to run holdout FE for (the training windows; "
                         "the script appends 'holdout_' to derive the actual labels)")
    args = ap.parse_args()
    main(args.windows)
