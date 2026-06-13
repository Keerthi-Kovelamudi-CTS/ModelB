"""
Held-out evaluation for the Lung model (touch-once labelled cohort).

Scores a TRAINED lookback model on the real held-out set, processed IDENTICALLY to training:
  heldout_test_{GAP}mo.sql  --(BigQuery export)-->  held-out events
     -> build_matrix(... xpoll_fit=False, xpoll_ref=<train ref>)  (FE + TRAIN xpoll percentiles,
        count->0 / value->NaN, reindexed to the held-out patients)
     -> predict_unseen.CancerPredictor(model).predict(matrix)      (TRAIN encoders/medians/scaler/
        feature-selector via value_cols; base-model proba; per-age-band King-Zeng to true prevalence;
        Sens/Spec/PPV/NPV vs the known labels)

Env:  GAP=12|1   NC_RATIO=1   WINDOW=5yr|10yr|20yr|lifetime
Run (on the VM, AFTER the matching model exists):  GAP=12 WINDOW=5yr python evaluate_heldout.py
Outputs -> {GAP}mo_1to{NC_RATIO}/lookback/{WINDOW}/heldout_{...}.csv + metrics printed.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
GAP = os.environ.get("GAP", "12")
NC_RATIO = os.environ.get("NC_RATIO", "1")
WINDOW = os.environ.get("WINDOW", "5yr")
L = {"5yr": 5, "10yr": 10, "20yr": 20, "lifetime": 100}[WINDOW]

# import the shared pipeline (build_matrix etc.); it reads GAP/NC_RATIO from env at import
sys.path.insert(0, os.path.join(ROOT, "2_FE"))
sys.path.insert(0, os.path.join(ROOT, "3_Modeling"))
import run_lookback_experiment as rl          # build_matrix, full_patient_frame, rp (export), fe, ef
from predict_unseen import CancerPredictor

OUT_DIR = os.path.join(ROOT, f"{GAP}mo_1to{NC_RATIO}", "lookback", WINDOW)
MODEL = os.path.join(OUT_DIR, f"model_{WINDOW}_1to1.joblib")
XREF = os.path.join(OUT_DIR, f"xpoll_ref_{WINDOW}.json")
SQL = os.path.join(ROOT, "2_FE", "SQL", f"heldout_test_{GAP}mo.sql")


def main():
    assert os.path.exists(MODEL), f"trained model missing: {MODEL} (train the {WINDOW} window first)"
    assert os.path.exists(SQL), f"held-out SQL missing: {SQL}"
    print(f"{'='*70}\nHELD-OUT EVAL — {GAP}mo 1:{NC_RATIO} {WINDOW}\n{'='*70}")

    # 1) export the held-out cohort from BigQuery
    ho_events = os.path.join(OUT_DIR, f"heldout_events_{WINDOW}.csv")
    rl.rp.export_query_to_csv(open(SQL).read(), ho_events,
                              gcs_prefix=rl.GCS, tag=f"heldout_{GAP}mo_r{NC_RATIO}")
    ev = pd.read_csv(ho_events, low_memory=False)
    print(f"[heldout] {len(ev):,} events | {ev['patient_guid'].nunique():,} patients")

    # 2) build the held-out FE matrix, applying the TRAIN xpoll reference (leakage-safe)
    frame = rl.full_patient_frame(ev)
    ho_matrix = os.path.join(OUT_DIR, f"heldout_features_{WINDOW}.csv")
    xref = XREF if os.path.exists(XREF) else None
    if xref is None:
        print(f"[heldout] WARNING: no train xpoll ref at {XREF} — percentiles will be absent.")
    rl.build_matrix(ev, L, ho_matrix, frame, xpoll_fit=False, xpoll_ref_path=xref)

    # 3) score with the trained model (TRAIN-fitted transforms applied inside CancerPredictor)
    res = CancerPredictor(MODEL).predict(ho_matrix)

    # 4) persist a tidy metrics row + the per-patient scores
    if res.get("metrics"):
        m = res["metrics"]
        pd.DataFrame([{**{"gap": GAP, "window": WINDOW, "n": res["n_samples"],
                          "n_pos": int(np.nansum(res.get("y_true", 0)))}, **m}]
                     ).to_csv(os.path.join(OUT_DIR, f"heldout_metrics_{WINDOW}.csv"), index=False)
        print(f"\n[heldout] metrics -> {OUT_DIR}/heldout_metrics_{WINDOW}.csv")
    np.savez(os.path.join(OUT_DIR, f"heldout_scores_{WINDOW}.npz"),
             y=np.asarray(res.get("y_true")) if res.get("y_true") is not None else np.array([]),
             p_raw=np.asarray(res.get("probabilities_raw")),
             p_cal=np.asarray(res.get("probabilities_calibrated")) if res.get("probabilities_calibrated") is not None else np.array([]))
    print(f"[heldout] scores -> {OUT_DIR}/heldout_scores_{WINDOW}.npz")
    print("HELD-OUT EVAL COMPLETE")


if __name__ == "__main__":
    main()
