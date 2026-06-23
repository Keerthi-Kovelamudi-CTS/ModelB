"""One-off: pull the held-out LIFETIME cohort events to the GCS cache (both horizons), so downstream
held-out evals reuse them with zero BigQuery. Run on a VM with BigQuery + the env."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))            # V3/4_Heldout
sys.path.insert(0, os.path.dirname(HERE))                    # V3 root, for `import events`
import events
for h, gap in [("12mo", "12"), ("1mo", "1")]:
    sql = os.path.join(HERE, "SQL", f"heldout_test_{gap}mo.sql")
    ev = events.load_cohort_events(h, events.LIFETIME_YEARS, sql_path=sql, tag="heldout_")
    print(f"[{h}] held-out lifetime cached: {len(ev):,} events | {ev.patient_guid.nunique():,} patients")
print("HELDOUT_LIFETIME_DONE")
