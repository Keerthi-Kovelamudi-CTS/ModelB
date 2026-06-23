"""
NOTE: SKIPPED when CATEGORIZED=True (run_v3 reads the hand-made category map directly instead of a
data-driven codelist). This runs only for the per-SNOMED path (V3_snomed).

Build the FE codelist (Phase-1 scoring -> 2_FE) for BOTH horizons — data-driven.

Pipeline:  build_score_counts.py -> "Combined Scoring (ML+Stat).py" -> build_codelist.py -> 2_FE

Built PER (horizon, lookback window) — each lookback re-discovers its codes on its OWN scores.
The codelist = a focused top-N data-driven set:
  * data-driven: codes passing 3 gates — Bonferroni p<0.05  AND  OR >= OR_MIN  AND  prevalence >= PREV_MIN
                 of cancer patients — then ranked by combined_rank and cut to the top TOP_N.

Per (horizon, lookback) it:
  1. reads the combined ranked scores  (output/{h}/{years}yr/Scores_lung/lung_combined_all.csv)
  2. builds the codelist (top-N eligible)
  3. one lean BigQuery aggregate marks each code value-bearing (>= MIN_VALUE_FRAC numeric)
  4. writes ../2_FE/codelist/{Astra|Nova}_{years}yr_OFF_codelist.csv  (Astra=12mo, Nova=1mo;
     columns: Code, Name, Value, combined_rank, odds_ratio, p_value_bonferroni, prevalence_pos; sorted by rank)

Run (VM/env with BigQuery, after Combined Scoring):  python build_codelist.py [12mo|1mo] [years]
Tunable via env: TOP_N (default 500), OR_MIN (2.0), PREV_MIN (0.01).
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # V3 root, for `import config` / `import events`
import config as C
import events
import splits                              # train-only filter for the value-bearing decision (leak-safe)
import artifacts                            # atomic write (no partial codelist on an interrupted run)

HORIZONS = C.horizons_from_argv(sys.argv[1:])   # e.g. `build_codelist.py 12mo 5`; hard-errors on a typo
# FE/scoring lookback in years: reads this lookback's scores and writes {Astra|Nova}_{YEARS}yr_OFF_codelist.csv
YEARS = next((int(a) for a in sys.argv[1:] if a.isdigit()), C.SCORING_YEARS_BEFORE)
MIN_VALUE_FRAC = C.MIN_VALUE_FRAC   # value-bearing threshold (KEEP IN SYNC with 2_FE/build_features)
TOP_N = C.TOP_N                     # # of top data-driven codes by combined_rank (eligible pool)
OR_MIN = C.OR_MIN                   # eligibility: risk-increasing odds ratio floor
PREV_MIN = C.PREV_MIN               # eligibility: >= this fraction of cancer patients


SCORES = os.path.join(HERE, "output", "{h}", f"{YEARS}yr", "Scores_lung", "lung_combined_all.csv")
OUT_DIR = os.path.join(HERE, "..", "2_FE", "codelist")


def value_bearing_flags(horizon):
    """Per-code fraction of events carrying a numeric value, computed LOCALLY from the shared GCS-cached
    cohort events (../events.py) — no BigQuery. Returns {code:int -> is_value_bearing:0/1}. The unified
    `code` is the SNOMED id for observations and the med_code_id for medications (matches build_features).
    Decided on the TRAIN split ONLY (leak-safe; the internal valid/test never informs the feature schema)."""
    ev = events.load_cohort_events(horizon, YEARS)
    ev = splits.filter_to_train(ev, horizon)   # TRAIN-only (matches build_features' value-code schema discipline)
    code = pd.Series(events.to_int64(ev["snomed_c_t_concept_id"]), index=ev.index).where(
        ev["event_type"].eq("observation"),
        pd.Series(events.to_int64(ev["med_code_id"]), index=ev.index))
    df = pd.DataFrame({"Code": code,
                       "num": pd.to_numeric(ev["value"], errors="coerce").notna().astype(int)})
    df = df.dropna(subset=["Code"])
    df["Code"] = df["Code"].astype("int64")
    g = df.groupby("Code")["num"].agg(["sum", "size"])
    is_vb = ((g["sum"] / g["size"].replace(0, pd.NA)) >= MIN_VALUE_FRAC).astype(int)
    print(f"  [{horizon}] value-bearing codes (local, from cached events): {int(is_vb.sum()):,} / {len(is_vb):,}")
    return dict(zip(is_vb.index, is_vb.values))


def build(horizon):
    d = pd.read_csv(SCORES.format(h=horizon))
    _cid = pd.Series(events.to_int64(d["code_id"]), index=d.index)          # exact (no float64 rounding)
    if "med_code_id" in d.columns:
        _cid = _cid.where(_cid.notna(), pd.Series(events.to_int64(d["med_code_id"]), index=d.index))
    d["Code"] = _cid
    d = d[d["Code"].notna()].copy(); d["Code"] = d["Code"].astype("int64")

    prev_pos = pd.to_numeric(d.get("prevalence_pos"), errors="coerce")
    if prev_pos.isna().all():                       # fallback if no prevalence_pos column
        prev_pos = pd.to_numeric(d.get("n_patient_count_pos"), errors="coerce") / \
                   pd.to_numeric(d.get("n_patient_count_total_pos"), errors="coerce")
    elig = d[(pd.to_numeric(d["p_value_bonferroni"], errors="coerce") < 0.05)
             & (pd.to_numeric(d["odds_ratio"], errors="coerce") >= OR_MIN)
             & (prev_pos >= PREV_MIN)]
    top = set(elig.nsmallest(TOP_N, "combined_rank")["Code"])   # combined_rank 1 = strongest
    sig = d[d["Code"].isin(top)].copy()
    print(f"  [{horizon}] eligible(Bonf<0.05, OR>={OR_MIN}, prev>={PREV_MIN:.0%})={len(elig):,} "
          f"-> top-{TOP_N} -> {sig['Code'].nunique():,} codes")

    vb = value_bearing_flags(horizon)            # code -> 0/1
    prev_sig = pd.to_numeric(sig.get("prevalence_pos"), errors="coerce")
    if prev_sig.isna().all():
        prev_sig = pd.to_numeric(sig.get("n_patient_count_pos"), errors="coerce") / \
                   pd.to_numeric(sig.get("n_patient_count_total_pos"), errors="coerce")
    # Codelist carries the SELECTION SCORES the top-N was chosen by (combined_rank = 1 strongest),
    # plus the eligibility values (odds ratio, Bonferroni p, prevalence in cancer patients). Sorted by rank.
    out = pd.DataFrame({
        "Code":  sig["Code"],
        "Name":  sig["term"].fillna(""),
        "Value": sig["Code"].map(vb).fillna(0).astype(int),       # 1 = value-bearing, 0 = code-only
        "combined_rank": pd.to_numeric(sig.get("combined_rank"), errors="coerce"),
        "odds_ratio": pd.to_numeric(sig.get("odds_ratio"), errors="coerce").round(3),
        "p_value_bonferroni": pd.to_numeric(sig.get("p_value_bonferroni"), errors="coerce"),
        "prevalence_pos": prev_sig.round(4),
    }).drop_duplicates("Code").sort_values("combined_rank")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, C.codelist_name(horizon, YEARS))
    artifacts.atomic_write(path, lambda t: out.to_csv(t, index=False))   # .tmp -> os.replace
    print(f"  [{horizon}] wrote {len(out):,} codes ({int(out.Value.sum())} value-bearing) -> {path}")


def main():
    for h in HORIZONS:
        print(f"\n===================  {h}  ===================")
        build(h)
    print("\nDone. FE reads 2_FE/codelist/{Astra,Nova}_{yr}yr_OFF_codelist.csv (Code column).")


if __name__ == "__main__":
    main()
