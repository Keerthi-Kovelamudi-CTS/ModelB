"""
╔══════════════════════════════════════════════════════════════════╗
║  LEUKAEMIA — Phase 1.4 (Clinical Review Bridge)                ║
║  Compare Ranking.py output vs. existing curated codelist.     ║
║                                                                ║
║  Inputs:                                                       ║
║    - Output/Scores_leukaemia/leukaemia_obs_all.csv               ║
║    - Output/Scores_leukaemia/leukaemia_meds_all.csv              ║
║    - ../codelists/leukaemia_curated_codes_v2.tsv                ║
║                                                                ║
║  Outputs (Output/Compare_to_curated/):                         ║
║    - leukaemia_obs_review.tsv                                   ║
║    - leukaemia_meds_review.tsv                                  ║
║    - leukaemia_combined_review.tsv                              ║
║    - compare_summary_YYYYMMDD_HHMMSS.log                       ║
║                                                                ║
║  Per-row columns:                                              ║
║    code, code_type, term, feature_type, in_curated_v2,         ║
║    current_category, combined_rank, stat_rank, ml_rank,        ║
║    odds_ratio, or_ci_lower, or_ci_upper, p_value_bonferroni,   ║
║    prevalence_pos, prevalence_neg, prevalence_diff,            ║
║    n_patient_count_pos, n_patient_count_neg,                   ║
║    combined_score, confidence_tier, stability_score,           ║
║    action_suggested, notes                                     ║
║                                                                ║
║  Action heuristic (clinician overrides anyway):                ║
║    KEEP             — in curated AND stat-significant + OR>1.2 ║
║    KEEP_BUT_WEAK    — in curated but low signal (review)       ║
║    KEEP_BUT_RARE    — in curated, didn't appear in ranking     ║
║    ADD_HIGH         — not in curated AND in top 150            ║
║    ADD_MEDIUM       — not in curated AND in top 500 (not 150)  ║
║    IGNORE           — not in curated AND below top 500         ║
║                                                                ║
║  Run: python compare_to_curated.py                             ║
║  Estimated time: <10s                                          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "cancer_type": "leukaemia",
    "scores_dir": os.path.join(_SCRIPT_DIR, "Output", "Scores_leukaemia"),
    "curated_tsv": os.path.join(
        _SCRIPT_DIR, "..", "codelists", "leukaemia_curated_codes_v2.tsv"
    ),
    "output_dir": os.path.join(_SCRIPT_DIR, "Output", "Compare_to_curated"),
    "obs_all_csv": "leukaemia_obs_all.csv",
    "meds_all_csv": "leukaemia_meds_all.csv",
    # Action heuristic thresholds — clinician overrides anyway.
    "top_high_threshold": 150,
    "top_medium_threshold": 500,
    "min_odds_ratio_for_keep": 1.2,
    "max_pbonf_for_keep": 0.05,
}


# ════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════


def setup_logging():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(CONFIG["output_dir"], f"compare_summary_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════


def _normalise_code(series):
    """Coerce to stripped strings, drop nan/empty so .merge works on the key."""
    return (
        series.astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def load_curated(curated_tsv_path):
    if not os.path.exists(curated_tsv_path):
        raise FileNotFoundError(f"Curated codelist not found: {curated_tsv_path}")

    df = pd.read_csv(curated_tsv_path, sep="\t", dtype={"CODE": str})
    df.columns = [c.strip() for c in df.columns]
    required = {"EVENT_TYPE", "CATEGORY", "TERM", "CODE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Curated TSV missing columns: {missing}. Found: {list(df.columns)}"
        )
    df["CODE"] = _normalise_code(df["CODE"])
    df["EVENT_TYPE"] = df["EVENT_TYPE"].str.upper().str.strip()
    df = df.dropna(subset=["CODE"]).copy()

    obs = df[df["EVENT_TYPE"] == "OBS"].copy()
    med = df[df["EVENT_TYPE"] == "MED"].copy()

    logger.info(
        f"  📋 Curated v2: {len(df):,} total — {len(obs):,} OBS + {len(med):,} MED"
    )
    return obs.rename(columns={"CODE": "code", "CATEGORY": "current_category",
                               "TERM": "curated_term"}), \
           med.rename(columns={"CODE": "code", "CATEGORY": "current_category",
                               "TERM": "curated_term"})


def load_ranked(csv_path, feature_type):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Ranking.py output not found: {csv_path}\n"
            "  → Run Ranking.py first to produce leukaemia_*_all.csv files."
        )

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Code column may be 'code_id', 'SNOMED_ID', 'DMD Code', or 'snomed_id_pos'/'med_code_id_pos'
    # depending on what Ranking.py emitted. We try them in order.
    code_col_candidates = ["code_id", "SNOMED_ID", "DMD Code", "snomed_id_pos",
                           "med_code_id_pos", "snomed_id", "med_code_id"]
    code_col = next((c for c in code_col_candidates if c in df.columns), None)
    if code_col is None:
        raise ValueError(
            f"No usable code column in {csv_path}. Tried: {code_col_candidates}. "
            f"Found: {list(df.columns)}"
        )

    df = df.rename(columns={code_col: "code"})
    df["code"] = _normalise_code(df["code"])
    df["feature_type"] = feature_type

    logger.info(f"  📊 {feature_type}: {len(df):,} ranked codes from {os.path.basename(csv_path)}")
    return df


def _safe_int(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _safe_float(series):
    return pd.to_numeric(series, errors="coerce")


def decide_action(row, top_high, top_medium, min_or, max_pbonf):
    """Vector-friendly via .apply(); per-row clinician-facing recommendation."""
    in_curated = bool(row["in_curated_v2"])
    rank = row.get("combined_rank")
    rank = int(rank) if pd.notna(rank) else None
    odds = row.get("odds_ratio")
    pbonf = row.get("p_value_bonferroni")
    odds = float(odds) if pd.notna(odds) else None
    pbonf = float(pbonf) if pd.notna(pbonf) else None

    if in_curated and rank is None:
        return "KEEP_BUT_RARE"

    significant = (odds is not None and odds > min_or) and (
        pbonf is not None and pbonf < max_pbonf
    )

    if in_curated:
        return "KEEP" if significant else "KEEP_BUT_WEAK"

    # Not in curated
    if rank is None:
        return "IGNORE"
    if rank <= top_high:
        return "ADD_HIGH"
    if rank <= top_medium:
        return "ADD_MEDIUM"
    return "IGNORE"


# ════════════════════════════════════════════════════════════════
# CORE COMPARISON
# ════════════════════════════════════════════════════════════════

REVIEW_COLS = [
    "code", "code_type", "term", "feature_type",
    "in_curated_v2", "current_category",
    "combined_rank", "stat_rank", "ml_rank",
    "odds_ratio", "or_ci_lower", "or_ci_upper",
    "p_value", "p_value_bonferroni",
    "n_patient_count_pos", "n_patient_count_neg",
    "prevalence_pos", "prevalence_neg", "prevalence_diff",
    "combined_score", "confidence_tier", "stability_score",
    "action_suggested", "notes",
]


def build_review_table(ranked_df, curated_df, feature_type):
    """Outer-join ranked + curated → per-code review table.

    Codes only in curated (never showed up in ranking) get RANK = NaN
    and action_suggested = 'KEEP_BUT_RARE'.
    """
    curated_subset = curated_df[["code", "current_category", "curated_term"]].copy()
    curated_subset["in_curated_v2"] = True

    merged = ranked_df.merge(curated_subset, on="code", how="outer")
    merged["in_curated_v2"] = merged["in_curated_v2"].fillna(False).astype(bool)
    merged["feature_type"] = merged["feature_type"].fillna(feature_type)

    # Prefer the ranked-pipeline term (more canonical / case-deduped); fall back to curated term.
    if "term" not in merged.columns:
        merged["term"] = pd.NA
    merged["term"] = merged["term"].fillna(merged["curated_term"])

    # Code type from ranked side; fall back to feature_type heuristic.
    if "code_type" not in merged.columns:
        merged["code_type"] = pd.NA
    fallback_code_type = "SNOMED" if feature_type == "observation" else "DMD"
    merged["code_type"] = merged["code_type"].fillna(fallback_code_type)

    # Coerce numerics so downstream sorts/filters behave.
    for c in ["combined_rank", "stat_rank", "ml_rank",
              "n_patient_count_pos", "n_patient_count_neg"]:
        if c in merged.columns:
            merged[c] = _safe_int(merged[c])
        else:
            merged[c] = pd.NA

    for c in ["odds_ratio", "or_ci_lower", "or_ci_upper",
              "p_value", "p_value_bonferroni",
              "prevalence_pos", "prevalence_neg", "prevalence_diff",
              "combined_score", "stability_score"]:
        if c in merged.columns:
            merged[c] = _safe_float(merged[c])
        else:
            merged[c] = np.nan

    if "confidence_tier" not in merged.columns:
        merged["confidence_tier"] = pd.NA

    merged["action_suggested"] = merged.apply(
        lambda r: decide_action(
            r,
            CONFIG["top_high_threshold"],
            CONFIG["top_medium_threshold"],
            CONFIG["min_odds_ratio_for_keep"],
            CONFIG["max_pbonf_for_keep"],
        ),
        axis=1,
    )

    # Notes column — a few clinician-relevant hints surfaced inline.
    notes = []
    for _, r in merged.iterrows():
        bits = []
        if r["in_curated_v2"] and pd.isna(r["combined_rank"]):
            bits.append("not in ranking — possibly zero positives in window")
        if r["in_curated_v2"] and pd.notna(r["combined_rank"]) and r["combined_rank"] > CONFIG["top_medium_threshold"]:
            bits.append(f"curated but rank {int(r['combined_rank'])} > {CONFIG['top_medium_threshold']}")
        if not r["in_curated_v2"] and pd.notna(r["odds_ratio"]) and r["odds_ratio"] > 5:
            bits.append(f"high OR ({r['odds_ratio']:.2f})")
        if pd.notna(r["stability_score"]) and r["stability_score"] >= 0.8:
            bits.append("stable across bootstraps")
        notes.append("; ".join(bits) if bits else "")
    merged["notes"] = notes

    # Final column order. Add any missing (rare) cols as empty.
    for c in REVIEW_COLS:
        if c not in merged.columns:
            merged[c] = pd.NA

    out = merged[REVIEW_COLS].copy()

    # Sort: curated first (so the clinician sees what's already in the codelist),
    # then non-curated sorted by combined_rank ascending.
    out["_sort_in_curated"] = (~out["in_curated_v2"]).astype(int)
    out["_sort_rank"] = out["combined_rank"].fillna(10**9)
    out = (
        out.sort_values(["_sort_in_curated", "_sort_rank"])
        .drop(columns=["_sort_in_curated", "_sort_rank"])
        .reset_index(drop=True)
    )

    return out


def summarise(df, name):
    counts = df["action_suggested"].value_counts().to_dict()
    in_curated = int(df["in_curated_v2"].sum())
    n = len(df)
    logger.info(f"\n📊 {name}: {n:,} unique codes ({in_curated:,} in curated, {n - in_curated:,} new candidates)")
    for action in ["KEEP", "KEEP_BUT_WEAK", "KEEP_BUT_RARE",
                   "ADD_HIGH", "ADD_MEDIUM", "IGNORE"]:
        c = counts.get(action, 0)
        if c:
            logger.info(f"    {action:<18} {c:>6,}")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════


def main():
    cancer = CONFIG["cancer_type"].upper()

    logger.info("=" * 80)
    logger.info(f"PHASE 1.4 — {cancer} — Compare Ranking output vs curated codelist")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    logger.info("\n📂 Loading inputs...")
    obs_curated, med_curated = load_curated(CONFIG["curated_tsv"])
    obs_ranked = load_ranked(
        os.path.join(CONFIG["scores_dir"], CONFIG["obs_all_csv"]), "observation"
    )
    med_ranked = load_ranked(
        os.path.join(CONFIG["scores_dir"], CONFIG["meds_all_csv"]), "medication"
    )

    logger.info("\n🔀 Building review tables...")
    obs_review = build_review_table(obs_ranked, obs_curated, "observation")
    med_review = build_review_table(med_ranked, med_curated, "medication")
    combined_review = pd.concat([obs_review, med_review], ignore_index=True)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    obs_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['cancer_type']}_obs_review.tsv")
    med_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['cancer_type']}_meds_review.tsv")
    cmb_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['cancer_type']}_combined_review.tsv")
    obs_review.to_csv(obs_path, sep="\t", index=False)
    med_review.to_csv(med_path, sep="\t", index=False)
    combined_review.to_csv(cmb_path, sep="\t", index=False)

    logger.info(f"  💾 {obs_path} ({len(obs_review):,} rows)")
    logger.info(f"  💾 {med_path} ({len(med_review):,} rows)")
    logger.info(f"  💾 {cmb_path} ({len(combined_review):,} rows)")

    summarise(obs_review, "Observations")
    summarise(med_review, "Medications")
    summarise(combined_review, "Combined")

    # Headline "what changed" view for the clinician.
    logger.info("\n" + "=" * 80)
    logger.info("HEADLINES FOR CLINICAL REVIEW")
    logger.info("=" * 80)

    add_high = combined_review[combined_review["action_suggested"] == "ADD_HIGH"]
    keep_weak = combined_review[combined_review["action_suggested"] == "KEEP_BUT_WEAK"]
    keep_rare = combined_review[combined_review["action_suggested"] == "KEEP_BUT_RARE"]

    logger.info(f"\n🆕 ADD_HIGH candidates (not in curated, top 150): {len(add_high)}")
    for _, r in add_high.head(20).iterrows():
        or_str = f"{r['odds_ratio']:.2f}" if pd.notna(r["odds_ratio"]) else "—"
        logger.info(
            f"   rank #{int(r['combined_rank']):<4} OR={or_str:<6} "
            f"[{r['code_type']}] {r['code']} — {r['term']}"
        )

    logger.info(f"\n⚠️  KEEP_BUT_WEAK (in curated, low signal): {len(keep_weak)}")
    for _, r in keep_weak.head(20).iterrows():
        or_str = f"{r['odds_ratio']:.2f}" if pd.notna(r["odds_ratio"]) else "—"
        rank_str = str(int(r["combined_rank"])) if pd.notna(r["combined_rank"]) else "—"
        logger.info(
            f"   [{r['current_category']}] {r['code']} ({r['term']}) "
            f"— rank #{rank_str}, OR={or_str}"
        )

    logger.info(f"\n👻 KEEP_BUT_RARE (in curated, never appeared in ranking): {len(keep_rare)}")
    for _, r in keep_rare.head(20).iterrows():
        logger.info(
            f"   [{r['current_category']}] {r['code']} — {r['term']}"
        )

    logger.info("\n" + "=" * 80)
    logger.info("✅ Phase 1.4 complete — ready for clinical review")
    logger.info(f"   Open: {cmb_path}")
    logger.info("=" * 80)

    return combined_review


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ FATAL: {e}", exc_info=True)
        sys.exit(1)
