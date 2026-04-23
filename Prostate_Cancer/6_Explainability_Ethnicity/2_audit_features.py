# ═══════════════════════════════════════════════════════════════
# AUDIT FEATURES — Find important-but-opaque features
# Reads SHAP importance, cross-references with feature_dictionary,
# and flags features that need attention.
#
# Usage:
#   python 2_audit_features.py --window 12mo
# ═══════════════════════════════════════════════════════════════

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'results'

from feature_dictionary import (
    get_explainability_class,
    get_clinical_description,
    classify_all_features,
    CLINICAL_DESCRIPTIONS,
)


def run_audit(window):
    """Audit features for explainability gaps."""

    logger.info(f"\n{'='*60}")
    logger.info(f"  FEATURE EXPLAINABILITY AUDIT - {window.upper()}")
    logger.info(f"{'='*60}")

    # Load global importance from step 1
    imp_path = RESULTS_DIR / window / f'global_feature_importance_{window}.csv'
    if not imp_path.exists():
        logger.error(f"  Run 1_explain_predictions.py first to generate {imp_path.name}")
        return

    df = pd.read_csv(imp_path)

    total = len(df)
    direct = df[df['explainability'] == 'direct']
    indirect = df[df['explainability'] == 'indirect']
    opaque = df[df['explainability'] == 'opaque']

    logger.info(f"\n  Feature Breakdown:")
    logger.info(f"    Total features:     {total}")
    logger.info(f"    Direct (D):         {len(direct):4d} ({len(direct)/total*100:.0f}%) — directly explainable")
    logger.info(f"    Indirect (I):       {len(indirect):4d} ({len(indirect)/total*100:.0f}%) — needs a sentence but OK")
    logger.info(f"    Opaque (X):         {len(opaque):4d} ({len(opaque)/total*100:.0f}%) — cannot explain to clinician")

    # Cumulative SHAP importance by class
    total_shap = df['mean_abs_shap'].sum()
    direct_shap = direct['mean_abs_shap'].sum()
    indirect_shap = indirect['mean_abs_shap'].sum()
    opaque_shap = opaque['mean_abs_shap'].sum()

    logger.info(f"\n  SHAP Importance Share:")
    logger.info(f"    Direct:     {direct_shap/total_shap*100:.1f}% of total model signal")
    logger.info(f"    Indirect:   {indirect_shap/total_shap*100:.1f}%")
    logger.info(f"    Opaque:     {opaque_shap/total_shap*100:.1f}%")

    # Key metric: what % of top 20 features are opaque?
    top20 = df.head(20)
    top20_opaque = top20[top20['explainability'] == 'opaque']
    top10 = df.head(10)
    top10_opaque = top10[top10['explainability'] == 'opaque']

    logger.info(f"\n  Opaque Features in Top Rankings:")
    logger.info(f"    Top 10: {len(top10_opaque)}/10 opaque")
    logger.info(f"    Top 20: {len(top20_opaque)}/20 opaque")

    # List the problematic opaque features (high importance)
    opaque_important = opaque[opaque['mean_abs_shap'] > opaque['mean_abs_shap'].quantile(0.5)]
    opaque_important = opaque_important.sort_values('mean_abs_shap', ascending=False)

    if len(opaque_important) > 0:
        logger.info(f"\n  OPAQUE FEATURES TO REVIEW ({len(opaque_important)}):")
        logger.info(f"  {'─'*65}")
        logger.info(f"  {'Rank':>4} {'Feature':45s} {'SHAP':>10}")
        logger.info(f"  {'─'*65}")
        for _, row in opaque_important.iterrows():
            rank = df[df['feature'] == row['feature']].index[0] + 1
            logger.info(f"  {rank:4d} {row['feature']:45s} {row['mean_abs_shap']:.6f}")

        logger.info(f"\n  ACTION ITEMS:")
        logger.info(f"  For each opaque feature above, choose one:")
        logger.info(f"    1. REMOVE — if it's statistical noise (embeddings, entropy, gini)")
        logger.info(f"    2. REPLACE — if the signal is real but the feature is opaque")
        logger.info(f"       (e.g., replace EMB_text_dim_3 with the keyword it correlates with)")
        logger.info(f"    3. RECLASSIFY — if you can write a clinical description for it")
        logger.info(f"       (update feature_dictionary.py, change 'opaque' → 'indirect')")
    else:
        logger.info(f"\n  No high-importance opaque features found.")

    # Check for features missing from the dictionary
    missing_desc = []
    for _, row in df.iterrows():
        desc = get_clinical_description(row['feature'])
        if desc == row['feature'] or desc.startswith('['):
            missing_desc.append(row['feature'])

    if missing_desc:
        logger.info(f"\n  FEATURES WITHOUT CLINICAL DESCRIPTION ({len(missing_desc)}):")
        for f in missing_desc[:20]:
            cls = get_explainability_class(f)
            if cls != 'opaque':  # only flag non-opaque features missing descriptions
                logger.info(f"    [{cls[0].upper()}] {f}")
        if len(missing_desc) > 20:
            logger.info(f"    ... and {len(missing_desc)-20} more")

    # Generate removal recommendation
    opaque_features = opaque['feature'].tolist()

    rec_path = RESULTS_DIR / window / f'audit_recommendations_{window}.csv'
    rec_df = pd.DataFrame({
        'feature': df['feature'],
        'mean_abs_shap': df['mean_abs_shap'],
        'rank': range(1, len(df) + 1),
        'explainability': df['explainability'],
        'recommendation': df.apply(lambda r: 'KEEP' if r['explainability'] in ('direct', 'indirect')
                                    else 'REMOVE' if r['explainability'] == 'opaque'
                                    else 'REVIEW', axis=1),
    })
    rec_df.to_csv(rec_path, index=False)
    logger.info(f"\n  Saved audit: {rec_path}")

    # Summary score
    explainability_score = (len(direct) + len(indirect)) / total * 100
    top10_score = (10 - len(top10_opaque)) / 10 * 100

    logger.info(f"\n  EXPLAINABILITY SCORES:")
    logger.info(f"    Overall:  {explainability_score:.0f}% of features are explainable")
    logger.info(f"    Top 10:   {top10_score:.0f}% of top 10 features are explainable")

    if top10_score < 80:
        logger.warning(f"\n  Top-10 score below 80% — run 3_enhancement_loop.py to improve")
    else:
        logger.info(f"\n  Explainability is acceptable for production")

    return opaque_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default='12mo', choices=['3mo', '6mo', '12mo'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_audit(args.window)
