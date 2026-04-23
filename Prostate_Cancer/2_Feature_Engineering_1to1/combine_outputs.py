# ═══════════════════════════════════════════════════════════════
# COMBINE OUTPUTS — Reference implementation for production handoff
#
# Joins the 4 FE outputs into a single patient x feature matrix.
# Supports an optional --features file for selective-mode computation.
#
# Use this as the reference for the production BigQuery transformer:
#   1. Replace the pd.read_csv(...) calls with BQ reads.
#   2. Replace the final .to_csv(...) with a BQ write.
#   3. Accept the same --features arg to support top-N selective mode.
#
# Note: this script currently builds ALL features first (via the normal
# 0_run_pipeline.py), then filters at the end. For true selective mode
# in prod (skip computing features that aren't needed), the underlying
# builders in 3_pipeline.py / 4_cancer_features.py would need to accept
# a feature-name whitelist. That's a second iteration; this file gives
# Alex the join + filter contract to work against.
# ═══════════════════════════════════════════════════════════════

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import config

logger = logging.getLogger(__name__)


def combine_window(window, features_whitelist=None):
    """Load cleanup + text + tfidf + bert for one window, join, optionally filter.

    Args:
        window: '3mo' | '6mo' | '12mo'
        features_whitelist: list[str] of column names to keep (LABEL always retained).
                            None = keep everything.

    Returns:
        pd.DataFrame indexed by PATIENT_GUID, first column LABEL.
    """
    cleanup = config.CLEANUP_RESULTS / window / f'feature_matrix_clean_{window}.csv'
    if not cleanup.exists():
        raise FileNotFoundError(f"Missing cleanup output: {cleanup}. Run 0_run_pipeline.py first.")

    fm = pd.read_csv(cleanup, index_col=0)
    logger.info(f"[{window}] cleanup: {fm.shape[0]} x {fm.shape[1]}")

    for name, fpath in [
        ('text',  config.TEXT_RESULTS / window / f'text_features_{window}.csv'),
        ('tfidf', config.EMB_RESULTS / window / f'text_embeddings_{window}.csv'),
        ('bert',  config.BERT_RESULTS / window / f'bert_embeddings_{window}.csv'),
    ]:
        if fpath.exists():
            tdf = pd.read_csv(fpath, index_col=0)
            fm = fm.join(tdf, how='left')
            logger.info(f"[{window}] + {name}: {tdf.shape[1]} cols -> total {fm.shape[1]}")
        else:
            logger.warning(f"[{window}] missing {name} file: {fpath}")

    fm = fm.fillna(0).loc[:, ~fm.columns.duplicated()]

    if features_whitelist:
        keep = ['LABEL'] + [c for c in features_whitelist if c in fm.columns and c != 'LABEL']
        missing = [c for c in features_whitelist if c not in fm.columns]
        fm = fm[keep]
        logger.info(f"[{window}] filtered to {len(keep)-1} features ({len(missing)} missing)")
        if missing:
            logger.warning(f"[{window}] missing features (will not be in output): {missing[:10]}...")

    return fm


def load_features_whitelist(path):
    """Accept either JSON list, JSON {'features': [...]}, or newline-delimited text."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Features file not found: {p}")
    content = p.read_text().strip()
    if p.suffix.lower() == '.json':
        data = json.loads(content)
        return data['features'] if isinstance(data, dict) else data
    return [line.strip() for line in content.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description='Combine FE outputs into a single feature matrix per window. '
                    'Optional --features for selective mode.')
    parser.add_argument('--features', type=str, default=None,
                        help='Path to JSON or .txt file listing feature names to keep. '
                             'Omit to keep all features.')
    parser.add_argument('--windows', type=str, default=None,
                        help='Comma-separated windows to process (default: all from config).')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/combined/).')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    whitelist = load_features_whitelist(args.features) if args.features else None
    if whitelist:
        logger.info(f"Selective mode: {len(whitelist)} requested features")

    windows = args.windows.split(',') if args.windows else config.WINDOWS
    out_dir = Path(args.output_dir) if args.output_dir else (SCRIPT_DIR / 'results' / 'combined')
    out_dir.mkdir(parents=True, exist_ok=True)

    for w in windows:
        try:
            fm = combine_window(w, features_whitelist=whitelist)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue
        out_path = out_dir / f'features_{w}.csv'
        fm.to_csv(out_path)
        logger.info(f"[{w}] wrote {out_path} ({fm.shape[0]} x {fm.shape[1]})")


if __name__ == '__main__':
    main()
