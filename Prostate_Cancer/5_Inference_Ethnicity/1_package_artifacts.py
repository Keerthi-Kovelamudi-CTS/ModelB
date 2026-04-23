# ═══════════════════════════════════════════════════════════════
# PACKAGE INFERENCE ARTIFACTS
# Copies ALL required artifacts + pipeline code into a single
# self-contained 5_Inference/ folder. Run ONCE after training.
# After this, 5_Inference/ can be deployed independently —
# no dependency on 2_Feature_Engineering/ or 3_Modeling/.
#
# Usage:
#   python 1_package_artifacts.py                   # default: 12mo
#   python 1_package_artifacts.py --window 6mo
#   python 1_package_artifacts.py --all-windows
# ═══════════════════════════════════════════════════════════════

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FE_DIR = PROJECT_DIR / '2_Feature_Engineering_Ethnicity'
MODELING_DIR = PROJECT_DIR / '3_Modeling_Ethnicity'
HOLDOUT_DIR = PROJECT_DIR / '4_ExpandedData_Test'
ARTIFACTS_DIR = SCRIPT_DIR / 'artifacts'
PIPELINE_DIR = SCRIPT_DIR / 'pipeline'


def package_pipeline_code():
    """Copy FE pipeline code into 5_Inference/pipeline/ for self-contained deployment."""
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        'config.py',
        '3_pipeline.py',
        '4_cancer_features.py',
        '6_text_features.py',
    ]

    copied = []
    for f in files_to_copy:
        src = FE_DIR / f
        dst = PIPELINE_DIR / f
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(f)
        else:
            logger.warning(f"  Missing: {src}")

    # Create __init__.py so it's importable as a package
    init_path = PIPELINE_DIR / '__init__.py'
    init_path.write_text('# Bundled FE pipeline for inference\n')
    copied.append('__init__.py')

    logger.info(f"  Pipeline code: {len(copied)} files copied to pipeline/")
    for f in copied:
        logger.info(f"    + pipeline/{f}")


def package_window(window):
    """Package all inference artifacts for one window."""
    out_dir = ARTIFACTS_DIR / window
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'models').mkdir(exist_ok=True)
    (out_dir / 'transformers').mkdir(exist_ok=True)

    copied = []
    missing = []

    # 1. Saved models
    model_dir = MODELING_DIR / 'results' / '1_training' / window / 'saved_models'
    for f in ['xgboost_model.pkl', 'lightgbm_model.pkl', 'catboost_model.pkl', 'config.json']:
        src = model_dir / f
        dst = out_dir / 'models' / f
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(f"models/{f}")
        else:
            missing.append(f"models/{f}")

    # 2. Text transformers
    transformer_dir = FE_DIR / 'results' / '6_text_features' / 'fitted_transformers' / window
    for f in ['tfidf_vectorizer.pkl', 'svd_transformer.pkl', 'bert_pca.pkl']:
        src = transformer_dir / f
        dst = out_dir / 'transformers' / f
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(f"transformers/{f}")
        else:
            missing.append(f"transformers/{f}")

    # 3. Code category mapping
    for candidate in [HOLDOUT_DIR / 'code_category_mapping.json',
                      FE_DIR / 'code_category_mapping.json']:
        if candidate.exists():
            shutil.copy2(candidate, out_dir / 'code_category_mapping.json')
            copied.append('code_category_mapping.json')
            break
    else:
        missing.append('code_category_mapping.json')

    # 4. Selected features list
    feat_src = MODELING_DIR / 'results' / '1_training' / window / f'selected_features_{window}.csv'
    if feat_src.exists():
        shutil.copy2(feat_src, out_dir / 'selected_features.csv')
        copied.append('selected_features.csv')

    # Summary
    logger.info(f"\n  {window} artifacts:")
    logger.info(f"    Copied: {len(copied)} files")
    for f in copied:
        logger.info(f"      + {f}")
    if missing:
        logger.warning(f"    Missing: {len(missing)} files (run training first)")
        for f in missing:
            logger.warning(f"      - {f}")

    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(description='Package inference artifacts')
    parser.add_argument('--window', default='12mo', choices=['3mo', '6mo', '12mo'])
    parser.add_argument('--all-windows', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    logger.info(f"{'='*60}")
    logger.info(f"  PACKAGE INFERENCE ARTIFACTS")
    logger.info(f"{'='*60}")

    # Package pipeline code
    logger.info(f"\n  Bundling pipeline code...")
    package_pipeline_code()

    # Package artifacts per window
    windows = ['3mo', '6mo', '12mo'] if args.all_windows else [args.window]
    for w in windows:
        package_window(w)

    logger.info(f"\n{'='*60}")
    logger.info(f"  DONE")
    logger.info(f"  5_Inference/ is now self-contained and ready for deployment.")
    logger.info(f"  Test: python 2_predict_single_patient.py --input sample_input/sample_patient.json")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
