"""v3: adds --cancer-class-weight, --sensitivity-weight overrides for k4."""
import argparse, csv, json, re, subprocess, sys
from pathlib import Path

LSTM_DIR = Path('/home/keerthikovelamudi_cthesigns_com/lstm_work')
TEMPLATE_DEFAULT = LSTM_DIR / "lung_lstm_v147_bq_v45.ipynb"
DEFAULT_SEED = 42


def _read_codelist(path):
    ids = []
    with open(path) as f:
        for row in csv.DictReader(f):
            sid = (row.get('id') or '').strip()
            if sid and sid.isdigit():
                ids.append(sid)
    return ids


def patch_config(nb, cancer, seed, split_seed, variant, overrides, suffix, codelist_ids, relevance_codelist_path=None):
    target = re.compile(r"TARGET_CANCER\s*=\s*'[^']+'")
    seed_re = re.compile(r"config\['SEED'\]\s*=\s*\d+")
    split_re = re.compile(r"config\['SPLIT_SEED'\]\s*=\s*\d+")
    save_re = re.compile(r"config\['saved_model_path'\]\s*=\s*f?'[^']+'")
    tuner_re = re.compile(r"config\['tuner_project_name'\]\s*=\s*f?'[^']+'")

    cv_token = f'_{variant}' if variant else ''
    artifact_tag = f"{cv_token}_{suffix}" if suffix else cv_token

    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        if not target.search(src): continue
        new_src = target.sub(f"TARGET_CANCER = '{cancer}'", src)
        new_src = seed_re.sub(f"config['SEED'] = {seed}", new_src)
        new_src = split_re.sub(f"config['SPLIT_SEED'] = {split_seed}", new_src)
        new_src = save_re.sub(
            f"config['saved_model_path'] = "
            f"'saved_models/{cancer}_lstm_v147_v45{artifact_tag}.keras'", new_src)
        new_src = tuner_re.sub(
            f"config['tuner_project_name'] = "
            f"'{cancer}_lstm_v147_v45{artifact_tag}_tuning'", new_src)

        for key, val in overrides.items():
            override_re = re.compile(rf"config\['{re.escape(key)}'\]\s*=\s*[^\n]+")
            replacement = f"config['{key}'] = {val!r}"
            if override_re.search(new_src):
                new_src = override_re.sub(replacement, new_src)
            else:
                new_src = new_src.rstrip() + f"\n{replacement}\n"

        if codelist_ids:
            inject = f"""

# ─── k2+ cancer-specific filter ─────────────────────────────────────
config['curated_codelist'] = {repr(codelist_ids)}
import pandas as _pd_k2
_orig_pre_process_k2 = pre_process
def pre_process(df, cfg):
    if cfg.get('curated_codelist'):
        df = df.copy()
        df.columns = df.columns.str.upper()
        allowed = set(str(x) for x in cfg['curated_codelist'])
        before = len(df)
        sid_mask = _pd_k2.Series(False, index=df.index)
        if 'SNOMED_C_T_CONCEPT_ID' in df.columns:
            sid_mask = df['SNOMED_C_T_CONCEPT_ID'].astype(str).isin(allowed)
        mid_mask = _pd_k2.Series(False, index=df.index)
        if 'MED_CODE_ID' in df.columns:
            mid_mask = df['MED_CODE_ID'].astype(str).isin(allowed)
        df = df[sid_mask | mid_mask].reset_index(drop=True)
        print(f"[k2 curated_codelist] kept {{len(df):,}} of {{before:,}} rows ({{len(df)/max(before,1):.1%}})")
    return _orig_pre_process_k2(df, cfg)
"""
            new_src = new_src.rstrip() + inject

        if relevance_codelist_path:
            relevance_inject = ("\n\n# --- k9 add-relevance-codelist: extend concept_codelist_paths ---\nconfig['concept_codelist_paths'] = list(config.get('concept_codelist_paths', [])) + [PATH_PLACEHOLDER]\nprint('[k9 relevance] added', PATH_PLACEHOLDER, 'to concept_codelist_paths')\n").replace('PATH_PLACEHOLDER', repr(relevance_codelist_path))
            new_src = new_src.rstrip() + relevance_inject

        if new_src != src:
            cell['source'] = new_src.splitlines(keepends=True)
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cancer', required=True,
                    choices=['lung', 'prostate', 'breast', 'colorectal',
                              'lymphoma', 'melanoma', 'bladder', 'pancreatic',
                              'ovarian', 'leukaemia'])
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--split-seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--variant', default='')
    ap.add_argument('--exclude-shortcuts', action='store_true')
    ap.add_argument('--use-med-branch', action='store_true')
    ap.add_argument('--max-snomed', type=int, default=None)
    ap.add_argument('--cancer-class-weight', type=float, default=None)
    ap.add_argument('--sensitivity-weight', type=float, default=None)
    ap.add_argument('--min-code-frequency', type=int, default=None)
    ap.add_argument('--min-snomed-seq', type=int, default=None,
                    help="override config['MIN_SNOMED_SEQ_LENGTH']")
    ap.add_argument('--min-med-seq', type=int, default=None,
                    help="override config['MIN_MED_SEQ_LENGTH']")
    ap.add_argument('--filter-codelist', default=None)
    ap.add_argument('--source-notebook', default=None, help='path to forked source notebook')
    ap.add_argument('--add-relevance-codelist', default=None, help='extra concept codelist path for relevance')
    ap.add_argument('--threshold-mode', choices=['sens_floor', 'sens_spec_balance'], default=None,
                    help='sens_floor (default): max spec st sens>=floor. sens_spec_balance: weighted (sw*sens+spw*spec). pair with --sensitivity-weight for weight')
    args = ap.parse_args()

    overrides = {}
    if args.exclude_shortcuts: overrides['exclude_shortcut_codes'] = True
    if args.use_med_branch: overrides['use_med_branch'] = True
    if args.max_snomed is not None: overrides['MAX_SNOMED_SEQ_LENGTH'] = args.max_snomed
    if args.cancer_class_weight is not None: overrides['cancer_class_weight'] = args.cancer_class_weight
    if args.sensitivity_weight is not None: overrides['sensitivity_weight'] = args.sensitivity_weight
    if args.min_code_frequency is not None: overrides['min_code_frequency'] = args.min_code_frequency
    if args.min_snomed_seq is not None: overrides['MIN_SNOMED_SEQ_LENGTH'] = args.min_snomed_seq
    if args.min_med_seq is not None: overrides['MIN_MED_SEQ_LENGTH'] = args.min_med_seq
    if args.threshold_mode == 'sens_spec_balance':
        overrides['use_sensitivity_floor'] = False
    elif args.threshold_mode == 'sens_floor':
        overrides['use_sensitivity_floor'] = True

    codelist_ids = []
    if args.filter_codelist:
        codelist_ids = _read_codelist(args.filter_codelist)
        print(f'Loaded {len(codelist_ids)} curated codes from {args.filter_codelist}')

    is_default_seed = (args.seed == DEFAULT_SEED and args.split_seed == DEFAULT_SEED)
    if is_default_seed: seed_suffix = ''
    elif args.seed == args.split_seed: seed_suffix = f'seed{args.seed}'
    else: seed_suffix = f'seed{args.seed}_split{args.split_seed}'

    if args.variant:
        nb_basename = f'{args.cancer}_lstm_v147_{args.variant}_bq_v45'
    else:
        nb_basename = f'{args.cancer}_lstm_v147_bq_v45'
    if seed_suffix: nb_basename = f'{nb_basename}_{seed_suffix}'
    nb_basename += '.ipynb'

    out_nb = LSTM_DIR / nb_basename
    tmp_nb = LSTM_DIR / f'.tmp_{nb_basename}'

    nb = json.load(open(args.source_notebook or TEMPLATE_DEFAULT))
    if not patch_config(nb, args.cancer, args.seed, args.split_seed,
                         args.variant, overrides, seed_suffix, codelist_ids,
                         relevance_codelist_path=args.add_relevance_codelist):
        print('ERROR: failed to patch config cell', file=sys.stderr); sys.exit(1)
    json.dump(nb, open(tmp_nb, 'w'))
    print(f'Patched: {args.cancer} v={args.variant!r} seed={args.seed} '
          f'overrides={overrides} codelist={len(codelist_ids)} codes')

    cmd = ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
           str(tmp_nb), '--output', str(out_nb), '--ExecutePreprocessor.timeout=3600']
    try: subprocess.run(cmd, check=True)
    finally:
        try: tmp_nb.unlink()
        except FileNotFoundError: pass
    print(f'DONE: {out_nb}')


if __name__ == '__main__':
    main()
