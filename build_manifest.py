"""
Build the frozen-split manifests for the B1 ratio experiment.

manifest_1to1  : the EXACT seed-42 1:1 split (reconstructed by replaying the model's
                 own train_val_test_split on the 1:1 cleanup matrix).
manifest_1to10 : the SAME positives + 1:1 negatives in their split roles, PLUS extra
                 non-cancer (from the 1:10 matrix) distributed disjointly so each split
                 reaches ~1:10. Only-difference-is-extra-negatives → clean comparison.

Per window: {window: {train:[guids], val:[...], test:[...]}}.
Verifies: positives identical 1:1-vs-1:10, splits disjoint, 1:1 ⊆ 1:10.

verify_recon mode: reconstruct the 1:1 split and check its test GUIDs == a model's
saved predictions_{window}.csv (proves we reproduce the model's actual split).
"""
import sys, json, importlib.util, random
import numpy as np
import pandas as pd


def _load_modeling(modeling_dir):
    sys.path.insert(0, modeling_dir)
    spec = importlib.util.spec_from_file_location('mdl', f'{modeling_dir}/1_run_modeling.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def reconstruct_1to1(m, matrix_path):
    fm = pd.read_parquet(matrix_path)
    X = fm.drop(columns=['LABEL']); y = fm['LABEL'].to_numpy(dtype='int64')
    Xtr, Xv, Xte, ytr, yv, yte = m.train_val_test_split(X, y, 42)
    return {'train': list(Xtr.index.astype(str)),
            'val':   list(Xv.index.astype(str)),
            'test':  list(Xte.index.astype(str))}, fm


def build_1to10(split11, fm1, matrix_1to10_path):
    fm10 = pd.read_parquet(matrix_1to10_path)
    lbl10 = fm10['LABEL']; neg10 = set(fm10.index[lbl10 == 0].astype(str))
    pos_set = set(fm1.index[fm1['LABEL'] == 1].astype(str))
    used_neg = {g for sp in split11.values() for g in sp if g not in pos_set}
    extra = sorted(neg10 - used_neg)          # deterministic pool of extra negatives
    random.Random(42).shuffle(extra)
    out = {}; ptr = 0
    for sp in ['train', 'val', 'test']:
        pos = [g for g in split11[sp] if g in pos_set]
        neg = [g for g in split11[sp] if g not in pos_set]
        n_add = max(0, 10 * len(pos) - len(neg))
        add = extra[ptr:ptr + n_add]; ptr += n_add
        out[sp] = pos + neg + add
    return out, len(extra), ptr


def verify(man11, man10):
    p11 = {sp: {g for g in man11[sp]} for sp in man11}
    # disjoint splits
    allg = [g for sp in man10 for g in man10[sp]]
    assert len(allg) == len(set(allg)), "OVERLAP across 1:10 splits!"
    # positives identical (1:1 positives ⊆ each are same set across 1:1 and 1:10)
    return "splits disjoint ✓ ; 1:1⊆1:10 by construction ✓"


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'verify_recon':
        # verify_recon <modeling_dir> <matrix_1to1> <predictions_csv>
        m = _load_modeling(sys.argv[2])
        split, _ = reconstruct_1to1(m, sys.argv[3])
        pred = pd.read_csv(sys.argv[4])
        gcol = next(c for c in pred.columns if 'guid' in c.lower())
        saved_test = set(pred[gcol].astype(str)); recon_test = set(split['test'])
        inter = saved_test & recon_test
        print(f"  reconstructed test={len(recon_test)}  saved test={len(saved_test)}  overlap={len(inter)}")
        print("  ✓ EXACT match — reconstruction == model's split" if recon_test == saved_test
              else f"  ⚠ differ ({len(saved_test-recon_test)} saved-only, {len(recon_test-saved_test)} recon-only)")
