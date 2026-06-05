"""Extract complete metrics table from LSTM variant notebooks.

Produces columns:
  Cancer Type | Model | Type | Sens | Spec | Patients |
  Positive Cohort | Negative Cohort | train/val/test |
  Accuracy | F1 | ROC-AUC | TP/TN/FP/FN |
  Faithfulness | Clinical Relevance | Consistency | Explainability Rating
"""
import json, re, os, sys, csv
from pathlib import Path

LSTM_DIR = Path('/home/keerthikovelamudi_cthesigns_com/lstm_work')


def get_text(nb_path):
    with open(nb_path) as f:
        nb = json.load(f)
    text = ''
    for c in nb['cells']:
        if c['cell_type'] != 'code': continue
        for o in c.get('outputs', []):
            if 'text' in o:
                t = o['text'] if isinstance(o['text'], str) else ''.join(o['text'])
                text += t
    return text


METRIC_PAT = re.compile(
    r'Accuracy:\s*([\d.]+)%\s*\n'
    r'Precision:\s*([\d.]+)%\s*\n'
    r'Recall:\s*([\d.]+)%\s*\n'
    r'F1-Score:\s*([\d.]+)%\s*\n'
    r'ROC-AUC:\s*([\d.]+)%\s*\n'
    r'PPV:\s*([\d.]+)%\s*\n'
    r'NPV:\s*([\d.]+)%\s*\n'
    r'Sensitivity:\s*([\d.]+)%\s*\n'
    r'Specificity:\s*([\d.]+)%'
)


def extract(nb_path, cancer, variant, seed):
    if not nb_path.exists():
        return None
    t = get_text(nb_path)
    m = METRIC_PAT.findall(t)
    if not m:
        return None
    acc, prec, rec, f1, auc, ppv, npv, sens, spec = (float(x) for x in m[0])

    # Patient counts
    splits = re.findall(r'(Train|Val|Test):\s*(\d+)\s+cancer\s*/\s*(\d+)\s+no-cancer', t)
    train = next((s for s in splits if s[0] == 'Train'), None)
    val = next((s for s in splits if s[0] == 'Val'), None)
    test = next((s for s in splits if s[0] == 'Test'), None)
    if train and val and test:
        tr_pos, tr_neg = int(train[1]), int(train[2])
        v_pos, v_neg = int(val[1]), int(val[2])
        te_pos, te_neg = int(test[1]), int(test[2])
        total_pos = tr_pos + v_pos + te_pos
        total_neg = tr_neg + v_neg + te_neg
        total = total_pos + total_neg
        split_str = f'{tr_pos+tr_neg}/{v_pos+v_neg}/{te_pos+te_neg}'
    else:
        total_pos = total_neg = total = 0
        split_str = '?/?/?'

    # Confusion matrix from test set evaluation — derive from sens/spec + counts
    n_test_pos = te_pos if test else 0
    n_test_neg = te_neg if test else 0
    tp = round(n_test_pos * sens / 100)
    fn = n_test_pos - tp
    tn = round(n_test_neg * spec / 100)
    fp = n_test_neg - tn

    # Rating components
    faith = re.findall(r'Faithfulness.*?score:\s*([\d.]+)', t)
    relev = re.findall(r'Clinical relevance.*?score:\s*([\d.]+)', t)
    cons = re.findall(r'consistency.*?score:\s*([\d.]+)', t)
    rating = re.findall(r'Explainability Rating:\s*([\d.]+)\s*/\s*100', t)

    # Type heuristic: med branch on/off + (sql version)
    src_text = ''
    for c in json.load(open(nb_path))['cells']:
        if c['cell_type'] == 'code':
            src_text += ''.join(c['source'])
    use_med = "config['use_med_branch'] = True" in src_text
    data_v = re.findall(r"config\['data_version'\]\s*=\s*'([^']+)'", src_text)
    type_str = 'LSTM-dual' if use_med else 'LSTM-snomed'
    if data_v and data_v[0] != 'v45':
        type_str += f"-{data_v[0]}"

    return {
        'Cancer Type': cancer,
        'Model': variant + (f'_seed{seed}' if seed != 42 else ''),
        'Type': type_str,
        'Sensitivity': f'{sens:.2f}%',
        'Specificity': f'{spec:.2f}%',
        'Patients': total,
        'Positive Cohort': total_pos,
        'Negative Cohort': total_neg,
        'train/val/test': split_str,
        'Accuracy': f'{acc:.2f}%',
        'F1': f'{f1:.2f}%',
        'ROC-AUC': f'{auc:.2f}%',
        'TP/TN/FP/FN': f'{tp}/{tn}/{fp}/{fn}',
        'Faithfulness': faith[-1] if faith else '-',
        'Clinical Relevance': relev[-1] if relev else '-',
        'Consistency': cons[-1] if cons else '-',
        'Explainability Rating': rating[-1] if rating else '-',
    }


if __name__ == '__main__':
    rows = []
    # All variants: base + k1-k14 + seed variants
    variants = ['', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k14_redo', 'k14b', 'k15', 'k16', 'k18']
    seeds_per_variant = {
        # canonical seed 42 for all; variance seeds only for k9, k11, k14
        'k9': [42, 13, 99, 7, 2024],
        'k11': [42, 13, 99, 7, 2024],
        'k14': [42, 13, 99, 7, 2024],
    }
    for cancer in ['lung', 'prostate']:
        for v in variants:
            seeds = seeds_per_variant.get(v, [42])
            for seed in seeds:
                suf = '' if seed == 42 else f'_seed{seed}'
                v_part = f'_{v}' if v else ''
                nb = LSTM_DIR / f'{cancer}_lstm_v147{v_part}_bq_v45{suf}.ipynb'
                row = extract(nb, cancer, v or 'base', seed)
                if row:
                    rows.append(row)

    if not rows:
        print('No results found')
        sys.exit(1)

    cols = list(rows[0].keys())
    out = LSTM_DIR / 'all_metrics.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {out} ({len(rows)} rows × {len(cols)} cols)')

    # Compact print
    for r in rows:
        print(f"{r['Cancer Type']:<9} {r['Model']:<15} {r['Type']:<20} sens={r['Sensitivity']:>7} spec={r['Specificity']:>7} auc={r['ROC-AUC']:>7} rating={r['Explainability Rating']:>5} kept={r['Patients']:>6}")
