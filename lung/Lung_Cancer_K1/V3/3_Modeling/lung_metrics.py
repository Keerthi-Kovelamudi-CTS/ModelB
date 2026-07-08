"""
Evaluation metrics with 95% bootstrap confidence intervals for the Lung model.

Reports the full clinical suite the project needs:
  AUROC, AUPRC (threshold-free) + Sensitivity, Specificity, PPV, NPV at a chosen
  operating threshold — each with a percentile bootstrap 95% CI.

Used for both the internal test split and the real-world held-out (500/50k) set.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def _point_metrics(y_true, y_score, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    out = {}
    # threshold-free
    out['auroc'] = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    out['auprc'] = average_precision_score(y_true, y_score) if y_true.sum() > 0 else np.nan
    # at threshold
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out['sensitivity'] = tp / (tp + fn) if (tp + fn) else np.nan   # recall
    out['specificity'] = tn / (tn + fp) if (tn + fp) else np.nan
    out['ppv'] = tp / (tp + fp) if (tp + fp) else np.nan           # precision
    out['npv'] = tn / (tn + fn) if (tn + fn) else np.nan
    out['prevalence'] = (tp + fn) / len(y_true)
    return out


def threshold_at_sensitivity(y_true, y_score, target_sens=0.90):
    """Lowest threshold achieving >= target sensitivity (clinical operating point)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos = np.sort(y_score[y_true == 1])
    if len(pos) == 0:
        return 0.5
    # to capture target_sens fraction of positives, threshold = the (1-target_sens) quantile of positive scores
    idx = int(np.floor((1 - target_sens) * len(pos)))
    idx = min(max(idx, 0), len(pos) - 1)
    return float(pos[idx])


def eval_with_ci(y_true, y_score, threshold=0.5, n_boot=1000, seed=42, label=""):
    """Point estimates + 95% percentile-bootstrap CIs for the full metric suite."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    point = _point_metrics(y_true, y_score, threshold)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ['auroc', 'auprc', 'sensitivity', 'specificity', 'ppv', 'npv']
    boot = {k: [] for k in keys}
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    for _ in range(n_boot):
        # stratified bootstrap (keeps both classes present each draw)
        bi = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                             rng.choice(neg_idx, len(neg_idx), replace=True)])
        m = _point_metrics(y_true[bi], y_score[bi], threshold)
        for k in keys:
            boot[k].append(m[k])
    res = {'label': label, 'n': n, 'n_pos': int(y_true.sum()),
           'prevalence': point['prevalence'], 'threshold': float(threshold)}
    for k in keys:
        lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
        res[k] = point[k]
        res[f'{k}_ci'] = (float(lo), float(hi))
    return res


def format_result(r):
    def f(k):
        lo, hi = r[f'{k}_ci']; return f"{r[k]:.3f} [{lo:.3f}-{hi:.3f}]"
    return (f"{r.get('label',''):<28} n={r['n']:>6} pos={r['n_pos']:>5} prev={r['prevalence']:.3f} thr={r['threshold']:.3f}\n"
            f"    AUROC {f('auroc')}  AUPRC {f('auprc')}\n"
            f"    Sens  {f('sensitivity')}  Spec {f('specificity')}\n"
            f"    PPV   {f('ppv')}  NPV  {f('npv')}")


def results_to_row(r, lookback, split):
    """Flat dict for a CSV summary table."""
    row = {'lookback': lookback, 'split': split, 'n': r['n'], 'n_pos': r['n_pos'],
           'prevalence': round(r['prevalence'], 4), 'threshold': round(r['threshold'], 4)}
    for k in ['auroc', 'auprc', 'sensitivity', 'specificity', 'ppv', 'npv']:
        row[k] = round(r[k], 4)
        row[f'{k}_lo'] = round(r[f'{k}_ci'][0], 4)
        row[f'{k}_hi'] = round(r[f'{k}_ci'][1], 4)
    return row
