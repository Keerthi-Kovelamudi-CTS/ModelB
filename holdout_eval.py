"""
Holdout evaluation — iso-sensitivity specificity comparison.

Single judge for every A/B (B1 ratio 1:1-vs-1:10, feature-scope all-vs-cumimp99-vs-topN,
v2 recency/xpoll). Calibration-proof: thresholds are set by SENSITIVITY (on the internal
test, which has positives), then applied to the 300K all-negative holdout for specificity
+ PPV at realistic prevalence. NEVER compares at a fixed probability threshold (the models
calibrate to different scales).

Inputs per model = two arrays:
  test_y, test_score   — internal test (has both classes); sets the thresholds
  holdout_score        — 300K holdout (all non-cancer); measures specificity / FP-rate
Reports, at Sens ∈ {80,85,90,95}%: threshold, test-spec, holdout-spec, PPV + alerts/1k
at the deployment prevalence; plus AUC (threshold-free).

Usage:
  from holdout_eval import evaluate_model, compare_models
  compare_models({'1:1': (ty,ts,hs), '1:10': (ty2,ts2,hs2)}, prevalence=0.0015)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SENS_POINTS = [0.80, 0.85, 0.90, 0.95]


def _threshold_for_sensitivity(test_y, test_score, target_sens):
    """Threshold on the test positives such that TPR == target_sens."""
    pos = np.asarray(test_score)[np.asarray(test_y) == 1]
    if len(pos) == 0:
        return np.nan
    # sens = P(score >= thr | positive); thr = the (1-sens) quantile of positive scores
    return float(np.quantile(pos, 1.0 - target_sens))


def evaluate_model(test_y, test_score, holdout_score, prevalence=0.0015):
    """Return (auc, DataFrame) for one model across the sensitivity points."""
    test_y = np.asarray(test_y); test_score = np.asarray(test_score)
    holdout_score = np.asarray(holdout_score)
    auc = roc_auc_score(test_y, test_score) if len(np.unique(test_y)) > 1 else np.nan
    neg = test_score[test_y == 0]
    rows = []
    for s in SENS_POINTS:
        thr = _threshold_for_sensitivity(test_y, test_score, s)
        spec_test = float((neg < thr).mean()) if len(neg) else np.nan
        spec_hold = float((holdout_score < thr).mean()) if len(holdout_score) else np.nan
        fp = 1.0 - spec_hold
        denom = s * prevalence + fp * (1 - prevalence)
        ppv = (s * prevalence) / denom if denom > 0 else 0.0
        alerts_per_1k = denom * 1000.0
        rows.append({
            'sens': s, 'threshold': round(thr, 4),
            'spec_test': round(spec_test, 4),
            'spec_holdout': round(spec_hold, 4),
            f'PPV@{prevalence:.4f}': round(ppv, 4),
            'alerts_per_1k': round(alerts_per_1k, 1),
        })
    return auc, pd.DataFrame(rows)


def compare_models(models: dict, prevalence=0.0015):
    """models: {name: (test_y, test_score, holdout_score)}. Prints AUC + a side-by-side
    spec_holdout table at each sensitivity, and flags the winner per row (highest
    holdout specificity = fewest false positives at equal catch-rate)."""
    aucs, tables = {}, {}
    for name, (ty, ts, hs) in models.items():
        aucs[name], tables[name] = evaluate_model(ty, ts, hs, prevalence)
    print("AUC (threshold-free):  " + "  ".join(f"{n}={aucs[n]:.4f}" for n in models))
    print(f"\nHoldout specificity @ matched sensitivity (prevalence={prevalence}):")
    hdr = f"{'Sens':>5} | " + " | ".join(f"{n:>12}" for n in models) + " | winner"
    print(hdr); print('-' * len(hdr))
    for i, s in enumerate(SENS_POINTS):
        vals = {n: tables[n].iloc[i]['spec_holdout'] for n in models}
        win = max(vals, key=vals.get)
        print(f"{int(s*100):>4}% | " + " | ".join(f"{vals[n]:>12.4f}" for n in models) + f" | {win}")
    print("\nPPV / alerts-per-1,000 at deployment prevalence (Sens=90%):")
    for n in models:
        r = tables[n].iloc[SENS_POINTS.index(0.90)]
        print(f"  {n:>12}: PPV={r[f'PPV@{prevalence:.4f}']:.4f}  alerts/1k={r['alerts_per_1k']}")
    return aucs, tables
