"""Compute Brier score + reliability for any saved {model}_y_proba_test.npy + _y_test.npy."""
import numpy as np
import os, glob, sys

LSTM_DIR = '/home/keerthikovelamudi_cthesigns_com/lstm_work'
proba_dir = f'{LSTM_DIR}/probas'

if not os.path.exists(proba_dir):
    print(f'No proba dir yet — run k14/k15 first to generate {proba_dir}/')
    sys.exit(0)

files = sorted(glob.glob(f'{proba_dir}/*_y_proba_test.npy'))
if not files:
    print(f'No proba files in {proba_dir} yet')
    sys.exit(0)

print(f'{"model":<55} {"brier":>8} {"BSS":>8} {"ECE":>8} {"slope":>7} {"intcp":>7}')
print('-' * 95)

for fp in files:
    tag = os.path.basename(fp).replace('_y_proba_test.npy', '')
    y_proba = np.load(fp)
    y_true = np.load(f'{proba_dir}/{tag}_y_test.npy')

    # Brier score
    brier = np.mean((y_proba - y_true) ** 2)

    # Brier Skill Score (BSS) = 1 - Brier / Brier_baseline (always predict mean)
    p_mean = y_true.mean()
    brier_base = np.mean((p_mean - y_true) ** 2)
    bss = 1 - brier / brier_base if brier_base > 0 else 0

    # Expected Calibration Error (ECE) — 10 bins
    bins = np.linspace(0, 1, 11)
    ece = 0
    for i in range(10):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_proba[mask].mean()
            ece += (mask.sum() / len(y_proba)) * abs(bin_acc - bin_conf)

    # Calibration slope/intercept (logistic regression of true ~ logit(proba))
    eps = 1e-6
    p_clip = np.clip(y_proba, eps, 1 - eps)
    logit = np.log(p_clip / (1 - p_clip))
    # Fit y_true ~ a + b * logit via simple LR
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones_like(logit), logit])
    coef, *_ = lstsq(X, y_true.astype(float), rcond=None)
    intercept, slope = coef
    # Perfect calibration: slope=1, intercept=0

    print(f'{tag:<55} {brier:>8.4f} {bss:>8.4f} {ece:>8.4f} {slope:>7.3f} {intercept:>7.3f}')

print()
print('Interpretation:')
print('  Brier: lower is better. Random guess = 0.25 for balanced. Lower bound = 0.')
print('  BSS:   higher is better. 1.0 = perfect, 0 = no skill, <0 = worse than baseline.')
print('  ECE:   lower is better. Expected calibration error across 10 bins.')
print('  slope: 1.0 = perfect. <1 = overconfident, >1 = underconfident.')
print('  intcp: 0.0 = perfect. >0 = predictions skewed too low.')
