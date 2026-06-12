"""Deployment calibration — per-age-band King & Zeng prior-correction to TRUE population prevalence.

    logit(p_cal) = logit(p_raw) + logit(tau_band(age)) - logit(ybar)      ybar = 0.5 (balanced training)

tau_band = adult lung-cancer prevalence by age band, computed from EMIS (N=3,112,596 adults,
reference 2026-01-01). Monotonic within band -> ranking unchanged; gives probabilities calibrated
to each patient's true age-specific base rate. Refs: King & Zeng 2001; Saerens 2002; Elkan 2001.
"""
import numpy as np
YBAR = 0.5
# (age_lo, age_hi, tau_band) — true adult lung prevalence per band, from BigQuery
TAU_BAND = [(0,55,0.000129),(55,65,0.001612),(65,70,0.003860),
            (70,75,0.006449),(75,80,0.008764),(80,200,0.008575)]
TAU_GLOBAL = 0.0018
def _logit(p):
    p = np.clip(p, 1e-9, 1-1e-9); return np.log(p/(1-p))
def _offset_for_age(a):
    for lo,hi,tau in TAU_BAND:
        if lo <= a < hi: return _logit(tau) - _logit(YBAR)
    return _logit(TAU_GLOBAL) - _logit(YBAR)
def calibrate(raw_prob, age):
    """raw_prob: model balanced-scale probability(ies); age: years. -> deployment-calibrated prob."""
    rp = np.clip(np.atleast_1d(np.asarray(raw_prob, float)), 1e-9, 1-1e-9)
    ages = np.atleast_1d(np.asarray(age, float))
    off = np.array([_offset_for_age(x) for x in ages])
    out = 1/(1+np.exp(-(np.log(rp/(1-rp)) + off)))
    return out if out.size > 1 else float(out[0])
