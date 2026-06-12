# Calibration notes — lung / prostate / bladder / breast (1:1 held-out)

_Generated 2026-06-11. Pre-deployment probability calibration on the held-out set._

## End-to-end pipeline (data -> calibrated model)
The same config-driven pipeline runs per cancer; only the cohort SQL + cancer-specific feature config differ.

1. **Cohort (BigQuery).** Cancer cases + non-cancer controls from the EMIS bulk data. Controls exclude any cancer-SNOMED and palliative-care patients; sex filter per cancer (prostate = M, breast = F, lung/bladder = none). An **anchor (index) date** per patient; label = a diagnosis *after* anchor; **all features strictly before anchor minus a gap** = the **prediction horizon** (`months_before` = 12 or 1). A separate **500-cancer / 50k-control held-out** is carved out and **excluded from training**.
2. **Event extraction.** One long table of pre-anchor clinical + medication events per patient (SNOMED concept, value, `days_before_anchor`, event_type, demographics), cached once.
3. **Feature engineering (per lookback window).** For each window (5/10/20yr/lifetime), filter to `days_before_anchor <= window`, build risk-factor features (counts, frequency/recency, trajectory/trend, lab values & deltas, demographics) + **lab percentiles (xpoll) fit on train only**, applied to held-out (leakage-safe). Reindexed to the full cohort (no-event patients -> zero rows).
4. **Training (per window × horizon).** 8-learner panel (LR/RF/ET/Ada/GB/XGB/LGBM/CatBoost), internal 80/10/10; **best single model or top-3 soft-voting ensemble** selected on internal validation -> `model_<window>_1to1.joblib`. **1:1 balanced**, both **12mo + 1mo** horizons.
5. **Internal evaluation.** Balanced test, **free/Youden** point, AUROC/AUPRC/Sens/Spec/PPV/NPV + bootstrap CIs.
6. **Held-out evaluation (touch-once).** The 500/50k enriched held-out (**0% train overlap, verified**) FE'd per window, scored once at the internal threshold.
7. **Probability calibration (this doc).** **Platt sigmoid on the held-out** (30% calib / 70% test) -> ECE ~0.001; saved `platt_calib_<window>.joblib`. Monotonic -> discrimination unchanged.
8. **Deployment (future).** Pick per-cancer window + horizon, load `model + platt_calib`, add the **King-Zeng tau-offset** for true population prevalence (see Deployment).

*(Parallel track - ensemble: for the lung **b12_v3** blending stack, the same structured model is retrained on the ensemble TRAIN partition and emits per-patient `proba` on VALID/TEST in the shared contract - separate from this calibration doc.)*

## Approach (all cancers, shared pipeline)
- **Model per window:** an 8-learner panel (LR/RF/ET/Ada/GB/XGB/LGBM/CatBoost) is trained; the **single best** or a **top-3 soft-voting ensemble** is selected on internal validation and saved as `model_<window>_1to1.joblib` (its members are the "Best Model" column below).
- **Held-out:** a touch-once 500/50k enriched set, **disjoint from training** (0% patient overlap, verified). Operating point = **free/Youden** (max sensitivity+specificity), AUROC-independent.
- **Calibration:** **Platt scaling (sigmoid)** fit **on the held-out** — see method below.
- **Horizons:** 12-month and 1-month (months before the index date); **lookback windows** 5yr/10yr/20yr/lifetime = how far back features look.

## Calibration method — Platt scaling (sigmoid), and why
Raw tree-ensemble probabilities are badly miscalibrated on the held-out (**ECE ≈ 0.2–0.5**) because the
models are trained on a **balanced 1:1** sample. We calibrate **on the held-out, to the held-out's own
prevalence** (we are **not deploying yet**, so no real-world-prevalence correction):

> raw probs → **stratified 30% calib / 70% test split** → fit a **1-parameter logistic (Platt) on the
> calib slice (on the logits)** → evaluate ECE/Brier on the **disjoint 70% test slice**.

- **Why Platt** (not isotonic): equally accurate here (both → ECE ~0.001) but Platt is a smooth 1-param
  fit — robust at our positive counts, no step artefacts in sparse regions. Isotonic kept as a cross-check.
- **Monotonic** → AUROC / AUPRC / Sens / Spec **unchanged**; only the probability scale is corrected.
- **Artifact per window:** `platt_calib_<window>.joblib` (the fitted sigmoid), `reliability_<window>.png`,
  `platt_calib_<window>.json` (ECE/Brier). Apply at inference: `p_cal = platt.predict_proba(logit(p_raw))`.

**References:** Platt 1999 (probabilistic outputs / sigmoid); Niculescu-Mizil & Caruana 2005 (calibration of
tree/boosted models); Guo et al. 2017 (ECE / modern calibration). For the *deployment* prevalence step, see §Deployment.

### Calibration split (of that same held-out — no separate dataset)
Platt is fit on a stratified **30% calib** slice and evaluated on the **70% test** slice (cancer / non-cancer):

| Cancer | Held-out total | Calib (fit) 30% — ca / non-ca | Test (eval) 70% — ca / non-ca |
|---|---|---|---|
| Lung | 50,500 (500 / 50,000) | 15,150 (150 / 15,000) | 35,350 (350 / 35,000) |
| Prostate | 51,500 (1,500 / 50,000) | 15,450 (450 / 15,000) | 36,050 (1,050 / 35,000) |
| Bladder | 50,600 (600 / 50,000) | 15,180 (180 / 15,000) | 35,420 (420 / 35,000) |
| Breast | 52,000 (2,000 / 50,000) | 15,600 (600 / 15,000) | 36,400 (1,400 / 35,000) |

_Performance metrics (Sens/Spec/PPV/NPV/AUROC/AUPRC) below are on the **full held-out**; the **ECE raw→Platt** column is on the 70% test slice._


## Lung

**12mo · 1:1 model** — held-out **N=50,500** (500 cancer / 50,000 non-cancer), prevalence 1.0%. Deployment pick (best held-out AUROC): **20yr**.

| Window | Best Model | Sens | Spec | PPV | NPV | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | ECE raw→Platt |
|---|---|---|---|---|---|---|---|---|---|---|
| 5yr | Ens(LGBM+XGB+Cat) | 93.0 | 70.5 | 3.1 | 99.9 | 0.902 | 0.891–0.911 | 0.102 | 0.087–0.124 | 0.293→0.0004 |
| 10yr | Ens(GB+XGB+Cat) | 93.4 | 70.9 | 3.1 | 99.9 | 0.902 | 0.889–0.912 | 0.115 | 0.096–0.141 | 0.332→0.0008 |
| 20yr ★ | Ens(XGB+Cat+RF) | 94.4 | 67.8 | 2.9 | 99.9 | 0.913 | 0.901–0.924 | 0.151 | 0.128–0.179 | 0.344→0.0009 |
| lifetime | XGB | 91.2 | 72.9 | 3.2 | 99.9 | 0.906 | 0.895–0.917 | 0.111 | 0.096–0.133 | 0.271→0.0007 |

**1mo · 1:1 model** — held-out **N=50,500** (500 cancer / 50,000 non-cancer), prevalence 1.0%. Deployment pick (best held-out AUROC): **10yr**.

| Window | Best Model | Sens | Spec | PPV | NPV | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | ECE raw→Platt |
|---|---|---|---|---|---|---|---|---|---|---|
| 5yr | Ens(XGB+LGBM+ET) | 97.8 | 58.1 | 2.3 | 100.0 | 0.923 | 0.913–0.932 | 0.164 | 0.138–0.195 | 0.505→0.0008 |
| 10yr ★ | Ens(XGB+Cat+MLP) | 96.8 | 65.4 | 2.7 | 100.0 | 0.930 | 0.920–0.939 | 0.182 | 0.156–0.216 | 0.449→0.0009 |
| 20yr | XGB | 88.0 | 81.1 | 4.5 | 99.9 | 0.922 | 0.910–0.933 | 0.182 | 0.155–0.215 | 0.318→0.0009 |
| lifetime | LGBM | 92.6 | 76.6 | 3.8 | 99.9 | 0.921 | 0.909–0.932 | 0.159 | 0.137–0.189 | 0.326→0.0008 |


## Prostate

**12mo · 1:1 model** — held-out **N=51,500** (1,500 cancer / 50,000 non-cancer), prevalence 2.9%. Deployment pick (best held-out AUROC): **20yr**.

| Window | Best Model | Sens | Spec | PPV | NPV | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | ECE raw→Platt |
|---|---|---|---|---|---|---|---|---|---|---|
| 5yr | Ens(LGBM+XGB+GB) | 91.3 | 81.3 | 12.8 | 99.7 | 0.930 | 0.925–0.934 | 0.291 | 0.270–0.313 | 0.197→0.0022 |
| 10yr | Ens(Cat+XGB+LGBM) | 92.8 | 81.6 | 13.2 | 99.7 | 0.938 | 0.934–0.942 | 0.339 | 0.318–0.362 | 0.192→0.0026 |
| 20yr ★ | Ens(LGBM+Cat+GB) | 89.6 | 84.6 | 14.9 | 99.6 | 0.942 | 0.938–0.946 | 0.349 | 0.327–0.373 | 0.180→0.0023 |
| lifetime | Cat | 91.5 | 83.3 | 14.1 | 99.7 | 0.942 | 0.938–0.945 | 0.347 | 0.325–0.370 | 0.189→0.0032 |

**1mo · 1:1 model** — held-out **N=51,500** (1,500 cancer / 50,000 non-cancer), prevalence 2.9%. Deployment pick (best held-out AUROC): **20yr**.

| Window | Best Model | Sens | Spec | PPV | NPV | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | ECE raw→Platt |
|---|---|---|---|---|---|---|---|---|---|---|
| 5yr | Ens(XGB+LGBM+Cat) | 95.5 | 83.1 | 14.5 | 99.8 | 0.959 | 0.956–0.962 | 0.526 | 0.500–0.552 | 0.198→0.0015 |
| 10yr | Ens(XGB+Cat+LGBM) | 96.9 | 79.0 | 12.1 | 99.9 | 0.956 | 0.952–0.959 | 0.520 | 0.495–0.545 | 0.195→0.0020 |
| 20yr ★ | Ens(XGB+Cat+LGBM) | 97.2 | 79.8 | 12.6 | 99.9 | 0.961 | 0.958–0.964 | 0.558 | 0.533–0.584 | 0.179→0.0022 |
| lifetime | Ens(XGB+LGBM+Cat) | 96.9 | 78.8 | 12.1 | 99.9 | 0.959 | 0.956–0.962 | 0.544 | 0.520–0.569 | 0.177→0.0020 |


## Bladder

**12mo · 1:1 model** — held-out **N=50,600** (600 cancer / 50,000 non-cancer), prevalence 1.2%. Deployment pick (best held-out AUROC): **lifetime**.

| Window | Best Model | Sens | Spec | PPV | NPV | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | ECE raw→Platt |
|---|---|---|---|---|---|---|---|---|---|---|
| 5yr | Ens(GB+LGBM+Ada) | 81.5 | 84.3 | 5.9 | 99.7 | 0.909 | 0.900–0.918 | 0.126 | 0.110–0.147 | 0.266→0.0005 |
| 10yr | Ens(RF+LGBM+ET) | 88.0 | 80.8 | 5.2 | 99.8 | 0.913 | 0.903–0.922 | 0.121 | 0.108–0.139 | 0.240→0.0004 |
| 20yr | Ens(Cat+ET+Ada) | 89.5 | 78.1 | 4.7 | 99.8 | 0.915 | 0.907–0.925 | 0.132 | 0.115–0.152 | 0.275→0.0002 |
| lifetime ★ | Ens(XGB+LGBM+ET) | 88.2 | 80.1 | 5.1 | 99.8 | 0.917 | 0.908–0.925 | 0.128 | 0.114–0.148 | 0.228→0.0007 |

**1mo · 1:1 model** — held-out **N=50,600** (600 cancer / 50,000 non-cancer), prevalence 1.2%. Deployment pick (best held-out AUROC): **10yr**.

| Window | Best Model | Sens | Spec | PPV | NPV | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | ECE raw→Platt |
|---|---|---|---|---|---|---|---|---|---|---|
| 5yr | Ens(XGB+Cat+GB) | 82.8 | 88.6 | 8.0 | 99.8 | 0.933 | 0.925–0.941 | 0.253 | 0.222–0.290 | 0.198→0.0015 |
| 10yr ★ | Ens(XGB+GB+Cat) | 89.7 | 83.0 | 5.9 | 99.9 | 0.934 | 0.926–0.942 | 0.245 | 0.212–0.281 | 0.196→0.0016 |
| 20yr | Ens(Cat+GB+XGB) | 89.2 | 84.2 | 6.3 | 99.9 | 0.933 | 0.925–0.941 | 0.258 | 0.224–0.294 | 0.208→0.0011 |
| lifetime | Ens(XGB+Cat+ET) | 87.3 | 85.0 | 6.5 | 99.8 | 0.932 | 0.925–0.941 | 0.267 | 0.233–0.301 | 0.215→0.0011 |


## Breast

**12mo · 1:1 model** — held-out **N=52,000** (2,000 cancer / 50,000 non-cancer), prevalence 3.9%. Deployment pick (best held-out AUROC): **lifetime**.

| Window | Best Model | Sens | Spec | PPV | NPV | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | ECE raw→Platt |
|---|---|---|---|---|---|---|---|---|---|---|
| 5yr | Ens(RF+Cat+LGBM) | 90.5 | 52.7 | 7.1 | 99.3 | 0.795 | 0.786–0.803 | 0.120 | 0.113–0.130 | 0.378→0.0014 |
| 10yr | Ens(RF+ET+LGBM) | 89.2 | 56.5 | 7.6 | 99.2 | 0.801 | 0.794–0.809 | 0.128 | 0.120–0.139 | 0.348→0.0003 |
| 20yr | Ens(GB+Cat+RF) | 89.3 | 57.3 | 7.7 | 99.3 | 0.807 | 0.799–0.815 | 0.134 | 0.125–0.145 | 0.327→0.0005 |
| lifetime ★ | Ens(LGBM+Cat+RF) | 90.2 | 56.3 | 7.6 | 99.3 | 0.809 | 0.801–0.817 | 0.144 | 0.134–0.157 | 0.340→0.0020 |

**1mo · 1:1 model** — held-out **N=52,000** (2,000 cancer / 50,000 non-cancer), prevalence 3.9%. Deployment pick (best held-out AUROC): **lifetime**.

| Window | Best Model | Sens | Spec | PPV | NPV | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | ECE raw→Platt |
|---|---|---|---|---|---|---|---|---|---|---|
| 5yr | Ens(LR+RF+XGB) | 93.8 | 47.1 | 6.6 | 99.5 | 0.797 | 0.789–0.806 | 0.129 | 0.120–0.139 | 0.440→0.0005 |
| 10yr | Ens(GB+XGB+LGBM) | 88.9 | 58.1 | 7.8 | 99.2 | 0.807 | 0.799–0.814 | 0.135 | 0.125–0.147 | 0.336→0.0011 |
| 20yr | Ens(XGB+Cat+LGBM) | 88.3 | 58.6 | 7.9 | 99.2 | 0.805 | 0.798–0.814 | 0.139 | 0.129–0.151 | 0.330→0.0010 |
| lifetime ★ | Ens(LGBM+Cat+GB) | 89.7 | 57.3 | 7.8 | 99.3 | 0.810 | 0.803–0.819 | 0.150 | 0.140–0.163 | 0.341→0.0010 |


## Deployment (future — not applied now)
When deploying to a real GP population, the held-out's enriched prevalence is no longer the target. Apply a
**King-Zeng prior-correction** on top of the Platt-calibrated score to re-scale to the **true population
prevalence τ** (per cancer band below): `logit(p) += logit(τ) − logit(0.5)`. Monotonic, so AUROC is unchanged;
it makes the probability and PPV meaningful at the real base rate.

| Cancer | τ band | τ | Source |
|---|---|---|---|
| Lung | all-adult | 0.0018 | BigQuery + CRUK/CPRD |
| Prostate | male | 0.002 | CRUK ~52k male cases/yr ÷ ~26M adult men |
| Bladder | all-adult | 0.0002 | CRUK ~10.5k cases/yr ÷ ~53M adults |
| Breast | female-adult | 0.002 | CRUK ~56k female cases/yr ÷ ~27M adult women |

τ values are literature/BigQuery estimates — finalise from the deployment cohort before go-live.
**References (deployment):** King & Zeng 2001; Elkan 2001; Saerens 2002; van Calster 2019; CRUK incidence statistics.

## Files
Per window dir `<cancer>/results/<horizon>_1to1/<window>/`:
`metrics.csv` (internal-test metrics) · `heldout_metrics.csv` (held-out 500/50k metrics) ·
`platt_calib_<window>.json` (ECE/Brier) · `platt_calib_<window>.joblib` (fitted calibrator) ·
`reliability_<window>.png` (reliability diagram). The trained model `model_<window>_1to1.joblib`
is kept local / on GCS (not in the repo).
