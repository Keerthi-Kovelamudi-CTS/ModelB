# Dual-horizon held-out evaluation

`evaluate_dual_horizon.py` — evaluate the **12mo and 1mo models together** on the touch-once
held-out, to answer one question: **does combining the two horizon models beat using a single
model, and if so, how should they be combined?**

This is the lung analogue of the Prostate dual-horizon deployment (run both models, combine per
patient). Everything below is measured on the fixed **500 cancer / 50,000 non-cancer** held-out.

---

## 1. Background — why "dual horizon" is even a question

We train **two** models per cohort ratio and lookback window:

| model | trained on | predicts |
|---|---|---|
| **12mo model** | features built from events ending **12 months** before diagnosis | "will this patient be diagnosed in ~12 months?" (early warning) |
| **1mo model** | features ending **1 month** before diagnosis | "…in ~1 month?" (imminent) |

The single-horizon held-out eval (`evaluate_heldout.py`) tests each model **only on its own
horizon**: 12mo-model → 12mo held-out, 1mo-model → 1mo held-out. The dual-horizon eval asks what
happens when you **run both models on the same patient and merge the two scores into one decision** —
because *at deployment you don't know a patient's lead time*, so you can't pick "the matched model."

---

## 2. The key fact that makes this clean

The two held-out cohorts are **the same patients**. `heldout_test_12mo.sql` and
`heldout_test_1mo.sql` are **byte-identical except one parameter**:

```
months_before = 12   (heldout_test_12mo.sql)
months_before = 1    (heldout_test_1mo.sql)
```

Same 500/50k patients, same anchors (diagnosis dates) — only the **feature gap** differs (how far
before diagnosis the feature window is cut). So every held-out patient has **two feature views**, and
we can score each view with its model and align the two scores by `patient_guid`.

**Deployment note:** at inference there is *no* 12mo-vs-1mo dataset — there's just the patient's EHR
**up to today**. Both models receive the *same* current feature vector; they differ only in what they
*learned to look for*. The two held-out datasets simulate "how does each model behave when the patient
is genuinely 12 / 1 months from diagnosis."

---

## 3. What the script does

For **each** held-out dataset (12mo and 1mo), it scores that dataset with **both** models:

```
DATASET heldout_test_12mo.sql :  12mo model (matched) | 1mo model (cross)  | combined
DATASET heldout_test_1mo.sql  :  1mo  model (matched) | 12mo model (cross) | combined
```

- **matched** = the model trained for this dataset's gap.
- **cross** = the *other* model applied to this dataset (each `CancerPredictor` reindexes the matrix
  to its own trained feature set, missing columns → 0).

Pipeline per dataset:
1. **Reuse** the `heldout_features_{window}.csv` cached by the single-horizon runs — **no re-FE**.
2. Score with both models → raw probabilities `raw12`, `raw1`.
3. Shared stratified **30% calib / 70% test** split (seed 42) — same partition for both models so the
   test patients align.
4. **Platt-calibrate** each model on the calib slice; report everything on the disjoint **70% test**.
5. Build all combine rules, report metrics + a target-free operating curve.

It is leakage-safe (touch-once: calibration/threshold tuning use only the 30% calib slice or are read
off the same test slice that's being reported, never the training data).

---

## 4. The combine rules (and why each)

Let `p12`, `p1` be the calibrated per-patient probabilities from the two models.

| rule | definition | character | use |
|---|---|---|---|
| **matched** | single matched model | baseline | "what one model gives" |
| **cross** | single other-horizon model | robustness check | how well a model generalises off-horizon |
| **max** | `max(p12, p1)` | recall-greedy — can only push scores **up**, so it inherits the weaker model's false positives | catches most, worst specificity |
| **mean** | `(p12 + p1) / 2` | tempered — a noisy high score is pulled back by the other model | blunt compromise |
| **stacker** | logistic regression on `[raw12, raw1]`, fit on the calib slice | **learns the weights** per data + its own calibration | principled; adapts to which model to trust |
| **OR** (reference) | flag if `p12 ≥ T12` **OR** `p1 ≥ T1` | recall-greedy, **uncontrolled** combined operating point | deployment "either fires"; kept only for reference |

**Why the stacker is preferred:** `max`/`OR` are recall-greedy — on a dataset where one model is
already strong (e.g. 1mo on 1mo data), they only *add* the weaker model's false positives and **lower
specificity**. The stacker instead *learns* to down-weight the weaker model, so it **does not degrade
the strong model**. **Noisy-OR** (`1−(1−p12)(1−p1)`) is intentionally **not** included: it assumes the
two models are independent, but they're highly correlated (same patients/features), so it would
over-combine and come out miscalibrated.

---

## 5. How to run

Runs **after** the single-horizon held-outs for the ratio exist (it reuses their feature matrices +
models). One invocation evaluates **both** datasets.

```bash
cd Lung_Cancer_K1/V1
NC_RATIO=1 WINDOW=5yr python 4_Heldout/evaluate_dual_horizon.py     # 1:1, 5yr
NC_RATIO=5 WINDOW=5yr python 4_Heldout/evaluate_dual_horizon.py     # 1:5
# DATASET=12 or DATASET=1 to evaluate only one dataset
```

Env: `NC_RATIO` (1|5|10), `WINDOW` (5yr|10yr|20yr|lifetime), `DATASET` (both|12|1), `CALIB_FRAC`
(default 0.30). Output: prints + saves `dual_heldout_{window}_1to{ratio}.txt` in this folder.

> Requires, per horizon: `{12,1}mo_1to{NC_RATIO}/lookback/{WINDOW}/heldout_features_{WINDOW}.csv` and
> `model_{WINDOW}_1to{NC_RATIO}.joblib` — i.e. run `evaluate_heldout.py` for both 12mo and 1mo first.

---

## 6. How to read the output

```
### DATASET = heldout_test_12mo.sql  | test n=35350 (pos 350, prev 0.99%)
model / combine            AUROC   AUPRC   Sens   Spec    PPV    NPV  (@Youden)
12mo model (matched)       0.911   0.129   88.0   79.2    4.1   99.8
1mo model (cross)          0.907   0.116   93.7   72.1    3.3   99.9
DUAL max(p12,p1)           0.912   0.119   89.1   77.9    3.9   99.9
DUAL mean(p12,p1)          0.913   0.127   88.0   78.7    4.0   99.8
DUAL stacker (LR)          0.912   0.127   89.7   77.1    3.8   99.9
DUAL OR (reference)            -       -   94.6   71.4    3.2   99.9
# stacker weights [12mo=+3.65, 1mo=+2.55]      <- which model the stacker trusts on this data
# coverage of 350 cancers: both=305  12mo-only=3  1mo-only=23  missed-by-both=19
# Spec @ fixed Sens (test slice) — which combine wins at each recall (higher=better):
 Sens |  matched     max    mean  stacker
  80% |     84.3    84.8    83.9     83.8
  ...
```

- **AUROC / AUPRC** — threshold-free ranking quality. `OR` has none (it's a fixed yes/no, not a score).
- **@Youden block** — Sens/Spec/PPV/NPV at each score's own Youden operating point (just one reference point).
- **stacker weights** — the logistic coefficients on `[12mo, 1mo]`; the larger one is the model the
  stacker leans on for this dataset (it should favour the matched model).
- **coverage** — of the true cancers in the test slice, how many are caught by **both** models,
  **matched-only**, **cross-only**, or **missed by both**. Shows whether the cross model adds real
  coverage. `missed-by-both` is the hard floor neither horizon reaches.
- **Spec @ fixed Sens** — the **target-free operating curve**: at each recall level, the specificity
  of each combine rule. This is the table to choose an operating point from (no threshold is hard-coded).

---

## 7. Findings (1:1, 5yr)

**`heldout_test_1mo.sql` — Spec @ fixed Sens:**

| Sens | matched (1mo) | max | mean | **stacker** |
|---|---|---|---|---|
| 80% | 89.4 | 89.0 | 89.2 | **89.6** |
| 85% | **86.6** | 84.3 | 84.6 | **85.9** |
| 90% | 82.5 | 81.7 | 81.7 | **81.9** |
| 95% | 75.9 | 73.5 | 74.2 | **75.9** |

**`heldout_test_12mo.sql` — Spec @ fixed Sens:** all rules within ~1 pp of each other (a wash).

Conclusions:
1. **The 1mo model is the strongest single component** — AUROC 0.937 on its own data and a robust
   0.907 even on 12mo data. The 12mo model: 0.911 / 0.923. Neither dominates, but 1mo is the standout.
2. **`max`/`OR` are recall-greedy and *degrade* the strong (1mo) scenario** — they can only add the
   weaker model's false positives (~2 pp specificity lost at 85% recall). Great for raw recall, bad
   for specificity at low prevalence where false-positive count dominates.
3. **The stacker fixes that** — it learns to weight the matched model heavier (1mo data:
   `[12mo +2.51, 1mo +4.31]`; 12mo data: `[12mo +3.65, 1mo +2.55]`), so it **recovers the matched
   model's specificity** while still using the other model where it helps. It never meaningfully
   underperforms the matched single model on either dataset.
4. **The honest ceiling:** even the stacker **≈ the matched single model** — it doesn't *beat* it, it
   *matches whichever model is right without needing to know the lead time.*

---

## 8. Recommendation

- **If shipping a dual model → ship the stacker.** It's lead-time-agnostic (you don't know a patient's
  lead time at inference), it never degrades the strong model, and it outputs a calibrated probability.
  **Do not** ship `OR`/`max` — recall-greedy, uncontrolled specificity.
- **Set the operating point from the `Spec @ fixed Sens` curve**, to an explicit **sensitivity target
  or alert budget** — at ~1% prevalence the absolute false-positive count is the binding constraint.
- **Simplest viable alternative:** ship the **1mo model alone** — strong on both scenarios; the dual's
  incremental value over it on a single snapshot is small.
- The dual's real, un-measured edge is **longitudinal** (re-screening over time: the 12mo model warns
  early, the 1mo model escalates as diagnosis nears) — the single-shot eval understates this.

## 9. Caveats

- Per-dataset the stacker is fit on **that dataset's** calib slice (this is the *ceiling* — as if you
  knew the lead time). For real deployment, train **one** stacker on mixed-lead-time data and verify on
  a mixed holdout; it should still avoid the max-degradation (it learns balanced weights), but confirm.
- All numbers above are **1:1, 5yr**. The **1:5 / 1:10** runs change the false-positive economics
  (more negatives) — the picture (especially specificity/PPV) may shift; re-read after those land.
- Cross-model scores are "off-horizon" (a model seeing a gap it wasn't trained on). Calibration adapts
  via Platt, and AUROC stays high, but treat cross rows as robustness evidence, not a deployment mode.
