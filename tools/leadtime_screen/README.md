# Lead-time leakage/signal screen (v2.1, 2026-06-08)

Per code: lift = P(code|cancer)/P(code|non-cancer) across months-before-dx bins
[0-3, 3-12, 12-24, 24-60, 60+]. Runs on existing preprocessed cohort (MONTHS_BEFORE_INDEX) —
no new extraction. Featurization ground-truth = CATEGORY non-null.

## Classification (CORRECTED rule — a near-dx spike is NOT leakage by itself)
- REACTIVE-LEAK : near-dx spike AND ~no earlier elevation AND term is a DIAGNOSIS-PROCESS
                  code (biopsy/referral/imaging/surgery/carcinoma/...). => propose for *_pathway_codes (human review).
- near-dx-symptom : spikes near dx but NOT process-like (sore nipple, menorrhagia) => KEEP (legitimate presenting signal).
- PRODROMAL : spikes near dx BUT still elevated 24-60mo (breast lump: 153x@0-3, 2.5x@24-60) => genuine.
- EARLY / LIFETIME : elevated in the long window => genuine early-warning / risk factor.
The screen NEVER auto-cuts symptoms; it only proposes process codes for review.

## Files
- leadtime_screen_allcancers.py        : all 7 cancers, corrected rule (this is the main one).
- leadtime_screen_breast_detailed.py   : single-cancer full per-code table + window-tuned lists.
- leadtime_screen_v1_codelist_annot.py : early version (codelist-path annotation; superseded).

## Findings (2026-06-08)
- Independently rediscovered the textbook early marker for every cancer (24-60mo lift):
  haematuria->Bladder, PSA/prostatic-hyperplasia->Prostate, actinic-keratosis->Melanoma,
  pelvic-mass/CA125->Ovarian, lymphocytosis->Leukaemia, paraprotein->Lymphoma.
- All 7 codelists comprehensive: every early signal is already featurized; ~0 process-leakage
  currently featurized (existing *_pathway_codes exclusions work).
- LIMITATION: preprocessed data = featurized-only (100% coverage) -> cannot see codes OUTSIDE
  the codelist. To find truly-missing codes: BigQuery all-codes lead-time re-aggregation
  (per cancer: code x bin x class COUNT DISTINCT patient), then this same classifier.
