"""
╔══════════════════════════════════════════════════════════════════╗
║  BREAST — Phase 1.4b — Curation Review                       ║
║                                                                ║
║  Produces _curation_reviews/breast_curation_review.csv in    ║
║  the same schema as bladder/melanoma/ovarian/lymphoma — so the ║
║  clinician can review the full ranked set in a single spread-  ║
║  sheet with consistent decision/reason vocabulary.             ║
║                                                                ║
║  Inputs:                                                       ║
║    - Output/Compare_to_curated/breast_combined_review.tsv    ║
║    - ../codelists/code_category_mapping_v2.json                ║
║                                                                ║
║  Outputs (written to project _curation_reviews/):              ║
║    - breast_curation_review.csv                              ║
║         cols: rank,score,code,term,type,decision,reason,       ║
║               already_curated,suggested_category               ║
║    - breast_skipped_no_category.csv                          ║
║         KEEP rows where no category could be auto-suggested    ║
║    - breast_curation_log_YYYYMMDD_HHMMSS.log                 ║
║                                                                ║
║  Decision vocabulary (mirrors prior reviews):                  ║
║    KEEP                     valid pre-dx signal                ║
║    REMOVE_LEAKAGE           cancer-name:<term> | cancer-pathway:<phrase>
║    REMOVE_NOISE             generic:<phrase>                   ║
║    BORDERLINE_DX_PATHWAY    dx-pathway:<phrase>                ║
║                                                                ║
║  Key clinical adjustment vs. naive substring matching:         ║
║    "Family history of X cancer" → KEEP (risk factor), NOT      ║
║    REMOVE_LEAKAGE. Prior reviews flagged these as leakage —    ║
║    that's a known false positive worth fixing here.            ║
║                                                                ║
║  Run:  python3 make_curation_review.py                         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import csv
import json
import logging
import os
import re
import sys
from datetime import datetime

import pandas as pd


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

CONFIG = {
    "v2_mapping":  os.path.join(_SCRIPT_DIR, "..", "codelists",
                                "code_category_mapping_v2.json"),
    "review_tsv":  os.path.join(_SCRIPT_DIR, "Output", "Compare_to_curated",
                                "breast_combined_review.tsv"),
    "out_dir":     os.path.join(_REPO_ROOT, "_curation_reviews"),
    "out_main":    "breast_curation_review.csv",
    "out_skipped": "breast_skipped_no_category.csv",
    "cancer_name": "breast",     # plain-language disease name
}


# ════════════════════════════════════════════════════════════════
# RULE PATTERNS (applied in cascade — first match wins)
# Order: family-history override → cancer-name → cancer-pathway
# → dx-pathway → noise → KEEP (default)
# ════════════════════════════════════════════════════════════════

# OVERRIDES — terms that LOOK like leakage/borderline but are clinically
# meaningful pre-dx signal. Checked BEFORE leakage/borderline rules.
# Substring match, lowercase, returns ("KEEP", "valid pre-dx signal — <category>")
# and the category is also recorded for suggest_category to honour.
# NOISE_OVERRIDES — substrings that look like dx-pathway (biopsy/endoscopy)
# but are unrelated to breast. Force REMOVE_NOISE before dx-pathway runs.
NOISE_OVERRIDES = [
    ("diagnostic nasendoscopy",   "minor procedure (ent)"),
    ("excision biopsy of skin",   "minor procedure (skin)"),
]

OVERRIDES = [
    # Bowel-cancer screening (NOT breast cancer) — screening invites + negatives
    ("bowel cancer screening",                         "CANCER_SCREENING"),
    ("screening for malignant neoplasm of large intestine", "CANCER_SCREENING"),
    ("bowel scope",                                    "CANCER_SCREENING"),
    ("flexible sigmoidoscopy",                         "CANCER_SCREENING"),
    ("flexible-sigmoidoscopy",                         "CANCER_SCREENING"),
    ("diagnostic endoscopic examination on colon",     "CANCER_SCREENING"),
    # Tumour markers (informative pre-dx workup signals — e.g. CA15-3, CA27.29, CEA)
    ("ca 19-9",                                        "CANCER_MARKERS"),
    ("ca 125",                                         "CANCER_MARKERS"),
    ("ca 15-3",                                        "CANCER_MARKERS"),
    # Prior non-breast skin cancers — risk factor, not breast leakage
    ("squamous cell carcinoma",                        "PRIOR_CANCER_HISTORY"),
    ("basal cell carcinoma",                           "PRIOR_CANCER_HISTORY"),
    # Breast biopsy / cystoscopy / urology referral — strong pre-12mo risk signal
    ("transrectal needle biopsy of breast",          "UROLOGY_PATHWAY"),
    ("transurethral biopsy breast",                  "UROLOGY_PATHWAY"),
    ("transperineal needle biopsy of breast",        "UROLOGY_PATHWAY"),
    ("flexible diagnostic cystoscopy",                 "UROLOGY_PATHWAY"),
    ("flexible check cystoscopy",                      "UROLOGY_PATHWAY"),
    ("diagnostic cystoscopy",                          "UROLOGY_PATHWAY"),
    ("referred to urologist",                          "UROLOGY_PATHWAY"),
    ("referral to urologist",                          "UROLOGY_PATHWAY"),
    ("seen in urology clinic",                         "UROLOGY_PATHWAY"),
    # Mis-categorised by keyword-match → redirect to correct category
    ("international normalised ratio using testing strip", "COAGULATION"),
    ("rheumatoid arthritis latex test",                "INFLAMMATORY"),
]


# Cancer-name terms (any of these in the term → REMOVE_LEAKAGE)
CANCER_NAME_TERMS = [
    "cancer", "carcinoma", "malignan", "neoplasm", "tumour", "tumor",
    "metastas", "adenocarcinoma", "sarcoma", "lymphoma", "leukaemia",
    "leukemia", "melanoma",
]

# Cancer-care pathway phrases
CANCER_PATHWAY_PHRASES = [
    ("2 week",          "2 week"),
    ("fast track",      "fast track"),
    ("multidisciplinary", "multidisciplinary"),
    ("mdt meeting",     "mdt"),
    ("oncolog",         "oncolog"),
    ("palliative",      "palliative"),
    ("chemotherap",     "chemotherap"),
    ("radiotherap",     "radiotherap"),
    ("brachytherap",    "brachytherap"),
]

# Cancer-treatment drug leakage — breast-cancer specific hormone therapy.
# Presence in the pre-dx window strongly implies a prior dx the model would otherwise miss.
CANCER_TREATMENT_DRUGS = [
    ("goserelin",       "goserelin (gnrh agonist)"),
    ("zoladex",         "goserelin (gnrh agonist)"),
    ("leuprorelin",     "leuprorelin (gnrh agonist)"),
    ("leuprolide",      "leuprorelin (gnrh agonist)"),
    ("prostap",         "leuprorelin (gnrh agonist)"),
    ("triptorelin",     "triptorelin (gnrh agonist)"),
    ("decapeptyl",      "triptorelin (gnrh agonist)"),
    ("degarelix",       "degarelix (gnrh antagonist)"),
    ("firmagon",        "degarelix (gnrh antagonist)"),
    ("bicalutamide",    "antiandrogen"),
    ("casodex",         "antiandrogen"),
    ("flutamide",       "antiandrogen"),
    ("nilutamide",      "antiandrogen"),
    ("enzalutamide",    "antiandrogen"),
    ("xtandi",          "antiandrogen"),
    ("apalutamide",     "antiandrogen"),
    ("darolutamide",    "antiandrogen"),
    ("abiraterone",     "cyp17 inhibitor"),
    ("zytiga",          "cyp17 inhibitor"),
    ("cyproterone",     "antiandrogen"),
    # Procedure-coded breast-ca hormone therapy (gonadorelin/GnRH analogues)
    ("gonadorelin analogue implant",   "gonadorelin implant (gnrh agonist)"),
    ("gonadorelin implant",            "gonadorelin implant (gnrh agonist)"),
    ("injection of gonadorelin",       "gonadorelin injection (gnrh agonist)"),
    ("gonadorelin",                    "gonadorelin (gnrh agonist) — breast-ca treatment"),
    ("injection of hormone antagonist", "hormone antagonist injection (likely gnrh)"),
    ("injection of hormone agonist",   "hormone agonist injection (likely gnrh)"),
]

# Diagnostic-pathway phrases (procedures very close to a cancer dx)
DX_PATHWAY_PHRASES = [
    ("biopsy",                 "biopsy"),
    ("cystoscop",              "cystoscopy"),
    ("colonoscop",             "colonoscopy"),
    ("sigmoidoscop",           "sigmoidoscopy"),
    ("endoscop",               "endoscopy"),
    ("colposcop",              "colposcopy"),
    ("bone marrow",            "bone marrow"),
    ("radical breastctomy",  "radical breastctomy"),
    ("transurethral resection of breast", "turp"),
    ("turp",                   "turp"),
    ("breast mri",           "breast mri"),
    ("mri of breast",        "mri breast"),
    ("referral to urolog",     "referral to urology"),
    ("referred to urolog",     "referred to urology"),
    ("urology clinic",         "urology clinic"),
    # (referral to dermatolog / haematolog / gynaecolog → NOT breast dx-pathway;
    #  handled as specialist-referral admin noise instead)
    ("referral to haematolog", "referral to haematology"),
    ("referral to gynaecolog", "referral to gynaecology"),
    ("seen in urology clinic", "seen in urology clinic"),
    ("seen by urolog",         "seen by urology"),
    # (seen in dermatology clinic → noise, not breast dx-pathway — handled below)
]

# Generic / admin / surveillance noise
NOISE_PHRASES = [
    # ── Topical / dermatology (no systemic relevance for breast cancer) ──
    ("diprobase",              "topical emollient"),
    ("aqueous cream",          "topical emollient"),
    ("hydrous ointment",       "topical emollient"),
    ("e45 cream",              "topical emollient"),
    ("daktacort",              "topical antifungal/steroid"),
    ("locorten vioform",       "topical antifungal/steroid"),
    ("alphosyl",               "topical (scalp)"),
    ("loceryl",                "topical (nail)"),
    ("medicated nail lacquer", "topical (nail)"),
    ("tioconazole",            "topical antifungal"),
    ("hydrocortisone 1%",      "topical steroid"),
    ("triamcinolone",          "topical/nasal steroid"),
    ("capsaicin",              "topical analgesic"),
    ("movelat",                "topical analgesic"),
    ("transvasin",             "topical analgesic"),
    ("algesal",                "topical analgesic"),
    ("fenbid forte",           "topical nsaid"),
    ("piroxicam 0.5% gel",     "topical nsaid"),
    ("diethylamine salicylate", "topical salicylate"),
    ("salicylic acid",         "topical salicylate"),
    ("mucopolysaccharide polysulfate", "topical"),
    ("hydrocortisone 1% / miconazole", "topical antifungal/steroid"),
    ("flumetasone 0.02% / clioquinol", "topical antifungal/steroid"),
    # ── Eye drops / ophthalmic (mostly age-related, not cancer signal) ──
    ("eye drop",               "eye drops"),
    ("eye ointment",           "eye drops"),
    ("lacri-lube",             "eye drops"),
    ("optive",                 "eye drops"),
    ("hypromellose",           "eye drops"),
    ("carmellose",             "eye drops"),
    ("viscotears",             "eye drops"),
    ("bimatoprost",            "eye drops"),
    ("travoprost",             "eye drops"),
    ("xalatan",                "eye drops"),
    ("nedocromil",             "eye drops"),
    ("fucithalmic",            "eye drops"),
    ("o/e - retinal",          "examination (eye)"),
    # ── Ear drops ──
    ("ear drop",               "ear drops"),
    ("sofradex",               "ear drops"),
    ("gentisone",              "ear drops"),
    ("cerumol",                "ear drops"),
    ("otosporin",              "ear drops"),
    ("o/e - hearing tested",   "examination (ear)"),
    ("assessment for syringing of ear", "examination (ear)"),
    # ── Cold / allergy / cough ──
    ("loratadine",             "antihistamine"),
    ("desloratadine",          "antihistamine"),
    ("pseudoephedrine",        "decongestant"),
    ("xylometazoline",         "decongestant"),
    ("beconase",               "nasal steroid"),
    ("pholcodine",             "antitussive"),
    # ── Antimalarials / travel ──
    ("malarone",               "antimalarial"),
    ("mefloquine",             "antimalarial"),
    ("proguanil",              "antimalarial"),
    ("atovaquone",             "antimalarial"),
    ("history of foreign travel", "travel"),
    # ── Mucosal / local anaesthetic ──
    ("difflam",                "oral rinse"),
    ("xylocaine",              "local anaesthetic"),
    ("lidocaine 200mg",        "local anaesthetic"),
    ("lidocaine 200mg/20ml",   "local anaesthetic"),
    # ── Wound supplies ──
    ("ethilon suture",         "wound supplies"),
    ("suture",                 "wound supplies"),
    ("attention to surgical dressings", "wound care admin"),
    # ── Smoking cessation (KEEP via SMOKING category instead) ──
    # NOTE: handled in CATEGORY_KEYWORDS below, NOT excluded here.
    # ── Admin certificates and form codes ──
    ("med3 certificate",       "med3 certificate"),
    ("med5",                   "med5 certificate"),
    ("fp1001",                 "fp1001"),
    ("fp7b",                   "fp7b/fp8b"),
    ("[v]issue of medical",    "v-code admin"),
    ("[v]issue of repeat",     "v-code admin"),
    ("[v]attention to",        "v-code admin"),
    ("[v]examination of",      "v-code admin"),
    # ── Prescription / pharmacy admin ──
    ("prescription collected", "prescription pharmacy admin"),
    ("repeat prescription",    "prescription pharmacy admin"),
    ("therapeutic prescription", "prescription pharmacy admin"),
    ("drug discontinued",      "prescription pharmacy admin"),
    ("medication review",      "medication review admin"),
    # ── Examination / O/E generic ──
    ("o/e - vibration sense",  "examination (neuro generic)"),
    ("o/e - peripheral pulses", "examination (cv generic)"),
    ("o/e - smell",            "examination generic"),
    ("o/e - itchy rash",       "examination generic"),
    ("o/e - foot",             "examination (diabetic foot)"),
    ("o/e -",                  "examination generic"),
    ("monofilament",           "examination (diabetic foot)"),
    ("predicted peak flow",    "examination (respiratory)"),
    # ── Generic health-check / screening invites (surveillance bias) ──
    ("adult health examination", "health check generic"),
    ("adult screening nos",    "health check generic"),
    ("geriatric screening",    "health check generic"),
    ("general health examination", "health check generic"),
    ("chronic disease monitoring", "monitoring admin"),
    ("disease monitoring nos", "monitoring admin"),
    ("urinary disorder monitoring", "monitoring admin"),
    ("ophthalmological monitoring", "monitoring admin"),
    ("cardiac disease monitoring", "monitoring admin"),
    ("ischaemic heart disease screening", "screening admin"),
    ("screening for cardiovascular", "screening admin"),
    ("vascular disease risk assessment", "screening admin"),
    ("cardiovascular disease risk assessment", "screening admin"),
    ("first recall",           "screening admin"),
    ("bp screening",           "screening admin"),
    ("first letter",           "screening admin"),
    # ── Blood-sent admin (not a result, just an order) ──
    ("blood sent for haematology", "blood sent admin"),
    ("blood sent for chemistry", "blood sent admin"),
    ("blood sample sent to",   "blood sent admin"),
    ("blood chemistry nos",    "blood sent admin"),
    # ── COVID-19 surveillance (calendar-bias artifact) ──
    ("sars-cov-2 not detected", "covid surveillance"),
    ("sars-cov-2 absent",      "covid surveillance"),
    ("coronavirus.*not detected", "covid surveillance"),
    ("coronavirus.*absent",    "covid surveillance"),
    ("shielding of uninfected", "covid surveillance"),
    ("self-isolation",         "covid surveillance"),
    ("risk of exposure to communicable", "covid surveillance"),
    ("moderate risk category for developing complication", "covid risk category"),
    ("imm. services admin",    "vaccine admin"),
    # ── Minor symptoms (low predictive value, often surveillance bias) ──
    ("chesty cough",           "minor symptom"),
    ("cough symptom nos",      "minor symptom"),
    ("otalgia",                "minor symptom"),
    ("tinea of body",          "minor symptom"),
    ("pruritus of skin",       "minor symptom"),
    ("external haemorrhoids",  "minor symptom"),
    ("wry neck",               "minor symptom"),
    ("torticollis",            "minor symptom"),
    ("sprain",                 "minor symptom"),
    ("strain of rotator cuff", "minor symptom"),
    ("shoulder tendonitis",    "minor symptom"),
    ("complaining of a rash",  "minor symptom"),
    ("ear/nose/throat symptoms", "minor symptom"),
    # ── Generic admin / surveillance ──
    ("registration transferred", "registration admin"),
    ("outpatients last attended", "registration admin"),
    ("additional information", "admin generic"),
    ("non coded",              "admin generic"),
    ("signposting to gp",      "admin generic"),
    ("community medicine referral", "admin generic"),
    ("reason for onward referral", "admin generic"),
    ("dissent from disclosure", "admin generic"),
    ("employment milestones",  "admin generic"),
    ("exercise grading",       "admin generic"),
    ("referral for participation in clinical trial", "admin generic"),
    ("illness and disease",    "admin generic"),
    ("injury and poisoning",   "admin generic"),
    # (earrs/qmortality/qfrailty intentionally NOT in noise — handled as FRAILTY category)
    ("non.specific",           "admin generic"),
    # ── Surgical procedures unrelated to breast ──
    ("replacement of total knee joint", "unrelated procedure"),
    ("diagnostic proctoscopy", "unrelated procedure"),
    ("diagnostic arthroscopic examination of knee", "unrelated procedure"),
    ("painful right knee",     "unrelated msk"),
    # ── Bowel polyps (colorectal; not breast signal) ──
    ("tubular adenoma",        "colorectal polyp"),
    # ── Pre-existing pre-12mo cancer-marker (handle separately) ──
    # CEA is borderline — flagged as a category below, not removed.
    # ── Foot care / diabetic-foot non-specific (kept under DIABETES_MGT category instead) ──
    # NOTE: diabetic foot stuff goes to category, not noise — handled in CATEGORY_KEYWORDS.
    # ── More topical preparations (wave 2) ──
    ("betnovate",              "topical steroid"),
    ("betamethasone valerate", "topical steroid"),
    ("clobetasone",            "topical steroid"),
    ("hydrocortisone 0.5%",    "topical steroid"),
    ("fucidin",                "topical antibiotic"),
    ("aciclovir 5% cream",     "topical antiviral"),
    ("ketoprofen 2.5% gel",    "topical nsaid"),
    ("emulsifying ointment",   "topical emollient"),
    ("aquamax cream",          "topical emollient"),
    ("cetomacrogol cream",     "topical emollient"),
    ("fluorouracil 5% cream",  "topical (actinic keratosis)"),
    ("solaraze 3% gel",        "topical nsaid (actinic keratosis)"),
    ("xyloproct",              "topical anaesthetic ointment"),
    ("proctosedyl",            "topical (haemorrhoids)"),
    ("chlorhexidine",          "antiseptic (oral/skin)"),
    ("melolin dressing",       "wound dressing"),
    ("senile hyperkeratosis",  "minor dermatology"),
    ("skin - benign mole",     "minor dermatology"),
    ("symptoms of skin and integumentary", "minor dermatology"),
    ("symptoms of skin",       "minor dermatology"),
    # ── Nasal sprays (apart from already-covered) ──
    ("flixonase",              "nasal steroid"),
    ("nasonex",                "nasal steroid"),
    ("sterimar",               "saline nasal spray"),
    ("fluticasone furoate",    "nasal steroid"),
    # ── GI / IBS / laxatives / antiemetics / antispasmodics ──
    ("mebeverine",             "ibs antispasmodic"),
    ("hyoscine butylbromide",  "antispasmodic"),
    ("movicol",                "laxative"),
    ("gaviscon",               "antacid otc"),
    ("metoclopramide",         "antiemetic"),
    ("prochlorperazine",       "antiemetic"),
    ("domperidone",            "antiemetic"),
    # ── Other meds (treated as noise — not breast-relevant) ──
    ("allopurinol",            "gout treatment"),
    ("quinine sulfate",        "leg cramps (elderly)"),
    ("flecainide",             "antiarrhythmic"),
    ("sotalol",                "antiarrhythmic"),
    ("nitrolingual",           "angina (GTN)"),
    # ── Vitamin D supplements (KEEP in VITAMIN_D category instead — handled below) ──
    # ── More healthcare utilization / admin ──
    ("reviewed at hospital",   "healthcare utilization"),
    ("inpatient stay",         "healthcare utilization"),
    ("clinic a monitoring",    "healthcare utilization"),
    ("consultation for minor illness", "minor illness encounter"),
    ("emergency consultation note", "encounter admin"),
    ("usual general practitioner", "encounter admin"),
    ("patient encounter procedure", "encounter admin"),
    ("report of clinical encounter", "encounter admin"),
    ("near-patient testing patient invitation", "encounter admin"),
    ("gp out of hours service", "encounter admin"),
    ("emergency consultation", "encounter admin"),
    ("provision of advice, assessment or treatment delayed", "covid surveillance"),
    ("provision of advice, assessment or treatment limited", "covid surveillance"),
    ("disease caused by 2019-ncov", "covid surveillance"),
    ("disease caused by severe acute respiratory syndrome", "covid surveillance"),
    ("wuhan",                  "covid surveillance"),
    ("low risk category for developing complication", "covid risk category"),
    ("covid-19 excluded by laboratory test", "covid surveillance"),
    ("provision of local nhs service information", "admin generic"),
    ("self-referral to accident and emergency", "encounter admin"),
    ("referred to chest physician", "specialist referral admin"),
    ("referral for other investigation", "specialist referral admin"),
    ("first episode",          "admin generic"),
    ("clinical trial",         "admin generic"),
    ("informed consent obtained to a nursing procedure", "consent admin"),
    ("consent to share demographic information", "consent admin"),
    ("dissent from secondary use", "consent admin"),
    ("patient medical record envelope", "registration admin"),
    ("prescription sent to patient", "prescription admin"),
    ("near-patient testing",   "encounter admin"),
    ("patient activation measure", "frailty score admin"),
    ("further miscellaneous scales", "admin generic"),
    ("goal achievement finding", "admin generic"),
    ("mobility - social functioning", "admin generic"),
    ("clinic a monitoring 1st letter", "monitoring admin"),
    # ── Minor symptoms (wave 2) ──
    ("complaining of a headache", "minor symptom"),
    ("[d]bloating",            "minor symptom"),
    ("general symptoms",       "minor symptom"),
    ("nose symptoms",          "minor symptom"),
    ("faeces appearance",      "minor symptom"),
    ("absence of sensation",   "minor symptom"),
    ("computed tomography result abnormal", "non-specific imaging"),
    ("further miscellaneous",  "admin generic"),
    # ── Surgery / procedures (minor, non-breast) ──
    ("minor surgery done",     "minor procedure"),
    ("other specified operations on coronary artery", "cv procedure"),
    ("cardiovascular angiography", "cv procedure"),
    # ── Test orders without results ──
    ("laboratory test requested", "blood sent admin"),
    ("haematology test requested", "blood sent admin"),
    ("test request : serum folate", "test request admin"),
    # ── Smoking-cessation services (already a category for varenicline) ──
    # No-history-of (negative info — KEEP) handled by NOT matching here
    # ── (qmortality/qfrailty intentionally NOT in noise — handled as FRAILTY category) ──
    # ── Antihistamines (no breast signal) ──
    ("cetirizine",             "antihistamine"),
    ("piriton",                "antihistamine"),
    ("chlorphenamine",         "antihistamine"),
    # ── Topical/minor meds (wave 3) ──
    ("fenbid 5% gel",          "topical nsaid"),
    ("simple linctus",         "antitussive"),
    ("loperamide",             "anti-diarrhoeal"),
    ("lactulose",              "laxative"),
    ("otomize",                "ear spray"),
    ("avamys",                 "nasal steroid"),
    ("depo-medrone",           "intra-articular steroid injection"),
    ("lidocaine 40mg",         "local anaesthetic"),
    ("lidocaine 2%",           "local anaesthetic"),
    ("lidocaine solution for injection", "local anaesthetic"),
    # ── Admin / encounter (wave 3) ──
    ("advised to contact nhs", "admin generic"),
    ("med3 - doctor",          "med3 certificate"),
    ("med3 doctor",            "med3 certificate"),
    ("well man health examination", "health check generic"),
    ("geriatric health examination", "health check generic"),
    ("patient not understood", "admin generic"),
    ("general pathology nos",  "admin generic"),
    ("counselling about disease", "admin generic"),
    ("report status",          "admin generic"),
    ("expert patient programme", "admin generic"),
    ("[rfc] reason for care",  "admin generic"),
    ("certificates administration", "admin generic"),
    ("optician",               "admin generic"),
    ("referral to clinical assessment service", "specialist referral admin"),
    ("referral to plastic surgeon", "specialist referral admin"),
    ("referral to surgeon",    "specialist referral admin"),
    ("repeat treatment nos",   "admin generic"),
    ("drug treatment",         "admin generic"),
    ("patient registration",   "registration admin"),
    ("patient given written advice", "admin generic"),
    ("normal mental state",    "admin generic"),
    ("acupuncture",            "admin generic"),
    ("counselling about disease", "admin generic"),
    ("education for care planning", "admin generic"),
    ("social assessment",      "admin generic"),
    ("in-house physiotherapy", "admin generic"),
    ("telephone in house",     "admin generic"),
    ("clinic b monitoring",    "monitoring admin"),
    ("clinic c monitoring",    "monitoring admin"),
    ("[v]personal history of drug allergy", "admin generic"),
    ("[v]pseudophakia",        "admin generic"),
    ("[v]",                    "v-code admin"),    # catch-all
    ("pre/post-operative procedures", "admin generic"),
    ("recurrence of problem",  "admin generic"),
    ("emis covid-19 care pathway", "covid surveillance"),
    ("patient status determination", "admin generic"),
    ("direct microscopy",      "lab generic"),
    ("culture for fungi",      "lab generic"),
    ("international normalised ratio requested", "test request admin"),
    ("haematology procedure",  "lab generic"),
    ("haematology test",       "lab generic"),
    ("blood chemistry",        "lab generic"),
    # (test request : esr / serum electrolytes intentionally NOT noise —
    #  handled as INFLAMMATORY / ELECTROLYTES categories)
    ("test request : bone profile", "test request admin"),
    # (test request alone is too broad to noise — falls through; specific
    #  "test request : <topic>" entries are handled by categories or by the
    #  specific noise entries below)
    ("test request : vitamin", "test request admin"),
    ("test request : b12",     "test request admin"),
    ("test request : urine",   "test request admin"),
    ("resource used",          "admin generic"),
    ("nhs prescription",       "admin generic"),
    ("chronic disease - drug compliance check", "admin generic"),
    ("patient activation",     "admin generic"),
    ("near-patient testing",   "admin generic"),
    ("written advice",         "admin generic"),
    ("ear syringing",          "ear procedure"),
    # ── Minor symptoms (wave 3) ──
    ("vomiting",               "minor symptom"),
    ("complaining of itching", "minor symptom"),
    ("sticky eye",             "minor symptom"),
    ("complaining of wax in ear", "minor symptom"),
    ("complaining of pain in hallux", "minor symptom"),
    ("complaining of postnasal drip", "minor symptom"),
    # (night cough absent/present → RESPIRATORY category, not noise)
    ("no general symptom",     "minor symptom"),
    ("pyrexia symptoms",       "minor symptom"),
    ("memory loss",            "minor symptom"),
    ("cervical spondylosis",   "minor msk"),
    ("diaphragmatic hernia",   "minor symptom"),
    ("lentigo",                "minor dermatology"),
    ("vitreous detachment",    "minor eye"),
    ("repair of bilateral inguinal hernias", "minor surgery"),
    ("grade b moderately active", "admin generic"),
    ("dyspnoea grade",         "admin generic"),
    ("mrc breathlessness",     "admin generic"),
    ("non-urgent hospital admission", "encounter admin"),
    ("plain x-ray pelvis",     "imaging generic"),
    ("plain x-ray",            "imaging generic"),
    ("mumps igg level",        "lab serology"),
    ("diagnostic fibreoptic gastroscopy", "gi procedure"),
    ("possibly eligible for participation", "research admin"),
    # ── Wave 5: long-tail mop-up ──
    # Ethnicity / demographic admin (separate from clinical signal)
    ("ethnic category",        "ethnicity admin"),
    ("black caribbean",        "ethnicity admin"),
    ("white british",          "ethnicity admin"),
    ("british or mixed british", "ethnicity admin"),
    ("irish - ethnic",         "ethnicity admin"),
    ("caribbean - ethnic",     "ethnicity admin"),
    ("speaks english",         "language admin"),
    # Healthcare-utilization admin (more)
    ("seen by deputising doctor", "encounter admin"),
    ("seen by specialty doctor", "encounter admin"),
    ("seen by community navigator", "encounter admin"),
    ("seen in hospital outpatient", "encounter admin"),
    ("hospital outpatient report", "encounter admin"),
    ("accident & emergency",   "encounter admin"),
    ("nurse telephone triage", "encounter admin"),
    ("nursing procedure",      "admin generic"),
    ("administrative procedure", "admin generic"),
    # ("administration" alone is too broad — matches valid codes)
    ("research administration", "admin generic"),
    ("extended hours access enhanced services administration", "admin generic"),
    ("prevention/screening administration", "admin generic"),
    ("computer record print",  "admin generic"),
    ("referral document",      "admin generic"),
    ("hospital outpatient",    "encounter admin"),
    ("lloyd george",           "admin generic"),
    ("transfer-degraded record", "admin generic"),
    ("preferred place of care", "admin generic"),
    ("preferred method of contact", "admin generic"),
    ("review of supportive care plan", "admin generic"),
    ("on integrated care pathway", "admin generic"),
    ("admission avoidance care", "admin generic"),
    ("provision of medical equipment", "admin generic"),
    ("provision of medical certificate", "admin generic"),
    ("patient examined",       "examination generic"),
    ("patient given verbal advice", "admin generic"),
    ("patient given written advice", "admin generic"),
    ("test result to patient", "admin generic"),
    ("test result given",      "admin generic"),
    ("discussed with patient", "admin generic"),
    ("declined consent for",   "consent admin"),
    ("consent given to participate", "consent admin"),
    ("consent for operation given", "consent admin"),
    ("informed consent obtained", "consent admin"),
    ("consent to share",       "consent admin"),
    ("patient has online access", "admin generic"),
    ("repeated prescription",  "prescription pharmacy admin"),
    ("prescription not collected", "prescription pharmacy admin"),
    ("batch prescription issued", "prescription pharmacy admin"),
    ("prescription issued for patient on holiday", "prescription pharmacy admin"),
    ("medication discontinued", "prescription pharmacy admin"),
    ("medication change to generic", "prescription pharmacy admin"),
    ("drug over usage checked", "prescription pharmacy admin"),
    ("able to use medication", "prescription pharmacy admin"),
    ("deleted from recall",    "admin generic"),
    ("refer for imaging nos",  "admin generic"),
    ("refer for imaging",      "admin generic"),
    ("x-ray report received",  "admin generic"),
    ("x-rays$",                "admin generic"),
    ("plain x-ray pelvis",     "imaging generic"),
    ("ct scan brain",          "imaging (non-breast)"),
    ("nuclear magn.reson. abnormal", "imaging generic"),
    ("radiology result",       "admin generic"),
    ("hospital anxiety and depression scale", "psych screening admin"),
    ("sample obtained",        "admin generic"),
    ("sample microscopy",      "lab generic"),
    ("collection of blood specimen", "blood sent admin"),
    ("fasting blood sample taken", "blood sent admin"),
    ("histopathology test",    "lab generic"),
    ("bacterial culture",      "lab generic"),
    ("microscopy$",            "lab generic"),
    ("palpation - action",     "examination generic"),
    ("cryotherapy",            "minor procedure"),
    ("removal other specified repair material from skin", "minor procedure"),
    ("removal of repair material", "minor procedure"),
    ("dressing of skin or wound", "wound care admin"),
    ("wound repair review",    "wound care admin"),
    ("therapeutic lumbar epidural injection", "pain procedure"),
    ("injection of steroid into subcutaneous", "minor procedure"),
    ("traumatic and/or non-traumatic injury", "injury admin"),
    ("superficial injury",     "injury admin"),
    # Demographics / risk admin
    ("has firearm certificate", "admin generic"),
    ("spouse deceased",        "social admin"),
    ("current non recreational drug user", "social admin"),
    # Minor symptoms (additional wave)
    # (night cough present → RESPIRATORY category)
    ("snoring symptoms",       "minor symptom"),
    ("snoring symptom",        "minor symptom"),
    ("tongue symptoms",        "minor symptom"),
    ("throat symptom",         "minor symptom"),
    ("ear symptom",            "minor symptom"),
    ("earache symptoms",       "minor symptom"),
    ("eye symptom",            "minor symptom"),
    ("ent symptoms",           "minor symptom"),
    ("indigestion symptoms",   "minor symptom"),
    ("diarrhoea symptoms",     "minor symptom"),
    ("temperature symptoms",   "minor symptom"),
    ("temperature normal",     "examination generic"),
    ("feels hot/feverish",     "minor symptom"),
    ("tooth symptoms",         "minor symptom"),
    ("pain in arm",            "minor symptom"),
    ("c/o: a pain",            "minor symptom"),
    ("visual symptoms",        "minor symptom"),
    ("itchy eye",              "minor eye"),
    ("complaining of catarrh", "minor symptom"),
    ("complaining of paraesthesia", "minor symptom"),
    ("complaining of pain in", "minor symptom"),
    ("perennial allergic rhinitis", "minor allergy"),
    ("chronic rhinitis",       "minor allergy"),
    ("hay fever",              "minor allergy"),
    ("acute maxillary sinusitis", "minor infection"),
    ("nonspecific abdominal pain", "minor symptom"),
    ("pruritus ani",           "minor symptom"),
    ("nose bleed",             "minor symptom"),
    ("bruising symptom",       "minor symptom"),
    ("seizure free",           "admin generic"),
    ("at risk of dementia",    "frailty admin"),
    ("at risk of emergency hospital", "frailty admin"),
    ("pts at risk of rehospitalisation", "frailty admin"),
    ("parr-30",                "frailty admin"),
    ("gp assessment of cognition", "frailty admin"),
    ("good compliance with diabetic diet", "diabetes admin"),
    ("access to online patient diabetes education", "diabetes admin"),
    ("has seen dietitian",     "diabetes admin"),
    ("retinopathy follow up",  "diabetes admin"),
    ("retinal photography",    "diabetes admin"),
    # (fundoscopy normal → DIABETES_MGT; ophthalmoscopy → kept as exam noise)
    ("ophthalmoscopy",         "examination (eye)"),
    # Other minor dermatology
    ("bowen's disease",        "minor dermatology"),
    ("onychomycosis",          "minor dermatology"),
    ("inflammatory dermatosis", "minor dermatology"),
    ("skin and subcutaneous tissue disease", "minor dermatology"),
    ("ingrowing nail",         "minor podiatry"),
    # H. pylori / GI (not breast)
    ("helicobacter",           "h pylori admin"),
    ("clo test positive",      "h pylori admin"),
    ("clo test",               "h pylori admin"),
    # (occult blood not detected in faeces — kept for FOBT screening signal)
    # Travel admin
    ("travel abroad",          "travel"),
    ("travel destination",     "travel"),
    ("antimalarial drug prophylaxis", "antimalarial"),
    # ENT minor
    ("ear wax",                "minor symptom"),
    ("wax in ear canal",       "minor symptom"),
    ("washout of ear",         "ear procedure"),
    ("sensorineural hearing loss", "minor symptom"),
    ("unspecified infective otitis externa", "minor symptom"),
    ("acute upper respiratory infections", "minor uri"),
    ("acute respiratory infections$", "minor uri"),
    ("other acute upper respiratory", "minor uri"),
    # Frailty / participant admin
    ("participant in research study", "research admin"),
    ("consent given to participate in research", "research admin"),
    ("invitation to participate in research", "research admin"),
    ("health coach",           "admin generic"),
    ("at risk of emergency hospital admission", "frailty admin"),
    # Beta blocker contraindicated, Vital capacity test, Generic — handled by category
    # IBS / GI minor
    ("irritable colon",        "ibs minor"),
    ("irritable bowel syndrome", "ibs minor"),
    # Test request admin (more)
    ("test request : serum ferritin", "test request admin"),
    ("h/o long term condition", "h/o admin"),
    # Image / document admin
    ("image \\(document\\)",   "admin generic"),
    ("radiology result",       "admin generic"),
    ("risk information",       "admin generic"),
    # Misc
    ("supraspinatus tendonitis", "minor msk"),
    ("rotator cuff",           "minor msk"),
    ("ent$",                   "specialist referral admin"),
    ("nuclear magn.reson.",    "imaging generic"),
    # (forced expired volume / vital capacity intentionally NOT noise — handled as RESPIRATORY)
    # ── Wave 6: past-tense referrals + final mop-up ──
    ("referred to physician",  "specialist referral admin"),
    ("referral to physician",  "specialist referral admin"),
    ("referred to rheumatolog", "specialist referral admin"),
    ("referral to rheumatolog", "specialist referral admin"),
    ("referred to plastic surgeon", "specialist referral admin"),
    ("referred to counsellor", "specialist referral admin"),
    ("referral to counsellor", "specialist referral admin"),
    ("referred to orthopaed",  "specialist referral admin"),
    ("referral to orthopaed",  "specialist referral admin"),
    ("referred to chest physician", "specialist referral admin"),
    ("referral to chest physician", "specialist referral admin"),
    # Non-breast specialist procedures / referrals
    ("referral to dermatolog", "specialist referral admin"),
    ("referred to dermatolog", "specialist referral admin"),
    ("seen in dermatology",    "specialist referral admin"),
    ("diagnostic nasendoscopy", "minor procedure (ent)"),
    ("excision biopsy of skin", "minor procedure (skin)"),
    # Generic antibiotics — not breast-specific (UTI antibiotics handled
    # separately via UTI_ANTIBIOTICS category)
    ("amoxicillin",            "generic antibiotic"),
    ("amoxil",                 "generic antibiotic"),
    ("clarithromycin",         "generic antibiotic"),
    ("erythromycin",           "generic antibiotic"),
    ("flucloxacillin",         "generic antibiotic"),
    ("oxytetracycline",        "generic antibiotic"),
    ("co-amoxiclav",           "generic antibiotic"),
    ("cefalexin",              "generic antibiotic"),
    ("doxycycline",            "generic antibiotic"),
    # Topical NSAID — different from systemic NSAIDs in PAIN_OTC
    ("ibuprofen 5% gel",       "topical nsaid"),
    ("ibuprofen gel",          "topical nsaid"),
    # Asthma admin / questionnaire codes (chronic-care monitoring admin)
    ("asthma monitoring call", "asthma monitoring admin"),
    ("asthma monitoring check done", "asthma monitoring admin"),
    ("royal college of physicians asthma", "asthma monitoring admin"),
    ("asthma review using royal college", "asthma monitoring admin"),
    ("asthma management",      "asthma monitoring admin"),
    # Cardiovascular invite/monitoring admin (already partly covered)
    ("high risk monitoring invitation letter", "monitoring admin"),
    ("coronary heart disease monitoring administration", "monitoring admin"),
    # Lifestyle admin
    ("patient-initiated diet", "lifestyle admin"),
    # ── Wave 8 — aggressive admin/severity cleanup ──
    # Asthma severity / symptom-frequency codes (redundant with inhaler meds + dx)
    ("asthma causes daytime symptoms", "asthma severity admin"),
    ("asthma causes night time symptoms", "asthma severity admin"),
    ("asthma causing night waking", "asthma severity admin"),
    ("asthma daytime symptoms",   "asthma severity admin"),
    ("asthma disturbing sleep",   "asthma severity admin"),
    ("asthma disturbs sleep weekly", "asthma severity admin"),
    ("asthma limiting activities", "asthma severity admin"),
    ("asthma limits walking",     "asthma severity admin"),
    ("asthma never causes daytime", "asthma severity admin"),
    ("asthma never causes night", "asthma severity admin"),
    ("asthma never disturbs sleep", "asthma severity admin"),
    ("asthma never restricts exercise", "asthma severity admin"),
    ("asthma not disturbing sleep", "asthma severity admin"),
    ("asthma not limiting activities", "asthma severity admin"),
    ("asthma sometimes restricts", "asthma severity admin"),
    ("asthma monitoring",         "asthma monitoring admin"),
    ("bronchodilators used a maximum", "asthma admin"),
    ("bronchodilators used more than", "asthma admin"),
    ("inhaler technique observed", "asthma admin"),
    ("unable to perform spirometry", "asthma admin"),
    ("respiratory flow rate measured", "respiratory admin"),
    ("respiratory system diseases nos", "respiratory admin generic"),
    # Smoking-cessation admin (keep direct smoker-status codes)
    ("attends stop smoking",      "smoking cessation admin"),
    ("smoking cessation milestones", "smoking cessation admin"),
    ("monitoring of smoking cessation therapy", "smoking cessation admin"),
    ("negotiated date for cessation of smoking", "smoking cessation admin"),
    ("smoking cessation programme start date", "smoking cessation admin"),
    ("smoking status at 4 weeks", "smoking cessation admin"),
    # CV monitoring admin (keep disease dx + lab tests)
    ("cardiovascular disease monitoring", "cv monitoring admin"),
    ("chd monitoring",            "cv monitoring admin"),
    ("stroke monitoring",         "cv monitoring admin"),
    ("follow-up cardiac assessment", "cv monitoring admin"),
    # MSK generic / admin
    ("musculoskeletal and connective tissue disorder", "msk generic"),
    ("musculoskeletal symptom",   "msk generic"),
    ("referral to back pain clinic", "specialist referral admin"),
    ("rheumatoid arthritis annual review", "ra admin"),
    # Adverse reaction admin
    ("adverse reaction caused by", "adverse event admin"),
    # Lifestyle counselling admin
    ("brief intervention for physical activity", "lifestyle admin"),
    ("dietary advice",            "lifestyle admin"),
    ("general practice physical activity questionnaire", "lifestyle admin"),
    ("lifestyle education",       "lifestyle admin"),
    # Family-history admin (just saying FH was taken)
    ("family history taken",      "fh admin"),
    ("family history with explicit context", "fh admin"),
    # Peripheral pulse exam findings (generic CV exam)
    ("on examination - left femoral pulse",        "peripheral exam admin"),
    ("on examination - left popliteal pulse",      "peripheral exam admin"),
    ("on examination - left posterior tibial pulse", "peripheral exam admin"),
    ("on examination - right femoral pulse",       "peripheral exam admin"),
    ("on examination - right posterior tibial pulse", "peripheral exam admin"),
    ("on examination - peripheral pulses",         "peripheral exam admin"),
    # Other generic exam findings
    ("c/o: a swelling",           "minor symptom"),
    # Generic lab descriptor / non-specific lab terms
    ("serum appearance",       "lab generic"),
    ("blood chemistry$",       "lab generic"),
    ("haematology test$",      "lab generic"),
    # Examination — peripheral pulses generic
    ("right-leg pulses",       "examination (cv generic)"),
    ("left leg pulses",        "examination (cv generic)"),
    ("dorsalis pedis",         "examination (cv generic)"),
    # Urine generic / non-specific
    ("urine sample for organism", "urine sample admin"),
    # (urine screening normal → URINE_MARKERS category, not noise)
    # Non-breast imaging
    ("ultrasound of kidney",   "imaging (non-breast)"),
    # Surgery / procedures
    ("total prosthetic replacement of hip", "unrelated procedure"),
    ("other excision of lesion of skin", "minor procedure"),
    # COVID
    ("covid-19 confirmed by laboratory", "covid surveillance"),
    # Admin
    ("questionable if patient telephone number", "admin generic"),
    ("emergency appointment",  "encounter admin"),
    ("postal invite to screening", "screening admin"),
    ("choice and booking enhanced services", "admin generic"),
    ("procedure refused",      "admin generic"),
    ("personal care plan completed", "admin generic"),
    ("over-the-counter medication education", "admin generic"),
    # (anchored "administration$" was a no-op via substring matching)
    # GI minor
    # (reflux oesophagitis / oesophagitis → GASTRIC_ACID category instead of noise)
    # Other minor
    ("night cramps",           "minor symptom"),
    ("assessment for dementia", "frailty admin"),
    ("declined consent for short message service", "consent admin"),
    # COVID full-name variant
    ("severe acute respiratory syndrome coronavirus 2 not detected", "covid surveillance"),
    ("severe acute respiratory syndrome coronavirus 2 absent", "covid surveillance"),
    # Minor / admin (final wave)
    ("solar keratosis",        "minor dermatology"),
    ("sebaceous cyst",         "minor dermatology"),
    ("tennis elbow",           "minor msk"),
    ("influenza-like illness", "minor uri"),
    ("upper respiratory infection", "minor uri"),
    ("allergic rhinitis",      "minor allergy"),
    ("throat soreness",        "minor symptom"),
    ("skin lesion",            "minor dermatology"),
    # (standard chest x-ray normal → RESPIRATORY)
    ("married",                "demographic admin"),
    ("motor car driver",       "demographic admin"),
    ("other occupations",      "demographic admin"),
    ("patient's condition improved", "admin generic"),
    ("patient review$",        "admin generic"),
    ("patient review",         "admin generic"),
    ("patient offered choice of provider", "admin generic"),
    ("post-surgical wound care", "wound care admin"),
    ("provision of written information", "admin generic"),
    ("injection of therapeutic substance into joint", "minor procedure"),
    ("injection given",        "minor procedure"),
    ("referral to general surgical service", "specialist referral admin"),
    ("referral$",              "admin generic"),
    ("appointment$",           "encounter admin"),
    ("prescription sent to pharmacy", "prescription pharmacy admin"),
    # Prednisolone gastro-resistant (low-dose; not specific to anything systemic)
    ("prednisolone 2.5mg",     "low-dose steroid"),
    # Minor ENT / respiratory infections
    ("myalgia",                "minor symptom"),
    ("acute pharyngitis",      "minor symptom"),
    ("acute sinusitis",        "minor symptom"),
    ("nasal obstruction",      "minor symptom"),
    ("polyp of nasal cavity",  "minor symptom"),
    # Admin (more)
    ("treatment plan given",   "admin generic"),
    ("nhs health check programme", "screening admin"),
    ("nhs health check",       "screening admin"),
    ("scanned document",       "admin generic"),
    ("histopathology report received", "admin generic"),
    ("new registration check", "registration admin"),
    ("had a discussion with patient", "admin generic"),
    ("postoperative monitoring", "admin generic"),
    ("primary repair of inguinal hernia", "minor surgery"),
    ("resp. system examined - nad", "examination generic"),
    ("seen in plastic surgery clinic", "encounter admin"),
    ("seen in haematology clinic", "encounter admin"),
    ("fasting sample",         "blood sent admin"),
    ("diabetes monitoring administration", "monitoring admin"),
    ("diagnostic procedure",   "admin generic"),
    ("foreign travel education", "travel"),
    ("over the counter aspirin therapy", "antiplatelet otc"),  # technically antiplatelet but flagged
    ("venesection",            "procedure (general)"),
    # ENT minor
    ("tinea",                  "minor dermatology"),

    ("[d]nervous system symptoms", "minor symptom"),
    ("[d]bloating",            "minor symptom"),
    ("[d]",                    "d-code admin"),
    # (Result/Microscopy as standalone codes — left to fall through to skipped CSV)
    ("eyelid inflammation",    "minor eye"),
    ("skin lesion$",           "minor dermatology"),
    ("bill/fee paid",          "admin generic"),
    ("under care of gp",       "admin generic"),
    ("patient encounter administration", "admin generic"),
    ("repeat dispensing",      "prescription pharmacy admin"),
    ("clinical report documentation", "admin generic"),
    ("patient's condition the same", "admin generic"),
    ("^ent$",                  "specialist referral admin"),
    ("non-european travel",    "travel"),
    ("european travel",        "travel"),
    ("benign paroxysmal positional vertigo", "minor symptom"),
    ("benign paroxysmal positional nystagmus", "minor symptom"),
    ("nystagmus",              "minor eye"),
    ("shotgun application certificate", "admin generic"),
    ("firearms certificate",   "admin generic"),
    ("marital state",          "demographic admin"),
    ("marital status",         "demographic admin"),
    ("encounter by computer",  "admin generic"),
    ("laboratory procedures",  "lab generic"),
    ("private referral",       "specialist referral admin"),
    ("referral to practice nurse", "specialist referral admin"),
    ("referral to nurse",      "specialist referral admin"),
    ("able to use medication", "admin generic"),
    ("excision of dupuytren",  "minor surgery"),
    ("dupuytren",              "minor msk"),
    ("eye trauma",             "minor eye"),
    ("conjunctivitis",         "minor eye"),
    ("epistaxis",              "minor symptom"),
    ("nose bleed",             "minor symptom"),
    ("dermatitis",             "minor dermatology"),
    ("eczema",                 "minor dermatology"),
    ("psoriasis",              "minor dermatology"),
    ("acne",                   "minor dermatology"),
    ("wart",                   "minor dermatology"),
    ("verruca",                "minor dermatology"),
    ("plantar fasciitis",      "minor msk"),
    ("bunion",                 "minor msk"),
    ("ingrowing toenail",      "minor podiatry"),
    ("athlete's foot",         "minor dermatology"),
    # Test-request: explicit non-clinical orders kept in noise; the rest fall
    # through to category matching (e.g., test request : lipids → LIPIDS).
    # ── Consent / admin (existing) ──
    ("implied consent",        "consent"),
    ("refused consent",        "consent"),
    ("consent for core",       "consent"),
    ("summary care record",    "summary care record"),
    # Vaccines / COVID admin
    ("comirnaty",              "comirnaty"),
    ("spikevax",               "spikevax"),
    ("vaxzevria",              "vaxzevria"),
    ("covid-19 vaccin",        "covid-19 vaccin"),
    ("covid 19 vaccin",        "covid-19 vaccin"),
    ("fluad",                  "fluad"),
    ("fluarix",                "fluarix"),
    ("enzira",                 "enzira"),
    ("pneumovax",              "pneumococcal vaccin"),
    ("pneumococcal vaccin",    "pneumococcal vaccin"),
    ("influenza vaccin",       "influenza vaccin"),
    ("vaccin",                 "vaccin"),
    ("immunisation",           "immunisation"),
    ("immunization",           "immunisation"),
    ("sars-cov-2",             "sars-cov-2"),
    ("high risk category for developing complication", "covid risk category"),
    # Communications
    ("letter sent",            "letter sent"),
    ("letter received",        "letter received"),
    ("letter from",            "letter from"),
    ("letter to",              "letter to"),
    ("referral letter",        "referral letter"),
    ("fax sent",               "fax sent"),
    ("fax received",           "fax received"),
    ("sms message",            "sms"),
    ("email",                  "email"),
    # Recall / scheduling
    ("did not attend",         "did not attend"),
    (" dna ",                  "dna"),
    ("appointment reminder",   "appointment reminder"),
    ("recall code",            "recall"),
    ("health check invitation", "health check invitation"),
    ("invitation for",         "invitation"),
    # Health-education / generic
    ("health education",       "health education"),
    ("language",               "language"),
    ("interpreter",            "interpreter"),
    ("ethnicity",              "ethnicity"),
    ("phone call",             "phone call"),
    ("telephone consult",      "telephone consult"),
    ("advance care plan",      "advance care plan"),
    ("fp1001",                 "fp1001"),
    ("med5 issued",            "med5 issued"),
    ("general medical",        "general medical"),
    # Pure surveillance / generic test markers
    ("frequency of encounter", "frequency of encounter"),
    ("qadmissions",            "qadmissions"),
    ("qrisk",                  "qrisk"),
    ("framingham",             "framingham"),
    ("test - laboratory",      "test - laboratory generic"),
    ("radiology/physics in medicine", "radiology generic"),
    ("ehr attachment",         "ehr attachment"),
    ("patient fp7b",           "fp7b/fp8b"),
    ("blood pressure reading", "blood pressure read"),
    ("alert",                  "alert"),    # generic 'Alert' code
    ("attachment for medical notes", "attachment for medical notes"),
]

# Suggested-category keyword matcher (KEEP rows only).
# Two tiers:
#   Tier 1 — breast-specific clinical categories (mammography, biopsy, etc. — stub for clinical team to fill in)
#   Tier 2 — broader categories observed in pre-anchor data (cancer-agnostic)
#            (METABOLIC, CV_RISK, STATIN, etc.)
CATEGORY_KEYWORDS = [
    # ── Tier 1: breast-specific clinical categories (STUB — fill in via curation) ───
    # The prostate template's Tier 1 (PSA / DRE / LUTS / urinary / erectile) has been
    # stripped because none apply to breast cancer. Clinical team to populate the
    # placeholders below with breast-relevant regex patterns.
    ("MAMMOGRAPHY",          []),   # e.g. "screening mammogram", "mammography abnormal"
    ("BREAST_IMAGING",       []),   # e.g. "breast ultrasound", "breast mri"
    ("BREAST_BIOPSY",        []),   # e.g. "breast biopsy", "core needle", "fine needle aspiration"
    ("BREAST_CONDITIONS",    []),   # e.g. "fibroadenoma", "mastitis", "ductal hyperplasia", "breast cyst"
    ("BREAST_SYMPTOMS",      []),   # e.g. "breast lump", "nipple discharge", "breast pain", "skin dimpling"
    ("HORMONE_RECEPTORS",    []),   # e.g. "oestrogen receptor", "progesterone receptor", "her2", "\\ber\\+\\b"
    ("BRCA",                 []),   # e.g. "brca1", "brca2", "hereditary breast cancer"
    ("BREAST_TUMOR_MARKERS", []),   # e.g. "ca 15-3", "ca 27.29", "carcinoembryonic"
    ("ALP_BONE_MARKER",      [r"alkaline phosphatase", r"\balp\b",
                              r"bone isoenzyme"]),
    ("CALCIUM",              [r"\bcalcium\b"]),
    ("ELECTROLYTES",         [r"\bsodium\b", r"\bpotassium\b",
                              r"urea and electrolytes", r"\bu&e\b"]),
    ("RENAL_FUNCTION",       [r"creatinine", r"glomerular filtration",
                              r"\begfr\b", r"\burea\b"]),
    ("LIVER_FUNCTION",       [r"bilirubin", r"\balt\b", r"\bast\b",
                              r"alanine amino", r"aspartate amino",
                              r"\bggt\b", r"gamma.glut", r"liver function"]),
    ("ALBUMIN_PROTEIN",      [r"albumin", r"globulin", r"total protein"]),
    ("FBC_HAEMATOLOGY",      [r"haemoglobin", r"haematocrit",
                              r"platelet count", r"white blood cell",
                              r"lymphocyte count", r"eosinophil count",
                              r"neutrophil count", r"monocyte count",
                              r"basophil count", r"red blood cell",
                              r"erythrocyte sedimentation", r"\bmcv\b",
                              r"\bmch\b", r"\bmchc\b", r"\bfbc\b",
                              r"full blood count"]),
    ("INFLAMMATORY",         [r"c reactive protein", r"\bcrp\b"]),
    ("HORMONAL",             []),  # STUB — populate with breast-relevant hormones (oestradiol, progesterone, LH, FSH, etc.)
    ("URINE_MARKERS",        [r"urine dipstick", r"urinalysis",
                              r"urine.*microscop", r"urine.*leucocyte",
                              r"urine.*protein", r"urine.*glucose",
                              r"urine.*specific gravity", r"urine.*nitrite"]),
    ("VITAMIN_D",            [r"vitamin d", r"25.hydroxyvitamin"]),
    ("COAGULATION",          [r"\binr\b", r"prothrombin", r"coagulation"]),
    ("CONSTITUTIONAL",       [r"tiredness", r"fatigue", r"tired all the time",
                              r"malaise", r"lethargy", r"night sweats"]),
    ("WEIGHT",               [r"weight loss", r"unintentional weight",
                              r"weight symptom"]),
    ("BMI",                  [r"body mass index", r"\bbmi\b"]),
    ("BODY_WEIGHT",          [r"body weight"]),
    ("SYSTOLIC_BP",          [r"systolic"]),
    ("DIASTOLIC_BP",         [r"diastolic"]),
    ("HEART_RATE",           [r"heart rate", r"pulse rate", r"\bpulse\b"]),
    ("HYPERTENSION",         [r"hypertension", r"high blood pressure"]),
    ("FAMILY_HISTORY",       [r"family history", r"\bfh:?\b"]),
    ("SMOKING",              [r"\bsmok", r"tobacco", r"cigarette",
                              r"ex.smoker"]),

    # ── Tier 2: new broader categories surfacing in pre-12mo data ──
    ("METABOLIC",            [r"fasting.*glucose", r"fasting blood sugar",
                              r"\bhba1c\b", r"glycated h.*moglobin",
                              r"plasma glucose", r"random glucose",
                              r"oral glucose tolerance", r"\bogtt\b",
                              r"impaired fasting"]),
    ("LIPIDS",               [r"\bhdl\b", r"\bldl\b", r"cholesterol",
                              r"triglycerid", r"lipid profile", r"lipoprotein"]),
    ("CV_RISK",              [r"chd risk", r"coronary.*risk", r"cardiovascular risk"]),
    ("STATIN",               [r"simvastatin", r"atorvastatin", r"rosuvastatin",
                              r"pravastatin", r"fluvastatin", r"simvador"]),
    ("ANTIHYPERTENSIVE",     [r"amlodipine", r"ramipril", r"lisinopril",
                              r"losartan", r"candesartan", r"atenolol",
                              r"bisoprolol", r"bendroflumethiazide", r"indapamide"]),
    ("GASTRIC_ACID",         [r"\bppi\b", r"omeprazole", r"lansoprazole",
                              r"esomeprazole", r"pantoprazole", r"ranitidine",
                              r"famotidine"]),
    ("RESPIRATORY",          [r"inhaler", r"beclometasone", r"salbutamol",
                              r"\bcopd\b", r"asthma"]),
    ("MUSCULOSKELETAL",      [r"knee pain", r"back pain", r"shoulder pain",
                              r"joint pain", r"arthritis"]),
    # (ANTIBIOTIC_GENERIC removed — generic antibiotics are non-specific noise;
    #  UTI-specific antibiotics handled by UTI_ANTIBIOTICS)
    ("DIABETES_MGT",         [r"testing strip", r"glucose meter",
                              r"insulin", r"metformin", r"gliclazide",
                              r"empagliflozin"]),
    ("LIFESTYLE",            [r"diet.*advice", r"patient.initiated diet",
                              r"physical activity"]),

    # ── Med-side categories (breast-specific) ───────────────────
    ("5ARI_med",             [r"finasteride", r"dutasteride"]),
    ("ALPHA_BLOCKERS_med",   [r"tamsulosin", r"alfuzosin", r"doxazosin",
                              r"terazosin", r"prazosin"]),
    ("ANTICHOLINERGICS_med", [r"oxybutynin", r"solifenacin", r"tolterodine",
                              r"fesoterodine", r"trospium", r"darifenacin"]),
    ("ED_MEDICATIONS_med",   [r"sildenafil", r"tadalafil", r"vardenafil",
                              r"avanafil"]),
    ("UTI_ANTIBIOTICS_med",  [r"nitrofurantoin", r"trimethoprim"]),
    ("ANTICOAGULANT_med",    [r"warfarin", r"apixaban", r"rivaroxaban",
                              r"dabigatran", r"edoxaban"]),

    # ── Additional med categories surfacing in pre-12mo data ──────
    ("STATIN",               [r"simvador", r"simvastatin", r"atorvastatin",
                              r"rosuvastatin", r"pravastatin", r"fluvastatin"]),
    ("ANTIHYPERTENSIVE",     [r"amlodipine", r"amlostin", r"felodipine",
                              r"ramipril", r"lisinopril", r"perindopril",
                              r"enalapril", r"losartan", r"candesartan",
                              r"irbesartan", r"valsartan", r"atenolol",
                              r"bisoprolol", r"bendroflumethiazide",
                              r"indapamide"]),
    ("GASTRIC_ACID",         [r"mepradec", r"omeprazole", r"lansoprazole",
                              r"esomeprazole", r"pantoprazole", r"rabeprazole",
                              r"ranitidine", r"famotidine"]),
    ("RESPIRATORY",          [r"\binhaler\b", r"salbutamol", r"ventolin",
                              r"beclometasone", r"seretide", r"tiotropium",
                              r"spiriva", r"\bcopd\b", r"asthma"]),
    ("ANTIPLATELET",         [r"aspirin 75", r"aspirin 75mg",
                              r"clopidogrel", r"dipyridamole"]),
    ("PSYCHOTROPIC",         [r"\btemazepam\b", r"\bdosulepin\b",
                              r"diazepam", r"amitriptyline", r"sertraline",
                              r"citalopram", r"mirtazapine", r"\bzopiclone\b"]),
    ("PAIN_OTC",             [r"co.codamol", r"co.proxamol", r"paracetamol",
                              r"ibuprofen", r"indometacin", r"naproxen"]),
    ("DIABETES_MGT",         [r"metformin", r"gliclazide", r"pioglitazone",
                              r"insulin", r"empagliflozin", r"dapagliflozin",
                              r"liraglutide", r"sitagliptin",
                              r"testing strip", r"lancet",
                              r"glucose meter", r"diabetic peripheral neuropathy"]),
    # (ANTIBIOTIC_GENERIC — removed; handled as noise)
    ("STEROID_SYSTEMIC",     [r"prednisolone 5mg", r"prednisolone tab"]),
    ("THYROID_LAB",          [r"plasma tsh", r"thyroid hormone",
                              r"thyroid stim", r"\btsh\b"]),
    ("CANCER_MARKERS",       [r"carcinoembryonic antigen"]),
    ("LIPIDS",               [r"\bhdl\b", r"\bldl\b", r"cholesterol",
                              r"triglycerid", r"lipid profile",
                              r"lipoprotein", r"fasting lipids",
                              r"fasting blood lipids"]),
    ("CV_RISK",              [r"chd risk", r"coronary heart disease.*risk",
                              r"cardiovascular.*risk", r"joint british societies",
                              r"qstroke", r"qrisk"]),
    # NOTE: QRisk and Framingham are already filtered as noise; only
    # CV_RISK survives for other risk-score variants.
    ("METABOLIC",            [r"fasting.*glucose", r"fasting blood sugar",
                              r"\bhba1c\b", r"glycated h.*moglobin",
                              r"plasma glucose", r"random glucose",
                              r"oral glucose tolerance", r"\bogtt\b",
                              r"impaired fasting", r"plasma fasting glucose"]),
    ("SMOKING_CESSATION",    [r"varenicline", r"champix",
                              r"\bnicotine.*replacement", r"\bnrt\b",
                              r"\bbupropion\b"]),
    ("OPHTH_GLAUCOMA",       [r"timolol.*eye", r"travoprost", r"bimatoprost"]),
    # LAB_OTHER kept narrow — uric acid (metabolic-syndrome marker)
    ("LAB_OTHER",            [r"serum uric acid"]),
    # PTT → COAGULATION (already a defined category)
    ("COAGULATION",          [r"partial thromboplastin time", r"\bpt ratio\b"]),
    # Serum/plasma proteins → ALBUMIN_PROTEIN (already a defined category)
    ("ALBUMIN_PROTEIN",      [r"serum/plasma proteins"]),

    # ── Wave-2 expansions ────────────────────────────────────────
    # More PAIN_OTC patterns (NSAIDs, weak opioids)
    ("PAIN_OTC",             [r"\btramadol\b", r"\bmeloxicam\b",
                              r"\bcelecoxib\b", r"\bnaproxen\b",
                              r"\bketoprofen tab", r"\bdiclofenac\b",
                              r"codeine.*linctus", r"\bzapain\b"]),
    # Add nasal-steroid / inhaler variants to RESPIRATORY (Eklira is LAMA,
    # nasal sprays are RESPIRATORY-adjacent)
    ("RESPIRATORY",          [r"\beklira\b", r"genuair", r"\bvolumatic\b",
                              r"serial peak expiratory flow",
                              r"bronchodilators used",
                              r"acute lower respiratory tract infection",
                              r"acute wheezy bronchitis",
                              r"respiratory system diseases nos",
                              r"no haemoptysis"]),
    # CARDIOVASCULAR (separate from CV_RISK — this is comorbidity/diagnosis)
    ("CARDIOVASCULAR",       [r"\bgtn\b", r"nitrolingual",
                              r"glyceryl trinitrate",
                              r"angina pectoris", r"\bccs\b",
                              r"cardiovascular disease monitoring",
                              r"coronary heart disease monitoring",
                              r"cardiovascular angiography",
                              r"follow.up cardiac assessment",
                              r"exercise tolerance test",
                              r"peripheral pulses",
                              r"phlebitis", r"thrombophlebitis",
                              r"canadian cardiovascular society"]),
    # MUSCULOSKELETAL (more)
    ("MUSCULOSKELETAL",      [r"musculoskeletal symptom",
                              r"musculoskeletal and connective tissue",
                              r"rotator cuff tear"]),
    # METABOLIC (more — random/test orders/glucose tolerance)
    ("METABOLIC",            [r"random blood sugar",
                              r"glucose tolerance test",
                              r"test request.*blood glucose",
                              r"no h/o.*diabetes mellitus"]),
    # DIABETES_MGT (more — clinical pathway + foot)
    ("DIABETES_MGT",         [r"diabetes clinical pathway",
                              r"foot care advice"]),
    # ANTIHYPERTENSIVE (lercanidipine)
    ("ANTIHYPERTENSIVE",     [r"\blercanidipine\b"]),
    # PSYCHOTROPIC (more)
    ("PSYCHOTROPIC",         [r"\bzolpidem\b", r"\bhydroxyzine\b",
                              r"\bgabapentin\b", r"\bquetiapine\b"]),
    # THYROID_LAB (more)
    ("THYROID_LAB",          [r"free thyroxine", r"plasma free t4",
                              r"test request.*thyroid function",
                              r"\bft4\b", r"\bft3\b"]),
    # ALCOHOL — known minor breast-cancer risk factor; KEEP
    ("ALCOHOL",              [r"alcohol questionnaire", r"alcohol intake",
                              r"alcohol consumption", r"\bcage\b",
                              r"\baudit-c\b"]),
    # FRAILTY — pre-12mo signal for older-male population
    ("FRAILTY",              [r"\bearrs\b", r"qmortality risk",
                              r"qfrailty category"]),

    # ── Wave-3 category expansions ────────────────────────────────
    # VITAMIN_D — supplement brands (real clinical relevance)
    ("VITAMIN_D",            [r"invita d3", r"pro d3", r"fultium-d3",
                              r"colecalciferol", r"cholecalciferol",
                              r"ergocalciferol"]),
    # DIABETES_MGT — Sukkarto = metformin brand; diabetic retinopathy review;
    # impaired-glucose-tolerance referral
    ("DIABETES_MGT",         [r"sukkarto", r"diabetic retinopathy",
                              r"impaired glucose tolerance"]),
    # SMOKING — exhaled carbon monoxide is a smoking biomarker
    ("SMOKING",              [r"carbon monoxide concentration",
                              r"expired carbon monoxide"]),
    # RESPIRATORY — lung-function tests, airways obstruction
    ("RESPIRATORY",          [r"fev1/fvc", r"\bfev1\b", r"\bfvc\b",
                              r"airways obstruction", r"respiratory flow rate"]),
    # URINE_MARKERS — MSU midstream urine
    ("URINE_MARKERS",        [r"\bmsu\b", r"midstream urine",
                              r"urine culture"]),
    # CARDIOVASCULAR — negative-history-of stroke/cva (relevant: tells model
    # patient has been screened/asked)
    ("CARDIOVASCULAR",       [r"no h/o.*cva", r"no h/o.*stroke",
                              r"no h/o.*ischaemic",
                              r"brain natriuretic peptide", r"\bbnp\b",
                              r"exercise ecg", r"\becg\b"]),
    # CV_RISK — more risk-score variants
    ("CV_RISK",              [r"chads2", r"chads.vasc", r"has-bled"]),
    # RESPIRATORY — spirometry / inhaled-steroid use
    ("RESPIRATORY",          [r"\bspirometry", r"using inhaled steroids",
                              r"not using inhaled steroids",
                              r"inhaled steroid"]),
    # ALCOHOL — screening tests / use disorder identification
    ("ALCOHOL",              [r"alcohol screen", r"alcohol use",
                              r"audit.questionnaire", r"\baudit\b"]),
    # STATIN — prophylactic use code (in addition to drug names)
    ("STATIN",               [r"statin prophylaxis", r"statin therapy"]),
    # LIPIDS — bare "lipids" term (covers "test request : lipids")
    ("LIPIDS",               [r"\blipids\b"]),
    # ELECTROLYTES — "test request : serum electrolytes"
    ("ELECTROLYTES",         [r"test request.*serum electrolytes"]),
    # INFLAMMATORY — ESR test request
    ("INFLAMMATORY",         [r"test request.*esr", r"\besr\b",
                              r"test request.*c.reactive protein",
                              r"plasma viscosity"]),
    # ── Wave-5 category expansions ───────────────────────────────
    # RESPIRATORY — FEV1/FVC by full term, peak flow, vital capacity, more
    ("RESPIRATORY",          [r"forced expired volume",
                              r"forced vital capacity",
                              r"\bpeak expiratory flow\b",
                              r"peak flow rate", r"vital capacity test",
                              r"acute bronchitis",
                              r"chronic obstructive pulmonary",
                              r"percent predicted fef",
                              r"\bfef.*25",
                              r"acute respiratory infection",
                              # Negative findings — clinically informative
                              r"night cough",
                              r"no haemoptysis",
                              r"standard chest x.ray"]),
    # COAGULATION (more — INR target range)
    ("COAGULATION",          [r"international normalised ratio",
                              r"\binr\b within target"]),
    # CARDIOVASCULAR — more
    ("CARDIOVASCULAR",       [r"\bchd monitoring\b",
                              r"chd monitoring administration",
                              r"\bcoronary heart disease\b",
                              r"exercise stress echocardiography",
                              r"beta blocker contraindicated",
                              r"beta blocker not indicated",
                              r"stroke monitoring",
                              r"chronic kidney disease stage"]),    # CKD ≈ vascular comorbidity
    # GASTRIC_ACID — oesophagitis / reflux
    ("GASTRIC_ACID",         [r"oesophagitis", r"reflux"]),
    # METABOLIC — "at risk of diabetes"
    ("METABOLIC",            [r"at risk of diabetes mellitus",
                              r"at risk of diabetes"]),
    # SEXUAL_REPRODUCTIVE — testicular pain
    ("SEXUAL_REPRODUCTIVE",  [r"testicular pain", r"testicular"]),
    # FRAILTY — memory assessment
    ("FRAILTY",              [r"initial memory assessment",
                              r"memory assessment"]),
    # Wave 7 — final clean-up categories
    ("ELECTROLYTES",         [r"serum electrolytes level"]),
    ("THYROID_LAB",          [r"thyroid function test"]),
    ("HYPERTENSION",         [r"hypertensiv"]),
    ("ANTIHYPERTENSIVE",     [r"angiotensin.*ii.*receptor",
                              r"angiotensin converting enzyme.*prophylaxis",
                              r"angiotensin converting enzyme.*inhibitor",
                              r"\bace\b inhibitor"]),
    ("CARDIOVASCULAR",       [r"beta blocker prophylaxis",
                              r"primary prevention of ischaemic"]),
    ("URINE_MARKERS",        [r"urine blood test", r"urine urobilinogen",
                              r"urine sent for culture",
                              # Negative finding
                              r"urine screening normal"]),
    ("UTI",                  [r"urinary tract infectious"]),
    ("DIABETES_MGT",         [r"fundoscopy.*diabetic", r"low carbohydrate diet",
                              # Negative finding — diabetic eye check
                              r"fundoscopy normal"]),
    ("LIFESTYLE",            [r"lifestyle education"]),
    # CANCER_SCREENING — colorectal/bowel screening + sigmoidoscopy
    ("CANCER_SCREENING",     [r"occult blood not detected in faeces",
                              r"\bfobt\b",
                              r"faecal occult blood",
                              r"bowel cancer screen",
                              r"bowel scope",
                              r"flexible.sigmoidoscop",
                              r"screening for malignant neoplasm of large intestine",
                              r"diagnostic endoscopic examination on colon"]),
    # PRIOR_CANCER_HISTORY — prior skin cancers (very common in older men;
    # risk factor not breast leakage)
    ("PRIOR_CANCER_HISTORY", [r"squamous cell carcinoma",
                              r"basal cell carcinoma"]),
    # UROLOGY_PATHWAY — pre-12mo urology workup (biopsy/cystoscopy/referrals)
    # Caught via OVERRIDES but listed here for documentation
    ("UROLOGY_PATHWAY",      [r"needle biopsy of breast",
                              r"transurethral biopsy breast",
                              r"diagnostic cystoscopy",
                              r"flexible.*cystoscop",
                              r"referred to urologist",
                              r"referral to urologist",
                              r"seen in urology clinic"]),

    # ── Earlier-wave additions (kept) ──
    # RENAL_FUNCTION — CKD staging
    ("RENAL_FUNCTION",       [r"chronic kidney disease stage",
                              r"\bckd\b"]),
    # METABOLIC — serum glucose, high risk of diabetes
    ("METABOLIC",            [r"serum glucose level",
                              r"high risk of diabetes",
                              r"plasma viscosity"]),
    # PSYCHOTROPIC — mental-health screening
    ("PSYCHOTROPIC",         [r"depression screening",
                              r"anxiety screening",
                              r"hospital anxiety and depression"]),
    # ELECTROLYTES — serum bicarbonate, urine electrolytes
    ("ELECTROLYTES",         [r"serum bicarbonate",
                              r"urine electrolytes"]),
    # FBC_HAEMATOLOGY — promyelocyte
    ("FBC_HAEMATOLOGY",      [r"promyelocyte"]),
    # MUSCULOSKELETAL — back / lumbar pain (could be late breast signal)
    ("MUSCULOSKELETAL",      [r"pain in lumbar spine",
                              r"supraspinatus tendonitis",
                              r"lumbar spine"]),
    # ALCOHOL — beverage intake
    ("ALCOHOL",              [r"alcoholic beverage intake"]),
    # ANTIPLATELET — salicylate prophylaxis (low-dose aspirin)
    ("ANTIPLATELET",         [r"salicylate prophylaxis"]),
    # FRAILTY — more frailty/risk scores
    ("FRAILTY",              [r"at risk of dementia",
                              r"at risk of emergency hospital",
                              r"parr-30",
                              r"gp assessment of cognition"]),
    # DIABETES_MGT — more
    ("DIABETES_MGT",         [r"high risk of diabetes",
                              r"good compliance with diabetic diet",
                              r"diabetes education",
                              r"has seen dietitian.*diabetes",
                              r"retinopathy follow up",
                              r"retinal photography"]),
    # THYROID_LAB — autoimmune thyroid antibodies
    ("THYROID_LAB",          [r"thyroid peroxidase antibody",
                              r"\btpo\b antibody"]),
    # CANCER_MARKERS — broader
    ("CANCER_MARKERS",       [r"\bafp\b",
                              r"alpha.fetoprotein",
                              r"\bca\s*15-?3\b",
                              r"\bca\s*19-?9\b",
                              r"\bca\s*125\b"]),
]


# ════════════════════════════════════════════════════════════════
# COMPILED REGEXES
# ════════════════════════════════════════════════════════════════

_CATEGORY_RX = [(cat, re.compile(p, re.I))
                for cat, pats in CATEGORY_KEYWORDS
                for p in pats]


# ════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════

def setup_logging():
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(CONFIG["out_dir"], f"breast_curation_log_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ════════════════════════════════════════════════════════════════
# CORE CLASSIFIER
# ════════════════════════════════════════════════════════════════

def is_family_history(term_lower: str) -> bool:
    """Family hx is a RISK FACTOR even when the term contains 'cancer'."""
    return (term_lower.startswith("family history")
            or term_lower.startswith("no family history")
            or term_lower.startswith("no relevant family history")
            or term_lower.startswith("no significant family history")
            or "family history of" in term_lower
            or "fh:" in term_lower)


# Side-channel: classify() sets this if an override matched, so main() can
# read the override category. Reset per call.
_OVERRIDE_CATEGORY = {"value": None}


def classify(term: str):
    """Return (decision, reason)."""
    _OVERRIDE_CATEGORY["value"] = None
    if not isinstance(term, str) or not term.strip():
        return "REMOVE_NOISE", "generic:empty term"
    t = term.lower()

    # NOISE_OVERRIDES — terms that look like dx-pathway but are non-breast;
    # checked before dx-pathway so they don't end up as BORDERLINE.
    for needle, label in NOISE_OVERRIDES:
        if needle in t:
            return "REMOVE_NOISE", f"generic:{label}"

    # OVERRIDES — clinically-meaningful pre-dx codes that look like leakage/
    # borderline by substring but should stay as KEEP. Checked first.
    for needle, category in OVERRIDES:
        if needle in t:
            _OVERRIDE_CATEGORY["value"] = category
            return "KEEP", f"valid pre-dx signal (override → {category})"

    # Family-history override — keep these as risk-factor signal.
    if is_family_history(t):
        return "KEEP", "valid pre-dx signal"

    # Breast-cancer treatment drug leakage (GnRH agonists/antagonists,
    # antiandrogens) — presence in pre-dx window implies prior diagnosis.
    for needle, label in CANCER_TREATMENT_DRUGS:
        if needle in t:
            return "REMOVE_LEAKAGE", f"cancer-treatment:{label}"

    # Cancer-name leakage
    for w in CANCER_NAME_TERMS:
        if w in t:
            return "REMOVE_LEAKAGE", f"cancer-name:{w}"

    # Cancer-pathway leakage
    for needle, label in CANCER_PATHWAY_PHRASES:
        if needle in t:
            return "REMOVE_LEAKAGE", f"cancer-pathway:{label}"

    # Borderline dx-pathway
    for needle, label in DX_PATHWAY_PHRASES:
        if needle in t:
            return "BORDERLINE_DX_PATHWAY", f"dx-pathway:{label}"

    # Generic noise
    for needle, label in NOISE_PHRASES:
        if needle in t:
            return "REMOVE_NOISE", f"generic:{label}"

    # Default
    return "KEEP", "valid pre-dx signal"


def suggest_category(term: str):
    if not isinstance(term, str) or not term.strip():
        return ""
    for cat, rx in _CATEGORY_RX:
        if rx.search(term):
            return cat[:-4] if cat.endswith("_med") else cat
    return ""


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 80)
    logger.info("BREAST — Phase 1.4b — Curation Review")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    logger.info("\n📂 Loading inputs...")
    with open(CONFIG["v2_mapping"]) as f:
        v2 = json.load(f)
    v2_codes = set(v2.get("obs", {}).keys()) | set(v2.get("med", {}).keys())
    logger.info(f"  📋 v2 mapping: {len(v2_codes)} curated codes")

    review = pd.read_csv(CONFIG["review_tsv"], sep="\t",
                         dtype={"code": str})
    review["code"] = review["code"].astype(str).str.strip()
    logger.info(f"  📊 Review TSV: {len(review):,} rows")

    # Sort by rank so the output mirrors prior reviews.
    review["_rank_sort"] = pd.to_numeric(review["combined_rank"], errors="coerce").fillna(10**9)
    review = review.sort_values("_rank_sort").reset_index(drop=True)

    rows_out = []
    decision_counts = {"KEEP": 0, "REMOVE_LEAKAGE": 0,
                       "REMOVE_NOISE": 0, "BORDERLINE_DX_PATHWAY": 0}
    reason_counts = {}
    n_already_curated = 0
    n_skipped_no_cat = 0
    skipped_rows = []

    for _, r in review.iterrows():
        code = str(r["code"]).strip()
        if not code or code.lower() in {"nan", "none"}:
            continue
        term = str(r["term"]) if pd.notna(r["term"]) else ""
        feat_type = "OBS" if r.get("feature_type") == "observation" else "MED"
        rank = (int(r["combined_rank"])
                if pd.notna(r["combined_rank"]) else "")
        score = (round(float(r["combined_score"]), 4)
                 if pd.notna(r["combined_score"]) else "")
        already_curated = "yes" if code in v2_codes else "no"
        if already_curated == "yes":
            n_already_curated += 1

        decision, reason = classify(term)
        # Capture override-category if classify() set one
        override_cat = _OVERRIDE_CATEGORY["value"]
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        suggested_cat = ""
        if decision == "KEEP" and already_curated == "no":
            # Prefer explicit override category if set; else fall back to
            # term-keyword matching.
            suggested_cat = override_cat or suggest_category(term)
            if not suggested_cat:
                n_skipped_no_cat += 1
                skipped_rows.append({
                    "rank": rank, "type": feat_type, "code": code,
                    "prev_pos":   (round(float(r["prevalence_pos"]) * 100, 2)
                                   if pd.notna(r.get("prevalence_pos")) else ""),
                    "ratio":      (round(float(r["odds_ratio"]), 2)
                                   if pd.notna(r.get("odds_ratio")) else ""),
                    "reason":     "no-match-skipped",
                    "term":       term,
                })

        rows_out.append({
            "rank":               rank,
            "score":              score,
            "code":               code,
            "term":               term,
            "type":               feat_type,
            "decision":           decision,
            "reason":             reason,
            "already_curated":    already_curated,
            "suggested_category": suggested_cat,
        })

    # Write main curation review
    out_path = os.path.join(CONFIG["out_dir"], CONFIG["out_main"])
    fieldnames = ["rank", "score", "code", "term", "type",
                  "decision", "reason", "already_curated", "suggested_category"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows_out)
    logger.info(f"\n  💾 {out_path} ({len(rows_out):,} rows)")

    # Write skipped-no-category
    skipped_path = os.path.join(CONFIG["out_dir"], CONFIG["out_skipped"])
    if skipped_rows:
        sk_fields = ["rank", "type", "code", "prev_pos", "ratio", "reason", "term"]
        with open(skipped_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sk_fields, quoting=csv.QUOTE_MINIMAL)
            w.writeheader()
            w.writerows(skipped_rows)
        logger.info(f"  💾 {skipped_path} ({len(skipped_rows):,} rows)")

    # ── Summary ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  Total ranked codes: {len(rows_out):,}")
    logger.info(f"  Already curated (in v2): {n_already_curated}")
    logger.info(f"  No-category-suggested (in skipped CSV): {n_skipped_no_cat}")
    logger.info("")
    logger.info("  Decisions:")
    for d in ["KEEP", "REMOVE_LEAKAGE", "BORDERLINE_DX_PATHWAY", "REMOVE_NOISE"]:
        logger.info(f"     {d:<25} {decision_counts.get(d, 0):>6,}")
    logger.info("\n  Top reason tags:")
    for reason, n in sorted(reason_counts.items(), key=lambda x: -x[1])[:20]:
        logger.info(f"     {n:>5}  {reason}")

    logger.info("\n  Per-category suggestions (KEEP + new):")
    cat_counts = {}
    for row in rows_out:
        if row["decision"] == "KEEP" and row["already_curated"] == "no" and row["suggested_category"]:
            cat_counts[row["suggested_category"]] = cat_counts.get(row["suggested_category"], 0) + 1
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        logger.info(f"     {cat:<25} {n:>4}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Curation review complete — open the CSV in a spreadsheet to triage.")
    logger.info(f"   Main:    {out_path}")
    logger.info(f"   Skipped: {skipped_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ FATAL: {e}", exc_info=True)
        sys.exit(1)
