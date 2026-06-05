"""Auto-generate feature_manifest_<cancer>_<window>.json sidecar after FE saves
the final feature matrix.

The manifest is a small JSON file documenting every column in the matrix:
  - name, dtype, semantic group (auto-classified by prefix)
  - group_counts summary
  - pipeline metadata (cancer, window, version, generated_at, n_patients)

Cancer-agnostic — uses cfg.PREFIX to identify cancer-specific columns.
Used for: documentation, modeling/inference column-validation, audit diffs.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


# Order matters — more specific patterns first. Tested against name AFTER stripping cfg.PREFIX.
GROUP_RULES = [
    # Age × interactions
    ("AGEX_",                      "age_interaction"),
    ("ELDERLY_",                   "age_interaction"),
    # Triples + biomarker pairs
    ("TRIPLE_",                    "triple_interaction"),
    ("LABPAIR_",                   "lab_pair"),
    ("PAIR_",                      "lab_pair"),
    # Lab dynamics + thresholds
    ("ZSCORE_",                    "lab_personal_zscore"),
    ("LAB_",                       "lab_threshold"),
    # Symptom dynamics
    ("VEL_",                       "velocity"),
    ("NEW_",                       "new_in_B_only"),
    ("FIRST_",                     "first_event_date"),
    ("RECENT_",                    "recent_onset"),
    ("SEQ_",                       "sequential_pattern"),
    # Comorbidity / activity / multisystem
    ("COMORBIDITY_",               "comorbidity_burden"),
    ("COMORB_",                    "weighted_comorbidity"),
    ("ACTIVITY_",                  "activity_escalation"),
    ("MULTISYSTEM_",               "multisystem"),
    ("TRIFECTA_",                  "multisystem"),
    # Investigation × symptom (multiple per-cancer prefixes)
    ("IMG_",                       "investigation_x_symptom"),
    ("PATHWAY_",                   "investigation_x_symptom"),
    ("HAEM_INV_",                  "investigation_x_symptom"),
    ("LAB_MARKERS_",               "investigation_x_symptom"),
    ("LYMPH_PROC_",                "investigation_x_symptom"),
    ("URINEINV_",                  "investigation_x_symptom"),
    ("URINELAB_",                  "investigation_x_symptom"),
    ("UROL_",                      "investigation_x_symptom"),
    ("DERM_",                      "investigation_x_symptom"),
    ("HISTO_",                     "investigation_x_symptom"),
    ("REFERRAL_",                  "investigation_x_symptom"),
    ("GYNAE_PROC_",                "investigation_x_symptom"),
    # Symptom burst
    ("SYMPTOM_burst",              "symptom_burst"),
    ("SYMPTOM_max",                "symptom_burst"),
    # NICE / clinical guideline
    ("NICE_",                      "nice_guideline"),
    # Risk / suspicion / IPI scores
    ("IPI_",                       "ipi_score"),
    ("SUSPICION_",                 "suspicion_score"),
    ("risk_",                      "risk_score"),
    # Risk factors
    ("RF_",                        "risk_factor"),
    # Treatment + diagnostic
    ("TX_",                        "treatment_pattern"),
    ("DX_",                        "diagnostic_pathway"),
    ("PATH_",                      "diagnostic_pathway"),
    # Mimic / cancer-specific composites
    ("MIMIC_",                     "mimic_pattern"),
    ("OV_",                        "cancer_specific_composite"),
    # Persistence
    ("PERSIST_",                   "persistence"),
    # Cancer-specific category groups (mostly bladder)
    ("HAEM_",                      "haematuria_deep"),
    ("URINE_",                     "urine_deep"),
    ("LUTS_",                      "luts_deep"),
    ("CATH_",                      "catheter_proxy"),
    ("RECUTI_",                    "recurrent_uti"),
    ("smoking_",                   "smoking"),
    # Generic
    ("INT_",                       "interaction_pair"),
    ("HAS_",                       "clinical_flag"),
    ("has_",                       "clinical_flag"),
    ("nodal_",                     "nodal_mass"),
    ("bsymptoms",                  "constitutional_symptom"),
    ("infection",                  "infection_pattern"),
    ("bleeding",                   "bleeding_pattern"),
    ("cardinal_count",             "nice_guideline"),
    ("2plus_cardinals",            "nice_guideline"),
    ("3plus_cardinals",            "nice_guideline"),
]


def classify(col: str, prefix: str) -> str:
    """Return semantic group for a feature column name."""
    name = col[len(prefix):] if col.startswith(prefix) else col
    for pat, group in GROUP_RULES:
        if name.startswith(pat):
            return group
    # Non-cancer-specific (spine/base) columns
    if not col.startswith(prefix):
        if col in ("AGE_AT_INDEX", "SEX", "INDEX_DATE", "PATIENT_GUID"):
            return "spine"
        if col.startswith("OBS_"):    return "base_obs"
        if col.startswith("MED_"):    return "base_med"
        if col.startswith("LAB_"):    return "base_lab"
        if col.startswith("CROSS_"):  return "cross_domain"
        if col.startswith("CLUSTER_"):return "symptom_cluster"
        return "base_feature"
    return "cancer_specific_other"


def _simplify_dtype(dt) -> str:
    s = str(dt)
    if s.startswith(("int", "Int")):  return "int"
    if s.startswith("float"):         return "float"
    if s == "bool":                   return "bool"
    if s.startswith("datetime"):      return "date"
    if s in ("object", "string"):     return "string"
    return s


def generate_manifest(fm: pd.DataFrame, cancer: str, window: str, prefix: str,
                      pipeline_version: str = "v4_1to1") -> Dict[str, Any]:
    """Build manifest dict from final feature matrix."""
    features = {}
    group_counts: Dict[str, int] = {}
    for col in fm.columns:
        if col == "LABEL":
            continue
        group = classify(col, prefix)
        features[col] = {
            "dtype":              _simplify_dtype(fm[col].dtype),
            "group":              group,
            "is_cancer_specific": col.startswith(prefix),
        }
        group_counts[group] = group_counts.get(group, 0) + 1

    return {
        "cancer":           cancer,
        "window":           window,
        "feature_prefix":   prefix,
        "pipeline_version": pipeline_version,
        "generated_at":     datetime.utcnow().isoformat() + "Z",
        "n_patients":       int(len(fm)),
        "n_features":       len(features),
        "group_counts":     dict(sorted(group_counts.items(), key=lambda kv: -kv[1])),
        "features":         features,
    }


def write_manifest(fm: pd.DataFrame, out_path, cancer: str, window: str,
                   prefix: str, pipeline_version: str = "v4_1to1") -> Path:
    """Generate manifest and write to JSON. Returns the path written."""
    out_path = Path(out_path)
    manifest = generate_manifest(fm, cancer, window, prefix, pipeline_version)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"  manifest → {out_path.name}  "
                f"({len(manifest['features'])} features, "
                f"{len(manifest['group_counts'])} groups)")
    return out_path


def load_and_validate(parquet_path, manifest_path,
                      require_all_present: bool = True) -> pd.DataFrame:
    """Load a feature matrix and validate against its manifest.

    require_all_present: if True, raise ValueError when any manifest column is
    missing from the parquet (catches silent drops between FE and modeling).
    Always warns about EXTRA columns in parquet that aren't in manifest.
    """
    fm = pd.read_parquet(parquet_path)
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        logger.warning(f"  no manifest at {manifest_path} — skipping validation")
        return fm
    with open(manifest_path) as f:
        manifest = json.load(f)
    expected = set(manifest["features"].keys())
    actual = set(fm.columns) - {"LABEL"}
    missing = expected - actual
    extra = actual - expected
    if missing and require_all_present:
        raise ValueError(
            f"feature drift: {len(missing)} columns expected by manifest "
            f"are missing from parquet (first 5: {sorted(missing)[:5]})"
        )
    if missing:
        logger.warning(f"  {len(missing)} expected columns missing")
    if extra:
        logger.warning(f"  {len(extra)} new columns not in manifest (rebuild?)")
    return fm
