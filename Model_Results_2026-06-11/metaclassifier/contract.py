"""Shared output contract for base-model predictions feeding the blending stack (Option A).

Every base team (LLM / LSTM / temporal_transformer / xgboost) emits ONE parquet per split:
    <model>_preds_valid.parquet   and   <model>_preds_test.parquet
with EXACTLY this schema, one row per patient per split.
"""
PRED_SCHEMA = {
    "patient_id":    "string",   # shared join key, identical across teams (normalised; see normalise_pid)
    "split":         "string",   # "valid" | "test"
    "proba_1":       "float",    # P(cancer) in [0,1]; the meta stacks on this. null iff abstained.
    "proba_0":       "float",    # 1 - proba_1 (sanity/logging)
    "model_name":    "string",   # "llm" | "lstm" | "temporal_transformer" | "xgboost"
    "model_version": "string",   # git sha / run id (frozen)
    "abstained":     "bool",     # true if patient unscoreable -> proba_1 = null (do NOT fake 0.5)
}
REQUIRED_COLS = list(PRED_SCHEMA)
VALID_SPLITS = {"valid", "test"}
KNOWN_MODELS = {"llm", "lstm", "temporal_transformer", "xgboost"}


def normalise_pid(s):
    """Canonical patient_id so every team joins on the SAME key. Strip braces/quotes, upper, trim."""
    import pandas as pd
    return (pd.Series(s).astype(str)
            .str.replace(r'[{}"]', "", regex=True).str.strip().str.upper())


def validate_df(df, expect_split=None):
    """Return (ok: bool, problems: list[str]) — structural + value checks against the contract."""
    import numpy as np, pandas as pd
    p = []
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, [f"missing columns: {missing}"]
    extra = [c for c in df.columns if c not in REQUIRED_COLS]
    if extra:
        p.append(f"WARN unexpected extra columns (will be ignored): {extra}")
    if df["patient_id"].isna().any():
        p.append("patient_id has nulls")
    if df["patient_id"].duplicated().any():
        p.append(f"duplicate patient_id rows: {int(df['patient_id'].duplicated().sum())} (must be one row/patient/split)")
    bad_split = set(df["split"].unique()) - VALID_SPLITS
    if bad_split:
        p.append(f"bad split values: {bad_split}")
    if expect_split and not (df["split"] == expect_split).all():
        p.append(f"split column is not all '{expect_split}'")
    ab = df["abstained"].fillna(False).astype(bool)
    pr = pd.to_numeric(df["proba_1"], errors="coerce")
    # abstained rows must have null proba_1; non-abstained must be a real prob in [0,1]
    if pr[ab].notna().any():
        p.append(f"{int(pr[ab].notna().sum())} abstained rows have non-null proba_1 (must be null)")
    live = pr[~ab]
    if live.isna().any():
        p.append(f"{int(live.isna().sum())} non-abstained rows have null/non-numeric proba_1")
    if ((live < 0) | (live > 1)).any():
        p.append(f"proba_1 outside [0,1] on {int(((live<0)|(live>1)).sum())} rows (raw logits? apply sigmoid)")
    # proba_0 ~ 1-proba_1 sanity (tolerate small fp error)
    s = pr[~ab] + pd.to_numeric(df["proba_0"], errors="coerce")[~ab]
    if (s.dropna().sub(1).abs() > 1e-3).any():
        p.append("proba_0 != 1 - proba_1 on some rows")
    if df["model_name"].nunique() != 1:
        p.append(f"mixed model_name in one file: {df['model_name'].unique()}")
    if df["model_version"].nunique() != 1 or df["model_version"].iloc[0] in ("", "nan", None):
        p.append("model_version must be a single frozen non-empty value (git sha / run id)")
    hard = [x for x in p if not x.startswith("WARN")]
    return (len(hard) == 0), p
