"""Make the patient-level explainability record dumps (FN/FP per-patient tabs) MATCH WHAT THE MODEL
CONSUMED — i.e. mirror the free-text dedup that `2_FE/free_text/build_merged_lifetime.py` applies when
building the FE-input cache. Run this on a folder of explainability workbooks, or import `match_model_input`.

WHY this exists
---------------
The record dumps are reconstructed from the RAW timeline (every recorded event). The FE merge, however,
only keeps the free-text the model actually used. To make the workbooks faithful to the model input we
apply the SAME three rules to the free-text rows (and ONLY the free-text rows):

  1. keep free-text only where assertion_status in {present, historical}   (drop absent / family)
  2. drop free-text-vs-free-text duplicates on (patient, code, day)         (keep first)
  3. drop free-text whose (patient, code, day) already exists in STRUCTURED (avoid double-count)

STRUCTURED rows are left 100% intact — including the same code recorded multiple times on the same day,
because FE features are EVENT COUNTS (two recordings = count 2). Collapsing them would misrepresent the
model input. This matches build_merged_lifetime.py exactly (text dedup only; structured untouched).

Workbook conventions handled
----------------------------
  *_combined_full_records*.xlsx / *_MODEL_USED_combined*.xlsx : per-patient tabs with a 'source' column
        ('free-text' vs structured) -> dedup the free-text rows in place, keep structured, preserve order.
  *_FREETEXT*.xlsx : per-patient tabs are free-text only (no 'source' col) -> needs structured (code,day)
        keys from the matching *_combined_full_records* workbook to apply rule 3; 'n_freetext_*' Summary
        column is recomputed.
  *_STRUCTURED*.xlsx : structured only -> left UNCHANGED (structured multiplicity is intentional).

Usage:  python explainability_record_dedup.py <folder-with-FN/FP-xlsx>   [--backup]
"""
import sys, os, glob, argparse, shutil
import pandas as pd

KEEP_ASSERTIONS = {"present", "historical"}
KEY = ["code", "days_before_anchor"]


def _keytuples(df):
    c = pd.to_numeric(df["code"], errors="coerce")
    d = pd.to_numeric(df["days_before_anchor"], errors="coerce")
    return list(zip(c.values, d.values))


def match_model_input(text_df, struct_keys):
    """Apply the build_merged_lifetime free-text dedup to a free-text DataFrame.
    `struct_keys` = set of (code, days_before_anchor) tuples from the patient's STRUCTURED rows."""
    if len(text_df) == 0:
        return text_df
    t = text_df.copy()
    t = t[t["assertion_status"].astype(str).str.lower().isin(KEEP_ASSERTIONS)]   # (1) drop absent/family
    t = t.drop_duplicates(KEY, keep="first")                                     # (2) text-vs-text
    if struct_keys:                                                              # (3) text dup of structured
        t = t[[k not in struct_keys for k in _keytuples(t)]]
    return t


def _dedup_combined(path):
    """combined / MODEL_USED workbook: each record sheet has a 'source' column."""
    xl = pd.ExcelFile(path)
    sheets = {}
    for s in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=s)
        if s == "Summary" or "source" not in df.columns:
            sheets[s] = df
            continue
        st = df[df["source"].astype(str) != "free-text"]
        ft = df[df["source"].astype(str) == "free-text"]
        ft = match_model_input(ft, set(_keytuples(st)))
        sheets[s] = pd.concat([st, ft]).sort_index()          # keep original row order
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for s in xl.sheet_names:
            sheets[s].to_excel(w, sheet_name=s, index=False)


def _struct_keys_by_patient(combined_path):
    """Per-patient structured (code,day) key sets, read from a *_combined_full_records* workbook."""
    xl = pd.ExcelFile(combined_path)
    out = {}
    for s in xl.sheet_names:
        if s == "Summary":
            continue
        d = pd.read_excel(combined_path, sheet_name=s)
        if "source" in d.columns:
            out[s] = set(_keytuples(d[d["source"].astype(str) != "free-text"]))
    return out


def _dedup_freetext(path, struct_by_patient):
    """FREETEXT workbook: free-text only, no 'source' column; recompute the n_freetext_* Summary column."""
    xl = pd.ExcelFile(path)
    sheets, counts = {}, {}
    for s in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=s)
        if s == "Summary":
            sheets[s] = df
            continue
        sheets[s] = match_model_input(df, struct_by_patient.get(s, set()))
        counts[s] = len(sheets[s])
    summ = sheets["Summary"].copy()
    ncol = next((c for c in summ.columns if c.startswith("n_freetext")), None)
    if ncol and "tab" in summ.columns:
        summ[ncol] = summ["tab"].map(counts).fillna(summ[ncol]).astype(int)
        sheets["Summary"] = summ
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for s in xl.sheet_names:
            sheets[s].to_excel(w, sheet_name=s, index=False)


def dedup_folder(folder, backup=False):
    """Dedup every FN/FP explainability workbook in `folder` to match the model input."""
    if backup:
        bk = os.path.join(folder, "_predupe_backup")
        os.makedirs(bk, exist_ok=True)
        for f in glob.glob(os.path.join(folder, "*.xlsx")):
            shutil.copy(f, bk)
    for grp in ("FN", "FP"):
        combined = glob.glob(os.path.join(folder, f"{grp}_combined_full_records*.xlsx"))
        if not combined:
            continue
        combined = combined[0]
        sk = _struct_keys_by_patient(combined)            # capture BEFORE editing
        _dedup_combined(combined)
        for mu in glob.glob(os.path.join(folder, f"{grp}_MODEL_USED_combined*.xlsx")):
            _dedup_combined(mu)
        for ftw in glob.glob(os.path.join(folder, f"{grp}_FREETEXT*.xlsx")):
            _dedup_freetext(ftw, sk)
        print(f"[{grp}] deduped combined + MODEL_USED + FREETEXT (STRUCTURED left unchanged)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder containing the FN/FP *.xlsx explainability workbooks")
    ap.add_argument("--backup", action="store_true", help="copy originals to <folder>/_predupe_backup first")
    a = ap.parse_args()
    dedup_folder(a.folder, backup=a.backup)
    print("done.")
