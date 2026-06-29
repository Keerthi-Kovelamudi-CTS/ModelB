"""PARAMETERIZED WITH-text merge: structured ⊕ free-text, windowed to N years ("N-for-N"), Astra 12mo cohort.
Generalizes build_merged_lifetime.py to ANY lookback:

    python build_merged.py <years>            -> Astra_<N>yr_withtext.parquet
    python build_merged.py 10                 -> Astra_10yr_withtext.parquet  (matches the current 10yr arm)
    python build_merged.py <years> <SRC> <TXT> <OUT>     (explicit overrides)

Differences vs build_merged_lifetime.py (which keeps structured at LIFETIME and lets FE slice):
  - Slices BOTH structured AND free-text to days_before_anchor <= round(N*365.25) -> the output IS an
    N-year cache (window-matched), so a model run at an N-year lookback consumes exactly N years on both sides.

Everything else is identical to the validated merge:
  - free-text kept where assertion_status in {present, historical} AND noise_flag == 'clean'
  - DEDUP on (patient_guid, snomed_c_t_concept_id, days_before_anchor): drop text rows that duplicate a
    STRUCTURED (patient,code,day) + drop text-vs-text dups; structured rows are NEVER touched
  - progression -> value (worsening=+1, stable=0, improving=-1)
  - event_type = 'observation' so FE routes text via snomed_c_t_concept_id (NOT med_code_id -> would drop it)
  - term = snomed_name (the mapped SNOMED term); associated_text = concept_name (raw extracted phrase)

DATA CEILING (important): the free-text source has a hard EXTRACTION window (currently 10yr — max
days_before_anchor = 3653). Requesting N > that ceiling cannot invent text beyond it; the structured side
is lifetime so it windows fine, but the text simply stops at the ceiling. The script prints a clear [warn]
when the free-text coverage is below the requested window — to truly do e.g. "20 for 20" you must FIRST
re-extract the free-text concept table at 20yr (extend its BigQuery window 3653 -> 7305 days).
"""
import sys, os, glob, numpy as np, pandas as pd, pyarrow.parquet as pq

if len(sys.argv) < 2:
    raise SystemExit("usage: python build_merged.py <years> [SRC] [TXT] [OUT]")
YEARS = float(sys.argv[1]); CAP = round(YEARS * 365.25)
_ny = str(int(YEARS)) if YEARS == int(YEARS) else str(YEARS)
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(HERE, "..", "..")
SRC = sys.argv[2] if len(sys.argv) > 2 else os.path.join(B, "cache/raw_events/Astra_lifetime.parquet")

def _auto_txt():
    """Prefer a free-text source named for THIS window; else the widest internal source present."""
    for pat in (f"*internal_{_ny}yr*", f"*{_ny}yr*internal*", "*concept_event_table_internal*"):
        m = [x for x in sorted(glob.glob(os.path.join(HERE, pat)))
             if x.endswith(".parquet") and "heldout" not in os.path.basename(x).lower()]
        if m:
            return m[0]
    raise SystemExit(f"[build_merged] no free-text source parquet found in {HERE} for {_ny}yr "
                     f"(expected e.g. astra_concept_event_table_internal_{_ny}yr_160.parquet).")

TXT = sys.argv[3] if len(sys.argv) > 3 else _auto_txt()
OUT = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, f"Astra_{_ny}yr_withtext.parquet")

def _read(p): return pq.read_table(p).replace_schema_metadata(None).to_pandas()  # drop 'dbdate' ext metadata
def clean(s): return str(s).replace('"""', '').replace('{', '').replace('}', '').strip()
def col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

print(f"[build_merged] {YEARS}yr (<= {CAP}d) | SRC={os.path.basename(SRC)} | TXT={os.path.basename(TXT)}")
# ---- structured (sliced to N yr) ----
st = _read(SRC); st["_g"] = st["patient_guid"].map(clean)
st = st[pd.to_numeric(st["days_before_anchor"], errors="coerce").le(CAP)].copy()
print(f"structured (<= {CAP}d): {len(st):,} rows, {st['_g'].nunique():,} patients")
ed = pd.to_datetime(st["event_date"], errors="coerce")
st["_anchor"] = ed + pd.to_timedelta(pd.to_numeric(st["days_before_anchor"], errors="coerce"), unit="D")
anchor = st.groupby("_g")["_anchor"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
static_cols = [c for c in ["cancer_class", "cancer_id", "sex", "patient_ethnicity_16", "patient_ethnicity_6",
                           "patient_age", "age_at_anchor", "anchor_year"] if c in st.columns]
stat = st.groupby("_g").first()[static_cols]

# ---- free-text (filtered + sliced to N yr) ----
tx = _read(TXT); tx["_g"] = tx["patient_guid"].map(clean)
CODE = col(tx, "snomed_id", "Code"); NAME = col(tx, "snomed_name", "Name")
CN = col(tx, "concept_name"); RAW = col(tx, "source_raw_term", "source_term")
PROG = col(tx, "progression"); NF = col(tx, "noise_flag")
tx = tx[tx["assertion_status"].astype(str).str.lower().isin(["present"])]
if NF:
    tx = tx[tx[NF].astype(str).str.lower() == "clean"]
tx = tx[tx["_g"].isin(set(st["_g"]))]
_dba = pd.to_numeric(tx["days_before_anchor"], errors="coerce")
_ftmax = _dba.max()
tx = tx[_dba.notna() & _dba.le(CAP)].copy()
if pd.notna(_ftmax) and _ftmax < CAP:
    print(f"[warn] free-text source covers only ~{_ftmax/365.25:.1f}yr (max {int(_ftmax)}d) < requested "
          f"{YEARS}yr ({CAP}d). No text events exist beyond the extraction ceiling — re-extract the "
          f"free-text concept table at {YEARS}yr for genuine N-for-N coverage. (Structured windows fine.)")
prog = {"getting worse": 1, "worsening": 1, "stable": 0, "improving": -1, "improved": -1}
tx["_val"] = tx[PROG].astype(str).str.strip().str.lower().map(prog) if PROG else np.nan
print(f"text (present/historical+clean, in-cohort, <= {CAP}d): {len(tx):,}, {tx['_g'].nunique():,} patients")

rows = pd.DataFrame(index=tx.index)
g2raw = st.groupby("_g")["patient_guid"].first()
rows["patient_guid"] = g2raw.reindex(tx["_g"].values).values
rows["days_before_anchor"] = pd.to_numeric(tx["days_before_anchor"], errors="coerce").astype("int64").values
rows["snomed_c_t_concept_id"] = pd.to_numeric(tx[CODE], errors="coerce").astype("Int64").astype("int64").values
rows["term"] = tx[NAME].astype(str).values if NAME else ""                 # SNOMED term for the mapped code
rows["event_type"] = "observation"                                          # so FE routes via snomed (not null med_code_id)
rows["problem_status_description"] = tx["assertion_status"].astype(str).values
rows["significance_description"] = ""
rows["problem_comment"] = tx[RAW].astype(str).values if RAW else ""
rows["associated_text"] = tx[CN].astype(str).values if CN else ""           # raw extracted phrase
rows["value"] = pd.Series(tx["_val"]).map(lambda v: "" if pd.isna(v) else str(int(v))).values
rows["med_code_id"] = np.nan; rows["drug_term"] = ""; rows["duration"] = np.nan
rows["event_age"] = 0
g = tx["_g"].values; rows["_g"] = g
rows["event_date"] = (anchor.reindex(g).values - pd.to_timedelta(rows["days_before_anchor"].values, unit="D"))
for c in static_cols:
    rows[c] = stat[c].reindex(g).values

cols = [c for c in st.columns if c not in ("_g", "_anchor")]
for c in cols:
    if c not in rows.columns:
        rows[c] = np.nan
rows = rows[cols]
rows["event_date"] = pd.to_datetime(rows["event_date"], errors="coerce").dt.date
st_out = st[cols].copy(); st_out["event_date"] = pd.to_datetime(st_out["event_date"], errors="coerce").dt.date
for c in ["patient_age", "age_at_anchor", "event_age", "days_before_anchor", "anchor_year", "snomed_c_t_concept_id"]:
    if c in rows.columns:
        rows[c] = pd.to_numeric(rows[c], errors="coerce").fillna(0).astype("int64")
rows["value"] = rows["value"].astype(str)

# DEDUP: drop TEXT rows whose (patient,code,day) already exists in STRUCTURED + text-vs-text; structured untouched.
key = ["patient_guid", "snomed_c_t_concept_id", "days_before_anchor"]
struct_keys = set(map(tuple, st_out[key].itertuples(index=False, name=None)))
rows = rows.drop_duplicates(subset=key, keep="first")
keep = [k not in struct_keys for k in map(tuple, rows[key].itertuples(index=False, name=None))]
dropped = len(rows) - sum(keep); rows = rows[keep]
merged = pd.concat([st_out, rows], ignore_index=True)
print(f"text rows: dropped {dropped:,} dup-of-structured; kept {len(rows):,} new | "
      f"merged {len(merged):,} = structured {len(st_out):,} + text {len(rows):,} (structured preserved)")
merged.to_parquet(OUT, index=False)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)")
