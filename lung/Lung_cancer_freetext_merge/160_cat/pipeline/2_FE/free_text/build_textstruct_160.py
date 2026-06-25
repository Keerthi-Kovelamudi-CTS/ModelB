# DESIGN A merge for the 160-category arm: free-text "as structured" (real snomed_id, no synthetic codes).
#   - Source = concept_event_table_snomed_mapped_*.parquet (free-text concepts already mapped to snomed_id).
#     Confirmed: all free-text snomed_ids ARE Codes in the 160 codelist (categorized_list_top160cat) -> they
#     fold cleanly into the codelist's Category via FE's Code->Category map. So NO generic-code dropping is
#     needed here (unlike the 10k build_textstruct, where ~140 ambiguous 1-snomed->many-grouping codes are cut).
#   - present/historical/absent/family ALL merged via the real snomed_id; assertion carried on `ft_assertion`
#     so build_features ASSERTION_FE counts POSITIVE findings in the base families and emits absent/family
#     counts + duration/frequency stats per category separately.
#   - duration/frequency parsed to numerics on `ft_dur`/`ft_freq`.
#   - value = concern ordinal (max of severity/progression).
#   - leakage discipline: keep noise_flag=='clean'; drop any LUNG_LEAK (lung-cancer-pathway) snomeds if present.
# args: SRC(structured cache) FT(free-text source) MAPIN(160 codelist, for logging only) CACHEOUT
import sys, re, pandas as pd, numpy as np, csv, os, pyarrow.parquet as pq
SRC, FT, MAPIN, CACHEOUT = sys.argv[1:5]
def clean(s): return str(s).replace('"""','').replace('{','').replace('}','').strip()
def rd(p): return pq.read_table(p).replace_schema_metadata(None).to_pandas()
def to_int_code(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.replace(r'\.0$', '', regex=True), errors="coerce")
def parse_dur(s):
    m = re.match(r'(\d+)\s*(day|week|month|year)', str(s).strip().lower())
    return int(m.group(1)) * {'day':1,'week':7,'month':30,'year':365}[m.group(2)] if m else np.nan
_FREQ = {'daily':3,'every day':3,'constant':3,'continuous':3,'twice daily':3,'twice a day':3,
         'weekly':2,'few times a week':2,'couple times a week':2,
         'intermittent':1,'occasional':1,'occasionally':1,'sometimes':1,'now and then':1}
def parse_freq(s):
    s = str(s).strip().lower()
    if s in _FREQ: return _FREQ[s]
    return 3 if re.match(r'(\d+)\s*(?:per|/|a)\s*day', s) else np.nan

st = rd(SRC); st["_g"] = st["patient_guid"].map(clean)
g2raw = st.groupby("_g")["patient_guid"].first()
static = ["cancer_class","cancer_id","sex","patient_ethnicity_16","patient_ethnicity_6","patient_age","age_at_anchor","anchor_year"]
static = [c for c in static if c in st.columns]
stat = st.groupby("_g").first()[static]
print(f"structured: {len(st):,} rows, {st['_g'].nunique():,} patients", flush=True)

m = list(csv.DictReader(open(MAPIN)))
code2cat = {int(r["Code"]): r["Category"] for r in m if r.get("Code","").strip().lstrip("-").isdigit()}
print(f"160 codelist: {len(code2cat):,} codes / {len(set(code2cat.values())):,} categories", flush=True)

tx = rd(FT)
# normalize source column names (160 source uses snomed_id / snomed_name)
tx["code"] = to_int_code(tx["snomed_id"])
tx["name"] = tx["snomed_name"].astype(str) if "snomed_name" in tx.columns else tx.get("concept_name", "").astype(str)
sev = {'mild':1,'slight':1,'moderate':2,'severe':3,'++':3,'+++':3}
prog = {'getting worse':3,'worsening':3,'increasing':3,'gain':3,'worse':3,'stable':1,'improving':0,'improved':0,'decreasing':0}
def conc(r):
    s = sev.get(str(r['severity']).strip().lower()); p = prog.get(str(r['progression']).strip().lower())
    v = [x for x in (s,p) if x is not None]; return max(v) if v else np.nan
tx["_concern"] = tx.apply(conc, axis=1)
tx["_dur"]  = tx["duration"].map(parse_dur)
tx["_freq"] = tx["frequency"].map(parse_freq)

LUNG_LEAK = {93880001,162573006,786838002,275981009,173171007,162572001}   # lung-cancer-pathway codes — never merge
tx = tx[(tx.noise_flag == "clean") & tx.code.notna() & ~tx.code.isin(LUNG_LEAK)].copy()
tx["code"] = tx["code"].astype("int64")
tx["_g"] = tx["patient_guid"].map(clean); tx = tx[tx["_g"].isin(set(st["_g"]))]
tx["dba"] = pd.to_numeric(tx["days_before_anchor"], errors="coerce"); tx = tx[tx.dba.notna()]
tx = tx[tx.assertion_status.isin(["present","historical","absent","family"])].copy()
tx["_asrt"] = tx.assertion_status.astype(str).str.lower()
# NO generic-code drop: all 160-arm free-text snomeds are specific 1:1 codes already in the codelist.
_n_in_map = int(tx.code.isin(set(code2cat)).sum())
print(f"free-text kept: {len(tx):,} ({_n_in_map:,} in codelist) | "
      + " ".join(f"{k} {int((tx._asrt==k).sum()):,}" for k in ["present","historical","absent","family"])
      + f" | dur {tx._dur.notna().sum():,} freq {tx._freq.notna().sum():,}", flush=True)

cols = [c for c in st.columns if c != "_g"]
r = pd.DataFrame(index=range(len(tx)))
r["patient_guid"] = g2raw.reindex(tx["_g"].values).values
r["days_before_anchor"] = tx.dba.astype("int64").values
r["snomed_c_t_concept_id"] = tx["code"].astype("int64").values   # REAL snomed_id -> FE maps Code->Category via 160 codelist
r["term"] = tx["name"].astype(str).values
r["event_type"] = "observation"
r["problem_status_description"] = tx.assertion_status.astype(str).values
r["value"] = pd.Series(tx["_concern"].values).map(lambda v: "" if pd.isna(v) else str(int(v))).values
r["event_date"] = pd.to_datetime(tx.event_date, errors="coerce").dt.date.values
r["event_age"] = 0
for c in static: r[c] = stat[c].reindex(tx["_g"].values).values
# free-text enrichment columns (consumed by build_features ASSERTION_FE)
r["ft_assertion"] = tx["_asrt"].values
r["ft_dur"]  = tx["_dur"].values
r["ft_freq"] = tx["_freq"].values
for c in cols:
    if c not in r.columns: r[c] = np.nan
r = r[cols + ["ft_assertion","ft_dur","ft_freq"]]

st_out = st[cols].copy()
st_out["ft_assertion"] = ""; st_out["ft_dur"] = np.nan; st_out["ft_freq"] = np.nan
for c in ["patient_age","age_at_anchor","event_age","days_before_anchor","anchor_year","snomed_c_t_concept_id"]:
    if c in r.columns: r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0).astype("int64")
r["value"] = r["value"].astype(str)
if "event_date" in st_out.columns: st_out["event_date"] = pd.to_datetime(st_out.event_date, errors="coerce").dt.date

# dedup: positive (present/historical) text events that duplicate a structured (patient,code,day) are dropped
# (avoid double-count of the same finding); absent/family kept (separate FE features). Dedup within text too.
key = ["patient_guid","snomed_c_t_concept_id","days_before_anchor"]
r = r.drop_duplicates(subset=key + ["ft_assertion"])
sk = set(map(tuple, st_out[key].itertuples(index=False, name=None)))
pos = r["ft_assertion"].isin(["present","historical"])
dup_pos = pos & pd.Series([t in sk for t in map(tuple, r[key].itertuples(index=False, name=None))], index=r.index)
dropped = int(dup_pos.sum()); r = r[~dup_pos]
merged = pd.concat([st_out, r], ignore_index=True)
print(f"text rows: dropped {dropped:,} positive dup of structured (patient,code,day); kept {len(r):,}", flush=True)
print(f"cache: structured {len(st_out):,} + text {len(r):,} = {len(merged):,} | patients {merged['patient_guid'].nunique()}", flush=True)
merged.to_parquet(CACHEOUT, index=False)
print(f"wrote {CACHEOUT} ({os.path.getsize(CACHEOUT)/1e6:.0f}MB)", flush=True)
