"""STEP 0: build the WITH-text merged lifetime cache for the 12mo (Astra) cohort.
Structured Astra_lifetime  ⊕  filtered text events (present/historical, clean), DEDUP'd on
(patient_guid, snomed_c_t_concept_id, days_before_anchor) keeping structured. Map identical for both arms;
only events differ. progression -> value (worsening=+1, stable=0, improving=-1)."""
import numpy as np, pandas as pd, os, pyarrow.parquet as pq
def _read(p):
    t=pq.read_table(p); return t.replace_schema_metadata(None).to_pandas()  # drop 'dbdate' ext metadata
HERE=os.path.dirname(os.path.abspath(__file__)); B=os.path.join(HERE,"..","..")
SRC=os.path.join(B,"cache/raw_events/Astra_lifetime.parquet")
TXT=os.path.join(HERE,"concept_event_table_snomed_mapped_bb_6_19_26.parquet")
OUT=os.path.join(HERE,"Astra_lifetime_withtext.parquet")
def clean(s): return str(s).replace('"""','').replace('{','').replace('}','').strip()
st=_read(SRC)
st["_g"]=st["patient_guid"].map(clean)
print(f"structured: {len(st):,} rows, {st['_g'].nunique():,} patients, {st['cancer_class'].mean():.3f} cancer-rate(row)")
# per-patient anchor_date = event_date + days_before_anchor  (constant per patient)
ed=pd.to_datetime(st["event_date"],errors="coerce")
st["_anchor"]=ed + pd.to_timedelta(pd.to_numeric(st["days_before_anchor"],errors="coerce"),unit="D")
anchor=st.groupby("_g")["_anchor"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
# per-patient static fields from structured (first row)
static_cols=["cancer_class","cancer_id","sex","patient_ethnicity_16","patient_ethnicity_6","patient_age","age_at_anchor","anchor_year"]
stat=st.groupby("_g").first()[static_cols]
# ---- text events ----
tx=pd.read_parquet(TXT); tx["_g"]=tx["patient_guid"].map(clean)
tx=tx[(tx["assertion_status"].isin(["present","historical"]))&(tx["noise_flag"]=="clean")].copy()
tx=tx[tx["_g"].isin(set(st["_g"]))]                          # only patients present in structured
tx=tx[pd.to_numeric(tx["days_before_anchor"],errors="coerce").notna()]
prog={"getting worse":1,"worsening":1,"stable":0,"improving":-1,"improved":-1}
tx["_val"]=tx["progression"].astype(str).str.strip().map(prog)
print(f"text events (present/historical+clean, in cohort): {len(tx):,}, {tx['_g'].nunique():,} patients; value-coded {tx['_val'].notna().sum():,}")
rows=pd.DataFrame(index=tx.index)
# attach text events to the STRUCTURED raw guid (so they merge into the same patient + match the split)
g2raw=st.groupby("_g")["patient_guid"].first()
rows["patient_guid"]=g2raw.reindex(tx["_g"].values).values
dba=pd.to_numeric(tx["days_before_anchor"],errors="coerce")
rows["days_before_anchor"]=dba.astype("int64").values
rows["snomed_c_t_concept_id"]=pd.to_numeric(tx["snomed_id"],errors="coerce").astype("Int64").astype("int64").values
rows["term"]=tx["snomed_name"].astype(str).values  # term = SNOMED term for the mapped code (snomed_id); the raw extracted phrase is kept in associated_text below
rows["event_type"]="observation"  # FIX: was "text" -> FE used med_code_id (null) -> events dropped
rows["problem_status_description"]=tx["assertion_status"].astype(str).values
rows["significance_description"]=""
rows["problem_comment"]=tx["source_raw_term"].astype(str).values
rows["associated_text"]=tx["concept_name"].astype(str).values
rows["value"]=tx["_val"].map(lambda v: "" if pd.isna(v) else str(int(v))).values
rows["med_code_id"]=np.nan; rows["drug_term"]=""; rows["duration"]=np.nan   # duration is numeric in structured
rows["event_age"]=0
g=tx["_g"].values
rows["_g"]=g
# event_date = anchor_date - days_before_anchor ; static from patient
rows["event_date"]=(anchor.reindex(g).values - pd.to_timedelta(rows["days_before_anchor"].values,unit="D"))
for c in static_cols: rows[c]=stat[c].reindex(g).values
# align columns to structured schema
cols=[c for c in st.columns if c!="_g" and c!="_anchor"]
for c in cols:
    if c not in rows.columns: rows[c]=np.nan
rows=rows[cols]
# normalize types to match structured
rows["event_date"]=pd.to_datetime(rows["event_date"],errors="coerce").dt.date
st_out=st[cols].copy(); st_out["event_date"]=pd.to_datetime(st_out["event_date"],errors="coerce").dt.date
for c in ["patient_age","age_at_anchor","event_age","days_before_anchor","anchor_year","snomed_c_t_concept_id"]:
    rows[c]=pd.to_numeric(rows[c],errors="coerce").fillna(0).astype("int64")
rows["value"]=rows["value"].astype(str)
# DEDUP: only drop TEXT rows whose (patient, code, day) already exists in STRUCTURED; structured untouched.
key=["patient_guid","snomed_c_t_concept_id","days_before_anchor"]
struct_keys=set(map(tuple, st_out[key].itertuples(index=False, name=None)))
rows=rows.drop_duplicates(subset=key, keep="first")                 # dedup text-vs-text first
rk=list(map(tuple, rows[key].itertuples(index=False, name=None)))
keep=[k not in struct_keys for k in rk]
dropped_on_struct=len(rows)-sum(keep)
rows=rows[keep]
merged=pd.concat([st_out,rows],ignore_index=True)
print(f"text rows: dropped {dropped_on_struct:,} that duplicate a structured (patient,code,day); kept {len(rows):,} new text events")
print(f"merged: {len(st_out):,} structured + {len(rows):,} text = {len(merged):,} rows (structured fully preserved)")
merged.to_parquet(OUT,index=False)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)")
