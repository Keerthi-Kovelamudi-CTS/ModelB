import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, csv, collections, os
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import xgboost as xgb
def clean(s): return str(s).replace('"""','').replace('{','').replace('}','').strip()
B="."
MAP=os.path.join(B,"2_FE/categorized_list_top160cat/Astra_10yr_categories.csv")
EV=os.path.join(B,"cache/raw_events/Astra_lifetime.parquet")
TXT=os.path.join(B,"2_FE/free_text/concept_event_table_snomed_mapped_bb_6_19_26.parquet")
SP=os.path.join(B,"cache/splits/lung_12mo_split.parquet")
# code->category (merged map)
c2c={}
for r in csv.DictReader(open(MAP)): c2c[str(r["Code"]).strip()]=r["Category"].strip()
# split + class + cohort = text-covered patients
sp=pd.read_parquet(SP); sp["_g"]=sp["patient_guid"].map(clean)
g2split=dict(zip(sp["_g"],sp["split"].astype(str)))
tt=pq.read_table(TXT,columns=["patient_guid","cancer_class"]).to_pandas(); tt["_g"]=tt["patient_guid"].map(clean)
g2cls=dict(zip(tt["_g"],tt["cancer_class"].astype(int)))
cohort=sorted(set(tt["_g"]) & set(g2split))
cohort_set=set(cohort)
print(f"cohort (text-covered ∩ split): {len(cohort)}")
# age
at=pq.read_table(EV,columns=["patient_guid","age_at_anchor"])
ag={}
for gg,aa in zip(at.column("patient_guid").to_pylist(),at.column("age_at_anchor").to_pylist()):
    if aa is not None: ag[clean(gg)]=aa
# accumulate per (guid,cat): [count, min_days]  -- STRUCTURED
acc=collections.defaultdict(lambda:[0,1e9])
pf=pq.ParquetFile(EV)
for b in pf.iter_batches(columns=["patient_guid","snomed_c_t_concept_id","days_before_anchor"],batch_size=500000):
    gs=b.column("patient_guid").to_pylist(); cs=b.column("snomed_c_t_concept_id").cast(pa.string()).to_pylist(); ds=b.column("days_before_anchor").to_pylist()
    for gg,cc,dd in zip(gs,cs,ds):
        cg=clean(gg)
        if cg not in cohort_set or dd is None or dd<365: continue   # 12mo gap
        cat=c2c.get(cc)
        if cat is None: continue
        a=acc[(cg,cat)]; a[0]+=1; a[1]=min(a[1],dd)
struct_keys=set(acc.keys())
# TEXT events (present/historical, clean) -> accumulate into a COPY for the with-text arm
acc_t=collections.defaultdict(lambda:[0,1e9,0])   # count,min_days,worsening
tx=pq.read_table(TXT,columns=["patient_guid","snomed_id","days_before_anchor","assertion_status","noise_flag","progression"]).to_pandas()
tx=tx[(tx["assertion_status"].isin(["present","historical"]))&(tx["noise_flag"]=="clean")]
for _,r in tx.iterrows():
    cg=clean(r["patient_guid"]); dd=r["days_before_anchor"]
    if cg not in cohort_set or pd.isna(dd) or dd<365: continue
    cat=c2c.get(str(r["snomed_id"]).strip())
    if cat is None: continue
    wr=1 if str(r["progression"]).strip() in ("getting worse","worsening") else 0
    a=acc_t[(cg,cat)]; a[0]+=1; a[1]=min(a[1],dd); a[2]+=wr
def build(with_text):
    cats=set(c for _,c in struct_keys)
    if with_text: cats|=set(c for _,c in acc_t.keys())
    rows={}
    for g in cohort:
        d={}
        for (gg,cat),v in []:  # placeholder
            pass
        rows[g]=d
    # build per category
    df=pd.DataFrame(index=cohort)
    # merge struct + (text) per (g,cat)
    comb=collections.defaultdict(lambda:[0,1e9,0])
    for (g,cat),v in acc.items(): c=comb[(g,cat)]; c[0]+=v[0]; c[1]=min(c[1],v[1])
    if with_text:
        for (g,cat),v in acc_t.items(): c=comb[(g,cat)]; c[0]+=v[0]; c[1]=min(c[1],v[1]); c[2]+=v[2]
    cnt=collections.defaultdict(dict); rec=collections.defaultdict(dict); wor=collections.defaultdict(dict)
    for (g,cat),v in comb.items():
        cnt[cat][g]=v[0]; rec[cat][g]=v[1]/30.0; 
        if with_text and v[2]>0: wor[cat][g]=v[2]
    parts=[]
    for cat in sorted(cats):
        parts.append(pd.Series(cnt[cat]).reindex(cohort).fillna(0).rename(f"{cat}_count"))
        parts.append(pd.Series(rec[cat]).reindex(cohort).fillna(999).rename(f"{cat}_recency"))
    if with_text:
        for cat in sorted(set(wor.keys())):
            parts.append(pd.Series(wor[cat]).reindex(cohort).fillna(0).rename(f"{cat}_txt_worsening"))
    return pd.concat(parts,axis=1).fillna(0)
y=np.array([g2cls[g] for g in cohort]); spl=np.array([g2split[g] for g in cohort])
age=np.array([ag.get(g,np.nan) for g in cohort],float)
tr=(spl=="train")|(spl=="valid"); te=spl=="test"
def evalm(X,tag):
    Xtr,Xte=X[tr.tolist()].values,X[te.tolist()].values; ytr,yte=y[tr],y[te]
    m=xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
        eval_metric="logloss",n_jobs=8,scale_pos_weight=(ytr==0).sum()/max(1,(ytr==1).sum()),random_state=42)
    m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]; au=roc_auc_score(yte,p); ap=average_precision_score(yte,p)
    fpr,tpr,_=roc_curve(yte,p); j=np.argmax(tpr-fpr)
    a_te=age[te]; aus=[]
    for lo,hi in [(0,55),(55,65),(65,70),(70,75),(75,80),(80,200)]:
        mm=(a_te>=lo)&(a_te<hi)
        if mm.sum()>=10 and len(set(yte[mm]))>1 and yte[mm].sum()>=3: aus.append(roc_auc_score(yte[mm],p[mm]))
    print(f"  {tag:<24} feats {X.shape[1]:>5}  AUROC {au:.4f}  AUPRC {ap:.4f}  Sens {100*tpr[j]:.1f}  Spec {100*(1-fpr[j]):.1f}  within-age {np.mean(aus):.4f}")
print(f"train {tr.sum()} test {te.sum()} | cancer(test) {y[te].sum()}\n=== 12mo with/without TEXT (controlled local, internal test) ===")
Xw=build(False); evalm(Xw,"WITHOUT text")
Xt=build(True);  evalm(Xt,"WITH text (occ+worsening)")
