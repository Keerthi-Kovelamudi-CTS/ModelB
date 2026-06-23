import pandas as pd, numpy as np, csv, collections, os
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import xgboost as xgb
def clean(s): return str(s).replace('"""','').replace('{','').replace('}','').strip()
V3="../../Lung_Cancer_K1/V3_categorized"
# frozen structured stable matrix (WITHOUT-text arm)
d=pd.read_parquet("structured_stable_12mo.parquet")
d["_g"]=d["patient_guid"].map(clean)
y=d["cancer_class"].astype(int).to_numpy(); spl=d["split"].astype(str).to_numpy()
age=pd.to_numeric(d.get("age_at_prediction"),errors="coerce").to_numpy()
drop=["patient_guid","_g","split","cancer_class"]
Xs=d.drop(columns=[c for c in drop if c in d.columns]).apply(pd.to_numeric,errors="coerce").fillna(0.0)
struct_feats=list(Xs.columns)
# ---- build TEXT feature columns (gap-applied 12mo: days_before_anchor>=365) ----
c2c={str(r["Code"]).strip():r["Category"].strip() for r in csv.DictReader(open(f"{V3}/2_FE/categorized_list_top160cat/Astra_10yr_categories.csv"))}
tx=pd.read_parquet(f"{V3}/2_FE/free_text/concept_event_table_snomed_mapped_bb_6_19_26.parquet")
tx=tx[(tx.assertion_status.isin(["present","historical"]))&(tx.noise_flag=="clean")].copy()
tx["_g"]=tx["patient_guid"].map(clean)
tx["dba"]=pd.to_numeric(tx["days_before_anchor"],errors="coerce")
tx=tx[tx["dba"]>=365]                              # SAME 12mo gap
tx["cat"]=pd.to_numeric(tx["snomed_id"],errors="coerce").astype("Int64").astype(str).map(c2c)
tx=tx.dropna(subset=["cat"])
tx["wor"]=tx["progression"].astype(str).str.strip().isin(["getting worse","worsening"]).astype(int)
cnt=collections.defaultdict(dict); rec=collections.defaultdict(dict); wor=collections.defaultdict(dict)
for (g,cat),grp in tx.groupby(["_g","cat"]):
    cnt[cat][g]=len(grp); rec[cat][g]=grp["dba"].min()/30.0; w=grp["wor"].sum();  wor[cat][g]=w if w>0 else np.nan
idx=d["_g"]
parts=[]
for cat in sorted(cnt):
    parts.append(pd.Series(cnt[cat]).reindex(idx).fillna(0).rename(f"txt::{cat}::count").reset_index(drop=True))
    parts.append(pd.Series(rec[cat]).reindex(idx).fillna(999).rename(f"txt::{cat}::recency").reset_index(drop=True))
    if any(not pd.isna(v) for v in wor[cat].values()):
        parts.append(pd.Series(wor[cat]).reindex(idx).fillna(0).rename(f"txt::{cat}::worsening").reset_index(drop=True))
Xt=pd.concat(parts,axis=1); Xt.index=Xs.index
print(f"structured features (frozen): {len(struct_feats)} | text columns bolted on: {Xt.shape[1]} (covering {len(cnt)} text categories)")
tr=np.isin(spl,["train","valid"]); te=spl=="test"
def evalm(X,tag):
    m=xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
        eval_metric="logloss",n_jobs=8,random_state=42)
    m.fit(X[tr].values,y[tr]); p=m.predict_proba(X[te].values)[:,1]
    au=roc_auc_score(y[te],p); ap=average_precision_score(y[te],p)
    fpr,tpr,_=roc_curve(y[te],p); j=np.argmax(tpr-fpr)
    a=age[te]; aus=[]
    for lo,hi in [(0,55),(55,65),(65,70),(70,75),(75,80),(80,200)]:
        mm=(a>=lo)&(a<hi)
        if mm.sum()>=10 and len(set(y[te][mm]))>1 and y[te][mm].sum()>=3: aus.append(roc_auc_score(y[te][mm],p[mm]))
    print(f"  {tag:<28} feats {X.shape[1]:>5} | AUROC {au:.4f} AUPRC {ap:.4f} | within-age {np.mean(aus):.4f} | Sens {100*tpr[j]:.1f} Spec {100*(1-fpr[j]):.1f}")
    return au,ap,np.mean(aus)
print(f"\n=== CLEAN BOLT-ON ISOLATION (structured FROZEN; same split/XGBoost) test n={te.sum()} ===")
b=evalm(Xs,"STRUCTURED only (baseline')")
t=evalm(pd.concat([Xs,Xt],axis=1),"STRUCTURED + TEXT (bolt-on)")
print(f"\n  Δ AUROC {t[0]-b[0]:+.4f} | Δ AUPRC {t[1]-b[1]:+.4f} | Δ within-age {t[2]-b[2]:+.4f}")

# --- matched-sensitivity specificity (the honest Sens/Spec comparison) ---
print("\n=== MATCHED-SENSITIVITY specificity (structured frozen) ===")
def spec_at_sens(X, targets):
    m=xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
        eval_metric="logloss",n_jobs=8,random_state=42)
    m.fit(X[tr].values,y[tr]); p=m.predict_proba(X[te].values)[:,1]
    fpr,tpr,thr=roc_curve(y[te],p); out={}
    for s in targets:
        i=np.argmin(np.abs(tpr-s)); out[s]=(tpr[i],1-fpr[i])
    return out
B=spec_at_sens(Xs,[0.88,0.90,0.95]); T=spec_at_sens(pd.concat([Xs,Xt],axis=1),[0.88,0.90,0.95])
print(f"  {'target Sens':>11} | {'Spec WITHOUT':>12} | {'Spec WITH':>10} | {'Δ Spec':>7}")
for s in [0.88,0.90,0.95]:
    print(f"  {int(s*100):>10}% | {100*B[s][1]:>11.1f}% | {100*T[s][1]:>9.1f}% | {100*(T[s][1]-B[s][1]):>+6.1f}")
