#!/usr/bin/env python3
"""Lead-time screen v2.1 — CORRECTED leakage rule (2026-06-08).
A near-dx spike is NOT leakage by itself. Leakage = the DIAGNOSIS PROCESS (biopsy/referral/
imaging/surgery/the cancer code), identified by TERM. Symptoms/labs/risk-factors that spike
near dx are KEPT (legitimate early-detection signal). Featurization ground-truth = CATEGORY non-null."""
import glob, os, re
import pandas as pd, numpy as np
H = os.path.expanduser("~")
CANCERS = ["Bladder","Prostate","Melanoma","Ovarian","Leukaemia","Lymphoma","Breast"]
BINS=[0,3,12,24,60,99999]; LAB=["0-3","3-12","12-24","24-60","60+"]
# diagnosis-PROCESS / outcome terms -> genuine leakage (regardless of timing)
PROC = re.compile(r"biops|\breferr|mammog|screen|excis|histolog|cytolog|aspirat|\bFNA\b|ultrasound|"
                  r"\bMRI\b|\bCT scan|imaging|resection|operation|surgery|2 ?week|fast.?track|colposcop|"
                  r"cystoscop|endoscop|radiothe|chemother|reconstruct|carcinoma|neoplasm|in situ|metastat", re.I)

for C in CANCERS:
    base=f"{H}/{C}_cancer_B+C_model/2_Feature_Engineering/data"; cl=C.lower()
    if not os.path.isdir(base): print(f"\n### {C}: no data"); continue
    fr=[]
    for w in os.listdir(base):
        if w=="raw": continue
        for kind in ["obs","med"]:
            for f in glob.glob(f"{base}/{w}/{cl}_{w}_{kind}_dropped.parquet"):
                try: fr.append(pd.read_parquet(f, columns=["PATIENT_GUID","CODE_ID","TERM","CATEGORY","MONTHS_BEFORE_INDEX","LABEL"]))
                except Exception: pass
    if not fr: print(f"\n### {C}: no parquet"); continue
    df=pd.concat(fr,ignore_index=True).dropna(subset=["CODE_ID","MONTHS_BEFORE_INDEX"])
    df["CODE_ID"]=df["CODE_ID"].astype("Int64").astype(str)
    tp=df.loc[df.LABEL==1,"PATIENT_GUID"].nunique(); tn=df.loc[df.LABEL==0,"PATIENT_GUID"].nunique()
    df["bin"]=pd.cut(df["MONTHS_BEFORE_INDEX"].astype(float),bins=BINS,right=False,labels=LAB)
    piv=df.groupby(["CODE_ID","bin","LABEL"],observed=True)["PATIENT_GUID"].nunique().reset_index(name="n").pivot_table(index="CODE_ID",columns=["bin","LABEL"],values="n",fill_value=0,observed=True)
    info=df.drop_duplicates("CODE_ID").set_index("CODE_ID")
    feat_codes=set(df.loc[df["CATEGORY"].notna() & (df["CATEGORY"].astype(str).str.strip()!="") & (df["CATEGORY"].astype(str)!="None"),"CODE_ID"].unique())
    def lf(code,b):
        a=piv.loc[code,(b,1)] if (b,1) in piv.columns else 0
        z=piv.loc[code,(b,0)] if (b,0) in piv.columns else 0
        return ((a+0.5)/tp)/((z+0.5)/tn), int(a)
    rows=[]
    for code in piv.index:
        Ls={b:lf(code,b)[0] for b in LAB}; npos=max(lf(code,b)[1] for b in LAB)
        near=Ls["0-3"]; mid=(Ls["12-24"]+Ls["24-60"])/2; far=Ls["60+"]
        term=str(info.loc[code,"TERM"]); is_proc=bool(PROC.search(term))
        spike = near>=3 and mid<1.5 and far<1.5
        if npos<50: c="rare"
        elif spike and is_proc:                 c="REACTIVE-LEAK"     # process-like + near-dx spike => exclude
        elif mid>=1.5 and near>=mid*1.6:         c="PRODROMAL"
        elif far>=1.4 and mid>=1.4:              c="LIFETIME"
        elif mid>=1.4 or far>=1.4:               c="EARLY"
        elif spike:                              c="near-dx-symptom"  # spikes but NOT process-like => KEEP (presenting sign)
        else:                                    c="weak"
        rows.append({"code":code,"cls":c,"npos":npos,"near":round(near,1),"mid":round(mid,1),"far":round(far,1),
                     "proc":is_proc,"feat":code in feat_codes,"term":term[:36]})
    R=pd.DataFrame(rows)
    leak=R[(R.cls=="REACTIVE-LEAK") & (R.feat)].sort_values("near",ascending=False)   # leakage that is currently a FEATURE = should exclude
    kept_sx=R[R.cls=="near-dx-symptom"]
    print(f"\n### {C}  cancer={tp} non={tn} | codes: prodromal/early={len(R[R.cls.isin(['PRODROMAL','EARLY','LIFETIME'])])} near-dx-symptom(KEPT)={len(kept_sx)} REACTIVE-LEAK(feat)={len(leak)}")
    if len(leak):
        print("   ⚠ process-like leakage currently FEATURIZED (review for exclusion):")
        for _,r in leak.head(6).iterrows(): print(f"     near={r.near:>5} far={r.far:>4}  {r.term}")
    if len(kept_sx):
        print("   ✓ near-dx symptoms KEPT (not leakage):", " | ".join(f"{r.term}" for _,r in kept_sx.sort_values('near',ascending=False).head(4).iterrows()))
