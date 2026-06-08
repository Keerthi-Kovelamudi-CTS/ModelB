#!/usr/bin/env python3
"""Generalized lead-time leakage/signal screen — all cancers. No new extraction.
Per code: lift (P(code|cancer)/P(code|non-cancer)) across months-before-dx bins ->
classify REACTIVE(leak)/PRODROMAL/EARLY/LIFETIME, and emit window-tuned candidate lists."""
import glob, os, re
import pandas as pd, numpy as np
H = os.path.expanduser("~")
CANCERS = ["Bladder","Prostate","Melanoma","Ovarian","Leukaemia","Lymphoma","Breast"]
BINS=[0,3,12,24,60,99999]; LAB=["0-3","3-12","12-24","24-60","60+"]

def load_codelist(C):
    for p in glob.glob(f"{H}/{C}_cancer_B+C_model/**/codelist2.0/*.csv", recursive=True)+glob.glob(f"{H}/{C}_cancer_B+C_model/codelist2.0/*.csv"):
        try:
            d=pd.read_csv(p,sep="\t",dtype=str)
            if "CODE" in d.columns: return set(d["CODE"].astype(str))
        except Exception: pass
    return set()
def load_pathway(C):
    for sq in glob.glob(f"{H}/{C}_cancer_B+C_model/2_Feature_Engineering/SQL/12mo.sql")+glob.glob(f"{H}/{C}_cancer_B+C_model/SQL Queries/v4/12mo_v4.sql"):
        try:
            m=re.search(r"pathway_codes AS.*?UNNEST\(\[(.*?)\]\)", open(sq).read(), re.S)
            if m: return set(re.findall(r"\d{6,}", m.group(1)))
        except Exception: pass
    return set()

for C in CANCERS:
    base=f"{H}/{C}_cancer_B+C_model/2_Feature_Engineering/data"; cl=C.lower()
    if not os.path.isdir(base): print(f"\n### {C}: no data dir"); continue
    frames=[]
    for w in os.listdir(base):
        if w=="raw": continue
        for kind in ["obs","med"]:
            for f in glob.glob(f"{base}/{w}/{cl}_{w}_{kind}_dropped.parquet"):
                try: frames.append(pd.read_parquet(f, columns=["PATIENT_GUID","CODE_ID","TERM","MONTHS_BEFORE_INDEX","LABEL"]))
                except Exception: pass
    if not frames: print(f"\n### {C}: no parquet"); continue
    df=pd.concat(frames,ignore_index=True).dropna(subset=["CODE_ID","MONTHS_BEFORE_INDEX"])
    df["CODE_ID"]=df["CODE_ID"].astype("Int64").astype(str)
    tp=df.loc[df.LABEL==1,"PATIENT_GUID"].nunique(); tn=df.loc[df.LABEL==0,"PATIENT_GUID"].nunique()
    df["bin"]=pd.cut(df["MONTHS_BEFORE_INDEX"].astype(float),bins=BINS,right=False,labels=LAB)
    piv=df.groupby(["CODE_ID","bin","LABEL"],observed=True)["PATIENT_GUID"].nunique().reset_index(name="n").pivot_table(index="CODE_ID",columns=["bin","LABEL"],values="n",fill_value=0,observed=True)
    terms=df.drop_duplicates("CODE_ID").set_index("CODE_ID")["TERM"]
    def lf(code,b):
        np_=piv.loc[code,(b,1)] if (b,1) in piv.columns else 0
        nn_=piv.loc[code,(b,0)] if (b,0) in piv.columns else 0
        return ((np_+0.5)/tp)/((nn_+0.5)/tn), int(np_)
    cset=load_codelist(C); pw=load_pathway(C); rows=[]
    for code in piv.index:
        Ls={b:lf(code,b)[0] for b in LAB}; npos=max(lf(code,b)[1] for b in LAB)
        near=Ls["0-3"]; mid=(Ls["12-24"]+Ls["24-60"])/2; far=Ls["60+"]
        if npos<30: c="rare"
        elif near>=3 and mid<1.5 and far<1.5: c="REACTIVE"
        elif mid>=1.5 and near>=mid*1.6: c="PRODROMAL"
        elif far>=1.4 and mid>=1.4: c="LIFETIME"
        elif mid>=1.4 or far>=1.4: c="EARLY"
        else: c="weak"
        rows.append({"code":code,"cls":c,"npos":npos,"near":near,"far":far,"incl":code in cset,"inpw":code in pw,"term":str(terms.get(code,""))[:34]})
    R=pd.DataFrame(rows)
    leak=R[(R.cls=="REACTIVE")&(~R.inpw)&(R.npos>=50)].sort_values("near",ascending=False)
    miss=R[(R.cls.isin(["PRODROMAL","EARLY","LIFETIME"]))&(~R.incl)&(~R.inpw)&(R.npos>=80)].sort_values("far",ascending=False)
    print(f"\n### {C}  cancer={tp} noncancer={tn} | codelist={len(cset)} pathway_excl={len(pw)}")
    print(f"   leak-candidates(reactive,not-excl)={len(leak)}  genuine-missing(not-in-codelist)={len(miss)}")
    if len(leak): print("   → top leak:", " | ".join(f"{r.term}(L0-3={r.near:.0f})" for _,r in leak.head(3).iterrows()))
    if len(miss): print("   → genuine-missing:", " | ".join(f"{r.term}(L24-60={r.far:.1f})" for _,r in miss.head(3).iterrows()))
    w12=R[(R.npos>=50)&(~R.inpw)].sort_values("far",ascending=False).head(4)
    print("   → 12mo-window top (far-lift, the early-warning signal):", " | ".join(f"{r.term}({r.far:.1f})" for _,r in w12.iterrows()))
