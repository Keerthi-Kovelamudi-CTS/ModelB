#!/usr/bin/env python3
"""Lead-time leakage/signal screen.
For each code, compute its enrichment (lift = P(code|cancer)/P(code|non-cancer)) in
months-before-diagnosis bins, then classify: REACTIVE (leakage), PRODROMAL/EARLY, LIFETIME, WEAK.
1mo data covers months 1-12; 12mo data covers 12-240 -> combined = full profile.
"""
import glob, os, re
import pandas as pd, numpy as np
H = os.path.expanduser("~")
BASE = f"{H}/Breast_cancer_B+C_model/2_Feature_Engineering/data"
cols = ["PATIENT_GUID", "CODE_ID", "TERM", "CATEGORY", "MONTHS_BEFORE_INDEX", "LABEL"]

frames = []
for w in ["1mo", "12mo"]:
    for kind in ["obs", "med"]:
        for f in glob.glob(f"{BASE}/{w}/breast_{w}_{kind}_dropped.parquet"):
            try: frames.append(pd.read_parquet(f, columns=cols))
            except Exception as e: print("  skip", f, e)
df = pd.concat(frames, ignore_index=True).dropna(subset=["CODE_ID", "MONTHS_BEFORE_INDEX"])
df["CODE_ID"] = df["CODE_ID"].astype("Int64").astype(str)
tot_pos = df.loc[df.LABEL == 1, "PATIENT_GUID"].nunique()
tot_neg = df.loc[df.LABEL == 0, "PATIENT_GUID"].nunique()
print(f"patients: cancer={tot_pos} non-cancer={tot_neg} | events={len(df):,}")

BINS = [0, 3, 12, 24, 60, 99999]; LAB = ["0-3", "3-12", "12-24", "24-60", "60+"]
df["bin"] = pd.cut(df["MONTHS_BEFORE_INDEX"].astype(float), bins=BINS, right=False, labels=LAB)
g = df.groupby(["CODE_ID", "bin", "LABEL"], observed=True)["PATIENT_GUID"].nunique().reset_index(name="n")
piv = g.pivot_table(index="CODE_ID", columns=["bin", "LABEL"], values="n", fill_value=0, observed=True)
terms = df.drop_duplicates("CODE_ID").set_index("CODE_ID")["TERM"]
cats  = df.drop_duplicates("CODE_ID").set_index("CODE_ID")["CATEGORY"]

def lift(code, b):
    try: npos = piv.loc[code, (b, 1)]
    except Exception: npos = 0
    try: nneg = piv.loc[code, (b, 0)]
    except Exception: nneg = 0
    pp = (npos + 0.5) / tot_pos; pn = (nneg + 0.5) / tot_neg
    return pp / pn, int(npos)

rows = []
for code in piv.index:
    L = {b: lift(code, b)[0] for b in LAB}
    npos_tot = max(lift(code, b)[1] for b in LAB)
    near = L["0-3"]; mid = (L["12-24"] + L["24-60"]) / 2; far = L["60+"]
    if npos_tot < 30:           cls = "(rare)"
    elif near >= 3 and mid < 1.5 and far < 1.5:  cls = "REACTIVE/LEAK"
    elif mid >= 1.5 and near >= mid * 1.6:        cls = "PRODROMAL"
    elif far >= 1.4 and abs(near - far) < max(near, far) * 0.5: cls = "LIFETIME"
    elif mid >= 1.4 or far >= 1.4:                cls = "EARLY-SIGNAL"
    else:                                          cls = "weak"
    rows.append({"code": code, "cls": cls, "npos": npos_tot,
                 **{f"L_{b}": round(L[b], 2) for b in LAB},
                 "cat": str(cats.get(code, ""))[:18], "term": str(terms.get(code, ""))[:40]})
R = pd.DataFrame(rows)

# annotate: in codelist? in breast_pathway_codes exclusion?
cl = set(pd.read_csv(f"{H}/Breast_cancer_B+C_model/codelist2.0/breast_curated_codes_2.0.csv", sep="\t", dtype=str)["CODE"].astype(str)) if os.path.exists(f"{H}/Breast_cancer_B+C_model/codelist2.0/breast_curated_codes_2.0.csv") else set()
sqltext = open(glob.glob(f"{H}/Breast_cancer_B+C_model/2_Feature_Engineering/SQL/12mo.sql")[0]).read()
m = re.search(r"breast_pathway_codes AS.*?UNNEST\(\[(.*?)\]\)", sqltext, re.S)
pathway = set(re.findall(r"\d{6,}", m.group(1))) if m else set()
R["in_codelist"] = R.code.isin(cl); R["in_pathway_excl"] = R.code.isin(pathway)

print("\n=== (A) REACTIVE/LEAK codes NOT yet excluded (candidates for breast_pathway_codes) ===")
a = R[(R.cls == "REACTIVE/LEAK") & (~R.in_pathway_excl)].sort_values("L_0-3", ascending=False)
print(a[["code","npos","L_0-3","L_3-12","L_12-24","L_24-60","L_60+","in_codelist","cat","term"]].head(15).to_string(index=False))
print("\n=== (B) PRODROMAL / EARLY-SIGNAL codes NOT in codelist (genuine add candidates) ===")
b = R[(R.cls.isin(["PRODROMAL","EARLY-SIGNAL","LIFETIME"])) & (~R.in_codelist) & (~R.in_pathway_excl) & (R.npos>=80)].sort_values("L_24-60", ascending=False)
print(b[["code","cls","npos","L_3-12","L_12-24","L_24-60","L_60+","term"]].head(15).to_string(index=False))
print("\n=== (C) sanity: classification of a few current codelist codes ===")
c = R[R.in_codelist].sort_values("L_0-3", ascending=False)
print(c[["code","cls","L_0-3","L_3-12","L_24-60","cat","term"]].head(12).to_string(index=False))
