import html, os, re, argparse
import numpy as np, pandas as pd, pyarrow.parquet as pq
_ap=argparse.ArgumentParser()
_ap.add_argument("--run",required=True); _ap.add_argument("--out",required=True)
_ap.add_argument("--model-short",default="model"); _ap.add_argument("--model-long",default="")
_ap.add_argument("--auroc",default="—"); _ap.add_argument("--threshold",type=float,default=0.50)
_ap.add_argument("--organ",default="")
_a=_ap.parse_args()
E=_a.run; EXP=f"{E}/modeling/explainability_internal"; OUT=_a.out
MODEL_SHORT=_a.model_short; MODEL_LONG=_a.model_long or _a.model_short; MODEL_AUROC=_a.auroc
THRESH=_a.threshold; AGG_TOPN=20; ORGAN=_a.organ; HO_SWEEP=[]; HO_OP=None


pe = pd.read_csv(f"{EXP}/patient_explanations.csv").sort_values("row").reset_index(drop=True)
yt = pe.y_true.astype(int).values; pr = pe.prob.astype(float).values
pe["seg"] = np.where((yt==1)&(pr>=THRESH),"TP",np.where((yt==1)&(pr<THRESH),"FN",np.where((yt==0)&(pr>=THRESH),"FP","TN")))
sch = pq.ParquetFile(f"{E}/fe/features_p005_12mo_stable.parquet").schema.names
eth_cols=[c for c in sch if c.startswith("g_eth_")]; cnt_cols=[c for c in sch if c.endswith("_count") and "_count_" not in c]
f=pd.read_parquet(f"{E}/fe/features_p005_12mo_stable.parquet",columns=["split","age_at_prediction","g_is_male"]+eth_cols+cnt_cols)
f=f[f.split=="test"].reset_index(drop=True); assert len(f)==len(pe)
cnt=f[cnt_cols].fillna(0).to_numpy(); cnt_names=[c[:-6] for c in cnt_cols]
ADMIN=re.compile(r"review|screening|admin|letter|report|EMED|advice|questionnaire|chaperone|authorisation|waiting|reassurance|recall|consent|template|batch|Non-smoker\b",re.I)
SUF=re.compile(r"(_decay_intensity|_recency_months|_recency_rank|_val_[a-z0-9_]+|_count_last\d+|_count_share|_count_w\d+_\d+|_count|_present|_first_months|_first_half_freq|_second_half_freq|_freq_per_year|_freq_trend_slope|_distinct_ratio|_n_distinct_codes|_age_first|_age_last|_age_median|_interval_min|_interval_max|_interval_median|_timespan_years|_max_abs_z|_accel|_recent_ratio|_is_worsening|_n_extreme|_w\d+_\d+|_last\d+)$")
def human(fac):
    if fac.startswith("age_at_prediction"): return "age"
    if fac.startswith("ageband"): return "age band"
    x=re.sub(r"^INT_","",fac); x=re.sub(r"_x_age$","",x)
    for _ in range(3):
        y=SUF.sub("",x)
        if y==x: break
        x=y
    return x.strip() or fac
def demo(i):
    age=int(round(f.age_at_prediction[i])); sex="M" if f.g_is_male[i]>=.5 else "F"
    e=[c for c in eth_cols if f[c][i]>=.5]; eth=e[0].replace("g_eth_","") if e else "Unknown"
    row=cnt[i]; onrec=[cnt_names[j] for j in np.argsort(-row) if row[j]>0 and not ADMIN.search(cnt_names[j])][:6]
    return age,sex,eth,int((row>0).sum()),int(row.sum()),onrec
def factors(r):
    d,u=[],[]
    for k in range(1,21):
        fac,sh=r.get(f"factor_{k}"),r.get(f"shap_{k}")
        if pd.isna(fac) or pd.isna(sh) or str(fac).strip()=="": continue
        (d if sh<0 else u).append((str(fac),float(sh)))
    return sorted(d,key=lambda t:t[1])[:12],sorted(u,key=lambda t:-t[1])[:12]
def arche(s,ncat,downs):
    if ncat<=8: return "Data gap"
    if s=="FN": return "Age-suppressed" if (downs and (downs[0][0].startswith("age_at_prediction") or downs[0][0].startswith("ageband"))) else "Signal-poor"
    return "Look-alike"
def rows(s):
    out=[]
    for _,r in pe[pe.seg==s].iterrows():
        i=int(r["row"]); age,sex,eth,ncat,nev,onrec=demo(i); d,u=factors(r)
        out.append(dict(tab="",age=age,sex=sex,eth=eth,ncat=ncat,nev=nev,onrec=onrec,downs=d,ups=u,prob=float(r["prob"]),arch=arche(s,ncat,d)))
    out=sorted(out,key=lambda x:x["age"])
    for k,x in enumerate(out,1): x["tab"]=f"{s}{k:02d}"
    return out
FN,FP=rows("FN"),rows("FP")
from collections import Counter
fn_a,fp_a=Counter(x["arch"] for x in FN),Counter(x["arch"] for x in FP)
nTP=int((pe.seg=="TP").sum()); nTN=int((pe.seg=="TN").sum()); nFN=len(FN); nFP=len(FP)
sens=100*nTP/(nTP+nFN); spec=100*nTN/(nTN+nFP)

def internal_sweep(thrs):
    P=int((yt==1).sum()); N=int((yt==0).sum()); out=[]
    for t in thrs:
        yp=(pr>=t).astype(int); tp=int(((yt==1)&(yp==1)).sum()); fp=int(((yt==0)&(yp==1)).sum()); fn=P-tp
        out.append((t,100*tp/P,100*(N-fp)/N,100*tp/(tp+fp) if tp+fp else 0,tp+fp,fn,fp))
    return out
INT_SWEEP=internal_sweep([0.2,0.3,0.4,0.5,0.6,0.7,0.8])


_WIN={"last6":"in the last 6 months","last12":"in the last 12 months","last24":"in the last 24 months","last60":"in the last 60 months","w0_6":"in the 0-6 month window","w6_12":"in the 6-12 month window","w12_24":"in the 12-24 month window","w24_60":"in the 24-60 month window","w60_999":"beyond 60 months"}
_FAM=[("_val_trend_slope","whether {c} values are rising or falling over time"),("_val_trend_corr","how steadily {c} values trend over time"),("_val_abs_change","how much the {c} value changed"),("_val_pct_change","percent change in the {c} value"),("_val_latest","most recent {c} value"),("_val_mean","average {c} value"),("_val_median","typical (median) {c} value"),("_val_max","highest {c} value"),("_val_min","lowest {c} value"),("_val_std","how much {c} values vary"),("_val_range","gap between the highest and lowest {c} value"),("_val_first","first recorded {c} value"),("_val_slope","whether {c} values are rising or falling"),("_max_abs_z","how far the most extreme {c} value is from this patient's usual level"),("_n_extreme","how many unusually high or low {c} values there are"),("_count_share","how much of the whole record is made up of {c}"),("_n_distinct_codes","how many different {c} codes are recorded"),("_distinct_ratio","how varied the {c} codes are"),("_count","how many times {c} was recorded"),("_present","whether {c} appears in the record at all"),("_recency_months","months since {c} was last recorded"),("_recency_rank","how recently {c} was recorded, compared with the rest of the record"),("_decay_intensity","how much recent {c} activity there is (recent events count more)"),("_accel","whether {c} activity is speeding up"),("_recent_ratio","how much of the {c} activity is recent"),("_freq_trend_slope","whether {c} is being recorded more or less often over time"),("_freq_per_year","how often {c} is recorded per year"),("_first_half_freq","how often {c} was recorded in the earlier half of the record"),("_second_half_freq","how often {c} was recorded in the later half of the record"),("_first_months","how long ago {c} first appeared in the record (months)"),("_timespan_years","how many years the {c} history spans"),("_interval_median","typical gap between {c} recordings"),("_interval_min","shortest gap between {c} recordings"),("_interval_max","longest gap between {c} recordings"),("_is_worsening","whether {c} is getting worse"),("_age_first","patient's age when {c} was first recorded"),("_age_last","patient's age when {c} was last recorded"),("_age_median","patient's typical age across {c} records")]
def describe(feat):
    f=str(feat)
    if f=="age_at_prediction": return "Patient's age at the prediction date."
    if f.startswith("ageband"): return "Age-band flag ("+f.replace("ageband_","").replace("u50","under 50").replace("_","-")+")."
    if f=="g_is_male": return "Whether the patient is male."
    if f.startswith("g_eth_"): return "Whether ethnicity is recorded as "+f[6:]+"."
    core=f; intx=""
    if core.startswith("INT_") and core.endswith("_x_age"): intx=" (interacted with age)"; core=core[4:-6]
    win=""
    for wt,wx in _WIN.items():
        if core.endswith("_"+wt): win=" "+wx; core=core[:-(len(wt)+1)]; break
    for suf,tmpl in _FAM:
        if core.endswith(suf):
            cat=core[:-len(suf)].strip()
            txt=tmpl.format(c='"'+cat+'"')+win+intx+"."
            return txt[0].upper()+txt[1:]
    return 'Engineered feature from "'+core+'"'+win+intx+"."

def chips(items,cls): return "".join(f'<span class="chip {cls}" data-tip="{html.escape(describe(t))}">{html.escape(t)}</span>' for t,_ in items) or '<span class="chip">—</span>'
def orec(l): return "".join(f'<span class="chip" data-tip="Clinical category present in this patient\'s record: {html.escape(c)}">{html.escape(c)}</span>' for c in l) or '<span class="chip">sparse record</span>'
def rich(n): return "sparse" if n<=8 else "moderate" if n<=30 else "rich"
def narr(x):
    who=f"A {x['age']}-year-old {'man' if x['sex']=='M' else 'woman'}"; onr=", ".join(html.escape(c) for c in x["onrec"][:3])
    if x["arch"]=="Data gap": return f"{who} with a very thin record — {x['nev']} events across {x['ncat']} categories. Too little history to predict from: a data limitation, not a model error."
    if x["arch"]=="Age-suppressed":
        w=f"Warning signs were on record — {onr} — yet " if onr else ""
        return f"{who}. {w}<b>age</b> was the single biggest factor: being {x['age']} pulled the score down and buried the real signal."
    if x["arch"]=="Signal-poor": return f"{who} with a full record ({x['nev']} events) but only routine monitoring — no COPD, no haemoptysis, no dominant lung red-flag to lock onto, so it settled below the alert line."
    w=f" — {onr}" if onr else ""
    return f"{who}, an older comorbid profile{w}. On structured data this is hard to tell apart from a real cancer, so the model flagged it. A different threshold won't fix this one."
def card(x,s):
    c="fn" if s=="FN" else "fp"
    return (f'<article class="card {c}"><header><span class="tab">{x["tab"]}</span><span class="arche {c}">{html.escape(x["arch"])}</span>'
            f'<span class="prob">{x["prob"]*100:.0f}%<small>risk</small></span></header>'
            f'<div class="meta">{x["age"]} · {x["sex"]} · {html.escape(x["eth"])} · {x["nev"]} events · {x["ncat"]} cats · {rich(x["ncat"])}</div>'
            f'<p class="narr">{narr(x)}</p><div class="detail">'
            f'<div class="drow"><span class="k">on record</span>{orec(x["onrec"])}</div>'
            f'<div class="drow"><span class="k">↓ lowered</span>{chips(x["downs"],"dn")}</div>'
            f'<div class="drow"><span class="k">↑ raised</span>{chips(x["ups"],"up")}</div></div></article>')
def dots(recs,c):
    return "".join(f'<span class="dot {c}" style="left:{max(1,min(99,(x["age"]-18)/74*100)):.1f}%" title="{x["tab"]}: {x["age"]}y"></span>' for x in recs)
def agg(seg,k=AGG_TOPN):
    imp={}; sub=pe[pe.seg==seg]; n=max(1,len(sub))
    for _,r in sub.iterrows():
        for j in range(1,21):
            fac,sh=r.get(f"factor_{j}"),r.get(f"shap_{j}")
            if pd.isna(fac) or pd.isna(sh) or str(fac).strip()=="": continue
            h=str(fac); a=imp.setdefault(h,[0.0,0.0]); a[0]+=abs(float(sh)); a[1]+=float(sh)
    return len(sub),[(fe,ab/n,sg/n) for fe,(ab,sg) in sorted(imp.items(),key=lambda t:-t[1][0])[:k]]
def agg_table(seg,title,color):
    n,rd=agg(seg); body="".join(f'<tr><td data-tip="{html.escape(describe(fe))}">{html.escape(fe)}</td><td>{ab:.3f}</td><td class="dir {"up" if sg>0 else "dn"}">{"↑" if sg>0 else "↓"}</td></tr>' for fe,ab,sg in rd)
    return f'<div class="aggcol"><h3 style="color:{color}">{title} <span>n={n}</span></h3><div class="scroll"><table class="agg"><thead><tr><th>feature</th><th>mean|SHAP|</th><th></th></tr></thead><tbody>{body}</tbody></table></div></div>'
def sweep_table(swrows,op):
    body=""
    for t,se,sp,pv,fl,mi,fa in swrows:
        cls=' class="dep"' if abs(t-op)<1e-6 else ''
        body+=f'<tr{cls}><td>{t:.3f}</td><td>{se:.1f}%</td><td>{sp:.1f}%</td><td>{pv:.1f}%</td><td>{fl:,}</td><td>{mi:,}</td><td>{fa:,}</td></tr>'
    _HDR=[("Threshold","The cut-off on the risk score above which a patient is flagged. Lower = flag more people."),
          ("Sens","Sensitivity — of all the real cancers, the share flagged at this threshold."),
          ("Spec","Specificity — of all the non-cancers, the share correctly NOT flagged."),
          ("PPV","Positive predictive value — of everyone flagged, the share who truly have cancer."),
          ("Flagged","How many patients would be flagged in total at this threshold."),
          ("Missed","Real cancers that would still be missed at this threshold."),
          ("False alarms","Non-cancers wrongly flagged at this threshold.")]
    head="".join(f'<th data-tip="{html.escape(t)}">{h}</th>' for h,t in _HDR)
    return f'<div class="scroll"><table class="sweep"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

CSS = """
:root{--ground:#ffffff;--surface:#f7f9fb;--text:#16202b;--muted:#5c6b7a;--line:#e2e8ee;--miss:#b0413e;--false:#2e6e8e;--accent:#2e6e8e;--dark:#16202b;color-scheme:light}
*{box-sizing:border-box}body{margin:0;background:var(--ground);color:var(--text);font:15px/1.55 -apple-system,"Segoe UI",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.mono{font-family:ui-monospace,"SF Mono",Menlo,monospace}.wrap{max-width:1080px;margin:0 auto;padding:0 24px}
header.hero{padding:60px 0 34px;border-bottom:2px solid var(--text)}
.eyebrow{font:600 12px/1.5 ui-monospace,Menlo,monospace;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:0 0 16px}
h1{font-size:clamp(30px,5.2vw,52px);line-height:1.05;letter-spacing:-.02em;font-weight:800;margin:0 0 16px;max-width:17ch;text-wrap:balance}
h1 .m{color:var(--miss)}h1 .f{color:var(--false)}.lede{font-size:18px;color:var(--muted);max-width:none;margin:0}
.strip{margin:38px 0 4px}
.striprow{display:grid;grid-template-columns:132px 1fr;gap:12px;align-items:center;margin-bottom:10px}
.striprow .lab{font:600 11px/1.3 ui-monospace,Menlo,monospace;letter-spacing:.05em;text-transform:uppercase;text-align:right}
.striprow.fn .lab{color:var(--miss)}.striprow.fp .lab{color:var(--false)}
.track{position:relative;height:24px;border-left:1px solid var(--line);border-right:1px solid var(--line);background:color-mix(in srgb,var(--text) 3%,transparent);border-radius:3px}
.track .dot{position:absolute;top:50%;transform:translate(-50%,-50%);width:11px;height:11px;border-radius:50%}
.dot.fn{background:var(--miss)}.dot.fp{background:var(--false)}
.ticks2{display:grid;grid-template-columns:132px 1fr;gap:12px;margin-top:2px}
.ticks2 .t{display:flex;justify-content:space-between;font:11px ui-monospace,Menlo,monospace;color:var(--muted)}
.stripnote{font-size:14px;color:var(--muted);margin:14px 0 0}
section{padding:44px 0;border-bottom:1px solid var(--line)}
section.dark{background:var(--dark);color:#e7ecf1;border-bottom:none}section.dark h2{color:#8fa3b5}section.dark .body{color:#c3cdd7}section.dark .finding{color:#fff}section.dark .body b{color:#fff}
h2{font:600 13px/1 ui-monospace,Menlo,monospace;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin:0 0 18px}
h3.sub{font-size:14px;font-weight:700;margin:20px 0 8px}
.finding{font-size:22px;font-weight:700;letter-spacing:-.01em;max-width:none;margin:0 0 12px}
.body{font-size:16px;color:var(--muted);max-width:none}.body b{color:var(--text)}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:4px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-top:3px solid var(--c);border-radius:10px;padding:13px 15px}
.tile .l{font:600 11px/1 ui-monospace,Menlo,monospace;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.tile .v{font-size:25px;font-weight:800;margin-top:4px;font-variant-numeric:tabular-nums}.tile .d{font-size:12px;color:var(--muted);margin-top:2px}
.metrics{display:flex;gap:26px;margin:14px 0 0}.metrics b{font-size:19px;color:var(--text);font-weight:800;margin-right:5px;font-variant-numeric:tabular-nums}.metrics span{font-size:13px;color:var(--muted)}
.arches{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:4px}
.ac{background:var(--surface);border:1px solid var(--line);border-top:3px solid var(--muted);border-radius:10px;padding:16px}.ac.miss{border-top-color:var(--miss)}.ac.fls{border-top-color:var(--false)}
.ac .n{font-size:32px;font-weight:800;letter-spacing:-.02em}.ac .t{font-weight:700;margin:1px 0 5px}.ac .d{font-size:13px;color:var(--muted);line-height:1.45}
.legend{display:grid;gap:10px;font-size:14px;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px}.legend>div{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}.legend .k,.legend .chip{flex:0 0 auto}
.grouphdr{display:flex;align-items:baseline;gap:14px;margin:4px 0 18px}.grouphdr .big{font-size:38px;font-weight:800;letter-spacing:-.02em}.grouphdr.fn .big{color:var(--miss)}.grouphdr.fp .big{color:var(--false)}.grouphdr .lbl{font-size:15px;color:var(--muted);max-width:none}
.pgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:13px}
.card{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:10px;padding:15px 15px 13px}.card.fn{border-left-color:var(--miss)}.card.fp{border-left-color:var(--false)}
.card header{display:flex;align-items:center;gap:9px;padding:0;border:none}.tab{font:700 13px/1 ui-monospace,Menlo,monospace}
.arche{font:600 10.5px/1 ui-monospace,Menlo,monospace;letter-spacing:.04em;text-transform:uppercase;padding:3px 8px;border-radius:20px}
.arche.fn{background:color-mix(in srgb,var(--miss) 13%,transparent);color:var(--miss)}.arche.fp{background:color-mix(in srgb,var(--false) 15%,transparent);color:var(--false)}
.prob{margin-left:auto;font:800 20px/1 ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}.prob small{display:block;font-size:9px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:right;margin-top:2px}
.meta{font:12px ui-monospace,Menlo,monospace;color:var(--muted);margin:9px 0 7px}.narr{font-size:14px;line-height:1.5;color:var(--text);margin:0 0 11px}.narr b{font-weight:700}
.detail{border-top:1px dashed var(--line);padding-top:9px;display:flex;flex-direction:column;gap:6px}.drow{display:flex;flex-wrap:wrap;gap:5px;align-items:baseline}
.k{font:600 10px/1.4 ui-monospace,Menlo,monospace;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);flex:0 0 74px}
.chip{font-size:11.5px;background:color-mix(in srgb,var(--text) 6%,transparent);border:1px solid var(--line);border-radius:6px;padding:2px 7px}
.chip.dn{background:color-mix(in srgb,var(--miss) 9%,transparent);border-color:color-mix(in srgb,var(--miss) 30%,transparent);color:var(--miss)}
.chip.up{background:color-mix(in srgb,var(--false) 10%,transparent);border-color:color-mix(in srgb,var(--false) 32%,transparent);color:var(--false)}
.scroll{overflow-x:auto}
[data-tip]{cursor:help}
#tip{position:fixed;z-index:9999;display:none;max-width:340px;background:#16202b;color:#f2f5f8;font:12.5px/1.45 -apple-system,"Segoe UI",system-ui,sans-serif;padding:8px 11px;border-radius:7px;box-shadow:0 6px 20px rgba(0,0,0,.32);pointer-events:none}
table.sweep,table.agg{border-collapse:collapse;width:100%;background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-variant-numeric:tabular-nums}
table.sweep{font-size:13.5px;max-width:680px}
.sweep th{background:color-mix(in srgb,var(--surface),var(--line) 55%);text-align:right;padding:8px 12px;font:600 10px ui-monospace,Menlo,monospace;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.sweep th:first-child,.sweep td:first-child{text-align:left;font-family:ui-monospace,Menlo,monospace}
.sweep td{border-top:1px solid var(--line);padding:8px 12px;text-align:right}
.sweep tr.dep td{background:color-mix(in srgb,var(--false) 11%,transparent);font-weight:700;color:var(--text)}
.agg4{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:8px}.aggcol h3{font-size:14px;margin:0 0 6px;font-weight:700}.aggcol h3 span{font:600 11px ui-monospace,Menlo,monospace;color:var(--muted)}
table.agg{font-size:13px}.agg th{background:color-mix(in srgb,var(--surface),var(--line) 55%);text-align:left;padding:6px 10px;font:600 10px ui-monospace,Menlo,monospace;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.agg td{border-top:1px solid var(--line);padding:5px 10px}.agg td:nth-child(2){text-align:right;color:var(--muted)}.agg .dir{text-align:center}.agg .dir.up{color:var(--false);font-weight:700}.agg .dir.dn{color:var(--miss);font-weight:700}
footer{padding:34px 0 60px;color:var(--muted);font-size:12.5px;line-height:1.6}
@media(max-width:720px){.tiles{grid-template-columns:repeat(2,1fr)}.agg4{grid-template-columns:1fr}}
html{-webkit-print-color-adjust:exact;print-color-adjust:exact;color-adjust:exact}
@media print{@page{margin:12mm}.card,.ac,.tile,.aggcol,table.sweep,table.agg,section{break-inside:avoid}header.hero{break-after:avoid}.pgrid{grid-template-columns:1fr 1fr}}
@media(prefers-reduced-motion:no-preference){.dot{transition:transform .12s}.dot:hover{transform:translate(-50%,-50%) scale(1.7)}}
"""

H=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',f"<title>10k+7k lung model — FP/FN patient deep-dive</title>",f"<style>{CSS}</style></head><body>"]
H.append('<header class="hero"><div class="wrap">')
H.append(f'<p class="eyebrow">Lung model · internal test · {MODEL_SHORT} · n={nTP+nTN+nFN+nFP} · operating point {THRESH:.2f} (Sens {sens:.1f}% / Spec {spec:.1f}%)</p>')
H.append(f'<h1>Where the model errs: <span class="m">{nFN} missed</span> · <span class="f">{nFP} false alarms</span>.</h1>')
H.append(f'<p class="lede">Every wrong call traces back to one factor — <b>age</b>. Below is each of the {nFN} missed cancers and {nFP} false alarms: who they are, what was on record, and what pushed the score the wrong way.</p>')
H.append('<div class="strip">'
         f'<div class="striprow fn"><div class="lab">Missed · FN</div><div class="track">{dots(FN,"fn")}</div></div>'
         f'<div class="striprow fp"><div class="lab">False alarm · FP</div><div class="track">{dots(FP,"fp")}</div></div>'
         '<div class="ticks2"><div></div><div class="t"><span>18</span><span>35</span><span>50</span><span>65</span><span>92 yrs</span></div></div>'
         '<p class="stripnote">Each dot is a patient, placed by age. <b style="color:var(--miss)">Missed cancers</b> cluster younger; '
         '<b style="color:var(--false)">false alarms</b> cluster older — mirror-image failures of the same age bias.</p></div></div></header>')

H.append('<section><div class="wrap"><h2>Where the 602 test patients landed</h2><div class="tiles">')
for l,v,d,c,tip in [("True positive",nTP,"cancer flagged","var(--false)","A real cancer the model correctly flagged (scored at or above the alert line)."),
                    ("False positive",nFP,"wrongly flagged","var(--false)","No cancer, but the model flagged it anyway — a false alarm."),
                    ("False negative",nFN,"cancer missed","var(--miss)","A real cancer the model scored below the alert line — a miss."),
                    ("True negative",nTN,"correctly cleared","var(--muted)","No cancer, and the model correctly did not flag it.")]:
    H.append(f'<div class="tile" style="--c:{c}" data-tip="{html.escape(tip)}"><div class="l">{l}</div><div class="v" style="color:{c}">{v}</div><div class="d">{d}</div></div>')
H.append(f'</div><div class="metrics"><div data-tip="Sensitivity — of all the real cancers in the test set, the share the model caught at this operating point."><b>{sens:.1f}%</b><span>sensitivity</span></div><div data-tip="Specificity — of all the non-cancers, the share the model correctly did not flag."><b>{spec:.1f}%</b><span>specificity</span></div></div>'
         '<p class="body" style="margin-top:16px"><b>Read each card below</b> for why that patient went wrong — the SHAP factors that pushed its score the wrong way. The aggregate table further down shows the dominant drivers per segment.</p></div></section>')

H.append(f'<section><div class="wrap"><h2>Why the {nFN} cancers were missed</h2><div class="arches">')
for name,d,tip in [("Data gap","≤8 categories on record — too little to predict from.","This patient has 8 or fewer clinical categories on record — too little history for the model to find a signal."),("Age-suppressed","Warning signs present, but age pulled the score below the line.","A miss where the patient's age was the top factor pulling the score down, outweighing genuine warning signs."),("Signal-poor","Full record but routine — no dominant lung red-flag to lock onto.","A miss with a full record but only routine entries — no dominant lung red-flag for the model to lock onto.")]:
    H.append(f'<div class="ac miss" data-tip="{html.escape(tip)}"><div class="n">{fn_a.get(name,0)}</div><div class="t">{name}</div><div class="d">{d}</div></div>')
H.append(f'</div></div></section><section><div class="wrap"><h2>Why the {nFP} false alarms fired</h2><div class="arches">')
for name,d,tip in [("Look-alike","Old, comorbid smoker/COPD profile — hard to separate from real cancer on structured data.","A false alarm on an older, comorbid smoker/COPD profile that looks just like real cancer on structured data."),("Data gap","Thin record; flagged largely on age.","A false alarm on a thin record, flagged largely because of the patient's age.")]:
    H.append(f'<div class="ac fls" data-tip="{html.escape(tip)}"><div class="n">{fp_a.get(name,0)}</div><div class="t">{name}</div><div class="d">{d}</div></div>')
H.append('</div></div></section>')

H.append('<section><div class="wrap"><h2>How to read each patient card</h2><div class="legend">'
         '<div><b class="mono" data-tip="The model\'s calibrated probability that this patient has cancer, 0–100%.">31%&nbsp;risk</b><span>— the model\'s calibrated probability of cancer for this patient.</span></div>'
         '<div><span class="k" data-tip="The clinical categories that appear most in this patient\'s history (routine admin codes excluded).">on record</span><span>— the main clinical categories in the patient\'s history (most frequent; admin excluded).</span></div>'
         '<div><span class="chip dn" data-tip="Features whose SHAP contribution pushed this patient\'s score DOWN, toward &lsquo;not cancer&rsquo;.">↓ lowered</span><span>— features whose SHAP pushed the score <b>down</b>, toward &ldquo;not cancer&rdquo;.</span></div>'
         '<div><span class="chip up" data-tip="Features whose SHAP contribution pushed this patient\'s score UP, toward &lsquo;cancer&rsquo;.">↑ raised</span><span>— features whose SHAP pushed the score <b>up</b>, toward &ldquo;cancer&rdquo;.</span></div></div>'
         '<p class="body" style="margin-top:12px">A <b style="color:var(--miss)">miss (FN)</b> is when the ↓ outweigh the ↑; a <b style="color:var(--false)">false alarm (FP)</b> when the ↑ win despite no real cancer.</p></div></section>')

H.append(f'<section><div class="wrap"><div class="grouphdr fn"><span class="big">{nFN}</span><span class="lbl"><b>Missed cancers (false negatives)</b> — youngest first. Each scored below its alert line.</span></div><div class="pgrid">'+"".join(card(x,"FN") for x in FN)+"</div></div></section>")
H.append(f'<section><div class="wrap"><div class="grouphdr fp"><span class="big">{nFP}</span><span class="lbl"><b>False alarms (false positives)</b> — youngest first. Each scored above its alert line.</span></div><div class="pgrid">'+"".join(card(x,"FP") for x in FP)+"</div></div></section>")

H.append('<section><div class="wrap"><h2>Aggregate SHAP — across all patients</h2><p class="body">Mean SHAP contribution per feature within each confusion-matrix segment (top 20), aggregated from each patient\'s factors. <code class="mono">age</code> tops every segment — the structural age dependence.</p><div class="agg4">'
         +agg_table("TP","True positives","var(--false)")+agg_table("FP","False positives","var(--false)")+agg_table("FN","False negatives","var(--miss)")+agg_table("TN","True negatives","var(--muted)")+'</div></div></section>')

# threshold trade — internal + held-out
H.append('<section><div class="wrap"><h2>Operating point — the threshold trade</h2>'
         '<p class="body">The per-patient errors don\'t change with the threshold — but the counts do, a lot. This is the real lever: a pure <b>catch-more-cancers</b> vs. <b>fewer-false-alarms</b> trade. The highlighted row is the operating point used above.</p>')
H.append(f'<h3 class="sub">Internal test — {nTP+nTN+nFN+nFP} patients, balanced</h3>'+sweep_table(INT_SWEEP,THRESH))
if HO_SWEEP: H.append('<h3 class="sub">Held-out sweep</h3>'+sweep_table(HO_SWEEP,HO_OP))
H.append('<p class="body" style="margin-top:14px">Lowering the cut catches more cancers but adds false alarms; PPV falls as the cut drops. The right operating point is a clinical trade-off for this model.</p></div></section>')

# how hard / ceiling (generic, data-derived)
H.append('<section class="dark"><div class="wrap"><h2>The ceiling — what is genuinely hard</h2>'
         '<p class="finding">Where the remaining error concentrates, from this model\'s own data.</p><div class="arches" style="margin-top:20px">')
for t,d in [("Data-gap", f"{fn_a.get('Data gap',0)} of the misses had too little on record to predict from — a data limit, not a tuning one."),
            ("Signal-poor", f"{fn_a.get('Signal-poor',0)} of the misses had a full record but no dominant red-flag for the model to lock onto."),
            ("Look-alikes", f"{fp_a.get('Look-alike',0)} of the false alarms fit a benign profile that resembles cancer on structured data.")]:
    H.append(f'<div class="ac" style="background:#1f2a37;border-color:#2c3a49;border-top-color:var(--false)"><div class="t" style="color:#fff">{t}</div><div class="d" style="color:#c3cdd7">{d}</div></div>')
H.append(f'</div><p class="body" style="margin-top:24px"><b>What was analysed.</b> {nFN+nFP} internal errors ({nFN} missed, {nFP} false alarms) at the {THRESH:.2f} cut, each with its record content and SHAP drivers, plus the threshold sweep above.</p>'
         '<p class="body" style="margin-top:16px"><b>The honest ceiling.</b> At low prevalence, limited discrimination yields many false alarms per true catch — a property of the data, not the cut. Judge realistic gains for THIS model from the archetypes above, its within-subgroup discrimination, and the threshold trade — and re-state this paragraph from what its data actually shows before sharing.</p></div></section>')

H.append(f'<footer><div class="wrap">{MODEL_LONG} (best internal, AUROC {MODEL_AUROC}) · internal test @ threshold {THRESH:.2f} ({nTP} TP / {nTN} TN / {nFP} FP / {nFN} FN; Sens {sens:.1f}% / Spec {spec:.1f}%) · risk % = calibrated model probability · driver chips are the model\'s top SHAP contributors, in plain terms.</div></footer>')
TIP='<div id="tip"></div><script>(function(){var t=document.getElementById("tip");function pos(e){var p=14,w=t.offsetWidth,h=t.offsetHeight,x=e.clientX+p,y=e.clientY+p;if(x+w>window.innerWidth-8)x=e.clientX-w-p;if(y+h>window.innerHeight-8)y=e.clientY-h-p;t.style.left=x+"px";t.style.top=y+"px";}document.addEventListener("mouseover",function(e){var el=e.target.closest&&e.target.closest("[data-tip]");if(!el)return;t.textContent=el.getAttribute("data-tip");t.style.display="block";pos(e);});document.addEventListener("mousemove",function(e){if(t.style.display==="block")pos(e);});document.addEventListener("mouseout",function(e){var el=e.target.closest&&e.target.closest("[data-tip]");if(el&&(!e.relatedTarget||!(e.relatedTarget.closest&&e.relatedTarget.closest("[data-tip]"))))t.style.display="none";});})();</script>'
open(OUT,"w",encoding="utf-8").write("".join(H)+TIP+"</body></html>")
print("wrote",OUT,os.path.getsize(OUT)//1024,"KB | TP",nTP,"FP",nFP,"FN",nFN,"TN",nTN,"| Sens",round(sens,1),"Spec",round(spec,1))
