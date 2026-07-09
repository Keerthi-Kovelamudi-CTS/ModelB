"""contrast_report — deterministic FN/FP per-patient sheets (+ deep-dive) for ANY cancer.
No LLM in the loop. Driven by a per-cancer YAML config. See configs/lung.yaml."""
import os, re, json, glob, subprocess, tempfile
import numpy as np, pandas as pd, yaml
from . import render as R

# ---------- config ----------
def load_config(path):
    cfg = yaml.safe_load(open(path))
    # join wrapped lines WITHOUT destroying intra-term spaces (each line ends with '|')
    join=lambda s:"".join(l.strip() for l in str(s or "").splitlines())
    cfg["workup_terms"]  = join(cfg.get("workup_terms",""))
    cfg["relevant_terms"]= join(cfg.get("relevant_terms",""))
    return cfg

CORE = (r"suspected( %(o)s)? cancer|fast[- ]track|two week wait|2 week rule|2ww|"
        r"refer\w*.*(oncolog|cancer|rapid|%(o)s)|referral to (oncolog|%(o)s)|oncolog|"
        r"malignan|neoplasm|carcinoma|metasta")

def setup_render(cfg, gap, organ):
    """Point the renderer's detectors/labels at this cancer (generic core + organ add-on)."""
    core = CORE % {"o": re.escape(organ)}
    workup = core + ("|" + cfg["workup_terms"] if cfg["workup_terms"] else "")
    R.WORKUP = re.compile(workup, re.I)
    R.LUNG   = re.compile(cfg["relevant_terms"] or re.escape(organ), re.I)
    R.ORGAN  = organ
    R.GAP    = gap

# ---------- run introspection ----------
def horizon_of(run):
    m = glob.glob(f"{run}/fe/features_p005_*_stable.parquet")
    if not m: raise SystemExit(f"no stable matrix under {run}/fe/")
    h = re.search(r"features_p005_([0-9]+mo)_stable", os.path.basename(m[0])).group(1)
    return h, m[0]

def resolve_params(run, cfg):
    h, stable = horizon_of(run)
    gap = cfg.get("gap_months") or int(re.match(r"(\d+)", h).group(1))
    # Internal-test error sheets default to 0.50 (balanced). NOTE: operating_threshold_{h}.json holds the
    # HELD-OUT / prevalence-corrected point (very low) — NOT what these sheets use. Override via config only
    # if you deliberately characterise errors at that operating point.
    thr = cfg.get("threshold")
    thr = 0.50 if thr is None else float(thr)
    return h, stable, gap, thr

def identify(run, stable, thr):
    pe = pd.read_csv(f"{run}/modeling/explainability_internal/patient_explanations.csv").sort_values("row").reset_index(drop=True)
    if "shap_1" not in pe.columns:
        raise SystemExit("patient_explanations.csv is not schema A (need signed factor_/shap_). Re-run explainability.")
    st = pd.read_parquet(stable, columns=["split","patient_guid","age_at_prediction"])
    test = st[st.split=="test"].reset_index(drop=True)
    if len(test)!=len(pe): raise SystemExit(f"stable test slice ({len(test)}) != explanations ({len(pe)})")
    pe["guid"]=[test.patient_guid.iloc[r] for r in pe.row]
    pe["age_stable"]=[int(test.age_at_prediction.iloc[r]) for r in pe.row]
    fn = pe[(pe.y_true==1)&(pe.prob<thr)].copy()
    fp = pe[(pe.y_true==0)&(pe.prob>=thr)].copy()
    return pe, fn, fp

def cache_local(cache_uri, workdir):
    if cache_uri.startswith("gs://"):
        dst=f"{workdir}/_cache.parquet"
        if not os.path.exists(dst):
            subprocess.run(["gcloud","storage","cp",cache_uri,dst],check=True)
        return dst
    return cache_uri

def anchors_pregap(cache_path, guids):
    cols=["patient_guid","event_date","days_before_anchor","event_type","snomed_c_t_concept_id",
          "term","problem_status_description","age_at_anchor","sex","cancer_class"]
    df=pd.read_parquet(cache_path, columns=cols)
    g=df[df.patient_guid.isin(guids)].copy(); g["event_date"]=pd.to_datetime(g.event_date)
    g["anchor"]=g.event_date+pd.to_timedelta(g.days_before_anchor,unit="D")
    anc=g.groupby("patient_guid").agg(anchor=("anchor","max"),age=("age_at_anchor","max"),
                                      sex=("sex","first")).reset_index()
    anc["anchor_date"]=anc.anchor.dt.strftime("%Y-%m-%d")
    pregap=(g.groupby(["patient_guid","snomed_c_t_concept_id"])
              .agg(term=("term","first"),n=("event_date","size"),etype=("event_type","first"),
                   psd=("problem_status_description",lambda s:s.dropna().iloc[0] if s.notna().any() else "")).reset_index())
    return {r.patient_guid:dict(age=int(r.age),sex=r.sex,anchor=r.anchor_date) for _,r in anc.iterrows()}, pregap

# ---------- plain-English SHAP decoder (cancer-agnostic feature families) ----------
_FAM=[("decay_intensity","recent {c} activity"),("distinct_ratio","how varied the {c} coding is"),
      ("recency_rank","how recently {c} was recorded"),("recency_months","how recently {c} was recorded"),
      ("count_last24","recent {c} entries"),("count_last60","recent {c} entries"),("count","how often {c} was recorded"),
      ("present","{c} on record"),("val_min","lowest {c} value"),("val_max","highest {c} value"),
      ("val_mean","average {c} value"),("val_latest","latest {c} value"),("val_std","spread of {c} values"),
      ("val_range","range of {c} values"),("max_abs_z","how unusual the {c} values are"),
      ("first_months","how early {c} appears"),("timespan_years","how long {c} spans"),
      ("interval","gaps between {c} records"),("accel","trend in {c}"),("freq","trend in {c}"),
      ("recent_ratio","how recent the {c} is"),("abs_change","change in {c}"),("pct_change","change in {c}")]
def describe(f):
    f=str(f)
    if f=="age_at_prediction": return "age"
    if f=="ageband_u50": return "being under 50"
    if f.startswith("g_eth"): return "ethnicity"
    for suf,t in _FAM:
        if f.endswith("_"+suf) or ("_"+suf+"_") in f or f.endswith(suf):
            return t.format(c=re.sub(r"_"+re.escape(suf)+r".*$","",f).replace("_"," ").strip())
    return f.replace("_"," ")
def _is_age(f): return f in ("age_at_prediction","ageband_u50") or str(f).startswith("ageband_")
def _downs(r): return [(r[f"factor_{i}"],float(r[f"shap_{i}"])) for i in range(1,21) if pd.notna(r.get(f"shap_{i}")) and float(r[f"shap_{i}"])<0]
def _ups(r):   return [(r[f"factor_{i}"],float(r[f"shap_{i}"])) for i in range(1,21) if pd.notna(r.get(f"shap_{i}")) and float(r[f"shap_{i}"])>0]

# ---------- BigQuery pull (cache-first; live via `bq`) ----------
def _anchors_values(anchors):
    return ",\n".join(f"    ('{g}', DATE '{a['anchor'][:10]}')" for g,a in anchors.items())

def _sql(anchors, gap, project, cohort):
    A=f"anchors AS (SELECT guid,anchor FROM UNNEST([STRUCT<guid STRING,anchor DATE>\n{_anchors_values(anchors)}\n]))"
    P=project
    dc=(f"dc AS (SELECT SAFE_CAST(code_id AS INT64) code_id, source_practice_code,"
        f" PARSE_DATE('%Y%m%d',REGEXP_EXTRACT(file_name,r'/([0-9]{{8}})/')) fd, term,"
        f" SAFE_CAST(snomed_c_t_concept_id AS INT64) sct"
        f" FROM `{P}.EMIS_BULK_DATA_PROCESSED.Coding_ClinicalCode`"
        f" WHERE snomed_c_t_concept_id IS NOT NULL AND term IS NOT NULL"
        f" QUALIFY ROW_NUMBER() OVER (PARTITION BY code_id,source_practice_code ORDER BY fd DESC)=1)")
    lo=f"DATE_SUB(a.anchor, INTERVAL {gap} MONTH)"; hi_fn="a.anchor"; hi_fp=f"DATE_ADD(a.anchor, INTERVAL {gap} MONTH)"
    hi = hi_fp if cohort=="fp" else hi_fn
    o=(f"o AS (SELECT co.patient_guid, PARSE_DATE('%Y-%m-%d',co.effective_date) ed, a.anchor,"
       f" SAFE_CAST(co.code_id AS INT64) code_id, co.source_practice_code, prob.problem_status_description psd"
       f" FROM `{P}.EMIS_BULK_DATA_PROCESSED.CareRecord_Observation` co JOIN anchors a ON a.guid=co.patient_guid"
       f" LEFT JOIN `{P}.EMIS_BULK_DATA_PROCESSED.CareRecord_Problem` prob ON prob.observation_guid=co.observation_guid"
       f" AND prob.patient_guid=co.patient_guid AND prob.source_practice_code=co.source_practice_code"
       f" WHERE co.effective_date IS NOT NULL AND PARSE_DATE('%Y-%m-%d',co.effective_date)>={lo}"
       f" AND PARSE_DATE('%Y-%m-%d',co.effective_date)<{hi}"
       f" QUALIFY ROW_NUMBER() OVER (PARTITION BY co.patient_guid,co.source_practice_code,co.observation_guid"
       f" ORDER BY co.source_date DESC, SAFE_CAST(co.code_id AS INT64))=1)")
    if cohort=="fp":
        obs=(f"WITH {A},\n{dc},\n{o}\nSELECT o.patient_guid,dc.sct,dc.term,"
             f"COUNTIF(o.ed<o.anchor) n_before, COUNTIF(o.ed>=o.anchor) n_after, MIN(o.ed) first_ed,"
             f"ARRAY_TO_STRING(ARRAY_AGG(DISTINCT o.psd IGNORE NULLS LIMIT 3),'; ') psd"
             f" FROM o JOIN dc ON dc.code_id=o.code_id AND dc.source_practice_code=o.source_practice_code GROUP BY 1,2,3")
    else:
        obs=(f"WITH {A},\n{dc},\n{o}\nSELECT o.patient_guid,dc.sct,dc.term,COUNT(*) n_gap,MIN(o.ed) first_in_gap,"
             f"ARRAY_TO_STRING(ARRAY_AGG(DISTINCT o.psd IGNORE NULLS LIMIT 3),'; ') psd"
             f" FROM o JOIN dc ON dc.code_id=o.code_id AND dc.source_practice_code=o.source_practice_code GROUP BY 1,2,3")
    # first-ever (new): min over ALL history, keep those whose first date is in the window
    o_all=o.replace(f" AND PARSE_DATE('%Y-%m-%d',co.effective_date)>={lo} AND PARSE_DATE('%Y-%m-%d',co.effective_date)<{hi}","")
    new=(f"WITH {A},\n{dc},\n{o_all},\nfs AS (SELECT patient_guid,sct,ANY_VALUE(term) term,MIN(ed) first_ed"
         f" FROM o JOIN dc ON dc.code_id=o.code_id AND dc.source_practice_code=o.source_practice_code GROUP BY 1,2)"
         f"\nSELECT fs.patient_guid,fs.sct,fs.term,fs.first_ed FROM fs JOIN anchors a ON a.guid=fs.patient_guid"
         f" WHERE fs.first_ed>={lo} AND fs.first_ed<{hi}")
    dr=(f"dr AS (SELECT TRIM(drug_record_guid) drug_record_guid,TRIM(patient_guid) patient_guid,source_practice_code,"
        f"PARSE_DATE('%Y%m%d',REGEXP_EXTRACT(file_name,r'/([0-9]{{8}})/')) fd,deleted"
        f" FROM `{P}.EMIS_BULK_DATA_PROCESSED.Prescribing_DrugRecord`"
        f" QUALIFY ROW_NUMBER() OVER (PARTITION BY drug_record_guid ORDER BY fd DESC)=1)")
    ir=(f"ir AS (SELECT TRIM(issue_record_guid) issue_record_guid,TRIM(drug_record_guid) drug_record_guid,"
        f"PARSE_DATE('%Y-%m-%d',TRIM(effective_date)) ed,SAFE_CAST(code_id AS INT64) code_id,"
        f"PARSE_DATE('%Y%m%d',REGEXP_EXTRACT(file_name,r'/([0-9]{{8}})/')) fd,deleted"
        f" FROM `{P}.EMIS_BULK_DATA_PROCESSED.Prescribing_IssueRecord`"
        f" QUALIFY ROW_NUMBER() OVER (PARTITION BY issue_record_guid ORDER BY fd DESC)=1)")
    ce=(f"ce AS (SELECT SAFE_CAST(code_id AS INT64) code_id,source_practice_code,term,"
        f"PARSE_DATE('%Y%m%d',REGEXP_EXTRACT(file_name,r'/([0-9]{{8}})/')) fd"
        f" FROM `{P}.EMIS_BULK_DATA_PROCESSED.Coding_DrugCode` WHERE dmd_product_code_id IS NOT NULL"
        f" QUALIFY ROW_NUMBER() OVER (PARTITION BY code_id,source_practice_code ORDER BY fd DESC)=1)")
    m=(f"m AS (SELECT dr.patient_guid,ir.ed,a.anchor,ce.term drug_term FROM dr JOIN anchors a ON a.guid=dr.patient_guid"
       f" LEFT JOIN ir USING(drug_record_guid) LEFT JOIN ce ON ce.code_id=ir.code_id AND ce.source_practice_code=dr.source_practice_code"
       f" WHERE COALESCE(dr.deleted,false)=false AND COALESCE(ir.deleted,false)=false AND ir.ed>={lo} AND ir.ed<{hi})")
    if cohort=="fp":
        med=(f"WITH {A},\n{dr},\n{ir},\n{ce},\n{m}\nSELECT patient_guid,drug_term,COUNTIF(ed<anchor) n_before,"
             f"COUNTIF(ed>=anchor) n_after,MIN(ed) first_ed FROM m WHERE drug_term IS NOT NULL GROUP BY 1,2")
    else:
        med=(f"WITH {A},\n{dr},\n{ir},\n{ce},\n{m}\nSELECT patient_guid,drug_term,COUNT(*) n_gap,MIN(ed) first_in_gap"
             f" FROM m WHERE drug_term IS NOT NULL GROUP BY 1,2")
    return {"obs":obs,"med":med,"new":new}

def _bq(sql, project):
    with tempfile.NamedTemporaryFile("w",suffix=".sql",delete=False) as f: f.write(sql); p=f.name
    out=subprocess.run(["bq","query","--nouse_legacy_sql",f"--project_id={project}","--format=csv","--max_rows=500000"],
                       stdin=open(p),capture_output=True,text=True)
    if out.returncode!=0: raise SystemExit("bq failed:\n"+out.stderr[-800:])
    import io; return pd.read_csv(io.StringIO(out.stdout))

def pull_gap(cohort, anchors, gap, project, cache_dir, live=True):
    q=_sql(anchors,gap,project,cohort); res={}
    for kind in ("obs","med","new"):
        cp=f"{cache_dir}/{cohort}_{kind}.csv"
        if os.path.exists(cp): res[kind]=pd.read_csv(cp)
        elif live:
            df=_bq(q[kind],project); os.makedirs(cache_dir,exist_ok=True); df.to_csv(cp,index=False); res[kind]=df
        else: raise SystemExit(f"missing {cp} and --no-bq set")
    return res

# ---------- output helpers ----------
DOC='<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"></head><body>{}</body></html>'
def _pdf(html_path, pdf_path):
    chrome="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if os.path.exists(chrome):
        subprocess.run([chrome,"--headless","--disable-gpu","--no-pdf-header-footer",
                        f"--print-to-pdf={os.path.abspath(pdf_path)}",f"file://{os.path.abspath(html_path)}"],capture_output=True)
def _xlsx(rows, hdr, path):
    import openpyxl
    from openpyxl.styles import Font, PatternFill
    wb=openpyxl.Workbook(); ws=wb.active; ws.append(hdr)
    for c in ws[1]: c.font=Font(bold=True,color="FFFFFF"); c.fill=PatternFill("solid",fgColor="16202B")
    for r in rows: ws.append((list(r)+[""]*len(hdr))[:len(hdr)])
    ws.freeze_panes="A2"; ws.auto_filter.ref=ws.dimensions
    wb.save(path)

# ---------- FN builder ----------
def build_fn(cfg, organ, gap, thr, anchors, pregap, fn_df, gd, out_dir):
    setup_render(cfg, gap, organ)
    fn_df=fn_df.set_index("guid")
    n_codes={g:int((pregap.patient_guid==g).sum()) for g in anchors}
    newkeys=set(zip(gd["new"].patient_guid, gd["new"].sct))
    obs=gd["obs"].copy(); obs["kind"]="obs"; obs["is_new"]=[(g,s) in newkeys for g,s in zip(obs.patient_guid,obs.sct)]
    med=gd["med"].rename(columns={"drug_term":"term"}).copy(); med["kind"]="med"; med["psd"]=""; med["is_new"]=False
    cols=["patient_guid","term","n_gap","first_in_gap","psd","is_new","kind"]
    gap_df=pd.concat([obs[cols],med[cols]],ignore_index=True)
    gap_df=(gap_df.groupby(["patient_guid","term","kind"],as_index=False)
            .agg(n_gap=("n_gap","sum"),first_in_gap=("first_in_gap","min"),
                 psd=("psd",lambda s:"; ".join(sorted({x for x in s.astype(str) if x and x!='nan'}))[:120]),
                 is_new=("is_new","max")))
    def classify(gp):
        if gp.empty: return "silent"
        t=gp.term.astype(str)
        if t.map(R.is_workup).any(): return "pathway"
        isprob=gp.psd.fillna("").str.contains("Problem")&(gp.kind=="obs")
        return "other" if (isprob&gp.is_new).any() else "silent"
    def why(g):
        if g not in fn_df.index: return ""
        r=fn_df.loc[g]; p=100*r.prob; age=anchors[g]["age"]; downs=_downs(r); ups=_ups(r)
        if n_codes[g]<=10 or (p<3 and n_codes[g]<20):
            return f"The record held only {n_codes[g]} coded items in the model's lookback window — too little to find cancer signal, so it defaulted to a low score ({p:.0f}%), driven mainly by the patient being {age}."
        top=[f for f,_ in downs[:3]]
        others=[describe(f) for f in top[1:] if not _is_age(f)][:2]; oc=(" Next strongest were "+", ".join(dict.fromkeys(others))+".") if others else ""
        real_ups=[describe(f) for f,_ in ups[:3] if not _is_age(f)][:2]
        up=(f" The only things nudging it up — {', '.join(dict.fromkeys(real_ups))} — weren't enough.") if real_ups else ""
        if downs and _is_age(downs[0][0]):
            return f"The model gave this {p:.0f}%. By far the biggest downward pull was the patient's age ({age}, SHAP {downs[0][1]:+.2f}) — it outweighed everything else.{oc}{up}"
        return f"The model gave this {p:.0f}%. The score was held down most by {describe(downs[0][0]) if downs else '—'} (SHAP {downs[0][1]:+.2f} if downs else 0).{oc}{up} No single {organ} red-flag was strong enough to lift it over the line."
    order=sorted(anchors,key=lambda g:(anchors[g]["age"],g)); blocks=[]; xrows=[]; cnt={"pathway":0,"other":0,"silent":0}
    for i,g in enumerate(order,1):
        pid=f"FN{i:02d}"; a=anchors[g]; pg=pregap[pregap.patient_guid==g]; gp=gap_df[gap_df.patient_guid==g]
        cat=classify(gp); cnt[cat]+=1
        meta=dict(age=a["age"],sex=a["sex"],anchor=a["anchor"],cat=cat,guid=g,why=why(g),why_label="Why missed",why_cls="")
        blocks.append(R.patient_block(pid,meta,pg,gp)); xrows+=R.xlsx_rows(cfg["cancer"],pid,meta,pg,gp)
    nP,nO,nS=cnt["pathway"],cnt["other"],cnt["silent"]
    key=(f'<div class="key"><span>{len(order)} patients</span>'
         f'<span class="k-path" style="color:var(--miss)">{nP} on cancer pathway</span>'
         f'<span class="k-new">{nO} other new dx</span><span>{nS} silent</span>'
         f'<span class="k-new">■ new (first-ever)</span><span style="color:var(--muted)">■ recurring</span>'
         f'<span style="color:var(--med)">■ medication</span><span class="k-lung">■ {organ}-relevant</span>'
         f'<span class="k-dx">bold = diagnosis</span></div>')
    sub=f"Internal test · {len(order)} {organ} cancers the model scored below its alert line."
    body=R.build_html(f"{cfg['cancer']} · missed cancers",sub,"".join(blocks),len(order),nP,nO,nS,
                      eyebrow_noun="missed cancers (false negatives)",key_html=key)
    html_p=f"{out_dir}/fn_contrast.html"; open(html_p,"w",encoding="utf-8").write(DOC.format(body))
    _xlsx(xrows,["cohort","patient","patient_guid","age","sex","anchor","window","term","snomed","type","relevant","count","problem_dx","first_seen","status"],f"{out_dir}/fn_before_vs_gap.xlsx")
    _pdf(html_p,f"{out_dir}/fn_contrast.pdf")
    return dict(patients=len(order),pathway=nP,other=nO,silent=nS,html=html_p)

# ---------- FP builder (3-column ±1yr) ----------
def _chip(term,n,first_ed,psd,is_new,is_med,side):
    lr=R.lung_rel(term); isdx=("Problem" in str(psd))
    ty,desc=("Medication","A prescribed / issued medication.") if is_med else R.code_type(term,psd)
    cls="chip"+(" med" if is_med else (" new" if is_new else " rec"))+(" lr" if lr else "")+(" dx" if isdx else "")
    cnt=f"<i>{int(n)}</i>" if n>1 else ""
    status="first-ever in the record" if (is_new and not is_med) else "seen before / repeated"
    tip=f"{ty} · {int(n)}× in the {side} · {status} (first {str(first_ed)[:10]}). {desc}"
    return f'<span class="{cls}" data-tip="{R.esc(tip)}">{R.esc(term)}{cnt}</span>'

def build_fp(cfg, organ, gap, thr, anchors, pregap, fp_df, gd, out_dir):
    setup_render(cfg, gap, organ)
    fp_df=fp_df.set_index("guid")
    newkeys=set(zip(gd["new"].patient_guid, gd["new"].sct))
    obs=gd["obs"].copy(); obs["kind"]="obs"; obs["is_new"]=[(g,s) in newkeys for g,s in zip(obs.patient_guid,obs.sct)]
    med=gd["med"].rename(columns={"drug_term":"term"}).copy(); med["kind"]="med"; med["psd"]=""; med["is_new"]=False; med["sct"]=-1
    def col(rows,side,anchor_date):
        nc="n_before" if side=="before" else "n_after"; sub=rows[rows[nc]>0].copy()
        if sub.empty: return '<span class="chip">(nothing recorded)</span>',0,0,0
        sub["lr"]=sub.term.map(R.lung_rel); sub["isdx"]=sub.psd.fillna("").str.contains("Problem"); sub["ismed"]=sub.kind=="med"
        anc=pd.Timestamp(anchor_date); fed=pd.to_datetime(sub.first_ed)
        sub["colnew"]=sub.is_new & ((fed<anc) if side=="before" else (fed>=anc))
        sub=sub.sort_values(["ismed","lr","colnew","isdx",nc],ascending=[True,False,False,False,False])
        html="".join(_chip(r.term,r[nc],r.first_ed,r.get("psd",""),bool(r.colnew),bool(r.ismed),
                     "12 months "+side+" the anchor") for _,r in sub.iterrows())
        return html,len(sub),int(sub.lr.sum()),int(sub.ismed.sum())
    def classify(rows):
        if rows.empty: return ("Genuine false alarm","b-silent")
        if rows.term.astype(str).map(R.is_workup).any(): return ("Cancer workup — negative","b-workup")
        isprob=rows.psd.fillna("").str.contains("Problem")&(rows.kind=="obs")
        return ("Other active disease","b-disease") if isprob.any() else ("Genuine false alarm","b-silent")
    def why(g):
        if g not in fp_df.index: return ""
        r=fp_df.loc[g]; p=100*r.prob; age=anchors[g]["age"]; ups=_ups(r)
        if not ups: return f"The model gave this {p:.0f}% despite no cancer."
        back=[describe(f) for f,_ in ups if not _is_age(f)][:3]; back=list(dict.fromkeys(back))
        if _is_age(ups[0][0]):
            nxt=next((s for f,s in ups if not _is_age(f)),0.0); ratio=f"{ups[0][1]/nxt:.0f}×" if nxt>0 else "far"
            return (f"The model gave this {p:.0f}% almost entirely on age: being {age} contributed SHAP {ups[0][1]:+.2f} — {ratio} the next factor"
                    +("; on top of "+", ".join(back) if back else "")+f". The over-flagged {cfg.get('lookalike','older-comorbid')} profile that looks like real {organ} cancer on structured data.")
        return f"The model gave this {p:.0f}% despite no cancer. Pushed up mainly by {', '.join(back)} — a {cfg.get('lookalike','comorbid')} picture the model reads as cancer-like."
    order=sorted(anchors,key=lambda g:(anchors[g]["age"],g)); blocks=[]; cnt={}; xrows=[]
    for i,g in enumerate(order,1):
        pid=f"FP{i:02d}"; a=anchors[g]; pg=pregap[pregap.patient_guid==g]
        o=obs[obs.patient_guid==g]; m=med[med.patient_guid==g]
        allr=pd.concat([o[["term","n_before","n_after","psd","kind","is_new","first_ed","sct"]],
                        m[["term","n_before","n_after","psd","kind","is_new","first_ed","sct"]]],ignore_index=True)
        badge,bcls=classify(allr); cnt[badge]=cnt.get(badge,0)+1
        bH,nb,nbl,nbm=col(allr,"before",a["anchor"]); aH,na,nal,nam=col(allr,"after",a["anchor"])
        wk=allr[(allr.kind=="obs")&allr.term.map(R.is_workup)]
        if len(wk):
            seen={}; 
            for _,r in wk.sort_values("first_ed").iterrows(): seen.setdefault(r.term,"before" if r.n_before>0 else "after")
            wl=f'<div class="workline hit" data-tip="Cancer-workup codes within a year of the flag (before or after) — a negative workup is a defensible flag."><b>Cancer workup within ±1 year:</b> '+"".join(f'<span class="wchip">{R.esc(t)} <em style="opacity:.7">({w})</em></span>' for t,w in seen.items())+'</div>'
        else:
            wl='<div class="workline none" data-tip="No cancer-workup code within a year either side — the flag was not vindicated by a cancer investigation."><b>No cancer workup within ±1 year</b> — see the code lists.</div>'
        whyl=f'<div class="whyfn" data-tip="Why the model scored this non-cancer patient above its alert line."><b>Why flagged:</b> {R.esc(why(g))}</div>'
        blocks.append(f'<article class="pc"><header><span class="tab">{pid}</span>'
            f'<span class="dem">{a["age"]} · {a["sex"]} · anchor {a["anchor"][:7]}</span>'
            f'<span class="guid" data-tip="Patient GUID — for tracing in the source data (local only).">{R.esc(g)}</span>'
            f'<span class="badge {bcls}">{badge}</span></header>{whyl}{wl}'
            f'<div class="cols cols3">'
            f'<div class="col"><div class="ch" data-tip="Every coded entry the model was given (up to {gap} months before the anchor).">MODEL WINDOW — what was used <i>{len(pg)} codes</i></div><div class="chipbox">{R.patient_left(pg)}</div></div>'
            f'<div class="col mid"><div class="ch" data-tip="The {gap} months before the anchor (the excluded gap). Blue = first-ever.">{gap} MONTHS BEFORE anchor <i>{nb} codes · {nbm} meds · {nbl} {organ}-relevant</i></div><div class="chipbox">{bH}</div></div>'
            f'<div class="col aft"><div class="ch" data-tip="The {gap} months AFTER the anchor — what happened next. A workup/disease here vindicates the flag.">{gap} MONTHS AFTER anchor <i>{na} codes · {nam} meds · {nal} {organ}-relevant</i></div><div class="chipbox">{aH}</div></div>'
            f'</div></article>')
        # xlsx
        for _,r in pg.iterrows():
            ty,_d=R.code_type(r.term,r.get("psd",""),r.get("etype","")); xrows.append([cfg["cancer"],pid,g,a["age"],a["sex"],a["anchor"],"model window",r.term,ty,"yes" if R.lung_rel(r.term) else "",int(r.n),"","",""])
        for _,r in o.iterrows():
            ty,_d=R.code_type(r.term,r.get("psd","")); s="new" if (g,r.sct) in newkeys else "recurring"; dx="yes" if "Problem" in str(r.psd) else ""
            if r.n_before>0: xrows.append([cfg["cancer"],pid,g,a["age"],a["sex"],a["anchor"],"12mo BEFORE",r.term,ty,"yes" if R.lung_rel(r.term) else "",int(r.n_before),dx,str(r.first_ed)[:10],s])
            if r.n_after>0:  xrows.append([cfg["cancer"],pid,g,a["age"],a["sex"],a["anchor"],"12mo AFTER",r.term,ty,"yes" if R.lung_rel(r.term) else "",int(r.n_after),dx,str(r.first_ed)[:10],s])
    nW=cnt.get("Cancer workup — negative",0); nD=cnt.get("Other active disease",0); nS=cnt.get("Genuine false alarm",0)
    key=(f'<div class="key"><span>{len(order)} patients</span>'
         f'<span class="k-lung" style="color:var(--workup)">{nW} cancer workup — negative</span>'
         f'<span class="k-lung">{nD} other active disease</span><span>{nS} genuine false alarm</span>'
         f'<span class="k-new">■ new</span><span style="color:var(--muted)">■ recurring</span>'
         f'<span style="color:var(--med)">■ medication</span><span class="k-lung">■ {organ}-relevant</span><span class="k-dx">bold = diagnosis</span></div>')
    lede=(f"Each patient has <b>three columns</b>: what the model saw · the {gap} months before the anchor · the {gap} months after.")
    body=R.build_html(f"{cfg['cancer']} · false alarms",f"Internal test · {len(order)} patients flagged with no cancer, ±1yr.",
                      "".join(blocks),len(order),nW,nD,nS,eyebrow_noun="false alarms (false positives) · ±1 year",key_html=key,lede=lede)
    html_p=f"{out_dir}/fp_contrast.html"; open(html_p,"w",encoding="utf-8").write(DOC.format(body))
    _xlsx(xrows,["cohort","patient","patient_guid","age","sex","anchor","window","term","type","relevant","count","problem_dx","first_seen","status"],f"{out_dir}/fp_before_after.xlsx")
    _pdf(html_p,f"{out_dir}/fp_contrast.pdf")
    return dict(patients=len(order),workup=nW,disease=nD,silent=nS,html=html_p)
