"""Per-patient before-gap vs in-gap contrast sheet (HTML + Excel).
LEFT  = model-visible history (all coded events up to anchor-12mo) -- the data the model uses.
RIGHT = the excluded 12-month window [anchor-12mo, anchor): codes appearing for the FIRST time.
"""
import re, html, json
import pandas as pd

# ---- code typing (for the hover description) ----
def code_type(term, psd, etype=""):
    t = ("" if pd.isna(term) else str(term)).lower()
    p = "" if pd.isna(psd) else str(psd)
    etype = "" if pd.isna(etype) else str(etype)
    if etype == "medication" or "medication" in t or "drug" in t:
        return "Medication", "A prescribed / issued medication."
    if "Problem" in p:
        return "Diagnosis", "A coded condition on the patient's problem list (an actual diagnosis)."
    if re.search(r"refer|fast track|2ww|two week", t):
        return "Referral", "A referral to another service or clinic."
    if re.search(r"x-?ray|\bct\b|computed tomog|scan|bronchoscop|spiromet|biopsy|endoscop|ultrasound|mri|imaging|angiogra", t):
        return "Investigation", "An imaging study or diagnostic procedure."
    if re.search(r"blood test|test|pressure|rate|weight|body mass|bmi|saturation|observ|measure|result|value|level|count|haemoglob|glucose", t):
        return "Measurement", "A test result or clinical measurement."
    if re.search(r"seen in|seen by|admission|attend|consult|contact|clinic|review|letter|telephone|advice|leaflet|plan|assessment", t):
        return "Encounter", "A clinical contact, appointment or administrative event."
    return "Clinical code", "A coded clinical entry."

# Lung-cancer WORKUP / pathway detector (imaging, chest/respiratory/oncology clinics &
# referrals, suspected-cancer/2ww, bronchoscopy/lobectomy, nodule/mass, malignancy dx).
# Symptoms (cough, chest pain) are deliberately NOT workup.
WORKUP = re.compile(
  r"suspected( lung)? cancer|fast[- ]track|two week wait|2 week rule|2ww|"
  r"refer\w*.*(lung|chest|respirat|oncolog|cancer|thorac|medicine|medical service|rapid)|"
  r"referral to (respiratory|chest|oncolog)|"
  r"chest x-?ray|standard chest|chest clinic|seen in chest|"
  r"respiratory (physician|medicine|clinic)|seen by respiratory|"
  r"bronchoscop|lobectomy|thoracoscop|pleural (asp|effusion|biopsy)|"
  r"oncolog|"
  r"lung (nodule|mass|cancer|lesion)|nodule of lung|solitary (pulmonary )?nodule|"
  r"malignan|neoplasm|metasta|carcinoma|"
  r"(ct|computed tomog).*(chest|thorax)", re.I)
def is_workup(term):
    t = "" if pd.isna(term) else str(term); tl = t.lower()
    if not WORKUP.search(t): return False
    if "not wanted" in tl or "declined" in tl: return False          # screening declines
    if "screening" in tl and ORGAN not in tl: return False          # non-lung screening (cervix/bowel/breast)
    return True

LUNG = re.compile(r"cough|h[ae]emoptysis|chest|pneumon|respirat|dyspn|breath|spo2|oxygen sat|copd|bronch|lung|pleur|wheez|lrti|lower respiratory|sputum|hoarse|x-?ray|\bct\b|computed tomog|spiromet|smok|weight loss|fatigue|anaemia|clubbing|lymph node|nodule|thoracic|malignan|neoplasm|metasta|cancer|tumour|mass|oncolog", re.I)
def lung_rel(term): return bool(LUNG.search("" if pd.isna(term) else str(term)))

def esc(s): return html.escape(str(s), quote=True)

# ---- one patient block ----

def patient_left(pregap):
    pg=pregap.copy(); pg["lr"]=pg.term.map(lung_rel); pg=pg.sort_values(["lr","n"],ascending=[False,False])
    out=[]
    for _,r in pg.iterrows():
        ty,desc=code_type(r.term,r.get("psd",""),r.get("etype",""))
        cls="chip"+(" lr" if r.lr else "")
        tip=ty+" \u00b7 seen "+str(int(r.n))+"x in the model window. "+desc
        out.append(f'<span class="{cls}" data-tip="{esc(tip)}">{esc(r.term)}<i>{int(r.n)}</i></span>')
    return "".join(out) or '<span class="chip">(no coded history)</span>'

def patient_block(pid, meta, pregap, gap):
    """pregap: df[snomed,term,n,etype,psd]; gap: df[term,first_ed,sct,psd,sig]"""
    # LEFT: sort lung-relevant first, then by count desc
    pg = pregap.copy()
    pg["lr"] = pg.term.map(lung_rel)
    pg = pg.sort_values(["lr", "n"], ascending=[False, False])
    left_chips = []
    for _, r in pg.iterrows():
        ty, desc = code_type(r.term, r.get("psd", ""), r.get("etype", ""))
        cls = "chip" + (" lr" if r.lr else "")
        tip = f"{ty} · seen {int(r.n)}× in the model window. {desc}"
        left_chips.append(f'<span class="{cls}" data-tip="{esc(tip)}">{esc(r.term)}<i>{int(r.n)}</i></span>')
    # RIGHT: the COMPLETE excluded year. cols: term,n_gap,first_in_gap,psd,is_new,kind
    gp = gap.copy()
    gp["lr"] = gp.term.map(lung_rel)
    gp["is_dx"] = gp.psd.fillna("").str.contains("Problem")
    gp["_ord"] = gp.kind.map({"obs": 0, "med": 1}).fillna(0)
    gp = gp.sort_values(["_ord", "lr", "is_new", "is_dx", "n_gap"],
                        ascending=[True, False, False, False, False])
    right_chips = []
    for _, r in gp.iterrows():
        is_med = (r.kind == "med")
        if is_med:
            ty, desc = "Medication", "A prescribed / issued medication."
        else:
            ty, desc = code_type(r.term, r.get("psd", ""))
        cls = "chip" + (" med" if is_med else (" new" if r.is_new else " rec")) \
              + (" lr" if r.lr else "") + (" dx" if r.is_dx else "")
        d = str(r.first_in_gap)[:10]; n = int(r.n_gap)
        status = "first-ever in the record" if (r.is_new and not is_med) else "recorded before, repeated here"
        extra = f" Flagged: {r.psd}." if str(r.get('psd', '')).strip() else ""
        tip = f"{ty} · {n}× in the excluded year (from {d}) · {status}. {desc}{extra}"
        cnt = f"<i>{n}</i>" if n > 1 else ""
        right_chips.append(f'<span class="{cls}" data-tip="{esc(tip)}">{esc(r.term)}{cnt}</span>')
    n_pre, n_pre_lr = len(pg), int(pg.lr.sum())
    n_gap = len(gp); n_gap_new = int((gp.is_new & (gp.kind == "obs")).sum())
    n_gap_med = int((gp.kind == "med").sum()); n_gap_lr = int(gp.lr.sum())
    # verdict badge (explicit override via meta['badge'], else the cohort classification)
    if meta.get("badge"):
        badge, bcls = meta["badge"], meta.get("bcls", "b-other")
    else:
        cat = meta.get("cat", "other")
        badge, bcls = {"pathway": ("On cancer pathway in gap", "b-path"),
                       "other": ("New diagnoses in gap", "b-other"),
                       "silent": ("Record went silent", "b-silent")}.get(cat, ("New diagnoses in gap", "b-other"))
    left_html = "".join(left_chips) or '<span class="chip">(no coded history)</span>'
    right_html = "".join(right_chips) or '<span class="chip">(nothing new in the excluded year)</span>'
    # workup summary line
    wk = gp[(gp.kind == "obs") & gp.term.map(is_workup)].sort_values("first_in_gap")
    if len(wk):
        wchips = "".join(f'<span class="wchip">{esc(t)}</span>' for t in dict.fromkeys(wk.term))
        workline = (f'<div class="workline hit" data-tip="Codes recorded in the excluded 12 months that are part of a '
                    f'cancer workup — referrals, imaging, clinics, biopsy or malignancy codes '
                    f'nodule/malignancy. The patient was being investigated for cancer inside the year the model excludes.">'
                    f'<b>Cancer workup / {ORGAN} investigations in the excluded year:</b> {wchips}</div>')
    else:
        none_tip = meta.get("workup_none", f"No cancer-workup code (referral, chest imaging, "
                    "respiratory/oncology clinic, bronchoscopy, nodule/malignancy) was recorded in the GP observation "
                    "record during the excluded {GAP} months. The cancer was likely diagnosed via a route not captured "
                    "here — e.g. an emergency/acute admission or a secondary-care workup coded only at diagnosis.")
        workline = (f'<div class="workline none" data-tip="{esc(none_tip)}">'
                    '<b>No coded cancer workup in this window</b> — see the full code list below.</div>')
    why = meta.get("why", ""); why_label = meta.get("why_label", "Why missed")
    why_tip = meta.get("why_tip", "Why the model scored this cancer below its alert line — the SHAP factors that pulled the risk down, from the model's own explanation.")
    whyline = (f'<div class="whyfn {meta.get("why_cls","")}" data-tip="{esc(why_tip)}"><b>{esc(why_label)}:</b> {esc(why)}</div>') if why else ""
    return f'''<article class="pc">
<header><span class="tab">{esc(pid)}</span><span class="dem">{esc(meta['age'])} · {esc(meta['sex'])} · anchor {esc(str(meta['anchor'])[:7])}</span>{('<span class="guid" data-tip="Patient GUID — for tracing this patient back in the source data (local only).">'+esc(meta['guid'])+'</span>') if meta.get('guid') else ''}<span class="badge {bcls}">{badge}</span></header>
{whyline}{workline}
<div class="cols">
  <div class="col left"><div class="ch" data-tip="Every coded entry in the patient's record up to {GAP} months before the anchor. This is exactly the data the model is given to make its prediction.">MODEL WINDOW — before the gap <i>{n_pre} codes · {n_pre_lr} {ORGAN}-relevant</i></div><div class="chipbox">{left_html}</div></div>
  <div class="col right"><div class="ch" data-tip="The final {GAP} months before the anchor (diagnosis), which the model excludes entirely. This shows EVERY code and medication recorded in that year, with how many times. Blue = first-ever in the record; grey = seen before and repeated; green = medication.">EXCLUDED {GAP} MONTHS — the full gap <i>{n_gap} codes · {n_gap_new} new · {n_gap_med} meds · {n_gap_lr} {ORGAN}-relevant</i></div><div class="chipbox">{right_html}</div></div>
</div></article>'''

CSS = '''<style>
:root{--ground:#fff;--surface:#f7f9fb;--text:#16202b;--muted:#5c6b7a;--line:#e2e8ee;--miss:#b0413e;--new:#2e6e8e;--lung:#a8560f;--med:#3f7a52;--workup:#6a4c93;color-scheme:light}
*{box-sizing:border-box}body{margin:0;background:var(--ground);color:var(--text);font:15px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.mono{font-family:ui-monospace,Menlo,monospace}.wrap{max-width:1180px;margin:0 auto;padding:0 24px}
header.hero{padding:48px 0 26px;border-bottom:2px solid var(--text)}
.eyebrow{font:600 12px/1.5 ui-monospace,Menlo,monospace;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin:0 0 14px}
h1{font-size:clamp(26px,4vw,40px);line-height:1.08;letter-spacing:-.02em;font-weight:800;margin:0 0 12px;max-width:22ch;text-wrap:balance}
.lede{font-size:17px;color:var(--muted);max-width:80ch;margin:0}
.key{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0 0}
.key span{font:600 12px/1 ui-monospace,Menlo,monospace;padding:6px 10px;border:1px solid var(--line);border-radius:20px;background:var(--surface);color:var(--muted);cursor:help}
.key .k-lung{color:var(--lung);border-color:color-mix(in srgb,var(--lung) 40%,transparent)}
.key .k-dx{color:var(--miss);border-color:color-mix(in srgb,var(--miss) 40%,transparent)}
.key .k-new{color:var(--new);border-color:color-mix(in srgb,var(--new) 40%,transparent)}
section{padding:26px 0}
.howto{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--new);border-radius:10px;padding:15px 18px;margin:0 0 20px;font-size:14px;line-height:1.6;color:var(--text)}
.howto .mono{font-size:13px}
.pc{border:1px solid var(--line);border-radius:12px;margin:0 0 16px;overflow:hidden;background:#fff}
.pc>header{display:flex;align-items:center;gap:12px;padding:11px 16px;background:var(--surface);border-bottom:1px solid var(--line)}
.tab{font:800 14px/1 ui-monospace,Menlo,monospace;color:var(--text)}.dem{font:12.5px ui-monospace,Menlo,monospace;color:var(--muted)}
.guid{font:11px ui-monospace,Menlo,monospace;color:var(--muted);opacity:.75;user-select:all}
.badge{margin-left:auto;font:700 10.5px/1 ui-monospace,Menlo,monospace;letter-spacing:.04em;text-transform:uppercase;padding:5px 10px;border-radius:20px}
.b-path{background:color-mix(in srgb,var(--miss) 13%,transparent);color:var(--miss)}
.b-other{background:color-mix(in srgb,var(--new) 13%,transparent);color:var(--new)}
.b-silent{background:color-mix(in srgb,var(--muted) 15%,transparent);color:var(--muted)}
.b-workup{background:color-mix(in srgb,var(--workup) 15%,transparent);color:var(--workup)}
.b-disease{background:color-mix(in srgb,var(--lung) 15%,transparent);color:var(--lung)}
.whyfn{padding:9px 16px;font-size:13.5px;line-height:1.5;color:var(--text);border-bottom:1px solid var(--line);background:color-mix(in srgb,var(--miss) 6%,transparent);cursor:help}
.whyfn b{color:var(--miss)}
.whyfn.fp{background:color-mix(in srgb,var(--new) 6%,transparent)}
.whyfn.fp b{color:var(--new)}
.reality{padding:9px 16px;font-size:13.5px;line-height:1.5;color:var(--text);border-bottom:1px solid var(--line);background:color-mix(in srgb,var(--muted) 7%,transparent);cursor:help}
.reality b{color:var(--text)}
.workline{padding:9px 16px;font-size:13px;border-bottom:1px solid var(--line);cursor:help}
.workline.hit{background:color-mix(in srgb,var(--workup) 8%,transparent);color:var(--text)}
.workline.hit b{color:var(--workup)}
.workline.none{background:color-mix(in srgb,var(--muted) 7%,transparent);color:var(--muted)}
.workline .wchip{display:inline-block;font-size:11.5px;background:color-mix(in srgb,var(--workup) 13%,transparent);border:1px solid color-mix(in srgb,var(--workup) 40%,transparent);color:var(--workup);border-radius:6px;padding:2px 7px;margin:2px 3px 0 0;font-weight:600}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:0}
.cols3{grid-template-columns:1fr 1fr 1fr}
.cols3 .col+.col{border-left:1px solid var(--line)}
.col.mid .ch{color:var(--miss)}.col.aft .ch{color:var(--med)}
.col{padding:13px 16px}.col.left{border-right:1px solid var(--line)}
.ch{font:600 11px/1.3 ui-monospace,Menlo,monospace;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:0 0 9px;cursor:help}
.ch i{font-style:normal;font-weight:400;text-transform:none;letter-spacing:0;color:var(--muted);display:block;margin-top:2px;font-size:11px}
.col.right .ch{color:var(--new)}
.chipbox{display:flex;flex-wrap:wrap;gap:5px;max-height:230px;overflow-y:auto;padding-right:4px}
.chip{font-size:11.5px;background:color-mix(in srgb,var(--text) 5%,transparent);border:1px solid var(--line);border-radius:6px;padding:2px 7px;cursor:help;line-height:1.5}
.chip i{font-style:normal;color:var(--muted);margin-left:5px;font-size:10px;font-variant-numeric:tabular-nums}
.chip.lr{background:color-mix(in srgb,var(--lung) 10%,transparent);border-color:color-mix(in srgb,var(--lung) 32%,transparent);color:var(--lung)}
.chip.new{background:color-mix(in srgb,var(--new) 11%,transparent);border-color:color-mix(in srgb,var(--new) 34%,transparent);color:var(--new)}
.chip.rec{background:color-mix(in srgb,var(--text) 4%,transparent);border-color:var(--line);color:var(--muted)}
.chip.med{background:color-mix(in srgb,var(--med) 11%,transparent);border-color:color-mix(in srgb,var(--med) 34%,transparent);color:var(--med)}
.chip.lr{background:color-mix(in srgb,var(--lung) 12%,transparent);border-color:color-mix(in srgb,var(--lung) 40%,transparent);color:var(--lung)}
.chip.dx{font-weight:700}
.summary{background:var(--dark,#16202b);color:#e7ecf1;border-radius:14px;padding:26px 26px 22px;margin:28px 0 8px}
.summary h2{font:600 12px/1 ui-monospace,Menlo,monospace;letter-spacing:.15em;text-transform:uppercase;color:#8fa3b5;margin:0 0 16px}
.summary .cmp{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.summary .box{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);border-radius:10px;padding:15px 17px}
.summary .box h3{font-size:14px;font-weight:800;margin:0 0 9px;color:#fff}
.summary .box.miss h3{color:#e88}.summary .box.good h3{color:#8fd0b0}
.summary .stat{display:flex;align-items:baseline;gap:9px;margin:6px 0;font-size:14px;color:#c3cdd7}
.summary .stat b{font-size:19px;color:#fff;font-variant-numeric:tabular-nums;min-width:44px;display:inline-block}
.summary .btm{font-size:16px;line-height:1.55;margin:18px 0 0;color:#fff}
.summary .btm b{color:#8fd0b0}
footer{padding:24px 0 60px;color:var(--muted);font-size:12.5px;line-height:1.6;border-top:1px solid var(--line);margin-top:20px}
@media(max-width:720px){.summary .cmp{grid-template-columns:1fr}}
#tip{position:fixed;z-index:99;max-width:320px;background:#16202b;color:#fff;font:12.5px/1.45 -apple-system,system-ui,sans-serif;padding:8px 11px;border-radius:7px;box-shadow:0 6px 24px rgba(0,0,0,.22);pointer-events:none;opacity:0;transition:opacity .05s}
@media(max-width:980px){.cols,.cols3{grid-template-columns:1fr}.col.left{border-right:none}.col+.col{border-left:none!important;border-top:1px solid var(--line)}}
@media print{html{-webkit-print-color-adjust:exact;print-color-adjust:exact}.chipbox{max-height:none}.pc{break-inside:avoid}@page{margin:10mm}}
</style>'''

TIP_JS = '''<div id="tip"></div><script>
(function(){var t=document.getElementById('tip');function show(e){var el=e.target.closest('[data-tip]');if(!el){hide();return}t.textContent=el.getAttribute('data-tip');t.style.opacity='1';move(e)}
function move(e){var x=e.clientX+14,y=e.clientY+16;var r=t.getBoundingClientRect();if(x+r.width>innerWidth-8)x=e.clientX-r.width-12;if(y+r.height>innerHeight-8)y=e.clientY-r.height-12;t.style.left=x+'px';t.style.top=y+'px'}
function hide(){t.style.opacity='0'}
document.addEventListener('mouseover',show);document.addEventListener('mousemove',function(e){if(t.style.opacity=='1')move(e)});document.addEventListener('mouseout',function(e){if(e.target.closest('[data-tip]'))hide()});})();
</script>'''

def build_html(cohort, subtitle, blocks_html, n_pat, n_path, n_other, n_silent, tail_html="",
               eyebrow_noun="missed cancers (false negatives)", key_html=None, lede=None, howto=None):
    lede = lede or ("For each patient: <b>left</b> = every coded entry the model was given (up to 12 months before the "
                    "anchor); <b>right</b> = the <b>complete</b> excluded final year — every code and prescription, new or "
                    "repeated. Hover any code for what it is.")
    howto = howto or ('<b>How to read this.</b> The model predicts at the anchor minus 12 months, so the final year — '
                    'including any cancer workup — is never shown to it. <b>Left</b> = the model\'s view: distinct codes in '
                    'its 10-year window (observations + medications, after the cohort filters), with a count of how often '
                    'each was recorded. <b>Right</b> = the <b>complete</b> excluded year: every observation code and every '
                    'prescription in <span class="mono">[anchor−12mo, anchor)</span>, with occurrence counts — '
                    '<span style="color:var(--new);font-weight:700">blue = first-ever in the record</span>, '
                    '<span style="color:var(--muted);font-weight:700">grey = seen before and repeated</span>, '
                    '<span style="color:var(--med);font-weight:700">green = medication</span>. Lung-relevant codes are '
                    'highlighted; problem-list diagnoses are bold.')
    default_key = f'''<div class="key">
<span data-tip="The {n_pat} lung cancers in this cohort that the model scored below its alert line — i.e. the patients it missed (false negatives).">{n_pat} patients</span>
<span class="k-path" style="color:var(--miss)" data-tip="Missed patients who ALREADY had a suspected-lung-cancer referral or workup recorded in the excluded 12-month window — e.g. fast-track lung referral, seen by a respiratory physician, chest X-ray, lung nodule, bronchoscopy. They were being diagnosed on time; the model's prediction point simply sits before this activity.">{n_path} on cancer pathway</span>
<span class="k-new" data-tip="Missed patients with no lung-cancer pathway yet, but who picked up OTHER new coded diagnoses in the excluded year — some are themselves lung red-flags (cough, haemoptysis, chest pain), others unrelated (e.g. heart failure, fracture).">{n_other} other new dx</span>
<span data-tip="Missed patients with no meaningful new coded activity in the excluded year — mostly young, thinly-documented records. Nothing was recorded for the model, or for later care, to act on.">{n_silent} silent</span>
<span class="k-new" data-tip="A code appearing for the FIRST time ever in the patient's record, during the excluded {GAP} months.">■ new (first-ever)</span>
<span style="color:var(--muted)" data-tip="A code that was already in the record before the excluded year and was recorded again during it.">■ recurring</span>
<span style="color:var(--med)" data-tip="A medication prescribed / issued during the excluded {GAP} months.">■ medication</span>
<span class="k-lung" data-tip="Codes related to the chest, respiratory system, or a cancer workup — e.g. cough, chest X-ray, COPD, lung nodule, respiratory referral, smoking.">■ lung-relevant</span>
<span class="k-dx" data-tip="Shown in bold: a code flagged on the patient's problem list as an actual diagnosis (not just a test, measurement or admin entry).">bold = diagnosis</span>
</div>'''
    return f'''<title>Before vs the excluded year ({cohort})</title>{CSS}
<header class="hero"><div class="wrap">
<p class="eyebrow">Lung model · {esc(cohort)} · {eyebrow_noun} · before-gap vs in-gap contrast</p>
<h1>What the model saw, and what happened in the year it can't see.</h1>
<p class="lede">{subtitle} {lede}</p>
{key_html or default_key}
</div></header>
<section><div class="wrap">
<div class="howto">{howto}</div>
{blocks_html}
{tail_html}
<footer>Source: EMIS observations + prescribing. This file is local — patient data stays on your infrastructure.</footer>
</div></section>{TIP_JS}'''

def xlsx_rows(cohort, pid, meta, pregap, gap):
    rows=[]; guid=meta.get("guid","")
    for _,r in pregap.iterrows():
        ty,_=code_type(r.term,r.get("psd",""),r.get("etype",""))
        rows.append([cohort,pid,guid,meta["age"],meta["sex"],str(meta["anchor"])[:10],"model window (before gap)",r.term,int(r.snomed_c_t_concept_id),ty,"yes" if lung_rel(r.term) else "",int(r.n),"","",""])
    for _,r in gap.iterrows():
        is_med=(r.kind=="med")
        ty=("Medication" if is_med else code_type(r.term,r.get("psd",""))[0])
        isdx="yes" if str(r.get("psd","")).find("Problem")>=0 else ""
        newness=("medication" if is_med else ("new (first-ever)" if r.is_new else "recurring"))
        rows.append([cohort,pid,guid,meta["age"],meta["sex"],str(meta["anchor"])[:10],"excluded 12mo (gap)",r.term,"",ty,"yes" if lung_rel(r.term) else "",int(r.n_gap),isdx,str(r.first_in_gap)[:10],newness])
    return rows
