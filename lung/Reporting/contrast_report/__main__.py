"""contrast_report — deterministic FN/FP sheets + deep-dive for ANY cancer. No Claude, PHI stays local.

  python -m contrast_report --run <RUN_DIR> --config configs/lung.yaml --cohort all --out <OUT_DIR>

--cohort: fn | fp | deepdive | all   ·   BigQuery is cached to <OUT_DIR>/bq/ (re-runs are free).
"""
import argparse, os, sys, subprocess, json
from . import pipeline as P

def main():
    ap=argparse.ArgumentParser(prog="contrast_report")
    ap.add_argument("--run", required=True, help="run dir with modeling/explainability_internal + fe/*_stable.parquet")
    ap.add_argument("--config", required=True, help="per-cancer YAML (see configs/)")
    ap.add_argument("--cohort", default="all", choices=["fn","fp","deepdive","all"])
    ap.add_argument("--out", default=None, help="output dir (default <run>/contrast_report_out)")
    ap.add_argument("--no-bq", action="store_true", help="require cached BQ CSVs; never scan")
    a=ap.parse_args()

    cfg=P.load_config(a.config); organ=cfg["organ"]
    out=a.out or f"{a.run}/contrast_report_out"; os.makedirs(f"{out}/bq", exist_ok=True)
    h, stable, gap, thr = P.resolve_params(a.run, cfg)
    print(f"[cfg] cancer={cfg['cancer']} organ={organ} horizon={h} gap={gap} threshold={thr}")
    _, fn, fp = P.identify(a.run, stable, thr)
    print(f"[errors] FN={len(fn)} FP={len(fp)} at threshold {thr}")
    cache = P.cache_local(cfg["cache_uri"], out)
    live = not a.no_bq
    results={}

    if a.cohort in ("fn","all"):
        anc,pregap = P.anchors_pregap(cache, set(fn.guid))
        gd = P.pull_gap("fn", anc, gap, cfg["bq_project"], f"{out}/bq", live)
        results["fn"]=P.build_fn(cfg, organ, gap, thr, anc, pregap, fn, gd, out)
        print("[fn]", results["fn"])
    if a.cohort in ("fp","all"):
        anc,pregap = P.anchors_pregap(cache, set(fp.guid))
        gd = P.pull_gap("fp", anc, gap, cfg["bq_project"], f"{out}/bq", live)
        results["fp"]=P.build_fp(cfg, organ, gap, thr, anc, pregap, fp, gd, out)
        print("[fp]", results["fp"])
    if a.cohort in ("deepdive","all"):
        dd=os.path.join(os.path.dirname(__file__),"deepdive.py")
        cmd=[sys.executable, dd, "--run", a.run, "--out", f"{out}/deepdive.html",
             "--threshold", str(thr), "--organ", organ,
             "--model-short", cfg.get("model_name", cfg["cancer"]), "--auroc", str(cfg.get("auroc","—"))]
        subprocess.run(cmd, check=True); results["deepdive"]=f"{out}/deepdive.html"
        print("[deepdive]", results["deepdive"])

    print("\nDONE. Local outputs in:", out, "\n(patient-identifiable — do NOT publish/upload)")
    return results

if __name__=="__main__":
    main()
