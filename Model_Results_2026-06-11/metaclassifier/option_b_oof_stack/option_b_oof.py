"""Option B — k-fold OOF stack, XGBoost base on the b12_v3 shared feature table.

Delivers two parquets (contract schema, braced PATIENT_GUID):
  xgboost_v<VER>_preds_train_oof.parquet  (4872 rows, each train patient predicted by a model NOT trained on its fold)
  xgboost_v<VER>_preds_test.parquet       (610 rows, predicted by a model trained on full TRAIN only)

Features: features_emis.lung_patient_snomed_json_no_heldout_b12_v3 -> transformed_features JSON expanded to wide.
Fold map: shared/fold_map_b12_v3_k5_seed42.parquet (NEVER regenerate).

Run on cpu-02 (bq + xgboost). Shared bundle must be in --shared dir.
"""
import argparse, os, json
import numpy as np, pandas as pd
from google.cloud import bigquery
import xgboost as xgb

FEAT = "prj-cts-ai-dev-sp.features_emis.lung_patient_snomed_json_no_heldout_b12_v3"
VER = "b12v3_oof_20260611"

def braced(s):
    g = str(s).replace("{", "").replace("}", "").replace('"', "").strip().upper()
    return "{" + g + "}"

def xgb_model():
    # balanced b12_v3 (50%) -> no scale_pos_weight; sensible defaults, deterministic
    return xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                             reg_lambda=1.0, eval_metric="logloss", n_jobs=8,
                             random_state=42, tree_method="hist")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", required=True)   # dir with fold_map + test_patient_ids parquet
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    bq = bigquery.Client()

    print("Pulling feature table...")
    raw = bq.query(f"SELECT patient_guid, cancer_class, transformed_features FROM `{FEAT}`").to_dataframe()
    raw["pid"] = raw["patient_guid"].map(braced)
    feats = pd.json_normalize(raw["transformed_features"].map(lambda x: json.loads(x) if isinstance(x, str) else (x or {})))
    obj = [c for c in feats.columns if feats[c].dtype == object]
    num = [c for c in feats.columns if c not in obj]
    # one-hot only genuine low-cardinality categoricals (e.g. *_TREND_DIRECTION); DROP high-card
    # object cols (absolute *_DATE strings) — they are not features + risk leakage; numeric trend/
    # change/slope features already carry the temporal signal.
    low_card = [c for c in obj if feats[c].nunique() <= 20]
    dropped = [c for c in obj if c not in low_card]
    Xnum = feats[num].apply(pd.to_numeric, errors="coerce")
    Xcat = pd.get_dummies(feats[low_card].astype(str), dummy_na=False) if low_card else pd.DataFrame(index=feats.index)
    X = pd.concat([raw[["pid"]].reset_index(drop=True), Xnum.reset_index(drop=True), Xcat.reset_index(drop=True)], axis=1).fillna(0.0)
    feat_cols = [c for c in X.columns if c != "pid"]
    print(f"  feature matrix: {X.shape} ({len(num)} numeric + {Xcat.shape[1]} one-hot from {len(low_card)} cats; "
          f"dropped {len(dropped)} high-card date/string cols)")

    fold_map = pd.read_parquet(os.path.join(a.shared, "fold_map_b12_v3_k5_seed42.parquet")); fold_map["pid"] = fold_map["patient_id"].map(braced)
    test_ids = pd.read_parquet(os.path.join(a.shared, "test_patient_ids.parquet"))
    tcol = [c for c in test_ids.columns if "id" in c.lower() or "guid" in c.lower()][0]
    test_ids["pid"] = test_ids[tcol].map(braced)

    train = fold_map.merge(X, on="pid", how="left")
    test = test_ids[["pid"]].merge(X, on="pid", how="left")
    assert train[feat_cols].isna().any(axis=1).sum() == 0, "train patients missing features"
    assert len(train) == 4872 and len(test) == 610, f"counts: train {len(train)} test {len(test)}"

    # ---- k-fold OOF ----
    print("OOF (5 folds)...")
    oof_parts = []
    for f in range(5):
        tr = train[train.fold != f]; va = train[train.fold == f]
        m = xgb_model(); m.fit(tr[feat_cols].values, tr["label"].values)
        p = m.predict_proba(va[feat_cols].values)[:, 1]
        oof_parts.append(pd.DataFrame({"patient_id": va["patient_id"].values, "split": "train_oof",
                                       "fold": f, "proba_1": p.astype(float)}))
        print(f"  fold {f}: train {len(tr)} predict {len(va)}")
    oof = pd.concat(oof_parts, ignore_index=True)

    # ---- full train -> test ----
    print("Full train -> test...")
    mf = xgb_model(); mf.fit(train[feat_cols].values, train["label"].values)
    pt = mf.predict_proba(test[feat_cols].values)[:, 1]
    tst = pd.DataFrame({"patient_id": test_ids[tcol].map(braced).values, "split": "test", "fold": -1, "proba_1": pt.astype(float)})

    for d in (oof, tst):
        d["proba_0"] = 1.0 - d["proba_1"]; d["model_name"] = "xgboost"; d["model_version"] = VER; d["abstained"] = False
    cols = ["patient_id", "split", "proba_1", "proba_0", "fold", "model_name", "model_version", "abstained"]
    oof, tst = oof[cols], tst[cols]

    # ---- self-check ----
    assert len(oof) == 4872 and oof.patient_id.nunique() == 4872, "OOF count"
    assert len(tst) == 610 and tst.patient_id.nunique() == 610, "TEST count"
    assert set(oof.patient_id) & set(tst.patient_id) == set(), "OOF/TEST overlap"
    chk = oof.merge(fold_map[["patient_id", "fold"]], on="patient_id", suffixes=("_p", "_m"))
    assert (chk.fold_p == chk.fold_m).all(), "fold mismatch vs fold_map"
    assert tst.fold.eq(-1).all() and oof.split.eq("train_oof").all() and tst.split.eq("test").all()
    print("SELF-CHECK PASSED ✅")

    of = os.path.join(a.out, f"xgboost_v{VER}_preds_train_oof.parquet"); oof.to_parquet(of, index=False)
    tf = os.path.join(a.out, f"xgboost_v{VER}_preds_test.parquet"); tst.to_parquet(tf, index=False)
    # quick OOF AUROC sanity (not used downstream)
    from sklearn.metrics import roc_auc_score
    print(f"OOF AUROC (sanity) = {roc_auc_score(fold_map.set_index('patient_id').loc[oof.patient_id,'label'].values, oof.proba_1.values):.4f}")
    print(f"-> {of}\n-> {tf}")

if __name__ == "__main__":
    main()
