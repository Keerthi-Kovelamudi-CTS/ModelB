"""Validate a base team's prediction parquet(s) against the shared contract BEFORE stacking.
Run this on every file each team delivers — fail fast on contract violations.

Usage:
  python validate_preds.py path/to/llm_preds_valid.parquet [more.parquet ...]
  python validate_preds.py ./preds/*.parquet
"""
import sys, os, pandas as pd
from contract import validate_df

def main(paths):
    rc = 0
    for p in paths:
        df = pd.read_parquet(p)
        split = "valid" if "_valid" in os.path.basename(p) else ("test" if "_test" in os.path.basename(p) else None)
        ok, probs = validate_df(df, expect_split=split)
        tag = "OK ✅" if ok else "FAIL ❌"
        m = df["model_name"].iloc[0] if "model_name" in df else "?"
        print(f"[{tag}] {os.path.basename(p)}  model={m} n={len(df)} "
              f"abstain={int(df['abstained'].fillna(False).sum()) if 'abstained' in df else '?'}")
        for x in probs: print("    " + x)
        rc = rc or (0 if ok else 1)
    sys.exit(rc)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python validate_preds.py <preds.parquet> [...]")
    main(sys.argv[1:])
