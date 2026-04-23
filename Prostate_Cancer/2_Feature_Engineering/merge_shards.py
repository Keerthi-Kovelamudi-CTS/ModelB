"""
Merge sharded obs CSVs into a single prostate_{window}_obs.csv per window.

Expects files in data/{3mo,6mo,12mo}/ named prostate_{window}_obs-0000000000NN.csv
(BigQuery export format). Strips headers from shards 2..N, streams contents in
16 MB chunks (memory-safe for multi-GB merges), deletes shards after success.

Idempotent: if the merged file already exists and no shards remain, does nothing.

Usage:
    python3 merge_shards.py                 # merge all windows in config.WINDOWS
    python3 merge_shards.py --window 12mo   # merge a single window
"""
import argparse
import sys
from pathlib import Path

import config


def merge_window(window: str) -> None:
    wdir = Path(config.BASE_PATH) / window
    if not wdir.exists():
        print(f"  [skip] {window}: {wdir} does not exist")
        return

    shards = sorted(wdir.glob(f"{config.DATA_PREFIX}_{window}_obs-*.csv"))
    out = wdir / f"{config.DATA_PREFIX}_{window}_obs.csv"

    if not shards:
        if out.exists():
            size_gb = out.stat().st_size / (1024 ** 3)
            print(f"  [ok] {window}: already merged ({size_gb:.2f} GB), no shards present")
        else:
            print(f"  [skip] {window}: no shards and no merged file")
        return

    print(f"=== {window}: merging {len(shards)} shards → {out.name} ===")
    first_header = None
    with open(out, "wb") as o:
        for i, s in enumerate(shards):
            with open(s, "rb") as f:
                header = f.readline()
                if i == 0:
                    first_header = header
                    o.write(header)
                else:
                    if header != first_header:
                        raise RuntimeError(
                            f"Header mismatch in {s.name}:\n  first: {first_header!r}\n  got:   {header!r}"
                        )
                while True:
                    buf = f.read(16 * 1024 * 1024)
                    if not buf:
                        break
                    o.write(buf)
            print(f"  + {s.name}  ({s.stat().st_size / (1024 ** 2):.1f} MB)")

    size_gb = out.stat().st_size / (1024 ** 3)
    print(f"  merged size: {size_gb:.2f} GB")

    for s in shards:
        s.unlink()
    print(f"  deleted {len(shards)} shard files")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", choices=config.WINDOWS + ["all"], default="all")
    args = ap.parse_args()
    windows = config.WINDOWS if args.window == "all" else [args.window]
    for w in windows:
        merge_window(w)
    print("--- done ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
