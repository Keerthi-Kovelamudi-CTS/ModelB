"""
Merge BigQuery-exported unified shards into a single per-window CSV.

Input:  data/{3mo,6mo,12mo}/Unified{w}mo-NNNNNNNNNN
        (no extension, header on first shard only, CSV format,
        obs + med rows mixed, distinguished by `event_type` column)

Output: data/{w}/prostate_{w}.csv
        (single unified CSV — keeps obs + med rows together;
         1_sanity_check.py splits them in memory at load time)

Streams in 16 MB chunks (memory-safe for multi-GB inputs). First shard's
header is kept; remaining shards' headers are stripped. Header mismatch
raises an error. Shards are deleted after a successful merge.

Usage:
    python3 merge_unified_shards.py
    python3 merge_unified_shards.py --window 12mo
    python3 merge_unified_shards.py --keep-shards
"""
import argparse
import sys
from pathlib import Path

import config


def merge_window(window: str, keep_shards: bool = False) -> None:
    wdir = Path(config.BASE_PATH) / window
    if not wdir.exists():
        print(f"  [skip] {window}: {wdir} does not exist")
        return

    shards = sorted([p for p in wdir.iterdir()
                     if p.is_file() and p.name.startswith(f"Unified{window}-")])
    out_path = wdir / f"{config.DATA_PREFIX}_{window}.csv"

    if not shards:
        if out_path.exists():
            size_gb = out_path.stat().st_size / (1024 ** 3)
            print(f"  [ok] {window}: already merged ({size_gb:.2f} GB), no shards present")
        else:
            print(f"  [skip] {window}: no shards and no merged file")
        return

    print(f"=== {window}: merging {len(shards)} shards → {out_path.name} ===")
    first_header: bytes | None = None
    with open(out_path, "wb") as out:
        for i, shard in enumerate(shards):
            with open(shard, "rb") as f:
                header = f.readline()
                if i == 0:
                    first_header = header
                    out.write(header)
                elif header != first_header:
                    raise RuntimeError(
                        f"Header mismatch in {shard.name}:\n"
                        f"  first: {first_header!r}\n"
                        f"  got:   {header!r}"
                    )
                while True:
                    buf = f.read(16 * 1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
            print(f"  + {shard.name} ({shard.stat().st_size / (1024 ** 2):.1f} MB)")

    size_gb = out_path.stat().st_size / (1024 ** 3)
    print(f"  merged size: {size_gb:.2f} GB")

    if not keep_shards:
        for s in shards:
            s.unlink()
        print(f"  deleted {len(shards)} shards")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", choices=config.WINDOWS + ["all"], default="all")
    ap.add_argument("--keep-shards", action="store_true",
                    help="Don't delete shards after merge (default: delete)")
    args = ap.parse_args()
    windows = config.WINDOWS if args.window == "all" else [args.window]
    for w in windows:
        merge_window(w, keep_shards=args.keep_shards)
    print("--- done ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
