#!/usr/bin/env python3
"""
Upload selected file types under the repo root to GCS, preserving relative paths.

Default: gs://gcs-ai-dev-model-artifacts/keerthi/<relative/path>

Examples:
  python3 scripts/upload_to_gcs.py --include joblib,pkl
  python3 scripts/upload_to_gcs.py --include csv,joblib,pkl,parquet
  python3 scripts/upload_to_gcs.py --dry-run --include joblib,pkl

Prerequisites:
  pip install google-cloud-storage
  gcloud auth application-default login
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, FrozenSet, List, Optional, Tuple

if TYPE_CHECKING:
    from google.cloud.storage import Client as StorageClient

# suffix -> glob fragment (lowercase)
KNOWN = ("csv", "joblib", "pkl", "pickle", "parquet")


def parse_include(s: str) -> FrozenSet[str]:
    parts = {x.strip().lower().lstrip(".") for x in s.split(",") if x.strip()}
    bad = parts - set(KNOWN)
    if bad:
        raise SystemExit(f"Unknown extension(s) in --include: {sorted(bad)}. Allowed: {', '.join(KNOWN)}")
    return frozenset(parts)


def find_files(root: Path, extensions: FrozenSet[str]) -> List[Path]:
    out: List[Path] = []
    for ext in sorted(extensions):
        pat = f"*.{ext}"
        for p in root.rglob(pat):
            if ".git" in p.parts:
                continue
            out.append(p)
    return sorted(set(out))


def upload_one(
    client: StorageClient, bucket_name: str, prefix: str, root: Path, path: Path
) -> Tuple[str, Optional[str]]:
    rel = path.relative_to(root).as_posix()
    dest = f"{prefix.strip('/')}/{rel}" if prefix.strip("/") else rel
    blob = client.bucket(bucket_name).blob(dest)
    try:
        blob.upload_from_filename(str(path))
        return (rel, None)
    except Exception as e:  # noqa: BLE001
        return (rel, str(e))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload CSV / model / table files to GCS with stable paths."
    )
    parser.add_argument(
        "--bucket",
        default="gcs-ai-dev-model-artifacts",
        help="GCS bucket name (no gs:// prefix)",
    )
    parser.add_argument(
        "--prefix",
        default="keerthi",
        help="Object prefix inside the bucket",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root to scan (default: repo root = parent of scripts/)",
    )
    parser.add_argument(
        "--include",
        default="joblib,pkl,parquet,pickle",
        help=f"Comma-separated extensions to upload. Allowed: {','.join(KNOWN)}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel upload threads",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files only",
    )
    args = parser.parse_args()

    root = (args.root or Path(__file__).resolve().parent.parent).resolve()
    extensions = parse_include(args.include)

    try:
        from google.cloud import storage
    except ImportError:
        print("Install: pip install google-cloud-storage", file=sys.stderr)
        return 1

    files = find_files(root, extensions)
    if not files:
        print(f"No files matching --include under {root}", file=sys.stderr)
        return 1

    print(f"Root: {root}")
    print(f"Include: {', '.join(sorted(extensions))}")
    print(f"Destination: gs://{args.bucket}/{args.prefix.strip('/')}/...")
    print(f"Files: {len(files)}")

    if args.dry_run:
        for p in files[:30]:
            print(" ", p.relative_to(root))
        if len(files) > 30:
            print(f"  ... and {len(files) - 30} more")
        return 0

    client = storage.Client()
    errors: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(upload_one, client, args.bucket, args.prefix, root, p) for p in files]
        done = 0
        for fut in as_completed(futs):
            rel, err = fut.result()
            done += 1
            if err:
                errors.append((rel, err))
                print(f"[{done}/{len(files)}] FAIL {rel}: {err}", file=sys.stderr)
            elif done % 25 == 0 or done == len(files):
                print(f"[{done}/{len(files)}] uploaded...")

    if errors:
        print(f"\nFinished with {len(errors)} error(s) out of {len(files)}.", file=sys.stderr)
        return 1

    print(f"Done. Uploaded {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
