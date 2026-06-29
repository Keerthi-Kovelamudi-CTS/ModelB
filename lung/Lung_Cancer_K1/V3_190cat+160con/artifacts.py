"""artifacts.py — atomic artifact writes + provenance/freshness manifests.

Every pipeline artifact can carry a sidecar ``{path}.manifest.json`` recording the git commit, a
timestamp, the *config slice* that shaped it, and the content hashes of its *upstream* inputs. This:

  (a) gives provenance — open any artifact and see exactly what code + config + inputs produced it; and
  (b) lets the runner bust a cached artifact when the code, the relevant config, or an upstream input
      changes — closing the "edited TOP_N but the codelist filename is unchanged so it's silently
      reused" gotcha and the "stale residue file left by an old run" gotcha.

Backward-compatible: an artifact with NO sidecar is treated as fresh-if-present (the legacy
existence-only behaviour), so existing caches keep working; content-aware busting starts the moment an
artifact is (re)written with a manifest.

Pure-Python; the only optional dependency is gcsfs (only when paths are gs://).
"""
import os
import json
import hashlib
import subprocess
import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))


def git_sha(_cache={}):
    """Current repo commit (or 'unknown'). Cached for the process."""
    if "v" not in _cache:
        try:
            r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_HERE,
                               capture_output=True, text=True, timeout=5)
            _cache["v"] = (r.stdout or "").strip() or "unknown"
        except Exception:
            _cache["v"] = "unknown"
    return _cache["v"]


def _is_gcs(p):
    return str(p).startswith("gs://")


def _fs():
    import gcsfs
    return gcsfs.GCSFileSystem()


def _gcs_blocked(path):
    """One global GCS-write switch (local writes always happen). Default = writes ON. Set GCS_WRITE=0 to turn
    OFF all GCS writes for read-only tooling / experiments so they can never overwrite production artifacts.
    Global by design — the zero-fit artifacts are produced together and must stay mutually consistent, so the
    unit is 'this run may publish or not', not per-artifact toggles.
        GCS_WRITE=0 -> OFF (no GCS writes)   |   GCS_WRITE=1 / unset -> ON (default, writes)"""
    if not _is_gcs(path):
        return False
    off = os.environ.get("GCS_WRITE", "").lower() in ("0", "false", "no")
    if off:
        print(f"[artifacts] GCS_WRITE=0 -> SKIP GCS write {path}")
    return off


def exists(path):
    if _is_gcs(path):
        try:
            return _fs().exists(path)
        except Exception:
            return False
    return os.path.exists(path)


def file_hash(path, _algo="sha256", _chunk=1 << 20):
    """Content identity of an artifact. Local: streamed sha256. GCS: the object's stored checksum
    (crc32c/md5/etag) so we never download. Returns '' if the file is absent/unreadable."""
    if _is_gcs(path):
        try:
            info = _fs().info(path)
            for k in ("crc32c", "md5Hash", "etag"):
                if info.get(k):
                    return f"{k}:{info[k]}"
            return f"size:{info.get('size', '?')}"
        except Exception:
            return ""
    if not os.path.exists(path):
        return ""
    h = hashlib.new(_algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(_chunk), b""):
            h.update(b)
    return f"{_algo}:{h.hexdigest()}"


def manifest_path(path):
    return str(path) + ".manifest.json"


def ensure_local(local, gcs):
    """Pull `gcs` -> `local` (+ its manifest sidecar) if local is missing and gcs exists. Returns True if
    `local` is present afterwards. Reads are ALWAYS allowed (GCS_WRITE only gates writes). No-op if `gcs`
    is falsy / not a gs:// path. Lets held-out reuse a GCS-published matrix instead of rebuilding — the same
    pull-before-(re)compute behaviour the internal run_v3 stages already have via their own ensure_local."""
    if os.path.exists(local):
        return True
    if not gcs or not _is_gcs(gcs):
        return False
    try:
        fs = _fs()
        if fs.exists(gcs):
            os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
            fs.get(gcs, local)
            print(f"[artifacts] pulled {gcs} -> {local}")
            mg = manifest_path(gcs)
            if fs.exists(mg):                       # sidecar too, so freshness is judged locally
                fs.get(mg, manifest_path(local))
    except Exception as e:
        print(f"[artifacts][warn] GCS pull failed for {gcs}: {e}")
    return os.path.exists(local)


def build_key(config_slice=None, upstream=None):
    """The cache identity of an artifact: producing commit + the config slice that shaped it + the
    content hashes of its upstream inputs. Two artifacts with the same key are interchangeable."""
    return {
        "git_sha": git_sha(),
        "config": {k: config_slice[k] for k in sorted(config_slice or {})},
        "upstream": {os.path.basename(str(u)): file_hash(u) for u in (upstream or [])},
    }


def _write_json_atomic(path, obj):
    if _is_gcs(path):
        if _gcs_blocked(path):
            return
        with _fs().open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def write_manifest(path, config_slice=None, upstream=None, extra=None):
    """Write {path}.manifest.json (atomically) capturing git sha + config slice + upstream hashes."""
    m = build_key(config_slice, upstream)
    m["created_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    m["artifact"] = os.path.basename(str(path))
    if extra:
        m.update(extra)
    _write_json_atomic(manifest_path(path), m)
    return m


def read_manifest(path):
    mp = manifest_path(path)
    if not exists(mp):
        return None
    try:
        if _is_gcs(mp):
            with _fs().open(mp) as f:
                return json.load(f)
        with open(mp) as f:
            return json.load(f)
    except Exception:
        return None


def is_fresh(path, config_slice=None, upstream=None):
    """Fresh = the artifact exists AND either it has no manifest (legacy artifact -> existence-only,
    backward compatible) OR its manifest key matches the expected (git sha + config + upstream)."""
    if not exists(path):
        return False
    m = read_manifest(path)
    if m is None:
        return True                       # legacy artifact, no manifest -> trust existence (back-compat)
    want = build_key(config_slice, upstream)
    return (m.get("git_sha") == want["git_sha"]
            and m.get("config") == want["config"]
            and m.get("upstream") == want["upstream"])


def stale_reason(path, config_slice=None, upstream=None):
    """Human-readable reason an artifact is NOT fresh (for logging) — '' if fresh."""
    if not exists(path):
        return "missing"
    m = read_manifest(path)
    if m is None:
        return ""                         # legacy, treated as fresh
    want = build_key(config_slice, upstream)
    if m.get("git_sha") != want["git_sha"]:
        return f"git_sha {m.get('git_sha','?')[:8]} != {want['git_sha'][:8]}"
    if m.get("config") != want["config"]:
        return "config changed"
    if m.get("upstream") != want["upstream"]:
        return "upstream input(s) changed"
    return ""


def upload_dir(local_dir, gcs_dir):
    """Mirror every file in `local_dir` up to the `gs://` prefix `gcs_dir` (flat — explainability
    writes a single flat folder). Returns the list of uploaded gs:// paths. No-op (returns []) if
    `gcs_dir` isn't a gs:// path or `local_dir` doesn't exist, so it's safe to call in local-only runs."""
    if not _is_gcs(gcs_dir) or not os.path.isdir(local_dir):
        return []
    if _gcs_blocked(gcs_dir):
        return []
    fs = _fs()
    sent = []
    for name in sorted(os.listdir(local_dir)):
        lp = os.path.join(local_dir, name)
        if os.path.isfile(lp):
            dst = gcs_dir.rstrip("/") + "/" + name
            fs.put(lp, dst)
            sent.append(dst)
    return sent


def atomic_write(path, writer):
    """`writer(target)` writes the artifact to `target`; we make the publish atomic. Local: write to
    a .tmp then os.replace (truly atomic — readers never see a partial file). GCS: a single-object
    write (readers see it only on close), so write directly. GCS_READONLY=1 skips GCS writes."""
    if _is_gcs(path):
        if _gcs_blocked(path):
            return path
        writer(path)
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = str(path) + ".tmp"
    writer(tmp)
    os.replace(tmp, path)
    return path


def atomic_write_text(path, text):
    """Atomically write a TEXT artifact. The handle is `with`-managed so the file is flushed+closed
    BEFORE the os.replace — guaranteed on any runtime (not relying on CPython refcount GC to flush a
    bare `open(t,'w').write(...)` before the rename). gs://-aware (uses gcsfs), mirroring the local/GCS
    split in atomic_write / _write_json_atomic, so it's a safe drop-in for any path."""
    def _w(target):
        if _is_gcs(target):
            if _gcs_blocked(target):
                return
            with _fs().open(target, "w") as f:    # gcsfs handle (NOT builtin open, which would make a bogus local 'gs:/…' file)
                f.write(text)
        else:
            with open(target, "w", encoding="utf-8") as f:
                f.write(text)
    return atomic_write(path, _w)
