"""manifest.py — freeze a model "book" into a versioned, provenance-bearing manifest.

WHY THIS EXISTS (read MODEL_REGISTRY.md): the frozen GBM books in this program are trained multithreaded
(`n_jobs/thread_count=20`) WITHOUT a seed on the estimators, so they are NOT bit-for-bit retrainable — the
persisted artifact is the ONLY verbatim copy. So the reproducibility strategy is FREEZE-THE-ARTIFACT +
RECORD-A-MANIFEST (not config-driven re-derivation). A manifest is a RECORD (written at save time), not a
DRIVER (read by training). It binds a stable semantic id -> {git sha, hyperparams, feature/data fingerprint,
env versions, artifact sha256s, metrics} so a model can be referenced directly and re-loaded verbatim later.

Use at the end of a production train() to stamp the book:
    import manifest
    m = manifest.build(book_id="EURUSD.m15.v1", timeframe="15m", side="combined", role="direction",
                       script="m15_production.py", summary="...", metrics={...},
                       artifacts=[art("direction_lgb.txt"), art("direction_xgb.json"), art("direction_cat.cbm")],
                       hyperparams={...}, strategy_json="models/m15_EURUSD_strategy.json",
                       feature_fingerprint=manifest.dir_fingerprint(H.FEAT_DIR, ("EURUSD_*.parquet",)))
    manifest.freeze(m, artifacts_src=[...], books_dir="books")   # copies artifacts + writes books/<id>.manifest.json
"""
import json, hashlib, subprocess, os, sys, glob, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def git_dirty():
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        return None


def file_sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_fingerprint(d, patterns=("*.parquet",)):
    """Fast content-change fingerprint of a data dir: hashes the sorted (name:size:mtime) listing.
    Does NOT read file contents (cheap), but detects any change to the feature set that produced a book."""
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(d, p))
    files = sorted(set(files))
    listing, total = [], 0
    for f in files:
        st = os.stat(f)
        listing.append(f"{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}")
        total += st.st_size
    return {"dir": d, "patterns": list(patterns), "n_files": len(files), "total_bytes": total,
            "sha256_of_listing": hashlib.sha256("\n".join(listing).encode()).hexdigest()}


def env_snapshot(py=None):
    """Capture the ACTUAL training-interpreter versions of the libraries that determine tree construction.
    (Hand-maintained notes drift — e.g. ENVIRONMENT_libs.txt said pandas 3.0.3 while the venv ran 2.3.3.)
    Uses importlib.metadata (works on pip-less uv venvs); reports the interpreter that imported this module."""
    from importlib import metadata as ilm
    keep = ["lightgbm", "xgboost", "catboost", "scikit-learn", "numpy", "pandas", "pyarrow", "scipy"]
    pinned = {}
    for name in keep:
        try:
            pinned[name] = ilm.version(name)
        except Exception:
            pinned[name] = None
    return {"python": py or sys.executable, "python_version": sys.version.split()[0], "key_libs": pinned}


def build(book_id, *, timeframe, side, role, script, summary, metrics, artifacts, hyperparams,
          strategy_json=None, feature_fingerprint=None, depends_on=None, notes=None, created_utc=None):
    """Assemble a manifest dict. `artifacts` = list of on-disk paths to the trained-model files (hashed here)."""
    arts = [{"file": os.path.basename(a), "bytes": os.path.getsize(a), "sha256": file_sha256(a)}
            for a in artifacts]
    return {
        "schema": "book-manifest/v1",
        "id": book_id,
        "currency": "EURUSD",
        "timeframe": timeframe,        # "1-5s" | "60s" | "5m" | "10m" | "15m" | "30m" | "120s"
        "side": side,                  # "up" | "down" | "combined"
        "role": role,                  # "direction" | "magnitude" | "gate" | "stack-meta"
        "summary": summary,
        "metrics": metrics,            # e.g. {"oos_2026": 0.663, "combined": 0.647, "cpcv_mean": 0.579, "breakeven": 0.541}
        "source": {"script": script, "git_sha": git_sha(), "git_dirty": git_dirty()},
        "hyperparams": hyperparams,    # best-effort; source.script @ git_sha is AUTHORITATIVE for full config
        "artifacts": arts,             # {file, bytes, sha256} — the verbatim model bytes
        "strategy_json": strategy_json,
        "feature_fingerprint": feature_fingerprint,
        "env": env_snapshot(),
        "determinism": {
            "estimator_seed": "unset",
            "threads": "n_jobs/thread_count=20",
            "bitwise_retrainable": False,
            "note": "GBMs trained multithreaded without a seed -> NOT bit-for-bit retrainable; the persisted "
                    "artifact is the only verbatim copy. Re-running the script yields a DIFFERENT model.",
        },
        "depends_on": depends_on,      # e.g. a stack book that consumes a parent book's artifacts
        "notes": notes,
        "created_utc": created_utc,    # pass in to keep manifests stable across regenerations
    }


def freeze(m, artifacts_src, books_dir=os.path.join(ROOT, "books")):
    """Copy the trained-model artifacts into books/<id>/ (a version-controlled frozen snapshot) and write
    books/<id>.manifest.json. The copy doubles as an on-drive backup independent of the working models/ dir."""
    bid = m["id"]
    dest = os.path.join(books_dir, bid)
    os.makedirs(dest, exist_ok=True)
    for src in artifacts_src:
        shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
    mpath = os.path.join(books_dir, f"{bid}.manifest.json")
    with open(mpath, "w") as f:
        json.dump(m, f, indent=2)
    return mpath
