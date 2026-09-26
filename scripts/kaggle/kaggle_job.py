"""AMLC26 Kaggle job driver (runs INSIDE a Kaggle kernel from the code bundle). Fail-closed at every step.

usage (from the kernel stub): python scripts/kaggle/kaggle_job.py '<JOB JSON>'
JOB = {"kind": "h4" | "dev" | "test_shards" | "merge",
       "bundle_commit": "<40-hex>", "config_sha256": "<sha of experiments/configs/h4_x2_frozen.json>",
       "shards": {"country": "India", "count": 6, "indices": [0, 1, 2, 3, 4, 5]}}      # test_shards only
Checks before any work: BUNDLE_COMMIT == job commit; canonical config sha == job sha; repository tests pass.
h4 / dev    : agy2_pipeline.py (mode) -> /kaggle/working/<kind>/{export,artifacts,pipeline_<kind>_results.json}
test_shards : export located under /kaggle/input (manifest mode must be h4) -> one pipeline run per shard index
merge       : every shard_*_meta.json under /kaggle/input -> agy2_pipeline_merge.py + organiser validator
All children run sequentially (one heavy process at a time); run_summary.json records exit codes, wall time, peak RSS.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INPUT = Path(os.environ.get("AMLC_INPUT", "/kaggle/input"))


def find_data_root(input_dir: Path = INPUT) -> Path:
    """Official data root under ANY mount layout: the single train_source1.tsv, climbed to the nearest ancestor that
    also holds train_ground_truth.tsv and test_source1.tsv. Fails closed on zero or several candidates."""
    hits = sorted(input_dir.rglob("train_source1.tsv"))
    if len(hits) != 1:
        raise SystemExit(f"FAIL-CLOSED: expected exactly one train_source1.tsv under {input_dir}, found {len(hits)}")
    d = hits[0].parent
    while True:
        if list(d.rglob("train_ground_truth.tsv")) and list(d.rglob("test_source1.tsv")):
            return d
        if d == input_dir or d.parent == d:
            raise SystemExit("FAIL-CLOSED: official data root (train + test files) not found")
        d = d.parent


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(job: dict) -> int:
    t0 = time.time()
    out = Path("/kaggle/working") / job["kind"]
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "stdout.log", "w")

    def run(cmd, env):
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
        return p.wait()

    summary = {"job": job, "steps": []}

    def fail(msg):
        summary["error"] = msg
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print("FAIL-CLOSED:", msg, flush=True)
        return 1

    commit = (REPO / "BUNDLE_COMMIT").read_text().strip() if (REPO / "BUNDLE_COMMIT").is_file() else ""
    if commit != job["bundle_commit"]:
        return fail(f"bundle commit {commit[:12]} != job {job['bundle_commit'][:12]}")
    if sha(REPO / "experiments/configs/h4_x2_frozen.json") != job["config_sha256"]:
        return fail("canonical config sha differs from the job")
    try:
        data = str(find_data_root())
    except SystemExit as e:
        return fail(str(e))
    summary["data_root"] = data
    env = dict(os.environ, DATA_ROOT=data, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    rc = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], env)
    summary["steps"].append({"tests": rc})
    if rc != 0:
        return fail("repository tests failed")

    kind = job["kind"]
    if kind in ("h4", "dev"):
        rc = run([sys.executable, "experiments/agy2_pipeline.py"], dict(env, PIPELINE_MODE=kind, OUT_DIR=str(out)))
        summary["steps"].append({kind: rc})
    elif kind == "test_shards":
        mans = [Path(p).parent for p in glob.glob(str(INPUT / "**" / "export/manifest.json"), recursive=True)]
        mans = [m for m in mans if json.loads((m / "manifest.json").read_text()).get("mode") == "h4"]
        if len(mans) != 1:
            return fail(f"expected exactly one H4 export, found {len(mans)}")
        sh = job["shards"]
        for i in sh["indices"]:
            rc = run([sys.executable, "experiments/agy2_pipeline.py"],
                     dict(env, PIPELINE_MODE="test", MODELS_DIR=str(mans[0]), SHARD_COUNTRY=sh["country"],
                          SHARD_INDEX=str(i), SHARD_COUNT=str(sh["count"]), OUT_DIR=str(out / "shards")))
            summary["steps"].append({f"shard_{sh['country']}_{i}of{sh['count']}": rc})
            if rc != 0:
                return fail(f"shard {i} failed")
    elif kind == "merge":
        mans = [Path(p).parent for p in glob.glob(str(INPUT / "**" / "export/manifest.json"), recursive=True)]
        mans = [m for m in mans if json.loads((m / "manifest.json").read_text()).get("mode") == "h4"]
        if len(mans) != 1:
            return fail(f"expected exactly one H4 export, found {len(mans)}")
        shard_dirs = sorted({str(Path(p).parent) for p in glob.glob(str(INPUT / "**" / "shard_*_meta.json"), recursive=True)})
        args = [sys.executable, "experiments/agy2_pipeline_merge.py", "--models", str(mans[0]), "--data-root", data,
                "--out", str(out / "output"), "--organiser-validator", str(REPO / "third_party/organiser/validate_submission.py")]
        for d in shard_dirs:
            args += ["--shards", d]
        rc = run(args, env)
        summary["steps"].append({"merge": rc})
    else:
        return fail(f"unknown job kind {kind}")
    summary.update({"exit_code": rc, "wall_min": round((time.time() - t0) / 60, 2), "bundle_commit": commit,
                    "child_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)})
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.close()
    return rc


if __name__ == "__main__":
    sys.exit(main(json.loads(sys.argv[1])))
