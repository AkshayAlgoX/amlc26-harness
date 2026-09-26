"""Kernel stub + data-root discovery work under any /kaggle/input mount layout and fail closed otherwise."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "kaggle"))
import kaggle_job as KJ  # noqa: E402
import make_kernels as MK  # noqa: E402

COMMIT = "a" * 40


def _bundle(d: Path, commit=COMMIT):
    (d / "scripts" / "kaggle").mkdir(parents=True)
    (d / "BUNDLE_COMMIT").write_text(commit + "\n")
    (d / "scripts" / "kaggle" / "kaggle_job.py").write_text(
        "import json, pathlib, sys, os\n"
        "pathlib.Path(os.environ['AMLC_WORK'], 'marker.json').write_text(json.dumps({'cwd': os.getcwd(), 'job': json.loads(sys.argv[1])}))\n")


def _stub(tmp: Path) -> Path:
    MK.kernel(tmp, "k", "amlc26-x2-h4", {"bundle_commit": COMMIT, "config_sha256": "x", "kind": "h4"}, "o/s", [])
    return tmp / "k" / "run.py"


def _run(stub, inp, work):
    e = dict(os.environ, AMLC_INPUT=str(inp), AMLC_WORK=str(work))
    return subprocess.run([sys.executable, str(stub)], env=e, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("layout", ["flat", "nested", "with_kernel_output_copy"])
def test_stub_finds_bundle_by_content(tmp_path, layout):
    inp, work = tmp_path / "input", tmp_path / "work"
    work.mkdir()
    ds = inp / ("amlc26-agy2-px" if layout != "nested" else "datasets/akshayalgox/amlc26-agy2-px")
    _bundle(ds)
    if layout == "with_kernel_output_copy":
        _bundle(inp / "amlc26-x2-h4" / "amlc26-agy2")          # repo copy inside a kernel output: not preferred
    (inp / "other").mkdir(parents=True)
    _bundle(inp / "other" / "stale", commit="b" * 40)        # different commit: never selected
    r = _run(_stub(tmp_path), inp, work)
    assert r.returncode == 0, r.stdout + r.stderr
    m = json.loads((work / "marker.json").read_text())
    assert m["job"]["bundle_commit"] == COMMIT and m["cwd"] == str(work / "amlc26-agy2")
    assert f"bundle: {ds}" in r.stdout


def test_stub_fails_closed_without_matching_bundle(tmp_path):
    inp, work = tmp_path / "input", tmp_path / "work"
    work.mkdir()
    _bundle(inp / "x", commit="b" * 40)
    r = _run(_stub(tmp_path), inp, work)
    assert r.returncode != 0 and "FAIL-CLOSED" in r.stdout + r.stderr and "input:" in r.stdout
    assert not (work / "marker.json").exists()


@pytest.mark.parametrize("layout", ["nested_split", "flat"])
def test_find_data_root_any_layout(tmp_path, layout):
    base = tmp_path / ("datasets/akshayalgox/amlc26-2026-official-data-v2" if layout == "nested_split" else "official")
    tr, te = (base / "train", base / "test") if layout == "nested_split" else (base, base)
    tr.mkdir(parents=True, exist_ok=True); te.mkdir(parents=True, exist_ok=True)
    for f in ("train_source1.tsv", "train_ground_truth.tsv"):
        (tr / f).write_text("x\n")
    (te / "test_source1.tsv").write_text("x\n")
    assert KJ.find_data_root(tmp_path) == base


def test_find_data_root_fails_closed_on_ambiguity(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "train_source1.tsv").write_text("x\n")
    with pytest.raises(SystemExit):
        KJ.find_data_root(tmp_path)
