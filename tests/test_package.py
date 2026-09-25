"""Submission package allowlist: required layout present, development / label / holdout files impossible."""
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import build_submission_package as B  # noqa: E402


@pytest.fixture
def inputs(tmp_path):
    out = tmp_path / "output"; out.mkdir()
    (out / "matching_results.tsv").write_text("source1_entity_id\tmatched_entity_ids\nS1-1\t\n")
    (out / "candidate_pairs.tsv").write_text("source1_entity_id\tcandidate_entity_ids\nS1-1\t\n")
    exp = tmp_path / "export"; exp.mkdir()
    for f in ("pruner_fold0.txt", "matcher_fold0.txt", "decision.json", "manifest.json"):
        (exp / f).write_text("x")
    doc = tmp_path / "Documentation_template.md"; doc.write_text("# doc")
    return out, exp, doc


def test_layout_and_exclusions(inputs, tmp_path):
    if not (ROOT / "experiments/configs/h4_frozen.json").is_file():
        pytest.skip("canonical h4_frozen.json not generated yet")
    out, exp, doc = inputs
    info = B.build("team", out, exp, doc, tmp_path / "team_submission.zip")
    names = zipfile.ZipFile(info["zip"]).namelist()
    for req in ("output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
                "code/business_entity_resolution/README.md", "code/business_entity_resolution/requirements.txt",
                "code/business_entity_resolution/src/experiments/agy2_pipeline.py",
                "code/business_entity_resolution/src/experiments/configs/h4_frozen.json",
                "code/business_entity_resolution/src/export/decision.json"):
        assert req in names, req
    for n in names:
        assert not n.endswith(".ids"), n
        for bad in ("/data/", "/expected/", "/exclude_ids/", "/scratch/", "romanization", "baseline_misses", "patch"):
            assert bad not in n, n


def test_forbidden_items_are_rejected(inputs, monkeypatch):
    out, exp, doc = inputs
    monkeypatch.setattr(B, "CODE_ALLOWLIST", B.CODE_ALLOWLIST + ["expected/h4_s1.ids"])
    with pytest.raises(SystemExit):
        B.plan("team", out, exp, doc)
