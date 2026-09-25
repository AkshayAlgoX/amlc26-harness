"""H4 holdout integrity (IDs only) + guard that no H4 path uses the retired R3 rule.
Supersedes tests/test_h4_r3.py (its R3 and exp8-superset tests were retired with the R3 H4 freeze)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.er.ids import source_of

ROOT = Path(__file__).resolve().parents[1]


def test_h4_file_and_hash():
    h4_path = ROOT / "expected" / "h4_s1.ids"
    assert h4_path.is_file(), f"Missing {h4_path}"
    
    ids = [line.strip() for line in h4_path.read_text().splitlines() if line.strip()]
    assert len(ids) == 50000, f"Expected 50,000 IDs, got {len(ids)}"
    assert len(set(ids)) == 50000, "Duplicate IDs in H4"
    
    for eid in ids:
        assert eid.startswith("S1-"), f"Invalid S1 prefix: {eid}"
        assert eid[3:].isdigit(), f"Non-numeric ID in S1: {eid}"
        
    actual_sha = hashlib.sha256(h4_path.read_bytes()).hexdigest()
    expected_sha = "237a24bb646ac3dd790c01718bf9f1c5226ecdcffec4cfa65b9d169a37ea3a0a"
    assert actual_sha == expected_sha, f"H4 SHA mismatch: {actual_sha} != {expected_sha}"


def test_h4_disjoint_from_all_prior_holdouts():
    h4_ids = set(ROOT.joinpath("expected", "h4_s1.ids").read_text().splitlines())
    
    priors = {
        "H0": ROOT / "exclude_ids" / "prev_val_s1_holdout_50k.ids",
        "H1": ROOT / "exclude_ids" / "prev_val_s1_50k.ids",
        "H2": ROOT / "expected" / "exp5_holdout_s1.ids",
        "H3": ROOT / "expected" / "h3_s1.ids",
    }
    
    all_sets = {"H4": h4_ids}
    for name, path in priors.items():
        assert path.is_file(), f"Missing prior list {path}"
        s = set(path.read_text().splitlines())
        assert len(s) == 50000, f"Expected 50k in {name}"
        overlap = len(h4_ids & s)
        assert overlap == 0, f"H4 overlaps with {name} by {overlap} IDs"
        all_sets[name] = s

    # Total union across all 5 disjoint holdouts must be exactly 250,000
    total_union = set.union(*all_sets.values())
    assert len(total_union) == 250000, f"Union count {len(total_union)} != 250,000"


@pytest.mark.parametrize("eid,expected_src", [
    ("S2-166376419", "S2"),
    ("S2-999999999", "S2"),
    ("S3-1", "S3"),
    ("S3-999999999", "S3"),
    ("S1-252242682", "S1"),
    ("S1-1", "S1"),
])
def test_prefix_source_classification(eid, expected_src):
    assert source_of(eid) == expected_src


def test_no_h4_path_uses_r3():
    for rel in ("experiments/agy2_pipeline.py", "experiments/agy2_pipeline_merge.py", "experiments/agy2_pipeline_freeze.py"):
        src = (ROOT / rel).read_text().lower()
        assert "dynamic_margin" not in src and '"r3"' not in src, rel
    cfg = ROOT / "experiments" / "configs" / "h4_frozen.json"
    if cfg.is_file():
        c = json.loads(cfg.read_text())
        assert c["candidate"]["name"] in ("B2", "B2nc")
        if c["candidate"]["name"] == "B2nc" or c["budget"]["top_n"] == 40:
            assert c["candidate"]["exp10_evidence"], "B2nc / Top-40 require pre-registered exp10 evidence"
        assert c["drop_features"] == (["country"] if c["candidate"]["name"] == "B2nc" else [])
        for stage in ("pruner", "matcher"):
            assert ("country" in c[stage]["features"]) == (c["candidate"]["name"] == "B2")
        assert not c["set_rules"]["thr_t"] and not c["set_rules"]["topn_n"] and not c["set_rules"]["gap_t"]
        assert c["set_rules"]["expF_s"], "expected-F0.5 must be the only decision family"
        assert not any("r3" in k for k in c["set_rules"])
