"""Unit tests for H4 holdout integrity and R3 dynamic margin decision rule."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import agy2_exp8_features as F
from experiments import agy2_exp8_store as S
from experiments.agy2_exp8_decide import Group
from src.er.ids import code_source, decode_target_id, encode_target_id, is_target_id, source_of

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


def test_frozen_retrieval_settings():
    assert F.SUPERSET["k_fwd"] == 300
    assert F.SUPERSET["rev_k"] == 10
    assert F.SUPERSET["cap"] == 1000


def test_r3_dynamic_margin_logic():
    # Construct a Group test case mirroring AGY1 test cases
    # S1-1: 3 candidates [0.95, 0.94, 0.70] -> floor=0.5, margin=0.2 -> thresh=max(0.5, 0.75)=0.75 -> keeps 2
    # S1-2: 2 candidates [0.40, 0.30] -> top=0.40 < 0.50 -> keeps 0
    # S1-3: 1 candidate [0.80] -> top=0.80 -> thresh=0.60 -> keeps 1
    df = pd.DataFrame({
        "q": [0, 0, 0, 1, 1, 2],
        "code": [10, 11, 12, 13, 14, 15],
        "p": [0.95, 0.94, 0.70, 0.40, 0.30, 0.80],
        "label": [1, 1, 0, 0, 0, 1],
    })
    q_ids = np.array([0, 1, 2])
    truth_n = np.array([2, 0, 1])

    grp = Group(df, q_ids, truth_n)
    
    sel = grp.decide("r3", {"floor": 0.5, "margin": 0.2})
    
    # Check selection:
    # Sorted order per group:
    # q=0: 0.95 (True), 0.94 (True), 0.70 (False)
    # q=1: 0.40 (False), 0.30 (False)
    # q=2: 0.80 (True)
    expected_sel = np.array([True, True, False, False, False, True])
    assert np.array_equal(sel, expected_sel)
    
    # Score calculation
    f05, _ = grp.score(sel)
    assert f05 == 1.0  # All 3 S1s achieve perfect F0.5
