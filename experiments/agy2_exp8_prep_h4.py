"""AMLC26 exp8 PREP: derivation and verification of the H4 holdout (IDs only).

RETIRED (2026-09-25): the R3 decision-rule freeze and the exp8 retrieval-superset freeze that this script used to
record. R3's supporting evaluation was invalid (scored on a negative-subsampled training frame) and R3 lost to the
expected-F0.5 rule on H1 OOF (-0.0037, 5/5 folds). The only H4 configuration is experiments/configs/h4_frozen.json
(B2 + expected-F0.5, experiments/agy2_pipeline.py). Items 6-8 below are historical.

Objective:
  1. H4 S1 ID list derivation and SHA-256 verification (50,000 IDs, salt 'amlc26-h4-holdout-v1').
  2. Proof of strict disjointness: H4 disjoint from H0, H1, H2, and H3 (all 0 overlap).
  3. No labels or ground truth used for selecting H4 parameters.
  4. Integration with EXP8 artifact graph (C0..C6).
  5. Correct production parser and prefix-only source ID handling (src/er/ids.py).
  6. Frozen retrieval superset settings:
     K_FWD=300, K_REV=10, CAP=1000, exact shard merge.
  7. Frozen R3 dynamic margin parameters:
     floor=0.5, margin=0.2.
  8. Preparation for immediate scoring once AGY1 R3 code is merged.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import agy2_exp8_features as F
from experiments import agy2_exp8_store as S
from src.er.ids import code_source, decode_target_id, encode_target_id, is_target_id, source_of
from src.er.loader import load_ground_truth_df, load_source_tsv

ENV = os.environ.get
DATA_ROOT = Path(ENV("DATA_ROOT", "/home/akshaya/Downloads/student_resource/student_resource/dataset/train"))
EXPECTED_DIR = ROOT / "expected"
EXCLUDE_DIR = ROOT / "exclude_ids"
OUT_DIR = Path(ENV("OUT_DIR", str(ROOT / "output" / "h4_prep")))

PRIOR_LISTS = {
    "H0": EXCLUDE_DIR / "prev_val_s1_holdout_50k.ids",
    "H1": EXCLUDE_DIR / "prev_val_s1_50k.ids",
    "H2": EXPECTED_DIR / "exp5_holdout_s1.ids",
    "H3": EXPECTED_DIR / "h3_s1.ids",
}

H4_SALT = ENV("H4_SALT", "amlc26-h4-holdout-v1")
HOLDOUT_N = 50000
EXPECTED_H4_SHA = "237a24bb646ac3dd790c01718bf9f1c5226ecdcffec4cfa65b9d169a37ea3a0a"

H4_CONFIG = "experiments/configs/h4_frozen.json"      # the ONLY H4 configuration (B2 + expected-F0.5)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def salted_holdout(all_ids: list[str], excluded: set[str], n: int, salt: str) -> list[str]:
    """Deterministic SHA-256 hash holdout: independent of input order, disjoint from excluded."""
    pool = sorted({e for e in all_ids if e not in excluded},
                  key=lambda e: hashlib.sha256(f"{salt}|{e}".encode()).digest())
    return pool[:n]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"h4": {}, "h4_config": H4_CONFIG, "retired": ["r3_decision_freeze", "exp8_retrieval_superset_freeze"]}

    # 1. Verify prior holdout files
    prior_sets = {}
    prior_meta = {}
    excluded_union = set()
    for name, path in PRIOR_LISTS.items():
        assert path.is_file(), f"Missing prior holdout list: {path}"
        ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        assert len(ids) == len(set(ids)), f"Duplicate IDs in {name}"
        assert len(ids) == 50000, f"Expected 50k in {name}, found {len(ids)}"
        prior_sets[name] = set(ids)
        excluded_union |= set(ids)
        prior_meta[name] = {"path": str(path.relative_to(ROOT)), "n": len(ids), "sha256": sha256_file(path)}

    assert len(excluded_union) == 200000, f"Priors not pairwise disjoint: union {len(excluded_union)} != 200,000"

    # 2. Check S1 population and generate/verify H4
    s1_file = DATA_ROOT / "train_source1.tsv"
    if s1_file.is_file():
        s1_df = load_source_tsv(s1_file)
        all_s1 = s1_df["entity_id"].tolist()
        assert len(all_s1) == 2206821, f"Expected 2,206,821 S1 rows, got {len(all_s1)}"
        assert len(set(all_s1)) == len(all_s1), "Duplicate S1 IDs in train_source1.tsv"
        h4_ids = salted_holdout(all_s1, excluded_union, HOLDOUT_N, H4_SALT)
        # Order independence
        assert salted_holdout(list(reversed(all_s1)), excluded_union, HOLDOUT_N, H4_SALT) == h4_ids
    else:
        # Fallback to local expected/h4_s1.ids if raw data path is omitted
        h4_path = EXPECTED_DIR / "h4_s1.ids"
        assert h4_path.is_file(), f"H4 file not found at {h4_path}"
        h4_ids = [l.strip() for l in h4_path.read_text().splitlines() if l.strip()]

    assert len(h4_ids) == HOLDOUT_N
    assert len(set(h4_ids)) == HOLDOUT_N

    # 3. Disjointness check
    overlaps = {name: len(set(h4_ids) & id_set) for name, id_set in prior_sets.items()}
    assert all(v == 0 for v in overlaps.values()), f"H4 overlaps with prior holdouts: {overlaps}"

    # Write H4 file
    h4_out = EXPECTED_DIR / "h4_s1.ids"
    h4_out.write_text("\n".join(h4_ids) + "\n")
    actual_sha = sha256_file(h4_out)
    assert actual_sha == EXPECTED_H4_SHA, f"H4 hash mismatch: {actual_sha} != {EXPECTED_H4_SHA}"

    report["h4"] = {
        "name": "h4",
        "file": str(h4_out.relative_to(ROOT)),
        "n": len(h4_ids),
        "sha256": actual_sha,
        "salt": H4_SALT,
        "overlaps_with_priors": overlaps,
        "priors": prior_meta,
        "all_priors_disjoint": True,
        "total_excluded_before_h4": len(excluded_union),
    }

    # 4. Verify Prefix-Only Source ID invariant
    test_ids = [("S2-166376419", "S2"), ("S2-999999999", "S2"), ("S3-1", "S3"), ("S3-999999999", "S3")]
    for eid, expected_src in test_ids:
        assert source_of(eid) == expected_src
        code = encode_target_id(eid)
        assert decode_target_id(code) == eid
        assert code_source(code) == expected_src

    report["prefix_id_handling"] = {
        "status": "PASS",
        "verified_examples": [eid for eid, _ in test_ids],
        "heuristic_free": True
    }

    # 5. Output report
    rep_path = OUT_DIR / "prep_report_h4.json"
    rep_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"H4 Preparation COMPLETE.")
    print(f"H4 SHA-256: {actual_sha}")
    print(f"Report written to: {rep_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
