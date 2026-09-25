"""Freeze the ONE H4 / production configuration: B2 features + expected-F0.5 decision (run before any H4 scoring).

Writes experiments/configs/h4_frozen.json (the canonical path; agy2_pipeline.py reads nothing else outside tests).
Pins: sha256 of every module the pipeline and merge import (fail-closed at run time), ID-list sha256 (H1, H4,
priors H0-H3), candidate B2 (exp9 H1 OOF 0.939414, results sha256), matcher cap 1000 / pruner cap 600, statistics
exclusion H2/H3/H4, and the decision family restricted to expected-F0.5 (thr/topn/gap grids empty; R3 retired).
usage: python experiments/agy2_pipeline_freeze.py --exp9-results PATH [--out experiments/configs/h4_frozen.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.agy2_exp7_prep_h3 import sha256_file  # noqa: E402
from experiments.agy2_exp9_h4_freeze import build_h4  # noqa: E402

PINNED_MODULES = [
    "experiments/agy2_pipeline.py", "experiments/agy2_pipeline_merge.py",
    "experiments/agy2_exp4_fresh_holdout_validation.py", "experiments/agy2_exp5_corrected_holdout.py",
    "experiments/agy2_exp6_integrated.py", "experiments/agy2_exp7_prep_h3.py", "experiments/agy2_exp8_store.py",
    "experiments/agy2_exp9_features.py", "src/er/__init__.py", "src/er/loader.py", "src/er/ids.py", "src/er/validator.py",
]
FILES = {"H0": "exclude_ids/prev_val_s1_holdout_50k.ids", "H1": "exclude_ids/prev_val_s1_50k.ids",
         "H2": "expected/exp5_holdout_s1.ids", "H3": "expected/h3_s1.ids", "H4": "expected/h4_s1.ids"}
EXP9_B2_H1_OOF = 0.9394136402248061


def build_pipeline_config(root: Path, files: dict, exp9: dict, exp9_sha: str, **kw) -> dict:
    ev = exp9["variants"]["B2"]
    if abs(ev["macro_f05"] - EXP9_B2_H1_OOF) > 1e-12 and not kw.pop("smoke", False):
        raise SystemExit("exp9 results are not the verified exp9 run (B2 H1 OOF 0.939414)")
    kw.pop("smoke", None)
    cfg = build_h4(root, files, "B2", exp9, exp9_sha, stats_exclude=("H2", "H3", "H4"), **kw)
    cfg["experiment"] = "agy2-er-010-pipeline-h4"
    cfg["code_sha256"] = {m: sha256_file(ROOT / m) for m in PINNED_MODULES}
    cfg["roles"]["test"] = "test_source1 shards scored with the exported fold models; decision frozen from H1 OOF"
    cfg["set_rules"].update({"thr_t": [], "topn_n": [], "topn_t": [], "gap_t": [], "gap_g": [],
                             "selection": "expected-F0.5 only (m in {0, H1 mean unseen links}, singleton gate grid), "
                                          "x {unconstrained, greedy one-to-one}; selected on H1 cross-fitted OOF only"})
    cfg["country_encoding"] = {"India": 0.0, "US": 1.0, "other": "NaN (unseen; France performance unknown - no labels)"}
    cfg["text_statistics"] = ("TF-IDF vocabulary/IDF, address-token DF and reverse DF mask fitted per country on train S1 "
                              "excluding H2, H3 and H4 (h4 mode) / on all test S1 of the shard's country (test mode)")
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp9-results", required=True)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "configs" / "h4_frozen.json"))
    a = ap.parse_args()
    exp9 = json.loads(Path(a.exp9_results).read_text())
    cfg = build_pipeline_config(ROOT, FILES, exp9, sha256_file(a.exp9_results))
    Path(a.out).write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {a.out}: candidate B2, {len(cfg['code_sha256'])} pinned modules")
