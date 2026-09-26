from experiments.agy2_x2_features import R_FEATURES
"""Freeze the ONE H4 / production configuration: B2 features + expected-F0.5 decision (run before any H4 scoring).

Writes experiments/configs/h4_frozen.json (the canonical path; agy2_pipeline.py reads nothing else outside tests).
Pins: sha256 of every module the pipeline and merge import (fail-closed at run time), ID-list sha256 (H1, H4,
priors H0-H3), candidate B2 (exp9 H1 OOF 0.939414, results sha256), matcher cap 1000 / pruner cap 600, statistics
exclusion H2/H3/H4, and the decision family restricted to expected-F0.5 (thr/topn/gap grids empty; R3 retired).
Variants: B2 (default) or B2nc (drop `country` from BOTH pruner and matcher); budget Top-80 (default) or Top-40.
B2nc / Top-40 are refused unless exp10 results are supplied with the B2 anchor reproduced and the corresponding
pre-registered exp10 decision true (fail-closed; no manual override).
--extra-train N (1..50000) adds N extra TRAINING S1 (salted SHA-256 draw, disjoint from H0-H4) to H1; it is
refused unless exp10 results show the B2 anchor reproduced and learning_curve.more_training_s1_run_justified true.
usage: python experiments/agy2_pipeline_freeze.py --exp9-results PATH [--variant B2|B2nc] [--top-n 80|40]
       [--extra-train N] [--exp10-results PATH] [--out experiments/configs/h4_frozen.json]
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
from experiments.agy2_exp7_h3 import M_FEATS, PR_FEATS  # noqa: E402
from experiments.agy2_exp9_features import NEW_FEATURES  # noqa: E402

PINNED_MODULES = [
    "experiments/agy2_pipeline.py", "experiments/agy2_pipeline_merge.py",
    "experiments/agy2_exp4_fresh_holdout_validation.py", "experiments/agy2_exp5_corrected_holdout.py",
    "experiments/agy2_exp6_integrated.py", "experiments/agy2_exp7_prep_h3.py", "experiments/agy2_exp8_store.py",
    "experiments/agy2_exp9_features.py", "src/er/__init__.py", "src/er/loader.py", "src/er/ids.py", "src/er/validator.py",
]
FILES = {"H0": "exclude_ids/prev_val_s1_holdout_50k.ids", "H1": "exclude_ids/prev_val_s1_50k.ids",
         "H2": "expected/exp5_holdout_s1.ids", "H3": "expected/h3_s1.ids", "H4": "expected/h4_s1.ids"}
EXP9_B2_H1_OOF = 0.9394136402248061
EXTRA_TRAIN_SALT = "amlc26-extra-train-v1"
MAX_EXTRA_TRAIN = 50000


def check_exp10(exp10: dict | None, variant: str, top_n: int, extra_train: int = 0) -> dict:
    """B2nc / Top-40 / extra training S1 need the pre-registered exp10 evidence: anchor reproduced + decision true."""
    need = []
    if extra_train:
        need.append("more_training_s1_run_justified")
    if variant == "B2nc":
        need.append("country_agnostic_for_h4")
    if top_n == 40:
        need.append("top40_budget_for_h4")
    if not need:
        return {}
    if not exp10 or not exp10.get("anchor", {}).get("b2_reproduced"):
        raise SystemExit("FAIL-CLOSED: B2nc/Top-40 need exp10 results with the B2 anchor reproduced")
    for d in need:
        got = (exp10.get("decisions", {}).get("learning_curve", {}).get(d) if d == "more_training_s1_run_justified"
               else exp10.get("decisions", {}).get(d))
        if got is not True:
            raise SystemExit(f"FAIL-CLOSED: exp10 pre-registered decision {d} is not true")
    return {d: True for d in need}


def build_pipeline_config(root: Path, files: dict, exp9: dict, exp9_sha: str, variant="B2", top_n=80,
                          exp10=None, exp10_sha=None, extra_train=0, **kw) -> dict:
    ev = exp9["variants"]["B2"]
    smoke = kw.pop("smoke", False)
    if abs(ev["macro_f05"] - EXP9_B2_H1_OOF) > 1e-12 and not smoke:
        raise SystemExit("exp9 results are not the verified exp9 run (B2 H1 OOF 0.939414)")
    if variant not in ("B2", "B2nc") or top_n not in (80, 40):
        raise SystemExit("FAIL-CLOSED: variant must be B2|B2nc and top_n 80|40")
    if not isinstance(extra_train, int) or extra_train < 0 or (extra_train > MAX_EXTRA_TRAIN and not smoke):
        raise SystemExit(f"FAIL-CLOSED: extra_train must be an int in 0..{MAX_EXTRA_TRAIN}")
    evidence = {} if smoke else check_exp10(exp10, variant, top_n, extra_train)
    cfg = build_h4(root, files, "B2", exp9, exp9_sha, stats_exclude=("H2", "H3", "H4"), **kw)
    cfg["experiment"] = "agy2-er-010-pipeline-h4"
    cfg["code_sha256"] = {m: sha256_file(ROOT / m) for m in PINNED_MODULES}
    cfg["roles"]["test"] = "test_source1 shards scored with the exported fold models; decision frozen from H1 OOF"
    cfg["set_rules"].update({"thr_t": [], "topn_n": [], "topn_t": [], "gap_t": [], "gap_g": [],
                             "selection": "expected-F0.5 only (m in {0, H1 mean unseen links}, singleton gate grid), "
                                          "x {unconstrained, greedy one-to-one}; selected on H1 cross-fitted OOF only"})
    drop = ["country"] if variant == "B2nc" else []
    cfg["drop_features"] = drop
    cfg["pruner"]["features"] = [f for f in PR_FEATS if f not in drop]
    from experiments.agy2_x2_features import R_FEATURES
    cfg["matcher"]["features"] = [f for f in list(M_FEATS) + list(NEW_FEATURES) + R_FEATURES if f not in drop]
    if not smoke:
        cfg["budget"]["top_n"] = top_n
    cfg["candidate"]["name"] = variant
    cfg["candidate"]["exp10_results_sha256"] = exp10_sha
    cfg["candidate"]["exp10_evidence"] = evidence
    if extra_train:
        cfg["extra_dev"] = {"n": extra_train, "salt": EXTRA_TRAIN_SALT}
        cfg["roles"]["dev"] = (f"H1 + {extra_train:,} extra training S1 (salted draw disjoint from H0-H4): "
                               "all fitting and selection")
    cfg["country_encoding"] = {"India": 0.0, "US": 1.0, "other": "NaN (unseen; France performance unknown - no labels)"}
    cfg["text_statistics"] = ("TF-IDF vocabulary/IDF, address-token DF and reverse DF mask fitted per country on train S1 "
                              "excluding H2, H3 and H4 (h4 mode) / on all test S1 of the shard's country (test mode)")
    return cfg


def dev_config(cfg: dict, lanes=(), extra_dev=None) -> dict:
    """Development-only config derived from a frozen one: optional retrieval lanes (dev mode only) and extra training
    S1 ({"n": int, "salt": str}, drawn disjoint from H0-H4). Code pinning and every other frozen field are kept."""
    import copy
    c = copy.deepcopy(cfg)
    c["experiment"] = c["experiment"] + "-dev"
    c["lanes"] = [dict(ln) for ln in lanes]
    lf = [f for ln in lanes for f in (f"flag_lane_{ln['name']}", f"lane_{ln['name']}_score", f"lane_{ln['name']}_rank")]
    drop = c.get("drop_features", [])
    c["pruner"]["features"] = [f for f in PR_FEATS + lf if f not in drop]
    c["matcher"]["features"] = [f for f in PR_FEATS + lf + list(M_FEATS)[len(PR_FEATS):] + list(NEW_FEATURES) + R_FEATURES if f not in drop]
    if extra_dev:
        c["extra_dev"] = dict(extra_dev)
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp9-results", required=True)
    ap.add_argument("--variant", default="B2", choices=["B2", "B2nc"])
    ap.add_argument("--top-n", type=int, default=80, choices=[80, 40])
    ap.add_argument("--exp10-results", default="")
    ap.add_argument("--extra-train", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "configs" / "h4_frozen.json"))
    a = ap.parse_args()
    exp9 = json.loads(Path(a.exp9_results).read_text())
    exp10 = json.loads(Path(a.exp10_results).read_text()) if a.exp10_results else None
    cfg = build_pipeline_config(ROOT, FILES, exp9, sha256_file(a.exp9_results), variant=a.variant, top_n=a.top_n,
                                exp10=exp10, exp10_sha=sha256_file(a.exp10_results) if a.exp10_results else None,
                                extra_train=a.extra_train)
    Path(a.out).write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {a.out}: candidate {a.variant}, Top-{cfg['budget']['top_n']}, extra training S1 {a.extra_train:,}, "
          f"{len(cfg['code_sha256'])} pinned modules")
