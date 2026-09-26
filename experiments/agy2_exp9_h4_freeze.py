"""Freeze the H4 evaluation config for ONE operator-approved exp9 candidate (run only after approval; never run H4 itself).

usage: python experiments/agy2_exp9_h4_freeze.py OUT_JSON --candidate B2 --exp9-results PATH [--root IDS_ROOT]
Pins: code sha of agy2_exp9_h4_eval.py, ID-list sha (H1, H4, priors H0-H3), candidate features and round cap,
statistics exclusion (H2, H3, H4 by default), decision-rule procedure (exp7 grids, DEV-only selection).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import agy2_exp9_features as F9  # noqa: E402
from experiments.agy2_exp7_freeze import build  # noqa: E402
from experiments.agy2_exp7_h3 import M_FEATS  # noqa: E402
from experiments.agy2_exp7_prep_h3 import sha256_file  # noqa: E402


def candidate_features(name: str, ev: dict) -> list:
    if name in ("B0", "B1"):
        return []
    if name == "B2":
        return list(F9.NEW_FEATURES)
    if name == "B3":
        return [f for g, fs in F9.GROUPS.items() if g != ev["dropped_group"] for f in fs]
    raise ValueError(name)


def build_h4(root: Path, files: dict, cand: str, exp9: dict, exp9_sha: str, stats_exclude=("H2", "H3", "H4"), **kw) -> dict:
    ev = exp9["variants"][cand]
    extra = candidate_features(cand, ev)
    cfg = build(root, {k: files[k] for k in ("H0", "H1", "H2", "H3")}, **kw)
    h4n = sum(1 for l in (root / files["H4"]).read_text().splitlines() if l.strip())
    cfg["experiment"] = "agy2-er-009-h4"
    cfg["code_sha256"] = {"experiments/agy2_exp9_h4_eval.py": sha256_file(ROOT / "experiments/agy2_exp9_h4_eval.py")}
    cfg["groups"] = {"dev": cfg["groups"]["dev"],
                     "h4": {"file": files["H4"], "sha256": sha256_file(root / files["H4"]), "n": h4n}}
    cfg["roles"] = {"dev": "H1: all fitting and selection", "h4": "sealed holdout: scored once, evaluation only"}
    cfg["h4"] = {"salt": "amlc26-h4-holdout-v1", "n": h4n,
                 "prior_list_files": {k: files[k] for k in ("H0", "H1", "H2", "H3")}}
    cfg.pop("h3", None); cfg.pop("h2_salt", None)
    cfg["stats_exclude"] = list(stats_exclude)
    from experiments.agy2_x2_features import R_FEATURES
    cfg["matcher"]["features"] = list(M_FEATS) + extra + R_FEATURES
    cfg["matcher"]["max_rounds"] = 1000 if cand != "B0" else 600
    cfg["candidate"] = {"name": cand, "extra_features": extra, "exp9_results_sha256": exp9_sha,
                        "exp9_h1_oof_macro_f05": ev["macro_f05"], "exp9_rule": ev["rule"], "exp9_params": ev["params"],
                        "note": "rule is re-selected on H1 OOF inside the H4 run by the same procedure; params are expected "
                                "to match exp9 (reported, not enforced, because text statistics also exclude H4)"}
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--candidate", required=True); ap.add_argument("--exp9-results", required=True)
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    exp9 = json.loads(Path(a.exp9_results).read_text())
    cfg = build_h4(Path(a.root), {"H0": "exclude_ids/prev_val_s1_holdout_50k.ids", "H1": "exclude_ids/prev_val_s1_50k.ids",
                                  "H2": "expected/exp5_holdout_s1.ids", "H3": "expected/h3_s1.ids", "H4": "expected/h4_s1.ids"},
                   a.candidate, exp9, sha256_file(a.exp9_results))
    Path(a.out).write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {a.out} for candidate {a.candidate}")
