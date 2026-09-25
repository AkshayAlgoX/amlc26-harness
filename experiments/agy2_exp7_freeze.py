"""Write the frozen, machine-readable exp7 configuration (run BEFORE any H3 scoring).

Every value that can change a result is fixed here: groups (+ SHA-256 of each ID list), H3 derivation, text
statistics, retrieval channels, fold assignment, pruner/matcher hyper-parameters and feature lists, the single
pre-registered budget, and the set-rule grids. exp7 refuses to run if its own code or an ID list differs.

usage: python experiments/agy2_exp7_freeze.py OUT_JSON [--root ROOT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.agy2_exp7_h3 import M_FEATS, PR_FEATS  # noqa: E402
from experiments.agy2_exp7_prep_h3 import sha256_file  # noqa: E402

LGB = {"learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.9,
       "bagging_fraction": 0.8, "lambda_l2": 1.0, "max_bin": 127, "max_rounds": 600, "early_stopping": 50}


def build(root: Path, files: dict, retrieval: dict | None = None, budget: dict | None = None, h3_n: int = 50000) -> dict:
    def ids_n(p):
        return sum(1 for l in (root / p).read_text().splitlines() if l.strip())
    groups = {g: {"file": f, "sha256": sha256_file(root / f), "n": ids_n(f)} for g, f in
              (("dev", files["H1"]), ("h2", files["H2"]), ("h3", files["H3"]))}
    return {
        "experiment": "agy2-er-007-h3",
        "code_sha256": {"experiments/agy2_exp7_h3.py": sha256_file(ROOT / "experiments/agy2_exp7_h3.py")},
        "seed": 42,
        "groups": groups,
        "roles": {"dev": "H1 previous 50k: ALL fitting and selection (cross-fitted OOF)",
                  "h2": "exp5 holdout: evaluation only (comparison; used for retrieval design before exp7)",
                  "h3": "new untouched holdout: evaluation only"},
        "h2_salt": "amlc26-exp5-holdout-v1",
        "h3": {"salt": "amlc26-h3-holdout-v1", "n": h3_n, "method": "sort train S1 ids by sha256(salt|id); take first n "
               "after excluding H0 u H1 u H2", "prior_list_files": {"H0": files["H0"], "H1": files["H1"], "H2": files["H2"]}},
        "text_statistics": "TF-IDF vocabulary/IDF, address-token DF and reverse DF mask fitted on train S1 excluding H2 and H3",
        "parser": "src/er/loader.py load_source_tsv / load_ground_truth_df (single canonical reader)",
        "retrieval": retrieval or {"pool_channels": ["A", "B", "F200", "R5", "B2C", "B3C", "B4C"], "max_df": 0.05,
                                   "rev_df_cap": 2000, "k_fwd": 200, "chunk": 20000, "cap": 200, "fwd_min_sim": 0.05,
                                   "canonical_view": False},
        "folds": {"salt": "amlc26-exp7-fold-v1", "nfold": 5, "shared_by": ["pruner", "matcher"],
                  "evaluation_scoring": "each eval S1 scored by the single pruner_k -> matcher_k chain of its own fold"},
        "pruner": {"features": PR_FEATS, **LGB},
        "budget": budget or {"top_n": 80, "floor": 0.0, "justification":
                             "pre-registered from exp6: max train-OOF oracle with mean<=80 and p99<=250 on H1 selected top80; "
                             "H2 showed top40/80/120 matcher F0.5 within 0.0004; no budget is chosen on H2 or H3"},
        "matcher": {"features": M_FEATS, **LGB},
        "calibration": "none (set rules on raw cross-fitted scores)",
        "set_rules": {"selection": "max DEV cross-fitted OOF macro F0.5 over rule x params x {unconstrained, greedy}",
                      "thr_t": [round(0.20 + 0.05 * i, 2) for i in range(14)],
                      "topn_n": [1, 2, 3, 4, 5, 6, 8, 11], "topn_t": [0.2, 0.3, 0.4, 0.5, 0.6],
                      "gap_t": [0.2, 0.3, 0.4, 0.5], "gap_g": [0.3, 0.5, 0.7, 0.85],
                      "expF_s": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                      "expF_m": "{0, DEV mean unseen links per S1}"},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    root = Path(a.root)
    cfg = build(root, {"H0": "exclude_ids/prev_val_s1_holdout_50k.ids", "H1": "exclude_ids/prev_val_s1_50k.ids",
                       "H2": "expected/exp5_holdout_s1.ids", "H3": "expected/h3_s1.ids"})
    Path(a.out).write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {a.out}")
