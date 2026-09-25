"""AMLC26 exp8 — SEALED holdout / test scorer (one expensive run per frozen decision).

Everything is fixed before this runs, in a frozen exp8 config (JSON):
  {"code_sha256": {<every exp8 file>: sha}, "store": <dev store root>, "c5": "c5_matcher/<key>",
   "c6": "c6_decide/<key>", "holdout": {"name": "h4", "file": <ids file>, "sha256": ...}, "sealed_store": <dir>,
   "dataset": "train" | "test"}
The chain C2 -> C3 -> C4 -> C5 -> C6 is resolved from the dev store (all verified), then:
  1. code-hash verification of every exp8 file against the config,
  2. holdout checks: ids sha, disjoint from DEV/H2, inside the statistics-exclusion set of the dev snapshot (so its
     texts never shaped IDF/DF), and not already scored (ledger in the sealed store; --allow-reuse to override),
  3. retrieval for the holdout queries with the SAME builder and parameters into a SEPARATE sealed store; the
     rebuilt statistics must hash-equal the dev C1 statistics,
  4. each holdout S1 is scored by the single chain pruner_k -> matcher_k of its own fold (saved boosters),
  5. the frozen C6 rule / gate / conflict setting is applied once; train holdouts are scored (per-S1 macro F0.5,
     cross-checked with src/er/metric.macro_f05); test writes predictions only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import agy2_exp8_build_snapshot as B  # noqa: E402
from experiments import agy2_exp8_decide as DC  # noqa: E402
from experiments import agy2_exp8_features as F  # noqa: E402
from experiments import agy2_exp8_matcher as MT  # noqa: E402
from experiments import agy2_exp8_store as S  # noqa: E402
from src.er.ids import decode_target_id  # noqa: E402

EXP8_FILES = ["experiments/agy2_exp8_store.py", "experiments/agy2_exp8_features.py",
              "experiments/agy2_exp8_build_snapshot.py", "experiments/agy2_exp8_pruner.py",
              "experiments/agy2_exp8_matcher.py", "experiments/agy2_exp8_decide.py", "experiments/agy2_exp8_final.py"]
STAT_FILES = ("vocab_", "idf_", "revkeep_", "addr_df_")


def freeze(store: S.Store, c6_ref: str, holdout_name: str, holdout_file, sealed_store, dataset="train",
           sealed_files=None) -> dict:
    """Write-once config for a sealed run (the scorer refuses to run if any file hash changes afterwards).
    sealed_files: id lists forming the dev store's sealed set (default: frozen H3 + EXP8_SEALED_LISTS)."""
    c6 = store.open(*c6_ref.split("/"))
    cfg = {"code_sha256": S.code_sha(*EXP8_FILES), "store": str(store.root), "c6": c6.ref, "c5": c6.man["parents"]["c5"],
           "holdout": {"name": holdout_name, "file": str(holdout_file), "sha256": S.sha256_file(holdout_file)},
           "sealed_store": str(sealed_store), "dataset": dataset}
    if sealed_files:
        cfg["sealed_files"] = {str(f): S.sha256_file(f) for f in sealed_files}
    return cfg


def score_sealed(cfg: dict, data_root, workers=None, allow_reuse=False, check_code=True) -> dict:
    t0 = time.time()
    if check_code:
        for f, sha in cfg["code_sha256"].items():
            if S.sha256_file(ROOT / f) != sha:
                raise S.StoreError(f"{f} differs from the frozen exp8 config")
    forbidden = None
    if cfg.get("sealed_files"):
        forbidden = set()
        for f, sha in cfg["sealed_files"].items():
            if S.sha256_file(f) != sha:
                raise S.StoreError(f"sealed list {f} changed")
            forbidden |= set(S.read_ids(f))
    dev = S.Store(cfg["store"], forbidden=forbidden)
    c6 = dev.open(*cfg["c6"].split("/"))
    c5 = dev.open(*cfg["c5"].split("/"))
    if c6.man["parents"]["c5"] != c5.ref:
        raise S.StoreError("C6 decision was not made on this C5")
    c4 = c5.parent("c4"); c3 = c4.parent("c3"); c2 = c3.source()
    if c2.kind != "c2_retrieval":
        raise S.StoreError("sealed scoring needs a real c2_retrieval snapshot (not the v0 sandbox)")
    c1 = c2.parent("c1")
    decision = json.loads(c6.file("decision.json").read_text())["selected"]

    H = cfg["holdout"]
    if H["name"] in S.DEV_GROUPS:
        raise S.StoreError("holdout name collides with a development group")
    if S.sha256_file(H["file"]) != H["sha256"]:
        raise S.StoreError("holdout id list differs from the frozen config")
    hold = S.read_ids(H["file"])
    devq = c2.table("queries", columns=["s1_id", "group"])
    if set(hold) & set(devq.s1_id):
        raise S.StoreError("holdout overlaps DEV/H2 queries")
    sealed_root = Path(cfg["sealed_store"])
    ledger = sealed_root / "ledger.jsonl"
    if ledger.exists() and not allow_reuse:
        used = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
        if any(u["holdout_sha256"] == H["sha256"] for u in used):
            raise S.StoreError(f"holdout {H['name']} was already scored — refusing a second look (--allow-reuse)")
    groups_dev = {"dev": devq.s1_id[devq.group == "dev"].tolist(), "h2": devq.s1_id[devq.group == "h2"].tolist()}
    stat_exclude = set(groups_dev["h2"]) | set(dev.forbidden)
    if B.ids_sha(stat_exclude) != c1.man["params"]["stat_exclude_sha"]:
        raise S.StoreError("cannot reproduce the dev statistics-exclusion set")
    if cfg["dataset"] == "train" and not set(hold) <= stat_exclude:
        raise S.StoreError("holdout S1 texts were used to fit the dev statistics — not a sealed holdout")

    # ---------------- retrieval for the holdout into the sealed store (same builder, same parameters)
    sealed = S.Store(sealed_root, forbidden=set(dev.forbidden) - set(hold), allowed_groups=(H["name"],))
    params = {k: v for k, v in c1.man["params"].items() if k in ("max_df", "min_df", "rev_df_cap")}
    params.update(fwd_min_sim=c2.man["params"]["fwd_min_sim"], chunk=c2.man["params"]["chunk"],
                  superset=c2.man["params"]["superset"])
    excl = stat_exclude if cfg["dataset"] == "train" else set()
    refs = B.build_snapshot(data_root, sealed, {H["name"]: hold}, excl, params, workers, dataset=cfg["dataset"])
    s2 = sealed.open(*refs["c2"].split("/"))
    s1 = sealed.open(*refs["c1"].split("/"))
    if cfg["dataset"] == "train":
        for fn, meta in c1.man["files"].items():
            if fn.startswith(STAT_FILES) and s1.man["files"].get(fn, {}).get("sha256") != meta["sha256"]:
                raise S.StoreError(f"sealed statistics differ from dev C1 ({fn})")

    # ---------------- frozen chain, each S1 through its own fold's pruner_k -> matcher_k
    nfold = S.fold_cfg()[1]
    qt = s2.table("queries")
    D = F.apply_view(s2.table("pool"), c3.man["params"]["view"], s2.man["params"]["superset"])
    fold = qt.fold.values.astype(np.int8)[D.q.values]
    p3 = c3.man["params"]
    D["prune_p"] = F.predict_by_fold(c3.dir, "pruner", D, p3["features"], fold, nfold, p3["neg_rate"])
    from experiments.agy2_exp8_pruner import prune_rank
    D["prune_rank"] = prune_rank(D.q.values, D.code.values, D.prune_p.values)
    p4 = c4.man["params"]
    M = D[(D.prune_rank.values <= p4["top_n"]) & (D.prune_p.values >= p4["floor"])].reset_index(drop=True)
    if p4["text_features"]:
        qx, tx = s2.table("q_text"), s2.table("t_text")
        B._G.update(M=M[["q", "t_idx", "exact_name", "exact_addr"]], qname=qx.name.tolist(), qaddr=qx.addr.tolist(),
                    tname=tx.name.tolist(), taddr=tx.addr.tolist())
        parts = B.pmap(MT._text_chunk, B.ranges(len(M), (workers or os.cpu_count()) * 4), workers or os.cpu_count())
        B._G.clear()
        for col in parts[0]:
            M[col] = np.concatenate([p[col] for p in parts])
    p5 = c5.man["params"]
    M["p"] = F.predict_by_fold(c5.dir, "matcher", M, p5["features"], qt.fold.values.astype(np.int8)[M.q.values],
                               nfold, p5["neg_rate"])
    G = DC.Group(M[["q", "code", "p", "label"]], np.arange(len(qt)), np.maximum(qt.truth_n.values, 0))
    sel = G.decide(decision["rule"], decision["params"])
    sel = G.constrain(sel) if decision["conflict"] == "greedy" else sel
    out = {"holdout": H, "dataset": cfg["dataset"], "decision": decision, "pairs_scored": int(len(M)),
           "predicted_pairs": int(sel.sum()), "sealed_c2": s2.ref, "runtime_min": None}
    pred = {}
    for q_, c_ in zip(G.q[sel], G.code[sel]):
        pred.setdefault(qt.s1_id.values[q_], []).append(decode_target_id(int(c_)))
    if cfg["dataset"] == "train":
        f = DC.per_s1_f05(G.q, G.label, sel, G.truth_n, G.nq)
        out["macro_f05"] = float(f.mean())
        out["singleton_f05"] = float(f[G.truth_n == 0].mean()) if (G.truth_n == 0).any() else None
        from src.er.metric import macro_f05
        tr = s2.table("truth")
        truth = {}
        for q_, c_ in zip(tr.q.values, tr.code.values):
            truth.setdefault(qt.s1_id.values[q_], set()).add(decode_target_id(int(c_)))
        out["macro_f05_official_check"] = macro_f05({e: truth.get(e, set()) for e in qt.s1_id},
                                                    {e: set(pred.get(e, [])) for e in qt.s1_id}, s1_ids=qt.s1_id.tolist())
        assert abs(out["macro_f05_official_check"] - out["macro_f05"]) < 1e-9, "metric mismatch"
    else:
        rows = [(e, ",".join(pred.get(e, []))) for e in qt.s1_id]
        pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
            sealed_root / f"predictions_{H['name']}.tsv", sep="\t", index=False)
    out["runtime_min"] = round((time.time() - t0) / 60, 2)
    (sealed_root / f"result_{H['name']}.json").write_text(json.dumps(out, indent=2, default=float))
    with open(ledger, "a") as fh:
        fh.write(json.dumps({"holdout": H["name"], "holdout_sha256": H["sha256"], "c6": c6.ref,
                             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/kaggle/input/datasets/akshayalgox/amlc26-2026-official-data-v2"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--allow-reuse", action="store_true")
    a = ap.parse_args()
    out = score_sealed(json.loads(Path(a.config).read_text()), a.data_root, a.workers, a.allow_reuse)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
