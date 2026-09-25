"""AMLC26 submission merge: shard scores -> frozen expected-F0.5 decision -> global one-to-one conflict resolution
-> output/candidate_pairs.tsv + output/matching_results.tsv -> validators.

Exactness / semantics
  * The decision functions (prep/decide/constrain) are imported from agy2_pipeline.py: the exact code that selected the
    rule on H1 OOF and scored H4. Rule + params come from the verified h4 export (decision.json), never re-tuned.
  * Conflict resolution (one S2/S3 record -> at most one S1) is applied over ALL test S1 of a country at once. The
    training ground truth links every target to at most one S1 (7,638,365 links, 0 repeated targets), and candidates
    are country-partitioned, so per-country merging is identical to one global pass.
  * Coverage is fail-closed: the shard S1 lists must partition test_source1 exactly (no missing / duplicated S1).
  * candidate_pairs.tsv = exactly the rows the matcher scored (post-pruner budget), as the organiser README requires.
Memory: one country at a time with integer ids; TSVs are streamed in test_source1 order.
usage: python experiments/agy2_pipeline_merge.py --shards DIR [--shards DIR ...] --models DIR --data-root DIR --out DIR
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import agy2_exp8_store as ST  # noqa: E402
from experiments import agy2_pipeline as PL  # noqa: E402
from src.er.ids import decode_target_id  # noqa: E402
from src.er.loader import load_source_tsv  # noqa: E402


def _sha(p) -> str:
    return PL.sha256_file(p)


def load_test_s1(data_root: Path) -> pd.DataFrame:
    hits = sorted(data_root.rglob("test_source1.tsv"))
    assert len(hits) == 1, f"expected one test_source1.tsv under {data_root}"
    return load_source_tsv(hits[0])[["entity_id", "country"]]


def check_coverage(metas: list, shard_dirs: list, s1: pd.DataFrame) -> dict:
    """Fail closed unless the shard S1 lists partition test_source1 exactly, per country, with one shard count."""
    by_country = {}
    for m in metas:
        by_country.setdefault(m["country"], []).append(m)
    report = {}
    for c, grp in s1.groupby("country", sort=True):
        ms = by_country.get(c, [])
        if not ms:
            raise SystemExit(f"FAIL-CLOSED: no shards for country {c}")
        counts = {m["count"] for m in ms}
        if len(counts) != 1 or sorted(m["index"] for m in ms) != list(range(counts.pop())):
            raise SystemExit(f"FAIL-CLOSED: incomplete/duplicated shard set for {c}: {sorted((m['index'], m['count']) for m in ms)}")
        seen = []
        for m in ms:
            f = [d / f"shard_{m['shard']}_s1.ids" for d in shard_dirs if (d / f"shard_{m['shard']}_s1.ids").is_file()]
            ids = [l for l in f[0].read_text().splitlines() if l]
            if hashlib.sha256("\n".join(ids).encode()).hexdigest() != m["s1_ids_sha256"]:
                raise SystemExit(f"FAIL-CLOSED: shard {m['shard']} S1 list changed")
            seen.extend(ids)
        if len(seen) != len(set(seen)) or set(seen) != set(grp.entity_id):
            raise SystemExit(f"FAIL-CLOSED: shards for {c} do not partition its {len(grp):,} test S1 exactly")
        report[c] = {"s1": int(len(grp)), "shards": len(ms)}
    extra = set(by_country) - set(report)
    if extra:
        raise SystemExit(f"FAIL-CLOSED: shards for countries absent from test_source1: {sorted(extra)}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", action="append", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--organiser-validator", default="")
    a = ap.parse_args(argv)
    t0 = time.time()
    models = Path(a.models)
    cfg = json.loads(PL.CONFIG.read_text())
    for rel, want in cfg["code_sha256"].items():                       # same fail-closed pinning as the pipeline
        if _sha(ROOT / rel) != want:
            raise SystemExit(f"FAIL-CLOSED: {rel} differs from the frozen config")
    PL.verify_export(models, cfg["code_sha256"], _sha(PL.CONFIG))
    dec = json.loads((models / "decision.json").read_text())
    man_sha = _sha(models / "manifest.json")

    shard_dirs = [Path(d) for d in a.shards]
    metas = [json.loads(Path(f).read_text()) for d in shard_dirs for f in sorted(glob.glob(str(d / "shard_*_meta.json")))]
    for m in metas:
        if m["export_manifest_sha256"] != man_sha or m["code_sha256"] != cfg["code_sha256"]:
            raise SystemExit(f"FAIL-CLOSED: shard {m['shard']} was scored with different models/code")
    s1 = load_test_s1(Path(a.data_root))
    cov = check_coverage(metas, shard_dirs, s1)
    n_s1 = len(s1)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cand_lists = {}                                                     # s1_pos -> list of target codes (prune order)
    match_lists = {}
    rep = {"decision": dec, "coverage": cov, "countries": {}}
    for c in sorted(cov):
        parts = []
        for m in metas:
            if m["country"] != c:
                continue
            f = [d / f"shard_{m['shard']}_scores.parquet" for d in shard_dirs if (d / f"shard_{m['shard']}_scores.parquet").is_file()]
            parts.append(ST.read_parquet(f[0]))
        D = pd.concat(parts, ignore_index=True)
        del parts
        if D.duplicated(["s1_pos", "code"]).any():
            raise SystemExit(f"FAIL-CLOSED: duplicated (S1, target) rows in {c}")
        D = D.rename(columns={"s1_pos": "q"})
        D["label"] = False                                              # required column shape only; never used
        D = D.sort_values(["q", "prune_rank"], kind="stable").reset_index(drop=True)
        grp_start = np.r_[0, np.flatnonzero(np.diff(D.q.values)) + 1]
        qs = D.q.values[grp_start]
        bounds = np.r_[grp_start, len(D)]
        codes = D.code.values
        for i, q in enumerate(qs):
            cand_lists[int(q)] = codes[bounds[i]:bounds[i + 1]]
        Dp = PL.prep(D[["q", "code", "p", "label"]])
        sel = PL.decide(Dp, dec["rule"], dec["params"])
        n_pre = int(sel.sum())
        if dec["constrained"]:
            sel = PL.constrain(Dp, sel)                                  # global over all S1 of the country
        chosen = Dp.loc[sel, ["q", "code", "p"]].sort_values(["q", "p"], ascending=[True, False], kind="stable")
        for q, g in chosen.groupby("q", sort=False):
            match_lists[int(q)] = g.code.values
        rep["countries"][c] = {"pairs_scored": int(len(D)), "s1_with_candidates": int(len(qs)),
                               "selected_before_conflicts": n_pre, "selected": int(sel.sum()),
                               "removed_by_conflicts": n_pre - int(sel.sum()),
                               "s1_with_matches": int(chosen.q.nunique())}
        del D, Dp, chosen

    ids = s1.entity_id.tolist()

    def write(path, header, lists):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            for i, e in enumerate(ids):
                lst = lists.get(i)
                f.write(e + "\t" + (",".join(decode_target_id(int(x)) for x in lst) if lst is not None and len(lst) else "") + "\n")

    cp, mr = out / "candidate_pairs.tsv", out / "matching_results.tsv"
    write(cp, "source1_entity_id\tcandidate_entity_ids", cand_lists)
    write(mr, "source1_entity_id\tmatched_entity_ids", match_lists)
    rep["outputs"] = {"candidate_pairs.tsv": {"sha256": _sha(cp), "bytes": cp.stat().st_size},
                      "matching_results.tsv": {"sha256": _sha(mr), "bytes": mr.stat().st_size}}
    rep["s1_total"] = n_s1
    rep["s1_empty_prediction"] = int(n_s1 - len(match_lists))

    from src.er.validator import validate_submission                    # repo validator (format + coverage + subset)
    test_dir = sorted(Path(a.data_root).rglob("test_source1.tsv"))[0].parent
    v = validate_submission(mr, cp, test_dir=test_dir, check_ids=True)
    rep["repo_validator"] = {"valid": bool(v.is_valid), "errors": v.errors[:20], "warnings": v.warnings[:20]}
    if a.organiser_validator:                                            # organiser stdlib validator (the real gate)
        import subprocess
        r = subprocess.run([sys.executable, a.organiser_validator, "--matching", str(mr), "--candidate", str(cp),
                            "--test-dir", str(test_dir)], capture_output=True, text=True)
        rep["organiser_validator"] = {"exit_code": r.returncode, "tail": (r.stdout + r.stderr)[-2000:]}
    rep["runtime_min"] = (time.time() - t0) / 60
    (out / "merge_report.json").write_text(json.dumps(rep, indent=2, default=float))
    ok = rep["repo_validator"]["valid"] and rep.get("organiser_validator", {"exit_code": 0})["exit_code"] == 0
    print(json.dumps({k: rep[k] for k in ("countries", "s1_total", "s1_empty_prediction", "repo_validator")}, default=float))
    if not ok:
        raise SystemExit("VALIDATION FAILED")
    return rep


if __name__ == "__main__":
    main()
