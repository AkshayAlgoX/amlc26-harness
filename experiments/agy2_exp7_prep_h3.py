"""AMLC26 exp7 PREP (Kaggle): parser contract on all 7 official TSVs + derivation of the H3 holdout (IDs only).

1. Parser contract: the production loader (src/er/loader.py) is the ONE canonical reader. For each of the 7 files,
   check that a reader with the contract kwargs (sep="\\t", dtype=str, keep_default_na=False, standard CSV quoting,
   fields stripped) returns exactly the loader's frame, and quantify how the legacy plain tab split (exp4-exp6)
   differs from it (row count and per-row field differences).
2. H3: SHA-256 salted hash holdout of HOLDOUT_N train S1 IDs, disjoint from every previous validation/holdout list
   (H0 exp4 holdout, H1 previous 50k, H2 exp5 holdout). Uses S1 IDs only — no labels, no text.
Outputs (OUT_DIR): prep_report.json, h3_s1.ids.   Nothing here prints data rows.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.er.loader import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, load_ground_truth_df, load_source_tsv  # noqa: E402

ENV = os.environ.get
DATA_ROOT = Path(ENV("DATA_ROOT", "/kaggle/input/datasets/akshayalgox/amlc26-2026-official-data-v2"))
OUT_DIR = Path(ENV("OUT_DIR", "/kaggle/working/exp7_prep"))
PRIOR_LISTS = [Path(p) for p in ENV("PRIOR_LISTS", ",".join(str(ROOT / p) for p in (
    "exclude_ids/prev_val_s1_holdout_50k.ids", "exclude_ids/prev_val_s1_50k.ids", "expected/exp5_holdout_s1.ids"))).split(",")]
H3_SALT = ENV("H3_SALT", "amlc26-h3-holdout-v1")
HOLDOUT_N = int(ENV("HOLDOUT_N", "50000"))
FILES = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv",
         "test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]


def read_tsv_contract(path) -> pd.DataFrame:
    """The parsing contract, written out: tab separator, all strings, no NA inference, standard CSV quoting, strip."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
    return df.apply(lambda s: s.str.strip())


def read_tsv_plain_split(path) -> pd.DataFrame:
    """Legacy exp4-exp6 reader (plain tab split, no quote handling) — kept only to measure the discrepancy."""
    with open(path, encoding="utf-8") as f:
        header = next(f).rstrip("\r\n").split("\t")
        rows = [line.rstrip("\r\n").split("\t") for line in f]
    width = len(header)
    rows = [(r + [""] * width)[:width] for r in rows]
    return pd.DataFrame(rows, columns=header).apply(lambda s: s.str.strip())


def load_canonical(path) -> pd.DataFrame:
    """Production loader output restricted to the official columns."""
    if Path(path).name.endswith("ground_truth.tsv"):
        # loader keeps matched_entity_ids unstripped and strips each id in parse_id_list; compare stripped strings
        return load_ground_truth_df(path)[GROUND_TRUTH_COLUMNS].fillna("").astype(str).apply(lambda s: s.str.strip())
    return load_source_tsv(path)[SOURCE_COLUMNS]


def salted_holdout(all_ids, excluded: set, n: int, salt: str):
    """Deterministic SHA-256 hash holdout: independent of input order, disjoint from `excluded`."""
    pool = sorted({e for e in all_ids if e not in excluded},
                  key=lambda e: hashlib.sha256(f"{salt}|{e}".encode()).digest())
    return pool[:n]


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def find(name):
    hits = sorted(DATA_ROOT.rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {name} under {DATA_ROOT}, found {len(hits)}")
    return hits[0]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep = {"parser": {}, "h3": {}}
    s1_ids = None
    for name in FILES:
        p = find(name)
        canon = load_canonical(p)
        contract = read_tsv_contract(p)[list(canon.columns)]
        plain = read_tsv_plain_split(p)[list(canon.columns)]
        with open(p, encoding="utf-8") as f:
            n_lines = sum(1 for _ in f) - 1
        with open(p, encoding="utf-8") as f:
            next(f)
            quote_lines = sum(1 for line in f if '"' in line)
        r = {"data_lines": n_lines, "loader_rows": len(canon), "contract_rows": len(contract), "plain_rows": len(plain),
             "lines_with_quote_char": quote_lines, "contract_equals_loader": bool(contract.equals(canon))}
        if len(plain) == len(canon):
            diff = (plain.values != canon.values)
            r["plain_vs_loader_rows_differing"] = int(diff.any(axis=1).sum())
            r["plain_vs_loader_cells_differing_by_column"] = {c: int(diff[:, i].sum()) for i, c in enumerate(canon.columns)}
            r["plain_vs_loader_id_column_identical"] = bool((plain.iloc[:, 0].values == canon.iloc[:, 0].values).all())
        else:
            r["plain_vs_loader_rows_differing"] = None
            r["plain_vs_loader_id_sets_identical"] = bool(set(plain.iloc[:, 0]) == set(canon.iloc[:, 0]))
        rep["parser"][name] = r
        print(name, r, flush=True)
        if name == "train_source1.tsv":
            s1_ids = canon["entity_id"].tolist()
        del canon, contract, plain
    rep["parser"]["all_contract_equals_loader"] = all(v["contract_equals_loader"] for v in rep["parser"].values())

    s1_set = set(s1_ids)
    assert len(s1_set) == len(s1_ids), "duplicate S1 ids"
    prior = {}
    excluded = set()
    for p in PRIOR_LISTS:
        ids = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        assert len(set(ids)) == len(ids), p
        prior[p.name] = {"n": len(ids), "sha256": sha256_file(p), "all_in_train_s1": bool(set(ids) <= s1_set)}
        excluded |= set(ids)
    h3 = salted_holdout(s1_ids, excluded, HOLDOUT_N, H3_SALT)
    assert len(h3) == HOLDOUT_N and len(set(h3)) == HOLDOUT_N
    overlaps = {name: len(set(h3) & {l.strip() for l in (next(q for q in PRIOR_LISTS if q.name == name)).read_text().splitlines() if l.strip()})
                for name in prior}
    assert all(v == 0 for v in overlaps.values()), overlaps
    assert salted_holdout(list(reversed(s1_ids)), excluded, HOLDOUT_N, H3_SALT) == h3, "order dependence"
    out = OUT_DIR / "h3_s1.ids"
    out.write_text("\n".join(h3) + "\n")
    rep["h3"] = {"salt": H3_SALT, "n": len(h3), "sha256": sha256_file(out), "train_s1_total": len(s1_ids),
                 "excluded_union": len(excluded), "prior_lists": prior, "overlap_with_prior": overlaps}
    (OUT_DIR / "prep_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["h3"], indent=1), flush=True)
    print("contract == loader on all 7 files:", rep["parser"]["all_contract_equals_loader"], flush=True)


if __name__ == "__main__":
    main()
