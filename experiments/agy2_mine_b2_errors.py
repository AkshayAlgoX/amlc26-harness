"""AMLC26 B2 error mining on H1 cross-fitted OOF (cache only; Kaggle). Prints/saves aggregates only, never rows.

Inputs (exp9 / exp10 store, e.g. the output of akshayalgox/amlc26-exp9-matcher-retrain): v0_seed (queries, pairs),
c4_mfeat (18 B2 features), c5_matcher (B2 scores). The decision is the exp7 expected-F0.5 family selected on these
OOF scores (same code as the pipeline), then errors are broken down by country (India first), cardinality, source,
address/number/name evidence and retrieval coverage, with per-cluster FN/FP counts and the macro-F0.5 mass at stake.
env: MINE_STORE (store root), MINE_C5 (c5 node key), OUT_DIR. No H2/H3/H4 data exists in these nodes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import agy2_exp8_store as S  # noqa: E402
from experiments import agy2_exp9_features as F9  # noqa: E402
from experiments.agy2_pipeline import constrain, decide, prep  # noqa: E402

ENV = os.environ.get


def _str_keys(o):
    if isinstance(o, dict):
        return {str(k): _str_keys(v) for k, v in o.items()}
    return o


def f05_vec(q, label, sel, truth_n, nq):
    k = np.bincount(q[sel], minlength=nq).astype(float)
    tp = np.bincount(q[sel & label], minlength=nq).astype(float)
    t = truth_n.astype(float)
    return np.where(t == 0, (k == 0) * 1.0, np.where(k == 0, 0.0, 1.25 * tp / np.maximum(k + 0.25 * t, 1e-12)))


def main():
    st = S.Store(root=Path(ENV("MINE_STORE")), forbidden=set())
    c5 = st.open("c5_matcher", ENV("MINE_C5"))
    c4 = c5.parent("c4")
    v0 = c4.parent("v0")
    Q = v0.table("queries")
    P = v0.table("pairs").sort_values(["q", "prune_rank"], kind="stable").reset_index(drop=True)
    NF = c4.table("newfeat")
    sc = c5.table("scores")
    P = P.merge(NF, on=["q", "code"], how="left").merge(sc.rename(columns={"p": "p_b2"}), on=["q", "code"], how="inner")
    nq = len(Q); truth_n = Q.truth_n.values; cty = Q.country.values
    D = prep(pd.DataFrame({"q": P.q.values, "code": P.code.values, "p": P.p_b2.values.astype(float), "label": P.label.values.astype(bool)}))
    hits = np.bincount(D.q.values[D.label.values], minlength=nq)
    m_dev = float((truth_n - hits).mean())
    best = None
    for prm in [{"m": m, "s": s} for m in (0.0, m_dev) for s in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]:
        base = decide(D, "expF", prm)
        for con in (False, True):
            sel = constrain(D, base) if con else base
            v = f05_vec(D.q.values, D.label.values, sel, truth_n, nq).mean()
            if best is None or v > best[0]:
                best = (v, prm, con, sel)
    macro, prm, con, sel = best
    f = f05_vec(D.q.values, D.label.values, sel, truth_n, nq)
    D["sel"] = sel
    X = D.merge(P.drop(columns=["label"]), on=["q", "code"], how="left")
    X["cty"] = np.where(cty[X.q.values] == 0, "India", "US")
    X["card"] = pd.cut(truth_n[X.q.values], [-1, 0, 1, 3, 5, 10**6], labels=["0", "1", "2-3", "4-5", "6+"]).astype(str)
    fn, fp = X[X.label & ~X.sel], X[~X.label & X.sel]
    rep = {"macro_f05": macro, "rule": {"params": prm, "constrained": con}, "by_country": {}}
    for c, ci in (("India", 0), ("US", 1)):
        mk = cty == ci
        rep["by_country"][c] = {"s1": int(mk.sum()), "macro_f05": float(f[mk].mean()),
                                "macro_mass_lost": float((1 - f[mk]).sum() / nq),
                                "fn": int((fn.cty == c).sum()), "fp": int((fp.cty == c).sum())}

    def clusters(E, name):
        out = {}
        for col, bins in (("name_jw", [0, .5, .7, .85, .95, 1.01]), ("addr_jw", [0, .5, .7, .85, .95, 1.01]),
                          ("name_idf_wjacc", [-.01, 0, .25, .5, .75, 1.01]), ("name_char3", [-.01, .3, .5, .7, .9, 1.01])):
            if col in E:
                out[col] = E.groupby([E.cty, pd.cut(E[col], bins)], observed=True).size().unstack(0).fillna(0).astype(int).to_dict()
        for col in ("num_conflict", "pin_conflict", "t_has_num", "both_addr_missing", "exact_name", "exact_addr", "src_s3"):
            if col in E:
                out[col] = E.groupby([E.cty, E[col]], observed=True).size().unstack(0).fillna(0).astype(int).to_dict()
        out["card"] = E.groupby([E.cty, E.card]).size().unstack(0).fillna(0).astype(int).to_dict()
        out["num_channels"] = E.groupby([E.cty, E.num_channels]).size().unstack(0).fillna(0).astype(int).to_dict()
        return _str_keys(out)
    rep["fn_clusters"] = clusters(fn, "fn")
    rep["fp_clusters"] = clusters(fp, "fp")
    out = Path(ENV("OUT_DIR", "/kaggle/working/mine_b2"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "b2_error_mining.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("macro_f05", "rule", "by_country")}, indent=1, default=str))


if __name__ == "__main__":
    main()
