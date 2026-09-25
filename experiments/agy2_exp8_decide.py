"""AMLC26 exp8 — C6 set-decision sweeps on CACHED matcher scores only (no corpus, no model, no reread).

Rules (each x singleton gate s x conflict resolution):
  thr   p >= t                      topn  rank <= n and p >= t          gap  p >= t and p >= g * max_p(S1)
  expF  adaptive set size: the k maximising expected F0.5 = 1.25*cumsum(p)/(k + 0.25*(sum(p) + m)), predicted only
        when it beats predicting nothing (P(no link) = prod(1-p)); m = 0 or DEV mean links lost before the matcher
  gate  singleton gate: predict nothing for an S1 whose best score < s (exp7 grid had it only inside expF)
  conflict  none | greedy (each target kept only for its highest-scoring S1, within the scored group)
Selection uses DEV (H1) cross-fitted OOF macro F0.5 only. H2 is reported for the selected decision only
(add --h2-per-rule to see H2 for every rule's DEV-best). H3 is not in the store.
Macro F0.5 per S1 = 1.25*TP/(k + 0.25*t); t=0 -> 1 iff k=0 (== src/er/metric.f05_score_single).
A v0_seed source has no per-S1 truth counts: it scores against links present in the exp6 pool
(score_kind = "pool_truth_proxy", biased upward; for plumbing/timing and relative comparison only).

  python -m experiments.agy2_exp8_decide --c5 c5_matcher/<key> [--gates 0,0.3,0.5] [--h2-per-rule]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import agy2_exp8_features as F  # noqa: E402
from experiments import agy2_exp8_store as S  # noqa: E402

CODE_FILES = ["experiments/agy2_exp8_decide.py"]


def per_s1_f05(q, label, sel, truth_n, n_q):
    k = np.bincount(q[sel], minlength=n_q).astype(float)
    tp = np.bincount(q[sel & label], minlength=n_q).astype(float)
    t = truth_n.astype(float)
    return np.where(t == 0, (k == 0).astype(float), np.where(k == 0, 0.0, 1.25 * tp / np.maximum(k + 0.25 * t, 1e-12)))


class Group:
    """One scored group (dev / h2): rows sorted by (q, p desc, code) with cached per-S1 aggregates."""

    def __init__(self, D: pd.DataFrame, q_ids: np.ndarray, truth_n: np.ndarray):
        o = np.lexsort((D.code.values, -D.p.values, D.q.values))
        self.code = D.code.values[o]
        self.p = D.p.values[o].astype(np.float64)
        self.label = D.label.values[o].astype(bool)
        remap = np.full(int(q_ids.max()) + 1 if len(q_ids) else 0, -1, np.int64)
        remap[q_ids] = np.arange(len(q_ids))
        self.q = remap[D.q.values[o]]
        assert (self.q >= 0).all()
        self.nq, self.truth_n = len(q_ids), truth_n
        st = np.r_[0, np.flatnonzero(self.q[1:] != self.q[:-1]) + 1] if len(self.q) else np.zeros(0, np.int64)
        ln = np.diff(np.r_[st, len(self.q)])
        self.r = np.arange(len(self.q)) - np.repeat(st, ln) + 1
        self.mx = np.repeat(self.p[st], ln)
        cs = np.cumsum(self.p)
        base = np.repeat(np.r_[0.0, cs][st], ln)
        self.cs = cs - base
        self.tot = np.repeat(np.add.reduceat(self.p, st) if len(st) else np.zeros(0), ln)
        l_ = np.log(np.clip(1 - self.p, 1e-9, 1))
        self.e0 = np.repeat(np.exp(np.add.reduceat(l_, st)) if len(st) else np.zeros(0), ln)
        self.st, self.ln = st, ln
        self._ef_cache = {}

    def decide(self, rule, prm):
        p, r = self.p, self.r
        if rule == "thr":
            sel = p >= prm["t"]
        elif rule == "topn":
            sel = (r <= prm["n"]) & (p >= prm["t"])
        elif rule == "gap":
            sel = (p >= prm["t"]) & (p >= prm["g"] * self.mx)
        elif rule == "expF":
            m = prm["m"]
            if m not in self._ef_cache:
                ef = 1.25 * self.cs / (r + 0.25 * (self.tot + m))
                best = np.repeat(np.maximum.reduceat(ef, self.st), self.ln)
                kst = np.where(ef >= best, r, 10 ** 9)
                kstar = np.repeat(np.minimum.reduceat(kst, self.st), self.ln)
                self._ef_cache[m] = (r <= kstar) & (best > self.e0)
            sel = self._ef_cache[m].copy()
        else:
            raise ValueError(rule)
        s = prm.get("s", 0.0)
        if s > 0:
            sel &= self.mx >= s
        return sel

    def constrain(self, sel):
        idx = np.flatnonzero(sel)
        o = np.lexsort((-self.p[idx], self.code[idx]))
        cs = self.code[idx][o]
        first = np.r_[True, cs[1:] != cs[:-1]] if len(cs) else np.zeros(0, bool)
        out = np.zeros(len(sel), bool)
        out[idx[o][first]] = True
        return out

    def score(self, sel):
        f = per_s1_f05(self.q, self.label, sel, self.truth_n, self.nq)
        return float(f.mean()), f


def grids(rules_cfg, m_dev, gates):
    R = rules_cfg
    g = {"thr": [{"t": t} for t in R["thr_t"]],
         "topn": [{"n": n, "t": t} for n in R["topn_n"] for t in R["topn_t"]],
         "gap": [{"t": t, "g": g2} for t in R["gap_t"] for g2 in R["gap_g"]],
         "expF": [{"m": mm, "s": s} for mm in (0.0, m_dev) for s in R["expF_s"]]}
    for rule in ("thr", "topn", "gap"):
        g[rule] = [{**prm, "s": s} for prm in g[rule] for s in gates]
    return g


def report(G: Group, sel, f):
    pred = int(sel.sum()); tp = int((sel & G.label).sum())
    return {"macro_f05": float(f.mean()), "singleton_f05": float(f[G.truth_n == 0].mean()) if (G.truth_n == 0).any() else None,
            "predicted_pairs": pred, "tp": tp, "precision": tp / max(pred, 1), "recall_links": tp / max(int(G.truth_n.sum()), 1)}


def run_decide(store: S.Store, c5_ref: str, gates=(0.0,), conflicts=("none", "greedy"), h2_per_rule=False) -> tuple:
    c5 = store.open(*c5_ref.split("/"))
    src = c5.source()
    rules_cfg = {k: v for k, v in S.frozen_config()["set_rules"].items() if isinstance(v, list)}
    params = {"rules": rules_cfg, "gates": list(gates), "conflicts": list(conflicts), "h2_per_rule": bool(h2_per_rule)}
    code = S.code_sha(*CODE_FILES)
    key = S.node_key("c6_decide", {"c5": c5.ref}, params, code)
    if store.exists("c6_decide", key):
        F.log(f"cache hit c6_decide/{key}")
        node = store.open("c6_decide", key)
        return node.ref, json.loads(node.file("decision.json").read_text())
    t0 = time.time()
    sc = c5.table("scores")
    qt = src.table("queries")
    proxy = (qt.truth_n.values < 0).any()
    truth_n_all = np.where(qt.truth_n.values < 0, qt.pool_hits.values if "pool_hits" in qt else 0, qt.truth_n.values)
    grp_of_q = qt.group.values
    G = {}
    for g in ("dev", "h2"):
        qids = np.flatnonzero(grp_of_q == g)
        m = grp_of_q[sc.q.values] == g
        G[g] = Group(sc[m], qids, truth_n_all[qids])
    t_load = time.time() - t0
    dev = G["dev"]
    m_dev = float((dev.truth_n - np.bincount(dev.q[dev.label], minlength=dev.nq)).mean())
    t1 = time.time()
    res, n_eval = {}, 0
    for rule, grid in grids(rules_cfg, m_dev, gates).items():
        best = None
        for prm in grid:
            base = dev.decide(rule, prm)
            for con in conflicts:
                sel = dev.constrain(base) if con == "greedy" else base
                s_, _ = dev.score(sel)
                n_eval += 1
                if best is None or s_ > best[0]:
                    best = (s_, prm, con)
        res[rule] = {"dev_oof_f05": best[0], "params": {k: float(v) for k, v in best[1].items()}, "conflict": best[2]}
    sweep_sec = time.time() - t1
    sel_rule = max(res, key=lambda k: res[k]["dev_oof_f05"])
    S_ = res[sel_rule]
    out = {"score_kind": "pool_truth_proxy" if proxy else "macro_f05", "selected_on": "DEV cross-fitted OOF only",
           "selected": {"rule": sel_rule, **S_}, "rules_dev": res, "m_dev": m_dev, "n_evaluations": n_eval,
           "timing_sec": {"load_and_prepare": round(t_load, 2), "sweep": round(sweep_sec, 2)}, "reports": {}}
    for g in ("dev", "h2"):
        sel = G[g].decide(sel_rule, S_["params"])
        sel = G[g].constrain(sel) if S_["conflict"] == "greedy" else sel
        out["reports"][g] = report(G[g], sel, G[g].score(sel)[1])
    if h2_per_rule:
        out["h2_per_rule_dev_best"] = {}
        for rule, v in res.items():
            sel = G["h2"].decide(rule, v["params"])
            sel = G["h2"].constrain(sel) if v["conflict"] == "greedy" else sel
            out["h2_per_rule_dev_best"][rule] = G["h2"].score(sel)[0]
    w = store.create("c6_decide", {"c5": c5.ref}, params, code, tags=["exp8"])
    w.put_json("decision.json", out)
    w.extra = {"selected": out["selected"], "score_kind": out["score_kind"], "timing_sec": out["timing_sec"]}
    node = w.finalize()
    F.log(f"C6 {node.ref}: {sel_rule} dev {S_['dev_oof_f05']:.6f} | sweep {n_eval} evals in {sweep_sec:.1f}s")
    return node.ref, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c5", required=True)
    ap.add_argument("--gates", default="0")
    ap.add_argument("--conflicts", default="none,greedy")
    ap.add_argument("--h2-per-rule", action="store_true")
    a = ap.parse_args()
    ref, out = run_decide(S.Store(), a.c5, tuple(float(x) for x in a.gates.split(",")), tuple(a.conflicts.split(",")),
                          a.h2_per_rule)
    print(ref)
    print(json.dumps({k: out[k] for k in ("score_kind", "selected", "reports", "timing_sec")}, indent=2))


if __name__ == "__main__":
    main()
