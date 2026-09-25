"""AMLC26 exp9 — matcher retrain from the IMMUTABLE exp7 H1 pair cache (Kaggle). Pre-registered: configs/exp9_prereg.json.

Artifact graph (exp8 content-addressed store; key = sha256(kind, parents, params, code)):
  v0_seed/<k>  exp7 H1 dump imported as-is (queries + pairs)       <- nothing upstream is recomputed
  c0_norm/<k>  normalised S1/target texts, street number/PIN, S2+S3 name-token DF
  c4_mfeat/<k> the 18 new pair features, row-aligned to the cached pairs
  c5_matcher/<k> per variant: cross-fitted OOF scores + model files
Future feature or model experiments open these nodes instead of re-running retrieval or pruning.
Variants B0 (exact exp7), B1 (cap 1000), B2 (+A,B,C,E), B3 (conditional ablation). H2/H3/H4 are never read.
"""
from __future__ import annotations

import gc
import glob
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import agy2_exp8_store as S  # noqa: E402
from experiments import agy2_exp9_features as F9  # noqa: E402
from experiments.agy2_exp4_fresh_holdout_validation import extract_address_components, norm_address, norm_name  # noqa: E402
from experiments.agy2_exp7_h3 import M_FEATS  # noqa: E402
from src.er.loader import load_source_tsv  # noqa: E402

ENV = os.environ.get
PREREG_PATH = Path(os.environ.get("EXP9_PREREG", str(ROOT / "experiments" / "configs" / "exp9_prereg.json")))
PREREG = json.loads(PREREG_PATH.read_text())
DATA_ROOT = Path(ENV("DATA_ROOT", "/kaggle/input/datasets/akshayalgox/amlc26-2026-official-data-v2"))
OUT = Path(ENV("OUT_DIR", "/kaggle/working/exp9"))
DUMP_GLOB = ENV("DUMP_GLOB", "/kaggle/input/**/h1_matcher_oof.parquet")
VARIANTS_ONLY = [v for v in ENV("VARIANTS", "B0,B1,B2,B3").split(",") if v]
T0 = time.time()
CODE = S.code_sha("experiments/agy2_exp9_retrain.py", "experiments/agy2_exp9_features.py", "experiments/agy2_exp7_h3.py")
LGB = PREREG["lightgbm"]


def log(m):
    print(f"[{(time.time() - T0) / 60:6.1f} min | peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:7.0f} MB] {m}", flush=True)


def find(name):
    hits = sorted(DATA_ROOT.rglob(name)); assert len(hits) == 1, name; return hits[0]


# ------------------------------------------------------------------ decision layer (exp7 logic, module-level)
def prep(D):
    D = D.sort_values(["q", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
    g = D.groupby("q", sort=False)
    D["r"] = g.cumcount().values + 1
    D["mx"] = g.p.transform("max").values
    D["cs"] = g.p.cumsum().values
    D["tot"] = g.p.transform("sum").values
    D["_l"] = np.log(np.clip(1 - D.p.values, 1e-9, 1))
    D["e0"] = np.exp(D.groupby("q", sort=False)._l.transform("sum").values)
    return D


def decide(D, rule, prm):
    r, p = D.r.values, D.p.values
    if rule == "thr":
        return p >= prm["t"]
    if rule == "topn":
        return (r <= prm["n"]) & (p >= prm["t"])
    if rule == "gap":
        return (p >= prm["t"]) & (p >= prm["g"] * D.mx.values)
    ef = 1.25 * D.cs.values / (r + 0.25 * (D.tot.values + prm["m"]))
    best = pd.Series(ef).groupby(D.q.values).transform("max").values
    kstar = pd.Series(np.where(ef >= best, r, 10**9)).groupby(D.q.values).transform("min").values
    return (r <= kstar) & (best > D.e0.values) & (D.mx.values >= prm["s"])


def constrain(D, sel):
    idx = np.flatnonzero(sel)
    o = np.lexsort((-D.p.values[idx], D.code.values[idx]))
    cs = D.code.values[idx][o]
    first = np.r_[True, cs[1:] != cs[:-1]]
    out = np.zeros(len(D), bool); out[idx[o][first]] = True
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store = S.Store()                                   # sealed: H3 (+ H4 through EXP8_SEALED_LISTS)
    res = {"prereg_file": str(PREREG_PATH), "prereg_sha256": S.sha256_file(PREREG_PATH), "code": CODE, "variants": {}}

    # ---------------- v0: import the immutable exp7 H1 cache (once)
    dump_dir = Path(sorted(glob.glob(DUMP_GLOB, recursive=True))[0]).parent
    dump_sha = {f: S.sha256_file(dump_dir / f) for f in ("h1_matcher_oof.parquet", "h1_queries.parquet", "h1_truth.parquet")}
    P = S.read_parquet(dump_dir / "h1_matcher_oof.parquet")
    Qd = S.read_parquet(dump_dir / "h1_queries.parquet")
    TR = S.read_parquet(dump_dir / "h1_truth.parquet")
    B = PREREG["base"]
    assert len(P) == B["expected_rows"] and int(P.selected.sum()) == B["expected_selected"], "cache differs from exp7 dump"
    assert set(Qd.s1_id) == set(S.group_ids("dev")), "cache S1 set != frozen H1"
    fold, es = S.folds_for(Qd.s1_id.tolist())
    assert (fold == Qd.fold.values).all(), "fold assignment differs from exp7"
    Qd["es"] = es
    P = P.sort_values(["q", "prune_rank"], kind="stable").reset_index(drop=True)    # exp7 matcher row order
    assert not P.duplicated(["q", "prune_rank"]).any()
    log(f"cache: {len(P):,} pairs, {len(Qd):,} S1, dump sha {dict((k, v[:12]) for k, v in dump_sha.items())}")
    v0p = {"source": "kaggle:akshayalgox/amlc26-exp7-h1-dump@v2", "dump_sha256": dump_sha}
    k0 = S.node_key("v0_seed", {}, v0p, CODE)
    if not store.exists("v0_seed", k0):
        w = store.create("v0_seed", {}, v0p, CODE, tags=["exp9", "exp7-h1-cache"])
        w.put_table("queries", Qd.assign(group="dev")[["q", "s1_id", "group", "fold", "es", "truth_n", "country"]])
        w.put_table("pairs", P); w.put_table("truth", TR)
        v0 = w.finalize()
    else:
        v0 = store.open("v0_seed", k0)
    log(f"v0 {v0.ref}")

    # ---------------- c0: normalised texts + numbers + name-token DF (production loader, one frame at a time)
    c0p = {"tokenizer": "norm_name().split()", "df_corpus": "all train S2+S3 names", "numbers": "extract_address_components"}
    k0c = S.node_key("c0_norm", {"v0": v0.ref}, c0p, CODE)
    if not store.exists("c0_norm", k0c):
        s1 = load_source_tsv(find("train_source1.tsv"))
        s1 = s1[s1.entity_id.isin(set(Qd.s1_id))]
        comp = [extract_address_components(a, c.upper()) for a, c in zip(s1.business_address, s1.country)]
        s1t = pd.DataFrame({"s1_id": s1.entity_id.values, "qn": [norm_name(x) for x in s1.business_name],
                            "qa": [norm_address(x) for x in s1.business_address],
                            "q_num": [c["num"] or "" for c in comp], "q_pin": [c["pin"] or "" for c in comp]})
        del s1, comp
        need = np.unique(P.code.values)
        df_names, parts = __import__("collections").Counter(), []
        n_docs = 0
        for f, tag in (("train_source2.tsv", 2), ("train_source3.tsv", 3)):
            src = load_source_tsv(find(f))
            names = [norm_name(x) for x in src.business_name]
            df_names.update(t for n in names for t in set(n.split())); n_docs += len(names)
            codes = src.entity_id.str[3:].astype(np.int64).values * 4 + tag
            m = np.isin(codes, need)
            sub = src[m]
            comp = [extract_address_components(a, c.upper()) for a, c in zip(sub.business_address, sub.country)]
            parts.append(pd.DataFrame({"code": codes[m], "tn": np.array(names, dtype=object)[m],
                                       "ta": [norm_address(x) for x in sub.business_address],
                                       "t_raw_empty": (sub.business_address.str.strip() == "").values,
                                       "t_num": [c["num"] or "" for c in comp], "t_pin": [c["pin"] or "" for c in comp]}))
            del src, names, codes, sub, comp; gc.collect()
            log(f"c0: {f} processed")
        tt = pd.concat(parts, ignore_index=True); del parts
        ndf = pd.DataFrame({"token": list(df_names), "df": np.fromiter(df_names.values(), np.int64)})
        w = store.create("c0_norm", {"v0": v0.ref}, c0p, CODE, tags=["exp9"])
        w.put_table("s1_text", s1t); w.put_table("t_text", tt); w.put_table("name_df", ndf)
        w.extra["n_docs"] = n_docs
        c0 = w.finalize()
        del df_names, ndf, tt, s1t; gc.collect()
    else:
        c0 = store.open("c0_norm", k0c)
    log(f"c0 {c0.ref}")

    # ---------------- c4: new features row-aligned to the cached pairs
    c4p = {"groups": F9.GROUPS, "char3": "HashingVectorizer char_wb 3-gram 2^20 TF l2", "strong_name": 0.9}
    k4 = S.node_key("c4_mfeat", {"v0": v0.ref, "c0": c0.ref}, c4p, CODE)
    if not store.exists("c4_mfeat", k4):
        s1t, tt, ndf = c0.table("s1_text"), c0.table("t_text"), c0.table("name_df")
        qmap = Qd[["q", "s1_id"]].merge(s1t, on="s1_id").sort_values("q").reset_index(drop=True)
        assert len(qmap) == len(Qd) and (qmap.q.values == np.arange(len(Qd))).all()
        tt = tt.sort_values("code").reset_index(drop=True)
        ti = np.searchsorted(tt.code.values, P.code.values); assert (tt.code.values[ti] == P.code.values).all()
        qi = P.q.values
        qn, tn = qmap.qn.values[qi], tt.tn.values[ti]
        idf = F9.idf_table(dict(zip(ndf.token, ndf.df)), c0.man["extra"]["n_docs"])
        wj, mx, cov = F9.name_idf_features(qn, tn, idf, default_idf=max(idf.values()))
        log("c4: group A idf done")
        ch = F9.char3_cosine(qmap.qn.tolist(), tt.tn.tolist(), qi, ti)
        am = F9.missing_semantics(P.addr_jw.values, P.addr_jacc.values, P.addr_cont.values,
                                  P.q_addr_missing.values, tt.t_raw_empty.values[ti], wj)
        cn = F9.number_features(qmap.q_num.values[qi], tt.t_num.values[ti], qmap.q_pin.values[qi], tt.t_pin.values[ti],
                                qmap.qa.values[qi], tt.ta.values[ti])
        ce = F9.s1_context(qi, P.name_jw.values, P.addr_jw.values)
        NF = pd.DataFrame({"q": P.q.values, "code": P.code.values})
        for name, arr in zip(F9.NEW_FEATURES, (wj, mx, cov, ch) + am + cn + ce):
            NF[name] = arr
        del qn, tn, wj, mx, cov, ch, am, cn, ce, s1t, tt, ndf, idf; gc.collect()
        w = store.create("c4_mfeat", {"v0": v0.ref, "c0": c0.ref}, c4p, CODE, tags=["exp9"])
        w.put_table("newfeat", NF)
        c4 = w.finalize()
    else:
        c4 = store.open("c4_mfeat", k4)
        NF = c4.table("newfeat")
    assert (NF.q.values == P.q.values).all() and (NF.code.values == P.code.values).all()
    for c in F9.NEW_FEATURES:
        P[c] = NF[c].values
    del NF; gc.collect()
    log(f"c4 {c4.ref}")

    # ---------------- models
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    y = P.label.values.astype(np.int8)
    fold_row = Qd.fold.values[P.q.values]; es_row = Qd.es.values[P.q.values]
    truth_n = Qd.truth_n.values.astype(float); cty = Qd.country.values; nq = len(Qd)
    cached_p = P.p.values.astype(np.float64)

    def fit_variant(name, feats, rounds):
        X = np.empty((len(P), len(feats)), np.float32)
        for j, f in enumerate(feats):
            X[:, j] = P[f].values
        params = dict(objective="binary", learning_rate=LGB["learning_rate"], num_leaves=LGB["num_leaves"],
                      min_data_in_leaf=LGB["min_data_in_leaf"], feature_fraction=LGB["feature_fraction"],
                      bagging_fraction=LGB["bagging_fraction"], bagging_freq=1, lambda_l2=LGB["lambda_l2"],
                      max_bin=LGB["max_bin"], seed=LGB["seed"], num_threads=os.cpu_count(), verbose=-1)
        pred = np.full(len(P), np.nan); iters, gains, fit_s = [], np.zeros(len(feats)), []
        c5p = {"variant": name, "features": feats, "max_rounds": rounds, "params": params}
        k5 = S.node_key("c5_matcher", {"c4": c4.ref}, c5p, CODE)
        w = store.create("c5_matcher", {"c4": c4.ref}, c5p, CODE, tags=["exp9", name])
        for k in range(5):
            t = time.time()
            tr = np.flatnonzero(fold_row != k); e = es_row[tr]
            dtr = lgb.Dataset(X[tr[~e]], y[tr[~e]])
            dva = lgb.Dataset(X[tr[e]], y[tr[e]], reference=dtr)
            m = lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(LGB["early_stopping"], verbose=False)])
            te = np.flatnonzero(fold_row == k)
            pred[te] = m.predict(X[te], num_iteration=m.best_iteration)
            iters.append(int(m.best_iteration)); gains += m.feature_importance("gain"); fit_s.append(round(time.time() - t, 1))
            m.save_model(str(w.tmp / f"{name}_fold{k}.txt"), num_iteration=m.best_iteration)
            del dtr, dva, m; gc.collect()
            log(f"{name} fold {k}: best_iter {iters[-1]} ({fit_s[-1]}s)")
        del X; gc.collect()
        w.put_table("scores", pd.DataFrame({"q": P.q.values, "code": P.code.values, "p": pred.astype(np.float32)}))
        w.extra.update({"best_iters": iters, "fit_sec": fit_s, "auc": float(roc_auc_score(y, pred))})
        node = w.finalize()
        return pred, iters, dict(zip(feats, gains.tolist())), node.ref

    grids = {"thr": [{"t": t} for t in [round(0.20 + 0.05 * i, 2) for i in range(14)]],
             "topn": [{"n": n, "t": t} for n in (1, 2, 3, 4, 5, 6, 8, 11) for t in (0.2, 0.3, 0.4, 0.5, 0.6)],
             "gap": [{"t": t, "g": g} for t in (0.2, 0.3, 0.4, 0.5) for g in (0.3, 0.5, 0.7, 0.85)]}

    def evaluate(pred):
        D = prep(pd.DataFrame({"q": P.q.values, "code": P.code.values, "label": P.label.values.astype(bool), "p": pred}))
        hits = np.bincount(D.q.values[D.label.values], minlength=nq)
        m_dev = float((truth_n - hits).mean())
        gr = dict(grids, expF=[{"m": mm, "s": s} for mm in (0.0, m_dev) for s in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)])

        def f05(sel):
            k = np.bincount(D.q.values[sel], minlength=nq).astype(float)
            tp = np.bincount(D.q.values[sel & D.label.values], minlength=nq).astype(float)
            return np.where(truth_n == 0, (k == 0) * 1.0, np.where(k == 0, 0.0, 1.25 * tp / np.maximum(k + 0.25 * truth_n, 1e-12))), k, tp
        best = None
        for rule, grid in gr.items():
            for prm in grid:
                base = decide(D, rule, prm)
                for con in (False, True):
                    sel = constrain(D, base) if con else base
                    sc = f05(sel)[0].mean()
                    if best is None or sc > best[0]:
                        best = (sc, rule, prm, con, sel)
        sc, rule, prm, con, sel = best
        f, k, tp = f05(sel)
        card = lambda lo, hi: float(f[(truth_n >= lo) & (truth_n <= hi)].mean())
        calib = (pd.DataFrame({"p": D.p.values, "y": D.label.values})
                 .assign(b=pd.cut(D.p.values, [0, .1, .3, .5, .7, .9, .99, 1.0001], right=False))
                 .groupby("b", observed=True).agg(n=("y", "size"), mean_p=("p", "mean"), obs=("y", "mean")))
        return {"macro_f05": float(f.mean()), "per_fold": [float(f[Qd.fold.values == j].mean()) for j in range(5)],
                "rule": rule, "params": {a: float(b) for a, b in prm.items()}, "constrained": bool(con),
                "singleton": card(0, 0), "card1": card(1, 1), "card2_5": card(2, 5), "card6p": card(6, 10**6),
                "India": float(f[cty == 0].mean()), "US": float(f[cty == 1].mean()),
                "precision": float(tp.sum() / max(k.sum(), 1)), "recall": float(tp.sum() / truth_n.sum()),
                "false_positives": int(k.sum() - tp.sum()), "true_links_rejected": int(D.label.sum() - tp.sum()),
                "calibration": {str(i): {"n": int(r.n), "mean_p": round(float(r.mean_p), 4), "obs": round(float(r.obs), 4)}
                                for i, r in calib.iterrows()}}

    base_feats = list(M_FEATS)
    res["cache_check"] = {"cached_exp7_eval": evaluate(cached_p)}
    log(f"cached exp7 OOF re-evaluated: {res['cache_check']['cached_exp7_eval']['macro_f05']:.9f}")
    plan = {"B0": (base_feats, 600), "B1": (base_feats, 1000), "B2": (base_feats + F9.NEW_FEATURES, 1000)}
    for name in ("B0", "B1", "B2"):
        if name not in VARIANTS_ONLY:
            continue
        t = time.time()
        pred, iters, gains, ref = fit_variant(name, *plan[name])
        ev = evaluate(pred)
        ev.update({"best_iters": iters, "node": ref, "minutes": round((time.time() - t) / 60, 2), "n_features": len(plan[name][0])})
        if name == "B0":
            ev["max_abs_diff_vs_cached_p"] = float(np.max(np.abs(pred - cached_p)))
        if name == "B2":
            ev["group_gain"] = {g: float(sum(gains[f] for f in fs)) for g, fs in F9.GROUPS.items()}
            ev["new_feature_gain_rank"] = sorted(((gains[f], f) for f in F9.NEW_FEATURES), reverse=True)[:10]
        res["variants"][name] = ev
        log(f"{name}: macro {ev['macro_f05']:.6f} folds {[round(x, 5) for x in ev['per_fold']]} iters {iters}")
        (OUT / "exp9_results.json").write_text(json.dumps(res, indent=2, default=float))
    V = res["variants"]
    if "B3" in VARIANTS_ONLY and {"B1", "B2"} <= set(V):
        wins = sum(a > b for a, b in zip(V["B2"]["per_fold"], V["B1"]["per_fold"]))
        res["b3_gate"] = {"B2_beats_B1_folds": wins, "run": wins >= 4}
        if wins >= 4:
            drop = max(V["B2"]["group_gain"], key=V["B2"]["group_gain"].get)
            feats = base_feats + [f for g, fs in F9.GROUPS.items() if g != drop for f in fs]
            t = time.time()
            pred, iters, _, ref = fit_variant("B3", feats, 1000)
            ev = evaluate(pred)
            ev.update({"dropped_group": drop, "best_iters": iters, "node": ref, "minutes": round((time.time() - t) / 60, 2), "n_features": len(feats)})
            V["B3"] = ev
            log(f"B3 (-{drop}): macro {ev['macro_f05']:.6f}")

    # ---------------- pre-registered acceptance vs B0 and ranking
    acc = PREREG["acceptance_vs_B0"]
    if "B0" in V:
        b0 = V["B0"]
        for name, ev in V.items():
            if name == "B0":
                continue
            folds_up = sum(a > b for a, b in zip(ev["per_fold"], b0["per_fold"]))
            drops = {c: ev[c] - b0[c] for c in acc["categories"]}
            ev["vs_B0"] = {"delta": ev["macro_f05"] - b0["macro_f05"], "folds_improved": folds_up, "category_delta": drops,
                           "accepted": bool(folds_up >= acc["min_folds_improved"] and np.nanmin(list(drops.values())) >= -acc["max_drop_any_category"])}
        ok = sorted(((ev["vs_B0"]["delta"], -ev["n_features"], n) for n, ev in V.items() if n != "B0" and ev["vs_B0"]["accepted"]), reverse=True)
        pick = None
        if ok:
            pick = ok[0][2]
            simpler = [x for x in ok if ok[0][0] - x[0] <= 0.0005 and -x[1] < -ok[0][1]]
            if simpler:
                pick = min(simpler, key=lambda x: -x[1])[2]
        res["recommended_h4_candidate"] = pick
    res["runtime_min"] = round((time.time() - T0) / 60, 2)
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    (OUT / "exp9_results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({n: {k: v for k, v in ev.items() if k not in ("calibration", "new_feature_gain_rank")} for n, ev in V.items()}, indent=1, default=float))
    print("recommended_h4_candidate:", res.get("recommended_h4_candidate"))
    log("done")


if __name__ == "__main__":
    main()
