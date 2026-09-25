"""AMLC26 Experiment 6 — integrated RETRIEVAL -> LEARNED PRUNER -> PAIR MATCHER -> SET DECISION (Kaggle-first).

Two disjoint 50k S1 query groups share ONE corpus pass:
  TRAIN group = previous 50k sample (PREV_IDS)          -> fits pruner, matcher, calibration, all rule choices
  FRESH group = exp5 fresh holdout (re-derived, asserted) -> evaluation only, never fitted or tuned on
Fresh-holdout isolation: IDF + address-token DF are fitted on train S1 EXCLUDING the fresh group; no canonical view;
every budget / threshold / rule is selected on TRAIN out-of-fold (GroupKFold by S1) results, then applied once to FRESH.
The fresh S1 rows stay in the reverse-retrieval corpus (they are competitors at test time too) — no labels are used.

Stages
  1 retrieval  : A + B + Forward TF-IDF@200 (rank kept) + Reverse TF-IDF@5 (all train S1) + B2 + B3 (global-freq cap,
                 city sub-block) + B4 -> exact unique (S1, target) pairs with cheap features
  2 pruner     : LightGBM on cheap features (all retrieved negatives = hard negatives by construction)
  3 budgets    : top-N {40,60,80,100,120} x probability floors -> oracle F0.5 + budget stats; select on TRAIN OOF
  4 matcher    : LightGBM on pruned pairs with Jaro-Winkler (numba) / token / digit / cross-field features
  5 set rules  : threshold, top-N+floor, score-gap, expected-F0.5 greedy (+ singleton gate) — fitted on TRAIN OOF
  6 constraint : unconstrained vs greedy one-S1-per-target conflict resolution
Outputs (OUT_DIR): exp6_results.json, pairs_pool.parquet, stdout via the runner.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
import warnings
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiments.agy2_exp5_corrected_holdout as X5  # noqa: E402  (verified exp5 retrieval helpers)
from experiments.agy2_exp4_fresh_holdout_validation import clean_text, norm_address, norm_name  # noqa: E402
from src.er.ids import code_source, encode_target_id  # noqa: E402

ENV = os.environ.get
DATA_ROOT = Path(ENV("DATA_ROOT", "/kaggle/input/datasets/akshayalgox/amlc26-2026-official-data-v2"))
OUT_DIR = Path(ENV("OUT_DIR", "/kaggle/working/exp6"))
EXCLUDE_IDS_DIR = Path(ENV("EXCLUDE_IDS_DIR", str(ROOT / "exclude_ids")))
PREV_IDS = Path(ENV("PREV_IDS", str(ROOT / "exclude_ids" / "prev_val_s1_50k.ids")))
EXPECTED_FRESH = ENV("EXPECTED_FRESH", str(ROOT / "expected" / "exp5_holdout_s1.ids"))
HOLDOUT_N = int(ENV("HOLDOUT_N", "50000"))
SALT = ENV("SALT", "amlc26-exp5-holdout-v1")
CAP = int(ENV("CAP", "200"))
REV_DF_CAP = int(ENV("REV_DF_CAP", "2000"))
FWD_MIN_SIM = float(ENV("FWD_MIN_SIM", "0.05"))
MAX_DF = float(ENV("MAX_DF", "0.05"))
CHUNK = int(ENV("CHUNK", "20000"))
K_FWD = int(ENV("K_FWD", "200"))
NFOLD = int(ENV("NFOLD", "5"))
SEED = 42
BUDGETS = [int(x) for x in ENV("BUDGETS", "40,60,80,100,120").split(",")]
FLOORS = [float(x) for x in ENV("FLOORS", "0,0.001,0.003,0.01").split(",")]
MAX_MEAN, MAX_P99 = float(ENV("MAX_MEAN", "80")), float(ENV("MAX_P99", "250"))
REQUIRE_EXPECTED = ENV("REQUIRE_EXPECTED", "1") == "1"
MATCH_EXTRA = [k for k in ENV("MATCH_EXTRA", "top40_floor0.0,top120_floor0.0").split(",") if k]   # comparison budgets

A, B, F100, F200, R1, R3, R5 = 1, 2, 4, 8, 16, 32, 64
B2U, B2C, B3U, B3C, B4U, B4C = 128, 256, 512, 1024, 2048, 4096
POOL = A | B | F200 | R5 | B2C | B3C | B4C
T0 = time.time()
warnings.filterwarnings("ignore", message="Overriding the parameters from Reference Dataset")


def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[{(time.time() - T0) / 60:6.1f} min | peak {rss:7.0f} MB] {msg}", flush=True)


# ------------------------------------------------------------------ Jaro-Winkler (numba if available)
def _jw_batch_py(bufA, offA, ia, bufB, offB, ib, out):
    for p in range(len(ia)):
        a0, a1 = offA[ia[p]], offA[ia[p] + 1]
        b0, b1 = offB[ib[p]], offB[ib[p] + 1]
        la, lb = a1 - a0, b1 - b0
        if la == 0 and lb == 0:
            out[p] = 1.0
            continue
        if la == 0 or lb == 0:
            out[p] = 0.0
            continue
        md = max(la, lb) // 2 - 1
        if md < 0:
            md = 0
        fa = np.zeros(la, np.bool_)
        fb = np.zeros(lb, np.bool_)
        m = 0
        for i in range(la):
            lo = max(0, i - md)
            hi = min(i + md + 1, lb)
            for j in range(lo, hi):
                if (not fb[j]) and bufA[a0 + i] == bufB[b0 + j]:
                    fa[i] = True
                    fb[j] = True
                    m += 1
                    break
        if m == 0:
            out[p] = 0.0
            continue
        t = 0
        k = 0
        for i in range(la):
            if fa[i]:
                while not fb[k]:
                    k += 1
                if bufA[a0 + i] != bufB[b0 + k]:
                    t += 1
                k += 1
        jaro = (m / la + m / lb + (m - t / 2.0) / m) / 3.0
        pre = 0
        for i in range(min(4, la, lb)):
            if bufA[a0 + i] == bufB[b0 + i]:
                pre += 1
            else:
                break
        out[p] = jaro + pre * 0.1 * (1.0 - jaro) if jaro > 0.7 else jaro


try:
    from numba import njit
    _jw_batch = njit(cache=False)(_jw_batch_py)
    JW_ENGINE = "numba"
except Exception:          # local smoke only; Kaggle has numba
    _jw_batch = _jw_batch_py
    JW_ENGINE = "python"


def pack(strings):
    lens = np.fromiter((len(s) for s in strings), dtype=np.int64, count=len(strings))
    off = np.zeros(len(strings) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    buf = np.frombuffer("".join(strings).encode("utf-32-le"), dtype=np.uint32)
    return buf, off


def jaro_winkler_pairs(strA, ia, strB, ib):
    bufA, offA = pack(strA)
    bufB, offB = pack(strB)
    out = np.zeros(len(ia), dtype=np.float64)
    _jw_batch(bufA, offA, np.asarray(ia, np.int64), bufB, offB, np.asarray(ib, np.int64), out)
    return out.astype(np.float32)


# ------------------------------------------------------------------ metrics
def per_s1_eval(q, label, sel, truth_n, n_q):
    """Macro F0.5 over the S1s in [0, n_q): F = 1.25*TP/(k + 0.25*t); t=0 -> 1 iff k=0; k=0,t>0 -> 0."""
    k = np.bincount(q[sel], minlength=n_q).astype(float)
    tp = np.bincount(q[sel & label], minlength=n_q).astype(float)
    t = truth_n.astype(float)
    f = np.where(t == 0, (k == 0).astype(float), np.where(k == 0, 0.0, 1.25 * tp / np.maximum(k + 0.25 * t, 1e-12)))
    return f


def budget_stats(q, label, keep, truth_n, n_q, is_s2):
    c = np.bincount(q[keep], minlength=n_q)
    hits = np.bincount(q[keep & label], minlength=n_q)
    orc = X5.oracle_f05(hits, truth_n)
    tot = truth_n.sum()
    s2_tot = None
    return {"unique_pairs": int(keep.sum()), "mean": float(c.mean()), "p50": float(np.percentile(c, 50)),
            "p95": float(np.percentile(c, 95)), "p99": float(np.percentile(c, 99)), "max": int(c.max()),
            "recall": float(hits.sum() / tot), "oracle_f05": float(orc.mean()),
            "singleton_mean_cands": float(c[truth_n == 0].mean()) if (truth_n == 0).any() else None,
            "_hits_s2": int((keep & label & is_s2).sum()), "_hits_s3": int((keep & label & ~is_s2).sum())}


def jw_self_test():
    """Fail fast if the (numba) Jaro-Winkler kernel disagrees with textbook values."""
    got = jaro_winkler_pairs(["martha", "dwayne", "dixon", "", "abc"], [0, 1, 2, 3, 4],
                             ["marhta", "duane", "dicksonx", "", "xyz"], [0, 1, 2, 3, 4])
    exp = np.array([0.961111, 0.84, 0.813333, 1.0, 0.0])
    assert np.allclose(got, exp, atol=1e-4), (got, exp)
    print(f"jaro-winkler self-test PASS ({JW_ENGINE})")


def main():
    X5.DATA_ROOT, X5.SALT = DATA_ROOT, SALT
    X5.self_test()
    jw_self_test()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {"meta": {"jw_engine": JW_ENGINE, "salt": SALT, "cap": CAP, "rev_df_cap": REV_DF_CAP, "max_df": MAX_DF,
                    "fwd_min_sim": FWD_MIN_SIM, "k_fwd": K_FWD, "nfold": NFOLD, "budgets": BUDGETS, "floors": FLOORS}}
    p_s1, p_s2, p_s3, p_gt = (X5.find(n) for n in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
                                                   "train_ground_truth.tsv"))

    # ---------------- S1, GT, groups
    s1_ids, s1_name, s1_addr_raw, s1_country = [], {}, {}, {}
    for f in X5.read_tsv(p_s1):
        e = f[0]; s1_ids.append(e)
        s1_name[e] = norm_name(f[1] if len(f) > 1 else "")
        s1_addr_raw[e] = f[2] if len(f) > 2 else ""
        s1_country[e] = (f[3] if len(f) > 3 else "").strip()
    gt = {f[0].strip(): [x.strip() for x in (f[1] if len(f) > 1 else "").split(",") if x.strip()] for f in X5.read_tsv(p_gt)}
    excluded = X5.load_excluded(EXCLUDE_IDS_DIR)
    fresh = X5.select_holdout(s1_ids, excluded, HOLDOUT_N)
    if REQUIRE_EXPECTED and not Path(EXPECTED_FRESH).exists():
        raise FileNotFoundError(f"EXPECTED_FRESH missing: {EXPECTED_FRESH}")
    if Path(EXPECTED_FRESH).exists():
        exp_ids = [l.strip() for l in open(EXPECTED_FRESH) if l.strip()]
        assert fresh == exp_ids, "fresh holdout differs from the exp5 holdout"
        log("fresh holdout == exp5 holdout (asserted)")
    train_ids = [l.strip() for l in open(PREV_IDS) if l.strip()]
    s1_set = set(s1_ids)
    assert all(e in s1_set for e in train_ids), "PREV_IDS contains unknown S1"
    assert not (set(train_ids) & set(fresh)), "train and fresh groups overlap"
    Q = train_ids + fresh
    n_tr, nq = len(train_ids), len(train_ids) + len(fresh)
    qi = {e: i for i, e in enumerate(Q)}
    fresh_set = set(fresh)
    truth = [set(encode_target_id(t) for t in gt.get(e, [])) for e in Q]
    truth_n = np.array([len(s) for s in truth])
    truth_keys = np.array(sorted((i << 32) | c for i, s in enumerate(truth) for c in s), dtype=np.int64)
    all_truth_codes = {c for s in truth for c in s}
    res["meta"].update({"train_s1": n_tr, "fresh_s1": nq - n_tr, "train_links": int(truth_n[:n_tr].sum()),
                        "fresh_links": int(truth_n[n_tr:].sum()),
                        "train_singleton_rate": float((truth_n[:n_tr] == 0).mean()),
                        "fresh_singleton_rate": float((truth_n[n_tr:] == 0).mean())})
    log(f"groups: train {n_tr:,} S1 ({truth_n[:n_tr].sum():,} links) | fresh {nq - n_tr:,} S1 ({truth_n[n_tr:].sum():,} links)")

    # ---------------- per-country TF-IDF (IDF + address DF fitted WITHOUT fresh S1)
    countries = sorted(set(s1_country.values()))
    V, XS1_revT, XQ, XQ_revT, hold_map, q_rows, addr_df = {}, {}, {}, {}, {}, {}, {}
    for c in countries:
        ids_c = [e for e in s1_ids if s1_country[e] == c]
        texts = [f"{s1_name[e]} {norm_address(s1_addr_raw[e])}".strip() for e in ids_c]
        fit_texts = [t for e, t in zip(ids_c, texts) if e not in fresh_set]
        vec = TfidfVectorizer(max_df=MAX_DF, min_df=2, token_pattern=r"\b[a-zA-Z]{3,}\b", dtype=np.float32).fit(fit_texts)
        X = vec.transform(texts).tocsr()
        not_fresh = np.array([e not in fresh_set for e in ids_c])
        keep = np.bincount(X[not_fresh].indices, minlength=X.shape[1]) <= REV_DF_CAP   # DF also without fresh S1
        V[c] = (vec, keep)
        XS1_revT[c] = X[:, keep].T.tocsr()
        hm = np.full(len(ids_c), -1, dtype=np.int64)
        rows = []
        for j, e in enumerate(ids_c):
            if e in qi:
                hm[j] = qi[e]; rows.append(j)
        hold_map[c] = hm
        q_rows[c] = np.array(rows, dtype=np.int64)
        XQ[c] = X[q_rows[c]]
        XQ_revT[c] = XS1_revT[c][:, q_rows[c]].tocsr()
        addr_df[c] = Counter(t for e in ids_c if e not in fresh_set
                             for t in {w for w in clean_text(s1_addr_raw[e]).split() if len(w) >= 4 and not w.isdigit()})
        log(f"{c}: S1 {len(ids_c):,} | IDF fitted on {len(fit_texts):,} (fresh excluded) | vocab {X.shape[1]:,} | queries {len(rows):,}")
        del texts, fit_texts
    gc.collect()

    # ---------------- query-side indexes
    idx = {k: defaultdict(list) for k in ("A", "B", "B2", "B3", "B4")}
    q_parts = {}
    for e in Q:
        i, c = qi[e], s1_country[e]
        nm, ad = s1_name[e], norm_address(s1_addr_raw[e])
        if len(nm) >= 3:
            idx["A"][hash((c, nm))].append((i, 0))
        if len(ad) >= 5:
            idx["B"][hash((c, ad))].append((i, 0))
        pin, num, city, rare = X5.addr_parts(s1_addr_raw[e], c, addr_df[c])
        q_parts[i] = (pin, num)
        if city and num:
            idx["B2"][hash((c, "b2", city, num))].append((i, 0))
        for rt in rare:
            if num:
                idx["B3"][hash((c, "b3", num, rt))].append((i, hash((c, "b3", num, rt, city)) if city else 0))
            if pin:
                idx["B4"][hash((c, "b4", pin, rt))].append((i, hash((c, "b4", pin, rt, num)) if num else 0))

    # ---------------- stream all S2+S3 once
    rq, rt_, rb, rk, rs, rfs, rrs, rrk = (array("q"), array("q"), array("q"), array("q"), array("q"),
                                          array("f"), array("f"), array("q"))
    kraw = array("q")
    fS = {c: np.full((len(q_rows[c]), K_FWD), -1.0, np.float32) for c in countries}
    fI = {c: np.full((len(q_rows[c]), K_FWD), -1, np.int64) for c in countries}
    fT = {c: np.full(len(q_rows[c]), -1.0, np.float32) for c in countries}
    buf = {c: ([], []) for c in countries}
    stats = Counter()

    def add(i, t, bit, key=0, sub=0, rsc=0.0, rrank=0):
        rq.append(i); rt_.append(t); rb.append(bit); rk.append(key); rs.append(sub)
        rfs.append(0.0); rrs.append(rsc); rrk.append(rrank)

    def flush(c):
        texts, codes = buf[c]
        if not texts:
            return
        vec, keep = V[c]
        Xt = vec.transform(texts).tocsr()
        ca = np.asarray(codes, dtype=np.int64)
        Fm = (XQ[c] @ Xt.T).tocsr()
        Fm.data[Fm.data < FWD_MIN_SIM] = 0; Fm.eliminate_zeros()
        S, I, thr = fS[c], fI[c], fT[c]
        for r in np.flatnonzero(np.diff(Fm.indptr)):
            a, b = Fm.indptr[r], Fm.indptr[r + 1]
            v = Fm.data[a:b]; m = v > thr[r]
            if not m.any():
                continue
            cs = np.concatenate([S[r], v[m]]); ci = np.concatenate([I[r], ca[Fm.indices[a:b][m]]])
            if len(cs) > K_FWD:
                top = np.argpartition(-cs, K_FWD - 1)[:K_FWD]; cs, ci = cs[top], ci[top]
            S[r], I[r] = cs, ci; thr[r] = cs.min()
        Xr = Xt[:, keep]
        touch = (Xr @ XQ_revT[c]).tocsr()
        rows = np.flatnonzero(np.diff(touch.indptr))
        hm = hold_map[c]
        for s0 in range(0, len(rows), 2000):
            rr = rows[s0:s0 + 2000]
            M = (Xr[rr] @ XS1_revT[c]).tocsr()
            for k, r in enumerate(rr):
                a, b = M.indptr[k], M.indptr[k + 1]
                if a == b:
                    continue
                v, cols = M.data[a:b], M.indices[a:b]
                if len(v) > 5:
                    sel = np.argpartition(-v, 4)[:5]; v, cols = v[sel], cols[sel]
                order = np.argsort(-v)
                for rank, (j, sc) in enumerate(zip(cols[order], v[order]), 1):
                    hi = hm[j]
                    if hi >= 0:
                        add(int(hi), int(ca[r]), R5 | (R3 if rank <= 3 else 0) | (R1 if rank == 1 else 0),
                            rsc=float(sc), rrank=rank)
        stats[f"rev_rows_{c}"] += len(rows)
        texts.clear(); codes.clear()

    t_s = time.time()
    for p in (p_s2, p_s3):
        for f in X5.read_tsv(p):
            code = encode_target_id(f[0]); c = (f[3] if len(f) > 3 else "").strip()
            stats["targets"] += 1
            if c not in V:
                stats["targets_unknown_country"] += 1
                continue
            name_raw, addr_raw = (f[1] if len(f) > 1 else ""), (f[2] if len(f) > 2 else "")
            nm, ad = norm_name(name_raw), norm_address(addr_raw)
            for i, _ in idx["A"].get(hash((c, nm)), ()):
                add(i, code, A)
            if len(ad) >= 5:
                for i, _ in idx["B"].get(hash((c, ad)), ()):
                    add(i, code, B)
            pin, num, city, rare = X5.addr_parts(addr_raw, c, addr_df[c])
            if city and num:
                k = hash((c, "b2", city, num)); kraw.append(k)
                for i, _ in idx["B2"].get(k, ()):
                    add(i, code, B2U, k)
            for rtok in rare:
                if num:
                    k = hash((c, "b3", num, rtok)); ks = hash((c, "b3", num, rtok, city)) if city else 0
                    kraw.append(k)
                    if ks:
                        kraw.append(ks)
                    for i, s1sub in idx["B3"].get(k, ()):
                        add(i, code, B3U, k, ks if (ks and ks == s1sub) else 0)
                if pin:
                    k = hash((c, "b4", pin, rtok)); ks = hash((c, "b4", pin, rtok, num)) if num else 0
                    kraw.append(k)
                    if ks:
                        kraw.append(ks)
                    for i, s1sub in idx["B4"].get(k, ()):
                        add(i, code, B4U, k, ks if (ks and ks == s1sub) else 0)
            bt, bc = buf[c]; bt.append(f"{nm} {ad}".strip()); bc.append(code)
            if len(bt) >= CHUNK:
                flush(c)
            if stats["targets"] % 1_000_000 == 0:
                log(f"stream {stats['targets']:,} targets ({(time.time()-t_s)/60:.1f} min) | records {len(rq):,}")
        for c in countries:
            flush(c)
    log(f"stream done {stats['targets']:,} in {(time.time()-t_s)/60:.1f} min | {dict(stats)}")
    res["meta"]["stream_min"] = (time.time() - t_s) / 60

    for c in countries:
        S, I = fS[c], fI[c]
        for r in range(S.shape[0]):
            ok = I[r] >= 0
            if not ok.any():
                continue
            order = np.argsort(-S[r][ok]); ids, scs = I[r][ok][order], S[r][ok][order]
            hi = int(hold_map[c][q_rows[c][r]])
            for rank, (code, sc) in enumerate(zip(ids, scs), 1):
                rq.append(hi); rt_.append(int(code)); rb.append(F200 | (F100 if rank <= 100 else 0))
                rk.append(0); rs.append(0); rfs.append(float(sc)); rrs.append(0.0); rrk.append(rank)   # rank reused below
    del fS, fI, fT
    gc.collect()

    # ---------------- caps from GLOBAL target-side key counts
    kc = np.frombuffer(kraw, dtype=np.int64)
    uk, cnt = np.unique(kc, return_counts=True)
    del kc, kraw

    def gcount(keys):
        if len(uk) == 0:
            return np.zeros(len(keys), np.int64)
        pos = np.clip(np.searchsorted(uk, keys), 0, len(uk) - 1)
        return np.where((keys != 0) & (uk[pos] == keys), cnt[pos], 0)
    qa = np.frombuffer(rq, np.int64); ta = np.frombuffer(rt_, np.int64)
    bits = np.frombuffer(rb, np.int64).copy()
    ka, sa = np.frombuffer(rk, np.int64), np.frombuffer(rs, np.int64)
    fsa, rsa = np.frombuffer(rfs, np.float32), np.frombuffer(rrs, np.float32)
    rka = np.frombuffer(rrk, np.int64)
    kcount, scount = gcount(ka), gcount(sa)
    for u, cb, has_sub in ((B2U, B2C, False), (B3U, B3C, True), (B4U, B4C, True)):
        m = (bits & u) > 0
        ok = kcount <= CAP
        if has_sub:
            ok |= (sa != 0) & (scount > 0) & (scount <= CAP)
        bits[m & ok] |= cb
    is_fwd = (bits & F200) > 0
    frank = np.where(is_fwd, rka, 9999).astype(np.int64)
    rrank = np.where((bits & R5) > 0, rka, 99).astype(np.int64)
    fsc = np.where(is_fwd, fsa, 0.0).astype(np.float32)
    rsc = np.where((bits & R5) > 0, rsa, 0.0).astype(np.float32)

    # ---------------- exact unique pairs with aggregated retrieval signals
    pk = (qa << 32) | ta
    o = np.argsort(pk, kind="stable")
    pk, bits, frank, rrank, fsc, rsc = pk[o], bits[o], frank[o], rrank[o], fsc[o], rsc[o]
    st = np.flatnonzero(np.r_[True, pk[1:] != pk[:-1]])
    upk = pk[st]
    ub = np.bitwise_or.reduceat(bits, st)
    ufr, urr = np.minimum.reduceat(frank, st), np.minimum.reduceat(rrank, st)
    ufs, urs = np.maximum.reduceat(fsc, st), np.maximum.reduceat(rsc, st)
    del qa, ta, bits, ka, sa, fsa, rsa, rka, kcount, scount, pk, frank, rrank, fsc, rsc, o
    for arr_ in (rq, rt_, rb, rk, rs, rfs, rrs, rrk):
        del arr_[:]
    gc.collect()
    inpool = (ub & POOL) > 0
    upk, ub, ufr, urr, ufs, urs = upk[inpool], ub[inpool], ufr[inpool], urr[inpool], ufs[inpool], urs[inpool]
    P = pd.DataFrame({"q": (upk >> 32).astype(np.int32), "code": (upk & 0xFFFFFFFF).astype(np.int64)})
    P["label"] = np.isin(upk, truth_keys)
    P["grp"] = pd.Categorical(np.where(P.q.values < n_tr, "train", "fresh"), categories=["train", "fresh"])
    P["is_s2"] = (P.code.values & 3) == 2          # explicit prefix tag written by encode_target_id
    for name_, bit in (("flag_A", A), ("flag_B", B), ("flag_F100", F100), ("flag_F200", F200), ("flag_R1", R1),
                       ("flag_R5", R5), ("flag_b2", B2C), ("flag_b3", B3C), ("flag_b4", B4C)):
        P[name_] = ((ub & bit) > 0).astype(np.int8)
    P["num_channels"] = P[["flag_A", "flag_B", "flag_F200", "flag_R5", "flag_b2", "flag_b3", "flag_b4"]].sum(axis=1).astype(np.int8)
    P["fwd_tfidf"], P["fwd_rank"], P["rev_tfidf"], P["rev_rank"] = ufs, ufr.astype(np.int32), urs, urr.astype(np.int16)
    del upk, ub, ufr, urr, ufs, urs
    gc.collect()
    log(f"pool: {len(P):,} unique pairs | positives {int(P.label.sum()):,}")

    # retrieval-stage reports (F@200 pool and F@100 pool)
    def pool_report(mask_series, g):
        sub = P[mask_series & (P.grp == g)]
        lo, hi = (0, n_tr) if g == "train" else (n_tr, nq)
        qq = sub.q.values - lo
        c = np.bincount(qq, minlength=hi - lo)
        h = np.bincount(qq[sub.label.values], minlength=hi - lo)
        tn = truth_n[lo:hi]
        orc = X5.oracle_f05(h, tn)
        s2t = sum(1 for i in range(lo, hi) for x in truth[i] if code_source(x) == "S2")
        return {"unique_pairs": int(len(sub)), "mean": float(c.mean()), "p50": float(np.percentile(c, 50)),
                "p95": float(np.percentile(c, 95)), "p99": float(np.percentile(c, 99)), "max": int(c.max()),
                "recall": float(h.sum() / tn.sum()), "oracle_f05": float(orc.mean()),
                "s2_recall": float((sub.label & sub.is_s2).sum() / s2t), "s3_recall": float((sub.label & ~sub.is_s2).sum() / (tn.sum() - s2t))}
    f100pool = (P.flag_A | P.flag_B | P.flag_F100 | P.flag_R5 | P.flag_b2 | P.flag_b3 | P.flag_b4) > 0
    res["retrieval"] = {g: {"pool_F200": pool_report(pd.Series(True, index=P.index), g), "pool_F100": pool_report(f100pool, g)}
                        for g in ("train", "fresh")}
    for g in ("train", "fresh"):
        r = res["retrieval"][g]["pool_F200"]
        log(f"retrieval {g}: recall {r['recall']:.4%} oracle {r['oracle_f05']:.6f} mean {r['mean']:.1f} p99 {r['p99']:.0f} max {r['max']}")

    # ---------------- target texts for candidates + truth (second pass)
    need = set(P.code.values.tolist()) | all_truth_codes
    tcode, tname, taddr, traw_empty, tpin, tnum = [], [], [], [], [], []
    for p in (p_s2, p_s3):
        for f in X5.read_tsv(p):
            code = encode_target_id(f[0])
            if code in need:
                c = (f[3] if len(f) > 3 else "").strip()
                araw = f[2] if len(f) > 2 else ""
                tcode.append(code); tname.append(norm_name(f[1] if len(f) > 1 else "")); taddr.append(norm_address(araw))
                traw_empty.append(not araw.strip())
                pin, num, _, _ = X5.addr_parts(araw, c, addr_df.get(c, Counter())) if c in addr_df else ("", "", "", [])
                tpin.append(pin); tnum.append(num)
    order_t = np.argsort(np.array(tcode, dtype=np.int64))
    tcode_s = np.array(tcode, dtype=np.int64)[order_t]
    tname = [tname[i] for i in order_t]; taddr = [taddr[i] for i in order_t]
    traw_empty = np.array(traw_empty)[order_t]; tpin = [tpin[i] for i in order_t]; tnum = [tnum[i] for i in order_t]
    del tcode, order_t, need
    gc.collect()
    log(f"target texts fetched: {len(tcode_s):,}")
    P["t_idx"] = np.searchsorted(tcode_s, P.code.values)
    qname = [s1_name[e] for e in Q]
    qaddr = [norm_address(s1_addr_raw[e]) for e in Q]
    qcountry = {c: i for i, c in enumerate(countries)}

    # ---------------- cheap features
    qn_tok = [set(s.split()) for s in qname]
    qa_tok = [set(s.split()) for s in qaddr]
    qa_dig = [{t for t in s.split() if t.isdigit()} for s in qaddr]
    n = len(P)
    f_nj, f_nc, f_aj, f_dj = (np.zeros(n, np.float32) for _ in range(4))
    f_en, f_ea, f_num, f_pin = (np.zeros(n, np.int8) for _ in range(4))
    qv, tv = P.q.values, P.t_idx.values
    for r in range(n):
        i, j = qv[r], tv[r]
        tn_, ta_ = tname[j], taddr[j]
        a, b = qn_tok[i], set(tn_.split())
        inter = len(a & b)
        f_nj[r] = inter / max(len(a | b), 1); f_nc[r] = inter / max(min(len(a), len(b)), 1)
        c_, d_ = qa_tok[i], set(ta_.split())
        f_aj[r] = len(c_ & d_) / max(len(c_ | d_), 1)
        dd = {t for t in d_ if t.isdigit()}
        f_dj[r] = len(qa_dig[i] & dd) / max(len(qa_dig[i] | dd), 1) if (qa_dig[i] or dd) else 0.0
        f_en[r] = qname[i] == tn_ and len(tn_) > 0
        f_ea[r] = qaddr[i] == ta_ and len(ta_) > 0
        qp, qn2 = q_parts[i]
        f_num[r] = bool(qn2) and qn2 == tnum[j]
        f_pin[r] = bool(qp) and qp == tpin[j]
    P["name_jacc"], P["name_cont"], P["addr_jacc"], P["digit_jacc"] = f_nj, f_nc, f_aj, f_dj
    P["exact_name"], P["exact_addr"], P["num_match"], P["pin_match"] = f_en, f_ea, f_num, f_pin
    q_amiss = np.array([not s1_addr_raw[e].strip() for e in Q], dtype=np.int8)
    q_cty = np.array([qcountry[s1_country[e]] for e in Q], dtype=np.int8)
    q_nlen = np.array([len(s) for s in qname]); t_nlen = np.array([len(s) for s in tname])
    P["q_addr_missing"] = q_amiss[qv]
    P["t_addr_missing"] = traw_empty[tv].astype(np.int8)
    P["country"] = q_cty[qv]
    P["src_s3"] = (~P.is_s2).astype(np.int8)
    P["name_len_diff"] = np.abs(q_nlen[qv] - t_nlen[tv]).astype(np.int16)
    g = P.groupby("q")
    P["cands_s1"] = g.q.transform("size").astype(np.int32)
    P["fwd_ratio_best"] = (P.fwd_tfidf / g.fwd_tfidf.transform("max").replace(0, np.nan)).fillna(0).astype(np.float32)
    P["name_jacc_rank"] = g.name_jacc.rank(ascending=False, method="min").astype(np.float32)
    del f_nj, f_nc, f_aj, f_dj, f_en, f_ea, f_num, f_pin
    gc.collect()
    log("cheap features built")
    try:
        P.drop(columns=["t_idx"]).to_parquet(OUT_DIR / "pairs_pool.parquet", index=False)
    except ImportError as ex:          # local smoke venv has no pyarrow; Kaggle image does
        log(f"pairs_pool.parquet skipped: {ex.__class__.__name__}")

    PR_FEATS = ["fwd_tfidf", "fwd_rank", "rev_tfidf", "rev_rank", "num_channels", "flag_A", "flag_B", "flag_F100",
                "flag_R1", "flag_R5", "flag_b2", "flag_b3", "flag_b4", "exact_name", "exact_addr", "country", "src_s3",
                "name_jacc", "name_cont", "addr_jacc", "digit_jacc", "num_match", "pin_match", "q_addr_missing",
                "t_addr_missing", "name_len_diff", "cands_s1", "fwd_ratio_best", "name_jacc_rank"]

    import lightgbm as lgb

    def fit_oof(df, feats, tag, rounds=600):
        """GroupKFold OOF on TRAIN rows; fold models averaged for FRESH. Early stop on an inner group split."""
        tr = df[df.grp == "train"]; fr = df[df.grp == "fresh"]
        oof = np.zeros(len(tr)); pf = np.zeros(len(fr))
        params = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=100,
                      feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED,
                      num_threads=os.cpu_count(), verbose=-1)
        imps = np.zeros(len(feats))
        for k, (a, b) in enumerate(GroupKFold(n_splits=NFOLD).split(tr, groups=tr.q.values)):
            ta_, va = tr.iloc[a], tr.iloc[b]
            inner = (ta_.q.values.astype(np.int64) * 2654435761 % 10) == 0   # inner early-stopping groups, never the scored fold
            y = ta_.label.values.astype(np.int8)
            dtr = lgb.Dataset(ta_[feats][~inner], y[~inner], params={"max_bin": 127})
            dva = lgb.Dataset(ta_[feats][inner], y[inner], reference=dtr)
            m = lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            oof[b] = m.predict(va[feats], num_iteration=m.best_iteration)
            pf += m.predict(fr[feats], num_iteration=m.best_iteration) / NFOLD
            imps += m.feature_importance("gain")
            log(f"{tag} fold {k}: best_iter {m.best_iteration}")
        from sklearn.metrics import roc_auc_score
        info = {"oof_auc_train": float(roc_auc_score(tr.label, oof)), "fresh_auc": float(roc_auc_score(fr.label, pf)),
                "top_features": [f for _, f in sorted(zip(imps, feats), reverse=True)[:12]]}
        return oof, pf, info

    # ---------------- stage 2: pruner
    oof, pf, pinfo = fit_oof(P, PR_FEATS, "pruner")
    P["prune_p"] = 0.0
    P.loc[P.grp == "train", "prune_p"] = oof
    P.loc[P.grp == "fresh", "prune_p"] = pf
    P.sort_values(["q", "prune_p"], ascending=[True, False], inplace=True, kind="stable")
    P["prune_rank"] = P.groupby("q").cumcount().astype(np.int32) + 1
    res["pruner"] = pinfo
    log(f"pruner: {pinfo}")

    # ---------------- stage 3: budgets (select on TRAIN OOF, report both)
    qv_all, lab = P.q.values, P.label.values
    s2v = P.is_s2.values
    def group_slice(g):
        m = (P.grp == g).values
        lo = 0 if g == "train" else n_tr
        return m, lo, (n_tr if g == "train" else nq - n_tr)
    s2_tot = {g: sum(1 for i in (range(0, n_tr) if g == "train" else range(n_tr, nq)) for x in truth[i]
                     if code_source(x) == "S2") for g in ("train", "fresh")}
    budgets = {}
    for N in BUDGETS:
        for fl in FLOORS:
            keep = (P.prune_rank.values <= N) & (P.prune_p.values >= fl)
            row = {}
            for g in ("train", "fresh"):
                m, lo, ng = group_slice(g)
                tn = truth_n[lo:lo + ng]
                s = budget_stats(qv_all[m] - lo, lab[m], keep[m], tn, ng, s2v[m])
                s2t = s2_tot[g]
                s["s2_recall"] = s.pop("_hits_s2") / s2t
                s["s3_recall"] = s.pop("_hits_s3") / (tn.sum() - s2t)
                row[g] = s
            budgets[f"top{N}_floor{fl}"] = row
    feasible = {k: v for k, v in budgets.items() if v["train"]["mean"] <= MAX_MEAN and v["train"]["p99"] <= MAX_P99}
    sel_budget = max(feasible, key=lambda k: feasible[k]["train"]["oracle_f05"]) if feasible else None
    res["budgets"] = budgets
    res["selected_budget"] = sel_budget
    log(f"selected budget (on TRAIN OOF): {sel_budget}")
    if sel_budget is None:
        (OUT_DIR / "exp6_results.json").write_text(json.dumps(res, indent=2, default=float))
        log("STOP: no budget meets mean<=%s and p99<=%s on TRAIN OOF" % (MAX_MEAN, MAX_P99))
        return
    res["timing_min"] = {"to_budgets": (time.time() - T0) / 60}
    M_FEATS = PR_FEATS + ["prune_p", "prune_rank", "name_jw", "addr_jw", "name_cont_q", "name_cont_t", "name_len_ratio",
                          "tok_count_diff", "addr_cont", "exact_both", "strong_n_strong_a", "strong_n_weak_a",
                          "weak_n_strong_a", "name_x_addr"]
    fresh_truth = truth_keys[(truth_keys >> 32) >= n_tr]
    pool_fresh_keys = np.sort((P.q.values[(P.grp == "fresh").values].astype(np.int64) << 32)
                              | P.code.values[(P.grp == "fresh").values])

    def prep(D):
        """Sort once by (q, p desc) and cache per-S1 aggregates used by every rule."""
        D = D.sort_values(["q", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
        g_ = D.groupby("q", sort=False)
        D["r"] = g_.cumcount().values + 1
        D["mx"] = g_.p.transform("max").values
        D["cs"] = g_.pc.cumsum().values
        D["tot"] = g_.pc.transform("sum").values
        D["_l"] = np.log(np.clip(1 - D.pc.values, 1e-9, 1))
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
        # expected-F0.5 greedy on calibrated probabilities (+ m expected unseen links) + singleton gate
        ef = 1.25 * D.cs.values / (r + 0.25 * (D.tot.values + prm["m"]))
        best = pd.Series(ef).groupby(D.q.values).transform("max").values
        kstar = pd.Series(np.where(ef >= best, r, 10**9)).groupby(D.q.values).transform("min").values
        return (r <= kstar) & (best > D.e0.values) & (D.mx.values >= prm["s"])

    def score(D, sel, g, constrain=False):
        """Macro F0.5 over ALL S1 of group g. constrain = greedy: each target kept only for its highest-p S1."""
        _, lo, ng = group_slice(g)
        if constrain:
            idx_sel = np.flatnonzero(sel)
            o_ = np.lexsort((-D.p.values[idx_sel], D.code.values[idx_sel]))
            cs_ = D.code.values[idx_sel][o_]
            first = np.r_[True, cs_[1:] != cs_[:-1]]
            sel = np.zeros(len(D), bool); sel[idx_sel[o_][first]] = True
        f = per_s1_eval(D.q.values - lo, D.label.values, sel, truth_n[lo:lo + ng], ng)
        return float(f.mean()), f, sel

    def run_matcher(key, full):
        t0 = time.time()
        N_, fl_ = int(key.split("_")[0][3:]), float(key.split("floor")[1])
        M = P[(P.prune_rank.values <= N_) & (P.prune_p.values >= fl_)].copy()
        tvm, qvm = M.t_idx.values, M.q.values
        M["name_jw"] = jaro_winkler_pairs(qname, qvm, tname, tvm)
        M["addr_jw"] = jaro_winkler_pairs(qaddr, qvm, taddr, tvm)
        ql = np.array([len(qn_tok[i]) for i in qvm]); tl = np.array([len(tname[j].split()) for j in tvm])
        inter = np.array([len(qn_tok[i] & set(tname[j].split())) for i, j in zip(qvm, tvm)])
        M["name_cont_q"] = inter / np.maximum(ql, 1); M["name_cont_t"] = inter / np.maximum(tl, 1)
        M["name_len_ratio"] = (q_nlen[qvm] + 1) / (t_nlen[tvm] + 1)
        M["tok_count_diff"] = np.abs(ql - tl)
        ainter = np.array([len(qa_tok[i] & set(taddr[j].split())) for i, j in zip(qvm, tvm)])
        M["addr_cont"] = ainter / np.maximum(np.array([len(qa_tok[i]) for i in qvm]), 1)
        M["exact_both"] = (M.exact_name & M.exact_addr).astype(np.int8)
        M["strong_n_strong_a"] = ((M.name_jw >= 0.9) & (M.addr_jw >= 0.9)).astype(np.int8)
        M["strong_n_weak_a"] = ((M.name_jw >= 0.9) & (M.addr_jw < 0.5)).astype(np.int8)
        M["weak_n_strong_a"] = ((M.name_jw < 0.5) & (M.addr_jw >= 0.9)).astype(np.int8)
        M["name_x_addr"] = M.name_jw * M.addr_jw
        log(f"[{key}] matcher pairs {len(M):,} (train {int((M.grp == 'train').sum()):,}) | JW engine {JW_ENGINE}")
        moof, mpf, minfo = fit_oof(M, M_FEATS, f"matcher[{key}]")
        is_tr = (M.grp == "train").values
        M["p"] = 0.0
        M.loc[is_tr, "p"] = moof
        M.loc[~is_tr, "p"] = mpf
        iso = IsotonicRegression(out_of_bounds="clip").fit(moof, M.label.values[is_tr])   # TRAIN OOF only
        M["pc"] = iso.predict(M.p.values)
        tr_hits = np.bincount(M.q.values[is_tr & M.label.values], minlength=n_tr)[:n_tr]
        miss_per_s1 = float((truth_n[:n_tr] - tr_hits).mean())
        Dtr, Dfr = prep(M[is_tr]), prep(M[~is_tr])
        grids = {"thr": [{"t": round(t, 2)} for t in np.arange(0.2, 0.86, 0.05)],
                 "topn": [{"n": nn, "t": t} for nn in (1, 2, 3, 4, 5, 6, 8, 11) for t in (0.2, 0.3, 0.4, 0.5, 0.6)],
                 "gap": [{"t": t, "g": gg} for t in (0.2, 0.3, 0.4, 0.5) for gg in (0.3, 0.5, 0.7, 0.85)],
                 "expF": [{"m": mm, "s": s} for mm in (0.0, miss_per_s1) for s in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]}
        rules = {}
        for rule, grid in grids.items():
            best = None
            for prm in grid:
                sc = score(Dtr, decide(Dtr, rule, prm), "train")[0]
                if best is None or sc > best[0]:
                    best = (sc, prm)
            out = {"params": {k: float(v) for k, v in best[1].items()}}
            for con in (False, True):
                tag = "constrained" if con else "unconstrained"
                out[f"train_oof_f05_{tag}"] = score(Dtr, decide(Dtr, rule, best[1]), "train", con)[0]
                out[f"fresh_f05_{tag}"] = score(Dfr, decide(Dfr, rule, best[1]), "fresh", con)[0]
            rules[rule] = out
            log(f"[{key}] rule {rule}: {out}")
        best_rule = max(rules, key=lambda k: max(rules[k]["train_oof_f05_unconstrained"], rules[k]["train_oof_f05_constrained"]))
        use_con = rules[best_rule]["train_oof_f05_constrained"] > rules[best_rule]["train_oof_f05_unconstrained"]
        prm = rules[best_rule]["params"]
        tr_mean, tr_vec, _ = score(Dtr, decide(Dtr, best_rule, prm), "train", use_con)
        f_mean, f_vec, sf = score(Dfr, decide(Dfr, best_rule, prm), "fresh", use_con)
        out = {"matcher": minfo, "matcher_pairs": int(len(M)), "train_expected_missed_per_s1": miss_per_s1,
               "set_rules": rules, "selected_rule": best_rule, "constrained": bool(use_con), "rule_params": prm,
               "train_oof_macro_f05": tr_mean, "train_oof_singleton_f05": float(tr_vec[truth_n[:n_tr] == 0].mean()),
               "fresh_macro_f05": f_mean, "fresh_singleton_f05": float(f_vec[truth_n[n_tr:] == 0].mean()),
               "minutes": (time.time() - t0) / 60}
        if full:
            tn_f = truth_n[n_tr:]
            pred = Dfr[sf]
            fp = pred[~pred.label.values]
            tp_keys = np.sort((pred.q.values[pred.label.values].astype(np.int64) << 32) | pred.code.values[pred.label.values])
            m_keys = np.sort((Dfr.q.values.astype(np.int64) << 32) | Dfr.code.values)
            fn = fresh_truth[~np.isin(fresh_truth, tp_keys)]
            in_pool, in_m = np.isin(fn, pool_fresh_keys), np.isin(fn, m_keys)
            fn_stage = {"not_retrieved": int((~in_pool).sum()), "pruned": int((in_pool & ~in_m).sum()),
                        "rejected_by_matcher_or_rule": int(in_m.sum())}
            fn_lex = Counter()
            for k in fn[~in_pool].tolist():
                i, c = k >> 32, k & 0xFFFFFFFF
                j = int(np.searchsorted(tcode_s, c))
                nj = len(qn_tok[i] & set(tname[j].split())); aj = len(qa_tok[i] & set(taddr[j].split()))
                fn_lex["no_name_or_addr_token_overlap" if nj == 0 and aj == 0 else
                       "addr_overlap_only" if nj == 0 else "name_overlap_only" if aj == 0 else "both_overlap"] += 1
            fp_cat = Counter()
            for en, ea, nj, aj in zip(fp.exact_name.values, fp.exact_addr.values, fp.name_jw.values, fp.addr_jw.values):
                fp_cat["exact_name_exact_addr" if en and ea else "exact_name_diff_addr" if en else
                       "exact_addr_diff_name" if ea else "name_jw>=0.9" if nj >= 0.9 else "addr_jw>=0.9" if aj >= 0.9 else "fuzzy_other"] += 1
            fp_single = int(np.isin(fp.q.values, np.flatnonzero(truth_n == 0)).sum())
            cty_f = q_cty[n_tr:]
            by_country = {c: float(f_vec[cty_f == ci].mean()) for c, ci in qcountry.items() if (cty_f == ci).any()}
            card = {}
            for lab_, lo_, hi_ in (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6+", 6, 10**6)):
                mm = (tn_f >= lo_) & (tn_f <= hi_)
                if mm.any():
                    card[lab_] = {"n": int(mm.sum()), "f05": float(f_vec[mm].mean())}
            out["fresh_detail"] = {"by_country": by_country, "cardinality": card, "predicted_pairs": int(sf.sum()),
                                   "false_positives": int(len(fp)), "fp_on_singleton_s1": fp_single,
                                   "fp_categories": dict(fp_cat), "fn_by_stage": fn_stage,
                                   "fn_not_retrieved_lexical": dict(fn_lex)}
        return out

    match_keys = [sel_budget] + [k for k in MATCH_EXTRA if k in budgets and k != sel_budget]
    res["matcher_runs"] = {}
    for key in match_keys:
        res["matcher_runs"][key] = run_matcher(key, full=(key == sel_budget))
        (OUT_DIR / "exp6_results.json").write_text(json.dumps(res, indent=2, default=float))
        gc.collect()
    res["meta"]["runtime_min"] = (time.time() - T0) / 60
    res["meta"]["peak_rss_mb_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (OUT_DIR / "exp6_results.json").write_text(json.dumps(res, indent=2, default=float))

    print("\n" + "=" * 110)
    for g in ("train", "fresh"):
        rf = res["retrieval"][g]["pool_F200"]
        print(f"RETRIEVAL ({g}) A+B+F@200+R@5+B2c+B3c+B4c: recall {rf['recall']:.4%} S2 {rf['s2_recall']:.3%} S3 {rf['s3_recall']:.3%} "
              f"oracle {rf['oracle_f05']:.6f} mean {rf['mean']:.1f} p50 {rf['p50']:.0f} p95 {rf['p95']:.0f} p99 {rf['p99']:.0f} max {rf['max']}")
    print(f"PRUNER AUC train-OOF {pinfo['oof_auc_train']:.4f} fresh {pinfo['fresh_auc']:.4f}")
    print(f"{'budget':22s} | {'TRAIN oracle':>12s} {'mean':>6s} {'p99':>5s} | {'FRESH oracle':>12s} {'recall':>8s} {'S2':>7s} {'S3':>7s} {'mean':>6s} {'p50':>4s} {'p95':>4s} {'p99':>4s} {'max':>5s}")
    for k, v in budgets.items():
        t_, f_ = v["train"], v["fresh"]
        print(f"{k:22s} | {t_['oracle_f05']:12.6f} {t_['mean']:6.1f} {t_['p99']:5.0f} | {f_['oracle_f05']:12.6f} {f_['recall']:8.4%} "
              f"{f_['s2_recall']:7.3%} {f_['s3_recall']:7.3%} {f_['mean']:6.1f} {f_['p50']:4.0f} {f_['p95']:4.0f} {f_['p99']:4.0f} {f_['max']:5d}"
              f"{' *' if k == sel_budget else ''}")
    for key, mr in res["matcher_runs"].items():
        print(f"MATCHER[{key}] AUC train-OOF {mr['matcher']['oof_auc_train']:.4f} fresh {mr['matcher']['fresh_auc']:.4f} | "
              f"rule {mr['selected_rule']} con={mr['constrained']} {mr['rule_params']} | TRAIN-OOF F0.5 {mr['train_oof_macro_f05']:.6f} "
              f"| FRESH F0.5 {mr['fresh_macro_f05']:.6f} singleton {mr['fresh_singleton_f05']:.4f} | {mr['minutes']:.1f} min")
        for rk_, rv in mr["set_rules"].items():
            print(f"    {rk_:5s} {rv}")
    print("FRESH detail:", json.dumps(res["matcher_runs"][sel_budget].get("fresh_detail"), default=float))
    log(f"done | results -> {OUT_DIR / 'exp6_results.json'}")


if __name__ == "__main__":
    main()
