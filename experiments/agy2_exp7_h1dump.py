"""AMLC26 exp7 H1 DUMP — byte-for-byte copy of agy2_exp7_h3.py (frozen config, same seed) that additionally writes the
H1 (DEV) cross-fitted OOF matcher pairs for error mining. Only DEV rows are dumped and only the DEV report is computed;
H2/H3 stay in the run solely so text statistics and pools are identical to the frozen exp7 run.

Original docstring:
AMLC26 Experiment 7 — pre-registered, cross-fitted pipeline scored on an untouched H3 holdout (Kaggle-first).

Fixes relative to exp6 (audit):
  * parsing  : the production loader (src/er/loader.py) is the ONE reader — no plain tab split anywhere.
  * cross-fit: ONE fixed S1 fold assignment (SHA-256, FOLD_SALT) shared by pruner AND matcher. For fold k the pruner
               and the matcher train only on DEV S1 with fold != k and score only rows with fold == k, so every
               model-derived training feature (prune_p, prune_rank) is out-of-fold for its own S1 group.
  * scoring  : an evaluation S1 is scored by the single chain (pruner_k -> matcher_k) of its own fold, so evaluation
               scores have exactly the distribution of DEV out-of-fold scores (no fold averaging).
  * no calibration: set rules run on raw OOF scores.
  * H3       : new salted holdout, disjoint from H0/H1/H2; every choice is frozen in a JSON config before scoring.
Groups: DEV = H1 (all fitting/selection), H2 and H3 = evaluation only, treated identically (H2 = comparison).
Text statistics (IDF, address DF, reverse DF mask) are fitted on train S1 excluding H2 and H3.
Outputs (OUT_DIR): exp7_results.json.
"""
from __future__ import annotations

import gc
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiments.agy2_exp5_corrected_holdout as X5  # noqa: E402  (addr_parts, oracle_f05, select_holdout)
from experiments.agy2_exp4_fresh_holdout_validation import clean_text, norm_address, norm_name  # noqa: E402
from experiments.agy2_exp6_integrated import jaro_winkler_pairs, jw_self_test, per_s1_eval, JW_ENGINE  # noqa: E402
from experiments.agy2_exp7_prep_h3 import salted_holdout, sha256_file  # noqa: E402
from src.er.ids import encode_target_id  # noqa: E402
from src.er.loader import load_ground_truth_df, load_source_tsv  # noqa: E402

ENV = os.environ.get
CONFIG = Path(ENV("EXP7_CONFIG", str(ROOT / "experiments" / "configs" / "exp7_h3_frozen.json")))
DATA_ROOT = Path(ENV("DATA_ROOT", "/kaggle/input/datasets/akshayalgox/amlc26-2026-official-data-v2"))
OUT_DIR = Path(ENV("OUT_DIR", "/kaggle/working/exp7"))
CHECK_CODE_SHA = False          # this copy differs from the frozen file by the dump only

A, B, F100, F200, R1, R3, R5 = 1, 2, 4, 8, 16, 32, 64
B2U, B2C, B3U, B3C, B4U, B4C = 128, 256, 512, 1024, 2048, 4096
BITS = {"A": A, "B": B, "F100": F100, "F200": F200, "R5": R5, "B2C": B2C, "B3C": B3C, "B4C": B4C}
PR_FEATS = ["fwd_tfidf", "fwd_rank", "rev_tfidf", "rev_rank", "num_channels", "flag_A", "flag_B", "flag_F100",
            "flag_R1", "flag_R5", "flag_b2", "flag_b3", "flag_b4", "exact_name", "exact_addr", "country", "src_s3",
            "name_jacc", "name_cont", "addr_jacc", "digit_jacc", "num_match", "pin_match", "q_addr_missing",
            "t_addr_missing", "name_len_diff", "cands_s1", "fwd_ratio_best", "name_jacc_rank"]
M_FEATS = PR_FEATS + ["prune_p", "prune_rank", "name_jw", "addr_jw", "name_cont_q", "name_cont_t", "name_len_ratio",
                      "tok_count_diff", "addr_cont", "exact_both", "strong_n_strong_a", "strong_n_weak_a",
                      "weak_n_strong_a", "name_x_addr"]
GROUPS = ("dev", "h2", "h3")
T0 = time.time()
warnings.filterwarnings("ignore", message="Overriding the parameters from Reference Dataset")


def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[{(time.time() - T0) / 60:6.1f} min | peak {rss:7.0f} MB] {msg}", flush=True)


# ------------------------------------------------------------------ leakage-critical helpers (unit-tested)
def _h(salt: str, e: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}|{e}".encode()).digest()[:8], "big")


def fold_of(ids, salt: str, nfold: int) -> np.ndarray:
    """ONE fixed fold id per S1 (pure function of the id) — shared by the pruner and the matcher."""
    return np.array([_h(salt, e) % nfold for e in ids], dtype=np.int8)


def inner_es_flag(ids, salt: str) -> np.ndarray:
    """Early-stopping subset marker (~10% of S1); only ever applied inside a model's own training folds."""
    return np.array([(_h(salt + "|es", e) % 10) == 0 for e in ids], dtype=bool)


def crossfit(fold_row: np.ndarray, train_row: np.ndarray, fit_fn, predict_fn, nfold: int) -> np.ndarray:
    """Model k trains on rows with train_row & fold != k and scores exactly the rows with fold == k.
    Evaluation rows (train_row False) are therefore scored by the single model of their own fold."""
    pred = np.full(len(fold_row), np.nan)
    for k in range(nfold):
        tr = np.flatnonzero(train_row & (fold_row != k))
        te = np.flatnonzero(fold_row == k)
        model = fit_fn(tr, k)
        if len(te):
            pred[te] = predict_fn(model, te)
    assert not np.isnan(pred).any(), "crossfit left rows unscored"
    return pred


def stat_fit_mask(ids, eval_ids: set) -> np.ndarray:
    """Rows allowed to contribute text statistics (IDF / DF): everything except evaluation S1."""
    return np.array([e not in eval_ids for e in ids], dtype=bool)


def read_ids(p) -> list:
    ids = [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]
    assert len(ids) == len(set(ids)), f"duplicate ids in {p}"
    return ids


def main():
    cfg = json.loads(CONFIG.read_text())
    code_sha = sha256_file(Path(__file__))
    if CHECK_CODE_SHA:
        assert code_sha == cfg["code_sha256"]["experiments/agy2_exp7_h3.py"], "exp7 code differs from the frozen config"
    R, PRN, MAT, RUL = cfg["retrieval"], cfg["pruner"], cfg["matcher"], cfg["set_rules"]
    assert PRN["features"] == PR_FEATS and MAT["features"] == M_FEATS, "feature lists differ from frozen config"
    NFOLD, FOLD_SALT = cfg["folds"]["nfold"], cfg["folds"]["salt"]
    TOPN, FLOOR = cfg["budget"]["top_n"], cfg["budget"]["floor"]
    POOL = 0
    for ch in R["pool_channels"]:
        POOL |= BITS[ch]
    X5.self_test(); jw_self_test()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {"config_file": CONFIG.name, "config_sha256": sha256_file(CONFIG), "code_sha256": code_sha, "meta": {"jw_engine": JW_ENGINE}}

    # ---------------- canonical load (production loader)
    def find(name):
        hits = sorted(DATA_ROOT.rglob(name))
        assert len(hits) == 1, f"expected one {name}"
        return hits[0]
    s1df = load_source_tsv(find("train_source1.tsv"))
    s1_ids = s1df.entity_id.tolist()
    s1_name = dict(zip(s1_ids, (norm_name(x) for x in s1df.business_name)))
    s1_addr_raw = dict(zip(s1_ids, s1df.business_address))
    s1_country = dict(zip(s1_ids, s1df.country))
    del s1df
    gtdf = load_ground_truth_df(find("train_ground_truth.tsv"))
    gt = dict(zip(gtdf.source1_entity_id, gtdf.matched_list))
    del gtdf
    log(f"loaded S1 {len(s1_ids):,} | GT rows {len(gt):,} (production loader)")

    # ---------------- groups (all verified; H3 re-derived and compared with the frozen list)
    G = cfg["groups"]
    ids_dev, ids_h2, ids_h3 = (read_ids(ROOT / G[g]["file"]) for g in GROUPS)
    for g, ids in zip(GROUPS, (ids_dev, ids_h2, ids_h3)):
        assert sha256_file(ROOT / G[g]["file"]) == G[g]["sha256"], f"{g} list differs from frozen config"
        assert len(ids) == G[g]["n"]
    prior = {n: set(read_ids(ROOT / f)) for n, f in cfg["h3"]["prior_list_files"].items()}
    s1_set = set(s1_ids)
    assert all(set(x) <= s1_set for x in (ids_dev, ids_h2, ids_h3))
    assert not (set(ids_dev) & set(ids_h2)) and not (set(ids_dev) & set(ids_h3)) and not (set(ids_h2) & set(ids_h3))
    assert all(not (set(ids_h3) & v) for v in prior.values()), "H3 overlaps a prior list"
    X5.SALT = cfg["h2_salt"]
    assert X5.select_holdout(s1_ids, prior["H0"] | prior["H1"], len(ids_h2)) == ids_h2, "H2 re-derivation mismatch"
    assert salted_holdout(s1_ids, set().union(*prior.values()), cfg["h3"]["n"], cfg["h3"]["salt"]) == ids_h3, "H3 re-derivation mismatch"
    log("groups verified: DEV=H1, H2, H3 pairwise disjoint; H2 and H3 re-derived exactly; H3 disjoint from H0/H1/H2")

    Q = ids_dev + ids_h2 + ids_h3
    nq = len(Q)
    bounds = {"dev": (0, len(ids_dev)), "h2": (len(ids_dev), len(ids_dev) + len(ids_h2)), "h3": (len(ids_dev) + len(ids_h2), nq)}
    grp_q = np.zeros(nq, np.int8); grp_q[bounds["h2"][0]:bounds["h2"][1]] = 1; grp_q[bounds["h3"][0]:] = 2
    qi = {e: i for i, e in enumerate(Q)}
    eval_set = set(ids_h2) | set(ids_h3)
    fold_q = fold_of(Q, FOLD_SALT, NFOLD)
    es_q = inner_es_flag(Q, FOLD_SALT)
    truth = [set(encode_target_id(t) for t in gt.get(e, [])) for e in Q]
    truth_n = np.array([len(s) for s in truth])
    truth_keys = np.array(sorted((i << 32) | c for i, s in enumerate(truth) for c in s), dtype=np.int64)
    all_truth_codes = {c for s in truth for c in s}
    for g in GROUPS:
        lo, hi = bounds[g]
        res["meta"][f"{g}_s1"] = hi - lo
        res["meta"][f"{g}_links"] = int(truth_n[lo:hi].sum())
        res["meta"][f"{g}_singleton_rate"] = float((truth_n[lo:hi] == 0).mean())
    del gt

    # ---------------- per-country TF-IDF, address DF and reverse DF mask: fitted WITHOUT H2/H3
    countries = sorted(set(s1_country.values()))
    V, XS1_revT, XQ, XQ_revT, hold_map, q_rows, addr_df = {}, {}, {}, {}, {}, {}, {}
    for c in countries:
        ids_c = [e for e in s1_ids if s1_country[e] == c]
        texts = [f"{s1_name[e]} {norm_address(s1_addr_raw[e])}".strip() for e in ids_c]
        fitm = stat_fit_mask(ids_c, eval_set)
        vec = TfidfVectorizer(max_df=R["max_df"], min_df=2, token_pattern=r"\b[a-zA-Z]{3,}\b", dtype=np.float32)
        vec.fit([t for t, ok in zip(texts, fitm) if ok])
        X = vec.transform(texts).tocsr()
        keep = np.bincount(X[fitm].indices, minlength=X.shape[1]) <= R["rev_df_cap"]
        V[c] = (vec, keep)
        XS1_revT[c] = X[:, keep].T.tocsr()
        hm = np.full(len(ids_c), -1, dtype=np.int64)
        rows = []
        for j, e in enumerate(ids_c):
            if e in qi:
                hm[j] = qi[e]; rows.append(j)
        hold_map[c], q_rows[c] = hm, np.array(rows, dtype=np.int64)
        XQ[c] = X[q_rows[c]]
        XQ_revT[c] = XS1_revT[c][:, q_rows[c]].tocsr()
        addr_df[c] = Counter(t for e, ok in zip(ids_c, fitm) if ok
                             for t in {w for w in clean_text(s1_addr_raw[e]).split() if len(w) >= 4 and not w.isdigit()})
        log(f"{c}: S1 {len(ids_c):,} | stats fitted on {int(fitm.sum()):,} (H2/H3 excluded) | vocab {X.shape[1]:,} | queries {len(rows):,}")
        del texts
    gc.collect()

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

    # ---------------- stream all S2+S3 (production loader frames)
    K_FWD, CHUNK, CAP = R["k_fwd"], R["chunk"], R["cap"]
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
        Fm.data[Fm.data < R["fwd_min_sim"]] = 0; Fm.eliminate_zeros()
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
        texts.clear(); codes.clear()

    tgt_frames = [load_source_tsv(find("train_source2.tsv")), load_source_tsv(find("train_source3.tsv"))]
    log(f"targets loaded (production loader): {sum(len(f) for f in tgt_frames):,}")
    t_s = time.time()
    tcodes_all = []
    for df in tgt_frames:
        codes_df = np.fromiter((encode_target_id(e) for e in df.entity_id), dtype=np.int64, count=len(df))
        tcodes_all.append(codes_df)
        for code, name_raw, addr_raw, c in zip(codes_df.tolist(), df.business_name, df.business_address, df.country):
            stats["targets"] += 1
            if c not in V:
                stats["targets_unknown_country"] += 1
                continue
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
            order = np.argsort(-S[r][ok]); ids_, scs = I[r][ok][order], S[r][ok][order]
            hi = int(hold_map[c][q_rows[c][r]])
            for rank, (code, sc) in enumerate(zip(ids_, scs), 1):
                rq.append(hi); rt_.append(int(code)); rb.append(F200 | (F100 if rank <= 100 else 0))
                rk.append(0); rs.append(0); rfs.append(float(sc)); rrs.append(0.0); rrk.append(rank)
    del fS, fI, fT
    gc.collect()

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
    P["g"] = grp_q[P.q.values]
    P["fold"] = fold_q[P.q.values]
    P["is_s2"] = (P.code.values & 3) == 2
    for name_, bit in (("flag_A", A), ("flag_B", B), ("flag_F100", F100), ("flag_F200", F200), ("flag_R1", R1),
                       ("flag_R5", R5), ("flag_b2", B2C), ("flag_b3", B3C), ("flag_b4", B4C)):
        P[name_] = ((ub & bit) > 0).astype(np.int8)
    P["num_channels"] = P[["flag_A", "flag_B", "flag_F200", "flag_R5", "flag_b2", "flag_b3", "flag_b4"]].sum(axis=1).astype(np.int8)
    P["fwd_tfidf"], P["fwd_rank"], P["rev_tfidf"], P["rev_rank"] = ufs, ufr.astype(np.int32), urs, urr.astype(np.int16)
    del upk, ub, ufr, urr, ufs, urs
    gc.collect()
    log(f"pool: {len(P):,} unique pairs")

    # ---------------- candidate + truth target texts (from the same loader frames)
    need = np.union1d(np.unique(P.code.values), np.fromiter(all_truth_codes, dtype=np.int64))
    tname, taddr, tpin, tnum, traw_empty, tcode = [], [], [], [], [], []
    for df, codes_df in zip(tgt_frames, tcodes_all):
        sel = np.flatnonzero(np.isin(codes_df, need))
        for j in sel.tolist():
            araw, c = df.business_address.iat[j], df.country.iat[j]
            tcode.append(int(codes_df[j])); tname.append(norm_name(df.business_name.iat[j])); taddr.append(norm_address(araw))
            traw_empty.append(not araw.strip())
            pin, num, _, _ = X5.addr_parts(araw, c, addr_df[c]) if c in addr_df else ("", "", "", [])
            tpin.append(pin); tnum.append(num)
    del tgt_frames, tcodes_all, need
    order_t = np.argsort(np.array(tcode, dtype=np.int64))
    tcode_s = np.array(tcode, dtype=np.int64)[order_t]
    tname = [tname[i] for i in order_t]; taddr = [taddr[i] for i in order_t]
    traw_empty = np.array(traw_empty)[order_t]; tpin = [tpin[i] for i in order_t]; tnum = [tnum[i] for i in order_t]
    del tcode, order_t
    gc.collect()
    log(f"target texts: {len(tcode_s):,}")
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
    g_ = P.groupby("q")
    P["cands_s1"] = g_.q.transform("size").astype(np.int32)
    P["fwd_ratio_best"] = (P.fwd_tfidf / g_.fwd_tfidf.transform("max").replace(0, np.nan)).fillna(0).astype(np.float32)
    P["name_jacc_rank"] = g_.name_jacc.rank(ascending=False, method="min").astype(np.float32)
    del f_nj, f_nc, f_aj, f_dj, f_en, f_ea, f_num, f_pin, g_
    gc.collect()
    log("cheap features built")
    res["timing_min"] = {"retrieval_and_features": (time.time() - T0) / 60}

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    def lgb_crossfit(D, feats, prm, tag):
        """Cross-fitted LightGBM: DEV rows of folds != k train model k; all rows of fold k are scored by model k."""
        X = np.empty((len(D), len(feats)), dtype=np.float32)       # column-wise fill: no float64 intermediate
        for j, f_ in enumerate(feats):
            X[:, j] = D[f_].values
        y = D.label.values.astype(np.int8)
        es = es_q[D.q.values]
        params = dict(objective="binary", learning_rate=prm["learning_rate"], num_leaves=prm["num_leaves"],
                      min_data_in_leaf=prm["min_data_in_leaf"], feature_fraction=prm["feature_fraction"],
                      bagging_fraction=prm["bagging_fraction"], bagging_freq=1, lambda_l2=prm["lambda_l2"],
                      max_bin=prm["max_bin"], seed=cfg["seed"], num_threads=os.cpu_count(), verbose=-1)
        imps = np.zeros(len(feats)); iters = []

        def fit(tr, k):
            assert (D.g.values[tr] == 0).all() and (D.fold.values[tr] != k).all()      # DEV only, other folds only
            e = es[tr]
            dtr = lgb.Dataset(X[tr[~e]], y[tr[~e]])
            dva = lgb.Dataset(X[tr[e]], y[tr[e]], reference=dtr)
            m = lgb.train(params, dtr, prm["max_rounds"], valid_sets=[dva],
                          callbacks=[lgb.early_stopping(prm["early_stopping"], verbose=False)])
            imps[:] += m.feature_importance("gain"); iters.append(m.best_iteration)
            log(f"{tag} fold {k}: trained on {len(tr):,} DEV rows, best_iter {m.best_iteration}")
            return m

        pred = crossfit(D.fold.values, D.g.values == 0, fit, lambda m, te: m.predict(X[te], num_iteration=m.best_iteration), NFOLD)
        info = {"best_iters": iters, "top_features": [f for _, f in sorted(zip(imps, feats), reverse=True)[:12]]}
        for gi, g in ((0, "dev"),):                   # DEV only: no H2/H3 labels read
            mk = D.g.values == gi
            info[f"auc_{g}"] = float(roc_auc_score(y[mk], pred[mk]))
        del X
        return pred, info

    # ---------------- stage 2: cross-fitted pruner
    t_p = time.time()
    P["prune_p"], pinfo = lgb_crossfit(P, PR_FEATS, PRN, "pruner")
    P.sort_values(["q", "prune_p"], ascending=[True, False], inplace=True, kind="stable")
    P["prune_rank"] = (P.groupby("q").cumcount() + 1).astype(np.int32)
    res["pruner"] = pinfo
    res["timing_min"]["pruner"] = (time.time() - t_p) / 60
    log(f"pruner {pinfo}")
    keep_budget = (P.prune_rank.values <= TOPN) & (P.prune_p.values >= FLOOR)

    # ---------------- stage 4: cross-fitted matcher on the FIXED pre-registered budget
    t_m = time.time()
    M = P[keep_budget].copy()
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
    log(f"matcher pairs {len(M):,} (DEV {int((M.g == 0).sum()):,}) | JW {JW_ENGINE}")
    M["p"], minfo = lgb_crossfit(M, M_FEATS, MAT, "matcher")
    res["matcher"] = minfo
    res["timing_min"]["matcher"] = (time.time() - t_m) / 60

    # ---------------- stage 5/6: set rules on RAW DEV OOF scores (no calibration), frozen grids
    def prep(D):
        D = D.sort_values(["q", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
        gg = D.groupby("q", sort=False)
        D["r"] = gg.cumcount().values + 1
        D["mx"] = gg.p.transform("max").values
        D["cs"] = gg.p.cumsum().values
        D["tot"] = gg.p.transform("sum").values
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
        idx_sel = np.flatnonzero(sel)
        o_ = np.lexsort((-D.p.values[idx_sel], D.code.values[idx_sel]))
        cs_ = D.code.values[idx_sel][o_]
        first = np.r_[True, cs_[1:] != cs_[:-1]]
        out = np.zeros(len(D), bool); out[idx_sel[o_][first]] = True
        return out

    def f05(D, sel, g):
        lo, hi = bounds[g]
        return per_s1_eval(D.q.values - lo, D.label.values, sel, truth_n[lo:hi], hi - lo)

    Dg = {g: prep(M[M.g.values == gi]) for gi, g in enumerate(GROUPS)}
    dev_hits = np.bincount(Dg["dev"].q.values[Dg["dev"].label.values], minlength=bounds["dev"][1])
    m_dev = float((truth_n[:bounds["dev"][1]] - dev_hits).mean())         # expected unseen links, DEV only
    grids = {"thr": [{"t": t} for t in RUL["thr_t"]],
             "topn": [{"n": n_, "t": t} for n_ in RUL["topn_n"] for t in RUL["topn_t"]],
             "gap": [{"t": t, "g": g2} for t in RUL["gap_t"] for g2 in RUL["gap_g"]],
             "expF": [{"m": mm, "s": s} for mm in (0.0, m_dev) for s in RUL["expF_s"]]}
    rules = {}
    for rule, grid in grids.items():
        best = None
        for prm in grid:
            for con in (False, True):
                sel = decide(Dg["dev"], rule, prm)
                sel = constrain(Dg["dev"], sel) if con else sel
                sc = float(f05(Dg["dev"], sel, "dev").mean())
                if best is None or sc > best[0]:
                    best = (sc, prm, con)
        rules[rule] = {"dev_oof_f05": best[0], "params": {k: float(v) for k, v in best[1].items()}, "constrained": best[2]}
        log(f"rule {rule}: {rules[rule]}")
    sel_rule = max(rules, key=lambda k: rules[k]["dev_oof_f05"])
    S = rules[sel_rule]
    res["set_rules_dev"] = rules
    res["selected"] = {"budget": f"top{TOPN}_floor{FLOOR}", "rule": sel_rule, **S, "m_dev": m_dev,
                       "selected_on": "DEV (H1) cross-fitted OOF only"}
    log(f"FROZEN DECISION (DEV only): {res['selected']}")

    # ---------------- stage 7: identical reports for DEV-OOF, H2, H3
    pool_keys = np.sort((P.q.values.astype(np.int64) << 32) | P.code.values)
    budget_keys = np.sort((P.q.values[keep_budget].astype(np.int64) << 32) | P.code.values[keep_budget])

    def cand_stats(q_rows_, lo, hi):
        c = np.bincount(q_rows_ - lo, minlength=hi - lo)
        return {"mean": float(c.mean()), "p50": float(np.percentile(c, 50)), "p95": float(np.percentile(c, 95)),
                "p99": float(np.percentile(c, 99)), "max": int(c.max())}

    def report(g):
        lo, hi = bounds[g]
        gi = GROUPS.index(g)
        tn = truth_n[lo:hi]
        gtruth = truth_keys[((truth_keys >> 32) >= lo) & ((truth_keys >> 32) < hi)]
        g_is_s2 = (gtruth & 3) == 2
        pm = P.g.values == gi
        in_pool = np.isin(gtruth, pool_keys); in_budget = np.isin(gtruth, budget_keys)
        D = Dg[g]
        sel = decide(D, sel_rule, S["params"])
        sel = constrain(D, sel) if S["constrained"] else sel
        f = f05(D, sel, g)
        pred_keys = np.sort((D.q.values[sel].astype(np.int64) << 32) | D.code.values[sel])
        tp_mask = np.isin(gtruth, pred_keys)
        n_pred, n_tp = int(sel.sum()), int(tp_mask.sum())
        pred_s2 = (D.code.values[sel] & 3) == 2
        tp_pred = D.label.values[sel]
        hits_pool = np.bincount(P.q.values[pm & P.label.values] - lo, minlength=hi - lo)
        hits_budget = np.bincount(P.q.values[pm & keep_budget & P.label.values] - lo, minlength=hi - lo)
        cty = q_cty[lo:hi]
        card = {}
        for lab_, a_, b_ in (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6+", 6, 10**6)):
            mm = (tn >= a_) & (tn <= b_)
            if mm.any():
                card[lab_] = {"n": int(mm.sum()), "f05": float(f[mm].mean())}
        fp = D[sel & ~D.label.values]
        fp_cat = Counter()
        for en, ea, nj, aj in zip(fp.exact_name.values, fp.exact_addr.values, fp.name_jw.values, fp.addr_jw.values):
            fp_cat["exact_name_exact_addr" if en and ea else "exact_name_diff_addr" if en else "exact_addr_diff_name" if ea
                   else "name_jw>=0.9" if nj >= 0.9 else "addr_jw>=0.9" if aj >= 0.9 else "fuzzy_other"] += 1
        return {
            "s1": hi - lo, "true_links": int(len(gtruth)), "singleton_s1": int((tn == 0).sum()),
            "retrieval": {"recall": float(in_pool.mean()), "s2_recall": float(in_pool[g_is_s2].mean()),
                          "s3_recall": float(in_pool[~g_is_s2].mean()),
                          "oracle_f05": float(X5.oracle_f05(hits_pool, tn).mean()), **cand_stats(P.q.values[pm], lo, hi)},
            "pruned": {"budget": f"top{TOPN}_floor{FLOOR}", "recall": float(in_budget.mean()),
                       "s2_recall": float(in_budget[g_is_s2].mean()), "s3_recall": float(in_budget[~g_is_s2].mean()),
                       "oracle_f05": float(X5.oracle_f05(hits_budget, tn).mean()),
                       **cand_stats(P.q.values[pm & keep_budget], lo, hi)},
            "matcher": {"macro_f05": float(f.mean()), "precision": n_tp / max(n_pred, 1), "recall": n_tp / len(gtruth),
                        "predicted_pairs": n_pred, "tp": n_tp, "fp": n_pred - n_tp,
                        "s2": {"precision": float(tp_pred[pred_s2].mean()) if pred_s2.any() else None,
                               "recall": float(tp_mask[g_is_s2].mean())},
                        "s3": {"precision": float(tp_pred[~pred_s2].mean()) if (~pred_s2).any() else None,
                               "recall": float(tp_mask[~g_is_s2].mean())},
                        "singleton_f05": float(f[tn == 0].mean()),
                        "fp_on_singleton_s1_pairs": int(np.isin(fp.q.values - lo, np.flatnonzero(tn == 0)).sum()),
                        "by_country": {c: float(f[cty == ci].mean()) for c, ci in qcountry.items() if (cty == ci).any()},
                        "cardinality": card, "fp_categories": dict(fp_cat)},
            "fn_by_stage": {"retrieval": int((~in_pool).sum()), "pruning": int((in_pool & ~in_budget).sum()),
                            "matching": int((in_budget & ~tp_mask).sum())},
        }

    res["reports"] = {"dev": report("dev")}                       # H2/H3 not re-scored
    Dd = Dg["dev"]
    sel_d = decide(Dd, sel_rule, S["params"])
    sel_d = constrain(Dd, sel_d) if S["constrained"] else sel_d
    dump = Dd[["q", "code", "label", "p", "fold", "r", "mx"] + [f_ for f_ in M_FEATS if f_ not in ("prune_p", "prune_rank")]
              + ["prune_p", "prune_rank"]].copy()
    dump["selected"] = sel_d
    dump["s1_id"] = np.array(Q, dtype=object)[dump.q.values]
    for c_ in dump.columns:
        if dump[c_].dtype == np.float64:
            dump[c_] = dump[c_].astype(np.float32)
    def _write(df, name):              # parquet on Kaggle; pickle only where pyarrow is absent (local smoke)
        try:
            df.to_parquet(OUT_DIR / name, index=False)
        except ImportError:
            df.to_pickle(OUT_DIR / name.replace(".parquet", ".pkl"))
    _write(dump, "h1_matcher_oof.parquet")
    lo_, hi_ = bounds["dev"]
    _write(pd.DataFrame({"q": np.arange(lo_, hi_), "s1_id": Q[lo_:hi_], "truth_n": truth_n[lo_:hi_], "fold": fold_q[lo_:hi_],
                  "country": q_cty[lo_:hi_]}), "h1_queries.parquet")
    dk = truth_keys[(truth_keys >> 32) < hi_]
    _write(pd.DataFrame({"q": (dk >> 32).astype(np.int32), "code": dk & 0xFFFFFFFF,
                         "in_pool": np.isin(dk, pool_keys), "in_budget": np.isin(dk, budget_keys)}), "h1_truth.parquet")
    log(f"H1 dump: {len(dump):,} matcher pairs, {int(dump.selected.sum()):,} selected, {len(dk):,} true links")
    res["meta"]["runtime_min"] = (time.time() - T0) / 60
    res["meta"]["peak_rss_mb_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (OUT_DIR / "exp7_h1dump_results.json").write_text(json.dumps(res, indent=2, default=float))

    print("\n" + "=" * 110)
    print(f"FROZEN: budget top{TOPN}_floor{FLOOR} | rule {sel_rule} {S['params']} constrained={S['constrained']} (chosen on DEV OOF {S['dev_oof_f05']:.6f})")
    print(f"pruner AUC dev {pinfo['auc_dev']:.4f} | matcher AUC dev {minfo['auc_dev']:.4f}")
    for g in ("dev",):
        r = res["reports"][g]
        print(f"--- {g.upper()} {'(DEV cross-fitted OOF)' if g == 'dev' else '(evaluation only)'}")
        print(json.dumps(r, default=float))
    log(f"done | results -> {OUT_DIR / 'exp7_results.json'}")


if __name__ == "__main__":
    main()
