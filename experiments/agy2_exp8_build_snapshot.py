"""AMLC26 exp8 — ONE expensive run: C0 normalize -> C1 corpus statistics/TF-IDF -> C2 retrieval SUPERSET (Kaggle).

Contract (identical to exp7 unless stated):
  * production loader (src/er/loader.py) for every TSV; ground truth parsed ONLY for query S1 rows,
  * text statistics (TF-IDF vocab/IDF, reverse DF mask, address-token DF) fitted on train S1 EXCLUDING H2 and H3
    (and any EXP8_SEALED_LISTS) — IDs only, no labels,
  * queries = DEV (H1) + H2 only; H3 S1 are reverse-retrieval competitors (as in exp7) but no H3 row, id or
    reverse hit is ever written to a development artifact,
  * SUPERSET instead of one setting: forward top-300, reverse top-10, capped channels kept up to effective bucket
    count 1000, with per-pair metadata (ranks, scores, effective counts) so any view with k<=300, rev_k<=10,
    cap<=1000 is an exact filter (agy2_exp8_features.apply_view; EXP7_VIEW reproduces exp7),
  * deterministic ties: every top-k is ordered by (score desc, id asc), so sharded == single-process exactly,
  * stable 64-bit blake2b keys for blocking (exp7 used per-process hash(); only collisions can differ).
Sharding: target rows are split into contiguous shards processed by forked workers (BLAS/OpenMP limited to 1 thread
per worker; launch the builder itself with OMP/OPENBLAS/MKL_NUM_THREADS=1 so the forked parent holds no idle BLAS
thread pool — the stage uses scipy.sparse and Python only, no BLAS). Merge is exact: forward top-K of the union == top-K of per-shard top-Ks; reverse top-k and blocking
records are per target row; bucket counts add.
Outputs: c0_norm (S1 open rows + all targets normalized), c1_stats (vocab, IDF, reverse mask, address DF, S1/target
TF-IDF CSR), c2_retrieval (queries, truth, superset pool with cheap features, text sidecars, query-independent reverse
cache over non-sealed S1, global bucket counts).
"""
from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import tempfile
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.agy2_exp4_fresh_holdout_validation import clean_text, extract_address_components, norm_address, norm_name  # noqa: E402
from experiments import agy2_exp8_features as F  # noqa: E402
from experiments import agy2_exp8_store as S  # noqa: E402
from src.er.ids import encode_target_id  # noqa: E402
from src.er.loader import load_source_tsv, parse_id_list  # noqa: E402

ENV = os.environ.get
CODE_FILES = ["experiments/agy2_exp8_build_snapshot.py", "experiments/agy2_exp8_features.py",
              "experiments/agy2_exp4_fresh_holdout_validation.py", "src/er/loader.py", "src/er/ids.py"]
CH_A, CH_B, CH_B2, CH_B3, CH_B4 = 1, 2, 3, 4, 5
TOKEN_PATTERN = r"\b[a-zA-Z]{3,}\b"


def default_params() -> dict:
    R = S.frozen_config()["retrieval"]
    return {"max_df": R["max_df"], "min_df": 2, "rev_df_cap": R["rev_df_cap"], "fwd_min_sim": R["fwd_min_sim"],
            "chunk": R["chunk"], "superset": dict(F.SUPERSET)}


def log(msg):
    F.log(msg)


def skey(*parts) -> int:
    """Stable signed 64-bit key (never 0) — identical in every process and run."""
    v = int.from_bytes(hashlib.blake2b("\x1f".join(parts).encode(), digest_size=8).digest(), "little", signed=True)
    return v or 1


# ------------------------------------------------------------------ worker plumbing (fork; 1 BLAS thread per worker)
_G: dict = {}


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def pmap(fn, jobs, workers):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, len(jobs)), initializer=_init_worker) as pool:
        return pool.map(fn, jobs, chunksize=1)


def ranges(n, parts):
    b = np.linspace(0, n, max(1, parts) + 1).astype(np.int64)
    return [(int(b[i]), int(b[i + 1])) for i in range(len(b) - 1) if b[i + 1] > b[i]]


# ------------------------------------------------------------------ loading
def find(data_root: Path, name: str) -> Path:
    hits = sorted(Path(data_root).rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {name} under {data_root}, found {hits}")
    return hits[0]


def read_truth(path, allowed: set) -> dict:
    """Ground truth for `allowed` S1 only: same read kwargs as src/er/loader.load_ground_truth; rows of any other S1
    (H3 included) are dropped BEFORE their matched-id field is parsed, and never stored."""
    df = pd.read_csv(path, sep="\t", dtype={"source1_entity_id": str, "matched_entity_ids": str},
                     keep_default_na=False, na_values=[""], low_memory=False)
    ids = df["source1_entity_id"].astype(str).str.strip()
    keep = ids.isin(allowed).values
    out = {e: parse_id_list(v) for e, v in zip(ids.values[keep], df["matched_entity_ids"].values[keep])}
    del df
    return out


def _norm_s1_chunk(rng):
    lo, hi = rng
    names, addrs, ctys = _G["s1_name_raw"], _G["s1_addr_raw"], _G["s1_cty"]
    out = []
    for i in range(lo, hi):
        a = addrs[i]
        comp = extract_address_components(a, ctys[i].upper())
        out.append((norm_name(names[i]), norm_address(a), clean_text(a), comp["pin"], comp["num"], comp["city"], not a.strip()))
    return out


# ------------------------------------------------------------------ target shard worker
def _shard(job):
    sid, lo, hi, outdir = job
    G = _G
    codes, names, addrs, ctys = G["t_code"], G["t_name"], G["t_addr"], G["t_cty"]
    V, XQ, XS1_revT, s1_open = G["V"], G["XQ"], G["XS1_revT"], G["s1_open_idx"]
    idx, addr_df, qrows_q = G["idx"], G["addr_df"], G["q_of_qrow"]
    K, RK, CHUNK, FMIN = G["k_fwd"], G["rev_k"], G["chunk"], G["fwd_min_sim"]
    bq, bc, bch, bk, bs = array("q"), array("q"), array("b"), array("q"), array("q")
    keys = array("q")
    rc, rs1, rsc, rrk = array("q"), array("q"), array("f"), array("b")
    fS = {c: np.full((XQ[c].shape[0], K), -1.0, np.float32) for c in V}
    fI = {c: np.full((XQ[c].shape[0], K), -1, np.int64) for c in V}
    fT = {c: np.full(XQ[c].shape[0], -1.0, np.float32) for c in V}
    xparts = {c: [] for c in V}
    norm = []
    buf = {c: ([], []) for c in V}
    stats = Counter()

    def rec(q, code, ch, k=0, s=0):
        bq.append(q); bc.append(code); bch.append(ch); bk.append(k); bs.append(s)

    def flush(c):
        texts, cds = buf[c]
        if not texts:
            return
        vec, keep = V[c]
        Xt = vec.transform(texts).tocsr().astype(np.float32)
        ca = np.asarray(cds, dtype=np.int64)
        xparts[c].append((Xt, ca))
        if XQ[c].shape[0]:                                   # forward: queries x chunk, running top-K per query
            Fm = (XQ[c] @ Xt.T).tocsr()
            Fm.data[Fm.data < FMIN] = 0
            Fm.eliminate_zeros()
            Sc, Ic, thr = fS[c], fI[c], fT[c]
            for r in np.flatnonzero(np.diff(Fm.indptr)):
                a, b = Fm.indptr[r], Fm.indptr[r + 1]
                v = Fm.data[a:b]
                m = v >= thr[r]                               # ties kept; resolved by (score desc, code asc)
                if not m.any():
                    continue
                cs = np.concatenate([Sc[r], v[m]]); ci = np.concatenate([Ic[r], ca[Fm.indices[a:b][m]]])
                o = np.lexsort((ci, -cs))[:K]
                Sc[r], Ic[r] = cs[o], ci[o]
                thr[r] = Sc[r][-1]
        Xr = Xt[:, keep].tocsr()                              # reverse: each target vs ALL S1 of the country
        nz = np.flatnonzero(np.diff(Xr.indptr))
        so = s1_open[c]
        for s0 in range(0, len(nz), 2000):
            rr = nz[s0:s0 + 2000]
            M = (Xr[rr] @ XS1_revT[c]).tocsr()
            for k, r in enumerate(rr):
                a, b = M.indptr[k], M.indptr[k + 1]
                if a == b:
                    continue
                v, cols = M.data[a:b], M.indices[a:b]
                if len(v) > RK:
                    t = np.partition(v, len(v) - RK)[len(v) - RK]
                    sel = v >= t
                    v, cols = v[sel], cols[sel]
                o = np.lexsort((cols, -v))[:RK]
                for rank, j in enumerate(o, 1):
                    oi = so[cols[j]]
                    if oi >= 0:                               # sealed S1 compete for rank but are never written
                        rc.append(int(ca[r])); rs1.append(int(oi)); rsc.append(float(v[j])); rrk.append(rank)
        stats[f"rev_rows_{c}"] += len(nz)
        texts.clear(); cds.clear()

    for i in range(lo, hi):
        code, c, name_raw, addr_raw = int(codes[i]), ctys[i], names[i], addrs[i]
        nm, ad = norm_name(name_raw), norm_address(addr_raw)
        comp = extract_address_components(addr_raw, c.upper())
        pin, num, city = comp["pin"], comp["num"], comp["city"]
        norm.append((code, c, nm, ad, pin, num, city, not addr_raw.strip()))
        stats["targets"] += 1
        if c not in V:
            stats["targets_unknown_country"] += 1
            continue
        for q in idx["A"].get(skey(c, nm), ()):
            rec(q, code, CH_A)
        if len(ad) >= 5:
            for q in idx["B"].get(skey(c, ad), ()):
                rec(q, code, CH_B)
        df = addr_df[c]
        toks = {t for t in clean_text(addr_raw).split() if len(t) >= 4 and not t.isdigit() and df.get(t, 0) > 0}
        rare = sorted(toks, key=lambda t: (df[t], t))[:2]
        if city and num:
            k = skey(c, "b2", city, num); keys.append(k)
            for q in idx["B2"].get(k, ()):
                rec(q, code, CH_B2, k)
        for rt in rare:
            if num:
                k = skey(c, "b3", num, rt); ks = skey(c, "b3", num, rt, city) if city else 0
                keys.append(k)
                if ks:
                    keys.append(ks)
                for q, s1sub in idx["B3"].get(k, ()):
                    rec(q, code, CH_B3, k, ks if (ks and ks == s1sub) else 0)
            if pin:
                k = skey(c, "b4", pin, rt); ks = skey(c, "b4", pin, rt, num) if num else 0
                keys.append(k)
                if ks:
                    keys.append(ks)
                for q, s1sub in idx["B4"].get(k, ()):
                    rec(q, code, CH_B4, k, ks if (ks and ks == s1sub) else 0)
        bt, bcd = buf[c]
        bt.append(f"{nm} {ad}".strip()); bcd.append(code)
        if len(bt) >= CHUNK:
            flush(c)
    for c in V:
        flush(c)

    uk, cnt = np.unique(np.frombuffer(keys, np.int64), return_counts=True) if len(keys) else (np.zeros(0, np.int64), np.zeros(0, np.int64))
    fw = {}
    for c in V:
        ok = fI[c] >= 0
        rows, cols_ = np.nonzero(ok)
        fw[c] = (qrows_q[c][rows], fI[c][ok], fS[c][ok])
    out = Path(outdir) / f"shard{sid:04d}.pkl"
    xp = {c: (sparse.vstack([x for x, _ in xparts[c]]).tocsr() if xparts[c] else None,
              np.concatenate([a for _, a in xparts[c]]) if xparts[c] else np.zeros(0, np.int64)) for c in V}
    with open(out, "wb") as f:
        pickle.dump({"blk": [np.frombuffer(a, t).copy() for a, t in ((bq, np.int64), (bc, np.int64), (bch, np.int8),
                                                                        (bk, np.int64), (bs, np.int64))],
                     "keys": (uk, cnt), "fwd": fw,
                     "rev": (np.frombuffer(rc, np.int64).copy(), np.frombuffer(rs1, np.int64).copy(),
                             np.frombuffer(rsc, np.float32).copy(), np.frombuffer(rrk, np.int8).copy()),
                     "norm": norm, "x": xp, "stats": dict(stats)}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(out)


def _cheap_chunk(rng):
    lo, hi = rng
    G = _G
    return F.cheap_pair_features(G["pq"][lo:hi], G["pt"][lo:hi], G["qname"], G["qaddr"], G["q_parts"],
                                 G["tname"], G["taddr"], G["tpin"], G["tnum"])


# ------------------------------------------------------------------ build
def ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def build_snapshot(data_root, store: S.Store, groups: dict, stat_exclude: set, params: dict = None, workers: int = None,
                   dataset: str = "train", shards_per_worker: int = 2) -> dict:
    """groups: ordered {name: [S1 ids]} (dev store: dev, h2). Returns {"c0": ref, "c1": ref, "c2": ref}."""
    t_all = time.time()
    P = {**default_params(), **(params or {})}
    sup = P["superset"]
    workers = workers or os.cpu_count()
    data_root = Path(data_root)
    code = S.code_sha(*CODE_FILES)
    files = {n: find(data_root, f"{dataset}_{n}.tsv") for n in ("source1", "source2", "source3")}
    gt_path = find(data_root, "train_ground_truth.tsv") if dataset == "train" else None
    data_sha = {n: S.sha256_file(p) for n, p in files.items()}
    sealed = store.forbidden
    Q = [e for g in groups for e in groups[g]]
    assert len(Q) == len(set(Q)), "query groups overlap"
    assert not (set(Q) & sealed), "sealed S1 among queries"
    c0_params = {"dataset": dataset, "data_sha": data_sha, "sealed_sha": ids_sha(sealed)}
    c1_params = {"max_df": P["max_df"], "min_df": P["min_df"], "rev_df_cap": P["rev_df_cap"],
                 "token_pattern": TOKEN_PATTERN, "stat_exclude_sha": ids_sha(stat_exclude)}
    c2_params = {"superset": sup, "fwd_min_sim": P["fwd_min_sim"], "chunk": P["chunk"],
                 "groups": {g: ids_sha(v) for g, v in groups.items()}, "fold": S.fold_cfg()}
    k0 = S.node_key("c0_norm", {}, c0_params, code)
    k1 = S.node_key("c1_stats", {"c0": f"c0_norm/{k0}"}, c1_params, code)
    k2 = S.node_key("c2_retrieval", {"c1": f"c1_stats/{k1}"}, c2_params, code)
    refs = {"c0": f"c0_norm/{k0}", "c1": f"c1_stats/{k1}", "c2": f"c2_retrieval/{k2}"}
    if store.exists("c2_retrieval", k2):
        store.open("c2_retrieval", k2)
        log(f"cache hit {refs['c2']} — nothing to build")
        return refs

    # ---------------- S1 (production loader) + normalization (parallel)
    s1df = load_source_tsv(files["source1"])
    s1_ids = s1df.entity_id.tolist()
    s1_cty = s1df.country.tolist()
    _G.clear()
    _G.update(s1_name_raw=s1df.business_name.tolist(), s1_addr_raw=s1df.business_address.tolist(), s1_cty=s1_cty)
    del s1df
    parts = pmap(_norm_s1_chunk, ranges(len(s1_ids), workers * 4), workers)
    s1n = [r for p in parts for r in p]
    del parts
    s1_name, s1_addr, s1_clean, s1_pin, s1_num, s1_city, s1_amiss = (list(x) for x in zip(*s1n))
    del s1n
    _G.clear()
    pos = {e: i for i, e in enumerate(s1_ids)}
    assert all(e in pos for e in Q), "query S1 missing from source1"
    log(f"S1 loaded + normalized: {len(s1_ids):,} ({workers} workers)")
    countries = sorted(set(s1_cty))
    is_sealed = np.array([e in sealed for e in s1_ids])
    open_rows = np.flatnonzero(~is_sealed)
    open_idx = np.full(len(s1_ids), -1, np.int64); open_idx[open_rows] = np.arange(len(open_rows))

    truth = {}
    if gt_path is not None:
        truth = read_truth(gt_path, set(Q))
    log(f"truth parsed for {len(truth):,} query S1 (other rows dropped unparsed)")

    # ---------------- C1 statistics (fit WITHOUT stat_exclude) + S1 TF-IDF
    fit_ok = np.array([e not in stat_exclude for e in s1_ids])
    V, XS1, XS1_revT, addr_df, rows_c = {}, {}, {}, {}, {}
    for c in countries:
        rc_ = np.flatnonzero(np.array([x == c for x in s1_cty]))
        rows_c[c] = rc_
        texts = [f"{s1_name[i]} {s1_addr[i]}".strip() for i in rc_]
        fm = fit_ok[rc_]
        vec = TfidfVectorizer(max_df=P["max_df"], min_df=P["min_df"], token_pattern=TOKEN_PATTERN, dtype=np.float32)
        vec.fit([t for t, ok in zip(texts, fm) if ok])
        X = vec.transform(texts).tocsr()
        keep = np.bincount(X[fm].indices, minlength=X.shape[1]) <= P["rev_df_cap"]
        V[c] = (vec, keep)
        XS1[c] = X
        XS1_revT[c] = X[:, keep].T.tocsr()
        addr_df[c] = Counter(t for i in rc_[fm] for t in {w for w in s1_clean[i].split() if len(w) >= 4 and not w.isdigit()})
        log(f"{c}: S1 {len(rc_):,} | stats fitted on {int(fm.sum()):,} (excluded {int((~fm).sum()):,}) | vocab {X.shape[1]:,}")
        del texts

    # ---------------- query-side indexes
    nq = len(Q)
    qpos = np.array([pos[e] for e in Q], dtype=np.int64)
    qi_of_row = np.full(len(s1_ids), -1, np.int64); qi_of_row[qpos] = np.arange(nq)
    idx = {k: defaultdict(list) for k in ("A", "B", "B2", "B3", "B4")}
    for q, r in enumerate(qpos.tolist()):
        c, nm, ad = s1_cty[r], s1_name[r], s1_addr[r]
        num, pin, city = s1_num[r], s1_pin[r], s1_city[r]
        if len(nm) >= 3:
            idx["A"][skey(c, nm)].append(q)
        if len(ad) >= 5:
            idx["B"][skey(c, ad)].append(q)
        df = addr_df[c]
        toks = {t for t in s1_clean[r].split() if len(t) >= 4 and not t.isdigit() and df.get(t, 0) > 0}
        rare = sorted(toks, key=lambda t: (df[t], t))[:2]
        if city and num:
            idx["B2"][skey(c, "b2", city, num)].append(q)
        for rt in rare:
            if num:
                idx["B3"][skey(c, "b3", num, rt)].append((q, skey(c, "b3", num, rt, city) if city else 0))
            if pin:
                idx["B4"][skey(c, "b4", pin, rt)].append((q, skey(c, "b4", pin, rt, num) if num else 0))
    idx = {k: dict(v) for k, v in idx.items()}
    XQ, q_of_qrow, s1_open_c = {}, {}, {}
    for c in countries:
        rc_ = rows_c[c]
        qrows = np.flatnonzero(qi_of_row[rc_] >= 0)
        XQ[c] = XS1[c][qrows]
        q_of_qrow[c] = qi_of_row[rc_][qrows]
        s1_open_c[c] = open_idx[rc_]

    # ---------------- stream targets in shards
    tdf = [load_source_tsv(files["source2"]), load_source_tsv(files["source3"])]
    t_code = np.concatenate([np.fromiter((encode_target_id(e) for e in d.entity_id), np.int64, len(d)) for d in tdf])
    assert t_code.min() > 0 and t_code.max() < (1 << 32), "target code does not fit the 32-bit pair key"
    _G.update(t_code=t_code, t_name=np.concatenate([d.business_name.values for d in tdf]),
              t_addr=np.concatenate([d.business_address.values for d in tdf]),
              t_cty=np.concatenate([d.country.values for d in tdf]),
              V=V, XQ=XQ, XS1_revT=XS1_revT, s1_open_idx=s1_open_c, idx=idx, addr_df=addr_df, q_of_qrow=q_of_qrow,
              k_fwd=sup["k_fwd"], rev_k=sup["rev_k"], chunk=P["chunk"], fwd_min_sim=P["fwd_min_sim"])
    del tdf
    nT = len(t_code)
    log(f"targets loaded (production loader): {nT:,}")
    (store.root / ".tmp").mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="exp8_shards_", dir=str(store.root / ".tmp")))
    t_s = time.time()
    try:
        jobs = [(i, lo, hi, str(tmp)) for i, (lo, hi) in enumerate(ranges(nT, workers * shards_per_worker))]
        shard_files = pmap(_shard, jobs, workers)
        log(f"stream done: {nT:,} targets in {(time.time() - t_s) / 60:.2f} min, {len(jobs)} shards / {workers} workers")
        _G.clear()
        stats = Counter()
        blk = [[] for _ in range(5)]
        ukl, cntl, rev = [], [], [[] for _ in range(4)]
        fwd = {c: [] for c in countries}
        xp = {c: [] for c in countries}
        norm = []
        for sf in shard_files:
            with open(sf, "rb") as f:
                d = pickle.load(f)
            for j in range(5):
                blk[j].append(d["blk"][j])
            ukl.append(d["keys"][0]); cntl.append(d["keys"][1])
            for j in range(4):
                rev[j].append(d["rev"][j])
            for c in countries:
                fwd[c].append(d["fwd"][c])
                if d["x"][c][0] is not None:
                    xp[c].append(d["x"][c])
            norm.extend(d["norm"])
            stats.update(d["stats"])
            del d
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bq, bc, bch, bk, bs = (np.concatenate(x) for x in blk)
    uk_all, cnt_all = np.concatenate(ukl), np.concatenate(cntl)
    uk, inv = np.unique(uk_all, return_inverse=True)
    cnt = np.bincount(inv, weights=cnt_all).astype(np.int64)
    del uk_all, cnt_all, inv, blk

    def gcount(keys_):
        if len(uk) == 0:
            return np.zeros(len(keys_), np.int64)
        p_ = np.clip(np.searchsorted(uk, keys_), 0, len(uk) - 1)
        return np.where((keys_ != 0) & (uk[p_] == keys_), cnt[p_], 0)

    INF = F.EFF_NONE
    kcount, scount = gcount(bk), gcount(bs)
    eff = np.where(bch == CH_B2, kcount, np.minimum(kcount, np.where((bs != 0) & (scount > 0), scount, INF)))
    eff = np.where(bch <= CH_B, 0, eff)
    ok = (bch <= CH_B) | (eff <= sup["cap"])
    bq, bc, bch, eff = bq[ok], bc[ok], bch[ok], eff[ok]
    del kcount, scount, bk, bs

    # forward: exact merge of shard top-K lists per query
    fq, fc, fr, fs = [], [], [], []
    K = sup["k_fwd"]
    for c in countries:
        qq = np.concatenate([x[0] for x in fwd[c]]); cc = np.concatenate([x[1] for x in fwd[c]])
        ss = np.concatenate([x[2] for x in fwd[c]])
        o = np.lexsort((cc, -ss, qq))
        qq, cc, ss = qq[o], cc[o], ss[o]
        st = np.r_[0, np.flatnonzero(qq[1:] != qq[:-1]) + 1]
        rank = np.arange(len(qq)) - np.repeat(st, np.diff(np.r_[st, len(qq)])) + 1
        m = rank <= K
        fq.append(qq[m]); fc.append(cc[m]); fr.append(rank[m]); fs.append(ss[m])
    fq, fc, fr, fs = (np.concatenate(x) for x in (fq, fc, fr, fs))
    r_code, r_s1, r_sc, r_rk = (np.concatenate(x) for x in rev)
    o = np.lexsort((r_rk, r_code)); r_code, r_s1, r_sc, r_rk = r_code[o], r_s1[o], r_sc[o], r_rk[o]
    q_of_open = np.full(len(open_rows), -1, np.int64)
    q_of_open[open_idx[qpos]] = np.arange(nq)
    rq = q_of_open[r_s1]
    rm = rq >= 0

    # exact unique pairs with aggregated metadata
    n_b, n_f, n_r = len(bq), len(fq), int(rm.sum())
    pk = np.concatenate([(bq << 32) | bc, (fq << 32) | fc, (rq[rm] << 32) | r_code[rm]])
    fA = np.concatenate([(bch == CH_A).astype(np.int8), np.zeros(n_f + n_r, np.int8)])
    fB = np.concatenate([(bch == CH_B).astype(np.int8), np.zeros(n_f + n_r, np.int8)])
    none = np.full(n_f + n_r, INF, np.int64)
    e2 = np.concatenate([np.where(bch == CH_B2, eff, INF), none])
    e3 = np.concatenate([np.where(bch == CH_B3, eff, INF), none])
    e4 = np.concatenate([np.where(bch == CH_B4, eff, INF), none])
    frk = np.concatenate([np.full(n_b, F.FWD_NONE, np.int64), fr, np.full(n_r, F.FWD_NONE, np.int64)])
    fsc = np.concatenate([np.zeros(n_b, np.float32), fs, np.zeros(n_r, np.float32)])
    rrk = np.concatenate([np.full(n_b + n_f, F.REV_NONE, np.int64), r_rk[rm].astype(np.int64)])
    rsc = np.concatenate([np.zeros(n_b + n_f, np.float32), r_sc[rm]])
    o = np.argsort(pk, kind="stable")
    pk = pk[o]
    st = np.flatnonzero(np.r_[True, pk[1:] != pk[:-1]])
    red = lambda f_, a: f_.reduceat(a[o], st)
    pool = pd.DataFrame({"q": (pk[st] >> 32).astype(np.int32), "code": (pk[st] & 0xFFFFFFFF).astype(np.int64)})
    pool["flag_A"], pool["flag_B"] = red(np.maximum, fA), red(np.maximum, fB)
    pool["fwd_rank"] = red(np.minimum, frk).astype(np.int32); pool["fwd_score"] = red(np.maximum, fsc).astype(np.float32)
    pool["rev_rank"] = red(np.minimum, rrk).astype(np.int16); pool["rev_score"] = red(np.maximum, rsc).astype(np.float32)
    pool["b2_eff"], pool["b3_eff"], pool["b4_eff"] = (red(np.minimum, e).astype(np.int64).clip(max=INF).astype(np.int32)
                                                      for e in (e2, e3, e4))
    upk = pk[st]
    del pk, o, fA, fB, e2, e3, e4, frk, fsc, rrk, rsc, bq, bc, bch, eff
    gc.collect()
    truth_codes = [sorted({encode_target_id(t) for t in truth.get(e, [])}) for e in Q]
    truth_n = np.array([len(x) for x in truth_codes], np.int32)
    truth_keys = np.array(sorted((q << 32) | c for q, cs in enumerate(truth_codes) for c in cs), dtype=np.int64)
    pool["label"] = np.isin(upk, truth_keys)
    pool["is_s2"] = (pool.code.values & 3) == 2
    log(f"pool: {len(pool):,} unique pairs from {n_b:,} blocking + {n_f:,} forward + {n_r:,} reverse records")

    # ---------------- normalized targets (C0) + sidecars
    tn = pd.DataFrame(norm, columns=["code", "country", "name", "addr", "pin", "num", "city", "raw_empty"])
    del norm
    tn = tn.iloc[np.argsort(tn.code.values, kind="stable")].reset_index(drop=True)
    need = np.union1d(np.unique(pool.code.values), np.array(sorted({c for x in truth_codes for c in x}), np.int64))
    first = np.searchsorted(tn.code.values, need)
    first = first[(first < len(tn)) & (tn.code.values[np.clip(first, 0, len(tn) - 1)] == need)]
    tt = tn.iloc[first].reset_index(drop=True)
    pool["t_idx"] = np.searchsorted(tt.code.values, pool.code.values).astype(np.int32)
    q_rows = qpos
    qname = [s1_name[r] for r in q_rows]; qaddr = [s1_addr[r] for r in q_rows]
    q_parts = [(s1_pin[r], s1_num[r]) for r in q_rows]
    qcountry = {c: i for i, c in enumerate(countries)}
    _G.update(pq=pool.q.values, pt=pool.t_idx.values, qname=qname, qaddr=qaddr, q_parts=q_parts,
              tname=tt.name.tolist(), taddr=tt.addr.tolist(), tpin=tt.pin.tolist(), tnum=tt.num.tolist())
    chunks = pmap(_cheap_chunk, ranges(len(pool), workers * 4), workers)
    _G.clear()
    for col in chunks[0]:
        pool[col] = np.concatenate([ch_[col] for ch_ in chunks])
    del chunks
    q_amiss = np.array([s1_amiss[r] for r in q_rows], np.int8)
    q_cty = np.array([qcountry[s1_cty[r]] for r in q_rows], np.int8)
    q_nlen = np.array([len(s) for s in qname]); t_nlen = tt.name.str.len().values
    qv, tv = pool.q.values, pool.t_idx.values
    pool["q_addr_missing"] = q_amiss[qv]
    pool["t_addr_missing"] = tt.raw_empty.values[tv].astype(np.int8)
    pool["country"] = q_cty[qv]
    pool["src_s3"] = (~pool.is_s2).astype(np.int8)
    pool["name_len_diff"] = np.abs(q_nlen[qv] - t_nlen[tv]).astype(np.int16)
    pool = pool[F.SUPERSET_COLS]
    log("cheap features built")

    # ---------------- write C0 / C1 / C2
    grp_of = {e: g for g, ids in groups.items() for e in ids}
    fold, es = S.folds_for(Q)
    queries = pd.DataFrame({"q": np.arange(nq, dtype=np.int32), "s1_id": Q, "group": [grp_of[e] for e in Q],
                            "fold": fold, "es": es, "truth_n": truth_n, "country": q_cty})
    if not store.exists("c0_norm", k0):
        w0 = store.create("c0_norm", {}, c0_params, code, tags=["exp8"])
        s1o = pd.DataFrame({"s1_open": np.arange(len(open_rows), dtype=np.int64), "s1_id": [s1_ids[r] for r in open_rows],
                            "country": [s1_cty[r] for r in open_rows],
                            "name": [s1_name[r] for r in open_rows], "addr": [s1_addr[r] for r in open_rows],
                            "addr_clean": [s1_clean[r] for r in open_rows], "pin": [s1_pin[r] for r in open_rows],
                            "num": [s1_num[r] for r in open_rows], "city": [s1_city[r] for r in open_rows],
                            "addr_missing": [s1_amiss[r] for r in open_rows]})
        w0.put_table("s1_open", s1o)
        w0.put_table("targets", tn)
        w0.extra = {"sealed_rows_not_persisted": int(is_sealed.sum()), "stats": dict(stats)}
        w0.finalize()
    if not store.exists("c1_stats", k1):
        w1 = store.create("c1_stats", {"c0": refs["c0"]}, c1_params, code, tags=["exp8"])
        for c in countries:
            vec, keep = V[c]
            w1.put_json(f"vocab_{c}.json", {t: int(i) for t, i in vec.vocabulary_.items()})
            w1.put_file(f"idf_{c}.npy", writer=lambda p, v=vec: np.save(p, v.idf_.astype(np.float32)))
            w1.put_file(f"revkeep_{c}.npy", writer=lambda p, k_=keep: np.save(p, k_))
            w1.put_table(f"addr_df_{c}", pd.DataFrame({"token": list(addr_df[c]), "df": list(addr_df[c].values())}))
            open_c = s1_open_c[c] >= 0
            w1.put_file(f"X_s1_open_{c}.npz", writer=lambda p, X=XS1[c][np.flatnonzero(open_c)]: sparse.save_npz(p, X, compressed=False))
            w1.put_file(f"X_s1_open_rows_{c}.npy", writer=lambda p, r=s1_open_c[c][open_c]: np.save(p, r))
            if xp[c]:
                w1.put_file(f"X_tgt_{c}.npz", writer=lambda p, c_=c: sparse.save_npz(p, sparse.vstack([x for x, _ in xp[c_]]).tocsr(), compressed=False))
                w1.put_file(f"X_tgt_codes_{c}.npy", writer=lambda p, c_=c: np.save(p, np.concatenate([a for _, a in xp[c_]])))
        w1.put_file("bucket_counts.npz", writer=lambda p: np.savez(p, keys=uk, counts=cnt))
        w1.finalize()
    del xp
    w2 = store.create("c2_retrieval", {"c1": refs["c1"]}, c2_params, code, tags=["exp8"])
    w2.put_table("queries", queries)
    w2.put_table("truth", pd.DataFrame({"q": (truth_keys >> 32).astype(np.int32), "code": truth_keys & 0xFFFFFFFF}))
    w2.put_table("pool", pool)
    w2.put_table("q_text", pd.DataFrame({"q": np.arange(nq, dtype=np.int32), "name": qname, "addr": qaddr,
                                         "pin": [p for p, _ in q_parts], "num": [n for _, n in q_parts]}))
    w2.put_table("t_text", tt[["code", "name", "addr", "pin", "num", "raw_empty"]])
    w2.put_table("rev_cache", pd.DataFrame({"code": r_code, "s1_open": r_s1, "score": r_sc, "rank": r_rk}))
    w2.extra = {"stats": dict(stats), "pairs": int(len(pool)), "positives": int(pool.label.sum()),
                "truth_complete": gt_path is not None, "countries": countries,
                "build_min": round((time.time() - t_all) / 60, 2), "workers": workers}
    node = w2.finalize()
    log(f"C2 written {node.ref} | {len(pool):,} pairs | {(time.time() - t_all) / 60:.2f} min total")
    return refs


def main():
    store = S.Store()
    groups = {"dev": S.group_ids("dev"), "h2": S.group_ids("h2")}
    stat_exclude = set(groups["h2"]) | set(store.forbidden)          # H2 + H3 (+ extra sealed lists)
    data_root = Path(ENV("DATA_ROOT", "/kaggle/input/datasets/akshayalgox/amlc26-2026-official-data-v2"))
    params = json.loads(ENV("EXP8_PARAMS", "{}"))
    workers = int(ENV("WORKERS", str(os.cpu_count())))
    refs = build_snapshot(data_root, store, groups, stat_exclude, params, workers)
    print(json.dumps(refs, indent=2))


if __name__ == "__main__":
    main()
