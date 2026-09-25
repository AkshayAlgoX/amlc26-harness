"""AMLC26 Experiment 5 — corrected fresh-hash holdout retrieval validation (Kaggle-first).

Replaces experiment 4, whose numbers are INVALID (S2/S3 source bug + reverse retrieval searched only the
holdout, "baseline" was F@100, TF-IDF fitted on the holdout only, B3 capped by holdout-side counts).

What this measures on a fresh deterministic SHA-256 50k S1 holdout (disjoint from every ID list in
EXCLUDE_IDS_DIR):
  BASELINE  = A + B + Forward TF-IDF@200 + Reverse TF-IDF@5
  BASE100   = A + B + Forward TF-IDF@100 + Reverse TF-IDF@5
  PROPOSED  = BASE100 + B2 + B3 (frequency-controlled) + B4 + Canon (non-destructive learned view)
  + leave-one-out marginals of B2, B3, B4, Canon, Reverse@5 from PROPOSED, and B3 uncapped vs capped.
Every configuration is an exact UNIQUE (S1, target) union: recall, S2/S3 recall, oracle macro F0.5,
candidates/S1 (mean/p50/p95/p99/max), US / India / empty-address / cardinality subgroups, links missed.

Leakage safety:
  * source identity only from the entity_id prefix (src/er/ids.py); self-test at start
  * TF-IDF IDF fitted on ALL train S1 of the country (unsupervised, no labels)
  * canonical substitutions mined only from links of S1 entities OUTSIDE the holdout
  * reverse retrieval ranks each S2/S3 record against ALL train S1 of its country
  * frequency caps use GLOBAL target-side key counts (all 10.3M records), never the sample

Environment variables (defaults in brackets):
  DATA_ROOT [/kaggle/input/datasets/akshayalgox/amlc26-2026-official-data-v2]  searched for train_*.tsv
  OUT_DIR [/kaggle/working/exp5]   EXCLUDE_IDS_DIR [<repo>/exclude_ids]   HOLDOUT_N [50000]
  SALT [amlc26-exp5-holdout-v1]     MINE_N [300000]   CAP [200]   REV_DF_CAP [2000]
  FWD_MIN_SIM [0.05]   MAX_DF [0.05]   CHUNK [20000]   REVERSE_MODE [full|truth_only]
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.er.ids import code_source, decode_target_id, encode_target_id, source_of  # noqa: E402
from experiments.agy2_exp4_fresh_holdout_validation import (  # noqa: E402  (same normalisation as exp2-4)
    canonicalize_name, clean_text, extract_address_components, norm_address, norm_name)

ENV = os.environ.get
DATA_ROOT = Path(ENV("DATA_ROOT", "/kaggle/input/datasets/akshayalgox/amlc26-2026-official-data-v2"))
OUT_DIR = Path(ENV("OUT_DIR", "/kaggle/working/exp5"))
EXCLUDE_IDS_DIR = Path(ENV("EXCLUDE_IDS_DIR", str(ROOT / "exclude_ids")))
HOLDOUT_N = int(ENV("HOLDOUT_N", "50000"))
SALT = ENV("SALT", "amlc26-exp5-holdout-v1")
MINE_N = int(ENV("MINE_N", "300000"))
CAP = int(ENV("CAP", "200"))
REV_DF_CAP = int(ENV("REV_DF_CAP", "2000"))
FWD_MIN_SIM = float(ENV("FWD_MIN_SIM", "0.05"))
MAX_DF = float(ENV("MAX_DF", "0.05"))
CHUNK = int(ENV("CHUNK", "20000"))
REVERSE_MODE = ENV("REVERSE_MODE", "full")
K_FWD = int(ENV("K_FWD", "200"))          # forward keep (BASELINE uses all K_FWD)
K_SPLIT = int(ENV("K_SPLIT", "100"))      # forward rank cut used by BASE100/PROPOSED

# channel bits
A, B, F100, F200, R1, R3, R5 = 1, 2, 4, 8, 16, 32, 64
B2U, B2C, B3U, B3C, B4U, B4C, CANON = 128, 256, 512, 1024, 2048, 4096, 8192


def self_test() -> None:
    for eid, src in [("S2-166376419", "S2"), ("S3-166376419", "S3"), ("S2-999999999", "S2"), ("S3-1", "S3")]:
        assert source_of(eid) == src, eid
        assert code_source(encode_target_id(eid)) == src and decode_target_id(encode_target_id(eid)) == eid, eid
    assert encode_target_id("S2-166376419") != encode_target_id("S3-166376419")
    print("self-test PASS: S2/S3 identity is prefix-only; S2-166376419 != S3-166376419")


def find(name: str) -> Path:
    hits = sorted(DATA_ROOT.rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {name} under {DATA_ROOT}, found {hits}")
    return hits[0]


def read_tsv(path: Path):
    """Plain tab split, no quote handling (4 train lines contain '\"'). Yields field lists."""
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            yield line.rstrip("\r\n").split("\t")


def hash_rank(eid: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SALT}|{eid}".encode()).digest()[:8], "big")


def select_holdout(all_ids, excluded: set, n: int):
    pool = [e for e in all_ids if e not in excluded]
    pool.sort(key=hash_rank)
    return pool[:n]


def load_excluded(d: Path) -> set:
    ex = set()
    if d.is_dir():
        for p in sorted(d.glob("*.ids")):
            ids = {l.strip() for l in p.read_text().splitlines() if l.strip()}
            print(f"  exclude list {p.name}: {len(ids):,} ids")
            ex |= ids
    return ex


def oracle_f05(hits_per_s1: np.ndarray, truth_per_s1: np.ndarray) -> np.ndarray:
    """Per-S1 best achievable F0.5 given candidates: 1.25c/(c+0.25t); singletons 1.0; c=0,t>0 -> 0."""
    t = truth_per_s1.astype(float); c = hits_per_s1.astype(float)
    out = np.where(t == 0, 1.0, np.where(c == 0, 0.0, 1.25 * c / np.maximum(c + 0.25 * t, 1e-12)))
    return out


def mine_canon(s1_name: dict, gt: dict, holdout: set, targets_path_list, order_ids) -> dict:
    """Single-token substitutions mined ONLY from links of S1 entities outside the holdout."""
    mine_ids = [e for e in order_ids if e not in holdout and gt.get(e)][:MINE_N]
    assert not (set(mine_ids) & holdout), "canon mining touched the holdout"
    need = {t for e in mine_ids for t in gt[e]}
    tname = {}
    for p in targets_path_list:
        for f in read_tsv(p):
            if f[0] in need:
                tname[f[0]] = norm_name(f[1] if len(f) > 1 else "")
    pc = Counter()
    for e in mine_ids:
        a = set(s1_name[e].split())
        for t in gt[e]:
            b = set(tname.get(t, "").split())
            d1, d2 = a - b, b - a
            if len(d1) == 1 and len(d2) == 1:
                w1, w2 = next(iter(d1)), next(iter(d2))
                if len(w1) >= 2 and len(w2) >= 2:
                    pc[(w1, w2)] += 1
    sub = {}
    for (w1, w2), c in pc.items():
        if c >= 5:
            lo, hi = sorted((w1, w2), key=lambda w: (len(w), w))
            sub[hi] = lo if len(lo) != len(hi) else min(w1, w2)
    print(f"  canon: mined {len(sub):,} substitutions from {len(mine_ids):,} non-holdout S1 ({len(need):,} targets)")
    return sub


def addr_parts(addr_raw: str, country: str, df: Counter):
    comp = extract_address_components(addr_raw, country.upper())
    toks = {t for t in clean_text(addr_raw).split() if len(t) >= 4 and not t.isdigit() and df.get(t, 0) > 0}
    rare = sorted(toks, key=lambda t: (df[t], t))[:2]          # rarest by S1 document frequency
    return comp["pin"], comp["num"], comp["city"], rare


def main():
    t_start = time.time()
    self_test()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p_s1, p_s2, p_s3, p_gt = (find(n) for n in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
                                                "train_ground_truth.tsv"))
    print(f"data: {p_s1.parent}")

    # ---------- S1 (all), ground truth
    s1_ids, s1_name, s1_addr_raw, s1_country = [], {}, {}, {}
    for f in read_tsv(p_s1):
        e = f[0]; s1_ids.append(e)
        s1_name[e] = norm_name(f[1] if len(f) > 1 else "")
        s1_addr_raw[e] = f[2] if len(f) > 2 else ""
        s1_country[e] = (f[3] if len(f) > 3 else "").strip()
    gt = {}
    for f in read_tsv(p_gt):
        gt[f[0].strip()] = [x.strip() for x in (f[1] if len(f) > 1 else "").split(",") if x.strip()]
    print(f"S1 {len(s1_ids):,} | GT rows {len(gt):,} | countries {Counter(s1_country.values())}")

    # ---------- fresh holdout
    excluded = load_excluded(EXCLUDE_IDS_DIR)
    hold = select_holdout(s1_ids, excluded, HOLDOUT_N)
    hold_set = set(hold)
    assert not (hold_set & excluded), "holdout overlaps an excluded sample"
    (OUT_DIR / "holdout_s1.ids").write_text("\n".join(hold) + "\n")
    h_idx = {e: i for i, e in enumerate(hold)}
    n = len(hold)
    truth_codes = [set() for _ in range(n)]
    for e in hold:
        for t in gt.get(e, []):
            truth_codes[h_idx[e]].add(encode_target_id(t))
    truth_n = np.array([len(s) for s in truth_codes])
    truth_keys = np.array(sorted((i << 32) | c for i, s in enumerate(truth_codes) for c in s), dtype=np.int64)
    print(f"holdout {n:,} S1 (excluded {len(excluded):,}) | true links {len(truth_keys):,} | singletons {(truth_n == 0).mean():.4%}")

    # ---------- canonical view (mined outside holdout)
    sub_map = mine_canon(s1_name, gt, hold_set, [p_s2, p_s3], sorted(s1_ids, key=hash_rank))

    # ---------- per-country TF-IDF on ALL train S1 (unsupervised) + address-token DF
    countries = sorted(set(s1_country.values()))
    V, XS1, XS1_revT, XQ_revT, hold_local_map, Xq, q_rows = {}, {}, {}, {}, {}, {}, {}
    addr_df = {}
    for c in countries:
        ids_c = [e for e in s1_ids if s1_country[e] == c]
        texts = [f"{s1_name[e]} {norm_address(s1_addr_raw[e])}".strip() for e in ids_c]
        vec = TfidfVectorizer(max_df=MAX_DF, min_df=2, token_pattern=r"\b[a-zA-Z]{3,}\b", dtype=np.float32)
        X = vec.fit_transform(texts).tocsr()
        df_cols = np.bincount(X.indices, minlength=X.shape[1])
        keep = df_cols <= REV_DF_CAP
        V[c] = (vec, keep)
        XS1[c] = X
        XS1_revT[c] = X[:, keep].T.tocsr()
        hm = np.full(len(ids_c), -1, dtype=np.int64)
        rows = []
        for j, e in enumerate(ids_c):
            if e in hold_set:
                hm[j] = h_idx[e]; rows.append(j)
        hold_local_map[c] = hm
        q_rows[c] = np.array(rows, dtype=np.int64)
        Xq[c] = X[q_rows[c]]
        XQ_revT[c] = XS1_revT[c][:, q_rows[c]].tocsr()
        addr_df[c] = Counter(t for e in ids_c for t in {w for w in clean_text(s1_addr_raw[e]).split()
                                                        if len(w) >= 4 and not w.isdigit()})
        print(f"  {c}: S1 {len(ids_c):,} | vocab {X.shape[1]:,} | reverse vocab {int(keep.sum()):,} | holdout queries {len(rows):,}")
        del texts
    gc.collect()

    # ---------- holdout-side exact / anchor indexes (hash keys)
    idx = {k: defaultdict(list) for k in ("A", "B", "CANON", "B2", "B3", "B4")}
    s1_empty_addr = np.zeros(n, dtype=bool)
    s1_is = {c: np.zeros(n, dtype=bool) for c in countries}
    for e in hold:
        i = h_idx[e]; c = s1_country[e]; s1_is[c][i] = True
        nm, ad = s1_name[e], norm_address(s1_addr_raw[e])
        s1_empty_addr[i] = not s1_addr_raw[e].strip()
        if len(nm) >= 3:
            idx["A"][hash((c, nm))].append((i, 0))
            idx["CANON"][hash((c, canonicalize_name(nm, sub_map)))].append((i, 0))
        if len(ad) >= 5:
            idx["B"][hash((c, ad))].append((i, 0))
        pin, num, city, rare = addr_parts(s1_addr_raw[e], c, addr_df[c])
        if city and num:
            idx["B2"][hash((c, "b2", city, num))].append((i, 0))
        for rt in rare:
            if num:
                idx["B3"][hash((c, "b3", num, rt))].append((i, hash((c, "b3", num, rt, city)) if city else 0))
            if pin:
                idx["B4"][hash((c, "b4", pin, rt))].append((i, hash((c, "b4", pin, rt, num)) if num else 0))

    # ---------- stream all S2 + S3
    rec_s1, rec_t, rec_bits, rec_key, rec_sub = array("q"), array("q"), array("q"), array("q"), array("q")
    key_counts_raw = array("q")                     # global target-side keys for capped channels
    fwd_S = {c: np.full((len(q_rows[c]), K_FWD), -1.0, np.float32) for c in countries}
    fwd_I = {c: np.full((len(q_rows[c]), K_FWD), -1, np.int64) for c in countries}
    fwd_thr = {c: np.full(len(q_rows[c]), -1.0, np.float32) for c in countries}
    target_empty = {}
    truth_targets = {c for s in truth_codes for c in s}
    buf = {c: ([], []) for c in countries}
    stats = Counter()

    def add(i, t, bit, key=0, sub=0):
        rec_s1.append(i); rec_t.append(t); rec_bits.append(bit); rec_key.append(key); rec_sub.append(sub)

    def flush(c):
        texts, codes = buf[c]
        if not texts:
            return
        vec, keep = V[c]
        Xt = vec.transform(texts).tocsr()
        codes_a = np.asarray(codes, dtype=np.int64)
        # forward top-200: holdout queries x chunk
        Fm = (Xq[c] @ Xt.T).tocsr()
        Fm.data[Fm.data < FWD_MIN_SIM] = 0; Fm.eliminate_zeros()
        S, I, thr = fwd_S[c], fwd_I[c], fwd_thr[c]
        for r in np.flatnonzero(np.diff(Fm.indptr)):
            a, b = Fm.indptr[r], Fm.indptr[r + 1]
            v = Fm.data[a:b]; m = v > thr[r]
            if not m.any():
                continue
            cs = np.concatenate([S[r], v[m]]); ci = np.concatenate([I[r], codes_a[Fm.indices[a:b][m]]])
            if len(cs) > K_FWD:
                top = np.argpartition(-cs, K_FWD - 1)[:K_FWD]; cs, ci = cs[top], ci[top]
            S[r], I[r] = cs, ci; thr[r] = cs.min()
        # reverse top-5 over ALL train S1 of the country (only rows that touch some holdout S1 can hit it)
        Xr = Xt[:, keep]
        if REVERSE_MODE == "truth_only":
            rows = np.flatnonzero([cd in truth_targets for cd in codes])
        else:
            touch = (Xr @ XQ_revT[c]).tocsr()
            rows = np.flatnonzero(np.diff(touch.indptr))
        hm = hold_local_map[c]
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
                for rank, j in enumerate(cols[order], 1):
                    hi = hm[j]
                    if hi >= 0:
                        add(int(hi), int(codes_a[r]), R5 | (R3 if rank <= 3 else 0) | (R1 if rank == 1 else 0))
        stats[f"rev_rows_{c}"] += len(rows)
        texts.clear(); codes.clear()

    t_stream = time.time()
    for p in (p_s2, p_s3):
        for f in read_tsv(p):
            tid = f[0]; c = (f[3] if len(f) > 3 else "").strip()
            code = encode_target_id(tid)
            stats["targets"] += 1
            if c not in V:
                stats["targets_unknown_country"] += 1
                continue
            name_raw = f[1] if len(f) > 1 else ""; addr_raw = f[2] if len(f) > 2 else ""
            if code in truth_targets:
                target_empty[code] = not addr_raw.strip()
            nm, ad = norm_name(name_raw), norm_address(addr_raw)
            for i, _ in idx["A"].get(hash((c, nm)), ()):
                add(i, code, A)
            if len(ad) >= 5:
                for i, _ in idx["B"].get(hash((c, ad)), ()):
                    add(i, code, B)
            if len(nm) >= 3:
                for i, _ in idx["CANON"].get(hash((c, canonicalize_name(nm, sub_map))), ()):
                    add(i, code, CANON)
            pin, num, city, rare = addr_parts(addr_raw, c, addr_df[c])
            if city and num:
                k = hash((c, "b2", city, num)); key_counts_raw.append(k)
                for i, _ in idx["B2"].get(k, ()):
                    add(i, code, B2U, k)
            for rt in rare:
                if num:
                    k = hash((c, "b3", num, rt)); ks = hash((c, "b3", num, rt, city)) if city else 0
                    key_counts_raw.append(k)
                    if ks:
                        key_counts_raw.append(ks)
                    for i, s1sub in idx["B3"].get(k, ()):
                        add(i, code, B3U, k, ks if (ks and ks == s1sub) else 0)
                if pin:
                    k = hash((c, "b4", pin, rt)); ks = hash((c, "b4", pin, rt, num)) if num else 0
                    key_counts_raw.append(k)
                    if ks:
                        key_counts_raw.append(ks)
                    for i, s1sub in idx["B4"].get(k, ()):
                        add(i, code, B4U, k, ks if (ks and ks == s1sub) else 0)
            bt, bc = buf[c]; bt.append(f"{nm} {ad}".strip()); bc.append(code)
            if len(bt) >= CHUNK:
                flush(c)
            if stats["targets"] % 1_000_000 == 0:
                el = time.time() - t_stream
                print(f"  {stats['targets']:,} targets | {el/60:.1f} min | records {len(rec_s1):,}", flush=True)
        for c in countries:
            flush(c)
    print(f"stream done: {stats['targets']:,} targets in {(time.time()-t_stream)/60:.1f} min | {dict(stats)}")

    # forward records with rank bits
    for c in countries:
        S, I = fwd_S[c], fwd_I[c]
        for r in range(S.shape[0]):
            ok = I[r] >= 0
            if not ok.any():
                continue
            order = np.argsort(-S[r][ok]); ids = I[r][ok][order]
            hi = int(hold_local_map[c][q_rows[c][r]])
            for rank, code in enumerate(ids, 1):
                add(hi, int(code), F200 | (F100 if rank <= K_SPLIT else 0))

    # ---------- frequency control from GLOBAL target-side counts
    kc = np.frombuffer(key_counts_raw, dtype=np.int64)
    uk, cnt = np.unique(kc, return_counts=True)
    del kc
    def gcount(keys):
        if len(uk) == 0:
            return np.zeros(len(keys), dtype=np.int64)
        pos = np.searchsorted(uk, keys); pos = np.clip(pos, 0, len(uk) - 1)
        return np.where((keys != 0) & (uk[pos] == keys), cnt[pos], 0)
    s1a = np.frombuffer(rec_s1, dtype=np.int64); ta = np.frombuffer(rec_t, dtype=np.int64)
    bits = np.frombuffer(rec_bits, dtype=np.int64).copy()
    ka = np.frombuffer(rec_key, dtype=np.int64); sa = np.frombuffer(rec_sub, dtype=np.int64)
    kcount, scount = gcount(ka), gcount(sa)
    for u, cbit, has_sub in ((B2U, B2C, False), (B3U, B3C, True), (B4U, B4C, True)):
        m = (bits & u) > 0
        keep = kcount <= CAP
        if has_sub:
            keep |= (sa != 0) & (scount > 0) & (scount <= CAP)
        bits[m & keep] |= cbit
    cap_report = {}
    for name_, u in (("B2", B2U), ("B3", B3U), ("B4", B4U)):
        m = (bits & u) > 0
        cap_report[name_] = {"pairs": int(m.sum()), "pairs_in_keys_over_cap": int((m & (kcount > CAP)).sum()),
                             "max_key_target_freq": int(kcount[m].max()) if m.any() else 0}

    # ---------- unique pair union with OR-ed channel bits
    pk = (s1a << 32) | ta
    order = np.argsort(pk, kind="stable"); pk, bits_s = pk[order], bits[order]
    starts = np.flatnonzero(np.r_[True, pk[1:] != pk[:-1]])
    upk = pk[starts]; ubits = np.bitwise_or.reduceat(bits_s, starts)
    us1 = (upk >> 32).astype(np.int64); ut = upk & 0xFFFFFFFF
    is_true = np.isin(upk, truth_keys)
    u_src_s2 = np.array([code_source(int(x)) == "S2" for x in ut[is_true]])
    del s1a, ta, bits, ka, sa, pk, bits_s
    gc.collect()
    tr_i = (truth_keys >> 32).astype(np.int64); tr_t = truth_keys & 0xFFFFFFFF
    tr_s2 = np.array([code_source(int(x)) == "S2" for x in tr_t])
    tr_empty = np.array([target_empty.get(int(x), False) for x in tr_t])

    def evaluate(mask_bits_any: int):
        sel = (ubits & mask_bits_any) > 0          # a pair is in the config if ANY included channel found it
        cands = np.bincount(us1[sel], minlength=n)
        hit = sel & is_true
        hits_s1 = np.bincount(us1[hit], minlength=n)
        orc = oracle_f05(hits_s1, truth_n)
        hit_keys = upk[hit]
        hit_s2 = u_src_s2[np.isin(upk[is_true], hit_keys)]
        tot, s2t = len(truth_keys), int(tr_s2.sum())
        emp = np.isin(truth_keys[tr_empty], hit_keys)
        out = {
            "recall": float(hit.sum() / tot), "s2_recall": float(hit_s2.sum() / s2t),
            "s3_recall": float((len(hit_s2) - hit_s2.sum()) / (tot - s2t)),
            "oracle_f05": float(orc.mean()), "missed_links": int(tot - hit.sum()),
            "cands_mean": float(cands.mean()), "cands_p50": float(np.percentile(cands, 50)),
            "cands_p95": float(np.percentile(cands, 95)), "cands_p99": float(np.percentile(cands, 99)),
            "cands_max": int(cands.max()), "unique_pairs": int(sel.sum()),
            "empty_addr_target_links_recall": float(emp.mean()) if len(emp) else None,
            "empty_addr_s1": {"n": int(s1_empty_addr.sum()), "oracle_f05": float(orc[s1_empty_addr].mean()) if s1_empty_addr.any() else None},
        }
        for c in countries:
            m = s1_is[c]
            tl = int(truth_n[m].sum())
            out[f"country_{c}"] = {"n_s1": int(m.sum()), "recall": float(hits_s1[m].sum() / tl) if tl else None,
                                   "oracle_f05": float(orc[m].mean()), "cands_mean": float(cands[m].mean())}
        card = {}
        for lab, lo, hi in (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5", 5, 5), ("6-11", 6, 99)):
            m = (truth_n >= lo) & (truth_n <= hi)
            if m.any():
                card[lab] = {"n_s1": int(m.sum()), "oracle_f05": float(orc[m].mean())}
        out["cardinality"] = card
        return out

    BASELINE = A | B | F200 | R5
    BASE100 = A | B | F100 | R5
    PROPOSED = BASE100 | B2C | B3C | B4C | CANON
    configs = {
        "BASELINE A+B+F@200+R@5": evaluate(BASELINE),
        "BASE100 A+B+F@100+R@5": evaluate(BASE100),
        "PROPOSED": evaluate(PROPOSED),
        "PROPOSED with B3 uncapped": evaluate(BASE100 | B2C | B3U | B4C | CANON),
        "PROPOSED with R@3": evaluate(A | B | F100 | R3 | B2C | B3C | B4C | CANON),
    }
    marg = {}
    for lab, bit in (("Reverse@5", R5 | R3 | R1), ("B2", B2C), ("B3", B3C), ("B4", B4C), ("Canon", CANON)):
        wo = evaluate(PROPOSED & ~bit)
        marg[lab] = {"recall_gain": configs["PROPOSED"]["recall"] - wo["recall"],
                     "oracle_gain": configs["PROPOSED"]["oracle_f05"] - wo["oracle_f05"],
                     "cands_added_mean": configs["PROPOSED"]["cands_mean"] - wo["cands_mean"]}
    standalone = {lab: evaluate(bit) for lab, bit in (("A", A), ("B", B), ("F@100", F100), ("F@200", F200),
                  ("R@1", R1), ("R@5", R5), ("B2 capped", B2C), ("B3 uncapped", B3U), ("B3 capped", B3C),
                  ("B4 capped", B4C), ("Canon", CANON))}
    result = {
        "meta": {"holdout_n": n, "true_links": int(len(truth_keys)), "salt": SALT, "excluded_ids": len(excluded),
                 "cap": CAP, "max_df": MAX_DF, "k_fwd": K_FWD, "k_split": K_SPLIT, "rev_df_cap": REV_DF_CAP, "fwd_min_sim": FWD_MIN_SIM, "reverse_mode": REVERSE_MODE,
                 "canon_substitutions": len(sub_map), "runtime_min": (time.time() - t_start) / 60,
                 "stats": dict(stats)},
        "configs": configs, "marginals_from_PROPOSED": marg, "standalone": standalone, "cap_report": cap_report,
    }
    (OUT_DIR / "exp5_results.json").write_text(json.dumps(result, indent=2))
    print("\n" + "=" * 100)
    hdr = f"{'config':34s} {'recall':>8s} {'S2':>7s} {'S3':>7s} {'oracle':>8s} {'mean':>6s} {'p50':>5s} {'p95':>5s} {'p99':>5s} {'max':>6s} {'missed':>7s}"
    print(hdr)
    for k, v in configs.items():
        print(f"{k:34s} {v['recall']:8.4%} {v['s2_recall']:7.3%} {v['s3_recall']:7.3%} {v['oracle_f05']:8.6f} "
              f"{v['cands_mean']:6.1f} {v['cands_p50']:5.0f} {v['cands_p95']:5.0f} {v['cands_p99']:5.0f} {v['cands_max']:6d} {v['missed_links']:7d}")
    print("\nmarginals (leave-one-out from PROPOSED):")
    for k, v in marg.items():
        print(f"  {k:10s} recall {v['recall_gain']:+.4%}  oracle {v['oracle_gain']:+.6f}  cands/S1 {v['cands_added_mean']:+.1f}")
    print(f"\ncap report: {cap_report}")
    print(f"REVERSE_MODE={REVERSE_MODE} | runtime {(time.time()-t_start)/60:.1f} min | wrote {OUT_DIR/'exp5_results.json'}")
    if REVERSE_MODE != "full":
        print("WARNING: truth_only reverse -> reverse RECALL exact, candidate BUDGET excludes non-true reverse hits (not a valid budget).")


if __name__ == "__main__":
    main()
