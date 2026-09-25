"""AMLC26 exp9 — evidence-driven matcher features (pure functions, unit-tested), derived from the exp7 H1 FN mining.

Groups (pre-registered in experiments/configs/exp9_prereg.json):
  A rare-token / name-only evidence  : IDF-weighted name overlap (IDF from S2+S3 name-token document frequency),
                                       max shared-token IDF, S1-side IDF coverage, hashed char-3gram name cosine.
      evidence: target address empty -> 76% FN rate; 28% of FN share no name token (needs character evidence).
  B explicit missing-address semantics: address similarities set to NaN when either side is empty, both-missing flag,
                                       name evidence exposed only when the target address is missing.
      evidence: 23.4% of FN have an empty target address.
  C street number / PIN               : has-number flags, number conflict (both present and different), PIN conflict,
                                       S1 digits contained in target digits.
      evidence: 77% of FN have num_match == 0 and 65% have disjoint digits, but absence and conflict are conflated today.
  E within-S1 competition context     : gap of name/address JW to the S1 maximum, name-JW rank, count of strong-name
                                       candidates in the S1.
      evidence: 69% of FN sit at rank 4-10, and the FN rate rises with cardinality (multi-match sets).
Retrieval coverage (flags, channel count, forward/reverse/pruner ranks) is already encoded in exp7 — no group D.
No feature uses labels. Everything is a function of the pair texts and of label-free corpus statistics.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

GROUPS = {
    "A": ["name_idf_wjacc", "name_idf_shared_max", "name_idf_qcov", "name_char3"],
    "B": ["addr_jw_m", "addr_jacc_m", "addr_cont_m", "both_addr_missing", "name_idf_if_taddr_missing"],
    "C": ["q_has_num", "t_has_num", "num_conflict", "pin_conflict", "q_digits_in_t"],
    "E": ["name_jw_gap_s1", "addr_jw_gap_s1", "name_jw_rank_s1", "n_strong_name_s1"],
}
NEW_FEATURES = [f for g in GROUPS.values() for f in g]


def token_df(names) -> Counter:
    """Document frequency of normalised name tokens (one count per record)."""
    df = Counter()
    for n in names:
        df.update(set(n.split()))
    return df


def idf_table(df: Counter, n_docs: int) -> dict:
    return {t: math.log((n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def name_idf_features(qn, tn, idf: dict, default_idf: float):
    """Per pair: IDF-weighted Jaccard, max IDF of a shared token, share of S1 IDF mass matched."""
    n = len(qn)
    wj, mx, cov = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
    for i in range(n):
        a, b = set(qn[i].split()), set(tn[i].split())
        if not a or not b:
            continue
        sh = a & b
        if not sh:
            continue
        w = lambda s: sum(idf.get(t, default_idf) for t in s)
        ws = w(sh)
        wj[i] = ws / w(a | b)
        mx[i] = max(idf.get(t, default_idf) for t in sh)
        cov[i] = ws / w(a)
    return wj, mx, cov


def char3_cosine(q_strings, t_strings, qi, ti, n_features=2 ** 20, chunk=500_000):
    """Cosine of hashed char_wb 3-gram TF vectors — no fitted vocabulary, identical at dev and holdout time."""
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=n_features, alternate_sign=False,
                           norm="l2", dtype=np.float32)
    Xq, Xt = hv.transform(q_strings), hv.transform(t_strings)
    out = np.zeros(len(qi), np.float32)
    for s in range(0, len(qi), chunk):
        out[s:s + chunk] = np.asarray(Xq[qi[s:s + chunk]].multiply(Xt[ti[s:s + chunk]]).sum(1)).ravel()
    return out


def missing_semantics(addr_jw, addr_jacc, addr_cont, q_missing, t_missing, name_idf_wjacc):
    miss = (q_missing.astype(bool)) | (t_missing.astype(bool))
    nan = np.float32(np.nan)
    return (np.where(miss, nan, addr_jw).astype(np.float32), np.where(miss, nan, addr_jacc).astype(np.float32),
            np.where(miss, nan, addr_cont).astype(np.float32),
            (q_missing.astype(bool) & t_missing.astype(bool)).astype(np.int8),
            np.where(t_missing.astype(bool), name_idf_wjacc, nan).astype(np.float32))


def number_features(q_num, t_num, q_pin, t_pin, q_addr, t_addr):
    """q_num/t_num/q_pin/t_pin: extracted street number / postal code strings ('' when absent)."""
    q_num, t_num, q_pin, t_pin = (np.asarray(x, dtype=object) for x in (q_num, t_num, q_pin, t_pin))
    qh, th = q_num != "", t_num != ""
    conflict = qh & th & (q_num != t_num)
    pconf = (q_pin != "") & (t_pin != "") & (q_pin != t_pin)
    qin = np.zeros(len(q_addr), np.int8)
    for i, (a, b) in enumerate(zip(q_addr, t_addr)):
        qd = {t for t in a.split() if t.isdigit()}
        if qd:
            qin[i] = qd <= {t for t in b.split() if t.isdigit()}
    return qh.astype(np.int8), th.astype(np.int8), conflict.astype(np.int8), pconf.astype(np.int8), qin


def s1_context(q, name_jw, addr_jw, strong=0.9):
    """Within-S1 competition context over the S1's candidate set (label-free)."""
    df = pd.DataFrame({"q": q, "n": name_jw, "a": addr_jw})
    g = df.groupby("q", sort=False)
    return ((g.n.transform("max") - df.n).values.astype(np.float32), (g.a.transform("max") - df.a).values.astype(np.float32),
            g.n.rank(ascending=False, method="min").values.astype(np.float32),
            g.n.transform(lambda s: (s >= strong).sum()).values.astype(np.float32))
