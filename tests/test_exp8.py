"""exp8 turbo-path invariants — synthetic data only (no official data is read)."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.agy2_exp5_corrected_holdout as X5
import experiments.agy2_exp7_h3 as E7
from experiments import agy2_exp8_build_snapshot as B
from experiments import agy2_exp8_decide as DC
from experiments import agy2_exp8_features as F
from experiments import agy2_exp8_final as FN
from experiments import agy2_exp8_matcher as MT
from experiments import agy2_exp8_pruner as PR
from experiments import agy2_exp8_store as S
from experiments.agy2_exp7_prep_h3 import salted_holdout, sha256_file
from src.er.metric import macro_f05

VIEW7 = {"k_fwd": 12, "rev_k": 5, "cap": 3, "channels": list(F.CHANNELS)}      # exp7-style view on synthetic data
PARAMS = {"max_df": 0.5, "min_df": 2, "rev_df_cap": 400, "fwd_min_sim": 0.05, "chunk": 23,
          "superset": {"k_fwd": 20, "rev_k": 6, "cap": 7}}
TINY = {"learning_rate": 0.3, "num_leaves": 7, "min_data_in_leaf": 5, "max_rounds": 25, "early_stopping": 10}
WORDS = [f"{a}{b}" for a in ("kor", "lum", "sab", "tev", "nar", "pol", "dim", "rax", "vel", "mun", "zor", "hak")
         for b in ("ano", "eli", "ora", "usk", "ith", "ept", "amo", "ivy", "ond", "erk")]
STREETS = ["oak", "maple", "cedar", "pine", "elm", "birch", "willow", "aspen", "laurel", "hazel", "juniper", "spruce"]
US_CITIES = ["springfield", "riverton", "fairview", "georgetown", "salem", "madison", "clinton", "franklin"]
IN_CITIES = ["pune", "nagpur", "indore", "bhopal", "surat", "jaipur"]


# ------------------------------------------------------------------ synthetic dataset
def _entity(rng, country):
    name = " ".join(rng.sample(WORDS, rng.choice([2, 3])))
    num = str(rng.randint(1, 25))
    if country == "US":
        addr = f"{num} {rng.choice(STREETS)} Street, {rng.choice(US_CITIES)}, {rng.randint(10000, 10012)}"
    else:
        addr = f"{num} {rng.choice(STREETS)} Road, {rng.choice(WORDS)} Nagar, {rng.choice(IN_CITIES)} {rng.randint(411000, 411008)}"
    return name.title(), addr


def _noisy(rng, name, addr):
    toks = name.split()
    if len(toks) > 2 and rng.random() < 0.5:
        toks.pop(rng.randrange(len(toks)))
    if rng.random() < 0.4:
        i = rng.randrange(len(toks)); w = toks[i]
        toks[i] = w[:-1] if len(w) > 4 else w
    a = addr
    if rng.random() < 0.3:
        a = a.replace("Street", "St").replace("Road", "Rd")
    if rng.random() < 0.15:
        a = ""
    return " ".join(toks), a


def make_dataset(root: Path, n_s1=420, seed=7):
    rng = random.Random(seed)
    d = root / "data" / "train"
    d.mkdir(parents=True)
    s1, s2, s3, gt = [], [], [], []
    ids2 = iter(rng.sample(range(10, 10 ** 6), 5000)); ids3 = iter(rng.sample(range(10, 10 ** 6), 5000))
    for i in range(n_s1):
        c = "US" if i % 5 < 3 else "India"
        name, addr = _entity(rng, c)
        e = f"S1-{1000 + i * 7}"
        s1.append((e, name, addr, c))
        links = []
        for _ in range(rng.choice([0, 1, 1, 2, 3])):
            src = rng.choice(["S2", "S3"])
            tid = f"{src}-{next(ids2) if src == 'S2' else next(ids3)}"
            n2, a2 = _noisy(rng, name, addr)
            (s2 if src == "S2" else s3).append((tid, n2, a2, c))
            links.append(tid)
        gt.append((e, ",".join(links)))
    for _ in range(n_s1 // 2):                                          # distractors
        c = rng.choice(["US", "India"]); n2, a2 = _entity(rng, c)
        src = rng.choice(["S2", "S3"])
        (s2 if src == "S2" else s3).append((f"{src}-{next(ids2) if src == 'S2' else next(ids3)}", n2, a2, c))
    hdr = "entity_id\tbusiness_name\tbusiness_address\tcountry"
    for fn, rows in (("train_source1.tsv", s1), ("train_source2.tsv", s2), ("train_source3.tsv", s3)):
        (d / fn).write_text(hdr + "\n" + "\n".join("\t".join(r) for r in rows) + "\n")
    (d / "train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\n" + "\n".join(f"{a}\t{b}" for a, b in gt) + "\n")
    ids = [r[0] for r in s1]
    h0 = ids[::11][:30]
    h1 = [e for e in ids if e not in set(h0)][5:125]
    X5.SALT, old = "syn-h2", X5.SALT
    h2 = X5.select_holdout(ids, set(h0) | set(h1), 70)
    X5.SALT = old
    h3 = salted_holdout(ids, set(h0) | set(h1) | set(h2), 70, "syn-h3")
    lists = {}
    for n_, l_ in (("h0", h0), ("dev", h1), ("h2", h2), ("h3", h3)):
        p = root / f"{n_}.ids"; p.write_text("\n".join(l_) + "\n"); lists[n_] = p
    return {"data": root / "data", "ids": {"h0": h0, "dev": h1, "h2": h2, "h3": h3}, "files": lists}


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    root = tmp_path_factory.mktemp("exp8syn")
    ds = make_dataset(root)
    ds["root"] = root
    return ds


def dev_store(syn, name="store"):
    return S.Store(syn["root"] / name, forbidden=set(syn["ids"]["h3"]))


def build(syn, store, workers=1, groups=None, params=None, shards_per_worker=2):
    g = groups or {"dev": syn["ids"]["dev"], "h2": syn["ids"]["h2"]}
    excl = set(syn["ids"]["h2"]) | set(syn["ids"]["h3"])
    return B.build_snapshot(syn["data"], store, g, excl, {**PARAMS, **(params or {})}, workers,
                            shards_per_worker=shards_per_worker)


@pytest.fixture(scope="module")
def built(syn):
    st = dev_store(syn)
    refs = build(syn, st, workers=1)
    return st, refs


def _pool(store, ref, view=VIEW7):
    node = store.open(*ref.split("/"))
    return F.apply_view(node.table("pool"), view, node.man["params"]["superset"]), node


# ------------------------------------------------------------------ 1. snapshot correctness vs the untouched exp7 code
def test_snapshot_matches_exp7_pool(syn, built, tmp_path, monkeypatch):
    cfg = json.loads(S.FROZEN_CONFIG.read_text())
    f = syn["files"]
    cfg["groups"] = {g: {"file": str(f[g]), "sha256": sha256_file(f[g]), "n": len(syn["ids"][g])} for g in ("dev", "h2", "h3")}
    cfg["h2_salt"] = "syn-h2"
    cfg["h3"] = {"salt": "syn-h3", "n": 70, "prior_list_files": {"H0": str(f["h0"]), "H1": str(f["dev"]), "H2": str(f["h2"])}}
    cfg["retrieval"] = {**cfg["retrieval"], "max_df": PARAMS["max_df"], "rev_df_cap": PARAMS["rev_df_cap"],
                        "k_fwd": VIEW7["k_fwd"], "chunk": 17, "cap": VIEW7["cap"], "fwd_min_sim": PARAMS["fwd_min_sim"]}
    cp = tmp_path / "exp7_syn.json"; cp.write_text(json.dumps(cfg))
    captured = {}

    class Stop(Exception):
        pass

    def grab(fold_row, train_row, fit_fn, predict_fn, nfold):
        D = fit_fn.__closure__[fit_fn.__code__.co_freevars.index("D")].cell_contents
        captured["P"] = D.copy()
        raise Stop

    for k, v in (("CONFIG", cp), ("DATA_ROOT", syn["data"]), ("OUT_DIR", tmp_path / "o7"), ("CHECK_CODE_SHA", False)):
        monkeypatch.setattr(E7, k, v)
    monkeypatch.setattr(E7, "crossfit", grab)
    salt = X5.SALT
    try:
        with pytest.raises(Stop):
            E7.main()
    finally:
        X5.SALT = salt
    P7full = captured["P"]
    assert "rev_rank" in P7full and len(P7full)
    st, refs = built
    P8, node = _pool(st, refs["c2"])
    _assert_equal_up_to_ties(syn, P7full, P8, node)


def _reverse_oracle(syn):
    """Independent recomputation of reverse scores (target vs every S1 of its country) for tie detection."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.er.ids import encode_target_id
    from src.er.loader import load_source_tsv
    from experiments.agy2_exp4_fresh_holdout_validation import norm_address, norm_name
    d = syn["data"] / "train"
    s1 = load_source_tsv(d / "train_source1.tsv")
    excl = set(syn["ids"]["h2"]) | set(syn["ids"]["h3"])
    out = {}
    tg = pd.concat([load_source_tsv(d / "train_source2.tsv"), load_source_tsv(d / "train_source3.tsv")])
    for c in sorted(set(s1.country)):
        sc = s1[s1.country == c]
        texts = [f"{norm_name(n)} {norm_address(a)}".strip() for n, a in zip(sc.business_name, sc.business_address)]
        fm = ~sc.entity_id.isin(excl).values
        vec = TfidfVectorizer(max_df=PARAMS["max_df"], min_df=2, token_pattern=B.TOKEN_PATTERN, dtype=np.float32)
        vec.fit([t for t, ok in zip(texts, fm) if ok])
        X = vec.transform(texts).tocsr()
        keep = np.bincount(X[fm].indices, minlength=X.shape[1]) <= PARAMS["rev_df_cap"]
        tc = tg[tg.country == c]
        Xt = vec.transform([f"{norm_name(n)} {norm_address(a)}".strip() for n, a in zip(tc.business_name, tc.business_address)])
        M = (Xt[:, keep] @ X[:, keep].T).toarray()
        for code, row in zip((encode_target_id(e) for e in tc.entity_id), M):
            out[code] = np.sort(row[row > 0])[::-1]
    return out


def _assert_equal_up_to_ties(syn, P7full, P8, node):
    """exp8 == exp7 on DEV/H2 rows except where exp7's arbitrary tie order at a top-k boundary picked another row."""
    P7 = P7full[P7full.g.values < 2]
    a, b = P7.set_index(["q", "code"]), P8.set_index(["q", "code"])
    k7, k8 = set(a.index), set(b.index)
    K, RK = VIEW7["k_fwd"], VIEW7["rev_k"]
    other = ["flag_A", "flag_B", "flag_b2", "flag_b3", "flag_b4"]
    oracle = _reverse_oracle(syn)

    def kth_fwd(D, q):
        s = np.sort(D.loc[q].query("flag_F200 == 1").fwd_tfidf.values)[::-1]
        return s[K - 1] if len(s) >= K else None

    affected = set()
    for (q, c) in (k7 - k8) | (k8 - k7):
        r = a.loc[(q, c)] if (q, c) in k7 else b.loc[(q, c)]
        affected.add(q)
        assert sum(int(r[x]) for x in other) == 0, ("non-tie difference", q, c)
        fwd_tie = r.flag_F200 == 1 and r.flag_R5 == 0 and np.isclose(r.fwd_tfidf, kth_fwd(a, q)) and np.isclose(r.fwd_tfidf, kth_fwd(b, q))
        sc = oracle[c]
        rev_tie = r.flag_R5 == 1 and r.flag_F200 == 0 and r.rev_rank == RK and len(sc) > RK and np.isclose(sc[RK - 1], sc[RK]) \
            and np.isclose(r.rev_tfidf, sc[RK - 1])
        assert fwd_tie or rev_tie, ("difference not explained by a boundary tie", q, c)
    common = sorted(k7 & k8)
    A, Bm = a.loc[common], b.loc[common]
    fwd_blk = ["flag_F100", "flag_F200", "fwd_tfidf", "fwd_rank"]
    rev_blk = ["flag_R1", "flag_R5", "rev_tfidf", "rev_rank"]

    def differs(cols):
        return np.any([~np.isclose(A[c].values.astype(float), Bm[c].values.astype(float), atol=1e-6) for c in cols], axis=0)

    # forward block: a flag/score difference on a shared row must be a boundary tie at the K-th forward score
    for i in np.flatnonzero(differs(["flag_F200"])):
        q, c = common[i]
        s = max(A.fwd_tfidf.values[i], Bm.fwd_tfidf.values[i])
        assert np.isclose(s, kth_fwd(a, q)) and np.isclose(s, kth_fwd(b, q)), ("forward flag differs off-tie", q, c)
        affected.add(q)
    for i in np.flatnonzero(differs(["flag_R5"])):
        q, c = common[i]
        sc = oracle[c]
        assert len(sc) > RK and np.isclose(sc[RK - 1], sc[RK]), ("reverse flag differs off-tie", q, c)
        affected.add(q)
    assert len(affected) < 0.25 * P8.q.nunique()             # synthetic vocab is small, so exact ties are common
    qa = np.array([q for q, _ in common])
    free = ~np.isin(qa, list(affected))
    same_f200 = ~differs(["flag_F200"]); same_r5 = ~differs(["flag_R5"])
    intrinsic = ["label", "is_s2", "flag_A", "flag_B", "flag_b2", "flag_b3", "flag_b4", "exact_name", "exact_addr",
                 "country", "src_s3", "name_jacc", "name_cont", "addr_jacc", "digit_jacc", "num_match", "pin_match",
                 "q_addr_missing", "t_addr_missing", "name_len_diff"]
    for col, m in [(c, np.ones(len(common), bool)) for c in intrinsic] + \
                  [(c, same_f200) for c in ("flag_F100", "flag_F200", "fwd_tfidf")] + \
                  [(c, same_r5) for c in ("flag_R1", "flag_R5", "rev_tfidf")] + \
                  [("num_channels", same_f200 & same_r5)] + \
                  [(c, free) for c in ("cands_s1", "fwd_ratio_best", "name_jacc_rank")]:
        assert np.allclose(A[col].values[m].astype(float), Bm[col].values[m].astype(float), atol=1e-6), col
    # ranks: equal unless the score is tied with another candidate (exp7 orders ties arbitrarily)
    for col, sc, m in (("fwd_rank", "fwd_tfidf", same_f200), ("rev_rank", "rev_tfidf", same_r5)):
        diff = (A[col].values != Bm[col].values) & m
        for i in np.flatnonzero(diff):
            q, c = common[i]
            s = Bm[sc].values[i]
            if col == "fwd_rank":
                assert np.isclose(b.loc[q].fwd_tfidf.values, s).sum() > 1, (col, q, c)
            else:
                assert np.isclose(oracle[c], s).sum() > 1, (col, q, c)
    print(f"exp7 parity: {len(common):,} shared pairs, {len((k7 - k8) | (k8 - k7))} tie-swapped pairs, "
          f"{len(affected)} of {P8.q.nunique()} queries touched by boundary ties")
    q8 = node.table("queries")
    assert (P7.fold.values == q8.fold.values[P7.q.values]).all()


# ------------------------------------------------------------------ 2. sharded == single-process
def test_sharded_equals_single_process(syn, built):
    st1, r1 = built
    st3 = dev_store(syn, "store_sharded")
    r3 = build(syn, st3, workers=3, params={"chunk": 11}, shards_per_worker=3)
    n1, n3 = st1.open(*r1["c2"].split("/")), st3.open(*r3["c2"].split("/"))
    for t in ("queries", "truth", "pool", "q_text", "t_text"):
        a, b = n1.table(t), n3.table(t)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
    ra = n1.table("rev_cache").sort_values(["code", "rank"]).reset_index(drop=True)
    rb = n3.table("rev_cache").sort_values(["code", "rank"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ra, rb, check_dtype=False)
    c1a, c1b = n1.parent("c1"), n3.parent("c1")
    for fn, meta in c1a.man["files"].items():
        if fn.startswith(("idf_", "vocab_", "revkeep_", "bucket_counts")):
            assert c1b.man["files"][fn]["sha256"] == meta["sha256"], fn


# ------------------------------------------------------------------ 3. query-set independence
def test_query_set_independence(syn, built):
    st1, r1 = built
    st2 = dev_store(syn, "store_devonly")
    r2 = build(syn, st2, groups={"dev": syn["ids"]["dev"]})
    a, _ = _pool(st1, r1["c2"]); b, _ = _pool(st2, r2["c2"])
    ndev = len(syn["ids"]["dev"])
    a = a[a.q < ndev].reset_index(drop=True)
    pd.testing.assert_frame_equal(a.drop(columns=["t_idx"]), b.drop(columns=["t_idx"]), check_dtype=False)
    # the reverse cache is query-independent: identical rows whatever the query set
    ra = st1.open(*r1["c2"].split("/")).table("rev_cache").sort_values(["code", "rank"]).reset_index(drop=True)
    rb = st2.open(*r2["c2"].split("/")).table("rev_cache").sort_values(["code", "rank"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ra, rb, check_dtype=False)


# ------------------------------------------------------------------ 4. H3 exclusion
def test_h3_never_enters_development_artifacts(syn, built):
    st, refs = built
    h3 = set(syn["ids"]["h3"])
    c2 = st.open(*refs["c2"].split("/"))
    c0 = c2.parent("c1").parent("c0")
    s1o = c0.table("s1_open")
    assert not (set(s1o.s1_id) & h3)
    assert len(s1o) == 420 - len(h3)                                    # H3 rows are not persisted at all
    rc = c2.table("rev_cache")
    assert rc.s1_open.max() < len(s1o)                                  # every cached reverse hit is an open S1
    q = c2.table("queries")
    assert set(q.group) == {"dev", "h2"} and not (set(q.s1_id) & h3)
    for p in st.root.rglob("*.parquet"):                                # no parquet anywhere carries an H3 id
        cols = S.parquet_columns(p)
        for c in ("s1_id", "entity_id", "source1_entity_id"):
            if c in cols:
                assert not (set(S.read_parquet(p, [c])[c]) & h3), p
    w = st.create("c2_retrieval", {}, {"probe": 1}, {})
    with pytest.raises(S.StoreError):
        w.put_table("queries", pd.DataFrame({"s1_id": [syn["ids"]["h3"][0]], "group": ["dev"]}))
    with pytest.raises(S.StoreError):
        w.put_table("queries", pd.DataFrame({"s1_id": [syn["ids"]["dev"][0]], "group": ["h3"]}))
    w.abort()
    with pytest.raises(AssertionError):
        build(syn, dev_store(syn, "store_bad"), groups={"dev": syn["ids"]["dev"], "h3": syn["ids"]["h3"]})


# ------------------------------------------------------------------ 5. cache invalidation
def test_cache_invalidation(syn, built, tmp_path):
    st, refs = built
    assert build(syn, st) == refs                                        # same inputs -> cache hit, same keys
    k = S.node_key("c3_pruner", {"src": refs["c2"]}, {"a": 1}, {"f": "x"})
    assert k != S.node_key("c3_pruner", {"src": refs["c2"]}, {"a": 2}, {"f": "x"})
    assert k != S.node_key("c3_pruner", {"src": refs["c1"]}, {"a": 1}, {"f": "x"})
    assert k != S.node_key("c3_pruner", {"src": refs["c2"]}, {"a": 1}, {"f": "y"})
    other = dev_store(syn, "store_params")
    r2 = build(syn, other, params={"superset": {"k_fwd": 18, "rev_k": 6, "cap": 7}})
    assert r2["c2"] != refs["c2"] and r2["c1"] == refs["c1"]            # retrieval change -> new C2 only
    # tampering: a changed file or a missing parent is refused
    import shutil
    tam = S.Store(tmp_path / "tam", forbidden=set(syn["ids"]["h3"]))
    shutil.copytree(st.root, tam.root)
    kind, key = refs["c2"].split("/")
    q = tam.path(kind, key) / "queries.parquet"
    q.write_bytes(q.read_bytes() + b"\0")
    with pytest.raises(S.StoreError):
        tam.open(kind, key)
    shutil.rmtree(tam.path(*refs["c1"].split("/")))
    with pytest.raises(S.StoreError):
        tam.open(kind, key, verify_files=False)


# ------------------------------------------------------------------ 6/7. folds + cross-fitted pruner/matcher integrity
@pytest.fixture(scope="module")
def chain(syn, built):
    st, refs = built
    c3 = PR.run_pruner(st, refs["c2"], view=VIEW7, params=TINY, threads=1, deterministic=True)
    c4 = MT.run_mfeat(st, c3, top_n=8, floor=0.0, workers=2)
    c5 = MT.run_matcher(st, c4, params=TINY, threads=1, deterministic=True)
    return st, refs, c3, c4, c5


def test_fold_consistency(syn, built, chain):
    st, refs, c3, c4, c5 = chain
    q = st.open(*refs["c2"].split("/")).table("queries")
    salt, nfold = S.fold_cfg()
    assert (q.fold.values == E7.fold_of(q.s1_id.tolist(), salt, nfold)).all()
    assert (q.es.values == E7.inner_es_flag(q.s1_id.tolist(), salt)).all()
    n3, n5 = st.open(*c3.split("/")), st.open(*c5.split("/"))
    assert list(n3.man["params"]["fold"]) == list(n5.man["params"]["fold"]) == [salt, nfold]


def test_every_row_is_scored_by_its_own_fold_model(syn, built, chain):
    st, refs, c3, c4, c5 = chain
    n2, n3, n4, n5 = (st.open(*r.split("/")) for r in (refs["c2"], c3, c4, c5))
    q = n2.table("queries")
    D, _ = _pool(st, refs["c2"])
    fold = q.fold.values[D.q.values]
    again = F.predict_by_fold(n3.dir, "pruner", D, F.PR_FEATS, fold, S.fold_cfg()[1], 1.0)
    assert np.allclose(again, n3.table("prune").prune_p.values, atol=1e-6)
    M = n4.table("mfeat")
    feats = n5.man["params"]["features"]
    assert "prune_p" in feats and "name_jw" in feats
    again = F.predict_by_fold(n5.dir, "matcher", M, feats, q.fold.values[M.q.values], S.fold_cfg()[1], 1.0)
    assert np.allclose(again, n5.table("scores").p.values, atol=1e-6)


def test_crossfit_label_perturbation(syn, built):
    """Flipping H2 labels changes nothing; flipping DEV fold-k labels never changes fold-k predictions."""
    st, refs = built
    q = st.open(*refs["c2"].split("/")).table("queries")
    D, _ = _pool(st, refs["c2"])
    fold, es, grp = q.fold.values[D.q.values], q.es.values[D.q.values], q.group.values[D.q.values]
    args = dict(prm={**PR.pruner_defaults(), **TINY}, fold=fold, is_dev=grp == "dev", es=es, nfold=5, seed=42,
                threads=1, deterministic=True)
    base, info = F.crossfit_lgb(D, F.PR_FEATS, **args)
    for k, tr in info["train_rows"].items():
        assert (fold[tr] != k).all() and (grp[tr] == "dev").all()
    D2 = D.copy(); m = grp == "h2"; D2.loc[m, "label"] = ~D2.label.values[m]
    assert np.array_equal(F.crossfit_lgb(D2, F.PR_FEATS, **args)[0], base)
    for k in (0, 3):
        D3 = D.copy(); m = (grp == "dev") & (fold == k); D3.loc[m, "label"] = ~D3.label.values[m]
        p3 = F.crossfit_lgb(D3, F.PR_FEATS, **args)[0]
        assert np.array_equal(p3[fold == k], base[fold == k])
    # negative subsampling keeps every positive and is a pure function of the pair
    u1 = F._unif(D.q.values, D.code.values, 42); u2 = F._unif(D.q.values[::-1], D.code.values[::-1], 42)[::-1]
    assert np.array_equal(u1, u2) and 0.3 < (u1 < 0.5).mean() < 0.7


# ------------------------------------------------------------------ decide: exact metric + cache
def test_decide_matches_official_metric(syn, chain):
    st, refs, c3, c4, c5 = chain
    ref, out = DC.run_decide(st, c5, gates=(0.0, 0.4))
    assert out["score_kind"] == "macro_f05" and out["selected_on"].startswith("DEV")
    n5, n2 = st.open(*c5.split("/")), st.open(*refs["c2"].split("/"))
    sc, q, tr = n5.table("scores"), n2.table("queries"), n2.table("truth")
    qids = np.flatnonzero(q.group.values == "dev")
    G = DC.Group(sc[q.group.values[sc.q.values] == "dev"], qids, q.truth_n.values[qids])
    for rule, prm in (("thr", {"t": 0.5}), ("gap", {"t": 0.3, "g": 0.7, "s": 0.4}), ("expF", {"m": 0.2, "s": 0.1})):
        for con in (False, True):
            sel = G.decide(rule, prm)
            sel = G.constrain(sel) if con else sel
            ids = q.s1_id.values[qids]
            pred = {e: set() for e in ids}; truth = {e: set() for e in ids}
            for qq, cc in zip(G.q[sel], G.code[sel]):
                pred[ids[qq]].add(int(cc))
            for qq, cc in zip(tr.q.values, tr.code.values):
                if q.group.values[qq] == "dev":
                    truth[q.s1_id.values[qq]].add(int(cc))
            assert abs(G.score(sel)[0] - macro_f05(truth, pred)) < 1e-12
    assert DC.run_decide(st, c5, gates=(0.0, 0.4))[0] == ref                 # cached


# ------------------------------------------------------------------ v0 seed from an exp6-shaped pool
def test_seed_exp6(syn, tmp_path, monkeypatch):
    dev, h2 = syn["ids"]["dev"], syn["ids"]["h2"]
    monkeypatch.setattr(S, "group_ids", lambda g: {"dev": dev, "h2": h2}[g])
    rng = np.random.default_rng(0)
    q = np.sort(rng.integers(0, len(dev) + len(h2), 3000))
    P = pd.DataFrame({"q": q.astype(np.int32), "code": rng.integers(1, 1 << 30, len(q)) * 4 + 2, "label": rng.random(len(q)) < 0.1,
                      "grp": np.where(q < len(dev), "train", "fresh"), "is_s2": True})
    for c in F.PR_FEATS:
        P[c] = rng.random(len(q)).astype(np.float32)
    P["country"] = (q % 2).astype(np.int8)
    pp = tmp_path / "pairs_pool.parquet"; S.write_parquet(P, pp)
    from experiments import agy2_exp8_seed_exp6 as SD
    st = dev_store(syn, "store_v0")
    ref, timing = SD.seed_exp6(st, pp)
    node = st.open(*ref.split("/"))
    assert S.sha256_file(node.file("pool.parquet")) == S.sha256_file(pp)
    qt = node.table("queries")
    assert (qt.s1_id.values[: len(dev)] == np.array(dev)).all() and (qt.truth_n == -1).all()
    assert qt.pool_hits.sum() == P.label.sum()
    assert "exp6-legacy" in node.man["tags"]
    D = F.load_view(node, SD.EXP6_VIEW)
    assert len(D) == len(P) and "grp" not in D
    assert SD.seed_exp6(st, pp)[1].get("cache_hit")


# ------------------------------------------------------------------ sealed scorer
def test_final_sealed_scoring_and_single_use(syn, chain, tmp_path):
    st, refs, c3, c4, c5 = chain
    c6, _ = DC.run_decide(st, c5)
    cfg = FN.freeze(st, c6, "h3", syn["files"]["h3"], tmp_path / "sealed", sealed_files=[syn["files"]["h3"]])
    # the dev store's forbidden set is H3; the test store root differs from STORE_ROOT
    out = FN.score_sealed(cfg, syn["data"], workers=1)
    assert 0.0 <= out["macro_f05"] <= 1.0
    assert abs(out["macro_f05_official_check"] - out["macro_f05"]) < 1e-9
    with pytest.raises(S.StoreError):
        FN.score_sealed(cfg, syn["data"], workers=1)                     # second look refused
    bad = dict(cfg, holdout={**cfg["holdout"], "name": "dev"})
    with pytest.raises(S.StoreError):
        FN.score_sealed(bad, syn["data"], workers=1, allow_reuse=True)
