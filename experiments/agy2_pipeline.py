"""AMLC26 production pipeline (H4 validation + test inference) — ONE code path for the validated and the submitted model.

Derived by anchored edits from experiments/agy2_exp9_h4_eval.py (exp7 retrieval/features + exp9 B2 features), which is
kept unchanged as the parity reference (tests/test_pipeline.py). Modes (PIPELINE_MODE):
  h4   : train pruner + matcher on H1 (cross-fitted), select the expected-F0.5 rule on H1 OOF only, score sealed H4 ONCE,
         and export the fold models + decision + C2/C4/C5 artifacts with a sha256 manifest (the production model).
  dev  : like h4 but H1 only (H4 is never queried, scored or read beyond its IDs for the statistics exclusion); used for
         every post-freeze development experiment (optional retrieval lanes, optional extra training S1).
  test : score one deterministic shard (country x SHA-256 bucket of s1_id) of test_source1 with the exported fold models
         (each S1 scored by the pruner_k -> matcher_k chain of its own fold, exactly as H4 evaluated); writes shard scores.
         The global expected-F0.5 decision + one-to-one conflict resolution + TSV writing is agy2_pipeline_merge.py.
Fail-closed: the config is read only from experiments/configs/h4_frozen.json (tests may override with PIPELINE_SMOKE=1)
and every pinned module sha256 must match; test mode additionally requires the export manifest to match this code.
Country: deterministic training encoding India=0, US=1 (identical to exp7/exp9 for every train/H1/H4 row); any other
country (test: France) is NaN = "unknown". France performance is NOT known — there are no France training labels.
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
CANONICAL_CONFIG = ROOT / "experiments" / "configs" / "h4_p2_frozen.json"
MODE = ENV("PIPELINE_MODE", "h4")
SMOKE = ENV("PIPELINE_SMOKE", "0") == "1"                      # synthetic tests only
# h4 / test read ONLY the canonical frozen config; dev experiments (and synthetic tests) may pass their own config
CONFIG = Path(ENV("PIPELINE_CONFIG", str(CANONICAL_CONFIG))) if (SMOKE or MODE == "dev") else CANONICAL_CONFIG
MODELS_DIR = Path(ENV("MODELS_DIR", "")) if ENV("MODELS_DIR") else None
SHARD_COUNTRY = ENV("SHARD_COUNTRY", "")
SHARD_INDEX, SHARD_COUNT = int(ENV("SHARD_INDEX", "0")), int(ENV("SHARD_COUNT", "1"))
SHARD_SALT = "amlc26-test-shard-v1"
COUNTRY_CODE = {"India": 0.0, "US": 1.0}   # == exp7/exp9 enumerate(sorted({"India", "US"})); unseen country -> NaN
ALLOWED_DROP = {"country"}                   # B2nc = drop_features ["country"] from BOTH pruner and matcher
FOLD_CONTRACT = ("amlc26-exp7-fold-v1", 5)
BUDGETS_ALLOWED = (40, 80)
MAX_EXTRA_TRAIN = 50000                      # H1 50k + extra 50k + H4 50k = 150k queries = the exp7 memory envelope
from experiments import agy2_exp9_features as F9  # noqa: E402
from experiments.agy2_lane_romanization import romanize, is_non_ascii
from experiments.agy2_x2_features import R_FEATURES, compute_r_features  # noqa: E402
from experiments import agy2_exp8_store as ST  # noqa: E402  (parquet I/O; S is the selected rule below)
DATA_ROOT = Path(ENV("DATA_ROOT", "/kaggle/input/datasets/akshayalgox/amlc26-2026-official-data-v2"))
OUT_DIR = Path(ENV("OUT_DIR", "/kaggle/working/pipeline"))

A, B, F100, F200, R1, R3, R5 = 1, 2, 4, 8, 16, 32, 64
F40, RROM5 = 4096 * 2, 4096 * 4
B2U, B2C, B3U, B3C, B4U, B4C = 128, 256, 512, 1024, 2048, 4096
BITS = {"A": A, "B": B, "F100": F100, "F40": F40, "F200": F200, "R5": R5, "RROM5": RROM5, "B2C": B2C, "B3C": B3C, "B4C": B4C}
PR_FEATS = ["fwd_tfidf", "fwd_rank", "rev_tfidf", "rev_rank", "num_channels", "flag_A", "flag_B", "flag_F100",
            "flag_R1", "flag_R5", "flag_b2", "flag_b3", "flag_b4", "exact_name", "exact_addr", "country", "src_s3",
            "name_jacc", "name_cont", "addr_jacc", "digit_jacc", "num_match", "pin_match", "q_addr_missing",
            "t_addr_missing", "name_len_diff", "cands_s1", "fwd_ratio_best", "name_jacc_rank"]
M_FEATS = PR_FEATS + ["prune_p", "prune_rank", "name_jw", "addr_jw", "name_cont_q", "name_cont_t", "name_len_ratio",
                      "tok_count_diff", "addr_cont", "exact_both", "strong_n_strong_a", "strong_n_weak_a",
                      "weak_n_strong_a", "name_x_addr"]
GROUPS = ("dev", "h4")
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


def shard_of(e: str, count: int) -> int:
    """Deterministic test shard of an S1 id (pure function of the id)."""
    return _h(SHARD_SALT, e) % count


def country_code(c: str) -> float:
    return COUNTRY_CODE.get(c, float("nan"))


def write_manifest(d: Path, extra: dict) -> dict:
    files = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(d.iterdir())
             if p.is_file() and p.name != "manifest.json"}
    man = {**extra, "files": files}
    (d / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    return man


def verify_export(d: Path, code: dict, cfg_sha: str) -> dict:
    """Test mode: exported models must be byte-identical to the manifest and produced by this exact code + config."""
    man = json.loads((d / "manifest.json").read_text())
    for name, meta in man["files"].items():
        if sha256_file(d / name) != meta["sha256"]:
            raise SystemExit(f"FAIL-CLOSED: export file {name} changed")
    if man["code_sha256"] != code or man["config_sha256"] != cfg_sha:
        raise SystemExit("FAIL-CLOSED: export was produced by different code/config")
    return man


def read_ids(p) -> list:
    ids = [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]
    assert len(ids) == len(set(ids)), f"duplicate ids in {p}"
    return ids


# ------------------------------------------------------------------ decision layer (exp7 logic; shared with the merge step)
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


def main():
    cfg = json.loads(CONFIG.read_text())
    code_sha = sha256_file(Path(__file__))
    if MODE not in ("h4", "test", "dev"):
        raise SystemExit(f"unknown PIPELINE_MODE {MODE}")
    if cfg["code_sha256"].get("experiments/agy2_pipeline.py") != code_sha:
        raise SystemExit("FAIL-CLOSED: experiments/agy2_pipeline.py differs from the frozen config")
    for rel, want in cfg["code_sha256"].items():               # every pinned module must match
        if sha256_file(ROOT / rel) != want:
            raise SystemExit(f"FAIL-CLOSED: {rel} differs from the frozen config")
    cfg_sha = sha256_file(CONFIG)
    R, PRN, MAT, RUL = cfg["retrieval"], cfg["pruner"], cfg["matcher"], cfg["set_rules"]
    EXTRA = cfg["candidate"]["extra_features"]
    assert set(EXTRA) <= set(F9.NEW_FEATURES)
    DROP = list(cfg.get("drop_features", []))
    if not set(DROP) <= ALLOWED_DROP:
        raise SystemExit(f"FAIL-CLOSED: unsupported drop_features {DROP}")
    LANES = list(cfg.get("lanes", []))
    if False:
        raise SystemExit("FAIL-CLOSED: retrieval lanes are development-only until validated (dev mode)")
    LANE_FEATS = [f for ln in LANES for f in (f"flag_lane_{ln['name']}", f"lane_{ln['name']}_score", f"lane_{ln['name']}_rank")]
    PR_RUN = [f for f in PR_FEATS + LANE_FEATS if f not in DROP]                     # pruner features
    M_RUN = [f for f in PR_FEATS + LANE_FEATS + M_FEATS[len(PR_FEATS):] + EXTRA + R_FEATURES if f not in DROP]   # == M_FEATS + EXTRA + R
    if PRN["features"] != PR_RUN or MAT["features"] != M_RUN:
        raise SystemExit("FAIL-CLOSED: pruner/matcher feature lists differ from the frozen config")
    if (cfg["folds"]["salt"], cfg["folds"]["nfold"]) != FOLD_CONTRACT:
        raise SystemExit("FAIL-CLOSED: fold configuration differs from the exp7 contract")
    if cfg["budget"]["top_n"] not in BUDGETS_ALLOWED and not SMOKE:
        raise SystemExit("FAIL-CLOSED: budget must be Top-40 or Top-80")
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
    split = "test" if MODE == "test" else "train"
    s1df = load_source_tsv(find(f"{split}_source1.tsv"))
    s1_ids = s1df.entity_id.tolist()
    s1_name = dict(zip(s1_ids, (norm_name(x) for x in s1df.business_name)))
    s1_addr_raw = dict(zip(s1_ids, s1df.business_address))
    s1_country = dict(zip(s1_ids, s1df.country))
    s1_rawname = dict(zip(s1_ids, s1df.business_name))          # B2R: raw names for romanization
    del s1df
    if MODE == "test":
        gt = {}                                                   # no labels exist for the test set
    else:
        gtdf = load_ground_truth_df(find("train_ground_truth.tsv"))
        gt = dict(zip(gtdf.source1_entity_id, gtdf.matched_list))
        del gtdf
    log(f"loaded S1 {len(s1_ids):,} | GT rows {len(gt):,} (production loader)")

    # ---------------- groups (all verified; H3 re-derived and compared with the frozen list)
    if MODE == "test":
        assert SHARD_COUNTRY and 0 <= SHARD_INDEX < SHARD_COUNT, "test mode needs SHARD_COUNTRY/SHARD_INDEX/SHARD_COUNT"
        ids_h4 = [e for e in s1_ids if s1_country[e] == SHARD_COUNTRY and shard_of(e, SHARD_COUNT) == SHARD_INDEX]
        ids_dev = []
        Q = ids_h4
        nq = len(Q)
        bounds = {"dev": (0, 0), "h4": (0, nq)}
        grp_q = np.ones(nq, np.int8)
        qi = {e: i for i, e in enumerate(Q)}
        eval_set = set()                                          # test statistics: all test S1 of the country
        log(f"test shard {SHARD_COUNTRY} {SHARD_INDEX}/{SHARD_COUNT}: {nq:,} S1")
    else:
        G = cfg["groups"]
        ids_dev, ids_h4 = (read_ids(ROOT / G[g]["file"]) for g in GROUPS)
        for g, ids in zip(GROUPS, (ids_dev, ids_h4)):
            assert sha256_file(ROOT / G[g]["file"]) == G[g]["sha256"], f"{g} list differs from frozen config"
            assert len(ids) == G[g]["n"]
        prior = {n: set(read_ids(ROOT / f)) for n, f in cfg["h4"]["prior_list_files"].items()}
        s1_set = set(s1_ids)
        assert set(ids_dev) <= s1_set and set(ids_h4) <= s1_set and not (set(ids_dev) & set(ids_h4))
        assert all(not (set(ids_h4) & v) for v in prior.values()), "H4 overlaps a prior list"
        assert salted_holdout(s1_ids, set().union(*prior.values()), cfg["h4"]["n"], cfg["h4"]["salt"]) == ids_h4, "H4 re-derivation mismatch"
        log("groups verified: DEV=H1, H4 disjoint from H0-H3 and re-derived exactly")
        if cfg.get("extra_dev"):                              # optional extra TRAINING S1, disjoint from every holdout
            xd = cfg["extra_dev"]
            if set(xd) != {"n", "salt"} or not isinstance(xd["n"], int) or xd["n"] <= 0 or not xd["salt"]:
                raise SystemExit("FAIL-CLOSED: malformed extra_dev")
            if xd["n"] > MAX_EXTRA_TRAIN and not SMOKE:
                raise SystemExit("FAIL-CLOSED: extra_dev above the validated memory envelope")
            extra = salted_holdout(s1_ids, set(ids_dev) | set(ids_h4) | set().union(*prior.values()), xd["n"], xd["salt"])
            assert not (set(extra) & (set(ids_h4) | set().union(*prior.values())))
            ids_dev = ids_dev + extra
            res["meta"]["extra_train_s1"] = len(extra)
            res["meta"]["extra_train_sha256"] = hashlib.sha256("\n".join(extra).encode()).hexdigest()
            log(f"extra training S1: {len(extra):,} (salt {xd['salt']})")
        Q = ids_dev + (ids_h4 if MODE == "h4" else [])
        nq = len(Q)
        bounds = {"dev": (0, len(ids_dev)), "h4": (len(ids_dev), nq)}
        grp_q = np.zeros(nq, np.int8); grp_q[bounds["h4"][0]:] = 1
        qi = {e: i for i, e in enumerate(Q)}
        eval_set = set().union(*(prior[n] for n in cfg["stats_exclude"] if n in prior))
        if "H4" in cfg["stats_exclude"]:
            eval_set |= set(ids_h4)
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
    countries = [SHARD_COUNTRY] if MODE == "test" else sorted(set(s1_country.values()))
    V, XS1_revT, XQ, XQ_revT, hold_map, q_rows, addr_df = {}, {}, {}, {}, {}, {}, {}
    V_rom, XS1_rom_revT, XQ_rom_revT = {}, {}, {}
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
        
        # Romanization lane
        rom_texts = [romanize(str(s1_rawname[e])) for e in ids_c]
        vec_rom = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_df=0.4, sublinear_tf=True, dtype=np.float32)
        vec_rom.fit([t for t, ok in zip(rom_texts, fitm) if ok])
        Xr_s1 = vec_rom.transform(rom_texts).tocsr()
        V_rom[c] = vec_rom
        XS1_rom_revT[c] = Xr_s1.T.tocsr()
        
        hm = np.full(len(ids_c), -1, dtype=np.int64)
        rows = []
        for j, e in enumerate(ids_c):
            if e in qi:
                hm[j] = qi[e]; rows.append(j)
        hold_map[c], q_rows[c] = hm, np.array(rows, dtype=np.int64)
        XQ[c] = X[q_rows[c]]
        XQ_revT[c] = XS1_revT[c][:, q_rows[c]].tocsr()
        XQ_rom_revT[c] = XS1_rom_revT[c][:, q_rows[c]].tocsr()
        addr_df[c] = Counter(t for e, ok in zip(ids_c, fitm) if ok
                             for t in {w for w in clean_text(s1_addr_raw[e]).split() if len(w) >= 4 and not w.isdigit()})
        log(f"{c}: S1 {len(ids_c):,} | stats fitted on {int(fitm.sum()):,} (stats_exclude groups excluded) | vocab {X.shape[1]:,} | queries {len(rows):,}")
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
    buf = {c: ([], [], []) for c in countries}
    stats = Counter()

    def add(i, t, bit, key=0, sub=0, rsc=0.0, rrank=0):
        rq.append(i); rt_.append(t); rb.append(bit); rk.append(key); rs.append(sub)
        rfs.append(0.0); rrs.append(rsc); rrk.append(rrank)

    def flush(c):
        texts, codes, rawnames = buf[c]
        if not texts:
            return
        vec, keep = V[c]
        Xt = vec.transform(texts).tocsr()
        ca = np.asarray(codes, dtype=np.int64)
        Fm = (XQ[c] @ Xt.T).tocsr()
        Fm.data[Fm.data < R["fwd_min_sim"]] = 0; Fm.eliminate_zeros()
        S, I, thr = fS[c], fI[c], fT[c]
        nz = np.flatnonzero(np.diff(Fm.indptr))
        if len(nz):                                               # EXACT pre-filter: a row changes only if max(v) > thr
            rmax = np.maximum.reduceat(Fm.data, Fm.indptr[nz])
            nz = nz[rmax > thr[nz]]
        for r in nz:
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
                
        
        # Romanization Reverse TF-IDF
        non_ascii = np.array([is_non_ascii(rn) for rn in rawnames])
        if non_ascii.any():
            trom = [romanize(rn) for rn, na in zip(rawnames, non_ascii) if na]
            trom_codes = [codes[i] for i, na in enumerate(non_ascii) if na]
            ca_rom = np.asarray(trom_codes, dtype=np.int64)
            Xt_rom = V_rom[c].transform(trom).tocsr()
            touch_rom = (Xt_rom @ XQ_rom_revT[c]).tocsr()
            rows_rom = np.flatnonzero(np.diff(touch_rom.indptr))
            for s0 in range(0, len(rows_rom), 2000):
                rr = rows_rom[s0:s0 + 2000]
                M = (Xt_rom[rr] @ XS1_rom_revT[c]).tocsr()
                for k, r in enumerate(rr):
                    a, b = M.indptr[k], M.indptr[k + 1]
                    if a == b: continue
                    v, cols = M.data[a:b], M.indices[a:b]
                    if len(v) > 5:
                        sel = np.argpartition(-v, 4)[:5]; v, cols = v[sel], cols[sel]
                    order = np.argsort(-v)
                    for rank, (j, sc) in enumerate(zip(cols[order], v[order]), 1):
                        hi = hm[j]
                        if hi >= 0:
                            add(int(hi), int(ca_rom[r]), RROM5, rsc=float(sc), rrank=rank)
        texts.clear(); codes.clear(); rawnames.clear()

    tgt_frames = [load_source_tsv(find(f"{split}_source2.tsv")), load_source_tsv(find(f"{split}_source3.tsv"))]
    log(f"targets loaded (production loader): {sum(len(f) for f in tgt_frames):,}")
    t_s = time.time()
    tcodes_all = []
    df_names = Counter(); n_docs = 0                             # S2+S3 name-token DF (exp9 group A)
    for df in tgt_frames:
        codes_df = np.fromiter((encode_target_id(e) for e in df.entity_id), dtype=np.int64, count=len(df))
        tcodes_all.append(codes_df)
        for code, name_raw, addr_raw, c in zip(codes_df.tolist(), df.business_name, df.business_address, df.country):
            stats["targets"] += 1
            nm = norm_name(name_raw)
            df_names.update(set(nm.split())); n_docs += 1              # name DF over the whole target corpus
            if c not in V:
                stats["targets_other_country"] += 1
                continue
            ad = norm_address(addr_raw)
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
            bt, bc, braw = buf[c]; bt.append(f"{nm} {ad}".strip()); bc.append(code); braw.append(str(df.business_name.iat[j]))
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
    is_fwd = (bits & (F200 | F40)) > 0
    frank = np.where(is_fwd, rka, 9999).astype(np.int64)
    rrank = np.where((bits & (R5 | RROM5)) > 0, rka, 99).astype(np.int64)
    fsc = np.where(is_fwd, fsa, 0.0).astype(np.float32)
    rsc = np.where((bits & (R5 | RROM5)) > 0, rsa, 0.0).astype(np.float32)
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
                       ("flag_R5", R5), ("flag_RROM5", RROM5), ("flag_b2", B2C), ("flag_b3", B3C), ("flag_b4", B4C)):
        P[name_] = ((ub & bit) > 0).astype(np.int8)
    P["num_channels"] = P[["flag_A", "flag_B", "flag_F200", "flag_R5", "flag_b2", "flag_b3", "flag_b4"]].sum(axis=1).astype(np.int8)
    P["fwd_tfidf"], P["fwd_rank"], P["rev_tfidf"], P["rev_rank"] = ufs, ufr.astype(np.int32), urs, urr.astype(np.int16)
    del upk, ub, ufr, urr, ufs, urs
    gc.collect()
    log(f"pool: {len(P):,} unique pairs")
    for ln in LANES:                                          # dev-only retrieval lanes (e.g. romanization), exact union
        lp = Path(ln["path"]) if Path(ln["path"]).is_absolute() else ROOT / ln["path"]
        if sha256_file(lp) != ln["sha256"]:
            raise SystemExit(f"FAIL-CLOSED: lane {ln['name']} file changed")
        L = ST.read_parquet(lp)
        L = L[L.s1_id.isin(qi)]
        lq = L.s1_id.map(qi).values.astype(np.int64)
        lk = (lq << 32) | L.code.values.astype(np.int64)
        fl, sc_, rk_ = f"flag_lane_{ln['name']}", f"lane_{ln['name']}_score", f"lane_{ln['name']}_rank"
        pk_ = (P.q.values.astype(np.int64) << 32) | P.code.values
        pos = np.searchsorted(pk_, lk); pos = np.minimum(pos, len(pk_) - 1)
        hit = pk_[pos] == lk
        P[fl] = np.int8(0); P[sc_] = np.float32(0); P[rk_] = np.int16(999)
        P.loc[P.index[pos[hit]], fl] = 1
        P.loc[P.index[pos[hit]], sc_] = L.score.values[hit].astype(np.float32)
        P.loc[P.index[pos[hit]], rk_] = L["rank"].values[hit].astype(np.int16)
        nw = L[~hit]
        if len(nw):
            nq_ = lq[~hit].astype(np.int32)
            N = pd.DataFrame({"q": nq_, "code": nw.code.values.astype(np.int64)})
            N["label"] = np.isin(lk[~hit], truth_keys)
            N["g"] = grp_q[nq_]; N["fold"] = fold_q[nq_]; N["is_s2"] = (N.code.values & 3) == 2
            for c_ in ("flag_A", "flag_B", "flag_F100", "flag_F200", "flag_R1", "flag_R5", "flag_b2", "flag_b3", "flag_b4", "num_channels"):
                N[c_] = np.int8(0)
            N["fwd_tfidf"], N["fwd_rank"], N["rev_tfidf"], N["rev_rank"] = np.float32(0), np.int32(9999), np.float32(0), np.int16(99)
            for c_ in P.columns:
                if c_.startswith("flag_lane_") or (c_.startswith("lane_") and c_ not in (sc_, rk_)):
                    N[c_] = P[c_].dtype.type(0) if not c_.endswith("_rank") else np.int16(999)
            N[fl], N[sc_], N[rk_] = np.int8(1), nw.score.values.astype(np.float32), nw["rank"].values.astype(np.int16)
            P = pd.concat([P, N[P.columns]], ignore_index=True).sort_values(["q", "code"], kind="stable").reset_index(drop=True)
        log(f"lane {ln['name']}: {int(hit.sum()):,} existing + {int((~hit).sum()):,} new pairs")

    # ---------------- candidate + truth target texts (from the same loader frames)
    need = np.union1d(np.unique(P.code.values), np.fromiter(all_truth_codes, dtype=np.int64))
    tname, taddr, tpin, tnum, traw_empty, tcode, trawname = [], [], [], [], [], [], []
    for df, codes_df in zip(tgt_frames, tcodes_all):
        sel = np.flatnonzero(np.isin(codes_df, need))
        for j in sel.tolist():
            araw, c = df.business_address.iat[j], df.country.iat[j]
            tcode.append(int(codes_df[j])); tname.append(norm_name(df.business_name.iat[j])); taddr.append(norm_address(araw))
            trawname.append(str(df.business_name.iat[j]))       # B2R: preserve raw target name
            traw_empty.append(not araw.strip())
            pin, num, _, _ = X5.addr_parts(araw, c, addr_df[c]) if c in addr_df else ("", "", "", [])
            tpin.append(pin); tnum.append(num)
    del tgt_frames, tcodes_all, need
    order_t = np.argsort(np.array(tcode, dtype=np.int64))
    tcode_s = np.array(tcode, dtype=np.int64)[order_t]
    tname = [tname[i] for i in order_t]; taddr = [taddr[i] for i in order_t]
    trawname = [trawname[i] for i in order_t]                    # B2R: reorder raw names
    traw_empty = np.array(traw_empty)[order_t]; tpin = [tpin[i] for i in order_t]; tnum = [tnum[i] for i in order_t]
    del tcode, order_t
    gc.collect()
    log(f"target texts: {len(tcode_s):,}")
    P["t_idx"] = np.searchsorted(tcode_s, P.code.values)
    qname = [s1_name[e] for e in Q]
    qaddr = [norm_address(s1_addr_raw[e]) for e in Q]
    qcountry = dict(COUNTRY_CODE)

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
    q_cty = np.array([country_code(s1_country[e]) for e in Q], dtype=np.float32)
    if MODE in ("h4", "dev"):
        assert not np.isnan(q_cty).any(), "training/validation groups must be US/India only"
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
    EXPORT = OUT_DIR / "export"
    if MODE in ("h4", "dev"):
        EXPORT.mkdir(parents=True, exist_ok=True)
    else:
        assert MODELS_DIR is not None and MODELS_DIR.is_dir(), "test mode needs MODELS_DIR (the h4 export)"
        export_man = verify_export(MODELS_DIR, cfg["code_sha256"], cfg_sha)
        if export_man.get("mode") != "h4":
            raise SystemExit("FAIL-CLOSED: test inference requires the H4-validated export (mode h4)")
        dec_ = json.loads((MODELS_DIR / "decision.json").read_text())
        if dec_.get("pruner_features") != PR_RUN or dec_.get("matcher_features") != M_RUN:
            raise SystemExit("FAIL-CLOSED: exported model features differ from the frozen config")
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
            m.save_model(str(EXPORT / f"{tag}_fold{k}.txt"), num_iteration=m.best_iteration)
            log(f"{tag} fold {k}: trained on {len(tr):,} DEV rows, best_iter {m.best_iteration}")
            return m

        if MODE == "test":                                           # exported fold k scores exactly the S1s of fold k
            pred = np.full(len(D), np.nan)
            for k in range(NFOLD):
                te = np.flatnonzero(D.fold.values == k)
                if len(te):
                    pred[te] = lgb.Booster(model_file=str(MODELS_DIR / f"{tag}_fold{k}.txt")).predict(X[te])
            assert not np.isnan(pred).any()
            del X
            return pred, {"tag": tag, "models": "exported"}
        pred = crossfit(D.fold.values, D.g.values == 0, fit, lambda m, te: m.predict(X[te], num_iteration=m.best_iteration), NFOLD)
        rel = np.full(len(D), np.nan)                              # export round-trip must reproduce the scores
        for k in range(NFOLD):
            te = np.flatnonzero(D.fold.values == k)
            if len(te):
                rel[te] = lgb.Booster(model_file=str(EXPORT / f"{tag}_fold{k}.txt")).predict(X[te])
        roundtrip = float(np.max(np.abs(rel - pred))) if len(pred) else 0.0
        assert roundtrip <= 1e-9, f"{tag} export round-trip differs by {roundtrip}"
        info = {"best_iters": iters, "top_features": [f for _, f in sorted(zip(imps, feats), reverse=True)[:12]],
                "export_roundtrip_max_abs_diff": roundtrip}
        for gi, g in enumerate(GROUPS):
            mk = D.g.values == gi
            if mk.any() and 0 < y[mk].sum() < mk.sum():
                info[f"auc_{g}"] = float(roc_auc_score(y[mk], pred[mk]))
        del X
        return pred, info

    # ---------------- stage 2: cross-fitted pruner
    t_p = time.time()
    P["prune_p"], pinfo = lgb_crossfit(P, PR_RUN, PRN, "pruner")
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
    if EXTRA:                                                   # exp9 features, same functions as the cached-H1 run
        idf = F9.idf_table(df_names, n_docs)
        wj, mx_, cov = F9.name_idf_features([qname[i] for i in qvm], [tname[j] for j in tvm], idf, default_idf=max(idf.values()))
        ch = F9.char3_cosine(qname, tname, qvm, tvm)
        am = F9.missing_semantics(M.addr_jw.values, M.addr_jacc.values, M.addr_cont.values, M.q_addr_missing.values,
                                  traw_empty[tvm], wj)
        cn = F9.number_features([q_parts[i][1] or "" for i in qvm], [tnum[j] or "" for j in tvm],
                                [q_parts[i][0] or "" for i in qvm], [tpin[j] or "" for j in tvm],
                                [qaddr[i] for i in qvm], [taddr[j] for j in tvm])
        ce = F9.s1_context(qvm, M.name_jw.values, M.addr_jw.values)
        for name_, arr_ in zip(F9.NEW_FEATURES, (wj, mx_, cov, ch) + am + cn + ce):
            if name_ in EXTRA:
                M[name_] = arr_
        del wj, mx_, cov, ch, am, cn, ce
    # ---------------- B2R: Group-R romanized-name features ----
    q_raw_arr = np.array([str(s1_rawname.get(Q[i], "")) for i in qvm], dtype=object)
    t_raw_arr = np.array([str(trawname[j]) for j in tvm], dtype=object)
    r_vals = compute_r_features(q_raw_arr, t_raw_arr)
    for rname, rarr in zip(R_FEATURES, r_vals):
        M[rname] = rarr
    del q_raw_arr, t_raw_arr, r_vals
    log(f"B2R: {len(R_FEATURES)} Group-R features computed")
    log(f"matcher pairs {len(M):,} (DEV {int((M.g == 0).sum()):,}) | JW {JW_ENGINE}")
    M["p"], minfo = lgb_crossfit(M, M_RUN, MAT, "matcher")

    # ---------------- stage 4.5: Stage-2 features
    if "matcher_s2" in cfg:
        t_s2 = time.time()
        # M has "p", "q", "code", "is_s2", "t_idx"
        M_s1 = M[["q", "p", "is_s2", "t_idx"]].copy()
        M_s1["rk"] = M_s1.groupby("q", sort=False).p.rank(method="first", ascending=False) - 1
        
        top1 = M_s1[M_s1.rk == 0].set_index("q")
        top2 = M_s1[M_s1.rk == 1].set_index("q")
        
        top1_score = top1.p
        top2_score = top2.p.reindex(top1.index).fillna(0.0)
        
        M_q = M.q.values
        
        M["top1_score_gap"] = top1_score.reindex(M_q).values - M.p.values
        M["top1_to_top2_ratio"] = (top1_score / np.maximum(top2_score, 1e-9)).reindex(M_q).values
        
        valid_anchor = top1_score >= 0.75
        has_sa = valid_anchor.reindex(M_q).fillna(False).values
        M["has_strong_anchor"] = has_sa.astype(np.int8)
        
        anchor_is_s2 = top1.is_s2.reindex(M_q).values
        M["anchor_source"] = np.where(has_sa, anchor_is_s2.astype(np.int8), -1).astype(np.int8)
        
        M["sibling_source_complement"] = np.where(has_sa, (M.is_s2.values != anchor_is_s2).astype(np.int8), 0).astype(np.int8)
        
        anchor_t_idx = top1.t_idx.reindex(M_q).values
        
        idx_range = np.arange(len(M), dtype=np.int64)
        anchor_names = [tname[j] if h else "" for j, h in zip(anchor_t_idx, has_sa)]
        cand_names = [tname[j] for j in M.t_idx.values]
        M["cand_to_anchor_name_jw"] = np.where(
            has_sa,
            jaro_winkler_pairs(cand_names, idx_range, anchor_names, idx_range),
            np.nan
        ).astype(np.float32)
        
        anchor_addrs = [taddr[j] if h else "" for j, h in zip(anchor_t_idx, has_sa)]
        cand_addrs = [taddr[j] for j in M.t_idx.values]
        M["cand_to_anchor_addr_jw"] = np.where(
            has_sa,
            jaro_winkler_pairs(cand_addrs, idx_range, anchor_addrs, idx_range),
            np.nan
        ).astype(np.float32)
        
        anchor_pins = [tpin[j] if h else "" for j, h in zip(anchor_t_idx, has_sa)]
        cand_pins = [tpin[j] for j in M.t_idx.values]
        pin_match = np.zeros(len(M), dtype=np.int8)
        for r in range(len(M)):
            if has_sa[r]:
                pin_match[r] = 1 if (cand_pins[r] and cand_pins[r] == anchor_pins[r]) else 0
            else:
                pin_match[r] = -1
        M["cand_to_anchor_pin_match"] = pin_match
        
        anchor_nums = [tnum[j] if h else "" for j, h in zip(anchor_t_idx, has_sa)]
        cand_nums = [tnum[j] for j in M.t_idx.values]
        num_match = np.zeros(len(M), dtype=np.int8)
        for r in range(len(M)):
            if has_sa[r]:
                num_match[r] = 1 if (cand_nums[r] and cand_nums[r] == anchor_nums[r]) else 0
            else:
                num_match[r] = -1
        M["cand_to_anchor_num_match"] = num_match
        
        log(f"Stage-2 features computed in {(time.time() - t_s2)/60:.1f} min")
        
        MAT_S2 = cfg["matcher_s2"]
        M_S2_RUN = MAT_S2["features"]
        M["p"], minfo_s2 = lgb_crossfit(M, M_S2_RUN, MAT_S2, "matcher_s2")
        res["matcher_s2"] = minfo_s2
        res["timing_min"]["matcher_s2"] = (time.time() - t_s2) / 60
    if MODE == "test":
        tag = f"{SHARD_COUNTRY}_{SHARD_INDEX}of{SHARD_COUNT}"
        pos = {e: i for i, e in enumerate(s1_ids)}               # position in test_source1 (file order)
        q_pos = np.array([pos[e] for e in Q], dtype=np.int32)
        sc = pd.DataFrame({"s1_pos": q_pos[M.q.values], "code": M.code.values.astype(np.int64),
                           "prune_p": M.prune_p.values.astype(np.float64), "prune_rank": M.prune_rank.values.astype(np.int32),
                           "p": M.p.values.astype(np.float64)})
        ST.write_parquet(sc, OUT_DIR / f"shard_{tag}_scores.parquet")
        if ENV("SAVE_FEATURES", "0") == "1":                      # optional test-side C4 (off at full scale by default)
            fe = M[M_RUN].copy()
            fe["s1_pos"], fe["code"], fe["p"] = q_pos[M.q.values], M.code.values.astype(np.int64), M.p.values
            ST.write_parquet(fe, OUT_DIR / f"shard_{tag}_features.parquet")
        meta = {"shard": tag, "country": SHARD_COUNTRY, "index": SHARD_INDEX, "count": SHARD_COUNT, "s1": nq,
                "s1_with_candidates": int(sc.s1_pos.nunique()), "pairs": int(len(sc)), "config_sha256": cfg_sha,
                "code_sha256": cfg["code_sha256"], "export_manifest_sha256": sha256_file(MODELS_DIR / "manifest.json"),
                "s1_ids_sha256": hashlib.sha256("\n".join(Q).encode()).hexdigest(), "stats": dict(stats),
                "runtime_min": (time.time() - T0) / 60,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
        (OUT_DIR / f"shard_{tag}_meta.json").write_text(json.dumps(meta, indent=2, default=float))
        (OUT_DIR / f"shard_{tag}_s1.ids").write_text("\n".join(Q) + "\n")
        log(f"test shard {tag} done: {meta['pairs']:,} pairs for {meta['s1_with_candidates']:,}/{nq:,} S1")
        return
    res["matcher"] = minfo
    res["timing_min"]["matcher"] = (time.time() - t_m) / 60

    # ---------------- stage 5/6: set rules on RAW DEV OOF scores (no calibration), frozen grids
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
    grids = {k: v for k, v in grids.items() if v}              # frozen config may restrict to one family (expF)
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
    (EXPORT / "decision.json").write_text(json.dumps({"rule": sel_rule, "params": S["params"], "constrained": S["constrained"],
                                                      "budget": {"top_n": TOPN, "floor": FLOOR}, "m_dev": m_dev,
                                                      "pruner_features": PR_RUN, "matcher_features": M_RUN,
                                                      "drop_features": DROP, "candidate": cfg["candidate"]["name"],
                                                      "lanes": [ln["name"] for ln in LANES]}, indent=2))
    ART = OUT_DIR / "artifacts"
    ART.mkdir(parents=True, exist_ok=True)
    keep_cols = [c for c in P.columns if c not in ("t_idx",)]
    pool = P[keep_cols].copy()
    pool["label"] = np.where(pool.g.values == 0, pool.label.values.astype(np.int8), -1).astype(np.int8)   # H4 labels never stored
    pool["s1_id"] = np.array(Q, dtype=object)[pool.q.values]
    ST.write_parquet(pool, ART / "c2c3_pool_prebudget.parquet")          # full candidate pool BEFORE the budget + pruner scores
    del pool
    mcols = ["q", "code", "g", "fold", "prune_p", "prune_rank", "p"] + [c for c in M_RUN if c not in ("prune_p", "prune_rank")]
    mf = M[mcols + ["label"]].copy()
    mf["label"] = np.where(mf.g.values == 0, mf.label.values.astype(np.int8), -1).astype(np.int8)
    mf["s1_id"] = np.array(Q, dtype=object)[mf.q.values]
    ST.write_parquet(mf, ART / "c4c5_matcher_rows.parquet")             # matcher features + OOF/holdout scores
    del mf
    import pickle                                             # C0 / C1: normalisation + retrieval statistics
    for c in countries:
        with open(ART / f"c1_stats_{c}.pkl", "wb") as fh:
            pickle.dump({"vectorizer": V[c][0], "rev_keep": V[c][1], "addr_df": dict(addr_df[c]), "stats_exclude": cfg["stats_exclude"]}, fh)
    ST.write_parquet(pd.DataFrame({"token": list(df_names), "df": np.fromiter(df_names.values(), np.int64, len(df_names))}),
                     ART / "c1_name_df.parquet")
    (ART / "c1_name_df_meta.json").write_text(json.dumps({"n_docs": n_docs}))
    ST.write_parquet(pd.DataFrame({"s1_id": Q, "qn": qname, "qa": qaddr, "q_pin": [q_parts[i][0] or "" for i in range(nq)],
                                   "q_num": [q_parts[i][1] or "" for i in range(nq)], "country": [s1_country[e] for e in Q]}),
                     ART / "c0_s1_texts.parquet")
    in_pool_t = np.isin(tcode_s, np.unique(P.code.values))   # ONLY pool targets (truth-only targets would leak labels)
    it_ = np.flatnonzero(in_pool_t)
    ST.write_parquet(pd.DataFrame({"code": tcode_s[it_], "tn": [tname[j] for j in it_], "ta": [taddr[j] for j in it_],
                                   "t_pin": [tpin[j] or "" for j in it_], "t_num": [tnum[j] or "" for j in it_],
                                   "t_raw_empty": traw_empty[it_]}), ART / "c0_target_texts.parquet")
    write_manifest(EXPORT, {"kind": "export", "mode": MODE, "code_sha256": cfg["code_sha256"], "config_sha256": cfg_sha,
                            "candidate": cfg["candidate"]["name"], "drop_features": DROP,
                            "pruner_best_iters": pinfo["best_iters"], "matcher_best_iters": minfo["best_iters"]})
    write_manifest(ART, {"kind": "artifacts", "mode": MODE, "code_sha256": cfg["code_sha256"], "config_sha256": cfg_sha,
                         "note": "label = -1 for H4 rows (H4 labels are never persisted); target texts restricted to pool"})

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

    res["reports"] = {g: report(g) for g in GROUPS if bounds[g][1] > bounds[g][0]}
    res["meta"]["runtime_min"] = (time.time() - T0) / 60
    res["meta"]["peak_rss_mb_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (OUT_DIR / f"pipeline_{MODE}_results.json").write_text(json.dumps(res, indent=2, default=float))

    print("\n" + "=" * 110)
    print(f"FROZEN: budget top{TOPN}_floor{FLOOR} | rule {sel_rule} {S['params']} constrained={S['constrained']} (chosen on DEV OOF {S['dev_oof_f05']:.6f})")
    print(f"pruner AUC {({k: round(v, 4) for k, v in pinfo.items() if k.startswith('auc_')})} | "
          f"matcher AUC {({k: round(v, 4) for k, v in minfo.items() if k.startswith('auc_')})}")
    for g in res["reports"]:
        r = res["reports"][g]
        print(f"--- {g.upper()} {'(DEV cross-fitted OOF)' if g == 'dev' else '(evaluation only)'}")
        print(json.dumps(r, default=float))
    log(f"done | results -> {OUT_DIR / f'pipeline_{MODE}_results.json'}")


if __name__ == "__main__":
    main()
