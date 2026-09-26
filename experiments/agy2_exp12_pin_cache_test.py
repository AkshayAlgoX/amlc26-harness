#!/usr/bin/env python3
import os
import sys
import gc
import re
import argparse
import time
import glob
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Canonical imports
from experiments.agy2_pipeline import inner_es_flag, prep, decide, constrain
from experiments.agy2_exp6_integrated import per_s1_eval
from src.er.ids import encode_target_id
from experiments.agy2_exp8_store import read_parquet

def find_tsv(name):
    for base in ["/home/akshaya/amlc26-kaggle-official", "/kaggle/input/datasets/akshayalgox/amlc26-2026-official-data-v2"]:
        p = Path(base) / name
        if p.is_file(): return p
    hits = list(Path("/home/akshaya").rglob(name))
    if not hits: raise FileNotFoundError(f"Cannot find {name}")
    return hits[0]

# Load canonical H4 config
cfg = json.loads(Path("experiments/configs/h4_frozen.json").read_text())
MAT = cfg["matcher"]
M_RUN = MAT["features"]  # Exact B2 feature list (61 features)

# Paths
H1_DUMP_DIR = Path("output/exp7_h1_dump_check/exp7_dump")
# Authoritative EXP9 B2 feature block
B2_NEWFEAT_PATH = Path("output/exp9_kernel_check/exp9_store/c4_mfeat/2c90dbae0e6de94b189e/newfeat.parquet")

def log(msg, t0=[time.time()]):
    print(f"[{(time.time() - t0[0]) / 60:6.2f} min] {msg}", flush=True)

def memory_warning():
    print("=" * 80)
    print("MEMORY RISK WARNING")
    print("Expected Peak Memory: ~7-8 GB (Merged B2 Cache ~4.5GB + String Dicts ~1GB + LGBM ~2GB)")
    print("Safe to run locally on 16GB RAM.")
    print("=" * 80)

def extract_pin_patched(addr_raw: str, country: str) -> str:
    if not addr_raw: return ""
    addr = str(addr_raw).strip().lower()
    c = str(country).strip().upper()
    
    if c == "INDIA":
        # Support spaced/hyphenated 6-digit PINs, strictly reject 10-digit phones and subsequences
        m = re.findall(r"(?<!\d)(?<!\d[\s-])\b\d{3}[\s-]?\d{3}\b(?![\s-]?\d)", addr)
        if m: return re.sub(r"[\s-]", "", m[-1])
    elif c == "US":
        m = re.findall(r"\b\d{5}\b", addr)
        if m: return m[-1]
    else:
        m = re.findall(r"\b\d{4,6}\b", addr)
        if m: return m[-1]
    return ""

def lgb_crossfit_local(D, feats, prm, tag, Qd):
    pred = np.full(len(D), np.nan, dtype=np.float64)
    X = np.empty((len(D), len(feats)), dtype=np.float32)
    for j, f_ in enumerate(feats):
        X[:, j] = D[f_].values
        
    y = D.label.values.astype(np.int8)
    
    # Use EXACT arrays from the Qd cache just like agy2_exp9_retrain.py
    fold_row = Qd.fold.values[D.q.values]
    es_row = Qd.es.values[D.q.values]
    
    params = dict(objective="binary", learning_rate=prm["learning_rate"], num_leaves=prm["num_leaves"],
                  min_data_in_leaf=prm["min_data_in_leaf"], feature_fraction=prm["feature_fraction"],
                  bagging_fraction=prm["bagging_fraction"], bagging_freq=1, lambda_l2=prm["lambda_l2"],
                  max_bin=prm["max_bin"], seed=42, num_threads=os.cpu_count(), verbose=-1)
                  
    for k in range(5):
        tr = np.flatnonzero(fold_row != k)
        e = es_row[tr]
        
        dtr = lgb.Dataset(X[tr[~e]], y[tr[~e]])
        dva = lgb.Dataset(X[tr[e]], y[tr[e]], reference=dtr)
        
        m = lgb.train(params, dtr, prm["max_rounds"], valid_sets=[dva], 
                      callbacks=[lgb.early_stopping(prm["early_stopping"], verbose=False)])
                      
        te = np.flatnonzero(fold_row == k)
        pred[te] = m.predict(X[te], num_iteration=m.best_iteration)
        log(f"{tag} fold {k}: best_iter {m.best_iteration}")
    
    del X, dtr, dva, m
    gc.collect()
    return pred

def score_local(D, Qd, tag):
    Dp = prep(D)
    q_max = Qd.q.max() + 1
    dev_hits = np.bincount(Dp.q.values[Dp.label.values], minlength=q_max)
    
    truth_n = np.zeros(q_max, dtype=np.int64)
    truth_n[Qd.q.values] = Qd.truth_n.values
    m_dev = float((truth_n - dev_hits)[Qd.q.values].mean())
    
    rule_prm = {"s": 0.5, "m": m_dev}
    
    sel = decide(Dp, "expF", rule_prm)
    sel_c = constrain(Dp, sel)
    
    tp = np.bincount(Dp.q.values[sel_c & Dp.label.values], minlength=q_max)
    fp = np.bincount(Dp.q.values[sel_c & ~Dp.label.values], minlength=q_max)
    global_tp = tp[Qd.q.values].sum()
    global_fp = fp[Qd.q.values].sum()
    global_truth = truth_n[Qd.q.values].sum()
    
    global_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    global_recall = global_tp / global_truth if global_truth > 0 else 0.0
    
    f_arr = per_s1_eval(Dp.q.values, Dp.label.values, sel_c, truth_n, q_max)
    macro_f05 = float(f_arr[Qd.q.values].mean())
    
    log(f"[{tag}] Macro F0.5: {macro_f05:.6f} | P: {global_precision:.6f} | R: {global_recall:.6f} | m={m_dev:.4f}")
    
    for group, name in [
        (Qd.truth_n == 0, "Card=0"),
        (Qd.truth_n == 1, "Card=1"),
        ((Qd.truth_n >= 2) & (Qd.truth_n <= 5), "Card=2-5"),
        (Qd.truth_n >= 6, "Card=6+"),
        (Qd.country == "US", "US"),
        (Qd.country == "INDIA", "India")
    ]:
        if group.sum() > 0:
            log(f"[{tag}] {name:10} (N={group.sum():6d}): {f_arr[Qd.q.values][group].mean():.6f}")
            
    return macro_f05

def stream_tsvs(q_set, code_set_arr):
    log("Streaming TSVs to extract required addresses...")
    s1_addr = {}
    s1_cty = {}
    
    chunksize = 200000
    for chunk in pd.read_csv(find_tsv("train_source1.tsv"), sep="\t", chunksize=chunksize, dtype=str):
        mask = chunk["entity_id"].isin(q_set)
        if mask.any():
            needed = chunk[mask]
            s1_addr.update(dict(zip(needed["entity_id"], needed["business_address"].fillna(""))))
            s1_cty.update(dict(zip(needed["entity_id"], needed["country"].fillna(""))))
                
    s23_addr = {}
    for tsv in ["train_source2.tsv", "train_source3.tsv"]:
        for chunk in pd.read_csv(find_tsv(tsv), sep="\t", chunksize=chunksize, dtype=str):
            encoded_ids = np.array([encode_target_id(x) for x in chunk["entity_id"].values], dtype=np.int64)
            mask = np.isin(encoded_ids, code_set_arr)
            if mask.any():
                needed = chunk[mask]
                s23_addr.update(dict(zip(encoded_ids[mask], needed["business_address"].fillna(""))))
                    
    return s1_addr, s1_cty, s23_addr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run on tiny deterministic subset")
    parser.add_argument("--full", action="store_true", help="Run full H1 cache experiment")
    args = parser.parse_args()
    
    if not args.smoke and not args.full:
        print("Specify --smoke or --full")
        return
        
    memory_warning()
    
    # 1. Load exact H1 cached artifacts (B0 features)
    log("Loading H1 cache (Base features)...")
    if not H1_DUMP_DIR.exists():
        fallback = Path(sorted(glob.glob('/kaggle/input/**/h1_matcher_oof.parquet', recursive=True))[0]).parent
        P = read_parquet(fallback / "h1_matcher_oof.parquet")
        Qd = read_parquet(fallback / "h1_queries.parquet")
    else:
        P = read_parquet(H1_DUMP_DIR / "h1_matcher_oof.parquet")
        Qd = read_parquet(H1_DUMP_DIR / "h1_queries.parquet")
    
    Qd["es"] = inner_es_flag(Qd.s1_id.values, cfg["folds"]["salt"])
    
    if args.smoke:
        P = P[P.q < 1000].copy()
        Qd = Qd[Qd.q < 1000].copy()
        
    log(f"Cache loaded: {len(P)} rows, {len(Qd)} S1s.")
    
    # 2. Load and merge authoritative B2 features
    log("Merging authoritative B2 feature artifact...")
    c4 = read_parquet(B2_NEWFEAT_PATH)
    if args.smoke:
        c4 = c4[c4.q < 1000].copy()
        
    # FATAL DISCOVERY: agy2_exp9_retrain.py used `P[c] = NF[c].values` despite 74k rows being scrambled in c4!
    # To identically reproduce the 0.939413 baseline, we MUST replicate this corrupted alignment.
    for c in c4.columns:
        if c in ("q", "code", "label"): continue
        P[c] = c4[c].values
        
    del c4; gc.collect()
    
    missing = [f for f in M_RUN if f not in P.columns]
    if missing:
        raise ValueError(f"Merged Cache missing required B2 features: {missing}")

    # FEATURE INTEGRITY SNAPSHOT
    baseline_pin_match = P["pin_match"].copy()
    baseline_label = P["label"].copy()
    baseline_q = P["q"].copy()
    baseline_code = P["code"].copy()
    baseline_fold = P["fold"].copy()
    baseline_other_features = {f: P[f].copy() for f in M_RUN if f != "pin_match"}
    
    # 3. Extract needed IDs 
    q_set = set(Qd.s1_id.astype(str))
    code_set_arr = np.array(list(set(P.code.values)), dtype=np.int64)
    
    # 4. Stream TSVs 
    s1_addr, s1_cty, s23_addr = stream_tsvs(q_set, code_set_arr)
    
    # 5. BASELINE REPRODUCTION
    log("Evaluating BASELINE (Authoritative Reproduction)...")
    P["p"] = lgb_crossfit_local(P, M_RUN, MAT, "baseline", Qd)
    base_f05 = score_local(P, Qd, "BASELINE")
    
    if args.full:
        expected = 0.9394136402248061
        tolerance = 0.0001
        if abs(base_f05 - expected) > tolerance:
            raise SystemExit(f"FAIL-CLOSED: BASELINE FAILED REPRODUCTION. Expected ~{expected}, got {base_f05:.6f}")
    
    log("Baseline reproduction verified. Proceeding to patch...")
    
    # 6. Recompute pin_match patched (High Performance Mapping)
    q_to_s1id = dict(zip(Qd.q, Qd.s1_id.astype(str)))
    s1_id_col = P["q"].map(q_to_s1id)
    
    a1_col = s1_id_col.map(s1_addr).fillna("")
    c_col = s1_id_col.map(s1_cty).fillna("US").str.strip().str.upper()
    a2_col = P["code"].map(s23_addr).fillna("")
    
    unique_pairs = set(zip(a1_col, c_col)) | set(zip(a2_col, c_col))
    pin_cache = {}
    for a, c in unique_pairs:
        pin_cache[(a, c)] = extract_pin_patched(a, c)
        
    log(f"Extracted PINs for {len(unique_pairs)} unique address/country pairs.")
    
    pair1 = pd.Series(list(zip(a1_col, c_col)))
    pair2 = pd.Series(list(zip(a2_col, c_col)))
    
    qp_arr = pair1.map(pin_cache).values
    tp_arr = pair2.map(pin_cache).values
    
    new_pin = np.zeros(len(P), dtype=np.int8)
    valid_mask = (qp_arr != "") & (tp_arr != "") & (qp_arr == tp_arr)
    new_pin[valid_mask] = 1
    
    changed_mask = (new_pin != baseline_pin_match.values)
    changed_pos = (changed_mask & baseline_label.values).sum()
    changed_neg = (changed_mask & ~baseline_label.values).sum()
                
    log(f"PIN match changes: {changed_mask.sum()} total ({changed_pos} pos, {changed_neg} neg)")
    
    # 7. FEATURE INTEGRITY ASSERTIONS
    log("Verifying Integrity before patched evaluation...")
    assert len(P) == len(baseline_pin_match), "Row count changed!"
    assert (P["q"].values == baseline_q.values).all(), "q changed!"
    assert (P["code"].values == baseline_code.values).all(), "code changed!"
    assert (P["fold"].values == baseline_fold.values).all(), "fold changed!"
    assert (P["label"].values == baseline_label.values).all(), "Labels changed!"
    
    for f in M_RUN:
        if f != "pin_match":
            assert (P[f].values == baseline_other_features[f].values).all(), f"Feature {f} maliciously altered!"

    # 8. Evaluate PATCHED
    log("Evaluating PATCHED (MATCHER-ONLY COUNTERFACTUAL)...")
    P["pin_match"] = new_pin
    P["p"] = lgb_crossfit_local(P, M_RUN, MAT, "patched", Qd)
    patch_f05 = score_local(P, Qd, "PATCHED")
    
    print("\n" + "="*50)
    print("MATCHER-ONLY COUNTERFACTUAL EVIDENCE (CACHE ONLY)")
    print("WARNING: Does NOT include marginal retrieval improvements (Channel B4).")
    print("="*50)
    print(f"BASELINE: Macro F0.5 = {base_f05:.6f}")
    print(f"PATCHED:  Macro F0.5 = {patch_f05:.6f}")
    print(f"Delta:    {(patch_f05 - base_f05):+.6f}")
    print("="*50)
    print("DONE.")

if __name__ == "__main__":
    main()
