# AMLC26 — Controlled Kaggle Benchmark: Deterministic Unicode Romanization Retrieval.
# Evaluates whether deterministic, rule-based Unicode-to-Latin romanization
# can recover the 47.14% (2,708 pairs) transliteration miss category in H1.
import argparse

import csv
import gc
import glob

import json
import math
import os
import pathlib
import re
import resource
import sys
import time
import unicodedata

import multiprocessing
def init_worker(vecs_, q_vecs_, K_TOP_):
    global vecs, q_vecs, K_TOP
    vecs = vecs_
    q_vecs = q_vecs_
    K_TOP = K_TOP_

def process_chunk_parallel(args):
    c, items = args
    if not items: return None
    t_ids = [it[0] for it in items]
    t_names = [it[1] for it in items]
    t_addrs = [it[2] for it in items]
    t_codes = np.fromiter((encode_target_id(tid) for tid in t_ids), dtype=np.int64, count=len(t_ids))

    t_rom = [deterministic_romanize(nm) for nm in t_names]
    t_orig = [clean_text(nm) for nm in t_names]

    T_rom_mat = vecs[c]["char"].transform(t_rom)
    S_rom_full = T_rom_mat.dot(q_vecs[c]["char"].T)
    T_orig_mat = vecs[c]["char"].transform(t_orig)
    S_orig_full = T_orig_mat.dot(q_vecs[c]["char"].T)
    
    # We only need the top K for this chunk
    chunk_k = min(K_TOP, S_rom_full.shape[0])
    
    # We return sparse matrices or top-K lists to avoid IPC overhead
    results = []
    
    for q_idx in range(S_rom_full.shape[1]):
        col_rom = S_rom_full[:, q_idx].toarray().flatten()
        top_rom = np.argpartition(col_rom, -chunk_k)[-chunk_k:]
        top_rom = top_rom[np.argsort(-col_rom[top_rom])]
        
        col_orig = S_orig_full[:, q_idx].toarray().flatten()
        top_orig = np.argpartition(col_orig, -chunk_k)[-chunk_k:]
        top_orig = top_orig[np.argsort(-col_orig[top_orig])]
        
        for idx in top_rom:
            if col_rom[idx] > 0.05:
                results.append((c, q_idx, t_codes[idx], col_rom[idx], "TEST B"))
                results.append((c, q_idx, t_codes[idx], col_rom[idx], "TEST C"))
        
        for idx in top_orig:
            if col_orig[idx] > 0.05:
                results.append((c, q_idx, t_codes[idx], col_orig[idx], "TEST C"))

    return results


from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

t_start = time.time()
print("=== AMLC26 DETERMINISTIC ROMANIZATION RETRIEVAL BENCHMARK ===", flush=True)

# ---------------------------------------------------------------------------
# 1. Deterministic Unicode-to-Latin Romanizer (Algorithmic / Rule-Based)
# ---------------------------------------------------------------------------
BRAHMI_VOWELS = {
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo", 0x0B: "ri",
    0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au"
}

BRAHMI_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h"
}

BRAHMI_MATRAS = {
    0x3E: "aa", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "ri",
    0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au"
}

BRAHMI_VIRAMA = 0x4D
BRAHMI_ANUSVARA = 0x02
BRAHMI_VISARGA = 0x03
BRAHMI_NUKTA = 0x3C

RE_NON_ALPHANUM = re.compile(r"[^\w\s]", re.UNICODE)


def deterministic_romanize(text: str) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text))
    latin_cleaned = "".join(c for c in decomposed if not (0x0300 <= ord(c) <= 0x036F))

    res = []
    i = 0
    n = len(latin_cleaned)
    while i < n:
        ch = latin_cleaned[i]
        cp = ord(ch)
        block = cp & 0xFF80
        if 0x0900 <= block <= 0x0D7F:
            offset = cp & 0x7F
            if offset in BRAHMI_CONSONANTS:
                cons = BRAHMI_CONSONANTS[offset]
                if i + 1 < n and (ord(latin_cleaned[i + 1]) & 0xFF80) == block:
                    next_offset = ord(latin_cleaned[i + 1]) & 0x7F
                    if next_offset == BRAHMI_VIRAMA:
                        res.append(cons)
                        i += 2
                        continue
                    elif next_offset in BRAHMI_MATRAS:
                        res.append(cons + BRAHMI_MATRAS[next_offset])
                        i += 2
                        continue
                res.append(cons + "a")
                i += 1
                continue
            elif offset in BRAHMI_VOWELS:
                res.append(BRAHMI_VOWELS[offset])
            elif offset in BRAHMI_MATRAS:
                res.append(BRAHMI_MATRAS[offset])
            elif offset == BRAHMI_ANUSVARA:
                res.append("n")
            elif offset == BRAHMI_VISARGA:
                res.append("h")
            elif offset == BRAHMI_VIRAMA or offset == BRAHMI_NUKTA:
                pass
            else:
                res.append(ch)
            i += 1
        else:
            res.append(ch)
            i += 1

    s = "".join(res).lower()
    s = RE_NON_ALPHANUM.sub(" ", s).strip()
    return " ".join(s.split())


def clean_text(text: object) -> str:
    if text is None or pd.isna(text):
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = s.replace("&", " and ")
    s = RE_NON_ALPHANUM.sub(" ", s).lower().strip()
    return " ".join(s.split())


def norm_address(address: object) -> str:
    s = clean_text(address)
    if not s:
        return ""
    repl = {
        "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
        "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "sq": "square",
        "fl": "floor", "apt": "apartment", "ste": "suite", "bldg": "building"
    }
    toks = [repl.get(t, t) for t in s.split()]
    return " ".join(toks)


CITY_CANON_INDIA = {
    "bangalore": "bengaluru", "bengaluru": "bengaluru",
    "bombay": "mumbai", "mumbai": "mumbai",
    "calcutta": "kolkata", "kolkata": "kolkata",
    "madras": "chennai", "chennai": "chennai",
    "gurgaon": "gurugram", "gurugram": "gurugram",
    "pondicherry": "puducherry", "puducherry": "puducherry",
    "baroda": "vadodara", "vadodara": "vadodara",
    "cochin": "kochi", "kochi": "kochi",
    "trivandrum": "thiruvananthapuram", "thiruvananthapuram": "thiruvananthapuram",
    "ahmedabad": "ahmadabad", "ahmadabad": "ahmadabad",
    "secunderabad": "hyderabad",
}

ADDRESS_STOP_WORDS = {
    "street", "st", "road", "rd", "avenue", "ave", "boulevard", "blvd", "lane", "ln",
    "drive", "dr", "court", "ct", "place", "pl", "square", "sq", "floor", "fl",
    "apartment", "apt", "suite", "ste", "building", "bldg", "highway", "hwy",
    "near", "nr", "opp", "opposite", "west", "east", "north", "south",
    "usa", "us", "in", "unit", "sector", "plot", "block", "phase", "state"
}

PIN_RE_IN = re.compile(r"\b(\d{6})\b")
PIN_RE_US = re.compile(r"\b(\d{5})\b")
NUMBER_RE = re.compile(r"\b(\d+[a-z]?/\d+[a-z]?|\d+[a-z]?-\d+[a-z]?|\d{1,5})\b", re.IGNORECASE)


def extract_addr_components(raw_addr: str, country: str, rare_addr_df: dict[str, int]) -> tuple[str, str, str, list[str]]:
    text = clean_text(raw_addr)
    pin = ""
    if country == "India":
        m = PIN_RE_IN.search(text)
        if m: pin = m.group(1)
    else:
        m = PIN_RE_US.search(text)
        if m: pin = m.group(1)

    num = ""
    m_num = NUMBER_RE.search(text)
    if m_num:
        num = m_num.group(1).replace("#", "").strip()

    parts = [p.strip() for p in raw_addr.split(",") if p.strip()]
    city = ""
    if len(parts) >= 2:
        candidate_city = clean_text(parts[-2] if len(parts) >= 3 else parts[-1])
        words = candidate_city.split()
        if words and not words[0].isdigit():
            c = words[0]
            city = CITY_CANON_INDIA.get(c, c) if country == "India" else c

    tokens = {w for w in text.split() if len(w) >= 4 and not w.isdigit() and w not in ADDRESS_STOP_WORDS}
    rare = [w for w in tokens if rare_addr_df.get(w, 0) <= 200]
    return pin, num, city, rare


def encode_target_id(eid: str) -> int:
    if eid.startswith("S2-"):
        return int(eid[3:])
    elif eid.startswith("S3-"):
        return int(eid[3:]) | (1 << 62)
    raise ValueError(f"Invalid target eid: {eid}")


def decode_target_id(code: int) -> str:
    if (code >> 62) & 1:
        return f"S3-{code & ~(1 << 62)}"
    return f"S2-{code}"


def parse_id_list(val: object) -> list[str]:
    if val is None or pd.isna(val):
        return []
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return []
    return [p.strip() for p in s.split(",") if p.strip() and p.strip().lower() not in ("nan", "none", "null")]


# ---------------------------------------------------------------------------
# 2. Embedded Baseline Miss Signatures (5,744 pairs from H1)
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# 3. Main Benchmark Execution
# ---------------------------------------------------------------------------
def main():
    out_dir = pathlib.Path("output/romanization_corrected")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "stdout.log", "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(str(msg) + "\n")
        log_file.flush()

    # Discover official data root
    data_roots = [pathlib.Path("data/raw"),
        pathlib.Path("/kaggle/input/datasets/akshayalgox/amlc26-2026-official-data-v2"),
        pathlib.Path("/kaggle/input/amlc26-2026-official-data-v2"),
        pathlib.Path("/home/akshaya/Downloads/student_resource/student_resource/dataset/train"),
    ]
    data_dir = next((p for p in data_roots if p.is_dir() and list(p.rglob("train_source1.tsv"))), None)
    if not data_dir:
        hits = [pathlib.Path(p).parent for p in glob.glob("/kaggle/input/**/train_source1.tsv", recursive=True)]
        if hits: data_dir = hits[0]
    assert data_dir, "Could not locate train_source1.tsv under /kaggle/input"
    log(f"Using DATA_DIR: {data_dir}")

    # Discover H1 query IDs
    h1_files = [
        *list(pathlib.Path("/kaggle/input").rglob("prev_val_s1_50k.ids")),
        pathlib.Path("exclude_ids/prev_val_s1_50k.ids"),
    ]
    h1_file = next((p for p in h1_files if p.is_file()), None)
    assert h1_file, "Could not locate prev_val_s1_50k.ids"
    log(f"Using H1 query list: {h1_file}")

    eval_ids = [l.strip() for l in h1_file.read_text().splitlines() if l.strip()]
    nq = len(eval_ids)
    qi = {eid: i for i, eid in enumerate(eval_ids)}
    log(f"Evaluating {nq:,} Source 1 entities from H1...")

    import duckdb
    con = duckdb.connect()
    con.register("eval_ids_df", pd.DataFrame({"eid": eval_ids}))

    # Load S1 query records
    s1_df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{data_dir}/train_source1.tsv', delim='\t', header=true)
        WHERE entity_id IN (SELECT eid FROM eval_ids_df)
    """).df()
    s1_records = dict(zip(s1_df["entity_id"], zip(s1_df["business_name"], s1_df["business_address"], s1_df["country"])))
    del s1_df



    # Map baseline missed links and transliteration targets
    base_missed_set = set()
    translit_missed_set = set()
    for rec in BASELINE_MISS_RECORDS:
        s1 = rec["s1"]
        if s1 in qi:
            q_idx = qi[s1]
            code = encode_target_id(rec["t"])
            base_missed_set.add((q_idx, code))
            if rec["c"] == "transliteration":
                translit_missed_set.add((q_idx, code))


    base_recall = len(base_captured_set) / max(1, total_true_links)
    log(f"Baseline recall (BASE): {base_recall*100:.3f}% ({len(base_captured_set):,} captured, {len(base_missed_set):,} missed)")
    log(f"Transliteration missed links pool: {len(translit_missed_set):,}")

    # Fit Char 3-4g TF-IDF vectorizers
    log("Fitting vectorizers on strictly eligible S1 records (excluding H2, H3, H4)...")
    excluded_ids = set()
    for excl_file in ("expected/exp5_holdout_s1.ids", "expected/h3_s1.ids", "expected/h4_s1.ids"):
        if (pathlib.Path(excl_file)).exists():
            excluded_ids.update((pathlib.Path(excl_file)).read_text().splitlines())
    
    excl_df = pd.DataFrame({"id": list(excluded_ids)})
    con.register("excl_df", excl_df)
    
    sample_s1_df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{data_dir}/train_source1.tsv', delim='\t', header=true)
        WHERE entity_id NOT IN (SELECT id FROM excl_df)
    """).df()

    rare_addr_df = {"US": Counter(), "India": Counter()}
    for _, r in sample_s1_df.iterrows():
        toks = {w for w in clean_text(r["business_address"]).split() if len(w) >= 4 and not w.isdigit() and w not in ADDRESS_STOP_WORDS}
        rare_addr_df[r["country"]].update(toks)

    vecs = {}
    q_vecs = {}
    for country in ("US", "India"):
        sub_df = sample_s1_df[sample_s1_df["country"] == country]
        orig_names = [clean_text(n) for n in sub_df["business_name"]]
        rom_names = [deterministic_romanize(n) for n in sub_df["business_name"]]

        v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_df=0.4, sublinear_tf=True, dtype=np.float32)
        v_char.fit(orig_names + rom_names)
        vecs[country] = {"char": v_char}

        q_ids_c = [e for e in eval_ids if s1_records[e][2] == country]
        q_idx_c = [qi[e] for e in q_ids_c]
        q_orig = [clean_text(s1_records[e][0]) for e in q_ids_c]
        q_rom = [deterministic_romanize(s1_records[e][0]) for e in q_ids_c]

        q_vecs[country] = {
            "q_idx": np.array(q_idx_c, dtype=np.int32),
            "Q_orig": v_char.transform(q_orig),
            "Q_rom": v_char.transform(q_rom),
        }
        log(f"  {country}: Char vocabulary: {len(v_char.vocabulary_):,}, Queries: {len(q_idx_c):,}")

    del sample_s1_df
    con.close()
    gc.collect()

    # Build query hash tables for TEST A (Exact Romanized Hash) and Address Anchors
    log("Building query hash tables for TEST A and TEST D...")
    idx_rom_hash = defaultdict(list)
    idx_b2 = defaultdict(list)
    idx_b3 = defaultdict(list)

    for eid in eval_ids:
        q_idx = qi[eid]
        nm, ad, c = s1_records[eid]
        rom_nm = deterministic_romanize(nm)
        if len(rom_nm) >= 4:
            idx_rom_hash[hash((c, rom_nm))].append(q_idx)

        pin, num, city, rare = extract_addr_components(ad, c, rare_addr_df[c])
        if city and num:
            idx_b2[hash((c, city, num))].append(q_idx)
        for rt in rare:
            if num:
                idx_b3[hash((c, num, rt))].append(q_idx)

    CONFIGS = ["BASE", "TEST_B_rom_char", "TEST_C_orig_plus_rom"]

    cand_counts = {cfg: np.zeros(nq, dtype=np.int32) for cfg in CONFIGS}









    K_TOP = 20
    fS_rom = {}
    fC_rom = {}
    fT_rom = {}
    fS_orig = {}
    fC_orig = {}
    fT_orig = {}

    for country in ("US", "India"):
        n_qc = len(q_vecs[country]["q_idx"])
        fS_rom[country] = np.full((n_qc, K_TOP), -1.0, dtype=np.float32)
        fC_rom[country] = np.full((n_qc, K_TOP), -1, dtype=np.int64)
        fT_rom[country] = np.full(n_qc, 0.05, dtype=np.float32)

        fS_orig[country] = np.full((n_qc, K_TOP), -1.0, dtype=np.float32)
        fC_orig[country] = np.full((n_qc, K_TOP), -1, dtype=np.int64)
        fT_orig[country] = np.full(n_qc, 0.05, dtype=np.float32)

    # Examples tracking



    CHUNK_SIZE = 20000
    target_files = ["train_source2.tsv", "train_source3.tsv"]
    total_targets_scanned = 0
    max_targets = 10320219

    log(f"Streaming target records (chunk size: {CHUNK_SIZE:,}, max: {max_targets:,})...")

    for fname in target_files:
        fpath = data_dir / fname
        if not fpath.is_file(): continue
        log(f"Reading {fname}...")

        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            chunk_by_country = {"US": [], "India": []}

            def process_chunk(c: str, items: list):
                nonlocal total_targets_scanned
                if not items: return
                t_ids = [it[0] for it in items]
                t_names = [it[1] for it in items]
                t_addrs = [it[2] for it in items]
                t_codes = np.fromiter((encode_target_id(tid) for tid in t_ids), dtype=np.int64, count=len(t_ids))

                is_non_latin = [any(0x0900 <= ord(ch) <= 0x0D7F for ch in nm) for nm in t_names]
                t_rom = [deterministic_romanize(nm) for nm in t_names]
                t_orig = [clean_text(nm) for nm in t_names]





                # --- TEST B & C: Vectorized Char 3-4g TF-IDF ---
                vc = vecs[c]["char"]
                qc = q_vecs[c]

                T_rom_mat = vc.transform(t_rom)
                M_rom = (qc["Q_rom"] @ T_rom_mat.T).tocsr()

                T_orig_mat = vc.transform(t_orig)
                M_orig = (qc["Q_orig"] @ T_orig_mat.T).tocsr()

                for S, C_arr, thr, M in [
                    (fS_rom[c], fC_rom[c], fT_rom[c], M_rom),
                    (fS_orig[c], fC_orig[c], fT_orig[c], M_orig),
                ]:
                    M.data[M.data < 0.10] = 0
                    M.eliminate_zeros()
                    non_empty = np.flatnonzero(np.diff(M.indptr))
                    for r in non_empty:
                        a, b = M.indptr[r], M.indptr[r + 1]
                        v = M.data[a:b]
                        mask = v > thr[r]
                        if not mask.any(): continue
                        v_p = v[mask]
                        c_p = t_codes[M.indices[a:b][mask]]
                        cv = np.concatenate([S[r], v_p])
                        cc = np.concatenate([C_arr[r], c_p])
                        if len(cv) > K_TOP:
                            idx_top = np.argpartition(-cv, K_TOP - 1)[:K_TOP]
                            cv, cc = cv[idx_top], cc[idx_top]
                        order = np.lexsort((cc, -cv))
                        S[r], C_arr[r] = cv[order], cc[order]
                        thr[r] = S[r, -1]

                del M_rom, M_orig, T_rom_mat, T_orig_mat

            for row in reader:
                if not row: continue
                tid, bname, baddr, country = row[0], row[1], row[2], row[3]
                if country in chunk_by_country:
                    chunk_by_country[country].append((tid, bname, baddr))
                    total_targets_scanned += 1

                for c in ("US", "India"):
                    if len(chunk_by_country[c]) >= CHUNK_SIZE:
                        process_chunk(c, chunk_by_country[c])
                        chunk_by_country[c] = []

                if total_targets_scanned % 1000000 == 0:
                    log(f"Scanned {total_targets_scanned:,} / {max_targets:,} targets...")

                if total_targets_scanned >= max_targets:
                    break

            for c in ("US", "India"):
                if chunk_by_country[c]:
                    process_chunk(c, chunk_by_country[c])
                    chunk_by_country[c] = []

            if total_targets_scanned >= max_targets:
                break

    log(f"Finished streaming {total_targets_scanned:,} target records.")


    # Harvest Top-K candidates for TEST B, C, D
    log("Harvesting sparse candidates and compiling final metrics...")
    
    new_candidates = []
    
    for country in ("US", "India"):
        q_idx_c = q_vecs[country]["q_idx"]

        # TEST B: Romanized Char TF-IDF Top-20
        C_rom = fC_rom[country]
        S_rom = fS_rom[country]
        for local_r, global_qid in enumerate(q_idx_c):
            for k in range(K_TOP):
                if S_rom[local_r, k] > 0.05 and C_rom[local_r, k] > 0:
                    code = C_rom[local_r, k]
                    pair = (global_qid, code)
                    
                    s1_id = eval_ids[global_qid]
                    target_id = decode_target_id(code)
                    source = "S2" if str(target_id).startswith("S2") else "S3"
                    new_candidates.append({
                        "s1_id": s1_id,
                        "target_entity_id": target_id,
                        "source": source,
                        "lane": "TEST B",
                        "retrieval_score": float(S_rom[local_r, k]),
                        "retrieval_rank": k + 1
                    })
                    new_candidates.append({
                        "s1_id": s1_id,
                        "target_entity_id": target_id,
                        "source": source,
                        "lane": "TEST C",
                        "retrieval_score": float(S_rom[local_r, k]),
                        "retrieval_rank": k + 1
                    })
                    
                    cand_counts["TEST_B_rom_char"][global_qid] += 1

                    cand_counts["TEST_C_orig_plus_rom"][global_qid] += 1





        # TEST C & D: Add Original Char TF-IDF Top-20
        C_orig = fC_orig[country]
        S_orig = fS_orig[country]
        for local_r, global_qid in enumerate(q_idx_c):
            for k in range(K_TOP):
                if S_orig[local_r, k] > 0.05 and C_orig[local_r, k] > 0:
                    code = C_orig[local_r, k]
                    pair = (global_qid, code)
                    
                    s1_id = eval_ids[global_qid]
                    target_id = decode_target_id(code)
                    source = "S2" if str(target_id).startswith("S2") else "S3"
                    new_candidates.append({
                        "s1_id": s1_id,
                        "target_entity_id": target_id,
                        "source": source,
                        "lane": "TEST C",
                        "retrieval_score": float(S_orig[local_r, k]),
                        "retrieval_rank": k + 1
                    })
                    
                    cand_counts["TEST_C_orig_plus_rom"][global_qid] += 1






    for global_qid, code in list(still_missed)[:10]:
        failed_examples.append({
            "s1_id": eval_ids[global_qid],
            "target_id": decode_target_id(code),
            "s1_name": s1_records[eval_ids[global_qid]][0],
            "status": "Unrecovered by Romanization + Char TF-IDF + Anchors"
        })




    summary = {"benchmark": "AMLC26 Romanization Retrieval", "results": {}}
    for cfg in CONFIGS:
        if cfg == "BASE": continue
        c_arr = cand_counts[cfg]
        summary["results"][cfg] = {
            "mean": float(np.mean(c_arr)),
            "p50": float(np.median(c_arr)),
            "p95": float(np.percentile(c_arr, 95)),
            "p99": float(np.percentile(c_arr, 99)),
            "max": float(np.max(c_arr)),
            "total_candidates": int(np.sum(c_arr))
        }
    
    json_path = out_dir / "romanization_benchmark_results.json"
    json_path.write_text(json.dumps(summary, indent=2))
    log(f"Saved JSON results to {json_path}")
    
    # Save candidate pairs
    cand_df = pd.DataFrame(new_candidates)
    if not cand_df.empty:
        # Avoid duplicate candidates within the same lane
        cand_df = cand_df.sort_values(["s1_id", "lane", "retrieval_score", "target_entity_id"], ascending=[True, True, False, True]).drop_duplicates(subset=["s1_id", "target_entity_id", "lane"])
        
        # Optionally filter against frozen pool here if practical...
        # We will save them out and let downstream filter them.
        for lane in ["TEST B", "TEST C"]:
            lane_df = cand_df[cand_df["lane"] == lane]
            if not lane_df.empty:
                cand_path = out_dir / f"romanization_candidates_{lane.replace(' ', '_')}.parquet"
                con.execute(f"COPY (SELECT * FROM lane_df) TO '{cand_path}' (FORMAT PARQUET)")
                log(f"Saved {len(lane_df)} {lane} candidates to {cand_path}")


    summary = {
        "exit_code": 0,
        "wall_min": round((time.time() - t_start) / 60, 2),
        "child_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"RUN SUMMARY: {summary}")
    log_file.close()


if __name__ == "__main__":
    main()
