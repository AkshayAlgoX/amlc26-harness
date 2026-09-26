import argparse
import csv
import gc
import json
import math
import multiprocessing
import os
import pathlib
import re
import resource
import sys
import time
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Import duckdb if needed, but for the correctness test we will just load a small mock or use the real df

def deterministic_romanize(text: str) -> str:
    # Simplified mock for correctness test speed, or we can use the real one.
    return str(text).lower()

def clean_text(text: object) -> str:
    return str(text).lower()

def worker_init(q_vecs_, vecs_, K_TOP_):
    global q_vecs, vecs, K_TOP
    q_vecs = q_vecs_
    vecs = vecs_
    K_TOP = K_TOP_

def process_chunk(c, t_ids, t_names, t_addrs, t_codes, is_non_latin):
    if not t_ids: return []
    t_rom = [deterministic_romanize(nm) for nm in t_names]
    t_orig = [clean_text(nm) for nm in t_names]
    
    T_rom_mat = vecs[c]["char"].transform(t_rom)
    S_rom_full = T_rom_mat.dot(q_vecs[c]["char"].T)
    T_orig_mat = vecs[c]["char"].transform(t_orig)
    S_orig_full = T_orig_mat.dot(q_vecs[c]["char"].T)
    
    chunk_k = min(K_TOP, S_rom_full.shape[0])
    results = []
    
    for q_idx in range(S_rom_full.shape[1]):
        col_rom = S_rom_full[:, q_idx].toarray().flatten()
        top_rom = np.argpartition(col_rom, -chunk_k)[-chunk_k:]
        top_rom = top_rom[np.argsort(-col_rom[top_rom])]
        
        col_orig = S_orig_full[:, q_idx].toarray().flatten()
        top_orig = np.argpartition(col_orig, -chunk_k)[-chunk_k:]
        top_orig = top_orig[np.argsort(-col_orig[top_orig])]
        
        # We just return the sparse top-K scores
        for idx in top_rom:
            if col_rom[idx] > 0.05:
                results.append((c, "TEST_B", q_idx, t_codes[idx], col_rom[idx]))
        for idx in top_orig:
            if col_orig[idx] > 0.05:
                results.append((c, "TEST_C", q_idx, t_codes[idx], col_orig[idx]))
                
    return results

def worker_task(args):
    worker_id, num_workers, data_file, max_targets, chunk_size, K_TOP = args
    import csv
    
    results = []
    with open(data_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        
        t_ids, t_names, t_addrs, t_codes, is_non_lat = [], [], [], [], []
        row_idx = 0
        total_scanned = 0
        
        for row in reader:
            if total_scanned >= max_targets:
                break
            total_scanned += 1
            if row_idx % num_workers == worker_id:
                if row:
                    tid, bname, baddr, country = row[0], row[1], row[2], row[3]
                    # code = encode
                    code = int(tid.split("-")[1])
                    if country == "US":
                        t_ids.append(tid)
                        t_names.append(bname)
                        t_addrs.append(baddr)
                        t_codes.append(code)
                        is_non_lat.append(False)
                        
                        if len(t_ids) >= chunk_size:
                            results.extend(process_chunk("US", t_ids, t_names, t_addrs, t_codes, is_non_lat))
                            t_ids, t_names, t_addrs, t_codes, is_non_lat = [], [], [], [], []
            row_idx += 1
            
        if t_ids:
            results.extend(process_chunk("US", t_ids, t_names, t_addrs, t_codes, is_non_lat))
            
    # Keep only global top K for this worker to reduce IPC
    # Actually, IPC is fine if the list isn't huge.
    return results

def test_correctness():
    # Mock data
    np.random.seed(42)
    # To be implemented
    pass

if __name__ == "__main__":
    pass
