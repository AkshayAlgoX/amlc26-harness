import sys

# Read B2R pipeline
code = open("experiments/agy2_x2_pipeline.py").read()

# Replace the feature names
code = code.replace(
    '        "t_name_kind",\n        "fx_name_char3",\n        "fx_name_jw",\n        "fx_name_tokjacc",\n        "fx_name_skeljacc",\n        "fx_name_skelcov",\n    ]',
    '        "t_name_kind",\n        "fx_name_char3",\n        "fx_name_jw",\n        "fx_name_tokjacc",\n        "fx_name_skeljacc",\n        "fx_name_skelcov",\n        "d_name_cov",\n        "d_name_jac",\n    ]'
)

# Load the dictionary at the top
load_block = """
import json
import os
import unicodedata
import re

# Load X2 name dictionary
dict_path = os.path.join(os.path.dirname(__file__), "x2_dict_name.json")
with open(dict_path, "r", encoding="utf-8") as fh:
    X2_DN = json.load(fh)

_isbr = lambda s: any(0x0900 <= ord(c) <= 0x0D7F for c in s)
_DEVTOK = re.compile(r"[^\\s,.\\-/()&:;#]+")
def _dtoks(raw): 
    return [unicodedata.normalize("NFC", t) for t in _DEVTOK.findall(str(raw)) if _isbr(t)]

def compute_dict_features(q_norm_names, t_raw_names):
    import numpy as np
    n = len(q_norm_names)
    cov = np.full(n, np.nan, dtype=np.float32)
    jac = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        qs = str(q_norm_names[i])
        raw = str(t_raw_names[i])
        if not raw or not _isbr(raw):
            continue
        L = set(qs.split())
        tr = [X2_DN.get(d) for d in _dtoks(raw)]
        if not tr:
            continue
        T = {t for t in tr if t}
        c = sum(1 for t in tr if t and t in L) / len(tr)
        j = len(T & L) / len(T | L) if (T or L) else np.nan
        cov[i] = np.float32(c)
        jac[i] = np.float32(j)
    return cov, jac
"""

# Inject before Feature Extractor
code = code.replace("from src.er.normalize import normalize_name, clean_text", "from src.er.normalize import normalize_name, clean_text\n" + load_block)

# Add feature call
old_call = """
        (kind, char3, jw, tokjacc, skeljacc, skelcov) = compute_r_features(
            q_raw, t_raw
        )
"""
new_call = """
        (kind, char3, jw, tokjacc, skeljacc, skelcov) = compute_r_features(
            q_raw, t_raw
        )
        
        q_norm = np.array([clean_text(normalize_name(q)) for q in q_raw], dtype=object)
        dcov, djac = compute_dict_features(q_norm, t_raw)
"""
code = code.replace(old_call, new_call)

# Add to stack
old_stack = "        E = np.column_stack([kind, char3, jw, tokjacc, skeljacc, skelcov])"
new_stack = "        E = np.column_stack([kind, char3, jw, tokjacc, skeljacc, skelcov, dcov, djac])"
code = code.replace(old_stack, new_stack)

open("experiments/agy2_x2_pipeline.py", "w").write(code)
