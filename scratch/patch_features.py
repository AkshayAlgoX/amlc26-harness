import re

code = open("experiments/agy2_x2_features.py").read()

code = code.replace(
    '"fx_name_skelcov",\n]',
    '"fx_name_skelcov",\n    "d_name_cov",\n    "d_name_jac",\n]'
)

dict_block = """
import json
import os
import unicodedata

dict_path = os.path.join(os.path.dirname(__file__), "x2_dict_name.json")
try:
    with open(dict_path, "r", encoding="utf-8") as fh:
        X2_DN = json.load(fh)
except Exception:
    X2_DN = {}

_isbr = lambda s: any(0x0900 <= ord(c) <= 0x0D7F for c in s)
_DEVTOK = re.compile(r"[^\\s,.\\-/()&:;#]+")
def _dtoks(raw): 
    return [unicodedata.normalize("NFC", t) for t in _DEVTOK.findall(str(raw)) if _isbr(t)]

def _compute_dict_features(q_raw_names, t_raw_names):
    import numpy as np
    n = len(q_raw_names)
    cov = np.full(n, np.nan, dtype=np.float32)
    jac = np.full(n, np.nan, dtype=np.float32)
    
    # We need norm names for q
    from src.er.normalize import clean_text, normalize_name
    for i in range(n):
        q_raw = str(q_raw_names[i])
        raw = str(t_raw_names[i])
        if not raw or not _isbr(raw):
            continue
            
        qs = str(clean_text(normalize_name(q_raw)))
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

code = code.replace("def fold(text: str) -> str:", dict_block + "\ndef fold(text: str) -> str:")

code = code.replace(
    'return kind, char3, jw, tokjacc, skeljacc_arr, skelcov_arr',
    'dcov, djac = _compute_dict_features(q_raw_names, t_raw_names)\n    return kind, char3, jw, tokjacc, skeljacc_arr, skelcov_arr, dcov, djac'
)

code = code.replace(
    'tuple of 6 np.ndarray',
    'tuple of 8 np.ndarray'
)
open("experiments/agy2_x2_features.py", "w").write(code)
