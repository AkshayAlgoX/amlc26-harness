"""B2R Group-R features: 6 romanized-name features for the matcher.

Features (all float32):
  t_name_kind      0 = target name is ASCII-only, 1 = non-ASCII (needs romanization)
  fx_name_char3    char 3-gram cosine on folded/romanized names (NaN when kind==0)
  fx_name_jw       Jaro-Winkler similarity on folded/romanized names (NaN when kind==0)
  fx_name_tokjacc  token Jaccard on folded/romanized names (NaN when kind==0)
  fx_name_skeljacc skeleton-token Jaccard on folded/romanized names (NaN when kind==0)
  fx_name_skelcov  skeleton-token coverage (intersection / min) on folded/romanized names (NaN when kind==0)

Romanization: deterministic schwa-aware transliteration from agy2_lane_romanization.romanize().
Folding: lowercase → vowel collapse (aa→a, ee→i, oo→u) → drop word-final 'a' →
         ph→f, w→v → collapse double letters.
Skeleton: remove digits → remove LEGAL tokens → strip vowels → collapse doubles.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# ---- import the pipeline's own romanize + is_non_ascii ----
from experiments.agy2_lane_romanization import romanize, is_non_ascii

# ---- import normalize helpers ----
from src.er.normalize import clean_text, normalize_name, LEGAL_SUFFIX_MAP

# ---- import the pipeline's own JW engine (numba or python fallback) ----
from experiments.agy2_exp6_integrated import jaro_winkler_pairs, pack

# ---- Group-R feature names (exact, do not rename) ----
R_FEATURES: List[str] = [
    "t_name_kind",
    "fx_name_char3",
    "fx_name_jw",
    "fx_name_tokjacc",
    "fx_name_skeljacc",
    "fx_name_skelcov",
]


# ---- fold(): documented phonetic folding ----
_RE_WORD_FINAL_A = re.compile(r"a\b")
_RE_DOUBLE = re.compile(r"(.)\1+")


def fold(text: str) -> str:
    """Fold a romanized/normalised name for fuzzy comparison.

    Steps (order matters):
      1. lowercase
      2. aa → a, ee → i, oo → u  (vowel digraph collapse)
      3. drop word-final inherent 'a'
      4. ph → f, w → v  (consonant normalisation)
      5. collapse consecutive identical characters
    """
    t = text.lower()
    # vowel digraphs — order: longest first so 'oo' doesn't eat 'o' twice
    t = t.replace("aa", "a").replace("ee", "i").replace("oo", "u")
    # drop word-final inherent 'a'
    t = _RE_WORD_FINAL_A.sub("", t)
    # consonant normalisation
    t = t.replace("ph", "f").replace("w", "v")
    # collapse doubles
    t = _RE_DOUBLE.sub(r"\1", t)
    return t


# ---- skel(): skeleton extraction ----
_RE_DIGITS = re.compile(r"\d+")
_RE_VOWELS = re.compile(r"[aeiou]")


def skel(text: str) -> str:
    """Extract consonant skeleton of a folded name.

    Steps:
      1. remove digits
      2. remove LEGAL suffix tokens (inc, ltd, pvt, llc, …)
      3. strip vowels
      4. collapse consecutive identical characters
    """
    t = _RE_DIGITS.sub("", text)
    tokens = [tok for tok in t.split() if tok not in LEGAL_SUFFIX_MAP]
    t = " ".join(tokens)
    t = _RE_VOWELS.sub("", t)
    t = _RE_DOUBLE.sub(r"\1", t)
    return t


# ---- main feature extraction ----
def compute_r_features(
    q_raw_names: np.ndarray,
    t_raw_names: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Compute the 6 Group-R features for n candidate pairs.

    Parameters
    ----------
    q_raw_names : array of str, shape (n,)
        Raw S1 business_name values (before any normalisation).
    t_raw_names : array of str, shape (n,)
        Raw target (S2/S3) business_name values.

    Returns
    -------
    tuple of 6 np.ndarray (float32), each shape (n,):
        (t_name_kind, fx_name_char3, fx_name_jw, fx_name_tokjacc,
         fx_name_skeljacc, fx_name_skelcov)

    Notes
    -----
    - t_name_kind: 1.0 when the raw target name contains any non-ASCII char, else 0.0
    - For kind==0 (ASCII target), all five fx_* features are NaN
    - S1 side: norm_name(raw S1 name) → fold()
    - Target side: clean_text(romanize(raw target name)) → fold()
    - skel() applied to both folded strings for skeljacc / skelcov
    """
    n = len(q_raw_names)
    assert len(t_raw_names) == n

    kind = np.zeros(n, dtype=np.float32)
    char3 = np.full(n, np.nan, dtype=np.float32)
    jw = np.full(n, np.nan, dtype=np.float32)
    tokjacc = np.full(n, np.nan, dtype=np.float32)
    skeljacc_arr = np.full(n, np.nan, dtype=np.float32)
    skelcov_arr = np.full(n, np.nan, dtype=np.float32)

    # Pre-compute which pairs need romanization
    nonascii_mask = np.array([is_non_ascii(str(t)) for t in t_raw_names], dtype=bool)
    kind[nonascii_mask] = 1.0

    if not nonascii_mask.any():
        return kind, char3, jw, tokjacc, skeljacc_arr, skelcov_arr

    indices = np.flatnonzero(nonascii_mask)

    # Compute folded strings for non-ASCII pairs
    fq_list: List[str] = []
    ft_list: List[str] = []
    sq_list: List[str] = []
    st_list: List[str] = []

    for i in indices:
        qn = normalize_name(q_raw_names[i])
        t2 = clean_text(romanize(str(t_raw_names[i])))
        fq = fold(qn)
        ft = fold(t2)
        fq_list.append(fq)
        ft_list.append(ft)
        sq_list.append(skel(fq))
        st_list.append(skel(ft))

    m = len(indices)

    # ---- fx_name_jw: use the pipeline's own JW engine ----
    idx_range = np.arange(m, dtype=np.int64)
    jw_vals = jaro_winkler_pairs(fq_list, idx_range, ft_list, idx_range)
    jw[indices] = jw_vals

    # ---- fx_name_char3: char_wb 3-gram cosine via HashingVectorizer ----
    hv = HashingVectorizer(
        analyzer="char_wb", ngram_range=(3, 3), n_features=2**20,
        alternate_sign=False, norm="l2", dtype=np.float32,
    )
    Xq = hv.transform(fq_list)
    Xt = hv.transform(ft_list)
    # element-wise cosine (both are l2-normalised)
    c3 = np.asarray(Xq.multiply(Xt).sum(axis=1)).ravel().astype(np.float32)
    char3[indices] = c3

    # ---- fx_name_tokjacc: token Jaccard ----
    for j, i in enumerate(indices):
        q_tok = set(fq_list[j].split())
        t_tok = set(ft_list[j].split())
        union = q_tok | t_tok
        if union:
            tokjacc[i] = np.float32(len(q_tok & t_tok) / len(union))
        else:
            tokjacc[i] = np.float32(1.0)  # both empty → identical

    # ---- fx_name_skeljacc / fx_name_skelcov: skeleton token Jaccard & coverage ----
    for j, i in enumerate(indices):
        sq_tok = set(sq_list[j].split())
        st_tok = set(st_list[j].split())
        union = sq_tok | st_tok
        inter = sq_tok & st_tok
        if union:
            skeljacc_arr[i] = np.float32(len(inter) / len(union))
        else:
            skeljacc_arr[i] = np.float32(1.0)  # both empty

        if sq_tok and st_tok:
            skelcov_arr[i] = np.float32(len(inter) / min(len(sq_tok), len(st_tok)))
        elif not sq_tok and not st_tok:
            skelcov_arr[i] = np.float32(1.0)
        else:
            skelcov_arr[i] = np.float32(0.0)

    return kind, char3, jw, tokjacc, skeljacc_arr, skelcov_arr
