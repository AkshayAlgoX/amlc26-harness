"""AMLC26 romanization retrieval LANE (leakage-safe rewrite; builds a reusable lane node, never evaluates labels).

Replaces the untracked benchmark scripts (which fitted statistics on the first 100k S1 rows without holdout
exclusion, bypassed the production loader and embedded H1 label records). This builder:
  * reads data only through src/er/loader.py and NEVER opens the ground truth;
  * fits the char 3-4gram TF-IDF (on deterministic romanization of names) on train S1 EXCLUDING the frozen
    statistics exclusion (H2, H3, H4 per experiments/configs/h4_frozen.json); holdout S1 stay in the reverse corpus
    only as competitors (exactly like the pipeline's reverse channel), never as queries;
  * lane "translit": every target whose raw name contains a non-ASCII character is romanized and matched against
    ALL S1 of its country (reverse direction, top-K per target); pairs whose S1 is a query (H1) are emitted.
    Cost scales with the number of non-ASCII targets, not with queries, so the same lane is feasible at test time;
  * memory bound: targets are processed in blocks of max(1, MAX_BLOCK_BYTES // (n_S1_country * 12)) rows, so the
    worst-case (fully dense) sparse product of one block stays <= MAX_BLOCK_BYTES;
  * deterministic top-K: score descending, then S1 corpus index ascending.
Output (OUT_DIR): lane_<name>.parquet (s1_id, code, score, rank) + lane_<name>_manifest.json (sha256, counts, config).
Downstream: add {"name", "path", "sha256"} to cfg["lanes"] and run agy2_pipeline.py in dev mode (lanes are dev-only).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import agy2_exp8_store as ST  # noqa: E402
from experiments.agy2_exp4_fresh_holdout_validation import RE_NON_ALPHANUM  # noqa: E402
from src.er.ids import encode_target_id  # noqa: E402
from src.er.loader import load_source_tsv  # noqa: E402

ENV = os.environ.get
CONFIG = Path(ENV("LANE_CONFIG", str(ROOT / "experiments" / "configs" / "h4_frozen.json")))
DATA_ROOT = Path(ENV("DATA_ROOT", "/kaggle/input/datasets/akshayalgox/amlc26-2026-official-data-v2"))
OUT_DIR = Path(ENV("OUT_DIR", "/kaggle/working/lane"))
LANE = ENV("LANE_NAME", "translit")
K = int(ENV("LANE_K", "5"))
MAX_BLOCK_BYTES = int(ENV("MAX_BLOCK_BYTES", str(256 * 2**20)))

BRAHMI_VOWELS = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo", 0x0B: "ri",
                 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au"}
BRAHMI_CONSONANTS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
                     0x1D: "jh", 0x1E: "ny", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
                     0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh",
                     0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v",
                     0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h"}
BRAHMI_MATRAS = {0x3E: "aa", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "ri", 0x46: "e", 0x47: "e",
                 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au"}
VIRAMA, ANUSVARA, VISARGA, NUKTA = 0x4D, 0x02, 0x03, 0x3C


def romanize(text: str) -> str:
    """Deterministic Brahmic-block offset transliteration + NFKD accent stripping (no lookup tables of names)."""
    if not text:
        return ""
    s = "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not (0x0300 <= ord(c) <= 0x036F))
    out, i, n = [], 0, len(s)
    while i < n:
        cp = ord(s[i]); block = cp & 0xFF80
        if 0x0900 <= block <= 0x0D7F:
            off = cp & 0x7F
            if off in BRAHMI_CONSONANTS:
                cons = BRAHMI_CONSONANTS[off]
                if i + 1 < n and (ord(s[i + 1]) & 0xFF80) == block:
                    nxt = ord(s[i + 1]) & 0x7F
                    if nxt == VIRAMA:
                        out.append(cons); i += 2; continue
                    if nxt in BRAHMI_MATRAS:
                        out.append(cons + BRAHMI_MATRAS[nxt]); i += 2; continue
                out.append(cons + "a")
            elif off in BRAHMI_VOWELS:
                out.append(BRAHMI_VOWELS[off])
            elif off in BRAHMI_MATRAS:
                out.append(BRAHMI_MATRAS[off])
            elif off == ANUSVARA:
                out.append("n")
            elif off == VISARGA:
                out.append("h")
            elif off not in (VIRAMA, NUKTA):
                out.append(s[i])
        else:
            out.append(s[i])
        i += 1
    t = RE_NON_ALPHANUM.sub(" ", "".join(out).lower()).strip()
    return " ".join(t.split())


def is_non_ascii(s: str) -> bool:
    return any(ord(c) > 127 for c in s)


def topk_rows(M, k: int):
    """Per row of a CSR matrix: (col, score, rank) of the top-k by score desc, then column asc (deterministic)."""
    out = []
    for r in range(M.shape[0]):
        a, b = M.indptr[r], M.indptr[r + 1]
        if a == b:
            out.append(([], [])); continue
        cols, v = M.indices[a:b], M.data[a:b]
        o = np.lexsort((cols, -v))[:k]
        out.append((cols[o], v[o]))
    return out


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    t0 = time.time()
    cfg = json.loads(CONFIG.read_text())
    ids_root = Path(ENV("LANE_IDS_ROOT", str(ROOT)))

    def read_ids(rel):
        p = Path(rel) if Path(rel).is_absolute() else ids_root / rel
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    dev = read_ids(cfg["groups"]["dev"]["file"])
    if sha(Path(cfg["groups"]["dev"]["file"]) if Path(cfg["groups"]["dev"]["file"]).is_absolute()
           else ids_root / cfg["groups"]["dev"]["file"]) != cfg["groups"]["dev"]["sha256"]:
        raise SystemExit("FAIL-CLOSED: H1 list differs from the frozen config")
    excl = set()
    for n in cfg["stats_exclude"]:
        if n == "H4":
            excl |= set(read_ids(cfg["groups"]["h4"]["file"]))
        elif n in cfg["h4"]["prior_list_files"]:
            excl |= set(read_ids(cfg["h4"]["prior_list_files"][n]))
    assert not (set(dev) & excl), "query S1 inside the statistics exclusion"

    def find(name):
        hits = sorted(DATA_ROOT.rglob(name)); assert len(hits) == 1, name; return hits[0]
    s1 = load_source_tsv(find("train_source1.tsv"))
    qset = set(dev)
    rows, man = [], {"lane": LANE, "k": K, "config_sha256": sha(CONFIG), "code_sha256": sha(__file__),
                     "stats_exclude": cfg["stats_exclude"], "countries": {}}
    tgt = [load_source_tsv(find("train_source2.tsv")), load_source_tsv(find("train_source3.tsv"))]
    for c, sc in s1.groupby("country", sort=True):
        ids = sc.entity_id.values
        rom = [romanize(x) for x in sc.business_name.values]
        fit = np.array([e not in excl for e in ids])
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_df=0.4, sublinear_tf=True,
                              dtype=np.float32).fit([r for r, f in zip(rom, fit) if f])
        XS = vec.transform(rom).T.tocsr()                       # vocab x S1
        is_q = np.array([e in qset for e in ids])
        n_s1 = len(ids)
        block = max(1, MAX_BLOCK_BYTES // (n_s1 * 12))
        n_sel = 0
        for df in tgt:
            d = df[df.country == c]
            m = np.fromiter((is_non_ascii(x) for x in d.business_name.values), bool, len(d))
            d = d[m]
            n_sel += len(d)
            codes = np.fromiter((encode_target_id(e) for e in d.entity_id.values), np.int64, len(d))
            trom = [romanize(x) for x in d.business_name.values]
            for s0 in range(0, len(d), block):
                T = vec.transform(trom[s0:s0 + block]).tocsr()
                Mx = (T @ XS).tocsr()
                for r, (cols, v) in enumerate(topk_rows(Mx, K)):
                    for rank, (j, sv) in enumerate(zip(cols, v), 1):
                        if is_q[j]:
                            rows.append((ids[j], int(codes[s0 + r]), float(sv), rank))
        man["countries"][c] = {"s1": int(n_s1), "stats_fit_rows": int(fit.sum()), "non_ascii_targets": int(n_sel),
                               "block_rows": int(block), "vocab": len(vec.vocabulary_)}
    L = pd.DataFrame(rows, columns=["s1_id", "code", "score", "rank"])
    L = L.sort_values(["s1_id", "code", "score"], ascending=[True, True, False]).drop_duplicates(["s1_id", "code"])
    L = L.reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"lane_{LANE}.parquet"
    ST.write_parquet(L, out)
    man.update({"pairs": int(len(L)), "s1_with_pairs": int(L.s1_id.nunique()), "file": out.name, "sha256": sha(out),
                "runtime_min": (time.time() - t0) / 60})
    (OUT_DIR / f"lane_{LANE}_manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
