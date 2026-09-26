"""AMLC26 Top Misses Analysis on H1 Development Data.

Extracts all 5,744 missed true links from the frozen EXP7/EXP8 candidate pool,
fetches the full text records for S1 and Target, and classifies every miss into
the 9 pre-registered categories:
  1. typo
  2. abbreviation
  3. word-order change
  4. transliteration
  5. missing address component
  6. numeric mismatch
  7. token substitution
  8. severe corruption
  9. genuinely semantic similarity
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.er.ids import decode_target_id
from src.er.loader import load_ground_truth_df, load_source_tsv
from experiments.agy2_exp4_fresh_holdout_validation import clean_text, norm_address, norm_name

DATA_DIR = Path("/home/akshaya/Downloads/student_resource/student_resource/dataset/train")
H1_IDS_FILE = ROOT / "exclude_ids" / "prev_val_s1_50k.ids"
STORE_POOL = ROOT / "artifacts/exp8_store/v0_seed/5bcbabbb227de073cb32/pool.parquet"
STORE_QUERIES = ROOT / "artifacts/exp8_store/v0_seed/5bcbabbb227de073cb32/queries.parquet"
OUT_DIR = ROOT / "output/top_misses"

# Common abbreviation expansions
ABBREV_MAP = {
    "pvt": "private", "ltd": "limited", "inc": "incorporated", "corp": "corporation",
    "co": "company", "llc": "limited liability company", "llp": "limited liability partnership",
    "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard", "dr": "drive",
    "ln": "lane", "hwy": "highway", "sq": "square", "ct": "court", "pl": "place",
    "apt": "apartment", "ste": "suite", "bldg": "building", "fl": "floor",
    "assoc": "associates", "assn": "association", "svc": "services", "svcs": "services",
    "intl": "international", "natl": "national", "mfg": "manufacturing", "mgmt": "management",
    "dept": "department", "univ": "university", "tech": "technologies", "sys": "systems",
    "soln": "solutions", "grp": "group", "ent": "enterprises", "hosp": "hospital",
    "ctr": "center", "ctr.": "center", "mtr": "motor", "auto": "automobile",
}

NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
DIGIT_RE = re.compile(r"\b\d+\b")


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr[j + 1] = min(insertions, deletions, substitutions)
        prev = curr
    return prev[-1]


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0
    match1 = [False] * len1
    match2 = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if match2[j] or s1[i] != s2[j]:
                continue
            match1[i] = True
            match2[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i in range(len1):
        if not match1[i]:
            continue
        while not match2[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0
    # Winkler bonus
    prefix = 0
    for i in range(min(4, min(len1, len2))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1.0 - jaro)


def classify_miss(s1_name_raw: str, s1_addr_raw: str,
                  tx_name_raw: str, tx_addr_raw: str,
                  country: str) -> tuple[str, dict]:
    """Classifies a missed link into one of the 9 categories with diagnostic metrics."""
    n1_norm = norm_name(s1_name_raw)
    n2_norm = norm_name(tx_name_raw)
    a1_norm = norm_address(s1_addr_raw)
    a2_norm = norm_address(tx_addr_raw)

    t1_norm = f"{n1_norm} {a1_norm}".strip()
    t2_norm = f"{n2_norm} {a2_norm}".strip()

    tokens1 = set(clean_text(s1_name_raw).split())
    tokens2 = set(clean_text(tx_name_raw).split())
    addr_toks1 = set(clean_text(s1_addr_raw).split())
    addr_toks2 = set(clean_text(tx_addr_raw).split())

    name_jacc = len(tokens1 & tokens2) / max(1, len(tokens1 | tokens2))
    addr_jacc = len(addr_toks1 & addr_toks2) / max(1, len(addr_toks1 | addr_toks2)) if (addr_toks1 and addr_toks2) else 0.0

    name_jw = jaro_winkler_similarity(n1_norm, n2_norm)
    addr_jw = jaro_winkler_similarity(a1_norm, a2_norm)

    n1_digits = set(DIGIT_RE.findall(s1_addr_raw + " " + s1_name_raw))
    n2_digits = set(DIGIT_RE.findall(tx_addr_raw + " " + tx_name_raw))

    diag = {
        "name_jacc": name_jacc,
        "addr_jacc": addr_jacc,
        "name_jw": name_jw,
        "addr_jw": addr_jw,
        "n1_norm": n1_norm,
        "n2_norm": n2_norm,
        "a1_norm": a1_norm,
        "a2_norm": a2_norm,
        "country": country,
    }

    # 1. Transliteration: Non-ASCII characters present in either name or address
    has_non_ascii_name = bool(NON_ASCII_RE.search(s1_name_raw)) or bool(NON_ASCII_RE.search(tx_name_raw))
    has_non_ascii_addr = bool(NON_ASCII_RE.search(s1_addr_raw)) or bool(NON_ASCII_RE.search(tx_addr_raw))
    if has_non_ascii_name or has_non_ascii_addr:
        return "transliteration", diag

    # 2. Missing address component: Target address is empty or trivial (< 4 chars)
    if not a2_norm or len(a2_norm) < 4 or not a1_norm or len(a1_norm) < 4:
        return "missing_address_component", diag

    # 3. Numeric mismatch: Significant address digits differ (e.g. street number or PIN code present in both but mismatch)
    if n1_digits and n2_digits:
        # If numbers exist in both, but overlap is 0 and both have >= 2-digit numbers
        significant_d1 = {d for d in n1_digits if len(d) >= 2}
        significant_d2 = {d for d in n2_digits if len(d) >= 2}
        if significant_d1 and significant_d2 and not (significant_d1 & significant_d2):
            if name_jw >= 0.70 or name_jacc >= 0.30:
                return "numeric_mismatch", diag

    # 4. Word-order change: Name tokens have very high overlap (Jaccard >= 0.70) but different order
    if name_jacc >= 0.70 and n1_norm != n2_norm:
        return "word_order_change", diag

    # 5. Typo: Low edit distance or high Jaro-Winkler on name despite low token overlap
    ed = levenshtein_distance(n1_norm, n2_norm)
    if (ed <= 3 and len(n1_norm) >= 6) or (name_jw >= 0.88 and name_jacc < 0.6):
        return "typo", diag

    # 6. Abbreviation: Expanded vs abbreviated forms
    # Check if expanded tokens match
    exp_tokens1 = {ABBREV_MAP.get(t, t) for t in tokens1}
    exp_tokens2 = {ABBREV_MAP.get(t, t) for t in tokens2}
    exp_jacc = len(exp_tokens1 & exp_tokens2) / max(1, len(exp_tokens1 | exp_tokens2))
    if exp_jacc >= 0.65 or any(
        (len(t1) <= 3 and any(t2.startswith(t1) for t2 in tokens2)) or
        (len(t2) <= 3 and any(t1.startswith(t2) for t1 in tokens1))
        for t1 in tokens1 for t2 in tokens2 if len(t1) > 1 and len(t2) > 1
    ):
        if name_jw >= 0.75 or exp_jacc >= 0.50:
            return "abbreviation", diag

    # 7. Token substitution: Core name token substituted (e.g. shared address, but name token swapped)
    if addr_jacc >= 0.40 and (tokens1 & tokens2):
        return "token_substitution", diag

    # 8. Severe corruption: Characters or strings appear heavily corrupted / truncated
    if len(n1_norm) < 4 or len(n2_norm) < 4 or (len(n1_norm) > 20 and ed > 15 and name_jw < 0.50 and addr_jw > 0.60):
        return "severe_corruption", diag

    # 9. Genuinely semantic similarity: Completely different name, but matching entity/address (DBA / trade name)
    if addr_jw >= 0.70 or addr_jacc >= 0.50:
        return "genuinely_semantic_similarity", diag

    # Default fallback
    if name_jw > 0.65:
        return "typo", diag
    return "genuinely_semantic_similarity", diag


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("--- 1. Loading H1 Ground Truth & Queries ---")
    h1_ids = [l.strip() for l in H1_IDS_FILE.read_text().splitlines() if l.strip()]
    h1_set = set(h1_ids)

    gtdf = load_ground_truth_df(DATA_DIR / "train_ground_truth.tsv")
    gt_dev = gtdf[gtdf["source1_entity_id"].isin(h1_set)]
    all_dev_pairs = set()
    for _, r in gt_dev.iterrows():
        s1 = r["source1_entity_id"]
        for t in r["matched_list"]:
            all_dev_pairs.add((s1, t))
    print(f"Total True Pairs in H1: {len(all_dev_pairs):,}")

    con = duckdb.connect()
    q_df = con.execute(f"""
        SELECT q, s1_id 
        FROM read_parquet('{STORE_QUERIES}')
        WHERE "group" = 'dev'
    """).df()
    q_map = dict(zip(q_df["q"], q_df["s1_id"]))

    pool_pairs = con.execute(f"""
        SELECT q, code 
        FROM read_parquet('{STORE_POOL}')
        WHERE grp = 'train' AND label = 1
    """).fetchall()

    retrieved_dev_pairs = set((q_map[q], decode_target_id(code)) for q, code in pool_pairs)
    print(f"Retrieved True Pairs in Baseline: {len(retrieved_dev_pairs):,}")

    missed_pairs = sorted(all_dev_pairs - retrieved_dev_pairs)
    print(f"Missed True Pairs in H1: {len(missed_pairs):,} ({len(missed_pairs)/len(all_dev_pairs)*100:.3f}%)")

    missed_s1_needed = {s1 for s1, _ in missed_pairs}
    missed_targets_needed = {t for _, t in missed_pairs}
    print(f"Unique S1s with misses: {len(missed_s1_needed):,} | Unique Targets missed: {len(missed_targets_needed):,}")

    print("\n--- 2. Fetching Raw Entity Records ---")
    s1_records = {}
    with open(DATA_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            eid = row[0]
            if eid in missed_s1_needed:
                s1_records[eid] = {
                    "business_name": row[1] if len(row) > 1 else "",
                    "business_address": row[2] if len(row) > 2 else "",
                    "country": row[3] if len(row) > 3 else "",
                }

    target_records = {}
    for fname in ("train_source2.tsv", "train_source3.tsv"):
        print(f"Scanning {fname}...")
        with open(DATA_DIR / fname, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            for row in reader:
                if not row:
                    continue
                eid = row[0]
                if eid in missed_targets_needed:
                    target_records[eid] = {
                        "business_name": row[1] if len(row) > 1 else "",
                        "business_address": row[2] if len(row) > 2 else "",
                        "country": row[3] if len(row) > 3 else "",
                    }

    print(f"Fetched {len(s1_records):,} S1 records, {len(target_records):,} Target records.")

    print("\n--- 3. Classifying Missed True Pairs ---")
    categories = Counter()
    country_breakdown = Counter()
    examples_by_cat = defaultdict(list)
    classified_rows = []

    for s1, tid in missed_pairs:
        s1_rec = s1_records.get(s1)
        tx_rec = target_records.get(tid)
        if not s1_rec or not tx_rec:
            continue
        country = s1_rec["country"]
        cat, diag = classify_miss(
            s1_rec["business_name"], s1_rec["business_address"],
            tx_rec["business_name"], tx_rec["business_address"],
            country
        )
        categories[cat] += 1
        country_breakdown[(cat, country)] += 1

        row = {
            "s1_id": s1,
            "target_id": tid,
            "category": cat,
            "country": country,
            "s1_name": s1_rec["business_name"],
            "target_name": tx_rec["business_name"],
            "s1_addr": s1_rec["business_address"],
            "target_addr": tx_rec["business_address"],
            **diag
        }
        classified_rows.append(row)
        if len(examples_by_cat[cat]) < 5:
            examples_by_cat[cat].append(row)

    tot = sum(categories.values())
    print(f"\nTotal Classified Missed Pairs: {tot:,}")
    print(f"{'Category':32s} | {'Count':6s} | {'Share':7s} | {'US':5s} | {'India':5s}")
    print("-" * 65)
    for cat, count in categories.most_common():
        us_c = country_breakdown.get((cat, "US"), 0)
        in_c = country_breakdown.get((cat, "India"), 0)
        print(f"{cat:32s} | {count:6d} | {count/tot*100:6.2f}% | {us_c:5d} | {in_c:5d}")

    # Save summary report
    report = {
        "total_missed_pairs": tot,
        "category_counts": dict(categories.most_common()),
        "category_percentages": {cat: count / tot * 100 for cat, count in categories.most_common()},
        "country_distribution": {f"{cat}_{cntry}": count for (cat, cntry), count in country_breakdown.items()},
        "examples_by_category": {cat: exs for cat, exs in examples_by_cat.items()},
    }

    rep_json = OUT_DIR / "top_misses_report.json"
    rep_json.write_text(json.dumps(report, indent=2))
    print(f"\nSaved JSON report to {rep_json}")

    # Also save detailed CSV
    df_out = pd.DataFrame(classified_rows)
    csv_path = OUT_DIR / "top_misses_classified.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved detailed classification CSV to {csv_path}")


if __name__ == "__main__":
    main()
