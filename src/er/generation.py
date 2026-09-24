"""TSV generation module for AMLC26 Business Entity Resolution submission files:

  1. candidate_pairs.tsv   (blocking output: source1_entity_id \t candidate_entity_ids)
  2. matching_results.tsv  (final matches:    source1_entity_id \t matched_entity_ids)

Guarantees exact TSV format, tab delimiters, comma-separated lists with no intra-list
duplicates, exactly one row per S1 entity, and S2-/S3- prefix validation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from .loader import format_id_list


def _sanitize_ids(ids: Iterable[Any], s1_id: str, field_name: str) -> List[str]:
    """Sanitize and validate candidate/matched IDs:

    - Removes duplicates preserving order
    - Forbids S1 self-matches
    - Enforces S2- or S3- prefixes
    """
    seen: Set[str] = set()
    cleaned: List[str] = []

    for item in ids:
        item_str = str(item).strip()
        if not item_str:
            continue
        if item_str == s1_id or item_str.startswith("S1-"):
            raise ValueError(f"Self-match error for {s1_id}: {field_name} cannot contain Source 1 ID '{item_str}'")
        if not (item_str.startswith("S2-") or item_str.startswith("S3-")):
            raise ValueError(f"Invalid prefix error for {s1_id}: ID '{item_str}' must start with 'S2-' or 'S3-'")
        if item_str not in seen:
            seen.add(item_str)
            cleaned.append(item_str)

    return cleaned


def generate_candidate_pairs_tsv(
    candidate_map: Dict[str, Iterable[str]],
    all_s1_ids: Iterable[str],
    output_path: str | Path,
) -> Path:
    """Generate candidate_pairs.tsv file.

    Format:
      source1_entity_id \t candidate_entity_ids
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in all_s1_ids:
            s1_str = str(s1).strip()
            raw_cands = candidate_map.get(s1_str, [])
            clean_cands = _sanitize_ids(raw_cands, s1_str, "candidate_entity_ids")
            cand_str = ",".join(clean_cands)
            f.write(f"{s1_str}\t{cand_str}\n")

    return out_file


def generate_matching_results_tsv(
    matching_map: Dict[str, Iterable[str]],
    all_s1_ids: Iterable[str],
    output_path: str | Path,
) -> Path:
    """Generate matching_results.tsv file.

    Format:
      source1_entity_id \t matched_entity_ids
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in all_s1_ids:
            s1_str = str(s1).strip()
            raw_matches = matching_map.get(s1_str, [])
            clean_matches = _sanitize_ids(raw_matches, s1_str, "matched_entity_ids")
            match_str = ",".join(clean_matches)
            f.write(f"{s1_str}\t{match_str}\n")

    return out_file


def check_candidate_subset(
    matching_map: Dict[str, Iterable[str]],
    candidate_map: Dict[str, Iterable[str]],
) -> List[Tuple[str, Set[str]]]:
    """Verify that all matched IDs are a subset of candidate IDs.

    Returns list of (s1_id, offending_ids_not_in_candidates).
    """
    violations = []
    for s1, matches in matching_map.items():
        s1_str = str(s1).strip()
        match_set = set(str(m).strip() for m in matches if str(m).strip())
        cand_set = set(str(c).strip() for c in candidate_map.get(s1_str, []) if str(c).strip())
        diff = match_set - cand_set
        if diff:
            violations.append((s1_str, diff))
    return violations


def generate_submission_package(
    matching_map: Dict[str, Iterable[str]],
    candidate_map: Dict[str, Iterable[str]],
    all_s1_ids: Iterable[str],
    output_dir: str | Path,
) -> Tuple[Path, Path]:
    """Generate both output files (matching_results.tsv and candidate_pairs.tsv) in output_dir.

    Warns if any matches are not in candidate_pairs.tsv.
    Returns (matching_path, candidate_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_list = [str(s).strip() for s in all_s1_ids]

    cand_path = out_dir / "candidate_pairs.tsv"
    match_path = out_dir / "matching_results.tsv"

    generate_candidate_pairs_tsv(candidate_map, s1_list, cand_path)
    generate_matching_results_tsv(matching_map, s1_list, match_path)

    # Check subset relationship
    violations = check_candidate_subset(matching_map, candidate_map)
    if violations:
        import warnings
        sample = violations[:3]
        warnings.warn(
            f"{len(violations)} S1 entities have matched IDs that are not present in candidate_pairs.tsv (e.g. {sample}). "
            f"Final matches should normally be a subset of candidate pairs."
        )

    return match_path, cand_path
