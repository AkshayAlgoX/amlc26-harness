"""Official Submission Validator integration for AMLC26 Business Entity Resolution.

Implements all formatting, syntax, and integrity checks from the official
utils/validate_submission.py tool.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DELIM = "\t"
MAX_EXAMPLES = 5
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    matched_mapping: Optional[Dict[str, Set[str]]] = None
    candidate_mapping: Optional[Dict[str, Set[str]]] = None


def read_ids(path: str | Path) -> Set[str]:
    """Return the set of first-column entity IDs from a source TSV."""
    with open(path, encoding="utf-8") as f:
        next(f, None)  # skip header
        return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}


def examples(items: Iterable[str]) -> str:
    """Return a short, human-readable sample of items for an error message."""
    items_sorted = sorted(items)
    shown = ", ".join(items_sorted[:MAX_EXAMPLES])
    if len(items_sorted) > MAX_EXAMPLES:
        return f"{len(items_sorted)} total, e.g. {shown}, ..."
    return shown


def load_match_targets(test_dir: str | Path, warnings: List[str]) -> Optional[Set[str]]:
    """Return the set of valid S2/S3 match IDs from test directory."""
    targets: Set[str] = set()
    test_path = Path(test_dir)
    for name in ("test_source2.tsv", "test_source3.tsv"):
        p = test_path / name
        if not p.is_file():
            warnings.append(
                f"{p} not found — skipping ID-existence check. Provide test_source2/3.tsv to enable it."
            )
            return None
        targets |= read_ids(p)
    return targets


def validate_id_list_file(
    path: str | Path,
    expected_header: List[str],
    col_label: str,
    required: Set[str],
    valid_ids: Optional[Set[str]],
    errors: List[str],
) -> Optional[Dict[str, Set[str]]]:
    """Validate one results-style TSV (matching or candidate)."""
    p = Path(path)
    if not p.is_file():
        errors.append(f"File not found: {p}")
        return None

    name = p.name
    mapping: Dict[str, Set[str]] = {}
    seen, dup_rows, intra_dupes = set(), set(), set()
    self_matches, wrong_prefix, unknown = set(), set(), set()
    n_rows = empties = 0

    with open(p, encoding="utf-8") as f:
        header = f.readline()
        if not header:
            errors.append(f"{name} is empty.")
            return None
        if DELIM not in header and "," in header:
            errors.append(
                f"{name}: header has no TAB but contains commas — the file looks "
                "COMMA-separated. Submissions must be TAB-separated (.tsv); "
                "write it with df.to_csv(sep='\\t', index=False)."
            )
            return None

        cols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
        if cols != expected_header:
            errors.append(
                f"{name}: unexpected header {cols}. "
                f"Expected exactly {expected_header} (tab-separated)."
            )
            return None

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(f"{name}: malformed row (no tab) at line {line_num}: {line.rstrip()!r}")
                continue

            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)

            ids = rest.rstrip("\n").split(",") if rest.strip() else []
            if not ids:
                empties += 1
                mapping[s1] = set()
                continue

            if len(ids) != len(set(ids)):
                intra_dupes.add(s1)

            id_set = set(ids)
            mapping[s1] = id_set
            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)
                elif valid_ids is not None and mid not in valid_ids:
                    unknown.add(mid)

    findings = [
        (dup_rows, f"{name}: duplicate source1_entity_id row(s): {{ex}}. Each S1 entity may appear on only one row."),
        (intra_dupes, f"{name}: repeated ID inside a {col_label} list for: {{ex}}. No duplicate IDs allowed within a list."),
        (self_matches, f"{name}: {col_label} contains Source-1 IDs (self-matches): {{ex}}. Only S2-/S3- IDs allowed."),
        (wrong_prefix, f"{name}: {col_label} contains IDs without an S2-/S3- prefix: {{ex}}."),
        (unknown, f"{name}: {col_label} references IDs not in the test Source-2/3 files: {{ex}}."),
        (required - seen, f"{name}: required S1 entity(ies) missing: {{ex}}. Every entity in test_source1.tsv needs a row."),
        (seen - required, f"{name}: row(s) using an S1 ID that is not in the test set: {{ex}}."),
    ]
    for offenders, msg in findings:
        if offenders:
            errors.append(msg.format(ex=examples(offenders)))

    return mapping


def validate_submission(
    matching_path: str | Path,
    candidate_path: Optional[str | Path] = None,
    test_dir: Optional[str | Path] = None,
    required_s1_ids: Optional[Set[str]] = None,
    check_ids: bool = False,
) -> ValidationResult:
    """Validate matching_results.tsv and optional candidate_pairs.tsv."""
    errors: List[str] = []
    warnings: List[str] = []

    required: Set[str] = set()
    if required_s1_ids is not None:
        required = set(required_s1_ids)
    elif test_dir is not None:
        s1_file = Path(test_dir) / "test_source1.tsv"
        if not s1_file.is_file():
            errors.append(f"Test source1 file not found: {s1_file}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        required = read_ids(s1_file)
    else:
        errors.append("Either test_dir or required_s1_ids must be provided to validate S1 entity coverage.")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    valid_ids = None
    if check_ids and test_dir:
        valid_ids = load_match_targets(test_dir, warnings)

    matched = validate_id_list_file(
        matching_path, MATCHING_HEADER, "matched_entity_ids", required, valid_ids, errors
    )

    candidate = None
    if candidate_path and Path(candidate_path).is_file():
        candidate = validate_id_list_file(
            candidate_path, CANDIDATE_HEADER, "candidate_entity_ids", required, valid_ids, errors
        )
    elif candidate_path:
        warnings.append(f"{candidate_path} not found — skipping candidate_pairs.tsv checks.")

    if matched is not None and candidate is not None:
        offenders = {s1 for s1, mids in matched.items() if mids - candidate.get(s1, set())}
        if offenders:
            warnings.append(
                f"{len(offenders)} S1 entity(ies) have matched IDs not present in candidate_pairs.tsv, "
                f"e.g. {examples(offenders)}. Final matches normally come from your blocking candidates."
            )

    return ValidationResult(
        is_valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
        matched_mapping=matched,
        candidate_mapping=candidate,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate AMLC26 ER submission files.")
    parser.add_argument("--matching", "-m", default="output/matching_results.tsv")
    parser.add_argument("--candidate", "-c", default="output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", "-t", default="dataset/test")
    parser.add_argument("--check-ids", action="store_true")
    args = parser.parse_args()

    res = validate_submission(
        matching_path=args.matching,
        candidate_path=args.candidate,
        test_dir=args.test_dir,
        check_ids=args.check_ids,
    )

    for w in res.warnings:
        print(f"WARNING: {w}")
    if res.errors:
        print(f"FAIL — {len(res.errors)} issue(s) found:")
        for i, err in enumerate(res.errors, 1):
            print(f"  {i}. {err}")
        sys.exit(1)
    print("PASS — no blocking issues found. Safe to submit.")
    sys.exit(0)


if __name__ == "__main__":
    main()
