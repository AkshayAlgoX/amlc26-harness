"""Entity-ID helpers — the ONLY place source identity is derived.

Source is determined exclusively from the entity_id prefix ("S1-", "S2-", "S3-").
Never infer the source from the numeric part's magnitude, a threshold/range, row position,
or string length: real IDs are unpadded numbers up to ~1e9 in every source, so S2 and S3
numeric ranges overlap completely.

For memory-compact integer sets, `encode_target_id` packs the numeric part together with an
explicit source tag taken from the prefix (code = number * 4 + tag). The tag is a lossless copy of
the prefix — not an inference — so `code_source` and `decode_target_id` are exact inverses.
"""
from __future__ import annotations

SOURCES = ("S1", "S2", "S3")
_TAG = {"S1": 1, "S2": 2, "S3": 3}
_SRC = {v: k for k, v in _TAG.items()}


def source_of(entity_id: str) -> str:
    """Return 'S1', 'S2' or 'S3' from the entity_id prefix. Raises ValueError otherwise."""
    s = str(entity_id).strip()
    prefix, sep, number = s.partition("-")
    if not sep or prefix not in _TAG or not number.isdigit():
        raise ValueError(f"not a valid entity_id (expected S1-/S2-/S3-<digits>): {entity_id!r}")
    return prefix


def is_target_id(entity_id: str) -> bool:
    """True for Source 2 / Source 3 IDs (the only IDs allowed in match/candidate lists)."""
    try:
        return source_of(entity_id) in ("S2", "S3")
    except ValueError:
        return False


def encode_target_id(entity_id: str) -> int:
    """Lossless int encoding of an S2/S3 entity_id (source tag carried explicitly in the low bits)."""
    src = source_of(entity_id)
    if src not in ("S2", "S3"):
        raise ValueError(f"only S2/S3 IDs can be encoded as targets: {entity_id!r}")
    number = str(entity_id).strip()[3:]
    if len(number) > 1 and number.startswith("0"):
        raise ValueError(f"leading zeros would not round-trip: {entity_id!r}")
    return int(number) * 4 + _TAG[src]


def code_source(code: int) -> str:
    """Source of an encoded target ID — read from the explicit tag written by encode_target_id."""
    src = _SRC.get(int(code) & 3)
    if src not in ("S2", "S3"):
        raise ValueError(f"not an encoded S2/S3 target code: {code!r}")
    return src


def decode_target_id(code: int) -> str:
    """Exact inverse of encode_target_id."""
    return f"{code_source(code)}-{int(code) >> 2}"
