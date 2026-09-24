"""Source identity must come only from the entity_id prefix (never numeric magnitude)."""
import pytest

from src.er.ids import code_source, decode_target_id, encode_target_id, is_target_id, source_of


@pytest.mark.parametrize("eid,src", [
    ("S2-166376419", "S2"),   # the reported misclassification (number >= 10,000,000)
    ("S2-999999999", "S2"),
    ("S3-1", "S3"),
    ("S3-999999999", "S3"),
    ("S1-252242682", "S1"),
    ("S2-1", "S2"),
])
def test_source_from_prefix_only(eid, src):
    assert source_of(eid) == src


@pytest.mark.parametrize("eid", ["S2-166376419", "S2-999999999", "S3-1", "S3-999999999", "S2-10000005", "S3-5"])
def test_encode_roundtrip_and_source(eid):
    code = encode_target_id(eid)
    assert decode_target_id(code) == eid
    assert code_source(code) == source_of(eid)


def test_no_s2_s3_collision_at_old_offset():
    # the removed heuristic mapped S2-10000005 and S3-5 to the same integer
    assert encode_target_id("S2-10000005") != encode_target_id("S3-5")


@pytest.mark.parametrize("bad", ["", "166376419", "S4-1", "S2-", "S2-12a", "s2-1", "S2_1"])
def test_invalid_ids_rejected(bad):
    with pytest.raises(ValueError):
        source_of(bad)


def test_target_rules():
    assert is_target_id("S2-1") and is_target_id("S3-9") and not is_target_id("S1-1") and not is_target_id("x")
    with pytest.raises(ValueError):
        encode_target_id("S1-5")
