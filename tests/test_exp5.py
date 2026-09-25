"""exp5 helpers: prefix-only source identity, deterministic disjoint holdout, oracle F0.5."""
import numpy as np

from experiments.agy2_exp5_corrected_holdout import oracle_f05, select_holdout, self_test


def test_self_test_distinguishes_s2_s3_same_number():
    self_test()   # raises if S2-166376419 and S3-166376419 could be confused


def test_holdout_deterministic_disjoint_and_sized():
    ids = [f"S1-{i}" for i in range(1, 5001)]
    excluded = set(ids[:1000])
    a = select_holdout(ids, excluded, 500)
    b = select_holdout(list(reversed(ids)), excluded, 500)
    assert a == b                                   # independent of input order
    assert len(a) == 500 and not (set(a) & excluded)


def test_oracle_f05():
    got = oracle_f05(np.array([0, 0, 2, 1, 3]), np.array([0, 2, 2, 2, 4]))
    np.testing.assert_allclose(got, [1.0, 0.0, 1.0, 1.25 / 1.5, 3.75 / 4.0])
