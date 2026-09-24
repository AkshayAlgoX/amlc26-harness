"""DAY 1: add the official worked example from the 2026 problem PDF as a test here."""
import pytest

from src.metric import f1_extraction, greater_is_better, is_better, macro_f05, mape_score, score, smape


def test_macro_f05_official_2026_example():
    # Official README:
    # S1-00001: pred = [S2-00047, S2-00193, S3-00812], gt = [S2-00047, S3-00812]
    # Precision = 2/3, Recall = 2/2 = 1.0 -> F_0.5 = 5/7 ≈ 0.7142857
    gt = ["S2-00047,S3-00812"]
    pred = ["S2-00047,S2-00193,S3-00812"]
    assert macro_f05(gt, pred) == pytest.approx(5 / 7, abs=1e-5)
    assert score(gt, pred, "macro_f05") == pytest.approx(0.7142857, abs=1e-5)


def test_macro_f05_singletons():
    # Singleton correctly predicted empty -> 1.0
    assert macro_f05([""], [""]) == 1.0
    # Singleton incorrectly predicted with false match -> 0.0
    assert macro_f05([""], ["S2-00001"]) == 0.0
    # Non-singleton incorrectly predicted empty -> 0.0
    assert macro_f05(["S2-00001"], [""]) == 0.0
    # Disjoint match -> 0.0
    assert macro_f05(["S2-00001"], ["S2-00002"]) == 0.0
    # Macro average across a singleton and the official example
    # (1.0 + 5/7) / 2 = 12/14 = 6/7 ≈ 0.8571428
    assert macro_f05(["", "S2-00047,S3-00812"], ["", "S2-00047,S2-00193,S3-00812"]) == pytest.approx(6 / 7, abs=1e-5)


def test_direction_helpers():
    assert greater_is_better("smape") is False
    assert greater_is_better("f1_extraction") is True
    assert greater_is_better("macro_f05") is True
    assert is_better(18.0, 19.0, "smape")
    assert is_better(0.9, 0.8, "f1_extraction")
    assert is_better(0.85, 0.75, "macro_f05")


def test_unknown_metric():
    with pytest.raises(KeyError):
        score([1], [1], "nope")

