"""DAY 1: add the official worked example from the 2026 problem PDF as a test here."""
import pytest

from src.metric import f1_extraction, greater_is_better, is_better, mape_score, score, smape


def test_smape_official_2025_example():
    # PDF: actual=100, predicted=120 -> 18.18%
    assert smape([100], [120]) == pytest.approx(18.1818, abs=1e-3)


def test_smape_edges():
    assert smape([0], [0]) == 0.0
    assert smape([1], [0]) == pytest.approx(200.0)
    assert smape([100, 100], [120, 80]) == pytest.approx((18.1818 + 22.2222) / 2, abs=1e-3)


def test_mape_score_2023_style():
    assert mape_score([100, 200], [90, 220]) == pytest.approx(90.0)
    assert mape_score([1], [100]) == 0.0  # floored at 0


def test_f1_extraction_2024_style():
    gt = ["10 gram", "", "5 ounce", "3 gram", ""]
    out = ["10 gram", "2 gram", "", "4 gram", ""]
    # TP=1; FP=2 (pred on blank gt, wrong value); FN=1; TN=1 -> P=1/3, R=1/2 -> F1=0.4
    assert f1_extraction(gt, out) == pytest.approx(0.4)
    assert f1_extraction(["a"], [""]) == 0.0


def test_direction_helpers():
    assert greater_is_better("smape") is False
    assert greater_is_better("f1_extraction") is True
    assert is_better(18.0, 19.0, "smape")
    assert is_better(0.9, 0.8, "f1_extraction")


def test_unknown_metric():
    with pytest.raises(KeyError):
        score([1], [1], "nope")
