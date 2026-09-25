"""exp6 helpers: Jaro-Winkler kernel and per-S1 macro F0.5."""
import numpy as np

from experiments.agy2_exp6_integrated import jaro_winkler_pairs, jw_self_test, per_s1_eval


def test_jaro_winkler_textbook_values():
    jw_self_test()
    assert jaro_winkler_pairs(["same"], [0], ["same"], [0])[0] == 1.0


def test_per_s1_eval_macro_f05():
    # S1 0: singleton, nothing predicted -> 1 ; S1 1: singleton, 1 FP -> 0 ; S1 2: t=2, 1 TP + 1 FP -> 1.25/(2+0.5)
    # S1 3: t=1, nothing predicted -> 0
    q = np.array([1, 2, 2]); label = np.array([False, True, False]); sel = np.array([True, True, True])
    f = per_s1_eval(q, label, sel, np.array([0, 0, 2, 1]), 4)
    np.testing.assert_allclose(f, [1.0, 0.0, 0.5, 0.0])
