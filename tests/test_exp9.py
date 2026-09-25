"""exp9 feature functions: correctness on tiny hand-made inputs (no data files)."""
from collections import Counter

import numpy as np

from experiments import agy2_exp9_features as F9


def test_groups_are_disjoint_and_named():
    feats = F9.NEW_FEATURES
    assert len(feats) == len(set(feats)) == 18
    assert set(F9.GROUPS) == {"A", "B", "C", "E"}


def test_idf_weighting_prefers_rare_shared_tokens():
    df = Counter({"store": 1000, "acme": 2})
    idf = F9.idf_table(df, 10000)
    wj, mx, cov = F9.name_idf_features(["acme store", "acme store", "acme store", ""],
                                       ["acme depot", "global store", "zzz", "acme"], idf, default_idf=max(idf.values()))
    assert wj[0] > wj[1] > 0 and wj[2] == 0 and wj[3] == 0      # rare 'acme' shared beats common 'store'
    assert mx[0] == idf["acme"] and mx[1] == idf["store"]
    assert 0 < cov[1] < cov[0] < 1


def test_char3_cosine_identity_and_disjoint():
    c = F9.char3_cosine(["acme traders", "abc"], ["acme traders", "xyz"], np.array([0, 1]), np.array([0, 1]))
    assert abs(c[0] - 1) < 1e-5 and c[1] == 0


def test_missing_semantics_masks_address_similarity():
    jw, jac, con, both, nm = F9.missing_semantics(np.array([.9, .8, .7]), np.array([.5, .4, .3]), np.array([1., 1., 1.]),
                                                  np.array([0, 1, 1]), np.array([0, 0, 1]), np.array([.2, .3, .4]))
    assert jw[0] == np.float32(.9) and np.isnan(jw[1]) and np.isnan(jw[2])
    assert both.tolist() == [0, 0, 1]
    assert np.isnan(nm[0]) and np.isnan(nm[1]) and nm[2] == np.float32(.4)


def test_number_features_separate_absence_from_conflict():
    qh, th, conf, pconf, qin = F9.number_features(["12", "12", "", "7"], ["12", "14", "14", ""],
                                                  ["560001", "560001", "", ""], ["560001", "560002", "", ""],
                                                  ["12 mg road", "12 mg road", "mg road", "7 lake"],
                                                  ["12 mg road 3", "14 mg road", "14 mg road", "lake"])
    assert conf.tolist() == [0, 1, 0, 0]          # absence is not a conflict
    assert pconf.tolist() == [0, 1, 0, 0]
    assert qh.tolist() == [1, 1, 0, 1] and th.tolist() == [1, 1, 1, 0]
    assert qin.tolist() == [1, 0, 0, 0]


def test_s1_context_is_within_group():
    gap_n, gap_a, rank, strong = F9.s1_context(np.array([0, 0, 0, 1]), np.array([.95, .5, .92, .3]), np.array([.1, .9, .5, .2]))
    assert np.allclose(gap_n, [0, .45, .03, 0], atol=1e-6)
    assert rank.tolist() == [1, 3, 2, 1] and strong.tolist() == [2, 2, 2, 0]
    assert np.allclose(gap_a, [.8, 0, .4, 0], atol=1e-6)


def test_h4_candidate_feature_sets():
    from experiments.agy2_exp9_h4_freeze import candidate_features
    assert candidate_features("B0", {}) == [] and candidate_features("B1", {}) == []
    assert candidate_features("B2", {}) == F9.NEW_FEATURES
    b3 = candidate_features("B3", {"dropped_group": "A"})
    assert set(b3) == set(F9.NEW_FEATURES) - set(F9.GROUPS["A"])
