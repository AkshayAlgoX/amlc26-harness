"""Synthetic tests for B2R Group-R features.

Tests:
  1. ASCII target → kind=0, all five fx_* = NaN
  2. Devanagari target → kind=1, all five fx_* have float values
  3. Mixed batch: correct kind per row, NaN only where kind=0
  4. All outputs are float32
  5. B2 parity: exp9 feature list unchanged at 18 (Groups A,B,C,E)
  6. B2R total = 61 base + 6 R = 67
  7. fold() rules: vowel collapse, word-final a drop, ph→f, w→v, double collapse
  8. skel() rules: digits removed, LEGAL removed, vowels removed, doubles collapsed
  9. Feature names are exactly as specified
  10. Empty-string edge cases
"""
import numpy as np
import pytest

from experiments.agy2_b2r_features import R_FEATURES, fold, skel, compute_r_features
from experiments.agy2_exp9_features import NEW_FEATURES as EXP9_FEATURES, GROUPS as EXP9_GROUPS


# ============================================================
# fold() unit tests
# ============================================================
class TestFold:
    def test_lowercase(self):
        assert fold("HELLO") == fold("hello")

    def test_aa_to_a(self):
        assert "aa" not in fold("Raamesh")
        assert fold("Raamesh") == "ramesh"

    def test_ee_to_i(self):
        assert "ee" not in fold("Sreemathi")

    def test_oo_to_u(self):
        assert "oo" not in fold("Pooja")
        assert fold("Pooja") == "puj"

    def test_word_final_a_drop(self):
        assert fold("rama") == "ram"
        assert fold("rama deva") == "ram dev"

    def test_ph_to_f(self):
        assert fold("Photograph") == "fotograf"

    def test_w_to_v(self):
        assert fold("water") == "vater"

    def test_double_collapse(self):
        assert fold("lloyds") == "loyds"

    def test_empty(self):
        assert fold("") == ""


# ============================================================
# skel() unit tests
# ============================================================
class TestSkel:
    def test_digit_removal(self):
        assert skel("abc 123 def") == "bc df"

    def test_legal_removal(self):
        # "inc" is in LEGAL_SUFFIX_MAP
        assert skel("test inc") == "tst"

    def test_vowel_removal(self):
        assert skel("apple") == "pl"

    def test_double_collapse(self):
        assert skel("broom") == "brm"

    def test_empty(self):
        assert skel("") == ""

    def test_all_legal(self):
        # Only legal tokens → empty after removal → empty skeleton
        assert skel("inc ltd") == ""


# ============================================================
# compute_r_features tests
# ============================================================
class TestComputeRFeatures:
    def test_ascii_target_all_nan(self):
        """ASCII target name → kind=0, five fx_* features are NaN."""
        q = np.array(["ABC Corp"], dtype=object)
        t = np.array(["ABC Corporation"], dtype=object)
        kind, c3, jw, tj, sj, sc = compute_r_features(q, t)
        assert kind[0] == 0.0
        for arr in (c3, jw, tj, sj, sc):
            assert np.isnan(arr[0]), f"Expected NaN for ASCII target, got {arr[0]}"

    def test_devanagari_target_has_values(self):
        """Devanagari target → kind=1, all fx_* are finite floats."""
        q = np.array(["Ramesh Enterprises"], dtype=object)
        t = np.array(["\u0930\u092e\u0947\u0936 \u090f\u0902\u091f\u0930\u092a\u094d\u0930\u093e\u0907\u091c\u0947\u091c"], dtype=object)
        kind, c3, jw, tj, sj, sc = compute_r_features(q, t)
        assert kind[0] == 1.0
        for name, arr in zip(R_FEATURES[1:], [c3, jw, tj, sj, sc]):
            assert not np.isnan(arr[0]), f"{name} should not be NaN for non-ASCII target"
            assert 0.0 <= arr[0] <= 1.0, f"{name}={arr[0]} not in [0,1]"

    def test_mixed_batch(self):
        """Mixed ASCII and non-ASCII targets in a single batch."""
        q = np.array(["Test Inc", "Sweets Ltd", "Hello"], dtype=object)
        t = np.array(["Test Inc", "\u0938\u094d\u0935\u0940\u091f\u094d\u0938", "World"], dtype=object)
        kind, c3, jw, tj, sj, sc = compute_r_features(q, t)
        # [0] ASCII → kind=0, NaN
        assert kind[0] == 0.0
        assert np.isnan(jw[0])
        # [1] Devanagari → kind=1, values
        assert kind[1] == 1.0
        assert not np.isnan(jw[1])
        # [2] ASCII → kind=0, NaN
        assert kind[2] == 0.0
        assert np.isnan(jw[2])

    def test_accented_latin_target(self):
        """French accented name (non-ASCII but Latin) → kind=1."""
        q = np.array(["Cafe Beaute"], dtype=object)
        t = np.array(["Café Beauté"], dtype=object)
        kind, c3, jw, tj, sj, sc = compute_r_features(q, t)
        assert kind[0] == 1.0
        assert not np.isnan(jw[0])

    def test_float32_dtype(self):
        """All outputs are float32."""
        q = np.array(["Test"], dtype=object)
        t = np.array(["\u0930\u092e\u0947\u0936"], dtype=object)
        for arr in compute_r_features(q, t):
            assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"

    def test_empty_strings(self):
        """Empty raw names do not crash."""
        q = np.array(["", "Test"], dtype=object)
        t = np.array(["", ""], dtype=object)
        kind, c3, jw, tj, sj, sc = compute_r_features(q, t)
        assert kind[0] == 0.0
        assert kind[1] == 0.0

    def test_kannada_target(self):
        """Kannada script target → kind=1."""
        q = np.array(["Bangalore Sweets"], dtype=object)
        t = np.array(["\u0cac\u0cc6\u0c82\u0c97\u0cb3\u0cc2\u0cb0\u0cc1 \u0cb8\u0ccd\u0cb5\u0cc0\u0c9f\u0ccd\u0cb8\u0ccd"], dtype=object)
        kind, c3, jw, tj, sj, sc = compute_r_features(q, t)
        assert kind[0] == 1.0
        assert not np.isnan(jw[0])


# ============================================================
# B2 parity & feature count tests
# ============================================================
class TestB2Parity:
    def test_exp9_features_unchanged(self):
        """exp9 still has exactly 18 features (Groups A,B,C,E) — no contamination."""
        assert len(EXP9_FEATURES) == 18
        assert set(EXP9_GROUPS.keys()) == {"A", "B", "C", "E"}

    def test_r_features_count(self):
        """Group R has exactly 6 features."""
        assert len(R_FEATURES) == 6

    def test_r_features_names_exact(self):
        """Feature names are exactly as specified, in order."""
        assert R_FEATURES == [
            "t_name_kind",
            "fx_name_char3",
            "fx_name_jw",
            "fx_name_tokjacc",
            "fx_name_skeljacc",
            "fx_name_skelcov",
        ]

    def test_no_overlap_with_exp9(self):
        """R features do not collide with exp9 features."""
        assert not set(R_FEATURES) & set(EXP9_FEATURES)

    def test_b2r_total_67(self):
        """B2R = 43 base + 18 exp9 + 6 R = 67."""
        from experiments.agy2_exp7_h3 import M_FEATS
        b2r_total = len(M_FEATS) + len(EXP9_FEATURES) + len(R_FEATURES)
        assert b2r_total == 67, f"Expected 67, got {b2r_total}"

# ============================================================
# Stage-2 Feature Layout Tests
# ============================================================
class TestStage2:
    def test_stage2_features_count(self):
        import json
        with open("experiments/configs/h4_p2_frozen.json", "r") as f:
            cfg = json.load(f)
        s2_features = cfg["matcher_s2"]["features"]
        assert len(s2_features) == 9
        
    def test_b2_features_unchanged(self):
        import json
        with open("experiments/configs/h4_p2_frozen.json", "r") as f:
            cfg = json.load(f)
        b2_features = cfg["matcher"]["features"]
        # Originally there were 67 features (61 B2 + 6 B2R)
        assert len(b2_features) == 67
