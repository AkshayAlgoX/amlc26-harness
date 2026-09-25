"""exp7 leakage and parser invariants."""
from pathlib import Path

import numpy as np

from experiments.agy2_exp7_h3 import crossfit, fold_of, inner_es_flag, stat_fit_mask
from experiments.agy2_exp7_prep_h3 import load_canonical, read_tsv_contract, read_tsv_plain_split, salted_holdout

ROOT = Path(__file__).resolve().parents[1]
SRC_ROWS = [
    "entity_id\tbusiness_name\tbusiness_address\tcountry",
    'S2-1\t"Joe\'s" Diner\t12 Main St\tUS',                  # quoted token at field start
    'S2-2\t"Quoted, Name Inc"\t"45 MG Road, Pune"\tIndia',   # fully quoted fields
    'S2-3\tAcme 5" Pipes\t7 Oak Ave\tUS',                     # quote inside a field
    'S2-4\t  Padded  \t\tUS',                                  # whitespace + empty address
    "S2-5\tPlain Name\t9 Lake Dr\tUS",
]


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def test_contract_reader_equals_production_loader_on_quoted_rows(tmp_path):
    p = _write(tmp_path, "train_source2.tsv", SRC_ROWS)
    canon = load_canonical(p)
    assert read_tsv_contract(p)[list(canon.columns)].equals(canon)
    assert canon.business_name.tolist()[:2] == ["Joe's Diner", "Quoted, Name Inc"]
    assert canon.business_address.tolist()[3] == ""
    plain = read_tsv_plain_split(p)[list(canon.columns)]
    assert (plain.values != canon.values).any(axis=1).sum() == 2      # legacy split keeps the quote chars


def test_contract_reader_equals_loader_ground_truth(tmp_path):
    p = _write(tmp_path, "train_ground_truth.tsv", ["source1_entity_id\tmatched_entity_ids",
                                                    "S1-1\tS2-1,S3-2", "S1-2\t", "S1-3\t S2-9 "])
    canon = load_canonical(p)
    assert read_tsv_contract(p)[list(canon.columns)].equals(canon)


def test_fold_assignment_is_a_pure_function_of_the_id():
    ids = [f"S1-{i}" for i in range(2000)]
    f1 = fold_of(ids, "salt", 5)
    f2 = fold_of(list(reversed(ids)), "salt", 5)[::-1]
    assert (f1 == f2).all() and set(f1.tolist()) == {0, 1, 2, 3, 4}
    assert not (f1 == fold_of(ids, "other", 5)).all()
    assert 0.05 < inner_es_flag(ids, "salt").mean() < 0.15


def test_crossfit_never_scores_a_group_with_a_model_trained_on_it():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 300, 5000)
    fold_g = fold_of([f"S1-{g}" for g in range(300)], "salt", 5)
    fold_row = fold_g[groups]
    train_row = groups < 200                                      # groups >= 200 play the evaluation role
    used = {}

    def fit(tr, k):
        used[k] = set(groups[tr].tolist())
        return k

    def predict(k, te):
        return np.array([1.0 if groups[i] in used[k] else 0.0 for i in te]) + 10 * k

    pred = crossfit(fold_row, train_row, fit, predict, 5)
    assert (pred % 10 == 0).all()                                   # no row scored by a model that saw its group
    assert (pred // 10 == fold_row).all()                           # every row (incl. eval) scored by its own fold
    assert all(g < 200 for s in used.values() for g in s)          # evaluation groups never trained on


def test_h3_holdout_disjoint_deterministic_and_salted():
    ids = [f"S1-{i}" for i in range(10000)]
    excluded = set(ids[:3000])
    h = salted_holdout(ids, excluded, 1000, "amlc26-h3-holdout-v1")
    assert len(h) == 1000 and not (set(h) & excluded)
    assert salted_holdout(list(reversed(ids)), excluded, 1000, "amlc26-h3-holdout-v1") == h
    assert salted_holdout(ids, excluded, 1000, "other") != h


def test_text_statistics_exclude_evaluation_s1():
    m = stat_fit_mask(["S1-1", "S1-2", "S1-3"], {"S1-2"})
    assert m.tolist() == [True, False, True]


def test_exp7_source_has_no_legacy_reader_or_calibration():
    src = (ROOT / "experiments" / "agy2_exp7_h3.py").read_text()
    assert "read_tsv(" not in src and ".split(\"\\t\")" not in src
    assert "Isotonic" not in src
    assert "lgb_crossfit(P, PR_FEATS" in src and "lgb_crossfit(M, M_FEATS" in src   # same crossfit for both stages
