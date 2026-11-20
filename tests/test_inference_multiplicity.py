"""
tests/test_inference_multiplicity.py  -  checks of Bonferroni, Holm and Benjamini-Hochberg (src/inference/multiplicity.py).

WHAT THIS FILE IS
    Automatic tests: a worked example done by hand, the order relations that must always hold, bad inputs, and a simulation
    that shows what the corrections are FOR (with ten true null hypotheses, uncorrected tests give a false alarm about 40% of
    the time, Holm about 5%).

WHY WE NEED IT
    The correction decides which of the ten hypotheses may be called "supported". An off-by-one in the step-down formula would
    change the verdicts without any error message.

WHAT HAPPENS WHEN IT RUNS
    pytest runs each test in a fraction of a second; the simulation uses a fixed seed.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Protect the one place where the project decides what it is allowed to claim.
"""
import numpy as np
import pytest

from src.inference.multiplicity import (adjust, benjamini_hochberg, bonferroni, family_wise_false_alarm_chance, holm)


def test_a_worked_example_done_by_hand():
    p = [0.01, 0.04, 0.03, 0.005]
    # Holm: sorted 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1 = 0.02, 0.03, 0.06, 0.04 -> running maximum 0.02, 0.03, 0.06, 0.06
    assert holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert bonferroni(p) == pytest.approx([0.04, 0.16, 0.12, 0.02])
    # BH: sorted times 4/1, 4/2, 4/3, 4/4 = 0.02, 0.02, 0.04, 0.04 -> running minimum from the top
    assert benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_a_second_textbook_example():
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    assert holm(p) == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09])
    assert benjamini_hochberg(p) == pytest.approx([0.05] * 5)


def test_one_p_value_is_left_alone():
    for method in ("bonferroni", "holm", "bh"):
        assert adjust([0.03], method) == pytest.approx([0.03])


def test_the_order_of_the_input_is_kept():
    p = np.array([0.5, 0.001, 0.2, 0.04])
    adjusted = holm(p)
    assert np.argmin(adjusted) == 1 and np.argmax(adjusted) == 0


def test_the_order_relations_hold_for_random_p_values():
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = rng.random(rng.integers(2, 15)) ** 2
        assert np.all(bonferroni(p) >= holm(p) - 1e-12)
        assert np.all(holm(p) >= benjamini_hochberg(p) - 1e-12)
        assert np.all(benjamini_hochberg(p) >= p - 1e-12)
        assert np.all(holm(p) <= 1.0)


def test_a_bigger_raw_p_value_never_gets_a_smaller_adjusted_one():
    rng = np.random.default_rng(2)
    for _ in range(100):
        p = rng.random(10)
        order = np.argsort(p)
        for method in (holm, benjamini_hochberg, bonferroni):
            assert np.all(np.diff(method(p)[order]) >= -1e-12)


def test_ties_are_treated_alike():
    assert holm([0.01, 0.01, 0.5])[0] == pytest.approx(holm([0.01, 0.01, 0.5])[1])


@pytest.mark.parametrize("bad", [[], [0.5, float("nan")], [-0.1, 0.2], [0.2, 1.5]])
def test_bad_input_is_refused(bad):
    with pytest.raises(ValueError):
        holm(bad)


def test_unknown_method_is_refused():
    with pytest.raises(ValueError):
        adjust([0.1, 0.2], "sidak-ish")


def test_the_chance_of_at_least_one_false_alarm():
    assert family_wise_false_alarm_chance(1) == pytest.approx(0.05)
    assert family_wise_false_alarm_chance(10) == pytest.approx(1 - 0.95 ** 10)
    with pytest.raises(ValueError):
        family_wise_false_alarm_chance(0)


def test_simulation_what_the_correction_is_for():
    rng = np.random.default_rng(3)
    families = rng.random((4000, 10))                      # 4,000 families of ten tests where every null hypothesis is true
    uncorrected = np.mean(families.min(axis=1) < 0.05)
    holm_rate = np.mean([holm(row).min() < 0.05 for row in families[:1500]])
    assert 0.36 < uncorrected < 0.44                       # about 1 - 0.95^10 = 40%
    assert holm_rate < 0.075                               # about 5%
