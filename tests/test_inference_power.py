"""
tests/test_inference_power.py  -  checks of power, sample size and equivalence tests (src/inference/power.py).

WHAT THIS FILE IS
    Automatic tests: textbook values (a standardised difference of 0.5 with 64 per group gives 80% power), the limit cases (no
    effect gives power equal to the significance level), a simulation that compares the formula with the rejection rate of the real
    test, and the equivalence-test identity (the test says "equivalent" exactly when the 90% confidence interval lies inside the
    margin).

WHY WE NEED IT
    The "smallest effect we could have detected" is the honest companion of every non-significant result in the notebook.

WHAT HAPPENS WHEN IT RUNS
    pytest runs the tests; the simulations use fixed seeds.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    A statement like "we found no difference" is only as good as the power behind it.
"""
import numpy as np
import pytest
from scipy import stats

from src.inference import classical, power


def test_power_with_no_effect_is_the_significance_level():
    assert power.power_two_proportions(0.3, 0.3, 200, 200) == pytest.approx(0.05, abs=1e-9)
    assert power.power_two_sample_t(0.0, 50, 50) == pytest.approx(0.05, abs=1e-9)


def test_textbook_values():
    assert power.power_two_sample_t(0.5, 64, 64) == pytest.approx(0.80, abs=0.005)
    assert power.required_n_per_group(0.6, 0.5) == (388, 388)
    assert power.power_two_proportions(0.6, 0.5, 388, 388) == pytest.approx(0.80, abs=0.005)
    n1, n2 = power.required_n_per_group(0.6, 0.5, ratio=2.0)
    assert n2 == 2 * n1 and n1 < 388


def test_power_grows_with_sample_size_and_effect_and_shrinks_with_the_design_effect():
    assert power.power_two_proportions(0.4, 0.3, 500, 500) > power.power_two_proportions(0.4, 0.3, 200, 200)
    assert power.power_two_proportions(0.5, 0.3, 200, 200) > power.power_two_proportions(0.4, 0.3, 200, 200)
    assert power.power_two_proportions(0.4, 0.3, 500, 500, design_effect=2.0) < power.power_two_proportions(0.4, 0.3, 500, 500)
    assert power.power_two_sample_t(0.3, 200, 200) > power.power_two_sample_t(0.3, 100, 100)
    with pytest.raises(ValueError):
        power.power_two_proportions(0.4, 0.3, 500, 500, design_effect=0.5)


def test_the_formula_agrees_with_a_simulation_of_the_real_test():
    rng = np.random.default_rng(1)
    p1, p2, n1, n2 = 0.36, 0.29, 495, 449
    rejected = np.mean([classical.two_proportion_ztest(rng.binomial(n1, p1), n1, rng.binomial(n2, p2), n2).p_value < 0.05 for _ in range(3000)])
    assert power.power_two_proportions(p1, p2, n1, n2) == pytest.approx(rejected, abs=0.04)


def test_t_test_power_agrees_with_a_simulation():
    rng = np.random.default_rng(2)
    rejected = np.mean([stats.ttest_ind(rng.normal(0.3, 1, 100), rng.normal(0, 1, 100)).pvalue < 0.05 for _ in range(3000)])
    assert power.power_two_sample_t(0.3, 100, 100) == pytest.approx(rejected, abs=0.04)


def test_minimum_detectable_effects_reach_the_target_power():
    mdd = power.minimum_detectable_difference(0.33, 122, 822)
    assert power.power_two_proportions(0.33 + mdd, 0.33, 122, 822) == pytest.approx(0.8, abs=0.005)
    assert 0.12 < mdd < 0.14
    assert power.minimum_detectable_difference(0.33, 122, 822, design_effect=1.5) > mdd
    assert power.minimum_detectable_difference(0.33, 500, 500) < mdd
    d = power.minimum_detectable_effect_t(100, 100)
    assert power.power_two_sample_t(d, 100, 100) == pytest.approx(0.8, abs=0.005)
    assert d == pytest.approx(0.396, abs=0.01)
    assert np.isnan(power.minimum_detectable_difference(0.9, 3, 3))


def test_minimum_detectable_effect_t_works_for_very_small_samples():
    # nine months against ten months: the old fixed search limit (d = 20) made the non-central t distribution return "not a number"
    d = power.minimum_detectable_effect_t(9, 10)
    assert np.isfinite(d)
    assert power.power_two_sample_t(d, 9, 10) == pytest.approx(0.8, abs=0.005)
    assert 1.2 < d < 1.6                                           # about 1.4 standard deviations, as every power table says
    assert power.minimum_detectable_effect_t(3, 3) > d             # fewer observations, a bigger effect is needed
    assert power.minimum_detectable_effect_t(9, 10, design_effect=2.0) > d
    assert power.minimum_detectable_effect_t(5000, 5000) < 0.1     # huge samples: tiny effects are visible


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        power.required_n_per_group(0.5, 0.5)
    with pytest.raises(ValueError):
        power.power_two_proportions(0.4, 0.3, 100, 100, alpha=1.5)
    with pytest.raises(ValueError):
        power.minimum_detectable_difference(0.3, 100, 100, power=1.2)


def test_tost_examples():
    clear = power.tost_equivalence(0.0, 0.02, 0.10)
    assert clear.p_value < 0.001 and clear.detail["equivalent"]
    unclear = power.tost_equivalence(0.08, 0.05, 0.10)
    assert unclear.p_value > 0.05 and not unclear.detail["equivalent"]
    assert clear.detail["confidence_level"] == pytest.approx(0.90)
    cluster = power.tost_equivalence(-0.043, 0.053, 0.10, df=242)
    assert cluster.df == 242 and not cluster.detail["equivalent"]


def test_tost_says_equivalent_exactly_when_the_90_percent_interval_is_inside_the_margin():
    rng = np.random.default_rng(3)
    for _ in range(300):
        estimate, se, margin = rng.normal(0, 0.05), rng.uniform(0.005, 0.08), rng.uniform(0.02, 0.15)
        for df in (None, 30):
            result = power.tost_equivalence(estimate, se, margin, df=df)
            inside = result.ci_low > -margin and result.ci_high < margin
            assert result.detail["equivalent"] == inside
            assert result.p_value == pytest.approx(max(result.detail["p_lower"], result.detail["p_upper"]))


def test_tost_refuses_a_zero_margin_or_error():
    with pytest.raises(ValueError):
        power.tost_equivalence(0.0, 0.0, 0.1)
    with pytest.raises(ValueError):
        power.tost_equivalence(0.0, 0.1, 0.0)


@pytest.mark.parametrize("p1, p2", [(0.6, 0.5), (0.36, 0.29), (0.2, 0.1), (0.55, 0.45)])
def test_power_and_required_sample_size_agree_to_the_last_row(p1, p2):
    n, _ = power.required_n_per_group(p1, p2)
    assert power.power_two_proportions(p1, p2, n, n) >= 0.8 > power.power_two_proportions(p1, p2, n - 1, n - 1)
