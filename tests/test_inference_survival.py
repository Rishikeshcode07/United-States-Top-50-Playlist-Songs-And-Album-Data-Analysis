"""
tests/test_inference_survival.py  -  checks of the Kaplan-Meier curve and the log-rank test (src/inference/survival.py).

WHAT THIS FILE IS
    Automatic tests: a textbook example worked by hand, the fact that without censoring the curve is simply the share of songs that
    stayed longer, agreement of the log-rank test with scipy.stats.logrank on data with ties and censoring, and simulations of
    its size (under "no difference" it rejects about 5% of the time) and its power.

WHY WE NEED IT
    The "singles stay longer" question is re-tested with these tools because they use the songs that are still on the chart at the
    end of the file instead of treating them as finished.

WHAT HAPPENS WHEN IT RUNS
    pytest runs the tests; the simulations use fixed seeds.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    A song that is still running is information, not an error.
"""
import numpy as np
import pytest
from scipy import stats

from src.inference.survival import kaplan_meier, logrank_test, median_survival


def test_kaplan_meier_textbook_example():
    curve = kaplan_meier([3, 5, 5, 8, 10, 12], [1, 1, 0, 1, 1, 0])
    assert curve["time"].tolist() == [0, 3, 5, 8, 10]
    assert curve["at_risk"].tolist() == [6, 6, 5, 3, 2]
    assert curve["events"].tolist() == [0, 1, 1, 1, 1]
    assert curve["survival"].tolist() == pytest.approx([1.0, 5 / 6, 5 / 6 * 4 / 5, 5 / 6 * 4 / 5 * 2 / 3, 5 / 6 * 4 / 5 * 2 / 3 / 2])
    assert median_survival(curve) == 8.0
    assert (curve["ci_low"] <= curve["survival"]).all() and (curve["survival"] <= curve["ci_high"]).all()
    assert curve["ci_low"].min() >= 0 and curve["ci_high"].max() <= 1


def test_without_censoring_the_curve_is_the_share_that_stayed_longer():
    rng = np.random.default_rng(1)
    durations = rng.integers(1, 40, 300)
    curve = kaplan_meier(durations, np.ones(300, bool))
    for row in curve.itertuples():
        assert row.survival == pytest.approx(np.mean(durations > row.time))


def test_the_median_is_missing_when_more_than_half_are_still_running():
    curve = kaplan_meier([5, 10, 20, 20, 20, 20], [1, 1, 0, 0, 0, 0])
    assert np.isnan(median_survival(curve))


def test_bad_input_is_refused():
    with pytest.raises(ValueError):
        kaplan_meier([1, 2, 3], [True, False])
    with pytest.raises(ValueError):
        kaplan_meier([-1, 2], [True, True])
    with pytest.raises(ValueError):
        logrank_test([], [], [1], [True])


def _random_groups(rng, n_a, n_b, rate_a, rate_b, censor_share=0.25):
    ta = np.ceil(rng.exponential(1 / rate_a, n_a)).astype(float)
    tb = np.ceil(rng.exponential(1 / rate_b, n_b)).astype(float)
    ea = rng.random(n_a) > censor_share
    eb = rng.random(n_b) > censor_share
    return ta, ea, tb, eb


def test_logrank_matches_scipy_with_ties_and_censoring():
    rng = np.random.default_rng(2)
    ta, ea, tb, eb = _random_groups(rng, 120, 150, 0.08, 0.05)
    ours = logrank_test(ta, ea, tb, eb)
    reference = stats.logrank(stats.CensoredData(uncensored=ta[ea], right=ta[~ea]), stats.CensoredData(uncensored=tb[eb], right=tb[~eb]))
    assert ours.statistic == pytest.approx(reference.statistic ** 2, rel=1e-9)
    assert ours.p_value == pytest.approx(reference.pvalue, rel=1e-6)
    assert ours.estimate > 1.0                                              # group A leaves faster
    assert ours.detail["observed_a"] + ours.detail["observed_b"] == pytest.approx(ours.detail["expected_a"] + ours.detail["expected_b"])


def test_simulation_logrank_size_and_power():
    rng = np.random.default_rng(3)
    size = np.mean([logrank_test(*_random_groups(rng, 40, 40, 0.1, 0.1)).p_value < 0.05 for _ in range(600)])
    strong = np.mean([logrank_test(*_random_groups(rng, 100, 100, 0.2, 0.1)).p_value < 0.05 for _ in range(200)])
    assert 0.03 < size < 0.075
    assert strong > 0.8
