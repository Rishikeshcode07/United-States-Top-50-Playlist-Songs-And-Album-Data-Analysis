"""
tests/test_survival_hazard.py  -  checks of the day-by-day hazard model and its tests (src/survival/hazard.py).

WHAT THIS FILE IS
    The coefficients, the usual standard errors, the artist-aware (cluster-robust) standard errors and the log-likelihood are compared
    with statsmodels 0.15 on a fixed set of 600 chart days (the numbers were calculated once with statsmodels and are written into the test,
    so statsmodels itself is not needed to run it). Then: the Wald test of one coefficient equals the t-test, the test of several is
    calibrated on data without an effect and finds an effect that is there, the age-window helpers, the separate-effects design of the
    "same effect at every age" check, and the refusals.

WHY WE NEED IT
    The hazard ratios, their intervals and the p-values of Phase 6B come from this model.

EXPECTED OUTCOME
    Agreement with statsmodels to at least six digits.

MOTIVE
    An interval is a promise, and this is the check that it is kept.
"""
import numpy as np
import pandas as pd
import pytest

from src.survival.hazard import (age_windows, equal_effects_contrasts, fit_cloglog, selection, wald_test, window_columns,
                                 window_interactions)


def toy_pd(n_rows=600):
    i = np.arange(n_rows)
    x1 = ((i * 37) % 100) / 50.0 - 1.0
    x2 = (i % 3 == 0).astype(float)
    age_window = (i // 7) % 3
    cluster = (i * 13) % 40
    u = ((i * 7919) % 1000) / 1000.0                          # a fixed pseudo-random number between 0 and 1, built from integers
    p = 1 - np.exp(-np.exp(-1.6 + 0.4 * age_window + 0.7 * x1 + 0.5 * x2))
    y = (u < p).astype(float)
    W = np.column_stack([(age_window == k).astype(float) for k in range(3)])
    return y, np.column_stack([W, x1, x2]), cluster


# statsmodels 0.15: GLM(Binomial, CLogLog) on the data above
COEF = [-1.66567338, -1.19950211, -0.91027162, 0.71418167, 0.5374551]
SE_MODEL = [0.17182443, 0.14165612, 0.13122277, 0.13406894, 0.153258]
SE_ROBUST = [0.12231877, 0.10874939, 0.08548435, 0.16222038, 0.11514405]
LOG_LIK = -343.89288343787763


def test_agrees_with_statsmodels():
    y, X, cl = toy_pd()
    fit = fit_cloglog(y, X, clusters=cl)
    assert fit.converged
    assert fit.coef == pytest.approx(COEF, abs=1e-6)
    assert np.sqrt(np.diag(fit.model_cov)) == pytest.approx(SE_MODEL, abs=1e-6)
    assert fit.se == pytest.approx(SE_ROBUST, abs=1e-6)
    assert fit.log_likelihood == pytest.approx(LOG_LIK, abs=1e-6)
    assert fit.groups == 40 and fit.df == 39 and fit.n == 600 and fit.k == 5


def test_without_clusters_the_covariance_is_the_textbook_one():
    y, X, _ = toy_pd()
    fit = fit_cloglog(y, X)
    assert fit.groups is None and fit.df is None
    assert fit.se == pytest.approx(SE_MODEL, abs=1e-6)


def test_hazard_ratios_and_intervals_use_the_t_reference_with_clusters():
    from scipy import stats
    y, X, cl = toy_pd()
    fit = fit_cloglog(y, X, clusters=cl)
    ratio, low, high = fit.hazard_ratios()
    assert ratio == pytest.approx(np.exp(fit.coef))
    crit = stats.t.ppf(0.975, 39)
    assert low == pytest.approx(np.exp(fit.coef - crit * fit.se)) and high == pytest.approx(np.exp(fit.coef + crit * fit.se))
    assert fit.p_values()[3] == pytest.approx(2 * stats.t.sf(abs(fit.coef[3] / fit.se[3]), 39))


def test_wald_test_of_one_coefficient_is_the_t_test():
    y, X, cl = toy_pd()
    fit = fit_cloglog(y, X, clusters=cl)
    result = wald_test(fit, selection(["a", "b", "c", "x1", "x2"], ["x1"]))
    assert result["q"] == 1 and result["df"] == 39
    assert result["p"] == pytest.approx(fit.p_values()[3], rel=1e-9)
    assert result["stat"] == pytest.approx((fit.coef[3] / fit.se[3]) ** 2, rel=1e-9)


def test_wald_test_of_several_coefficients():
    y, X, cl = toy_pd()
    fit = fit_cloglog(y, X, clusters=cl)
    both = wald_test(fit, selection(["a", "b", "c", "x1", "x2"], ["x1", "x2"]))
    assert both["q"] == 2 and both["p"] < 0.001
    plain = wald_test(fit_cloglog(y, X), selection(["a", "b", "c", "x1", "x2"], ["x1", "x2"]))
    assert plain["df"] is None and plain["p"] < 0.001                                     # chi-square without clusters


def simulated(effect_by_window, n=6000, seed=3):
    """Chart days with one input whose effect is given for each of three age windows."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    window = rng.integers(0, 3, size=n)
    eta = -1.5 + 0.3 * window + np.array(effect_by_window)[window] * x
    y = (rng.random(n) < 1 - np.exp(-np.exp(eta))).astype(float)
    return y, x, window, rng.integers(0, 150, size=n)


def check_equal_effects(effect_by_window):
    y, x, window, cluster = simulated(effect_by_window)
    design = pd.DataFrame({**{f"age{k}": (window == k).astype(float) for k in range(3)}, "x": x})
    split = window_interactions(design, ["x"], window, 3)
    fit = fit_cloglog(y, split.to_numpy(), clusters=cluster)
    names = list(split.columns)
    return fit, wald_test(fit, equal_effects_contrasts(names, ["x"], 3)), names


def test_equal_effects_check_does_not_reject_when_the_effect_is_the_same_at_every_age():
    fit, result, names = check_equal_effects([0.5, 0.5, 0.5])
    assert result["q"] == 2 and result["p"] > 0.05
    assert fit.coef[names.index("x @ 1")] == pytest.approx(0.5, abs=0.15)


def test_equal_effects_check_finds_an_effect_that_changes_with_age():
    _, result, _ = check_equal_effects([0.9, 0.5, 0.0])
    assert result["p"] < 0.001


def test_window_interactions_replace_the_column_by_one_column_per_window():
    design = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 0.0, 1.0]})
    split = window_interactions(design, ["a"], [0, 1, 2, 1], 3)
    assert list(split.columns) == ["b", "a @ 1", "a @ 2", "a @ 3"]
    assert split["a @ 1"].tolist() == [1, 0, 0, 0] and split["a @ 2"].tolist() == [0, 2, 0, 4] and split["a @ 3"].tolist() == [0, 0, 3, 0]
    assert split[["a @ 1", "a @ 2", "a @ 3"]].sum(axis=1).tolist() == design["a"].tolist()
    contrasts = equal_effects_contrasts(list(split.columns), ["a"], 3)
    assert contrasts.shape == (2, 4) and contrasts[0].tolist() == [0, 1, -1, 0] and contrasts[1].tolist() == [0, 0, 1, -1]


def test_age_windows_and_columns():
    edges = [0, 1, 2, 3, 7, 14, 30, 60, 120, 10000]
    ages = np.array([1, 2, 3, 4, 7, 8, 14, 15, 30, 31, 60, 61, 120, 121, 500])
    assert list(age_windows(ages, edges)) == [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    columns = window_columns(ages, edges)
    assert list(columns.columns[:5]) == ["age 1", "age 2", "age 3", "age 4-7", "age 8-14"] and columns.columns[-1] == "age 121+"
    assert columns.to_numpy().sum(axis=1).tolist() == [1.0] * len(ages)
    with pytest.raises(ValueError):
        age_windows([0, 1], edges)


def test_perfect_separation_is_reported():
    x = np.r_[np.zeros(30), np.ones(30)]
    y = x.copy()
    X = np.column_stack([np.ones(60), x])
    assert not fit_cloglog(y, X).converged


def test_recovers_the_coefficient_of_simulated_data():
    y, x, window, cluster = simulated([0.6, 0.6, 0.6], n=30000, seed=11)
    X = np.column_stack([(window == k).astype(float) for k in range(3)] + [x])
    fit = fit_cloglog(y, X, clusters=cluster)
    assert fit.converged and fit.coef[3] == pytest.approx(0.6, abs=0.06)
    assert fit.coef[:3] == pytest.approx([-1.5, -1.2, -0.9], abs=0.1)


def test_bad_input_is_refused():
    y, X, cl = toy_pd()
    with pytest.raises(ValueError):
        fit_cloglog(y[:-1], X)
    with pytest.raises(ValueError):
        fit_cloglog(np.where(y > 0, 2, 0), X)
    with pytest.raises(ValueError):
        fit_cloglog(np.zeros(600), X)
    with pytest.raises(ValueError):
        fit_cloglog(y, np.where(X > 5, np.nan, X) + np.nan)
    with pytest.raises(ValueError):
        fit_cloglog(y, X, clusters=np.zeros(600))
    with pytest.raises(ValueError):
        fit_cloglog(y[:3], X[:3])
