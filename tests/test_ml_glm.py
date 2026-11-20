"""
tests/test_ml_glm.py  -  checks of the logistic regression with cluster-robust standard errors (src/ml/glm.py).

WHAT THIS FILE IS
    The coefficients, the ordinary standard errors, the cluster-robust standard errors and the log-likelihood are compared with
    statsmodels 0.15 on a fixed data set of 72 rows (the numbers were calculated once with statsmodels and are written into the
    test, so statsmodels itself is not needed to run it). Further tests: the same coefficients as scikit-learn without a penalty,
    perfect separation is reported instead of hidden, and bad input is refused.

WHY WE NEED IT
    The odds ratios and their intervals of the notebook come from this function. A standard error that is wrong by 10% changes which
    inputs look "significant", so the numbers are pinned to a trusted implementation.

WHAT HAPPENS WHEN IT RUNS
    The data set is built from integer formulas (no random numbers), so it is identical on every machine.

EXPECTED OUTCOME
    Agreement to at least eight digits.

MOTIVE
    A confidence interval is a promise; this is the check that it is kept.
"""
import numpy as np
import pytest
from scipy import stats

from src.ml.glm import fit_logit

Y = np.array(list("101001010011000010000110001110011100011100111101111011111011100111001110"), dtype=int)
I = np.arange(72)
X1 = ((I * 37) % 17 - 8) / 4.0
X2 = ((I * 11) % 7 - 3) / 3.0
X = np.column_stack([np.ones(72), X1, X2])
CLUSTERS = I // 6                                                                  # twelve clusters of six rows

# calculated with statsmodels 0.15.0: sm.Logit(y, X).fit() and sm.GLM(y, X, Binomial()).fit(cov_type="cluster", cov_kwds={"groups": ...})
COEF = [0.29402548698842185, 0.6858855242210927, -0.4451491858276867]
SE = [0.2601991566405994, 0.2276230860073959, 0.3900663876531151]
SE_CLUSTER = [0.2726905590566163, 0.3734625743162842, 0.33442086090026285]
LOG_LIKELIHOOD = -43.63093153448331


def test_coefficients_standard_errors_and_log_likelihood_equal_statsmodels():
    fit = fit_logit(Y, X)
    assert fit.converged and fit.iterations < 12
    assert fit.coef == pytest.approx(COEF, abs=1e-9)
    assert fit.se == pytest.approx(SE, abs=1e-9)
    assert fit.log_likelihood == pytest.approx(LOG_LIKELIHOOD, abs=1e-9)
    assert fit.groups is None and fit.df is None


def test_cluster_robust_standard_errors_equal_statsmodels():
    fit = fit_logit(Y, X, clusters=CLUSTERS)
    assert fit.coef == pytest.approx(COEF, abs=1e-9)
    assert fit.se == pytest.approx(SE_CLUSTER, abs=1e-9)
    assert fit.groups == 12 and fit.df == 11


def test_p_values_use_the_t_distribution_with_clusters_and_the_normal_without():
    plain = fit_logit(Y, X)
    assert plain.p_values() == pytest.approx(2 * stats.norm.sf(np.abs(np.array(COEF) / np.array(SE))))
    robust = fit_logit(Y, X, clusters=CLUSTERS)
    assert robust.p_values() == pytest.approx(2 * stats.t.sf(np.abs(np.array(COEF) / np.array(SE_CLUSTER)), 11))


def test_odds_ratios_and_intervals():
    fit = fit_logit(Y, X, clusters=CLUSTERS)
    ratio, low, high = fit.odds_ratios(0.95)
    assert ratio == pytest.approx(np.exp(COEF))
    assert (low < ratio).all() and (ratio < high).all()
    crit = stats.t.ppf(0.975, 11)
    assert low == pytest.approx(np.exp(np.array(COEF) - crit * np.array(SE_CLUSTER)))
    narrower = fit_logit(Y, X, clusters=CLUSTERS).odds_ratios(0.5)
    assert (narrower[2] - narrower[1] < high - low).all()


def test_same_coefficients_as_scikit_learn_without_a_penalty():
    linear = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(300, 3))
    y = (rng.random(300) < 1 / (1 + np.exp(-(0.5 + Z @ [1.0, -0.5, 0.0])))).astype(int)
    mine = fit_logit(y, np.column_stack([np.ones(300), Z]))
    reference = linear.LogisticRegression(C=1e10, max_iter=10000, tol=1e-12).fit(Z, y)          # a penalty of 1e-10: no penalty in practice
    assert mine.coef[0] == pytest.approx(reference.intercept_[0], abs=1e-5)
    assert mine.coef[1:] == pytest.approx(reference.coef_[0], abs=1e-5)


def test_predict_proba_is_the_logistic_function_of_the_linear_score():
    fit = fit_logit(Y, X)
    assert fit.predict_proba(X) == pytest.approx(1 / (1 + np.exp(-(X @ fit.coef))))


def test_perfect_separation_is_reported_not_hidden():
    x = np.arange(20.0)
    y = (x >= 10).astype(int)
    fit = fit_logit(y, np.column_stack([np.ones(20), x]))
    assert not fit.converged


def test_bad_input_is_refused():
    with pytest.raises(ValueError):
        fit_logit([0, 1, 2], np.ones((3, 1)))
    with pytest.raises(ValueError):
        fit_logit([1, 1, 1, 1], np.ones((4, 1)))
    with pytest.raises(ValueError):
        fit_logit(Y, X[:-1])
    with pytest.raises(ValueError):
        fit_logit(Y, np.where(X > 100, X, np.nan))
    with pytest.raises(ValueError):
        fit_logit(Y, X, clusters=np.zeros(72))
