"""
tests/test_inference_dependence.py  -  checks of the tools for related rows (src/inference/dependence.py).

WHAT THIS FILE IS
    Automatic tests of the cluster-robust regression, the artist bootstrap, the intraclass correlation, the permutation tests and
    the time-series tools. Each tool is checked three ways: against an independent calculation (scipy, or a formula done by hand
    in the test), on small examples whose answer is known, and by SIMULATION: data are generated in which the artists (or the
    days) really are related, and the tests must behave the way the theory promises.

WHY WE NEED IT
    The headline results of Phase 5 depend on these tools. The most important claim of the module, that the classical test is too
    optimistic when rows are related and the cluster-robust test is not, is proved here by simulation, not just stated.

WHAT HAPPENS WHEN IT RUNS
    pytest runs every test with fixed random seeds; the simulations take a few seconds in total.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Trust the standard errors before trusting the p-values.
"""
import numpy as np
import pytest
from scipy import signal, stats

from src.inference import dependence as d


# ================================================================================================ regression
def test_ols_matches_numpy_and_scipy_linregress():
    rng = np.random.default_rng(1)
    x = rng.normal(size=80)
    y = 2.0 + 0.7 * x + rng.normal(size=80)
    fit = d.fit_ols(y, np.column_stack([np.ones(80), x]))
    reference = stats.linregress(x, y)
    assert fit.coef == pytest.approx([reference.intercept, reference.slope])
    assert fit.se[1] == pytest.approx(reference.stderr)
    result = d.coefficient_test(fit, 1)
    assert result.p_value == pytest.approx(reference.pvalue)
    margin = stats.t.ppf(0.975, 78) * reference.stderr
    assert (result.ci_low, result.ci_high) == pytest.approx((reference.slope - margin, reference.slope + margin))


def test_with_one_row_per_cluster_the_cluster_robust_covariance_is_the_white_hc1_one():
    rng = np.random.default_rng(2)
    x = rng.normal(size=60)
    y = 1.0 + x + rng.normal(size=60) * (1 + np.abs(x))
    X = np.column_stack([np.ones(60), x])
    fit = d.fit_ols(y, X, clusters=np.arange(60))
    xtx_inv = np.linalg.inv(X.T @ X)
    meat = (X * fit.residuals[:, None]).T @ (X * fit.residuals[:, None])
    assert fit.cov == pytest.approx(xtx_inv @ meat @ xtx_inv * 60 / 58)
    assert fit.groups == 60 and fit.df == 59


def test_bad_regression_input_is_refused():
    X = np.column_stack([np.ones(3), [1.0, 2.0, 3.0], [1.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        d.fit_ols([1.0, 2.0, 3.0], X)                                   # as many coefficients as rows
    with pytest.raises(ValueError):
        d.fit_ols([1.0, 2.0, 3.0, 4.0], X[:, :2])                       # lengths differ
    with pytest.raises(ValueError):
        d.fit_ols(np.arange(10.0), np.column_stack([np.ones(10), np.arange(10.0)]), clusters=np.zeros(10))   # one cluster only
    with pytest.raises(ValueError):
        d.fit_ols([1.0, np.nan, 3.0, 4.0], np.column_stack([np.ones(4), np.arange(4.0)]))


def test_without_clusters_the_ready_made_tests_are_the_classical_ones():
    rng = np.random.default_rng(3)
    groups = rng.integers(0, 3, 150)
    y = rng.normal(size=150) + groups * 0.4
    result = d.cluster_robust_k_group_test(y, groups)
    reference = stats.f_oneway(*[y[groups == g] for g in range(3)])
    assert result.statistic == pytest.approx(reference.statistic) and result.p_value == pytest.approx(reference.pvalue)
    assert result.df == (2, 147)
    flag = groups == 2
    two = d.cluster_robust_two_group_test(y, flag)
    student = stats.ttest_ind(y[flag], y[~flag], equal_var=True)
    assert two.statistic == pytest.approx(student.statistic) and two.p_value == pytest.approx(student.pvalue)
    x = rng.normal(size=150)
    slope = d.cluster_robust_slope_test(y, x)
    assert slope.p_value == pytest.approx(stats.linregress(x, y).pvalue)


def test_the_interaction_t_squared_is_the_balanced_anova_f():
    rng = np.random.default_rng(4)
    a = np.repeat([0, 0, 1, 1], 30)
    b = np.repeat([0, 1, 0, 1], 30)
    y = rng.normal(size=120) + 0.5 * a + 0.3 * b + 0.6 * a * b
    result = d.cluster_robust_interaction_test(y, a, b)
    cells = [y[(a == i) & (b == j)] for i in (0, 1) for j in (0, 1)]
    did = cells[3].mean() - cells[2].mean() - cells[1].mean() + cells[0].mean()
    pooled = sum(((c - c.mean()) ** 2).sum() for c in cells) / (120 - 4)
    assert result.statistic ** 2 == pytest.approx(did ** 2 / (pooled * 4 / 30))
    assert result.estimate == pytest.approx(did)


def test_wald_test_of_several_restrictions_and_of_a_single_one():
    rng = np.random.default_rng(5)
    X = np.column_stack([np.ones(100), rng.normal(size=100), rng.normal(size=100)])
    y = X @ np.array([1.0, 0.5, 0.0]) + rng.normal(size=100)
    fit = d.fit_ols(y, X)
    single = d.wald_test(fit, [0, 1, 0])
    assert single.statistic == pytest.approx(d.coefficient_test(fit, 1).statistic ** 2)
    both = d.wald_test(fit, [[0, 1, 0], [0, 0, 1]])
    assert both.df == (2, 97)


def test_simulation_the_naive_test_is_too_optimistic_for_related_rows_and_the_cluster_robust_one_is_not():
    rng = np.random.default_rng(6)
    groups, size, runs = 40, 10, 800
    cluster = np.repeat(np.arange(groups), size)
    naive = robust = 0
    for _ in range(runs):
        y = rng.normal(size=groups)[cluster] + rng.normal(size=groups * size)             # intraclass correlation 0.5
        flag = np.isin(cluster, rng.permutation(groups)[: groups // 2])                  # the covariate is fixed inside a cluster
        naive += d.cluster_robust_two_group_test(y, flag).p_value < 0.05
        robust += d.cluster_robust_two_group_test(y, flag, clusters=cluster).p_value < 0.05
    assert naive / runs > 0.25                                                           # about 40% false alarms instead of 5%
    assert 0.03 < robust / runs < 0.09


# ================================================================================================ clustering
def test_intraclass_correlation_recovers_the_true_value():
    rng = np.random.default_rng(7)
    clusters = np.repeat(np.arange(300), 6)
    y = rng.normal(0, np.sqrt(0.4), 300)[clusters] + rng.normal(0, np.sqrt(0.6), 1800)
    result = d.intraclass_correlation(y, clusters)
    assert result["icc"] == pytest.approx(0.4, abs=0.06)
    assert result["design_effect"] == pytest.approx(1 + (6 - 1) * result["icc"])
    assert result["effective_n"] == pytest.approx(1800 / result["design_effect"])
    independent = d.intraclass_correlation(rng.normal(size=1800), clusters)
    assert abs(independent["icc"]) < 0.05


def test_intraclass_correlation_is_one_when_rows_of_a_cluster_are_identical():
    clusters = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    y = np.array([1.0, 1, 1, 5, 5, 9, 9, 9, 9])
    assert d.intraclass_correlation(y, clusters)["icc"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        d.intraclass_correlation([1.0, 2.0], [0, 1])


# ================================================================================================ bootstrap and permutation
def test_cluster_bootstrap_weights_are_whole_cluster_counts():
    clusters = np.array(["a", "a", "b", "c", "c", "c", "d"])
    batches = list(d.cluster_bootstrap_weights(clusters, n_boot=1200, seed=1, chunk=500))
    weights = np.vstack(batches)
    assert weights.shape == (1200, 7) and len(batches) == 3
    assert np.all(weights >= 0) and np.all(weights == np.round(weights))
    assert np.all(weights[:, 0] == weights[:, 1]) and np.all(weights[:, 3] == weights[:, 4]) and np.all(weights[:, 4] == weights[:, 5])   # rows of one cluster share a weight
    per_cluster = weights[:, [0, 2, 3, 6]]
    assert np.all(per_cluster.sum(axis=1) == 4)                            # four clusters drawn every time
    assert per_cluster.mean(axis=0) == pytest.approx([1, 1, 1, 1], abs=0.1)
    again = np.vstack(list(d.cluster_bootstrap_weights(clusters, n_boot=1200, seed=1, chunk=500)))
    assert np.array_equal(weights, again)
    other = np.vstack(list(d.cluster_bootstrap_weights(clusters, n_boot=1200, seed=2, chunk=500)))
    assert not np.array_equal(weights, other)


def test_the_cluster_bootstrap_standard_error_agrees_with_the_cluster_robust_one():
    rng = np.random.default_rng(8)
    sizes = rng.integers(1, 12, 60)
    clusters = np.repeat(np.arange(60), sizes)
    y = rng.normal(size=60)[clusters] + rng.normal(size=sizes.sum())
    means = []
    for weights in d.cluster_bootstrap_weights(clusters, n_boot=4000, seed=3):
        means.append(weights @ y / weights.sum(axis=1))
    bootstrap_se = np.concatenate(means).std(ddof=1)
    analytic_se = d.fit_ols(y, np.ones((len(y), 1)), clusters).se[0]
    assert bootstrap_se == pytest.approx(analytic_se, rel=0.15)


def test_bootstrap_interval_and_p_value():
    replicates = np.linspace(-1, 3, 4001)
    low, high = d.bootstrap_interval(replicates, 0.95)
    assert (low, high) == pytest.approx((-0.9, 2.9), abs=0.01)
    assert d.bootstrap_p_value(replicates, null=-5) == pytest.approx(2 / 4002)          # one replicate-equivalent on one side, none on the other
    assert d.bootstrap_p_value(replicates, null=1.0) == pytest.approx(1.0, abs=0.01)
    assert 0 < d.bootstrap_p_value(replicates, null=0.0) < 1
    with pytest.raises(ValueError):
        d.bootstrap_interval([])


def test_permutation_p_value_small_examples():
    null = np.arange(9.0)
    assert d.permutation_p_value(8.0, null, "greater") == pytest.approx(2 / 10)
    assert d.permutation_p_value(8.0, null, "two-sided") == pytest.approx(2 / 10)
    assert d.permutation_p_value(8.0, null, "less") == pytest.approx(1.0)
    assert d.permutation_p_value(100.0, null) == pytest.approx(1 / 10)
    with pytest.raises(ValueError):
        d.permutation_p_value(1.0, null, "sideways")


def test_shuffled_group_differences_centre_and_spread():
    rng = np.random.default_rng(9)
    y = rng.normal(5, 2, 200)
    flag = np.arange(200) < 60
    null = d.shuffled_group_differences(y, flag, n_perm=4000, seed=5)
    assert abs(null.mean()) < 0.05
    assert null.std() == pytest.approx(y.std() * np.sqrt(1 / 60 + 1 / 140), rel=0.08)
    assert np.array_equal(null, d.shuffled_group_differences(y, flag, n_perm=4000, seed=5))
    with pytest.raises(ValueError):
        d.shuffled_group_differences(y, np.zeros(200, bool))


def test_permutation_test_keeps_its_size_under_the_null():
    rng = np.random.default_rng(10)
    rejected = 0
    runs = 400
    for run in range(runs):
        y = rng.exponential(size=40)
        flag = np.arange(40) < 15
        observed = y[flag].mean() - y[~flag].mean()
        rejected += d.permutation_p_value(observed, d.shuffled_group_differences(y, flag, n_perm=199, seed=run)) < 0.05
    assert 0.02 < rejected / runs < 0.085


def test_exact_split_permutation_small_example_and_guard():
    result = d.exact_split_permutation_test([1, 2, 3, 4, 5, 6], 3)
    assert result.estimate == pytest.approx(-3.0)
    assert result.detail["splits"] == 20
    assert result.p_value == pytest.approx(2 / 20)                         # only {1,2,3} and {4,5,6} are as extreme
    assert d.exact_split_permutation_test(np.arange(19.0), 9).detail["splits"] == 92378
    with pytest.raises(ValueError):
        d.exact_split_permutation_test(np.arange(60.0), 30)
    with pytest.raises(ValueError):
        d.exact_split_permutation_test([1.0, 2.0], 2)


# ================================================================================================ time series
def test_autocorrelation_and_effective_sample_size():
    rng = np.random.default_rng(11)
    ar = signal.lfilter([1], [1, -0.7], rng.normal(size=20000))
    assert d.autocorrelation(ar) == pytest.approx(0.7, abs=0.04)
    assert abs(d.autocorrelation(rng.normal(size=20000))) < 0.04
    assert d.effective_sample_size_ar1(100, 0.0) == pytest.approx(100.0)
    assert d.effective_sample_size_ar1(100, 0.6) == pytest.approx(25.0)
    assert d.effective_sample_size_ar1(100, -0.5) == pytest.approx(100.0)       # never above n
    assert 1.0 <= d.effective_sample_size_ar1(100, 0.999) < 2.0
    with pytest.raises(ValueError):
        d.autocorrelation([1.0, 2.0, 3.0], lag=3)


def test_default_lags_rule_of_thumb():
    assert d.default_lags(100) == 4 and d.default_lags(555) == 5


def test_hac_trend_recovers_a_slope_and_has_a_wider_error_for_correlated_noise():
    rng = np.random.default_rng(12)
    n = 400
    t = np.arange(n)
    white = d.hac_trend_test(0.05 * t + rng.normal(size=n))
    assert white.estimate == pytest.approx(0.05, abs=0.005)
    assert white.detail["se"] == pytest.approx(white.detail["naive_se"], rel=0.35)
    noise = signal.lfilter([1], [1, -0.7], rng.normal(size=n))
    correlated = d.hac_trend_test(0.05 * t + noise)
    assert correlated.detail["se"] > 1.3 * correlated.detail["naive_se"]
    assert correlated.ci_low < correlated.estimate < correlated.ci_high


def test_simulation_hac_is_closer_to_its_size_than_the_naive_trend_test():
    rng = np.random.default_rng(13)
    n, runs = 300, 500
    t = np.arange(n, dtype=float)
    naive = hac = 0
    for _ in range(runs):
        y = signal.lfilter([1], [1, -0.6], rng.normal(size=n))                          # no trend, but correlated days
        hac += d.hac_trend_test(y).p_value < 0.05
        slope = stats.linregress(t, y)
        naive += slope.pvalue < 0.05
    assert naive / runs > 0.2
    assert hac / runs < 0.14 and hac < naive / 1.5


def test_moving_block_indices_shape_range_and_blocks():
    indices = d.moving_block_indices(100, 14, n_boot=50, seed=1)
    assert indices.shape == (50, 100)
    assert indices.min() >= 0 and indices.max() <= 99
    first = indices[0]
    assert np.all(np.diff(first[:14]) == 1) and np.all(np.diff(first[14:28]) == 1)      # a block is a run of consecutive days
    assert np.array_equal(indices, d.moving_block_indices(100, 14, n_boot=50, seed=1))
    with pytest.raises(ValueError):
        d.moving_block_indices(10, 20)


def test_block_bootstrap_trend_is_centred_on_the_observed_slope_and_wider_than_the_naive_error():
    rng = np.random.default_rng(14)
    n = 400
    y = 0.03 * np.arange(n) + signal.lfilter([1], [1, -0.7], rng.normal(size=n))
    slopes = d.block_bootstrap_trend(y, block=20, n_boot=4000, seed=2)
    observed = d.hac_trend_test(y)
    assert slopes.shape == (4000,)
    assert slopes.mean() == pytest.approx(observed.estimate, abs=3 * slopes.std() / np.sqrt(4000) + 1e-4)
    assert slopes.std() > observed.detail["naive_se"]
    assert np.array_equal(slopes, d.block_bootstrap_trend(y, block=20, n_boot=4000, seed=2))


def test_mann_kendall_perfect_trend_and_no_trend():
    up = d.mann_kendall_trend(np.arange(50.0) * 2 + 1)
    assert up.statistic == pytest.approx(1.0) and up.p_value < 1e-10
    assert up.estimate == pytest.approx(2.0) and up.ci_low <= 2.0 <= up.ci_high
    down = d.mann_kendall_trend(-np.arange(50.0))
    assert down.statistic == pytest.approx(-1.0)
    flat = d.mann_kendall_trend(np.random.default_rng(15).normal(size=200))
    assert flat.p_value > 0.01


def test_newey_west_matches_a_hand_calculation():
    rng = np.random.default_rng(16)
    n, lags = 40, 3
    X = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
    u = rng.normal(size=n)
    got = d.newey_west_cov(X, u, lags)
    middle = np.zeros((2, 2))
    for t in range(n):
        middle += u[t] ** 2 * np.outer(X[t], X[t])
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)                                     # the Bartlett weights: 3/4, 2/4, 1/4
        for t in range(lag, n):
            cross = u[t] * u[t - lag] * np.outer(X[t], X[t - lag])
            middle += weight * (cross + cross.T)
    bread = np.linalg.inv(X.T @ X)
    assert got == pytest.approx(bread @ middle @ bread * n / (n - 2))


def test_cluster_bootstrap_runs_a_statistic_on_every_replicate():
    rng = np.random.default_rng(17)
    clusters = np.repeat(np.arange(30), rng.integers(1, 8, 30))
    y = rng.normal(size=len(clusters))
    replicates = d.cluster_bootstrap(clusters, lambda w: w @ y / w.sum(axis=1), n_boot=1300, seed=4, chunk=400)
    assert replicates.shape == (1300,)
    again = np.concatenate([w @ y / w.sum(axis=1) for w in d.cluster_bootstrap_weights(clusters, 1300, 4, 400)])
    assert np.array_equal(replicates, again)
    with pytest.raises(ValueError):
        d.cluster_bootstrap(clusters, lambda w: w.sum(), n_boot=10)             # returns one number instead of one per replicate


# ---------------------------------------------------------------------------------------------------- the wild cluster bootstrap
def _unbalanced_clusters(rng, big=60, others=80):
    sizes = np.r_[big, rng.integers(1, 6, others)]
    return np.repeat(np.arange(len(sizes)), sizes)


def test_wild_bootstrap_matches_a_closed_form_for_two_groups():
    """For two groups the whole bootstrap collapses into sums over clusters; written out by hand it must give the same p-value."""
    rng = np.random.default_rng(11)
    clusters = _unbalanced_clusters(rng)
    y = rng.normal(size=len(clusters)) + 0.5 * rng.normal(size=clusters.max() + 1)[clusters]
    flag = rng.random(len(clusters)) < 0.4
    n_boot, seed = 400, 5
    result = d.wild_two_group_test(y, flag, clusters, n_boot=n_boot, seed=seed)

    codes, g = np.unique(clusters, return_inverse=True)[1], clusters.max() + 1
    n1, n0, n = flag.sum(), (~flag).sum(), len(y)
    r = y - y.mean()
    R1, R0 = np.bincount(codes, r * flag, g), np.bincount(codes, r * ~flag, g)
    c1, c0 = np.bincount(codes, flag * 1.0, g), np.bincount(codes, ~flag * 1.0, g)
    S1, S0 = np.bincount(codes, y * flag, g), np.bincount(codes, y * ~flag, g)
    m1, m0 = S1.sum() / n1, S0.sum() / n0
    factor = g / (g - 1) * (n - 1) / (n - 2)
    t_obs = (m1 - m0) / np.sqrt(factor * np.sum(((S1 - c1 * m1) / n1 - (S0 - c0 * m0) / n0) ** 2))
    v = np.where(np.random.default_rng(seed).random((n_boot, g)) < 0.5, -1.0, 1.0)
    d1, d0 = v @ R1 / n1, v @ R0 / n0
    infl = (-c1[None, :] * d1[:, None] + v * R1[None, :]) / n1 - (-c0[None, :] * d0[:, None] + v * R0[None, :]) / n0
    t_star = (d1 - d0) / np.sqrt(factor * (infl ** 2).sum(axis=1))
    expected_p = (1 + np.sum(np.abs(t_star) >= abs(t_obs) * (1 - 1e-12))) / (n_boot + 1)
    assert result.statistic == pytest.approx(t_obs, rel=1e-9)
    assert result.p_value == pytest.approx(expected_p, abs=1e-12)


def test_wild_bootstrap_matches_a_plain_loop_for_three_groups():
    """The vectorised bootstrap must equal a slow loop that refits the model with the library's own least squares, replicate by replicate."""
    rng = np.random.default_rng(12)
    clusters = _unbalanced_clusters(rng, big=25, others=30)
    group = rng.integers(0, 3, len(clusters))
    y = rng.normal(size=len(clusters)) + 0.4 * (group == 2) + 0.4 * rng.normal(size=clusters.max() + 1)[clusters]
    n_boot, seed = 150, 8
    result = d.wild_k_group_test(y, group, clusters, n_boot=n_boot, seed=seed)

    X = np.column_stack([np.ones(len(y)), group == 1, group == 2]).astype(float)
    R = np.array([[0, 1, 0], [0, 0, 1]], dtype=float)
    f_obs = d.wald_test(d.fit_ols(y, X, clusters), R).statistic
    g = len(np.unique(clusters))
    v = np.where(np.random.default_rng(seed).random((n_boot, g)) < 0.5, -1.0, 1.0)
    centre, residual = y.mean(), y - y.mean()                       # the restricted fit of "all group means equal" is the overall mean
    extreme = 0
    for b in range(n_boot):
        y_star = centre + v[b][clusters] * residual
        f_star = d.wald_test(d.fit_ols(y_star, X, clusters), R).statistic
        extreme += f_star >= f_obs * (1 - 1e-12)
    assert result.statistic == pytest.approx(f_obs, rel=1e-9)
    assert result.p_value == pytest.approx((1 + extreme) / (1 + n_boot), abs=1e-12)
    assert result.statistic == pytest.approx(d.cluster_robust_k_group_test(y, group, clusters).statistic, rel=1e-9)


def test_wild_bootstrap_keeps_its_five_percent_when_one_cluster_dominates():
    """Under a true null (a coin-flip label for whole clusters) the wild test rejects about 5% of the time even with one huge cluster."""
    rng = np.random.default_rng(13)
    clusters = _unbalanced_clusters(rng, big=80, others=60)
    n_clusters = clusters.max() + 1
    rejections = 0
    runs = 300
    for run in range(runs):
        y = rng.normal(size=len(clusters)) + 1.0 * rng.normal(size=n_clusters)[clusters]
        flag = (rng.random(n_clusters) < 0.5)[clusters]
        rejections += d.wild_two_group_test(y, flag, clusters, n_boot=99, seed=run).p_value < 0.05
    rate = rejections / runs
    assert 0.01 <= rate <= 0.09, rate                               # nominal 5%; the Monte Carlo noise of 300 runs is about 1.3 points


def test_wild_bootstrap_finds_a_real_effect_and_its_interval_agrees():
    rng = np.random.default_rng(14)
    clusters = _unbalanced_clusters(rng)
    flag = rng.random(len(clusters)) < 0.5
    y = rng.normal(size=len(clusters)) + 1.5 * flag
    result = d.wild_two_group_test(y, flag, clusters, n_boot=499, seed=2)
    assert result.p_value <= 0.01
    assert result.ci_low > 0 and result.ci_low < result.estimate < result.ci_high
    assert result.estimate == pytest.approx(d.cluster_robust_two_group_test(y, flag, clusters).estimate)
    nothing = d.wild_two_group_test(rng.normal(size=len(clusters)), flag, clusters, n_boot=499, seed=2)
    assert nothing.p_value > 0.05 and nothing.ci_low < 0 < nothing.ci_high          # no effect: the interval holds zero, as the p-value says


def test_wild_wrappers_estimate_what_the_robust_functions_estimate():
    rng = np.random.default_rng(15)
    clusters = _unbalanced_clusters(rng)
    a, b = rng.random(len(clusters)) < 0.5, rng.random(len(clusters)) < 0.4
    x = rng.normal(size=len(clusters))
    y = rng.normal(size=len(clusters)) + 0.5 * a * b + 0.3 * x
    slope = d.wild_slope_test(y, x, clusters, n_boot=99, seed=1)
    assert slope.estimate == pytest.approx(d.cluster_robust_slope_test(y, x, clusters).estimate)
    assert slope.statistic == pytest.approx(d.cluster_robust_slope_test(y, x, clusters).statistic, rel=1e-9)
    inter = d.wild_interaction_test(y, a, b, clusters, n_boot=99, seed=1)
    assert inter.estimate == pytest.approx(d.cluster_robust_interaction_test(y, a, b, clusters).estimate)
    assert inter.statistic == pytest.approx(d.cluster_robust_interaction_test(y, a, b, clusters).statistic, rel=1e-9)
    again = d.wild_interaction_test(y, a, b, clusters, n_boot=99, seed=1)
    assert again.p_value == inter.p_value and again.ci_low == inter.ci_low                     # the same seed gives the same answer


def test_wild_bootstrap_refuses_bad_input():
    y, clusters = np.arange(10.0), np.repeat([0, 1], 5)
    with pytest.raises(ValueError):
        d.wild_two_group_test(y, np.arange(10) % 2, np.zeros(10))                              # one cluster only
    with pytest.raises(ValueError):
        d.wild_two_group_test(y, np.arange(10) % 2, clusters[:-1])                             # one cluster label too few
    with pytest.raises(ValueError):
        d.wild_cluster_bootstrap_test(y, np.ones((10, 2)), [0, 1, 2], clusters)                # a restriction with the wrong length
    with pytest.raises(ValueError):
        d.wild_k_group_test(y, np.zeros(10), clusters)                                          # one group only


def test_wild_bootstrap_with_a_null_value_other_than_zero_matches_a_plain_loop():
    """Testing 'the two group effects are 0.1 and 0.2' (not zero): the restricted fit must keep those values."""
    rng = np.random.default_rng(16)
    clusters = _unbalanced_clusters(rng, big=25, others=30)
    group = rng.integers(0, 3, len(clusters))
    y = rng.normal(size=len(clusters)) + 0.3 * (group == 1) + 0.6 * (group == 2) + 0.4 * rng.normal(size=clusters.max() + 1)[clusters]
    X = np.column_stack([np.ones(len(y)), group == 1, group == 2]).astype(float)
    R = np.array([[0, 1, 0], [0, 0, 1]], dtype=float)
    null = np.array([0.1, 0.2])
    n_boot, seed = 120, 21
    result = d.wild_cluster_bootstrap_test(y, X, R, clusters, values=null, n_boot=n_boot, seed=seed)

    f_obs = d.wald_test(d.fit_ols(y, X, clusters), R, values=null).statistic
    g = len(np.unique(clusters))
    v = np.where(np.random.default_rng(seed).random((n_boot, g)) < 0.5, -1.0, 1.0)
    fitted = (y - 0.1 * X[:, 1] - 0.2 * X[:, 2]).mean() + 0.1 * X[:, 1] + 0.2 * X[:, 2]       # the best fit that has exactly these two effects
    residual = y - fitted
    extreme = 0
    for b in range(n_boot):
        f_star = d.wald_test(d.fit_ols(fitted + v[b][clusters] * residual, X, clusters), R, values=null).statistic
        extreme += f_star >= f_obs * (1 - 1e-12)
    assert result.statistic == pytest.approx(f_obs, rel=1e-9)
    assert result.p_value == pytest.approx((1 + extreme) / (1 + n_boot), abs=1e-12)


# ================================================================================================ comparing rows of the same cluster only
def test_within_cluster_demean_removes_every_cluster_average():
    rng = np.random.default_rng(5)
    clusters = np.repeat(np.arange(30), rng.integers(1, 6, size=30))
    values = rng.normal(size=len(clusters)) + clusters * 0.3
    centred = d.within_cluster_demean(values, clusters)
    sums = np.bincount(clusters, weights=centred, minlength=30)
    assert np.allclose(sums, 0.0)
    # the same as the pandas way of doing it
    import pandas as pd
    reference = values - pd.Series(values).groupby(clusters).transform("mean").to_numpy()
    assert np.allclose(centred, reference)
    # a cluster with one row has nothing to be compared with
    assert d.within_cluster_demean([4.0, 1.0, 3.0], ["a", "b", "b"])[0] == 0.0
    # several columns at once, column by column
    matrix = np.column_stack([values, values ** 2])
    both = d.within_cluster_demean(matrix, clusters)
    assert both.shape == matrix.shape
    assert np.allclose(both[:, 0], centred)
    with pytest.raises(ValueError):
        d.within_cluster_demean([1.0, 2.0], [1, 2, 3])


def test_fixed_effects_wild_test_matches_the_same_model_with_a_dummy_for_every_cluster():
    # Frisch-Waugh: removing cluster averages gives the same slope as putting one dummy per cluster into the model, and the same
    # bootstrap p-value, because every bootstrapped statistic is only rescaled by the same constant as the observed one.
    rng = np.random.default_rng(8)
    groups_n = 25
    clusters = np.repeat(np.arange(groups_n), rng.integers(2, 7, size=groups_n))
    n = len(clusters)
    x = rng.normal(size=n) + 0.5 * rng.normal(size=groups_n)[clusters]
    y = 0.12 * x + rng.normal(size=groups_n)[clusters] + rng.normal(size=n)          # a weak effect: the p-value lands in the middle of its range
    dummies = (clusters[:, None] == np.arange(groups_n)[None, :]).astype(float)
    full_X = np.column_stack([x, dummies])
    reference = np.linalg.lstsq(full_X, y, rcond=None)[0][0]
    restriction = np.r_[1.0, np.zeros(groups_n)]
    for null in (None, [0.25]):                                                      # a null value other than zero must be handed on, too
        within = d.wild_fixed_effects_test(y, x, [1], clusters, values=null, n_boot=999, seed=3, chunk=250)
        with_dummies = d.wild_cluster_bootstrap_test(y, full_X, restriction, clusters, values=null, n_boot=999, seed=3, chunk=250)
        assert within.estimate == pytest.approx(reference)
        assert within.estimate == pytest.approx(with_dummies.estimate)
        assert within.p_value == pytest.approx(with_dummies.p_value)
        assert 0.01 < within.p_value < 0.99
        # the only difference is the degrees-of-freedom factor: the intercepts of the clusters are not counted in the fixed-effects version
        scale = np.sqrt((n - 1) / (n - groups_n - 1))
        assert within.statistic == pytest.approx(with_dummies.statistic * scale, rel=1e-9)
        assert within.detail["se"] == pytest.approx(with_dummies.detail["se"] / scale, rel=1e-9)
    assert within.ci_low < within.estimate < within.ci_high


def test_fixed_effects_wild_test_separates_a_difference_between_clusters_from_one_inside_clusters():
    rng = np.random.default_rng(12)
    groups_n = 60
    sizes = rng.integers(3, 8, size=groups_n)
    clusters = np.repeat(np.arange(groups_n), sizes)
    cluster_level = rng.normal(size=groups_n)
    # case 1: the outcome follows the cluster, the row-level flag follows the cluster too, nothing inside the clusters
    flag_between = (rng.random(len(clusters)) < 1 / (1 + np.exp(-2.5 * cluster_level[clusters]))).astype(float)
    y_between = cluster_level[clusters] + 0.3 * rng.normal(size=len(clusters))
    overall = d.wild_cluster_bootstrap_test(y_between, np.column_stack([np.ones(len(flag_between)), flag_between]), [0, 1], clusters, n_boot=999, seed=1)
    inside = d.wild_fixed_effects_test(y_between, flag_between, [1], clusters, n_boot=999, seed=1)
    assert overall.p_value < 0.01                        # an artist-level difference shows up as a group difference
    assert inside.p_value > 0.05                         # but it is not there within the same artist
    # case 2: a real effect inside the clusters is found
    flag_within = (rng.random(len(clusters)) < 0.5).astype(float)
    y_within = cluster_level[clusters] + 0.8 * flag_within + 0.5 * rng.normal(size=len(clusters))
    found = d.wild_fixed_effects_test(y_within, flag_within, [1], clusters, n_boot=999, seed=2)
    assert found.p_value < 0.01
    assert found.estimate == pytest.approx(0.8, abs=0.15)
    assert found.ci_low < 0.8 < found.ci_high
