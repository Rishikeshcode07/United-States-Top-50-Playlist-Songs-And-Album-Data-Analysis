"""
tests/test_inference_classical.py  -  checks of the classical tests (src/inference/classical.py).

WHAT THIS FILE IS
    Automatic tests of every test function: its numbers are compared with scipy or with a calculation done by hand in the test,
    known identities are checked (for a 2 x 2 table the square of the z statistic IS the chi-square statistic), and a few
    simulations check that a test rejects a true null hypothesis about 5% of the time and that a test which should be fooled
    by skewed data really is.

WHY WE NEED IT
    The formulas were written out by hand so that a reader can follow them. A hand-written formula is only worth trusting once
    it has been shown to give the same answer as an independent implementation.

WHAT HAPPENS WHEN IT RUNS
    pytest runs every test; the simulations use fixed seeds and finish in a few seconds.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    No test result in the notebook rests on a formula that has not been checked.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.inference import classical as c


# ================================================================================================ proportions
def test_wilson_interval_known_value_and_range():
    low, high = c.wilson_interval(5, 10)
    assert (low, high) == pytest.approx((0.2366, 0.7634), abs=5e-4)
    assert c.wilson_interval(0, 20)[0] == 0.0 and c.wilson_interval(20, 20)[1] <= 1.0
    with pytest.raises(ValueError):
        c.wilson_interval(11, 10)


def test_one_proportion_ztest_by_hand():
    result = c.one_proportion_ztest(560, 1000, 0.5)
    z = (0.56 - 0.5) / np.sqrt(0.25 / 1000)
    assert result.statistic == pytest.approx(z)
    assert result.p_value == pytest.approx(2 * stats.norm.sf(z))
    assert result.estimate == pytest.approx(0.56)
    assert result.ci_low < 0.56 < result.ci_high
    assert c.one_proportion_ztest(560, 1000, 0.5, alternative="greater").p_value == pytest.approx(stats.norm.sf(z))
    with pytest.raises(ValueError):
        c.one_proportion_ztest(5, 10, 1.0)


def test_two_proportion_z_squared_is_the_chi_square_statistic():
    x1, n1, x2, n2 = 177, 495, 129, 449
    z = c.two_proportion_ztest(x1, n1, x2, n2)
    chi = c.chi_square_independence([[x1, n1 - x1], [x2, n2 - x2]])
    assert z.statistic ** 2 == pytest.approx(chi.statistic)
    assert z.p_value == pytest.approx(chi.p_value)
    assert chi.p_value == pytest.approx(stats.chi2_contingency([[x1, n1 - x1], [x2, n2 - x2]], correction=False)[1])


def test_two_proportion_interval_and_effect_sizes():
    result = c.two_proportion_ztest(177, 495, 129, 449)
    assert result.estimate == pytest.approx(177 / 495 - 129 / 449)
    assert result.ci_low < result.estimate < result.ci_high
    assert result.ci_low > 0                                     # the difference is significant, so the interval excludes zero
    h = 2 * np.arcsin(np.sqrt(177 / 495)) - 2 * np.arcsin(np.sqrt(129 / 449))
    assert result.detail["cohens_h"] == pytest.approx(h)
    assert result.detail["odds_ratio"] == pytest.approx((177 * (449 - 129)) / ((495 - 177) * 129))
    assert result.detail["odds_ratio_low"] < result.detail["odds_ratio"] < result.detail["odds_ratio_high"]


def test_odds_ratio_handles_a_zero_cell():
    odds, low, high = c.odds_ratio(5, 0, 2, 10)
    assert np.isfinite([odds, low, high]).all() and low < odds < high


def test_two_proportion_test_keeps_its_size_under_the_null():
    rng = np.random.default_rng(4)
    p_values = [c.two_proportion_ztest(rng.binomial(300, 0.3), 300, rng.binomial(300, 0.3), 300).p_value for _ in range(2000)]
    assert 0.035 < np.mean(np.array(p_values) < 0.05) < 0.065


# ================================================================================================ tables
def test_chi_square_independence_matches_scipy_and_cramers_v():
    table = np.array([[30, 20, 10], [20, 25, 35]])
    result = c.chi_square_independence(table)
    reference = stats.chi2_contingency(table, correction=False)
    assert result.statistic == pytest.approx(reference[0]) and result.p_value == pytest.approx(reference[1]) and result.df == reference[2]
    assert result.detail["cramers_v"] == pytest.approx(np.sqrt(reference[0] / table.sum()))
    assert result.detail["min_expected"] == pytest.approx(reference[3].min())


def test_cramers_v_is_one_for_a_perfect_link_and_zero_for_none():
    assert c.chi_square_independence([[50, 0], [0, 50]]).detail["cramers_v"] == pytest.approx(1.0)
    assert c.chi_square_independence([[25, 25], [25, 25]]).detail["cramers_v"] == pytest.approx(0.0)


def test_chi_square_needs_a_real_table():
    with pytest.raises(ValueError):
        c.chi_square_independence([1, 2, 3])


def test_fisher_exact_matches_scipy():
    table = [[8, 2], [1, 5]]
    result = c.fisher_exact_2x2(table)
    assert result.p_value == pytest.approx(stats.fisher_exact(table)[1])
    assert result.statistic == pytest.approx(20.0)
    with pytest.raises(ValueError):
        c.fisher_exact_2x2([[1, 2, 3], [4, 5, 6]])


def test_goodness_of_fit_matches_scipy_and_w():
    observed = [357, 240, 48, 65, 61, 42, 81]
    shares = np.array([80, 79, 79, 80, 77, 80, 80]) / 555
    result = c.chi_square_goodness_of_fit(observed, shares)
    reference = stats.chisquare(observed, f_exp=shares * sum(observed))
    assert result.statistic == pytest.approx(reference.statistic) and result.p_value == pytest.approx(reference.pvalue)
    assert result.df == 6
    assert result.detail["cohens_w"] == pytest.approx(np.sqrt(reference.statistic / sum(observed)))
    with pytest.raises(ValueError):
        c.chi_square_goodness_of_fit([1, 2], [0.5, 0.0])


def test_cochran_armitage_squared_equals_n_times_r_squared():
    successes, totals = np.array([30, 57, 70, 73, 76]), np.array([125, 191, 218, 201, 209])
    result = c.cochran_armitage_trend(successes, totals)
    scores = np.repeat(np.arange(5), totals)
    outcomes = np.concatenate([np.r_[np.ones(s), np.zeros(t - s)] for s, t in zip(successes, totals)])
    r = np.corrcoef(scores, outcomes)[0, 1]
    assert result.statistic ** 2 == pytest.approx(totals.sum() * r ** 2)
    assert result.estimate > 0 and result.p_value < 0.05


def test_cochran_armitage_has_more_power_than_the_omnibus_test_for_a_steady_trend():
    successes, totals = np.array([24, 28, 32, 36, 40]), np.array([100, 100, 100, 100, 100])
    trend = c.cochran_armitage_trend(successes, totals)
    omnibus = c.chi_square_independence(np.column_stack([successes, totals - successes]))
    assert trend.p_value < omnibus.p_value


def test_cochran_armitage_no_trend_gives_a_large_p_value():
    assert c.cochran_armitage_trend([30, 30, 30, 30], [100, 100, 100, 100]).p_value == pytest.approx(1.0)
    with pytest.raises(ValueError):
        c.cochran_armitage_trend([1, 2], [10, 10])


# ================================================================================================ spread
def test_f_test_by_hand_and_interval():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0, 2, 40), rng.normal(0, 1, 60)
    result = c.f_test_two_variances(a, b)
    ratio = a.var(ddof=1) / b.var(ddof=1)
    assert result.statistic == pytest.approx(ratio) and result.df == (39, 59)
    assert result.p_value == pytest.approx(2 * min(stats.f.cdf(ratio, 39, 59), stats.f.sf(ratio, 39, 59)))
    assert result.ci_low < ratio < result.ci_high
    assert result.p_value < 0.01                                  # variance ratio of about 4 with these sample sizes is clearly detected


def test_f_test_keeps_its_size_for_normal_data_but_is_fooled_by_heavy_tails():
    rng = np.random.default_rng(6)
    normal = np.mean([c.f_test_two_variances(rng.normal(size=30), rng.normal(size=30)).p_value < 0.05 for _ in range(1500)])
    heavy = np.mean([c.f_test_two_variances(rng.standard_t(3, 30), rng.standard_t(3, 30)).p_value < 0.05 for _ in range(1500)])
    assert 0.035 < normal < 0.07
    assert heavy > 0.12                                           # the same null hypothesis is true, but the F test rejects far too often


def test_levene_matches_scipy_and_is_robust_to_heavy_tails():
    rng = np.random.default_rng(7)
    a, b, d = rng.normal(0, 1, 50), rng.normal(0, 1.5, 60), rng.normal(0, 1, 40)
    result = c.levene_test([a, b, d])
    reference = stats.levene(a, b, d, center="median")
    assert result.statistic == pytest.approx(reference.statistic) and result.p_value == pytest.approx(reference.pvalue)
    assert result.df == (2, 147)
    robust = np.mean([c.levene_test([rng.standard_t(3, 40), rng.standard_t(3, 40)]).p_value < 0.05 for _ in range(1000)])
    assert robust < 0.10


def test_bartlett_matches_scipy():
    rng = np.random.default_rng(8)
    a, b = rng.normal(size=30), rng.normal(0, 2, 30)
    assert c.bartlett_test([a, b]).p_value == pytest.approx(stats.bartlett(a, b).pvalue)


# ================================================================================================ means
def test_welch_matches_scipy_including_the_interval():
    rng = np.random.default_rng(9)
    a, b = rng.normal(5, 3, 80), rng.normal(4, 1, 120)
    result = c.two_sample_ttest(a, b)
    reference = stats.ttest_ind(a, b, equal_var=False)
    interval = reference.confidence_interval(0.95)
    assert result.statistic == pytest.approx(reference.statistic) and result.p_value == pytest.approx(reference.pvalue)
    assert result.df == pytest.approx(reference.df)
    assert (result.ci_low, result.ci_high) == pytest.approx((interval.low, interval.high))
    assert result.estimate == pytest.approx(a.mean() - b.mean())


def test_student_matches_scipy_and_hedges_g_is_a_shrunk_d():
    rng = np.random.default_rng(10)
    a, b = rng.normal(1, 1, 25), rng.normal(0, 1, 30)
    result = c.two_sample_ttest(a, b, equal_var=True)
    reference = stats.ttest_ind(a, b, equal_var=True)
    assert result.statistic == pytest.approx(reference.statistic) and result.p_value == pytest.approx(reference.pvalue)
    pooled = np.sqrt((24 * a.var(ddof=1) + 29 * b.var(ddof=1)) / 53)
    d = (a.mean() - b.mean()) / pooled
    assert abs(result.detail["hedges_g"]) < abs(d)
    assert result.detail["hedges_g"] == pytest.approx(d * (1 - 3 / (4 * 55 - 9)))
    assert result.detail["hedges_g_low"] < result.detail["hedges_g"] < result.detail["hedges_g_high"]


def test_one_sample_and_paired_t_match_scipy():
    rng = np.random.default_rng(11)
    x = rng.normal(0.4, 1, 60)
    y = x + rng.normal(0.3, 0.5, 60)
    one = c.one_sample_ttest(x, 0.0)
    assert one.statistic == pytest.approx(stats.ttest_1samp(x, 0.0).statistic) and one.p_value == pytest.approx(stats.ttest_1samp(x, 0.0).pvalue)
    paired = c.paired_ttest(y, x)
    reference = stats.ttest_rel(y, x)
    assert paired.statistic == pytest.approx(reference.statistic) and paired.p_value == pytest.approx(reference.pvalue)
    assert paired.estimate == pytest.approx((y - x).mean())
    with pytest.raises(ValueError):
        c.paired_ttest([1, 2, 3], [1, 2])


def test_anova_matches_scipy_and_eta_squared_by_hand():
    rng = np.random.default_rng(12)
    groups = [rng.normal(0, 1, 30), rng.normal(0.5, 1, 40), rng.normal(1, 1, 50)]
    result = c.one_way_anova(groups)
    reference = stats.f_oneway(*groups)
    assert result.statistic == pytest.approx(reference.statistic) and result.p_value == pytest.approx(reference.pvalue)
    everyone = np.concatenate(groups)
    ss_between = sum(len(g) * (g.mean() - everyone.mean()) ** 2 for g in groups)
    ss_total = ((everyone - everyone.mean()) ** 2).sum()
    assert result.detail["eta_squared"] == pytest.approx(ss_between / ss_total)
    assert 0 <= result.detail["omega_squared"] < result.detail["eta_squared"]


def test_welch_anova_keeps_its_size_when_the_classical_anova_does_not():
    rng = np.random.default_rng(13)
    sizes, spreads = (10, 40, 100), (4.0, 2.0, 1.0)             # the smallest group has the largest spread: bad for the classical F test
    classical = welch = 0
    runs = 1500
    for _ in range(runs):
        groups = [rng.normal(0, s, n) for n, s in zip(sizes, spreads)]
        classical += c.one_way_anova(groups).p_value < 0.05
        welch += c.welch_anova(groups).p_value < 0.05
    assert classical / runs > 0.09
    assert 0.035 < welch / runs < 0.07


def test_welch_anova_matches_a_hand_calculation():
    groups = [np.array([4.1, 5.2, 6.3, 5.5, 4.9]), np.array([6.8, 7.9, 8.1, 7.2, 9.0, 6.5]), np.array([5.1, 5.9, 6.0, 5.5])]
    n = np.array([len(g) for g in groups], dtype=float)
    m = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) for g in groups])
    w = n / v
    mw = (w * m).sum() / w.sum()
    lam = 3 * (((1 - w / w.sum()) ** 2) / (n - 1)).sum() / (3 ** 2 - 1)
    expected_f = ((w * (m - mw) ** 2).sum() / 2) / (1 + 2 * lam * (3 - 2) / 3)
    result = c.welch_anova(groups)
    assert result.statistic == pytest.approx(expected_f)
    assert result.df[1] == pytest.approx(1 / lam)


# ================================================================================================ ranks
def test_mann_whitney_matches_scipy_and_the_effect_sizes_by_brute_force():
    rng = np.random.default_rng(14)
    a, b = rng.poisson(8, 60).astype(float), rng.poisson(6, 70).astype(float)
    result = c.mann_whitney_test(a, b)
    reference = stats.mannwhitneyu(a, b, alternative="two-sided")
    assert result.statistic == pytest.approx(reference.statistic) and result.p_value == pytest.approx(reference.pvalue)
    wins = (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
    assert result.estimate == pytest.approx(wins / (len(a) * len(b)))
    assert result.detail["cliffs_delta"] == pytest.approx(2 * result.estimate - 1)
    assert result.detail["cliffs_delta"] == pytest.approx(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


def test_hodges_lehmann_shift_small_example():
    assert c.hodges_lehmann_shift([1, 2, 3], [0, 1, 2]) == pytest.approx(1.0)
    assert c.hodges_lehmann_shift([5, 5, 5], [5, 5, 5]) == 0.0


def test_kruskal_wallis_matches_scipy_and_epsilon_squared():
    rng = np.random.default_rng(15)
    groups = [rng.exponential(1.0, 40), rng.exponential(1.5, 50), rng.exponential(2.0, 60)]
    result = c.kruskal_wallis_test(groups)
    reference = stats.kruskal(*groups)
    assert result.statistic == pytest.approx(reference.statistic) and result.p_value == pytest.approx(reference.pvalue)
    n = 150
    assert result.detail["epsilon_squared"] == pytest.approx(reference.statistic * (n + 1) / (n ** 2 - 1))
    assert 0 < result.detail["epsilon_squared"] < 1


def test_wilcoxon_matches_scipy_and_the_rank_biserial_effect():
    rng = np.random.default_rng(16)
    x = rng.normal(0, 1, 50)
    y = x - rng.normal(0.4, 0.6, 50)
    result = c.wilcoxon_signed_rank_test(x, y)
    assert result.p_value == pytest.approx(stats.wilcoxon(x - y).pvalue)
    assert c.wilcoxon_signed_rank_test([2, 3, 4, 5], [1, 1, 1, 1]).estimate == pytest.approx(1.0)       # every difference is positive
    assert c.wilcoxon_signed_rank_test([1, 1, 1, 1], [2, 3, 4, 5]).estimate == pytest.approx(-1.0)


def test_friedman_matches_scipy_and_kendalls_w_range():
    rng = np.random.default_rng(17)
    table = rng.normal(size=(40, 4)) + np.array([0, 0.3, 0.6, 0.9])
    result = c.friedman_test(table)
    reference = stats.friedmanchisquare(*[table[:, j] for j in range(4)])
    assert result.statistic == pytest.approx(reference.statistic) and result.p_value == pytest.approx(reference.pvalue)
    assert 0 < result.detail["kendalls_w"] < 1
    perfect = np.tile([1.0, 2.0, 3.0], (20, 1))
    assert c.friedman_test(perfect).detail["kendalls_w"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        c.friedman_test(np.ones((5, 2)))


# ================================================================================================ after the F test
def test_dunn_for_two_groups_is_the_mann_whitney_z_without_continuity_correction():
    rng = np.random.default_rng(18)
    a, b = rng.poisson(5, 45).astype(float), rng.poisson(6, 55).astype(float)         # plenty of ties on purpose
    table = c.dunn_posthoc([a, b], ["a", "b"])
    reference = stats.mannwhitneyu(a, b, use_continuity=False, method="asymptotic")
    assert table["p"].iloc[0] == pytest.approx(reference.pvalue)


def test_dunn_for_three_groups_by_hand():
    groups = [np.array([1.0, 2, 3, 4]), np.array([3.0, 4, 5, 6]), np.array([10.0, 11, 12, 13])]
    table = c.dunn_posthoc(groups, ["x", "y", "z"])
    ranks = stats.rankdata(np.concatenate(groups))
    mean_ranks = [ranks[0:4].mean(), ranks[4:8].mean(), ranks[8:12].mean()]
    _, counts = np.unique(np.concatenate(groups), return_counts=True)
    variance = 12 * 13 / 12 - (counts ** 3 - counts).sum() / (12 * 11)
    z_xz = (mean_ranks[0] - mean_ranks[2]) / np.sqrt(variance * (1 / 4 + 1 / 4))
    row = table[(table["group_a"] == "x") & (table["group_b"] == "z")].iloc[0]
    assert row["z"] == pytest.approx(z_xz)
    assert (table["p_holm"] >= table["p"]).all() and len(table) == 3


def test_games_howell_for_two_groups_is_the_welch_test():
    rng = np.random.default_rng(19)
    a, b = rng.normal(0, 3, 40), rng.normal(1, 1, 70)
    table = c.games_howell([a, b], ["a", "b"])
    reference = stats.ttest_ind(a, b, equal_var=False)
    assert table["p"].iloc[0] == pytest.approx(reference.pvalue, rel=1e-4)
    assert table["df"].iloc[0] == pytest.approx(reference.df)


def test_games_howell_p_values_are_wider_than_pairwise_welch_for_three_groups():
    rng = np.random.default_rng(20)
    groups = [rng.normal(m, 1 + m, 40) for m in (0, 0.6, 1.2)]
    table = c.games_howell(groups, ["a", "b", "c"])
    for row in table.itertuples():
        i, j = "abc".index(row.group_a), "abc".index(row.group_b)
        assert row.p >= stats.ttest_ind(groups[i], groups[j], equal_var=False).pvalue - 1e-12
        assert row.ci_low < row.difference < row.ci_high


# ================================================================================================ assumptions
def test_normality_summary_tells_a_bell_curve_from_a_skewed_sample():
    rng = np.random.default_rng(21)
    normal = c.normality_summary(rng.normal(size=400))
    skewed = c.normality_summary(rng.exponential(size=400))
    assert normal["shapiro_p"] > 0.01 and abs(normal["skewness"]) < 0.4
    assert skewed["shapiro_p"] < 1e-6 and skewed["skewness"] > 1.0
    assert c.normality_summary(rng.exponential(size=6000), max_n=1000)["n"] == 6000       # the Shapiro-Wilk test used a subset, the summary the whole sample
    again = c.normality_summary(np.arange(6000.0) ** 1.5, max_n=1000)
    assert again == c.normality_summary(np.arange(6000.0) ** 1.5, max_n=1000)           # the subset is the same every time
    with pytest.raises(ValueError):
        c.normality_summary([1, 2, 3])


def test_bad_values_are_refused():
    for bad in ([], [1.0, float("nan")], [1.0, float("inf")]):
        with pytest.raises(ValueError):
            c.two_sample_ttest(bad, [1.0, 2.0, 3.0])
