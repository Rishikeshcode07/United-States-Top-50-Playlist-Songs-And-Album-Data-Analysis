"""
src/inference/classical.py  -  the classical tests, written out so that you can see every formula.

WHAT THIS FILE IS
    One function per test. Each returns an `InferenceResult` (test statistic, degrees of freedom, p-value, the effect in its
    own units with a confidence interval, and the extras in `detail`).

        proportions        wilson_interval, one_proportion_ztest, two_proportion_ztest, odds_ratio
        counts and tables  chi_square_independence, fisher_exact_2x2, chi_square_goodness_of_fit, cochran_armitage_trend
        spread             f_test_two_variances, levene_test (Brown-Forsythe), bartlett_test
        means              one_sample_ttest, two_sample_ttest (Student or Welch), paired_ttest, one_way_anova, welch_anova
        ranks              mann_whitney_test, hodges_lehmann_shift, kruskal_wallis_test, wilcoxon_signed_rank_test, friedman_test
        after the F test   dunn_posthoc (for ranks), games_howell (for means with unequal spread)
        assumptions        normality_summary

WHY WE NEED IT
    scipy can run most of these in one line, and the notebook uses scipy as a cross-check. But a number that comes out of a black
    box teaches nothing. Here the formula is in the code, the test compares it with scipy, and the notebook shows the result with
    the effect size that scipy does not give you. Three habits run through the file:

        * Every test comes with an effect size: a p-value only says "detectable", the effect size says "how big".
        * Every interval is a confidence interval of the EFFECT (a difference of proportions, a difference of means), not just a
          p-value in disguise.
        * Tests that compare groups of a few hundred rows against each other use the versions that do not assume equal spread
          (Welch, Games-Howell) unless the equal-spread version is asked for by name.

    All tests here assume independent rows. The notebook uses `dependence.py` for the same questions when rows are related.

WHAT HAPPENS WHEN IT RUNS
    Functions only calculate; nothing is drawn or saved. Every p-value is two-sided unless `alternative` says otherwise.

EXPECTED OUTCOME
    The same statistics and p-values as scipy (the tests check this), plus effect sizes with intervals.

MOTIVE
    Know what is inside the test before trusting what comes out of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.inference.multiplicity import holm
from src.inference.results import InferenceResult


# ================================================================================================ helpers
def _z(confidence: float) -> float:
    return float(stats.norm.ppf(0.5 + confidence / 2.0))


def _values(sample, name="sample") -> np.ndarray:
    array = np.asarray(sample, dtype=float).ravel()
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty list of finite numbers (no NaN, no infinity)")
    return array


def _p_value_from_z(z: float, alternative: str) -> float:
    if alternative == "two-sided":
        return float(2 * stats.norm.sf(abs(z)))
    if alternative == "greater":
        return float(stats.norm.sf(z))
    if alternative == "less":
        return float(stats.norm.cdf(z))
    raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")


# ================================================================================================ proportions
def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    The Wilson score interval for a proportion. Better than the textbook "p +- z * sqrt(p (1 - p) / n)" when the proportion is
    near 0 or 1 or the sample is small, because it can never leave the range 0 to 1.
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n and n > 0")
    z = _z(confidence)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def one_proportion_ztest(successes: int, n: int, p0: float, alternative: str = "two-sided", confidence: float = 0.95) -> InferenceResult:
    """
    z test of one proportion against a stated value p0:  z = (p_hat - p0) / sqrt(p0 (1 - p0) / n).
    The standard error uses p0 because the test asks "if p0 were true, how surprising is p_hat?". The interval is Wilson's.
    """
    if not 0 < p0 < 1:
        raise ValueError("p0 must be between 0 and 1")
    p_hat = successes / n
    z = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
    low, high = wilson_interval(successes, n, confidence)
    return InferenceResult("one-proportion z test", float(z), _p_value_from_z(float(z), alternative), estimate=float(p_hat), ci_low=low,
                           ci_high=high, n=int(n), detail={"p0": p0, "cohens_h": float(2 * np.arcsin(np.sqrt(p_hat)) - 2 * np.arcsin(np.sqrt(p0)))})


def odds_ratio(a: int, b: int, c: int, d: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Odds ratio of a 2 x 2 table [[a, b], [c, d]] (a = exposed with the outcome, b = exposed without, c = not exposed with, d = not
    exposed without), with the Woolf interval: exp(log OR +- z * sqrt(1/a + 1/b + 1/c + 1/d)). Half is added to every cell when
    one of them is zero.
    """
    cells = np.array([a, b, c, d], dtype=float)
    if (cells == 0).any():
        cells = cells + 0.5
    a, b, c, d = cells
    log_or = np.log(a * d / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = _z(confidence)
    return float(np.exp(log_or)), float(np.exp(log_or - z * se)), float(np.exp(log_or + z * se))


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int, alternative: str = "two-sided", confidence: float = 0.95) -> InferenceResult:
    """
    z test of "the two proportions are equal":  z = (p1 - p2) / sqrt(p (1 - p) (1/n1 + 1/n2))  with p = (x1 + x2) / (n1 + n2).
    The estimate is the difference p1 - p2 with Newcombe's interval (built from the two Wilson intervals; it behaves better than
    the simple Wald interval). For a 2 x 2 table, z squared equals the chi-square statistic without continuity correction.
    """
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    l1, u1 = wilson_interval(x1, n1, confidence)
    l2, u2 = wilson_interval(x2, n2, confidence)
    difference = p1 - p2
    low = difference - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = difference + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    odds, odds_low, odds_high = odds_ratio(x1, n1 - x1, x2, n2 - x2, confidence)
    return InferenceResult("two-proportion z test", float(z), _p_value_from_z(float(z), alternative), estimate=float(difference), ci_low=float(low),
                           ci_high=float(high), n=int(n1 + n2),
                           detail={"p1": p1, "p2": p2, "pooled": pooled, "cohens_h": float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))),
                                   "odds_ratio": odds, "odds_ratio_low": odds_low, "odds_ratio_high": odds_high})


# ================================================================================================ counts and tables
def chi_square_independence(table, correction: bool = False) -> InferenceResult:
    """
    Chi-square test of independence for a table of counts: sum of (observed - expected)^2 / expected, with expected counts from
    the row and column totals. Without continuity correction by default (the correction is for 2 x 2 tables with small counts).
    Effect size: Cramer's V = sqrt(chi2 / (n * min(rows - 1, columns - 1))), from 0 (no link) to 1 (perfect link).
    """
    observed = np.asarray(table, dtype=float)
    if observed.ndim != 2 or min(observed.shape) < 2:
        raise ValueError("need a table with at least two rows and two columns")
    chi2, p_value, dof, expected = stats.chi2_contingency(observed, correction=correction)
    n = observed.sum()
    v = float(np.sqrt(chi2 / (n * (min(observed.shape) - 1))))
    return InferenceResult("chi-square test of independence" + (" (Yates)" if correction else ""), float(chi2), float(p_value), df=int(dof),
                           n=int(n), detail={"cramers_v": v, "expected": expected, "min_expected": float(expected.min()),
                                             "residuals": (observed - expected) / np.sqrt(expected)})


def fisher_exact_2x2(table) -> InferenceResult:
    """Fisher's exact test for a 2 x 2 table (no large-sample approximation). The statistic is the sample odds ratio."""
    observed = np.asarray(table, dtype=float)
    if observed.shape != (2, 2):
        raise ValueError("Fisher's exact test here is for a 2 x 2 table")
    result = stats.fisher_exact(observed)
    return InferenceResult("Fisher's exact test", float(result.statistic), float(result.pvalue), n=int(observed.sum()))


def chi_square_goodness_of_fit(observed, expected_shares) -> InferenceResult:
    """
    Chi-square goodness-of-fit test: do the observed counts follow the stated shares?  Sum of (observed - expected)^2 / expected
    with degrees of freedom = categories - 1. Effect size: Cohen's w = sqrt(chi2 / n).
    """
    observed = np.asarray(observed, dtype=float)
    shares = np.asarray(expected_shares, dtype=float)
    if observed.shape != shares.shape or observed.ndim != 1 or len(observed) < 2:
        raise ValueError("observed and expected_shares must be lists of the same length (at least 2)")
    if (shares <= 0).any():
        raise ValueError("every expected share must be above zero")
    shares = shares / shares.sum()
    n = observed.sum()
    expected = shares * n
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    dof = len(observed) - 1
    return InferenceResult("chi-square goodness-of-fit test", chi2, float(stats.chi2.sf(chi2, dof)), df=dof, n=int(n),
                           detail={"cohens_w": float(np.sqrt(chi2 / n)), "expected": expected, "residuals": (observed - expected) / np.sqrt(expected),
                                   "min_expected": float(expected.min())})


def cochran_armitage_trend(successes, totals, scores=None) -> InferenceResult:
    """
    Cochran-Armitage test for a trend in proportions across ORDERED groups ("the chi-square test for trend"). It asks one
    focused question (does the proportion go steadily up, or steadily down, with the score?) and is therefore stronger than the
    ordinary chi-square test, which asks whether the groups differ in any way at all.
        z = sum(s_i (x_i - n_i p)) / sqrt( p (1 - p) (sum(n_i s_i^2) - (sum(n_i s_i))^2 / N) )     p = overall proportion
    The estimate is the weighted slope: the change of the proportion per one unit of score.
    """
    x = np.asarray(successes, dtype=float)
    n = np.asarray(totals, dtype=float)
    s = np.arange(len(x), dtype=float) if scores is None else np.asarray(scores, dtype=float)
    if not (len(x) == len(n) == len(s)) or len(x) < 3:
        raise ValueError("need at least three ordered groups, with one count, one total and one score each")
    total = n.sum()
    p = x.sum() / total
    statistic = (s * (x - n * p)).sum()
    variance = p * (1 - p) * ((n * s ** 2).sum() - (n * s).sum() ** 2 / total)
    z = float(statistic / np.sqrt(variance))
    proportions = x / n
    s_bar = (n * s).sum() / total
    slope = float((n * (s - s_bar) * (proportions - p)).sum() / (n * (s - s_bar) ** 2).sum())
    return InferenceResult("Cochran-Armitage test for trend", z, _p_value_from_z(z, "two-sided"), estimate=slope, n=int(total),
                           detail={"proportions": proportions, "chi_square": z * z})


# ================================================================================================ spread
def f_test_two_variances(a, b, confidence: float = 0.95) -> InferenceResult:
    """
    The F test for equal variances: F = var(a) / var(b) with (n_a - 1, n_b - 1) degrees of freedom, two-sided. The estimate is
    the variance ratio with its interval. WARNING: this test is very sensitive to non-normal data (a few extreme values are
    enough to make it reject). Levene's test is the safer choice for real data; the F test is here because everyone is asked
    about it and because the notebook shows the difference.
    """
    a, b = _values(a, "a"), _values(b, "b")
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each sample needs at least two values")
    ratio = float(a.var(ddof=1) / b.var(ddof=1))
    df1, df2 = len(a) - 1, len(b) - 1
    p_value = float(min(1.0, 2 * min(stats.f.cdf(ratio, df1, df2), stats.f.sf(ratio, df1, df2))))
    tail = (1 - confidence) / 2
    low = ratio / float(stats.f.ppf(1 - tail, df1, df2))
    high = ratio / float(stats.f.ppf(tail, df1, df2))
    return InferenceResult("F test for equal variances", ratio, p_value, df=(df1, df2), estimate=ratio, ci_low=low, ci_high=high, n=len(a) + len(b),
                           detail={"sd_a": float(a.std(ddof=1)), "sd_b": float(b.std(ddof=1))})


def levene_test(groups, center: str = "median") -> InferenceResult:
    """
    Levene's test for equal spread across groups. With center='median' (the Brown-Forsythe version) it is robust to skewed data.
    It is a one-way ANOVA on the absolute distances of the values from their group's centre.
    """
    arrays = [_values(g) for g in groups]
    result = stats.levene(*arrays, center=center)
    k = len(arrays)
    return InferenceResult("Brown-Forsythe test for equal spread" if center == "median" else "Levene test for equal spread", float(result.statistic),
                           float(result.pvalue), df=(k - 1, int(sum(len(a) for a in arrays) - k)), n=int(sum(len(a) for a in arrays)))


def bartlett_test(groups) -> InferenceResult:
    """Bartlett's test for equal variances (the most powerful one for normal data, and the least safe for anything else)."""
    arrays = [_values(g) for g in groups]
    result = stats.bartlett(*arrays)
    return InferenceResult("Bartlett test for equal variances", float(result.statistic), float(result.pvalue), df=len(arrays) - 1,
                           n=int(sum(len(a) for a in arrays)))


# ================================================================================================ means
def _hedges_g(a: np.ndarray, b: np.ndarray, confidence: float) -> tuple[float, float, float]:
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / pooled_sd
    g = d * (1 - 3 / (4 * (n1 + n2) - 9))
    se = np.sqrt((n1 + n2) / (n1 * n2) + g * g / (2 * (n1 + n2 - 2)))
    z = _z(confidence)
    return float(g), float(g - z * se), float(g + z * se)


def one_sample_ttest(sample, mu0: float, confidence: float = 0.95) -> InferenceResult:
    """t test of one mean against mu0:  t = (mean - mu0) / (sd / sqrt(n)) with n - 1 degrees of freedom. Effect: Cohen's d."""
    x = _values(sample)
    n = len(x)
    if n < 2:
        raise ValueError("need at least two values")
    se = x.std(ddof=1) / np.sqrt(n)
    t = (x.mean() - mu0) / se
    margin = float(stats.t.ppf(0.5 + confidence / 2, n - 1)) * se
    return InferenceResult("one-sample t test", float(t), float(2 * stats.t.sf(abs(t), n - 1)), df=n - 1, estimate=float(x.mean() - mu0),
                           ci_low=float(x.mean() - mu0 - margin), ci_high=float(x.mean() - mu0 + margin), n=n,
                           detail={"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "cohens_d": float((x.mean() - mu0) / x.std(ddof=1))})


def two_sample_ttest(a, b, equal_var: bool = False, confidence: float = 0.95) -> InferenceResult:
    """
    t test of "the two means are equal". equal_var=False (the default) is Welch's test, which does not assume equal spread:
        t = (mean_a - mean_b) / sqrt(var_a / n_a + var_b / n_b), degrees of freedom by the Welch-Satterthwaite formula.
    equal_var=True is Student's test with one pooled variance. The estimate is the difference of means with its interval;
    Hedges' g (Cohen's d with a small-sample correction) is in `detail`.
    """
    a, b = _values(a, "a"), _values(b, "b")
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("each sample needs at least two values")
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    difference = float(a.mean() - b.mean())
    if equal_var:
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = np.sqrt(pooled * (1 / n1 + 1 / n2))
        df = float(n1 + n2 - 2)
        name = "Student t test (equal variances)"
    else:
        se = np.sqrt(v1 / n1 + v2 / n2)
        df = float((v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)))
        name = "Welch t test"
    t = difference / se
    margin = float(stats.t.ppf(0.5 + confidence / 2, df)) * se
    g, g_low, g_high = _hedges_g(a, b, confidence)
    return InferenceResult(name, float(t), float(2 * stats.t.sf(abs(t), df)), df=df, estimate=difference, ci_low=difference - margin,
                           ci_high=difference + margin, n=n1 + n2,
                           detail={"mean_a": float(a.mean()), "mean_b": float(b.mean()), "sd_a": float(np.sqrt(v1)), "sd_b": float(np.sqrt(v2)),
                                   "hedges_g": g, "hedges_g_low": g_low, "hedges_g_high": g_high, "se": float(se)})


def paired_ttest(first, second, confidence: float = 0.95) -> InferenceResult:
    """Paired t test: a one-sample t test of the pairwise differences (first - second) against zero. Effect: d_z = mean / sd of the differences."""
    x, y = _values(first, "first"), _values(second, "second")
    if len(x) != len(y):
        raise ValueError("paired samples must have the same length")
    result = one_sample_ttest(x - y, 0.0, confidence)
    return InferenceResult("paired t test", result.statistic, result.p_value, df=result.df, estimate=result.estimate, ci_low=result.ci_low,
                           ci_high=result.ci_high, n=result.n, detail={"mean_first": float(x.mean()), "mean_second": float(y.mean()),
                                                                       "cohens_dz": result.detail["cohens_d"]})


def one_way_anova(groups) -> InferenceResult:
    """
    One-way ANOVA F test of "all group means are equal":  F = (between-group mean square) / (within-group mean square).
    Effect sizes: eta squared = SS_between / SS_total (the share of the variance that the groups explain) and the less biased
    omega squared. Assumes equal spread; see `welch_anova` and `levene_test`.
    """
    arrays = [_values(g) for g in groups]
    k = len(arrays)
    if k < 2:
        raise ValueError("need at least two groups")
    sizes = np.array([len(a) for a in arrays], dtype=float)
    means = np.array([a.mean() for a in arrays])
    n = sizes.sum()
    grand = np.concatenate(arrays).mean()
    ss_between = float((sizes * (means - grand) ** 2).sum())
    ss_within = float(sum(((a - a.mean()) ** 2).sum() for a in arrays))
    ms_between, ms_within = ss_between / (k - 1), ss_within / (n - k)
    f_value = ms_between / ms_within
    ss_total = ss_between + ss_within
    return InferenceResult("one-way ANOVA F test", float(f_value), float(stats.f.sf(f_value, k - 1, n - k)), df=(k - 1, int(n - k)), n=int(n),
                           detail={"eta_squared": ss_between / ss_total, "omega_squared": max(0.0, (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)),
                                   "group_means": means.tolist(), "group_sizes": sizes.astype(int).tolist(), "ms_within": ms_within})


def welch_anova(groups) -> InferenceResult:
    """
    Welch's ANOVA: the F test of equal means that does not assume equal spread. Groups are weighted by n / variance; the
    denominator degrees of freedom are reduced to match (Welch, 1951).
    """
    arrays = [_values(g) for g in groups]
    k = len(arrays)
    if k < 2 or any(len(a) < 2 for a in arrays):
        raise ValueError("need at least two groups with at least two values each")
    sizes = np.array([len(a) for a in arrays], dtype=float)
    means = np.array([a.mean() for a in arrays])
    variances = np.array([a.var(ddof=1) for a in arrays])
    weights = sizes / variances
    weighted_mean = (weights * means).sum() / weights.sum()
    tail = (((1 - weights / weights.sum()) ** 2) / (sizes - 1)).sum()
    numerator = (weights * (means - weighted_mean) ** 2).sum() / (k - 1)
    denominator = 1 + 2 * (k - 2) / (k ** 2 - 1) * tail
    f_value = numerator / denominator
    df2 = (k ** 2 - 1) / (3 * tail)
    return InferenceResult("Welch ANOVA", float(f_value), float(stats.f.sf(f_value, k - 1, df2)), df=(k - 1, float(df2)), n=int(sizes.sum()),
                           detail={"group_means": means.tolist()})


# ================================================================================================ ranks
def mann_whitney_test(a, b) -> InferenceResult:
    """
    Mann-Whitney U (Wilcoxon rank-sum) test: do values of one group tend to be larger than values of the other? It compares
    ranks, so skewed data and outliers do not matter. The estimate is the probability of superiority P(a > b) + half of P(a = b)
    (0.5 = no difference); Cliff's delta = 2 * that - 1 (from -1 to +1) is in `detail`, as is the common-language statement.
    """
    a, b = _values(a, "a"), _values(b, "b")
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    superiority = float(result.statistic / (len(a) * len(b)))
    return InferenceResult("Mann-Whitney U test", float(result.statistic), float(result.pvalue), estimate=superiority, n=len(a) + len(b),
                           detail={"cliffs_delta": 2 * superiority - 1, "median_a": float(np.median(a)), "median_b": float(np.median(b))})


def hodges_lehmann_shift(a, b) -> float:
    """The Hodges-Lehmann estimate of the shift between two groups: the median of ALL pairwise differences a_i - b_j."""
    a, b = _values(a, "a"), _values(b, "b")
    return float(np.median((a[:, None] - b[None, :]).ravel()))


def kruskal_wallis_test(groups) -> InferenceResult:
    """
    Kruskal-Wallis H test: the rank-based version of the one-way ANOVA ("do the groups have the same distribution?").
    Effect size: epsilon squared = H (n + 1) / (n^2 - 1), the rank-based share of the variance explained by the groups.
    """
    arrays = [_values(g) for g in groups]
    result = stats.kruskal(*arrays)
    n = sum(len(a) for a in arrays)
    return InferenceResult("Kruskal-Wallis H test", float(result.statistic), float(result.pvalue), df=len(arrays) - 1, n=n,
                           detail={"epsilon_squared": float(result.statistic * (n + 1) / (n ** 2 - 1)), "medians": [float(np.median(a)) for a in arrays]})


def wilcoxon_signed_rank_test(first, second) -> InferenceResult:
    """
    Wilcoxon signed-rank test for paired data (the rank-based partner of the paired t test). The effect size is the matched-pairs
    rank-biserial correlation, (sum of positive ranks - sum of negative ranks) / total of ranks, from -1 to +1. Zero differences are dropped.
    """
    x, y = _values(first, "first"), _values(second, "second")
    if len(x) != len(y):
        raise ValueError("paired samples must have the same length")
    differences = x - y
    differences = differences[differences != 0]
    result = stats.wilcoxon(differences)
    ranks = stats.rankdata(np.abs(differences))
    effect = float((ranks[differences > 0].sum() - ranks[differences < 0].sum()) / ranks.sum())
    return InferenceResult("Wilcoxon signed-rank test", float(result.statistic), float(result.pvalue), estimate=effect, n=len(x),
                           detail={"rank_biserial": effect, "median_difference": float(np.median(x - y))})


def friedman_test(table) -> InferenceResult:
    """
    Friedman test: the same blocks (rows) measured under several conditions (columns); do the conditions differ? Within each
    block the conditions are ranked. Effect size: Kendall's W = chi2 / (blocks * (conditions - 1)), from 0 (the conditions are
    ranked differently in every block) to 1 (every block ranks them the same way).
    """
    data = np.asarray(table, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("need a table with blocks as rows and at least three conditions as columns")
    result = stats.friedmanchisquare(*[data[:, j] for j in range(data.shape[1])])
    blocks, conditions = data.shape
    return InferenceResult("Friedman test", float(result.statistic), float(result.pvalue), df=conditions - 1, n=blocks,
                           detail={"kendalls_w": float(result.statistic / (blocks * (conditions - 1))), "column_means": data.mean(axis=0).tolist()})


# ================================================================================================ after the F test
def dunn_posthoc(groups, labels, adjust: bool = True) -> pd.DataFrame:
    """
    Dunn's test for all pairs of groups after a Kruskal-Wallis test. All values are ranked together; two groups differ when
    their mean ranks differ by more than chance allows (with the tie correction). Returns one row per pair with z, the raw p-value
    and, when `adjust` is true, the Holm-adjusted p-value (the pairs are a family of tests).
    """
    arrays = [_values(g) for g in groups]
    pooled = np.concatenate(arrays)
    ranks = stats.rankdata(pooled)
    n = len(pooled)
    _, counts = np.unique(pooled, return_counts=True)
    tie_term = (counts ** 3 - counts).sum() / (12.0 * (n - 1))
    variance = n * (n + 1) / 12.0 - tie_term
    edges = np.cumsum([0] + [len(a) for a in arrays])
    mean_ranks = [ranks[edges[i]:edges[i + 1]].mean() for i in range(len(arrays))]
    rows = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            z = (mean_ranks[i] - mean_ranks[j]) / np.sqrt(variance * (1.0 / len(arrays[i]) + 1.0 / len(arrays[j])))
            rows.append({"group_a": labels[i], "group_b": labels[j], "mean_rank_a": mean_ranks[i], "mean_rank_b": mean_ranks[j], "z": float(z),
                         "p": float(2 * stats.norm.sf(abs(z)))})
    table = pd.DataFrame(rows)
    if adjust:
        table["p_holm"] = holm(table["p"].to_numpy())
    return table


def games_howell(groups, labels, confidence: float = 0.95) -> pd.DataFrame:
    """
    Games-Howell test for all pairs of means after an ANOVA when the groups have unequal spread and size. Each pair gets a Welch
    t statistic, Welch degrees of freedom and a p-value from the studentized range distribution (which already accounts for the
    number of pairs). Returns the difference, its interval, the statistic and the p-value of every pair.
    """
    arrays = [_values(g) for g in groups]
    k = len(arrays)
    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = arrays[i], arrays[j]
            v1, v2 = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
            se = np.sqrt(v1 + v2)
            df = (v1 + v2) ** 2 / (v1 ** 2 / (len(a) - 1) + v2 ** 2 / (len(b) - 1))
            difference = a.mean() - b.mean()
            q = abs(difference) / se * np.sqrt(2.0)
            critical = float(stats.studentized_range.ppf(confidence, k, df)) / np.sqrt(2.0)
            rows.append({"group_a": labels[i], "group_b": labels[j], "difference": float(difference), "ci_low": float(difference - critical * se),
                         "ci_high": float(difference + critical * se), "t": float(difference / se), "df": float(df),
                         "p": float(stats.studentized_range.sf(q, k, df))})
    return pd.DataFrame(rows)


# ================================================================================================ assumptions
def normality_summary(sample, max_n: int = 5000, seed: int = 20240518) -> dict:
    """
    How far from a bell curve is a sample? Skewness (0 for a symmetric curve), excess kurtosis (0 for a normal curve), and the
    Shapiro-Wilk test (null hypothesis: the sample is normal). Shapiro-Wilk is only reliable up to about 5,000 values, so a larger
    sample is cut down to a random 5,000. With hundreds of values the test rejects tiny departures that do not matter for a t test:
    look at the skewness and the Q-Q plot as well.
    """
    x = _values(sample)
    if len(x) < 8:
        raise ValueError("need at least 8 values")
    used = x
    if len(x) > max_n:
        keep = np.argsort(np.random.default_rng(seed).random(len(x)))[:max_n]       # a random subset, the same on every computer
        used = x[keep]
    shapiro = stats.shapiro(used)
    return {"n": len(x), "skewness": float(stats.skew(x)), "excess_kurtosis": float(stats.kurtosis(x)), "shapiro_w": float(shapiro.statistic),
            "shapiro_p": float(shapiro.pvalue)}
