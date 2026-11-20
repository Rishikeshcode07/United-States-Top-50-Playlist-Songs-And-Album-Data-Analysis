"""
src/inference/dependence.py  -  tests and intervals that allow for rows that are NOT independent of each other.

WHAT THIS FILE IS
    The classical tests (t, F, chi-square) assume that every row is an independent draw. Two kinds of dependence break that
    assumption in this project:

        songs of the SAME ARTIST resemble each other (Taylor Swift alone has 96 songs on the chart), and
        DAYS in a row resemble each other (the average popularity of today is close to the one of yesterday).

    This file holds the tools that stay honest under both:

        fit_ols, coefficient_test, wald_test   a regression that can use "cluster-robust" standard errors: rows of one cluster
                                               (one artist) may be related in any way, only different clusters must be independent
        cluster_robust_*                       ready-made versions for the questions of the project: two groups, several groups,
                                               a slope, and a 2 x 2 interaction
        wild_cluster_bootstrap_test, wild_*    the same questions with a p-value from the wild cluster bootstrap-t, which keeps its 5% when a few
                                               artists own many songs (the t distribution of the plain sandwich test is then too optimistic)
        wild_fixed_effects_test                the same wild bootstrap, but comparing rows of the SAME artist only (what is left after every artist's
                                               own average is taken away), so that "which artist is it" can no longer explain a difference
        cluster_bootstrap_weights, cluster_bootstrap   the artist bootstrap: resample whole artists, not single songs
        intraclass_correlation                 how strongly rows of one cluster resemble each other, and the design effect
        newey_west_cov, hac_trend_test         a trend test whose standard error allows for correlation from day to day
        moving_block_indices, block_bootstrap_trend, autocorrelation   the block bootstrap for a time series, and the correlation from day to day
        permutation_p_value, exact_split_permutation_test   permutation tests (shuffle the labels, see how rare the result is)
        mann_kendall_trend                     the rank-based trend test

WHY WE NEED IT
    If 96 songs of one artist count as 96 independent pieces of evidence, the standard error is too small, the p-value too
    small and the confidence interval too narrow. The honest standard error treats an artist as ONE cluster. For a variable
    that is hardly clustered (reaching the top 10 has an intraclass correlation near zero) the two give almost the same
    answer; for a variable that is strongly clustered (explicit or not: 0.57) the cluster-robust one is the one to trust.

WHAT HAPPENS WHEN IT RUNS
    Functions return `InferenceResult` objects (see results.py) or plain numpy arrays. Every random method takes a `seed` and
    uses only the integer and uniform generators of numpy, whose streams do not change between versions, so a result is the
    same on every computer and every numpy version.

EXPECTED OUTCOME
    With `clusters=None` the regression tools reproduce the classical t and F tests exactly (the tests check this); with
    clusters the standard errors grow in proportion to how much the rows of a cluster resemble each other.

MOTIVE
    Ask "how many independent pieces of evidence do we really have?" before asking "how small is the p-value?".
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.inference.results import InferenceResult


# ================================================================================================ small helpers
def _codes(labels) -> tuple[np.ndarray, int]:
    """Turn any labels (names, numbers) into the codes 0 ... G-1 and return the codes and G."""
    unique, codes = np.unique(np.asarray(labels), return_inverse=True)
    return codes.astype(int), len(unique)


def _as_float(values, name="values") -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty list of finite numbers (no NaN, no infinity)")
    return array


# ================================================================================================ regression with clusters
@dataclass(frozen=True)
class OlsFit:
    """A fitted least-squares line or plane. `groups` is the number of clusters, or None when no clusters were given."""

    coef: np.ndarray
    cov: np.ndarray
    n: int
    k: int
    groups: int | None
    df: int                         # degrees of freedom of the t and F reference: G - 1 with clusters, n - k without
    residuals: np.ndarray

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))


def fit_ols(y, X, clusters=None) -> OlsFit:
    """
    Ordinary least squares, with the classical covariance matrix (clusters=None) or the cluster-robust one.

    The cluster-robust ("CR1", the sandwich) covariance is
        (X'X)^-1 [ sum over clusters g of  X_g' u_g u_g' X_g ] (X'X)^-1  *  G/(G-1) * (n-1)/(n-k)
    where u_g are the residuals of cluster g. The middle part lets the residuals of one cluster be related in any way.
    """
    y = _as_float(y, "y")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    if len(y) != n:
        raise ValueError("y and X must have the same number of rows")
    if n <= k:
        raise ValueError("there must be more rows than coefficients")
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ (X.T @ y)
    residuals = y - X @ coef
    if clusters is None:
        sigma2 = float(residuals @ residuals) / (n - k)
        return OlsFit(coef, sigma2 * xtx_inv, n, k, None, n - k, residuals)
    codes, groups = _codes(clusters)
    if len(codes) != n:
        raise ValueError("clusters must have one entry per row")
    if groups < 2:
        raise ValueError("cluster-robust standard errors need at least two clusters")
    scores = X * residuals[:, None]                                  # each row's contribution to the gradient
    cluster_sums = np.zeros((groups, k))
    np.add.at(cluster_sums, codes, scores)                           # add the rows of each cluster together
    meat = cluster_sums.T @ cluster_sums
    correction = groups / (groups - 1.0) * (n - 1.0) / (n - k)
    return OlsFit(coef, correction * xtx_inv @ meat @ xtx_inv, n, k, groups, groups - 1, residuals)


def coefficient_test(fit: OlsFit, index: int, null: float = 0.0, confidence: float = 0.95, name: str = "coefficient test") -> InferenceResult:
    """t test of one coefficient against `null`, with its confidence interval (t distribution with fit.df degrees of freedom)."""
    estimate = float(fit.coef[index])
    se = float(np.sqrt(fit.cov[index, index]))
    t_value = (estimate - null) / se
    p_value = float(2 * stats.t.sf(abs(t_value), fit.df))
    margin = float(stats.t.ppf(0.5 + confidence / 2, fit.df)) * se
    return InferenceResult(name, float(t_value), p_value, df=fit.df, estimate=estimate, ci_low=estimate - margin,
                           ci_high=estimate + margin, n=fit.n, detail={"se": se, "clusters": fit.groups})


def wald_test(fit: OlsFit, restrictions, values=None, name: str = "Wald F test") -> InferenceResult:
    """
    F test of the linear restrictions  R b = r  (all rows of R together). With cluster-robust errors the reference is the F
    distribution with (q, G - 1) degrees of freedom, q being the number of restrictions.
    """
    R = np.atleast_2d(np.asarray(restrictions, dtype=float))
    r = np.zeros(R.shape[0]) if values is None else np.asarray(values, dtype=float)
    gap = R @ fit.coef - r
    middle = R @ fit.cov @ R.T
    q = R.shape[0]
    f_value = float(gap @ np.linalg.solve(middle, gap)) / q
    return InferenceResult(name, f_value, float(stats.f.sf(f_value, q, fit.df)), df=(q, fit.df), n=fit.n,
                           detail={"clusters": fit.groups})


# ================================================================================================ ready-made comparisons
def cluster_robust_two_group_test(y, in_group, clusters=None, confidence: float = 0.95, name: str | None = None) -> InferenceResult:
    """
    Difference between two groups, estimate = mean(y | group) - mean(y | not group). For a yes/no outcome this is the difference
    between two proportions (a "linear probability model"). With clusters=None it is the classical equal-variance t test.
    """
    flag = np.asarray(in_group, dtype=float).ravel()
    fit = fit_ols(y, np.column_stack([np.ones(len(flag)), flag]), clusters)
    label = name or ("cluster-robust difference between two groups" if clusters is not None else "difference between two groups (rows treated as independent)")
    return coefficient_test(fit, 1, confidence=confidence, name=label)


def cluster_robust_k_group_test(y, groups, clusters=None, name: str | None = None) -> InferenceResult:
    """F test that the means of k groups are all equal (a one-way ANOVA; with clusters, the cluster-robust version)."""
    codes, k = _codes(groups)
    if k < 2:
        raise ValueError("there must be at least two groups")
    dummies = (codes[:, None] == np.arange(1, k)[None, :]).astype(float)
    fit = fit_ols(y, np.column_stack([np.ones(len(codes)), dummies]), clusters)
    restrictions = np.hstack([np.zeros((k - 1, 1)), np.eye(k - 1)])
    label = name or ("cluster-robust test of equal group means" if clusters is not None else "test of equal group means (rows treated as independent)")
    result = wald_test(fit, restrictions, name=label)
    means = [float(np.mean(np.asarray(y, dtype=float)[codes == g])) for g in range(k)]
    return InferenceResult(result.name, result.statistic, result.p_value, df=result.df, n=result.n,
                           detail={"clusters": fit.groups, "group_means": means})


def cluster_robust_slope_test(y, x, clusters=None, confidence: float = 0.95, name: str | None = None) -> InferenceResult:
    """Slope of y on a numeric x with its test and interval (estimate = change in y for one more unit of x)."""
    x = _as_float(x, "x")
    fit = fit_ols(y, np.column_stack([np.ones(len(x)), x]), clusters)
    label = name or ("cluster-robust slope" if clusters is not None else "slope (rows treated as independent)")
    return coefficient_test(fit, 1, confidence=confidence, name=label)


def cluster_robust_interaction_test(y, a, b, clusters=None, confidence: float = 0.95, name: str | None = None) -> InferenceResult:
    """
    Model y ~ a + b + a*b for two yes/no factors. The estimate is the interaction: how much the effect of `a` differs between
    the two levels of `b` (a "difference of differences"). The test statistic is its t value.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    fit = fit_ols(y, np.column_stack([np.ones(len(a)), a, b, a * b]), clusters)
    label = name or ("cluster-robust interaction" if clusters is not None else "interaction (rows treated as independent)")
    return coefficient_test(fit, 3, confidence=confidence, name=label)


# ================================================================================================ wild cluster bootstrap
def wild_cluster_bootstrap_test(y, X, restrictions, clusters, values=None, n_boot: int = 9_999, seed: int = 20240518, chunk: int = 500,
                                confidence: float = 0.95, name: str | None = None) -> InferenceResult:
    """
    The wild cluster bootstrap-t test of the linear restrictions  R b = r  in the regression y = X b + error, with clusters.

    WHY: the ordinary cluster-robust test refers its statistic to a t (or F) distribution with G - 1 degrees of freedom. That is only
    accurate when the clusters are about equally big. Here one artist owns 96 of 944 songs, so the sandwich test is too eager (a test
    of a meaningless label rejects 7 to 10% of the time at the 5% level). The wild bootstrap does not rely on that distribution:

        1. fit the model WITH the restriction imposed (the restricted fit) and keep its residuals;
        2. make a new outcome  y* = restricted fit + v_g * residual,  where v_g is +1 or -1 by a coin flip, the SAME for every row of
           cluster g (so the relatedness inside a cluster is kept) and independent between clusters;
        3. refit the full model to y*, compute the same cluster-robust t (one restriction) or F (several) statistic;
        4. repeat n_boot times. The p-value is the share of repeats whose statistic is at least as extreme as the observed one,
           (1 + number as extreme) / (1 + n_boot).

    One restriction: the statistic is the signed t value, the estimate is R b with an interval  estimate +- c * se,  where c is the
    `confidence` quantile of the bootstrapped |t| (so the interval agrees with the p-value: it excludes `r` exactly when p < 1 - confidence).
    Several restrictions: the statistic is the Wald F value. `detail` also holds the ordinary sandwich p-value ("p_cr1") for comparison.
    """
    y = _as_float(y, "y")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    if len(y) != n:
        raise ValueError("y and X must have the same number of rows")
    codes, groups = _codes(clusters)
    if len(codes) != n:
        raise ValueError("clusters must have one entry per row")
    if groups < 2:
        raise ValueError("the wild cluster bootstrap needs at least two clusters")
    if n <= k or n_boot < 1:
        raise ValueError("there must be more rows than coefficients, and at least one bootstrap repeat")
    R = np.atleast_2d(np.asarray(restrictions, dtype=float))
    if R.shape[1] != k:
        raise ValueError("every restriction needs one number for each column of X")
    q = R.shape[0]
    r = np.zeros(q) if values is None else np.asarray(values, dtype=float).ravel()

    order = np.argsort(codes, kind="stable")                         # rows sorted by cluster, so that cluster sums are one reduceat away
    Xs, ys, cs = X[order], y[order], codes[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    xtx_inv = np.linalg.inv(Xs.T @ Xs)
    projector = xtx_inv @ Xs.T                                       # (k, n): coefficients of any outcome are projector @ outcome
    factor = groups / (groups - 1.0) * (n - 1.0) / (n - k)

    def statistic(coefficients, residuals):
        """The cluster-robust t (q = 1) or F (q > 1) statistic of R b = r for one or many fits. Returns (statistic, se of R b)."""
        coefficients, residuals = np.atleast_2d(coefficients), np.atleast_2d(residuals)
        scores = np.stack([np.add.reduceat(residuals * Xs[:, j][None, :], starts, axis=1) for j in range(k)], axis=2)   # (fits, G, k)
        meat = np.einsum("mgk,mgl->mkl", scores, scores)
        covariance = factor * (xtx_inv @ meat @ xtx_inv)
        gap = coefficients @ R.T - r                                 # (fits, q)
        middle = R @ covariance @ R.T                                # (fits, q, q)
        if q == 1:
            se = np.sqrt(middle[:, 0, 0])
            with np.errstate(divide="ignore", invalid="ignore"):
                return gap[:, 0] / se, se
        try:
            solved = np.linalg.solve(middle, gap[..., None])[..., 0]
        except np.linalg.LinAlgError:
            solved = (np.linalg.pinv(middle) @ gap[..., None])[..., 0]
        return np.einsum("mq,mq->m", gap, solved) / q, None

    coef = projector @ ys
    observed, observed_se = statistic(coef, ys - Xs @ coef)
    observed = float(observed[0])
    # the restricted fit: the least-squares coefficients forced to satisfy R b = r
    adjustment = xtx_inv @ R.T @ np.linalg.solve(R @ xtx_inv @ R.T, R @ coef - r)
    fitted_restricted = Xs @ (coef - adjustment)
    residual_restricted = ys - fitted_restricted

    rng = np.random.default_rng(seed)
    scale = abs(observed) if q == 1 else observed
    extreme = 0
    draws = []
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        signs = np.where(rng.random((m, groups)) < 0.5, -1.0, 1.0)             # a +1 / -1 coin flip for every cluster
        y_star = fitted_restricted[None, :] + signs[:, cs] * residual_restricted[None, :]
        coef_star = y_star @ projector.T
        stat_star, _ = statistic(coef_star, y_star - coef_star @ Xs.T)
        stat_star = np.abs(stat_star) if q == 1 else stat_star
        extreme += int(np.sum(~np.isfinite(stat_star) | (stat_star >= scale * (1 - 1e-12))))
        draws.append(stat_star[np.isfinite(stat_star)])
        done += m
    p_value = (1 + extreme) / (1 + n_boot)

    fit = fit_ols(y, X, clusters)                                    # the ordinary sandwich fit, for the comparison and the standard error
    detail = {"clusters": groups, "n_boot": n_boot, "q": q}
    label = name or "wild cluster bootstrap test"
    if q == 1:
        estimate = float(R[0] @ coef)
        se = float(observed_se[0])
        critical = float(np.quantile(np.concatenate(draws), confidence)) if draws else float("nan")
        detail.update({"se": se, "critical_value": critical,
                       "p_cr1": float(2 * stats.t.sf(abs(observed), fit.df))})
        return InferenceResult(label, observed, float(p_value), df=None, estimate=estimate, ci_low=estimate - critical * se,
                               ci_high=estimate + critical * se, n=n, detail=detail)
    detail["p_cr1"] = float(stats.f.sf(observed, q, fit.df))
    return InferenceResult(label, observed, float(p_value), df=None, n=n, detail=detail)


def wild_two_group_test(y, in_group, clusters, n_boot: int = 9_999, seed: int = 20240518, confidence: float = 0.95,
                        name: str | None = None) -> InferenceResult:
    """Difference between two groups, mean(y | group) - mean(y | not group), with the wild cluster bootstrap-t p-value and interval."""
    flag = np.asarray(in_group, dtype=float).ravel()
    return wild_cluster_bootstrap_test(y, np.column_stack([np.ones(len(flag)), flag]), [0, 1], clusters, n_boot=n_boot, seed=seed,
                                       confidence=confidence, name=name or "difference between two groups (wild cluster bootstrap-t)")


def wild_k_group_test(y, groups, clusters, n_boot: int = 9_999, seed: int = 20240518, name: str | None = None) -> InferenceResult:
    """Test that the means of k groups are all equal (a one-way ANOVA) with the wild cluster bootstrap-t p-value."""
    codes, k = _codes(groups)
    if k < 2:
        raise ValueError("there must be at least two groups")
    dummies = (codes[:, None] == np.arange(1, k)[None, :]).astype(float)
    restrictions = np.hstack([np.zeros((k - 1, 1)), np.eye(k - 1)])
    return wild_cluster_bootstrap_test(y, np.column_stack([np.ones(len(codes)), dummies]), restrictions, clusters, n_boot=n_boot,
                                       seed=seed, name=name or "test of equal group means (wild cluster bootstrap-t)")


def wild_slope_test(y, x, clusters, n_boot: int = 9_999, seed: int = 20240518, confidence: float = 0.95,
                    name: str | None = None) -> InferenceResult:
    """Slope of y on a number x (change in y per unit of x), with the wild cluster bootstrap-t p-value and interval."""
    x = _as_float(x, "x")
    return wild_cluster_bootstrap_test(y, np.column_stack([np.ones(len(x)), x]), [0, 1], clusters, n_boot=n_boot, seed=seed,
                                       confidence=confidence, name=name or "slope (wild cluster bootstrap-t)")


def wild_interaction_test(y, a, b, clusters, n_boot: int = 9_999, seed: int = 20240518, confidence: float = 0.95,
                          name: str | None = None) -> InferenceResult:
    """The interaction of two yes/no factors (difference of differences) with the wild cluster bootstrap-t p-value and interval."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return wild_cluster_bootstrap_test(y, np.column_stack([np.ones(len(a)), a, b, a * b]), [0, 0, 0, 1], clusters, n_boot=n_boot,
                                       seed=seed, confidence=confidence, name=name or "interaction (wild cluster bootstrap-t)")


def within_cluster_demean(values, clusters) -> np.ndarray:
    """
    Take every cluster's own average away from its rows (the "fixed effects" transformation). What is left says how a row differs
    from the OTHER rows of the same cluster. Works for one column (a vector) or several (a matrix, column by column). A cluster with a
    single row becomes 0.
    """
    codes, groups = _codes(clusters)
    array = np.asarray(values, dtype=float)
    if len(array) != len(codes):
        raise ValueError("clusters must have one entry per row")
    columns = array[:, None] if array.ndim == 1 else array
    sizes = np.bincount(codes, minlength=groups).astype(float)
    centred = np.empty_like(columns)
    for j in range(columns.shape[1]):
        means = np.bincount(codes, weights=columns[:, j], minlength=groups) / sizes
        centred[:, j] = columns[:, j] - means[codes]
    return centred[:, 0] if array.ndim == 1 else centred


def wild_fixed_effects_test(y, X, restrictions, clusters, values=None, n_boot: int = 9_999, seed: int = 20240518, chunk: int = 250,
                            confidence: float = 0.95, name: str | None = None) -> InferenceResult:
    """
    The wild cluster bootstrap-t test of R b = r when every cluster has its OWN intercept, so that only rows of the same cluster are
    compared (songs of one artist with each other, never with songs of another artist). `X` holds the columns to estimate and must
    NOT contain a constant: the cluster intercepts are taken out by `within_cluster_demean`.

    WHY: a difference between two groups can come from the clusters ("the artists who make long songs are also the artists who reach the
    top 10") and not from the groups themselves. If the difference is still there inside the clusters, it is not only the artist.

    Convention: like the usual "fixed effects with clustered errors" routine, the degrees-of-freedom factor does not count the removed
    intercepts, because each cluster's intercept is absorbed by that cluster. The bootstrap p-value does not depend on this factor
    (it multiplies the observed statistic and every bootstrapped one alike); only the standard error and the interval do.
    Clusters that never vary in X (one row, or all rows alike) take no part in the estimate.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return wild_cluster_bootstrap_test(within_cluster_demean(y, clusters), within_cluster_demean(X, clusters), restrictions, clusters, values=values,
                                       n_boot=n_boot, seed=seed, chunk=chunk, confidence=confidence,
                                       name=name or "within-cluster test (wild cluster bootstrap-t)")


# ================================================================================================ how clustered is a variable?
def intraclass_correlation(values, clusters) -> dict:
    """
    How strongly do the rows of one cluster resemble each other? The one-way analysis-of-variance estimator for clusters of
    different sizes. 0 = rows of a cluster are no more alike than rows of different clusters; 1 = identical within a cluster.
    A negative estimate (noise) is reported as it is.

    Returns {"icc", "mean_cluster_size" (the adjusted average n0), "design_effect", "effective_n", "clusters", "n"}. The design
    effect 1 + (n0 - 1) * icc says how many times larger the variance of an average is than it would be for independent rows;
    effective_n = n / design effect is the number of independent rows that the data are worth.
    """
    y = _as_float(values, "values")
    codes, groups = _codes(clusters)
    n = len(y)
    if groups < 2 or n <= groups:
        raise ValueError("need at least two clusters and more rows than clusters")
    sizes = np.bincount(codes, minlength=groups).astype(float)
    sums = np.bincount(codes, weights=y, minlength=groups)
    means = sums / sizes
    grand = y.mean()
    between = float((sizes * (means - grand) ** 2).sum() / (groups - 1))
    within = float(((y - means[codes]) ** 2).sum() / (n - groups))
    n0 = float((n - (sizes ** 2).sum() / n) / (groups - 1))
    denominator = between + (n0 - 1.0) * within
    icc = float((between - within) / denominator) if denominator > 0 else 0.0
    design_effect = float(max(1.0 + (n0 - 1.0) * icc, 1e-12))
    return {"icc": icc, "mean_cluster_size": n0, "design_effect": design_effect, "effective_n": n / design_effect,
            "clusters": groups, "n": n}


# ================================================================================================ resampling
def cluster_bootstrap_weights(clusters, n_boot: int = 10_000, seed: int = 20240518, chunk: int = 500):
    """
    The cluster (artist) bootstrap, as row weights. Drawing G clusters with replacement means that cluster c is used w_c times
    (w_c = 0, 1, 2 ...). Every row of cluster c therefore gets the weight w_c. A statistic of a replicate is simply the weighted
    version of the statistic (a weighted mean, a weighted proportion ...), so no data have to be copied.

    A generator: each step yields a matrix of shape (chunk, n_rows) with the weights of `chunk` replicates.
    """
    codes, groups = _codes(clusters)
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        draws = rng.integers(0, groups, size=(size, groups))
        flat = (draws + np.arange(size)[:, None] * groups).ravel()
        counts = np.bincount(flat, minlength=size * groups).reshape(size, groups)
        yield counts[:, codes].astype(float)
        done += size


def cluster_bootstrap(clusters, statistic, n_boot: int = 10_000, seed: int = 20240518, chunk: int = 500) -> np.ndarray:
    """
    Run a statistic on every cluster-bootstrap replicate. `statistic(weights)` receives a matrix of row weights with one replicate per
    row (shape (replicates, n_rows)) and must return one number per replicate, for example a weighted difference of two means.
    Returns all n_boot numbers.
    """
    pieces = []
    for weights in cluster_bootstrap_weights(clusters, n_boot, seed, chunk):
        values = np.asarray(statistic(weights), dtype=float)
        if values.shape != (weights.shape[0],):
            raise ValueError("statistic must return one number for every replicate (row of the weight matrix)")
        pieces.append(values)
    return np.concatenate(pieces)


def bootstrap_interval(replicates, confidence: float = 0.95) -> tuple[float, float]:
    """The percentile confidence interval of a set of bootstrap replicates."""
    values = _as_float(replicates, "replicates")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def bootstrap_p_value(replicates, null: float = 0.0) -> float:
    """
    A two-sided p-value from bootstrap replicates: twice the smaller share of replicates on either side of `null`, with one
    added to numerator and denominator so that it is never exactly zero. It is approximate; use it as a cross-check.
    """
    values = _as_float(replicates, "replicates")
    below = (np.sum(values <= null) + 1.0) / (len(values) + 1.0)
    above = (np.sum(values >= null) + 1.0) / (len(values) + 1.0)
    return float(min(1.0, 2.0 * min(below, above)))


def permutation_p_value(observed: float, null_values, alternative: str = "two-sided") -> float:
    """
    The share of shuffled results at least as extreme as the observed one, counted as (1 + number as extreme) / (1 + shuffles)
    so that the p-value is never zero (the observed arrangement is itself one of the possible shuffles).
    """
    null_values = _as_float(null_values, "null_values")
    if alternative == "two-sided":
        extreme = np.sum(np.abs(null_values) >= abs(observed) - 1e-12)
    elif alternative == "greater":
        extreme = np.sum(null_values >= observed - 1e-12)
    elif alternative == "less":
        extreme = np.sum(null_values <= observed + 1e-12)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    return float((1.0 + extreme) / (1.0 + len(null_values)))


def shuffled_group_differences(values, in_group, n_perm: int = 10_000, seed: int = 20240518, chunk: int = 1000) -> np.ndarray:
    """
    The permutation null of "mean in the group minus mean outside": shuffle which rows are in the group (keeping the group size)
    n_perm times and recompute the difference. Returns the n_perm differences.
    """
    y = _as_float(values, "values")
    flag = np.asarray(in_group, dtype=bool).ravel()
    size = int(flag.sum())
    if size == 0 or size == len(y):
        raise ValueError("both groups must have at least one row")
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    done = 0
    total = y.sum()
    while done < n_perm:
        step = min(chunk, n_perm - done)
        order = np.argsort(rng.random((step, len(y))), axis=1)[:, :size]       # the rows drawn into the group
        group_sum = y[order].sum(axis=1)
        out[done:done + step] = group_sum / size - (total - group_sum) / (len(y) - size)
        done += step
    return out


def exact_split_permutation_test(values, size_first: int, name: str = "exact permutation test of the difference between two parts") -> InferenceResult:
    """
    Split the values into a first part (the first `size_first` values) and the rest. The test statistic is mean(first) - mean(rest).
    Under "the order does not matter" every way of choosing which values form the first part is equally likely, so ALL
    C(n, size_first) ways are enumerated: an exact p-value, no random numbers. The statistic of every split is in detail["null"].
    """
    y = _as_float(values, "values")
    n = len(y)
    if not 0 < size_first < n:
        raise ValueError("size_first must be between 1 and n - 1")
    if math.comb(n, size_first) > 3_000_000:
        raise ValueError(f"{math.comb(n, size_first):,} splits are too many to enumerate; use shuffled_group_differences instead")
    observed = float(y[:size_first].mean() - y[size_first:].mean())
    total = y.sum()
    splits = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(n), size_first)), dtype=np.int32).reshape(-1, size_first)
    first_sum = y[splits].sum(axis=1)
    null = first_sum / size_first - (total - first_sum) / (n - size_first)
    p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return InferenceResult(name, observed, p_value, estimate=observed, n=n, detail={"splits": int(len(null)), "null": null})


# ================================================================================================ time series
def autocorrelation(values, lag: int = 1) -> float:
    """The correlation of a series with itself `lag` steps later (after removing the mean)."""
    y = _as_float(values, "values")
    if not 1 <= lag < len(y):
        raise ValueError("lag must be at least 1 and shorter than the series")
    centred = y - y.mean()
    return float((centred[:-lag] * centred[lag:]).sum() / (centred ** 2).sum())


def effective_sample_size_ar1(n: int, rho: float) -> float:
    """
    How many independent days a run of n correlated days is worth, if each day is correlated with the previous one by rho:
    n (1 - rho) / (1 + rho), never below 1 and never above n.
    """
    rho = float(np.clip(rho, -0.99, 0.99))
    return float(min(max(n * (1.0 - rho) / (1.0 + rho), 1.0), n))


def default_lags(n: int) -> int:
    """The Newey-West (1994) rule of thumb for the number of lags: floor(4 * (n / 100) ** (2 / 9))."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_cov(X, residuals, lags: int) -> np.ndarray:
    """
    The Newey-West (HAC) covariance matrix of least-squares coefficients: it allows the residuals of days up to `lags` apart to be
    correlated (weights 1, then falling linearly towards zero) and the spread to change over time.
    """
    X = np.asarray(X, dtype=float)
    u = np.asarray(residuals, dtype=float)
    n, k = X.shape
    scores = X * u[:, None]
    omega = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        omega += weight * (gamma + gamma.T)
    xtx_inv = np.linalg.inv(X.T @ X)
    return xtx_inv @ omega @ xtx_inv * (n / (n - k))


def hac_trend_test(values, lags: int | None = None, confidence: float = 0.95, name: str = "trend (least squares, Newey-West standard error)") -> InferenceResult:
    """
    A straight-line trend of a daily series against the day number 0, 1, 2 ... The estimate is the change per day; the standard
    error allows for correlation from day to day. The reference distribution is t with n - 2 degrees of freedom.
    """
    y = _as_float(values, "values")
    n = len(y)
    lags = default_lags(n) if lags is None else int(lags)
    X = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
    fit = fit_ols(y, X)
    cov = newey_west_cov(X, fit.residuals, lags)
    slope = float(fit.coef[1])
    se = float(np.sqrt(cov[1, 1]))
    t_value = slope / se
    df = n - 2
    margin = float(stats.t.ppf(0.5 + confidence / 2, df)) * se
    return InferenceResult(name, float(t_value), float(2 * stats.t.sf(abs(t_value), df)), df=df, estimate=slope,
                           ci_low=slope - margin, ci_high=slope + margin, n=n,
                           detail={"se": se, "lags": lags, "naive_se": float(fit.se[1]), "lag1_autocorrelation": autocorrelation(fit.residuals)})


def moving_block_indices(n: int, block: int, n_boot: int = 10_000, seed: int = 20240518) -> np.ndarray:
    """
    Row indices for the moving-block bootstrap of a series of length n: glue together randomly chosen runs of `block` consecutive
    days until n days are filled. Keeping the days of a block together keeps the correlation from day to day. Shape (n_boot, n).
    """
    if not 1 <= block <= n:
        raise ValueError("block must be between 1 and n")
    rng = np.random.default_rng(seed)
    blocks_needed = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, blocks_needed))
    indices = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, blocks_needed * block)
    return indices[:, :n]


def block_bootstrap_trend(values, block: int = 14, n_boot: int = 10_000, seed: int = 20240518) -> np.ndarray:
    """
    The residual block bootstrap of a straight-line trend. Fit the line, cut the residuals into runs of `block` days, glue random
    runs together into new residual series, add the fitted line back and refit. Keeping whole runs together keeps the correlation
    from day to day, so the spread of the n_boot slopes is an honest spread for the slope. Returns the n_boot slopes (change per
    day), centred on the observed slope.
    """
    y = _as_float(values, "values")
    n = len(y)
    time = np.arange(n, dtype=float)
    centred = time - time.mean()
    fit = fit_ols(y, np.column_stack([np.ones(n), time]))
    indices = moving_block_indices(n, block, n_boot, seed)
    return float(fit.coef[1]) + (fit.residuals[indices] @ centred) / float(centred @ centred)


def mann_kendall_trend(values, confidence: float = 0.95) -> InferenceResult:
    """
    The Mann-Kendall trend test: Kendall's tau between the day number and the value (does the series tend to go up or down,
    whatever the shape?). The estimate is the Theil-Sen slope (the median of all pairwise slopes) with its interval.
    It does not assume a straight line or a normal distribution, but it does assume independent days: use it next to a
    block bootstrap when the days are correlated.
    """
    y = _as_float(values, "values")
    time = np.arange(len(y), dtype=float)
    tau = stats.kendalltau(time, y)
    slope, _, low, high = stats.theilslopes(y, time, alpha=confidence)
    return InferenceResult("Mann-Kendall trend test", float(tau.statistic), float(tau.pvalue), estimate=float(slope), ci_low=float(low),
                           ci_high=float(high), n=len(y), detail={"tau": float(tau.statistic)})
