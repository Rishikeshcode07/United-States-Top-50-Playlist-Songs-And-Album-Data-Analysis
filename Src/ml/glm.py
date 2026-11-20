"""
src/ml/glm.py  -  logistic regression with honest standard errors: the "explain" side of the classification models.

WHAT THIS FILE IS
    `fit_logit` fits a logistic regression (the chance of "yes" as a function of several inputs) by Newton's method and returns the
    coefficients with their standard errors, p-values, odds ratios and confidence intervals. With `clusters` (the artist of every
    song) the standard errors are the cluster-robust "sandwich" ones, the same idea as `fit_ols` in src/inference/dependence.py.

WHY WE NEED IT
    The prediction models of Phase 6A answer "how well can we forecast?". A reader also asks "which inputs matter, and by how much,
    holding the others fixed?". An odds ratio answers that in one number: 2.0 means that the odds of "yes" double when the input
    goes up by one unit. scikit-learn does not give standard errors, and the usual ones would be too small here because the songs of
    one artist are not independent. So the doubt is calculated here, allowing for related songs.

WHAT HAPPENS WHEN IT RUNS
    Newton's method with step halving (the log-likelihood never goes down), usually 5 to 8 rounds. If the data separate perfectly (an
    input that decides the outcome by itself) the coefficients run off to infinity; the function then reports `converged=False` instead
    of printing nonsense. The covariance is
        (X'WX)^-1 [ sum over clusters g of  s_g s_g' ] (X'WX)^-1  *  G/(G-1) * (n-1)/(n-k),     s_g = sum over rows of g of x (y - p)
    p-values and intervals use the t distribution with G-1 degrees of freedom when clusters are given (cautious with few clusters), and
    the normal distribution otherwise. The tests compare the numbers with statsmodels on a fixed data set.

EXPECTED OUTCOME
    Coefficients equal to those of an ordinary (unpenalised) logistic regression; standard errors that grow when songs of one artist
    behave alike.

MOTIVE
    Say not only what goes with the outcome but how sure we are, given that the songs are not independent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import expit


@dataclass(frozen=True)
class LogitFit:
    """A fitted logistic regression. `groups` is the number of clusters (None without clusters); `df` the reference of t (None = normal)."""

    coef: np.ndarray
    cov: np.ndarray
    n: int
    k: int
    groups: int | None
    df: int | None
    converged: bool
    iterations: int
    log_likelihood: float

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def p_values(self) -> np.ndarray:
        z = self.coef / self.se
        return 2 * (stats.norm.sf(np.abs(z)) if self.df is None else stats.t.sf(np.abs(z), self.df))

    def interval(self, confidence: float = 0.95):
        """Confidence interval of every coefficient (low, high)."""
        crit = stats.norm.ppf(0.5 + confidence / 2) if self.df is None else stats.t.ppf(0.5 + confidence / 2, self.df)
        return self.coef - crit * self.se, self.coef + crit * self.se

    def odds_ratios(self, confidence: float = 0.95):
        """exp(coefficient) with its interval: (odds ratio, low, high)."""
        low, high = self.interval(confidence)
        return np.exp(self.coef), np.exp(low), np.exp(high)

    def predict_proba(self, X) -> np.ndarray:
        return expit(np.asarray(X, dtype=float) @ self.coef)


def _log_likelihood(y, eta):
    # y*log(p) + (1-y)*log(1-p) written with logaddexp so that large |eta| does not overflow
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def fit_logit(y, X, clusters=None, max_iter: int = 100, tol: float = 1e-10) -> LogitFit:
    """
    Logistic regression of the 0/1 outcome `y` on the columns of `X` (add a column of ones for the intercept yourself).
    `clusters` (one label per row) switches on the cluster-robust covariance. Raises ValueError for a y that is not 0/1.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    if len(y) != n:
        raise ValueError("y and X must have the same number of rows")
    if not np.isin(y, [0.0, 1.0]).all() or y.min() == y.max():
        raise ValueError("y must hold only 0 and 1, and both must occur")
    if not np.isfinite(X).all():
        raise ValueError("X must not contain NaN or infinity")
    if n <= k:
        raise ValueError("there must be more rows than coefficients")

    beta = np.zeros(k)
    log_lik = _log_likelihood(y, X @ beta)
    converged, iterations = False, 0
    for iterations in range(1, max_iter + 1):
        p = expit(X @ beta)
        weight = p * (1 - p)
        gradient = X.T @ (y - p)
        hessian = X.T @ (X * weight[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        scale = 1.0
        while True:                                       # halve the step until the log-likelihood does not go down
            candidate = beta + scale * step
            candidate_ll = _log_likelihood(y, X @ candidate)
            if candidate_ll >= log_lik - 1e-12 or scale < 1e-8:
                break
            scale /= 2
        change = abs(candidate_ll - log_lik)
        beta, log_lik = candidate, candidate_ll
        if change < tol and np.max(np.abs(scale * step)) < 1e-6:
            converged = True
            break
    if np.max(np.abs(beta)) > 25:                          # perfect separation: the coefficients drift to infinity
        converged = False

    p = expit(X @ beta)
    weight = p * (1 - p)
    bread = np.linalg.pinv(X.T @ (X * weight[:, None]))
    if clusters is None:
        return LogitFit(beta, bread, n, k, None, None, converged, iterations, log_lik)
    _, codes = np.unique(np.asarray(clusters), return_inverse=True)
    groups = int(codes.max()) + 1
    if groups < 2 or len(codes) != n:
        raise ValueError("clusters must have one label per row and at least two different labels")
    scores = X * (y - p)[:, None]
    sums = np.zeros((groups, k))
    np.add.at(sums, codes, scores)
    meat = sums.T @ sums
    correction = groups / (groups - 1.0) * (n - 1.0) / (n - k)
    return LogitFit(beta, correction * bread @ meat @ bread, n, k, groups, groups - 1, converged, iterations, log_lik)
