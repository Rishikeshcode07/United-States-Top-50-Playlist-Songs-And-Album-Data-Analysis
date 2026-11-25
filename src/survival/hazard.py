"""
src/survival/hazard.py  -  the day-by-day hazard model: which inputs go with a chart day being the last one?

WHAT THIS FILE IS
    A regression for the chance that a chart day is the LAST day of a stay, given that the song reached it. The table has one row for each
    chart day of each stay (`person_days` in lives.py) and an `event` column that is 1 on the last day of a finished stay. The model is the
    complementary log-log one,

        chance that the day is the last  =  1 - exp( - exp( baseline for this age  +  inputs x coefficients ) )

    which is the discrete-time version of the proportional-hazards (Cox) model: exp(coefficient) is a HAZARD RATIO, the number by which an
    input multiplies the hazard on every chart day (2.0 means that a chart day is twice as likely to be the last one; 0.5 that it is half as
    likely). The baseline is a separate level for each age window (day 1, day 2, day 3, days 4-7 ...), which is why no separate intercept
    is used: the age-window columns are the baseline.

        fit_cloglog        fit the model and give coefficients with artist-aware standard errors
        wald_test          test that several coefficients are all zero (or equal), with the artist-aware covariance
        age_windows        the age window of every chart day
        window_interactions  separate effects of an input in each age window, to test the "same effect at every age" assumption

WHY WE NEED IT
    The curves of the groups (album drop or single, debut at 1-10 or at 41-50) mix several things: songs of an album drop also debut
    higher, singles arrive alone, and so on. A regression lets us ask "holding the other inputs fixed" for each input in turn. The
    day-by-day form uses every chart day, handles the many ties of a daily chart without approximation, and lets the hazard of the
    baseline fall as the song ages without any assumption about its shape.

    The songs of one artist are not independent. The standard errors are therefore the cluster-robust ("sandwich") ones, with the artist as
    the cluster, and p-values and intervals use the t distribution with (artists - 1) degrees of freedom, as `fit_logit` does in Phase 6A.

WHAT HAPPENS WHEN IT RUNS
    Fisher scoring with step halving (the log-likelihood never goes down). The covariance is
        bread x [ sum over artists g of  s_g s_g' ] x bread  x  G/(G-1) x (n-1)/(n-k),     s_g = sum over the rows of g of  x (score of the row)
    where the bread is the inverse OBSERVED information, which is what statsmodels uses for its robust covariance. The tests compare the
    coefficients and both kinds of standard errors with statsmodels on fixed data.

EXPECTED OUTCOME
    On the real file: a debut at 31 to 40 goes with a hazard about four times that of a debut in the Top 10, and an album drop one
    standard deviation larger goes with a hazard about 1.5 times as high.

MOTIVE
    Say what goes with a short stay, one input at a time, and how sure we are given that songs of one artist behave alike.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

_ETA_LOW, _ETA_HIGH = -40.0, 8.0


@dataclass(frozen=True)
class CloglogFit:
    """A fitted day-by-day hazard model. `groups` = number of clusters (None without clusters), `df` = degrees of freedom of the t reference."""

    coef: np.ndarray
    cov: np.ndarray                  # the covariance used for tests and intervals (cluster-robust when clusters were given)
    model_cov: np.ndarray            # the textbook covariance (inverse expected information)
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
        crit = stats.norm.ppf(0.5 + confidence / 2) if self.df is None else stats.t.ppf(0.5 + confidence / 2, self.df)
        return self.coef - crit * self.se, self.coef + crit * self.se

    def hazard_ratios(self, confidence: float = 0.95):
        """exp(coefficient) with its interval: (hazard ratio, low, high)."""
        low, high = self.interval(confidence)
        return np.exp(self.coef), np.exp(low), np.exp(high)


def _terms(eta):
    eta = np.clip(eta, _ETA_LOW, _ETA_HIGH)
    mu = np.exp(eta)
    return mu, np.exp(-mu), -np.expm1(-mu)                    # mu = exp(eta), 1 - p, p


def _log_likelihood(y, eta):
    mu, _, p = _terms(eta)
    return float(np.sum(y * np.log(np.maximum(p, 1e-300)) - (1.0 - y) * mu))


def _score(y, eta):
    """d(log-likelihood of a row)/d eta and the expected information weight of the row."""
    mu, r, p = _terms(eta)
    q = mu * r / np.maximum(p, 1e-300)
    return y * q - (1.0 - y) * mu, mu * q


def _observed_weight(y, eta):
    mu, r, p = _terms(eta)
    positive = mu * r * (mu + np.expm1(-mu)) / np.maximum(p, 1e-300) ** 2          # minus the second derivative on a row with y = 1
    return y * positive + (1.0 - y) * mu


def fit_cloglog(y, X, clusters=None, max_iter: int = 100, tol: float = 1e-10) -> CloglogFit:
    """
    The complementary log-log regression of the 0/1 outcome `y` on the columns of `X` (the age-window columns are the baseline, so no extra
    intercept). `clusters` (one label for each row, the artist) switches on the cluster-robust covariance.
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

    start_level = np.log(-np.log(1.0 - np.clip(y.mean(), 1e-6, 1 - 1e-6)))
    beta = np.linalg.lstsq(X, np.full(n, start_level), rcond=None)[0]
    log_lik = _log_likelihood(y, X @ beta)
    converged, iterations = False, 0
    for iterations in range(1, max_iter + 1):
        score, weight = _score(y, X @ beta)
        gradient = X.T @ score
        information = X.T @ (X * weight[:, None])
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            break
        scale = 1.0
        while True:
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
    if np.max(np.abs(beta)) > 25:
        converged = False

    eta = X @ beta
    score, weight = _score(y, eta)
    model_cov = np.linalg.pinv(X.T @ (X * weight[:, None]))
    if clusters is None:
        return CloglogFit(beta, model_cov, model_cov, n, k, None, None, converged, iterations, log_lik)
    _, codes = np.unique(np.asarray(clusters), return_inverse=True)
    groups = int(codes.max()) + 1
    if groups < 2 or len(codes) != n:
        raise ValueError("clusters must have one label for each row and at least two different labels")
    bread = np.linalg.pinv(X.T @ (X * _observed_weight(y, eta)[:, None]))
    sums = np.zeros((groups, k))
    np.add.at(sums, codes, X * score[:, None])
    correction = groups / (groups - 1.0) * (n - 1.0) / (n - k)
    robust = correction * bread @ (sums.T @ sums) @ bread
    return CloglogFit(beta, robust, model_cov, n, k, groups, groups - 1, converged, iterations, log_lik)


def age_windows(ages, edges) -> np.ndarray:
    """The window (0, 1, 2 ...) of every age: window k holds the ages edges[k] < age <= edges[k+1]."""
    ages = np.asarray(ages)
    edges = np.asarray(edges)
    if ages.min() <= edges[0] or ages.max() > edges[-1]:
        raise ValueError("some ages are outside the windows")
    return np.searchsorted(edges, ages, side="left") - 1


def window_columns(ages, edges, prefix: str = "age") -> pd.DataFrame:
    """One 0/1 column for each age window: the baseline hazard of the model."""
    index = age_windows(ages, edges)
    names = [f"{prefix} {low + 1}" + ("+" if high >= 10_000 else ("" if high == low + 1 else f"-{high}")) for low, high in zip(edges[:-1], edges[1:])]
    return pd.DataFrame({name: (index == k).astype(float) for k, name in enumerate(names)})


def selection(names, chosen) -> np.ndarray:
    """The contrast matrix that picks the coefficients `chosen` (a list of column names), one row each."""
    names = list(names)
    rows = np.zeros((len(chosen), len(names)))
    for i, name in enumerate(chosen):
        rows[i, names.index(name)] = 1.0
    return rows


def wald_test(fit: CloglogFit, contrasts) -> dict:
    """
    Test R b = 0 for the contrast matrix `contrasts` (q rows). With clusters the statistic W / q is compared with an F distribution with
    (q, artists - 1) degrees of freedom; without clusters W is compared with chi-square. Returns stat, q, df and p.
    """
    R = np.atleast_2d(np.asarray(contrasts, dtype=float))
    b = R @ fit.coef
    middle = R @ fit.cov @ R.T
    w = float(b @ np.linalg.pinv(middle) @ b)
    q = int(np.linalg.matrix_rank(R))
    if fit.df is None:
        return {"stat": w, "q": q, "df": None, "p": float(stats.chi2.sf(w, q))}
    stat = w / q
    return {"stat": stat, "q": q, "df": fit.df, "p": float(stats.f.sf(stat, q, fit.df))}


def window_interactions(design: pd.DataFrame, columns, window_of_row, n_windows: int) -> pd.DataFrame:
    """
    A design in which every column of `columns` is replaced by one column for each window: "<column> @ 1", "<column> @ 2" ... (the
    column multiplied by 1 inside that window of age and by 0 outside it). The other columns stay as they are. In the fitted model the
    coefficients of "<column> @ k" are the effect of the input in window k of age.
    """
    window_of_row = np.asarray(window_of_row)
    out = design.drop(columns=list(columns)).copy()
    for column in columns:
        for k in range(n_windows):
            out[f"{column} @ {k + 1}"] = design[column].to_numpy() * (window_of_row == k)
    return out


def equal_effects_contrasts(names, columns, n_windows: int) -> np.ndarray:
    """The contrasts of "the input has the same effect in every window": b(window k) - b(window k+1) = 0, for each column and each k."""
    names = list(names)
    rows = []
    for column in columns:
        for k in range(1, n_windows):
            row = np.zeros(len(names))
            row[names.index(f"{column} @ {k}")] = 1.0
            row[names.index(f"{column} @ {k + 1}")] = -1.0
            rows.append(row)
    return np.array(rows)
