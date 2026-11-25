"""
src/survival/cox.py  -  a Cox model that ranks songs by how soon they will leave, and the C-index that says how well it ranks them.

WHAT THIS FILE IS
    Three tools for the question "if we only know what was known on the arrival day, can we tell which of two songs leaves first?"

        fit_cox                 the Cox proportional-hazards model: one number (a risk score) for each song, higher = leaves sooner
        CoxScorer               fit_cox plus the preparation of the inputs (logs, filling gaps, one scale), so that it can be applied
                                to songs it has not seen
        cross_validated_risk    the risk score of every song from models that never saw the song's artist (artist-grouped folds)
        concordance             the C-index: among pairs of songs where the order of leaving is known, the share in which the song with
                                the higher risk score left first (0.5 = a coin flip, 1 = always right)
        concordance_pairs, weighted_concordance   the same for many bootstrap draws at once

WHY WE NEED IT
    A hazard ratio says "an album drop goes with a higher hazard". It does not say how much of the difference between songs that is. The
    C-index does: it is the chance that, for two songs picked at random, the one the model rates as riskier is the one that leaves first.
    It cannot be inflated by a lucky threshold, it uses the censored stays too, and it needs no cut-off. As in Phase 6A, every score is
    made by a model that never saw the artist (the folds are made by artist), because otherwise the model could recognise artists.

WHAT HAPPENS WHEN IT RUNS
    `fit_cox` maximises the Efron partial likelihood (the standard way to handle ties, of which a daily chart has many: 155 songs leave
    on their first chart day) with a ridge penalty of half the sum of squared coefficients times `ridge`, by Newton's method with step
    halving. The inputs are put on one scale first, so that the penalty treats them alike. Pairs of songs are "comparable" when the
    first song's exit was seen and it left before the other song did (or on the same day as a song that was still there): two songs that
    left on the same day are not compared, exactly as in the lifelines package.

EXPECTED OUTCOME
    On the real file: a C-index of about 0.55 for what is known about the release, and about 0.64 when the debut day (place, popularity
    and the size of the album drop) is added.

MOTIVE
    Put a number on "how much of who stays longer can be read from the arrival day", with honest cross-validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from src.ml.validation import repeated_grouped_folds


@dataclass(frozen=True)
class CoxFit:
    coef: np.ndarray
    log_likelihood: float          # the Efron partial log-likelihood at the solution (without the penalty)
    penalised: float               # the penalised value that was maximised
    n_events: int
    iterations: int
    converged: bool

    def risk(self, X) -> np.ndarray:
        """The linear predictor of every row: a higher number leaves sooner."""
        return np.asarray(X, dtype=float) @ self.coef


def _efron(beta, ordered, ridge, want_hessian=True):
    """Penalised Efron log-likelihood, gradient and Hessian for inputs that are already sorted by time."""
    t, e, X, times, starts, stops = ordered
    p = X.shape[1]
    eta = np.clip(X @ beta, -50.0, 50.0)
    theta = np.exp(eta)
    risk0 = np.cumsum(theta[::-1])[::-1]
    weighted = X * theta[:, None]
    risk1 = np.cumsum(weighted[::-1], axis=0)[::-1]
    if want_hessian:
        outer = np.einsum("ni,nj->nij", weighted, X)
        risk2 = np.cumsum(outer[::-1], axis=0)[::-1]
    log_lik = float((X[e] @ beta).sum())
    gradient = X[e].sum(axis=0).astype(float)
    hessian = np.zeros((p, p))
    for start, stop in zip(starts, stops):
        tied = np.arange(start, stop)[e[start:stop]]
        d = len(tied)
        s0, s1 = risk0[start], risk1[start]
        t0, t1 = theta[tied].sum(), weighted[tied].sum(axis=0)
        fraction = np.arange(d) / d
        phi = s0 - fraction * t0                                         # one value for each tied exit
        v = s1[None, :] - fraction[:, None] * t1[None, :]
        log_lik -= float(np.log(phi).sum())
        gradient -= (v / phi[:, None]).sum(axis=0)
        if want_hessian:
            s2 = risk2[start]
            t2 = outer[tied].sum(axis=0)
            hessian -= s2 * (1.0 / phi).sum() - t2 * (fraction / phi).sum() - (v / phi[:, None]).T @ (v / phi[:, None])
    penalised = log_lik - 0.5 * ridge * float(beta @ beta)
    return penalised, log_lik, gradient - ridge * beta, hessian - ridge * np.eye(p)


def fit_cox(time, event, X, ridge: float = 1.0, max_iter: int = 60, tol: float = 1e-9) -> CoxFit:
    """The ridge-penalised Cox model with Efron's handling of ties. `time` in whole chart days, `event` True when the exit was seen."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(bool)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if len(time) != len(event) or len(time) != len(X):
        raise ValueError("time, event and X must have the same number of rows")
    if not np.isfinite(X).all():
        raise ValueError("X must not contain NaN or infinity")
    if not event.any():
        raise ValueError("there is no exit in the data")
    if ridge < 0:
        raise ValueError("ridge cannot be negative")
    order = np.argsort(time, kind="mergesort")
    t, e, Xs = time[order], event[order], X[order]
    times = np.unique(t[e])
    ordered = (t, e, Xs, times, np.searchsorted(t, times, side="left"), np.searchsorted(t, times, side="right"))
    beta = np.zeros(X.shape[1])
    value, log_lik, gradient, hessian = _efron(beta, ordered, ridge)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        try:
            step = np.linalg.solve(-hessian, gradient)
        except np.linalg.LinAlgError:
            break
        scale = 1.0
        while True:
            candidate = beta + scale * step
            new_value = _efron(candidate, ordered, ridge, want_hessian=False)[0]
            if new_value >= value - 1e-12 or scale < 1e-8:
                break
            scale /= 2
        change = abs(new_value - value)
        beta = candidate
        value, log_lik, gradient, hessian = _efron(beta, ordered, ridge)
        if change < tol and np.max(np.abs(scale * step)) < 1e-7:
            converged = True
            break
    return CoxFit(beta, log_lik, value, int(event.sum()), iterations, converged)


class CoxScorer:
    """
    A Cox model that can score songs it has not seen. `columns` are the inputs; the columns in `log_columns` are put on the log(1 + x)
    scale; a gap is filled with the median of the training songs and, if there was a gap in the training songs, flagged by an extra
    0/1 column; every column is put on one scale (mean 0, standard deviation 1 of the training songs) so that the ridge penalty treats
    the inputs alike.
    """

    def __init__(self, columns, log_columns=(), ridge: float = 1.0):
        self.columns = list(columns)
        self.log_columns = set(log_columns)
        self.ridge = ridge
        self.fit_: CoxFit | None = None

    def _matrix(self, frame, fitting: bool) -> np.ndarray:
        values = frame[self.columns].astype(float).copy()
        for column in self.columns:
            if column in self.log_columns:
                values[column] = np.log1p(values[column])
        if fitting:
            self.medians_ = values.median()
            self.flagged_ = [c for c in self.columns if values[c].isna().any()]
        flags = values[self.flagged_].isna().astype(float).add_suffix(" (missing)")
        values = values.fillna(self.medians_)
        values = pd.concat([values, flags], axis=1)
        if fitting:
            self.mean_ = values.mean()
            self.scale_ = values.std(ddof=0).replace(0.0, 1.0)
        return ((values - self.mean_) / self.scale_).to_numpy()

    def fit(self, frame, time, event):
        self.fit_ = fit_cox(time, event, self._matrix(frame, True), ridge=self.ridge)
        return self

    def score(self, frame) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("fit the scorer first")
        return self.fit_.risk(self._matrix(frame, False))


def cross_validated_risk(frame, time, event, groups, columns, log_columns=(), ridge: float = 1.0, n_splits: int = 5, n_repeats: int = 3, seed: int = 0):
    """
    The risk score of every song from a model that never saw the song's artist: for each of `n_repeats` different artist-grouped
    splits and each of its `n_splits` parts, a model is trained on the other parts and scores the held-out part. Returns an array with one
    row for each repeat and one column for each song. Also returns nothing else: use `concordance` on each row.
    """
    time = np.asarray(time)
    event = np.asarray(event).astype(bool)
    out = np.full((n_repeats, len(time)), np.nan)
    for r, folds in enumerate(repeated_grouped_folds(groups, n_splits=n_splits, n_repeats=n_repeats, seed=seed)):
        for fold in np.unique(folds):
            test = np.flatnonzero(folds == fold)
            train = np.flatnonzero(folds != fold)
            scorer = CoxScorer(columns, log_columns, ridge).fit(frame.iloc[train], time[train], event[train])
            out[r, test] = scorer.score(frame.iloc[test])
    return out


def concordance_pairs(time, event):
    """The comparable pairs (i, j): the exit of song i was seen and i left before j did, or on the same day as a song j that was still there."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(bool)
    earlier = time[:, None] < time[None, :]
    same_day = (time[:, None] == time[None, :]) & ~event[None, :]
    i, j = np.nonzero(event[:, None] & (earlier | same_day))
    return i, j


def concordance(time, event, risk) -> float:
    """The C-index: of all comparable pairs, the share in which the song that left first has the higher risk score (a tie in risk counts 1/2)."""
    i, j = concordance_pairs(time, event)
    if len(i) == 0:
        raise ValueError("there is no comparable pair")
    risk = np.asarray(risk, dtype=float)
    return float(((risk[i] > risk[j]) + 0.5 * (risk[i] == risk[j])).mean())


def weighted_concordance(pairs, risk, weights) -> np.ndarray:
    """
    The C-index for bootstrap draws: a pair counts w_i * w_j times. `weights` is one vector or a matrix with one row for each draw; the answer
    has one value for each row (NaN when a draw holds no comparable pair).
    """
    i, j = pairs
    risk = np.asarray(risk, dtype=float)
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    n = len(risk)
    agree = ((risk[i] > risk[j]) + 0.5 * (risk[i] == risk[j])).astype(float)
    numerator = sparse.csr_matrix((agree, (i, j)), shape=(n, n))
    denominator = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    top = np.asarray(((numerator.T @ w.T).T * w).sum(axis=1)).ravel()
    bottom = np.asarray(((denominator.T @ w.T).T * w).sum(axis=1)).ravel()
    out = np.divide(top, bottom, out=np.full(len(top), np.nan), where=bottom > 0)
    return out[0] if np.ndim(weights) == 1 else out
