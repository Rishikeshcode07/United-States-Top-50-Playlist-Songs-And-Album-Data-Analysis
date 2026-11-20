"""
src/survival/estimators.py  -  the survival curve, the hazard and the log-rank test, with weights so that the artist bootstrap is cheap.

WHAT THIS FILE IS
    The descriptive tools of Phase 6B. Time is counted in whole CHART DAYS (1 on the arrival day), a stay lasts `t` chart days and either
    ended after them (`observed` = True) or may still be going (False, "censored").

        survival_grid       S(a): the share of songs whose stay is longer than a chart days (Kaplan-Meier), for a list of ages
        window_hazard       the chance that a chart day is the last one, for age windows (day 1, day 2, day 3, days 4-7 ...)
        life_table          the same, as a table with one row for each age window
        median_stay         the first age at which half of the songs have gone
        median_further_stay the median further stay of a song that has already spent `age` chart days on the chart
        restricted_mean     the average number of chart days used out of the first `horizon` (the area under the curve)
        logrank_k           the log-rank test of k groups, in the usual form and in the artist-aware form

WHY WE NEED IT
    Songs of one artist are not independent: they arrive together, share a fan base and leave together. A band around a survival
    curve that treats 894 songs as 894 independent draws is too narrow. The remedy used throughout this phase is the artist
    bootstrap: draw whole artists with replacement, many times, and see how much the answer moves. Every function here therefore takes
    an optional `weights` argument. A bootstrap draw is a weight for each song (how many times the song's artist was drawn), and a
    matrix of weights (one row per draw) gives the answers of all draws at once, without copying any data.

    The log-rank test has the same problem. `logrank_k(..., clusters=artist)` replaces the textbook variance by one that adds up the
    evidence of each artist before squaring it, so that a group of songs by a single artist counts as the single piece of evidence it is.

WHAT HAPPENS WHEN IT RUNS
    Durations must be whole numbers of 1 or more. All functions work on counts per chart day, so a run on 894 songs, even with a
    thousand weight vectors, takes a fraction of a second. The Kaplan-Meier estimate is the product over chart days a of
    (1 - exits on day a / songs still on the chart on day a).

EXPECTED OUTCOME
    On the real file: S(3) = 63%, S(7) = 43%, S(30) = 19%, a median stay of 6 chart days, and a hazard of 17% on the first chart day
    that falls below 1% from the 121st chart day on.

MOTIVE
    Describe how long songs stay, and say how sure we are, without pretending that songs are independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _check(durations, observed):
    t = np.asarray(durations)
    e = np.asarray(observed).astype(bool)
    if t.ndim != 1 or t.shape != e.shape:
        raise ValueError("durations and observed must be one-dimensional and of the same length")
    if t.size == 0:
        raise ValueError("there are no songs")
    if not np.all(np.equal(np.mod(t, 1), 0)) or t.min() < 1:
        raise ValueError("durations must be whole numbers of chart days, 1 or more")
    return t.astype(int), e


def _counts(t, e, weights):
    """Weighted exits and songs at risk on every chart day 1 ... max(t), one row for each weight vector."""
    n_days = int(t.max())
    w = np.atleast_2d(np.asarray(weights, dtype=float)) if weights is not None else np.ones((1, len(t)))
    if w.shape[1] != len(t):
        raise ValueError("weights must have one value for each song")
    exits = np.zeros((w.shape[0], n_days + 1))
    ends = np.zeros((w.shape[0], n_days + 1))
    for column in np.unique(t):                                   # at most a few hundred distinct durations
        in_day = t == column
        ends[:, column] = w[:, in_day].sum(axis=1)
        exits[:, column] = w[:, in_day & e].sum(axis=1)
    at_risk = np.cumsum(ends[:, ::-1], axis=1)[:, ::-1]           # songs with a stay of `day` or more
    return exits[:, 1:], at_risk[:, 1:], w.shape[0]


def survival_grid(durations, observed, grid, weights=None) -> np.ndarray:
    """
    S(a) for every age a in `grid` (0 allowed: S(0) = 1; an age beyond the longest stay repeats the last value). `weights` can be one vector
    (the answer is a 1-d array) or a matrix with one row for each bootstrap draw (the answer has one row for each draw).
    """
    t, e = _check(durations, observed)
    exits, at_risk, rows = _counts(t, e, weights)
    hazard = np.divide(exits, at_risk, out=np.zeros_like(exits), where=at_risk > 0)
    survival = np.concatenate([np.ones((rows, 1)), np.cumprod(1.0 - hazard, axis=1)], axis=1)       # column a = S(a)
    grid = np.asarray(grid, dtype=int)
    if grid.min() < 0:
        raise ValueError("ages cannot be negative")
    out = survival[:, np.minimum(grid, survival.shape[1] - 1)]
    return out[0] if (weights is None or np.ndim(weights) == 1) else out


def window_hazard(durations, observed, edges, weights=None):
    """
    For the windows (edges[k], edges[k+1]] of chart days: exits in the window, chart days spent in the window, and their ratio, the
    chance that a chart day in the window is the last one. Returns three arrays (one row for each weight vector when `weights` is a matrix).
    """
    t, e = _check(durations, observed)
    edges = np.asarray(edges, dtype=int)
    if np.any(np.diff(edges) <= 0) or edges[0] != 0:
        raise ValueError("edges must start at 0 and increase")
    w = np.atleast_2d(np.asarray(weights, dtype=float)) if weights is not None else np.ones((1, len(t)))
    low, high = edges[:-1], edges[1:]
    in_window = (t[:, None] > low[None, :]) & e[:, None]
    exits = (w @ (in_window & (t[:, None] <= high[None, :])).astype(float))
    days = w @ np.clip(np.minimum(t[:, None], high[None, :]) - low[None, :], 0, None).astype(float)
    hazard = np.divide(exits, days, out=np.full_like(exits, np.nan), where=days > 0)
    if weights is None or np.ndim(weights) == 1:
        return exits[0], days[0], hazard[0]
    return exits, days, hazard


def window_labels(edges) -> list:
    """'1', '2', '3', '4-7', '8-14' ... and '121+' for the last, open window."""
    labels = []
    for low, high in zip(edges[:-1], edges[1:]):
        first = low + 1
        labels.append(f"{first}+" if high >= 10_000 else (str(first) if first == high else f"{first}-{high}"))
    return labels


def life_table(durations, observed, edges) -> pd.DataFrame:
    """One row for each age window: songs that reach the window, exits, chart days, hazard and survival at the end of the window."""
    t, e = _check(durations, observed)
    exits, days, hazard = window_hazard(t, e, edges)
    edges = np.asarray(edges, dtype=int)
    ends = np.minimum(edges[1:], int(t.max()))
    survival = survival_grid(t, e, ends)
    return pd.DataFrame({"window": window_labels(edges), "from_day": edges[:-1] + 1, "to_day": edges[1:],
                         "songs_reaching": [int((t > low).sum()) for low in edges[:-1]], "exits": exits.astype(int), "chart_days": days.astype(int),
                         "hazard": hazard, "survival_at_end": survival})


def median_stay(survival, ages=None):
    """The first age at which S is at or below one half (NaN when the curve never gets there). `survival` is S(0), S(1), ...; 2-d: one row each."""
    s = np.atleast_2d(np.asarray(survival, dtype=float))
    ages = np.arange(s.shape[1]) if ages is None else np.asarray(ages)
    reached = s <= 0.5
    first = np.argmax(reached, axis=1)
    out = np.where(reached.any(axis=1), ages[first], np.nan)
    return float(out[0]) if np.ndim(survival) == 1 else out


def median_further_stay(survival, age: int):
    """Median further stay of a song that has already spent `age` chart days: the first u with S(age + u) <= S(age) / 2 (NaN if never)."""
    s = np.atleast_2d(np.asarray(survival, dtype=float))
    if age >= s.shape[1]:
        raise ValueError("the age is beyond the end of the curve")
    rest = s[:, age:] / s[:, [age]]
    reached = rest <= 0.5
    out = np.where(reached.any(axis=1), np.argmax(reached, axis=1), np.nan)
    return float(out[0]) if np.ndim(survival) == 1 else out


def restricted_mean(survival, horizon: int):
    """Chart days used out of the first `horizon`: S(0) + S(1) + ... + S(horizon - 1). `survival` as in median_stay."""
    s = np.asarray(survival, dtype=float)
    if s.shape[-1] < horizon:
        raise ValueError("the curve is shorter than the horizon")
    return s[..., :horizon].sum(axis=-1)


def logrank_k(durations, observed, groups, clusters=None) -> dict:
    """
    The log-rank test that all groups leave the chart at the same rate on every chart day. Returns a dict with
        observed, expected   exits seen and exits expected in each group if all groups were alike
        chi2, df, p          the usual test (exits of different songs treated as independent)
        chi2_artist, p_artist  the artist-aware test, when `clusters` (one label for each song) is given: the evidence of each artist is added
                              up before it is squared. This is the score test of a Cox model with the Lin-Wei variance.
    """
    t, e = _check(durations, observed)
    labels, g = np.unique(np.asarray(groups), return_inverse=True)
    k = len(labels)
    if k < 2:
        raise ValueError("at least two groups are needed")
    n_days = int(t.max())
    at_risk = np.zeros((k, n_days + 1))
    exits = np.zeros((k, n_days + 1))
    for group in range(k):
        mask = g == group
        at_risk[group] = np.cumsum(np.bincount(t[mask], minlength=n_days + 1)[::-1])[::-1]
        exits[group] = np.bincount(t[mask & e], minlength=n_days + 1)
    at_risk, exits = at_risk[:, 1:], exits[:, 1:]
    total_risk, total_exits = at_risk.sum(axis=0), exits.sum(axis=0)
    h = np.divide(total_exits, total_risk, out=np.zeros(n_days), where=total_risk > 0)             # pooled hazard of each chart day
    expected = (at_risk * h).sum(axis=1)
    observed_count = exits.sum(axis=1)
    score = (observed_count - expected)[:-1]                                                        # the last group is implied by the others
    share = np.divide(at_risk, total_risk, out=np.zeros_like(at_risk), where=total_risk > 0)
    factor = np.divide(total_exits * (total_risk - total_exits), total_risk - 1, out=np.zeros(n_days), where=total_risk > 1)
    covariance = np.zeros((k - 1, k - 1))
    for a in range(k - 1):
        for b in range(k - 1):
            covariance[a, b] = (factor * share[a] * ((a == b) - share[b])).sum()
    chi2 = float(score @ np.linalg.pinv(covariance) @ score)
    result = {"groups": list(labels), "observed": observed_count, "expected": expected, "df": k - 1,
              "chi2": chi2, "p": float(stats.chi2.sf(chi2, k - 1))}
    if clusters is not None:
        _, codes = np.unique(np.asarray(clusters), return_inverse=True)
        if len(codes) != len(t):
            raise ValueError("clusters must have one label for each song")
        z = (g[:, None] == np.arange(k - 1)[None, :]).astype(float)                                 # group indicators of each song
        zbar = share[: k - 1].T                                                                      # share of each group among the songs at risk, by day
        cum_h = np.concatenate([[0.0], np.cumsum(h)])
        cum_hz = np.vstack([np.zeros(k - 1), np.cumsum(h[:, None] * zbar, axis=0)])
        # the contribution of every song to the score: its own exit, minus what the pooled hazard expected of it
        own = np.where(e[:, None], z - zbar[t - 1], 0.0)
        contribution = own - z * cum_h[t][:, None] + cum_hz[t]
        per_cluster = np.zeros((codes.max() + 1, k - 1))
        np.add.at(per_cluster, codes, contribution)
        n_clusters = per_cluster.shape[0]
        meat = per_cluster.T @ per_cluster * (n_clusters / max(n_clusters - 1, 1))
        chi2_artist = float(score @ np.linalg.pinv(meat) @ score)
        result.update({"chi2_artist": chi2_artist, "p_artist": float(stats.chi2.sf(chi2_artist, k - 1)), "n_clusters": int(n_clusters)})
    return result
