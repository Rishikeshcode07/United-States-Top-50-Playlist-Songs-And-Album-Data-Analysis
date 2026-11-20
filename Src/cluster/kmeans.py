"""
src/cluster/kmeans.py  -  k-means and the three tools that judge a grouping.

WHAT THIS FILE IS
    kmeans              k-means with a k-means++ start, several starts, and the groups numbered by size (the biggest group is 0)
    assign              the group of new rows: the nearest of the given centres
    silhouette          how well every row fits its own group compared with the next-nearest group (average, from -1 to 1)
    adjusted_rand       how much two groupings agree, corrected for chance (1 = the same grouping, about 0 = no agreement)
    match_labels        which label of one grouping is which label of another (best one-to-one match of the centres)

WHY WE NEED IT
    k-means needs a random start and the library versions do not all draw the same random numbers or use the same tie rules. Writing
    the few lines here (about 40) keeps the groups of the notebook identical on every setup, and the tests compare it with
    scikit-learn on data whose groups are clear.

WHAT HAPPENS WHEN IT RUNS
    `kmeans(X, k, n_init, seed)` starts n_init times from k-means++ centres, improves each start (every row to its nearest centre, every
    centre to the mean of its rows) until nothing moves, and keeps the start with the smallest total squared distance (the inertia).

EXPECTED OUTCOME
    The same labels for the same data and seed, on any machine.

MOTIVE
    A grouping that cannot be reproduced cannot be defended.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass
class KMeansResult:
    labels: np.ndarray          # group of every row, 0 = the biggest group
    centers: np.ndarray         # (k, n_columns) mean of every group
    inertia: float              # total squared distance of the rows to their centres
    n_iter: int                 # steps the best start needed


def squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """(rows, centres) matrix of squared distances."""
    d = (X * X).sum(axis=1)[:, None] - 2.0 * (X @ centers.T) + (centers * centers).sum(axis=1)[None, :]
    return np.maximum(d, 0.0)


def plus_plus_start(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++: the first centre at random, every next one drawn with a chance proportional to its squared distance to the nearest centre."""
    n = len(X)
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.integers(n)]
    nearest = squared_distances(X, centers[:1])[:, 0]
    for j in range(1, k):
        total = nearest.sum()
        pick = rng.integers(n) if total <= 0 else rng.choice(n, p=nearest / total)
        centers[j] = X[pick]
        nearest = np.minimum(nearest, squared_distances(X, centers[j:j + 1])[:, 0])
    return centers


def _improve(X: np.ndarray, centers: np.ndarray, max_iter: int):
    """Alternate 'every row to its nearest centre' and 'every centre to the mean of its rows' until no row changes group."""
    centers = centers.copy()
    labels = None
    for step in range(1, max_iter + 1):
        distances = squared_distances(X, centers)
        new = distances.argmin(axis=1)
        if labels is not None and np.array_equal(new, labels):
            return labels, centers, step - 1
        labels = new
        for j in range(len(centers)):
            members = labels == j
            if members.any():
                centers[j] = X[members].mean(axis=0)
            else:                                           # an empty group takes the row that is farthest from its own centre
                far = distances[np.arange(len(X)), labels].argmax()
                centers[j], labels[far] = X[far], j
    return labels, centers, max_iter


def kmeans(X: np.ndarray, k: int, n_init: int = 10, seed: int = 0, max_iter: int = 100) -> KMeansResult:
    """Best of n_init k-means runs (smallest inertia); the groups are numbered by size, the biggest first."""
    X = np.asarray(X, dtype=float)
    if not 1 <= k <= len(X):
        raise ValueError("k must be between 1 and the number of rows")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        labels, centers, steps = _improve(X, plus_plus_start(X, k, rng), max_iter)
        inertia = float(squared_distances(X, centers)[np.arange(len(X)), labels].sum())
        if best is None or inertia < best[0] - 1e-9:
            best = (inertia, labels, centers, steps)
    inertia, labels, centers, steps = best
    order = np.argsort(-np.bincount(labels, minlength=k), kind="stable")          # biggest group first
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return KMeansResult(labels=remap[labels], centers=centers[order], inertia=inertia, n_iter=steps)


def assign(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """The nearest centre of every row."""
    return squared_distances(np.asarray(X, dtype=float), centers).argmin(axis=1)


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Average silhouette: for every row (b - a) / max(a, b), a = mean distance to its own group, b = mean distance to the nearest other group."""
    labels = np.asarray(labels)
    groups = np.unique(labels)
    if len(groups) < 2:
        raise ValueError("the silhouette needs at least two groups")
    distances = cdist(X, X)
    sizes = np.array([(labels == g).sum() for g in groups])
    means = np.column_stack([distances[:, labels == g].sum(axis=1) for g in groups])          # total distance of every row to every group
    own = np.searchsorted(groups, labels)
    n = len(labels)
    a = means[np.arange(n), own] / np.maximum(sizes[own] - 1, 1)                            # the row itself adds a distance of 0
    others = means / sizes[None, :]
    others[np.arange(n), own] = np.inf
    b = others.min(axis=1)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-300)
    s[sizes[own] == 1] = 0.0                                                                 # a group of one row has no silhouette
    return float(s.mean())


def adjusted_rand(first: np.ndarray, second: np.ndarray) -> float:
    """Adjusted Rand index of two groupings of the same rows (the numbering of the groups does not matter)."""
    first, second = np.asarray(first), np.asarray(second)
    if len(first) != len(second):
        raise ValueError("the two groupings must cover the same rows")
    n = len(first)
    _, a = np.unique(first, return_inverse=True)
    _, b = np.unique(second, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    pairs = lambda x: (x * (x - 1) / 2.0).sum()
    together, rows, columns, total = pairs(table), pairs(table.sum(axis=1)), pairs(table.sum(axis=0)), n * (n - 1) / 2.0
    expected = rows * columns / total
    maximum = 0.5 * (rows + columns)
    if maximum == expected:                                          # both groupings are a single group or all singletons: they agree
        return 1.0
    return float((together - expected) / (maximum - expected))


def match_labels(reference_centers: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """mapping[j] = the label of the reference grouping that group j of `centers` matches best (one-to-one, by distance of the centres)."""
    cost = cdist(centers, reference_centers)
    rows, columns = linear_sum_assignment(cost)
    mapping = np.empty(len(centers), dtype=int)
    mapping[rows] = columns
    return mapping
