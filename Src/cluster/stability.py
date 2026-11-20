"""
src/cluster/stability.py  -  how firm is a grouping?  Re-draw whole artists, and hold whole artists out.

WHAT THIS FILE IS
    artist_folds          deal the artists into folds of about the same number of rows (an artist is never split)
    bootstrap_stability   re-fit the grouping on artists drawn again with replacement, many times, and compare every re-fit with the original
    heldout_assignment    fit on some artists, give the rows of the other artists to the nearest centre, and compare with the original

WHY WE NEED IT
    The songs of one artist are alike (one album drop is dozens of songs that arrive together), so a grouping that looks firm when the
    songs are re-drawn one by one can hang on two or three artists. Here the unit that is re-drawn or held out is the ARTIST.
    Three numbers come out of the re-draws:
        the adjusted Rand index between the original grouping and a re-fit (1 = the same grouping),
        the Jaccard score of every original group (1 = a re-fit finds exactly the same rows as a group; below 0.6 = the group dissolves),
        for every row, the share of re-fits that put it in the group it started in (low = a row on a border).

WHAT HAPPENS WHEN IT RUNS
    Every re-fit gives ALL original rows to its nearest new centre, so the original and the re-fit can be compared on the same rows. The
    groups of a re-fit are matched to the original groups one-to-one by the distance of their centres.

EXPECTED OUTCOME
    Numbers between 0 and 1 (the Rand index can be slightly below 0 for a grouping that is worse than chance).

MOTIVE
    Do not describe a pattern that exists only in one draw of the artists.
"""
from __future__ import annotations

import numpy as np

from .kmeans import adjusted_rand, assign, kmeans, match_labels


def artist_folds(artists: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """fold[i] for every row: the artists are dealt, biggest first, into the fold that has the fewest rows so far (ties broken at random)."""
    artists = np.asarray(artists)
    names, inverse, counts = np.unique(artists, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.permutation(len(names)), -counts))              # biggest artists first, equal sizes in random order
    fold_of_artist = np.empty(len(names), dtype=int)
    load = np.zeros(n_folds)
    for a in order:
        fold = int(np.argmin(load))
        fold_of_artist[a] = fold
        load[fold] += counts[a]
    return fold_of_artist[inverse]


def _draw_rows(rows_of_artist: list, n_artists: int, rng: np.random.Generator) -> np.ndarray:
    drawn = rng.integers(0, n_artists, n_artists)                               # as many artists as there are, drawn with replacement
    return np.concatenate([rows_of_artist[a] for a in drawn])


def bootstrap_stability(X: np.ndarray, artists: np.ndarray, labels: np.ndarray, centers: np.ndarray, n_boot: int = 200, seed: int = 0,
                        n_init: int = 3) -> dict:
    """
    Re-fit k-means (same k) on artists drawn again with replacement n_boot times.
    Returns {'ari': (n_boot,), 'jaccard': (n_boot, k), 'agree': (rows,)} - see the header of this file.
    """
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    k = len(centers)
    names, inverse = np.unique(np.asarray(artists), return_inverse=True)
    rows_of_artist = [np.flatnonzero(inverse == a) for a in range(len(names))]
    rng = np.random.default_rng(seed)
    ari, jaccard, agree = np.empty(n_boot), np.empty((n_boot, k)), np.zeros(len(X))
    sizes = np.bincount(labels, minlength=k)
    for b in range(n_boot):
        rows = _draw_rows(rows_of_artist, len(names), rng)
        fit = kmeans(X[rows], k, n_init=n_init, seed=int(rng.integers(2 ** 31)))
        new = assign(X, fit.centers)                                             # every original row goes to its nearest new centre
        ari[b] = adjusted_rand(labels, new)
        table = np.zeros((k, k))
        np.add.at(table, (labels, new), 1)
        union = sizes[:, None] + table.sum(axis=0)[None, :] - table
        jaccard[b] = (table / np.maximum(union, 1)).max(axis=1)                  # best match of every original group among the new groups
        agree += match_labels(centers, fit.centers)[new] == labels
    return {"ari": ari, "jaccard": jaccard, "agree": agree / n_boot}


def heldout_assignment(X: np.ndarray, artists: np.ndarray, labels: np.ndarray, centers: np.ndarray, n_folds: int = 5, n_repeats: int = 3,
                       seed: int = 0, n_init: int = 5) -> dict:
    """
    Fit on all folds but one, give the rows of the held-out artists to the nearest centre, match the groups to the original ones.
    Returns {'ari': (n_repeats,), 'same': (n_repeats,) share of rows that land in their original group, 'recall': (n_repeats, k) the same per group}.
    """
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    k = len(centers)
    ari, same, recall = np.empty(n_repeats), np.empty(n_repeats), np.empty((n_repeats, k))
    for r in range(n_repeats):
        folds = artist_folds(artists, n_folds, seed + r)
        predicted = np.empty(len(X), dtype=int)
        for f in range(n_folds):
            test = folds == f
            fit = kmeans(X[~test], k, n_init=n_init, seed=seed + 100 * r + f)
            predicted[test] = match_labels(centers, fit.centers)[assign(X[test], fit.centers)]
        ari[r] = adjusted_rand(labels, predicted)
        same[r] = float((predicted == labels).mean())
        recall[r] = [float((predicted[labels == g] == g).mean()) for g in range(k)]
    return {"ari": ari, "same": same, "recall": recall}
