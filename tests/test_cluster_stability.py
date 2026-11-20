"""
tests/test_cluster_stability.py  -  checks of the artist folds, the artist bootstrap and the held-out-artist assignment (src/cluster/stability.py).

WHAT THIS FILE IS
    Folds that never split an artist and carry about the same number of rows; a bootstrap and a held-out run on data with clear groups (everything
    close to 1) and on pure noise (clearly lower); the shapes and ranges of the outputs; the same result for the same seed; and a data set in
    which ONE artist makes a whole group: re-drawing artists must lose that group in about a third of the draws, while re-drawing rows one by one
    would not notice.

WHY WE NEED IT
    The decision how many groups the notebook reports rests on these numbers.

EXPECTED OUTCOME
    Clear groups: scores above 0.95. Noise: the weakest group clearly lower. The one-artist group: low in many draws.

MOTIVE
    A stability check that cannot tell a firm grouping from an accident is worse than none.
"""
import numpy as np
import pytest

from src.cluster.kmeans import kmeans
from src.cluster.stability import artist_folds, bootstrap_stability, heldout_assignment


def clear_data(n_artists=24, rows_each=5, seed=1):
    """Three far-apart groups; every artist has rows in ONE group only (like a real artist whose songs are alike)."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]])
    group = np.arange(n_artists) % 3
    artists = np.repeat(np.arange(n_artists), rows_each)
    X = centres[group[artists]] + 0.4 * rng.standard_normal((len(artists), 2))
    return X, artists


def fitted(X, k):
    fit = kmeans(X, k, n_init=5, seed=0)
    return fit.labels, fit.centers


def test_folds_never_split_an_artist_and_balance_the_rows():
    sizes = np.array([40, 30, 20, 10, 10, 8, 6, 5, 5, 4, 3, 2, 2, 1, 1, 1])
    artists = np.repeat(np.arange(len(sizes)), sizes)
    folds = artist_folds(artists, 4, seed=3)
    for a in range(len(sizes)):
        assert len(set(folds[artists == a])) == 1
    load = np.bincount(folds, minlength=4)
    assert load.sum() == sizes.sum() and load.max() - load.min() <= sizes.max()
    assert load.max() - load.min() <= 10                                                  # the biggest artists were dealt first


def test_folds_are_reproducible_and_depend_on_the_seed():
    artists = np.repeat(np.arange(30), 3)
    assert np.array_equal(artist_folds(artists, 5, 7), artist_folds(artists, 5, 7))
    assert not np.array_equal(artist_folds(artists, 5, 7), artist_folds(artists, 5, 8))


def test_bootstrap_shapes_ranges_and_reproducibility():
    X, artists = clear_data()
    labels, centers = fitted(X, 3)
    out = bootstrap_stability(X, artists, labels, centers, n_boot=20, seed=4, n_init=2)
    assert out["ari"].shape == (20,) and out["jaccard"].shape == (20, 3) and out["agree"].shape == (len(X),)
    assert out["jaccard"].min() >= 0 and out["jaccard"].max() <= 1 and out["agree"].min() >= 0 and out["agree"].max() <= 1
    again = bootstrap_stability(X, artists, labels, centers, n_boot=20, seed=4, n_init=2)
    for key in out:
        assert np.array_equal(out[key], again[key])


def test_bootstrap_of_clear_groups_is_near_perfect():
    X, artists = clear_data()
    labels, centers = fitted(X, 3)
    out = bootstrap_stability(X, artists, labels, centers, n_boot=60, seed=1, n_init=3)
    assert out["ari"].min() > 0.95 and out["jaccard"].mean(axis=0).min() > 0.95 and out["agree"].min() > 0.95


def test_bootstrap_of_noise_is_clearly_less_stable():
    rng = np.random.default_rng(9)
    artists = np.repeat(np.arange(60), 4)
    X = rng.standard_normal((len(artists), 3))                                         # no groups at all
    labels, centers = fitted(X, 5)
    noise = bootstrap_stability(X, artists, labels, centers, n_boot=60, seed=2, n_init=3)
    Xc, ac = clear_data()
    lc, cc = fitted(Xc, 3)
    clear = bootstrap_stability(Xc, ac, lc, cc, n_boot=60, seed=2, n_init=3)
    assert noise["jaccard"].mean(axis=0).min() < 0.75 < clear["jaccard"].mean(axis=0).min()
    assert noise["ari"].mean() < clear["ari"].mean() - 0.2


def test_a_group_made_by_one_artist_is_lost_when_artists_are_redrawn_but_not_when_rows_are():
    rng = np.random.default_rng(3)
    small_group = np.array([30.0, 30.0]) + 0.3 * rng.standard_normal((40, 2))          # 40 rows of ONE artist, far from everything
    others = []
    for c in ([0.0, 0.0], [8.0, 0.0]):
        others.append(np.array(c) + 1.0 * rng.standard_normal((60, 2)))
    X = np.vstack([small_group] + others)
    artists = np.concatenate([np.zeros(40, dtype=int), 1 + np.repeat(np.arange(24), 5)])      # artist 0 owns the whole small group
    labels, centers = fitted(X, 3)
    own = int(labels[0])                                                                    # the group that holds artist 0
    assert (labels == own).sum() == 40
    by_artist = bootstrap_stability(X, artists, labels, centers, n_boot=80, seed=5, n_init=3)["jaccard"][:, own]
    by_row = bootstrap_stability(X, np.arange(len(X)), labels, centers, n_boot=80, seed=5, n_init=3)["jaccard"][:, own]
    assert by_row.mean() > 0.95
    assert by_artist.mean() < 0.9 and (by_artist < 0.8).mean() > 0.15                  # artist 0 is missing from about a third of the draws


def test_heldout_assignment_of_clear_groups_is_near_perfect_and_shapes_fit():
    X, artists = clear_data(n_artists=30)
    labels, centers = fitted(X, 3)
    out = heldout_assignment(X, artists, labels, centers, n_folds=5, n_repeats=2, seed=1, n_init=3)
    assert out["ari"].shape == (2,) and out["same"].shape == (2,) and out["recall"].shape == (2, 3)
    assert out["ari"].min() > 0.95 and out["same"].min() > 0.97 and out["recall"].min() > 0.95
    again = heldout_assignment(X, artists, labels, centers, n_folds=5, n_repeats=2, seed=1, n_init=3)
    assert all(np.array_equal(out[key], again[key]) for key in out)


def test_heldout_of_noise_is_clearly_worse():
    rng = np.random.default_rng(12)
    artists = np.repeat(np.arange(50), 4)
    X = rng.standard_normal((len(artists), 3))
    labels, centers = fitted(X, 5)
    out = heldout_assignment(X, artists, labels, centers, n_folds=5, n_repeats=2, seed=1, n_init=3)
    assert out["same"].mean() < 0.9 and out["ari"].mean() < 0.8
