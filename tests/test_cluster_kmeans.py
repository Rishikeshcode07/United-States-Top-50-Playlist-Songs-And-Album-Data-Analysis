"""
tests/test_cluster_kmeans.py  -  checks of the k-means, silhouette, adjusted Rand index and label matching (src/cluster/kmeans.py).

WHAT THIS FILE IS
    Our own few-line k-means compared with scikit-learn on data whose groups are clear, the silhouette worked out by hand and compared with
    scikit-learn on random data, the adjusted Rand index compared with scikit-learn (also for relabelled groupings and the degenerate cases),
    the numbering of the groups by size, the repair of an empty group, and the matching of the centres of two groupings.

WHY WE NEED IT
    Every group, silhouette and agreement score of notebook 06c comes from these functions.

EXPECTED OUTCOME
    Agreement with scikit-learn to at least nine digits where both are exact, and the same groups (up to the numbering) on clear data.

MOTIVE
    A grouping method that is a little off looks exactly like one that is right, so the numbers are checked.
"""
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import KMeans

from src.cluster.kmeans import adjusted_rand, assign, kmeans, match_labels, silhouette, squared_distances


def blobs(sizes=(60, 40, 20), spread=0.3, seed=1):
    """Well separated groups in three columns: the centres are far apart compared with the spread."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 10], [10, 10, -10]], dtype=float)[:len(sizes)]
    X = np.vstack([c + spread * rng.standard_normal((n, 3)) for c, n in zip(centres, sizes)])
    truth = np.repeat(np.arange(len(sizes)), sizes)
    return X, truth


def test_squared_distances_agree_with_scipy():
    rng = np.random.default_rng(0)
    X, C = rng.standard_normal((50, 4)), rng.standard_normal((6, 4))
    assert np.allclose(squared_distances(X, C), cdist(X, C, "sqeuclidean"), atol=1e-12)


def test_clear_groups_are_found_exactly():
    X, truth = blobs()
    fit = kmeans(X, 3, n_init=5, seed=0)
    assert adjusted_rand(fit.labels, truth) == 1.0
    assert sorted(np.bincount(fit.labels)) == [20, 40, 60]


def test_agrees_with_scikit_learn_on_clear_groups():
    X, _ = blobs(sizes=(70, 50, 30, 15))
    ours = kmeans(X, 4, n_init=10, seed=3)
    theirs = KMeans(n_clusters=4, n_init=10, random_state=3).fit(X)
    assert adjusted_rand(ours.labels, theirs.labels_) == 1.0
    assert ours.inertia == pytest.approx(theirs.inertia_, rel=1e-9)
    order = np.lexsort(ours.centers.T)
    assert np.allclose(ours.centers[order], theirs.cluster_centers_[np.lexsort(theirs.cluster_centers_.T)], atol=1e-9)


def test_same_seed_same_result_and_inertia_is_the_sum_of_squares():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((200, 5))
    a, b = kmeans(X, 4, n_init=4, seed=11), kmeans(X, 4, n_init=4, seed=11)
    assert np.array_equal(a.labels, b.labels) and np.allclose(a.centers, b.centers) and a.inertia == b.inertia
    assert a.inertia == pytest.approx(((X - a.centers[a.labels]) ** 2).sum(), rel=1e-12)


def test_more_starts_never_give_a_worse_fit():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((300, 6))
    one, many = kmeans(X, 6, n_init=1, seed=4).inertia, kmeans(X, 6, n_init=12, seed=4).inertia
    assert many < one - 1.0                                                              # for this data one start is clearly worse than the best of twelve


def test_groups_are_numbered_by_size_and_centres_follow_the_labels():
    X, _ = blobs(sizes=(15, 70, 30))
    fit = kmeans(X, 3, n_init=5, seed=2)
    sizes = np.bincount(fit.labels)
    assert list(sizes) == [70, 30, 15]
    for j in range(3):
        assert np.allclose(fit.centers[j], X[fit.labels == j].mean(axis=0))


def test_one_group_is_the_mean_and_n_groups_have_no_distance_left():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 3))
    one = kmeans(X, 1, n_init=2, seed=0)
    assert np.allclose(one.centers[0], X.mean(axis=0)) and set(one.labels) == {0}
    every = kmeans(X, 30, n_init=1, seed=0)
    assert every.inertia == pytest.approx(0.0, abs=1e-9)


def test_k_outside_the_range_is_refused():
    X = np.arange(12, dtype=float).reshape(6, 2)
    for k in (0, 7):
        with pytest.raises(ValueError):
            kmeans(X, k)


def test_empty_group_is_repaired_when_rows_repeat():
    X = np.vstack([np.zeros((10, 2)), np.ones((10, 2))])                      # only two different rows, three groups asked for
    fit = kmeans(X, 3, n_init=3, seed=0)
    assert fit.labels.min() >= 0 and fit.labels.max() <= 2 and np.isfinite(fit.inertia) and fit.inertia == pytest.approx(0.0, abs=1e-9)


def test_assign_gives_the_nearest_centre():
    C = np.array([[0.0, 0.0], [10.0, 10.0]])
    assert list(assign(np.array([[1, 1], [9, 8], [4, 4], [6, 6]]), C)) == [0, 1, 0, 1]


def test_silhouette_by_hand():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    labels = np.array([0, 0, 1, 1])
    # row 0: a = 1, b = (10 + 11) / 2 = 10.5;  row 1: a = 1, b = (9 + 10) / 2 = 9.5;  the other two groups mirror them
    expected = ((10.5 - 1) / 10.5 + (9.5 - 1) / 9.5) / 2
    assert silhouette(X, labels) == pytest.approx(expected, rel=1e-12)


def test_silhouette_agrees_with_scikit_learn():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((120, 4))
    for k in (2, 3, 5):
        labels = rng.integers(0, k, 120)
        assert silhouette(X, labels) == pytest.approx(metrics.silhouette_score(X, labels), abs=1e-12)
    clear, truth = blobs()
    assert silhouette(clear, truth) == pytest.approx(metrics.silhouette_score(clear, truth), abs=1e-12) and silhouette(clear, truth) > 0.9


def test_silhouette_needs_two_groups_and_scores_a_single_row_group_as_zero():
    X = np.arange(10, dtype=float).reshape(5, 2)
    with pytest.raises(ValueError):
        silhouette(X, np.zeros(5, dtype=int))
    labels = np.array([0, 0, 0, 0, 1])
    assert silhouette(X, labels) == pytest.approx(metrics.silhouette_score(X, labels), abs=1e-12)


def test_adjusted_rand_agrees_with_scikit_learn():
    rng = np.random.default_rng(6)
    for _ in range(5):
        a, b = rng.integers(0, 4, 90), rng.integers(0, 5, 90)
        assert adjusted_rand(a, b) == pytest.approx(metrics.adjusted_rand_score(a, b), abs=1e-12)


def test_adjusted_rand_ignores_the_numbering_and_sees_a_real_difference():
    a = np.array([0, 0, 1, 1, 2, 2, 2])
    assert adjusted_rand(a, np.array([5, 5, 3, 3, 9, 9, 9])) == 1.0
    assert adjusted_rand(a, np.array([0, 1, 0, 1, 0, 1, 0])) < 0.2


def test_adjusted_rand_degenerate_cases_and_length_check():
    assert adjusted_rand(np.zeros(6, dtype=int), np.zeros(6, dtype=int)) == 1.0           # one group against one group
    assert adjusted_rand(np.arange(6), np.arange(6)) == 1.0                              # all singletons against all singletons
    with pytest.raises(ValueError):
        adjusted_rand(np.zeros(5), np.zeros(6))


def test_match_labels_recovers_a_permutation_of_the_centres():
    reference = np.array([[0.0, 0.0], [5.0, 5.0], [-5.0, 5.0]])
    shuffled = reference[[2, 0, 1]] + 0.1                                                  # new group j is reference group [2, 0, 1][j]
    assert list(match_labels(reference, shuffled)) == [2, 0, 1]
    assert sorted(match_labels(reference, shuffled)) == [0, 1, 2]                        # one-to-one
