"""
tests/test_cluster_artists.py  -  checks of the artist table, the six grouping columns and the PCA (src/cluster/artists.py).

WHAT THIS FILE IS
    Two artists worked out by hand (one song is exactly on place 10, the edge of the top 10; songs, chart days, days present, best and average place, top-10 share, chart days per song, most songs on one
    day), the standardized matrix (mean 0, standard deviation 1, the six columns in the documented order), and the PCA compared with scikit-learn
    (shares of the variance, orthogonal scores, scores that rebuild the matrix, sign rule).

WHY WE NEED IT
    The artist segments and the PCA picture of notebook 06c come from these functions.

EXPECTED OUTCOME
    Exact agreement with the hand-made numbers, and agreement with scikit-learn to at least ten digits.

MOTIVE
    The footprint of an artist is only useful when the table says what it claims to say.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from src.cluster.artists import FEATURES, artist_matrix, artist_table, pca

ROWS = pd.DataFrame({
    "primary_artist": ["A", "A", "A", "A", "B", "B"],
    "track_id":       ["t1", "t1", "t1", "t2", "t3", "t3"],
    "date":           pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-01", "2024-01-02"]),
    "position":       [1, 2, 11, 10, 50, 40],
})


def test_artist_table_by_hand():
    t = artist_table(ROWS)
    assert list(t.index) == ["A", "B"]
    a, b = t.loc["A"], t.loc["B"]
    assert (a["songs"], a["chart_days"], a["days_present"], a["best_place"], a["peak_songs"]) == (2, 4, 3, 1, 2)
    assert a["mean_place"] == pytest.approx((1 + 2 + 11 + 10) / 4) and a["top10_share"] == pytest.approx(3 / 4) and a["days_per_song"] == pytest.approx(2.0)
    assert (b["songs"], b["chart_days"], b["days_present"], b["best_place"], b["peak_songs"]) == (1, 2, 2, 40, 1)
    assert b["mean_place"] == pytest.approx(45.0) and b["top10_share"] == 0.0 and b["days_per_song"] == pytest.approx(2.0)


def synthetic_table(n=40, seed=0):
    rng = np.random.default_rng(seed)
    songs = rng.integers(1, 30, n)
    return pd.DataFrame({"songs": songs, "chart_days": songs * rng.integers(2, 60, n), "days_present": rng.integers(2, 400, n),
                         "mean_place": rng.uniform(5, 45, n), "top10_share": rng.uniform(0, 1, n), "peak_songs": rng.integers(1, 8, n)}
                        ).assign(days_per_song=lambda d: d["chart_days"] / d["songs"])


def test_matrix_columns_order_logs_and_standardization():
    table = synthetic_table()
    Z, frame = artist_matrix(table)
    assert list(frame.columns) == FEATURES and Z.shape == (len(table), 6)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-12) and np.allclose(Z.std(axis=0), 1, atol=1e-12)
    assert np.allclose(frame["log_songs"], np.log(table["songs"])) and np.allclose(frame["log_peak_songs"], np.log(table["peak_songs"]))
    assert np.allclose(frame["mean_place"], table["mean_place"]) and np.allclose(frame["top10_share"], table["top10_share"])
    assert np.allclose(frame["log_days_per_song"], np.log(table["days_per_song"])) and np.allclose(frame["log_days_present"], np.log(table["days_present"]))


def test_pca_agrees_with_scikit_learn_and_rebuilds_the_matrix():
    Z, _ = artist_matrix(synthetic_table(60, seed=3))
    scores, share, loadings = pca(Z)
    ref = PCA(n_components=6).fit(Z)
    assert np.allclose(share, ref.explained_variance_ratio_, atol=1e-12) and share.sum() == pytest.approx(1.0)
    assert np.all(np.diff(share) <= 1e-12)                                                 # largest component first
    assert np.allclose(scores @ loadings, Z - Z.mean(axis=0), atol=1e-10)
    gram = scores.T @ scores
    assert np.allclose(gram - np.diag(np.diag(gram)), 0, atol=1e-8)                      # the scores of different components are uncorrelated
    for i in range(6):
        assert abs(abs(loadings[i] @ ref.components_[i]) - 1) < 1e-8                      # the same axes as scikit-learn, up to the sign
        assert loadings[i][np.abs(loadings[i]).argmax()] > 0                              # our sign rule
