"""
tests/test_ml_validation.py  -  checks of the fold-makers and the cross-validation loop (src/ml/validation.py).

WHAT THIS FILE IS
    Tests that artists are never split over folds, that folds are balanced and reproducible, that the forward-in-time split leaves
    a gap, that every row is predicted exactly once per repeat by a model that never saw it, and that a "memoriser" (a model that can
    only look up what it has seen) is exposed by the artist split but rewarded by a random split.

WHY WE NEED IT
    The whole point of the notebook's validation is that the model is tested on artists it has not met. If one artist leaked across a
    fold boundary, every score would be inflated and nobody would notice. The last test shows the mechanism with a model that is
    useless on new artists: with a random split it looks excellent, with the artist split it falls to chance.

WHAT HAPPENS WHEN IT RUNS
    Small random data sets with fixed seeds.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Make the most important safeguard of the modeling phase the best-tested one.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from src.ml import metrics as M
from src.ml.validation import (assert_no_group_overlap, cross_validated_predictions, fit_predict, forward_split, grouped_folds,
                               random_folds, repeated_grouped_folds)


@pytest.fixture(scope="module")
def groups():
    rng = np.random.default_rng(0)
    sizes = np.r_[92, 37, 34, 34, 32, rng.integers(1, 6, 120)]                   # one dominant artist, a few big ones, many small
    return np.repeat([f"artist {i:03d}" for i in range(len(sizes))], sizes)


# ================================================================================ folds
def test_no_artist_is_split_over_two_folds(groups):
    for seed in range(5):
        folds = grouped_folds(groups, 5, seed)
        assert_no_group_overlap(groups, folds)
        assert set(folds) == set(range(5))


def test_folds_are_balanced_in_size(groups):
    sizes = np.bincount(grouped_folds(groups, 5, 0))
    assert sizes.max() - sizes.min() <= 92 - 1                                    # never worse than the size of the biggest artist
    assert sizes.max() / sizes.min() < 1.25


def test_same_seed_same_folds_and_other_seeds_other_folds(groups):
    assert np.array_equal(grouped_folds(groups, 5, 3), grouped_folds(groups, 5, 3))
    assert not np.array_equal(grouped_folds(groups, 5, 3), grouped_folds(groups, 5, 4))
    repeats = repeated_grouped_folds(groups, 5, 3, seed=10)
    assert len(repeats) == 3 and not np.array_equal(repeats[0], repeats[1])


def test_the_dominant_artist_sits_in_one_fold_only(groups):
    folds = grouped_folds(groups, 5, 1)
    assert len(set(folds[groups == "artist 000"])) == 1


def test_bad_fold_counts_are_refused():
    with pytest.raises(ValueError):
        grouped_folds(["a", "a", "b"], 3)
    with pytest.raises(ValueError):
        grouped_folds(["a", "b", "c"], 1)


def test_assert_no_group_overlap_catches_a_leak():
    with pytest.raises(AssertionError, match="spread over several folds"):
        assert_no_group_overlap(np.array(["a", "a", "b"]), np.array([0, 1, 1]))


def test_random_folds_ignore_the_group_and_are_balanced():
    folds = random_folds(103, 5, 0)
    assert np.bincount(folds).tolist() in ([21, 21, 21, 20, 20], [21, 21, 21, 20, 20][::-1]) or np.bincount(folds).max() - np.bincount(folds).min() <= 1


# ================================================================================ forward split
def test_forward_split_leaves_a_gap_between_train_and_test():
    time = np.arange(100)
    train, test = forward_split(time, cutoff=70, gap=30)
    assert time[train].max() == 39 and time[test].min() == 70
    assert set(train).isdisjoint(test) and len(train) + len(test) == 70            # the 30 rows between 40 and 69 are used by neither
    with pytest.raises(ValueError):
        forward_split(time, cutoff=20, gap=30)


# ================================================================================ cross-validated predictions
class _TrainMean(BaseEstimator, RegressorMixin):
    """Predicts the mean of the training outcomes: lets a test know exactly which rows the model saw."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_every_row_is_predicted_once_per_repeat_by_a_model_that_never_saw_it():
    rng = np.random.default_rng(1)
    n = 60
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = rng.normal(size=n)
    labels = np.repeat(np.arange(12), 5)
    fold_sets = repeated_grouped_folds(labels, 4, 2, seed=0)
    P = cross_validated_predictions(_TrainMean, X, y, fold_sets, kind="value")
    assert P.shape == (2, n) and not np.isnan(P).any()
    for r, folds in enumerate(fold_sets):
        for fold in range(4):
            held_out = folds == fold
            assert np.allclose(P[r, held_out], y[~held_out].mean())                # the mean of the OTHER folds, nothing else


class _Memoriser(BaseEstimator, ClassifierMixin):
    """Remembers the outcome of every group it was trained on; for an unseen group it can only say 'I do not know' (0.5)."""

    def fit(self, X, y):
        table = pd.Series(np.asarray(y, dtype=float), index=X["artist"].to_numpy()).groupby(level=0).mean()
        self.table_ = table
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = X["artist"].map(self.table_).fillna(0.5).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def test_a_memoriser_looks_brilliant_with_random_folds_and_useless_with_artist_folds():
    rng = np.random.default_rng(2)
    n_artists, per = 80, 10
    artist = np.repeat(np.arange(n_artists), per)
    rate = rng.random(n_artists)
    y = (rng.random(len(artist)) < rate[artist]).astype(int)                      # the artist decides the outcome; nothing else does
    X = pd.DataFrame({"artist": artist})
    careless = cross_validated_predictions(_Memoriser, X, y, [random_folds(len(y), 5, 0)], kind="proba")[0]
    honest = cross_validated_predictions(_Memoriser, X, y, [grouped_folds(artist, 5, 0)], kind="proba")[0]
    assert M.roc_auc(y, careless) > 0.65
    assert M.roc_auc(y, honest) == 0.5                                             # it knows nothing about a new artist: every answer is a tie


def test_fit_predict_trains_on_the_train_rows_only():
    X = pd.DataFrame({"x": np.arange(10.0)})
    y = np.arange(10.0)
    out = fit_predict(_TrainMean, X, y, np.arange(0, 4), np.arange(6, 10), kind="value")
    assert np.allclose(out, 1.5)


def test_unknown_prediction_kind_is_refused():
    X = pd.DataFrame({"x": np.arange(10.0)})
    with pytest.raises(ValueError):
        cross_validated_predictions(_TrainMean, X, np.arange(10.0), [np.arange(10) % 2], kind="classes")
