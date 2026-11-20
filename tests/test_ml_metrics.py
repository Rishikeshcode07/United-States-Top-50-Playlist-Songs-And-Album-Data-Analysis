"""
tests/test_ml_metrics.py  -  checks of the scoring functions (src/ml/metrics.py) against scikit-learn and against hand-worked cases.

WHAT THIS FILE IS
    Every score is compared with scikit-learn's version on random data that contains ties, with and without integer weights, and a
    weight MATRIX (many re-drawn data sets at once) is compared column by column with a loop. The artist bootstrap weights, the gains
    table, the break-even table and the calibration table are checked on small examples worked out on paper.

WHY WE NEED IT
    The functions are written to score thousands of re-drawn data sets at once, which is not how scikit-learn does it. A fast function
    that is slightly wrong would give confidence intervals that look fine and mean nothing. Agreement with an independent, trusted
    implementation is the check.

WHAT HAPPENS WHEN IT RUNS
    Random data with fixed seeds; scikit-learn is imported only inside this test file.

EXPECTED OUTCOME
    All tests pass; the largest difference to scikit-learn is at the level of rounding.

MOTIVE
    The intervals in the notebook are only as good as the scores under them.
"""
import numpy as np
import pandas as pd
import pytest

sk = pytest.importorskip("sklearn.metrics")

from src.ml import metrics as M


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    n = 300
    y = (rng.random(n) < 0.2).astype(int)
    score = np.round(rng.normal(size=n) + 1.0 * y, 1)                 # rounded on purpose: many ties
    weights = rng.integers(0, 4, size=n).astype(float)
    matrix = rng.integers(0, 4, size=(n, 40)).astype(float)
    return y, score, weights, matrix


# ================================================================================ two classes
def test_auc_and_average_precision_equal_sklearn_with_ties(data):
    y, score, _, _ = data
    assert M.roc_auc(y, score) == pytest.approx(sk.roc_auc_score(y, score), abs=1e-12)
    assert M.average_precision(y, score) == pytest.approx(sk.average_precision_score(y, score), abs=1e-12)


def test_auc_and_average_precision_equal_sklearn_with_integer_weights(data):
    y, score, weights, _ = data
    assert M.roc_auc(y, score, weights) == pytest.approx(sk.roc_auc_score(y, score, sample_weight=weights), abs=1e-12)
    assert M.average_precision(y, score, weights) == pytest.approx(sk.average_precision_score(y, score, sample_weight=weights), abs=1e-12)


def test_a_weight_matrix_gives_the_same_as_one_column_at_a_time(data):
    y, score, _, matrix = data
    auc = M.roc_auc(y, score, matrix)
    ap = M.average_precision(y, score, matrix)
    assert auc.shape == (40,) and ap.shape == (40,)
    for column in (0, 7, 39):
        assert auc[column] == pytest.approx(sk.roc_auc_score(y, score, sample_weight=matrix[:, column]), abs=1e-12)
        assert ap[column] == pytest.approx(sk.average_precision_score(y, score, sample_weight=matrix[:, column]), abs=1e-12)


def test_a_constant_score_has_auc_one_half_and_average_precision_equal_to_the_base_rate():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    constant = np.full(10, 0.3)
    assert M.roc_auc(y, constant) == 0.5
    assert M.average_precision(y, constant) == pytest.approx(0.2)


def test_perfect_and_reversed_scores():
    y = np.array([0, 0, 1, 1, 0, 1])
    assert M.roc_auc(y, y * 1.0) == 1.0
    assert M.roc_auc(y, 1.0 - y) == 0.0
    assert M.average_precision(y, y * 1.0) == 1.0


def test_auc_is_missing_when_one_class_is_absent():
    assert np.isnan(M.roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert np.isnan(M.average_precision([0, 0], [0.1, 0.2]))


def test_brier_and_log_loss_equal_sklearn_with_weights(data):
    y, score, weights, _ = data
    p = 1 / (1 + np.exp(-score))
    assert M.brier_score(y, p, weights) == pytest.approx(sk.brier_score_loss(y, p, sample_weight=weights), abs=1e-12)
    assert M.log_loss(y, p, weights) == pytest.approx(sk.log_loss(y, p, sample_weight=weights), abs=1e-12)
    assert M.log_loss(y, np.where(y == 1, 1.0, 0.0)) < 1e-10                      # certain and right: no loss, and no infinity


def test_confusion_counts_and_threshold_scores():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    flagged = np.array([1, 1, 0, 1, 0, 0, 0, 0], dtype=bool)
    tp, fp, fn, tn = M.confusion_counts(y, flagged)
    assert (tp, fp, fn, tn) == (2, 1, 1, 4)
    s = M.threshold_scores(tp, fp, fn, tn)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3) and s["specificity"] == pytest.approx(0.8)
    assert s["f1"] == pytest.approx(2 / 3) and s["accuracy"] == pytest.approx(6 / 8) and s["balanced_accuracy"] == pytest.approx((2 / 3 + 0.8) / 2)
    nothing = M.threshold_scores(0, 0, 3, 5)                                       # no song flagged: precision is undefined, not zero
    assert np.isnan(nothing["precision"]) and nothing["recall"] == 0.0


def test_top_share_flags_pick_the_highest_scores():
    score = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
    assert M.top_share_flags(score, 0.4).tolist() == [False, True, False, True, False]
    assert M.top_share_flags(score, 0.0).sum() == 1                                # at least one song


# ================================================================================ many classes
def test_matrix_scores_equal_sklearn():
    rng = np.random.default_rng(8)
    truth = rng.integers(0, 3, 400)
    guess = np.where(rng.random(400) < 0.5, truth, rng.integers(0, 3, 400))
    matrix = M.confusion_matrix(truth, guess, 3)
    assert matrix.sum() == 400 and matrix.shape == (3, 3)
    scores = M.matrix_scores(matrix)
    assert scores["accuracy"] == pytest.approx(sk.accuracy_score(truth, guess))
    assert scores["balanced_accuracy"] == pytest.approx(sk.balanced_accuracy_score(truth, guess))
    assert scores["macro_f1"] == pytest.approx(sk.f1_score(truth, guess, average="macro"))
    assert scores["weighted_kappa"] == pytest.approx(sk.cohen_kappa_score(truth, guess, weights="quadratic"))


def test_a_class_that_is_never_forecast_counts_as_f1_zero_like_sklearn():
    """The 'always guess the most common class' baseline: its macro F1 must be one third of that class's F1, not the F1 of that class alone."""
    truth = np.array([0] * 5 + [1] * 3 + [2] * 2)
    guess = np.zeros(10, dtype=int)                                                  # classes 1 and 2 are never forecast
    scores = M.matrix_scores(M.confusion_matrix(truth, guess, 3))
    assert scores["macro_f1"] == pytest.approx(sk.f1_score(truth, guess, average="macro", zero_division=0))
    assert scores["macro_f1"] == pytest.approx((2 * 5 / (2 * 5 + 5)) / 3)
    assert scores["balanced_accuracy"] == pytest.approx(1 / 3)


def test_matrix_scores_on_a_stack_of_matrices_equal_one_at_a_time():
    rng = np.random.default_rng(9)
    truth = rng.integers(0, 3, 200)
    guess = rng.integers(0, 3, 200)
    weights = rng.integers(0, 3, size=(200, 5)).astype(float)
    stack = M.confusion_matrix(truth, guess, 3, weights)
    assert stack.shape == (5, 3, 3)
    stacked = M.matrix_scores(stack)
    for b in range(5):
        single = M.matrix_scores(M.confusion_matrix(truth, guess, 3, weights[:, b]))
        for key in single:
            assert stacked[key][b] == pytest.approx(single[key])


def test_kappa_is_one_for_perfect_and_penalises_far_misses_more():
    truth = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 2])
    assert M.matrix_scores(M.confusion_matrix(truth, truth, 3))["weighted_kappa"] == pytest.approx(1.0)
    near = truth.copy()
    near[0] = 1                                                                      # one song off by one class
    far = truth.copy()
    far[0] = 2                                                                       # one song off by two classes
    assert M.matrix_scores(M.confusion_matrix(truth, near, 3))["weighted_kappa"] > M.matrix_scores(M.confusion_matrix(truth, far, 3))["weighted_kappa"]


def test_multiclass_log_loss_equals_sklearn():
    rng = np.random.default_rng(4)
    probabilities = rng.dirichlet(np.ones(3), size=100)
    truth = rng.integers(0, 3, 100)
    assert M.multiclass_log_loss(truth, probabilities) == pytest.approx(sk.log_loss(truth, probabilities, labels=[0, 1, 2]))


# ================================================================================ regression
def test_regression_scores_equal_sklearn_with_weights(data):
    _, _, weights, _ = data
    rng = np.random.default_rng(10)
    y = rng.normal(size=300)
    prediction = 0.5 * y + rng.normal(size=300)
    assert M.r_squared(y, prediction, weights) == pytest.approx(sk.r2_score(y, prediction, sample_weight=weights))
    assert M.mean_absolute_error(y, prediction, weights) == pytest.approx(sk.mean_absolute_error(y, prediction, sample_weight=weights))
    assert M.root_mean_squared_error(y, prediction, weights) == pytest.approx(np.sqrt(sk.mean_squared_error(y, prediction, sample_weight=weights)))
    assert M.r_squared(y, np.full(300, y.mean())) == pytest.approx(0.0, abs=1e-12)           # the mean predictor
    assert M.r_squared(y, np.full(300, y.mean() + 5)) < 0                                      # worse than the mean


# ================================================================================ uncertainty
def test_artist_weights_are_constant_inside_an_artist_and_draw_as_many_artists_as_there_are():
    groups = np.array(["a", "a", "a", "b", "c", "c", "d", "e", "e", "e", "e"])
    weights = M.artist_resample_weights(groups, 200, seed=1)
    assert weights.shape == (11, 200)
    for artist in "abcde":
        rows = weights[groups == artist]
        assert (rows == rows[0]).all()
    draws = np.array([weights[np.flatnonzero(groups == a)[0]] for a in "abcde"])
    assert (draws.sum(axis=0) == 5).all()                                      # five artists drawn in every replicate
    assert abs(draws.mean() - 1.0) < 0.1                                       # an artist is drawn once on average
    assert np.array_equal(weights, M.artist_resample_weights(groups, 200, seed=1))      # same seed, same weights
    assert not np.array_equal(weights, M.artist_resample_weights(groups, 200, seed=2))


def test_bootstrap_interval_covers_the_estimate_and_shrinks_with_more_data():
    rng = np.random.default_rng(12)
    widths = []
    for n_artists in (30, 300):
        groups = np.repeat(np.arange(n_artists), 3)
        y = (rng.random(len(groups)) < 0.3).astype(int)
        score = rng.normal(size=len(groups)) + 0.8 * y
        low, high = M.interval(M.roc_auc(y, score, M.artist_resample_weights(groups, 400, seed=2)))
        assert low < M.roc_auc(y, score) < high
        widths.append(high - low)
    assert widths[1] < widths[0]


def test_interval_ignores_missing_replicates():
    assert M.interval([1.0, 2.0, np.nan, 3.0, 4.0], 0.5) == pytest.approx((1.75, 3.25))
    assert all(np.isnan(v) for v in M.interval([np.nan]))


# ================================================================================ calibration, gains, decisions
def test_calibration_table_groups_equal_sized_and_reports_observed_shares():
    p = np.linspace(0.05, 0.95, 20)
    y = (p > 0.5).astype(int)
    table = M.calibration_table(y, p, n_bins=4)
    assert table["songs"].tolist() == [5, 5, 5, 5]
    assert table["observed"].tolist() == [0.0, 0.0, 1.0, 1.0]                      # the song at p = 0.47 is the last of the second group
    assert (table["ci_low"] <= table["observed"]).all() and (table["observed"] <= table["ci_high"]).all()
    assert table["mean_predicted"].is_monotonic_increasing


def test_gains_table_for_a_perfect_and_for_a_useless_score():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    perfect = M.gains_table(y, y * 1.0)
    assert perfect["share_flagged"].iloc[1] == pytest.approx(0.2) and perfect["gain"].iloc[1] == pytest.approx(1.0)
    assert perfect["lift"].iloc[1] == pytest.approx(5.0)
    useless = M.gains_table(y, np.zeros(10))
    assert useless["gain"].iloc[-1] == 1.0 and len(useless) == 2                  # all ties: one step, from nothing to everything
    assert useless["lift"].iloc[-1] == pytest.approx(1.0)


def test_decision_table_arithmetic_and_break_even():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    table = M.decision_table(y, score, value_ratios=[5, 20], shares=[0.2, 0.3])
    row = table[(table["value_ratio"] == 5) & (table["share_promoted"] == 0.2)].iloc[0]
    assert (row["promoted"], row["hits"], row["precision"]) == (2, 2, 1.0)
    assert row["net"] == pytest.approx(5 * 2 - 2) and row["roi"] == pytest.approx(4.0) and row["break_even"] == pytest.approx(0.2)
    row = table[(table["value_ratio"] == 5) & (table["share_promoted"] == 0.3)].iloc[0]
    assert (row["promoted"], row["hits"]) == (3, 2) and row["roi"] == pytest.approx((10 - 3) / 3)
    losing = M.decision_table(y, score, value_ratios=[2], shares=[1.0]).iloc[0]            # promote everything at 20% hit rate and ratio 2
    assert losing["roi"] == pytest.approx((2 * 2 - 10) / 10) and losing["roi"] < 0


def test_a_binary_outcome_must_be_zero_or_one():
    with pytest.raises(ValueError):
        M.roc_auc([0, 1, 2], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        M.roc_auc([0, 1, 1], [0.1, 0.2, 0.3], weights=np.ones(2))


# ================================================================================ curves
def test_roc_curve_equals_sklearn_and_its_area_is_the_auc(data):
    y, score, _, _ = data
    fpr, tpr = M.roc_curve(y, score)
    ref_fpr, ref_tpr, _ = sk.roc_curve(y, score, drop_intermediate=False)
    assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1
    assert np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2) == pytest.approx(M.roc_auc(y, score), abs=1e-12)
    # every point of the reference curve lies on my curve (mine has one point per distinct score, the reference one per threshold)
    mine = set(zip(np.round(fpr, 12), np.round(tpr, 12)))
    assert {(round(a, 12), round(b, 12)) for a, b in zip(ref_fpr, ref_tpr)} <= mine


def test_precision_recall_curve_equals_sklearn(data):
    y, score, _, _ = data
    recall, precision = M.precision_recall_curve(y, score)
    ref_precision, ref_recall, _ = sk.precision_recall_curve(y, score)
    pairs = set(zip(np.round(recall[1:], 12), np.round(precision[1:], 12)))
    assert {(round(a, 12), round(b, 12)) for a, b in zip(ref_recall, ref_precision) if a > 0} <= pairs
    assert recall[0] == 0 and recall[-1] == 1
    assert np.sum(np.diff(recall) * precision[1:]) == pytest.approx(M.average_precision(y, score), abs=1e-12)    # the step area is the average precision
