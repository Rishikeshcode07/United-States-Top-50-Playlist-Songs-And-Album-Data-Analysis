"""
src/ml/metrics.py  -  the scores of Phase 6A, written so that one call can score thousands of re-sampled data sets at once.

WHAT THIS FILE IS
    Scoring functions and the helpers around them.

        classification   roc_auc, average_precision, brier_score, log_loss, confusion_counts, threshold_scores, calibration_table
        many classes     confusion_matrix, accuracy / balanced accuracy / macro F1 / quadratic weighted kappa from a matrix
        regression       r_squared, mean_absolute_error, root_mean_squared_error
        uncertainty      artist_resample_weights (the weights of a bootstrap that re-draws whole ARTISTS), interval
        decisions        gains_table, decision_table (net value and return of flagging the top share of songs)

    Every score accepts `weights`. A weight vector gives one weighted score. A weight MATRIX (rows x replicates) gives one score per
    column, calculated for all replicates together with array arithmetic and no Python loop over replicates.

WHY WE NEED IT
    A single AUC says nothing about how much to trust it. With 56 songs that climb, one more or one fewer changes it visibly. The
    honest way to show the doubt is to re-draw the data many times and watch how the score moves. The re-drawing must keep
    the songs of an artist together (they are not independent), so what is re-drawn is the artists, not the songs: an artist that
    is drawn twice counts all of its songs twice, one that is not drawn counts none. In weight terms: a song's weight is the number
    of times its artist was drawn. Scoring with weights is exactly scoring the re-drawn data.

    The functions agree with scikit-learn's (the tests compare them on random data with integer weights), they just do many weight
    columns at once.

WHAT HAPPENS WHEN IT RUNS
    Nothing by itself. Tie handling follows scikit-learn: songs with the same score are one step of the curve (AUC counts a tie as
    one half; average precision treats them as one threshold).

EXPECTED OUTCOME
    `roc_auc(y, score)` equals sklearn's `roc_auc_score`; `roc_auc(y, score, W)` gives one AUC for each column of W.

MOTIVE
    Report every score with its doubt, and make the doubt cheap enough to compute for every model and every chart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ================================================================================================ helpers
def _weights(weights, n):
    """Weights as a 2-D float array (rows, columns); a missing or 1-D weight becomes one column. Returns (array, was_1d)."""
    if weights is None:
        return np.ones((n, 1)), True
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = w[:, None]
        single = True
    else:
        single = False
    if w.shape[0] != n:
        raise ValueError("weights must have one row per observation")
    return w, single


def _out(values, single):
    return float(values[0]) if single else values


def _binary(y):
    y = np.asarray(y)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("a binary outcome must contain only 0 and 1")
    return y.astype(bool)


def _tied_groups(score, y, w):
    """Sum the positive weight and the negative weight of every distinct score value, in ascending order of the score."""
    _, inverse = np.unique(np.asarray(score, dtype=float), return_inverse=True)
    n_groups = int(inverse.max()) + 1
    positive = np.zeros((n_groups, w.shape[1]))
    negative = np.zeros((n_groups, w.shape[1]))
    np.add.at(positive, inverse[y], w[y])
    np.add.at(negative, inverse[~y], w[~y])
    return positive, negative


# ================================================================================================ two classes
def roc_auc(y, score, weights=None):
    """
    Area under the ROC curve: the chance that a random positive song gets a higher score than a random negative one (a tie counts as
    one half). NaN when the weighted data has no positive or no negative song.
    """
    y = _binary(y)
    w, single = _weights(weights, len(y))
    positive, negative = _tied_groups(score, y, w)
    below = np.cumsum(negative, axis=0) - negative
    numerator = (positive * (below + 0.5 * negative)).sum(axis=0)
    denominator = positive.sum(axis=0) * negative.sum(axis=0)
    safe = np.where(denominator > 0, denominator, 1.0)
    return _out(np.where(denominator > 0, numerator / safe, np.nan), single)


def average_precision(y, score, weights=None):
    """Area under the precision-recall curve, step version (the sum over thresholds of the gain in recall times the precision)."""
    y = _binary(y)
    w, single = _weights(weights, len(y))
    positive, negative = _tied_groups(score, y, w)
    positive, negative = positive[::-1], negative[::-1]                    # highest score first
    true_positive = np.cumsum(positive, axis=0)
    flagged = true_positive + np.cumsum(negative, axis=0)
    total_positive = positive.sum(axis=0)
    precision = np.divide(true_positive, flagged, out=np.zeros_like(true_positive), where=flagged > 0)
    safe = np.where(total_positive > 0, total_positive, 1.0)
    value = (positive / safe * precision).sum(axis=0)
    return _out(np.where(total_positive > 0, value, np.nan), single)


def roc_curve(y, score):
    """
    The ROC curve as two arrays (false positive rate, true positive rate), from (0, 0) to (1, 1), with one step per distinct score value
    (songs with the same score move the curve diagonally). Its area equals `roc_auc`.
    """
    y = _binary(y)
    positive, negative = _tied_groups(score, y, np.ones((len(y), 1)))
    positive, negative = positive[::-1, 0], negative[::-1, 0]                # highest score first
    tpr = np.r_[0.0, np.cumsum(positive) / positive.sum()]
    fpr = np.r_[0.0, np.cumsum(negative) / negative.sum()]
    return fpr, tpr


def precision_recall_curve(y, score):
    """
    The precision-recall curve as two arrays (recall, precision), one step per distinct score value, starting at recall 0 with the
    precision of the first step. Precision is the share of flagged songs that are positive, recall the share of positives flagged.
    """
    y = _binary(y)
    positive, negative = _tied_groups(score, y, np.ones((len(y), 1)))
    positive, negative = positive[::-1, 0], negative[::-1, 0]
    true_positive = np.cumsum(positive)
    flagged = true_positive + np.cumsum(negative)
    recall = true_positive / positive.sum()
    precision = true_positive / flagged
    return np.r_[0.0, recall], np.r_[precision[0], precision]


def brier_score(y, probability, weights=None):
    """Mean squared gap between the predicted probability and what happened (0 = perfect; a model that always says the base rate scores p(1-p))."""
    y = _binary(y).astype(float)
    w, single = _weights(weights, len(y))
    squared = (np.asarray(probability, dtype=float) - y) ** 2
    return _out((w * squared[:, None]).sum(axis=0) / w.sum(axis=0), single)


def log_loss(y, probability, weights=None, eps: float = 1e-15):
    """Mean negative log of the probability given to what happened (lower is better)."""
    y = _binary(y).astype(float)
    w, single = _weights(weights, len(y))
    p = np.clip(np.asarray(probability, dtype=float), eps, 1 - eps)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return _out((w * loss[:, None]).sum(axis=0) / w.sum(axis=0), single)


def confusion_counts(y, flagged, weights=None):
    """(true positives, false positives, false negatives, true negatives); each is a number, or one value per weight column."""
    y = _binary(y)
    flagged = np.asarray(flagged, dtype=bool)
    w, single = _weights(weights, len(y))
    tp = (w * (y & flagged)[:, None]).sum(axis=0)
    fp = (w * (~y & flagged)[:, None]).sum(axis=0)
    fn = (w * (y & ~flagged)[:, None]).sum(axis=0)
    tn = (w * (~y & ~flagged)[:, None]).sum(axis=0)
    return tuple(_out(part, single) for part in (tp, fp, fn, tn))


def threshold_scores(tp, fp, fn, tn) -> dict:
    """Precision, recall (sensitivity), specificity, F1, accuracy and balanced accuracy from the four counts (NaN where a denominator is 0)."""
    tp, fp, fn, tn = (np.asarray(part, dtype=float) for part in (tp, fp, fn, tn))

    def ratio(a, b):
        return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b > 0)

    precision, recall, specificity = ratio(tp, tp + fp), ratio(tp, tp + fn), ratio(tn, tn + fp)
    f1 = ratio(2 * precision * recall, precision + recall)
    accuracy = ratio(tp + tn, tp + fp + fn + tn)
    return {"precision": precision, "recall": recall, "specificity": specificity, "f1": f1, "accuracy": accuracy,
            "balanced_accuracy": (recall + specificity) / 2}


def top_share_flags(score, share: float):
    """Flag the songs with the highest scores: the top `share` of all songs (at least one). Ties at the edge are broken by order."""
    score = np.asarray(score, dtype=float)
    k = max(1, int(round(share * len(score))))
    flagged = np.zeros(len(score), dtype=bool)
    flagged[np.argsort(-score, kind="stable")[:k]] = True
    return flagged


def calibration_table(y, probability, n_bins: int = 5, confidence: float = 0.95) -> pd.DataFrame:
    """
    Reliability table: songs are put into `n_bins` groups of (nearly) equal size by predicted probability; for each group the mean
    prediction, the observed share of positives and its Wilson interval. A calibrated model has the two shares close together.
    """
    from scipy import stats
    y = _binary(y).astype(float)
    p = np.asarray(probability, dtype=float)
    order = np.argsort(p, kind="stable")
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    rows = []
    for part in np.array_split(order, n_bins):
        n, hits = len(part), float(y[part].sum())
        share = hits / n
        centre = (share + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        rows.append({"songs": n, "mean_predicted": float(p[part].mean()), "observed": share,
                     "ci_low": float(max(0.0, centre - half)), "ci_high": float(min(1.0, centre + half))})
    return pd.DataFrame(rows)


# ================================================================================================ many classes
def confusion_matrix(y_true, y_pred, n_classes: int, weights=None):
    """Counts with the true class in the rows and the predicted class in the columns: shape (classes, classes), or (columns, classes, classes) for a weight matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    w, single = _weights(weights, len(y_true))
    out = np.zeros((w.shape[1], n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            out[:, i, j] = (w * ((y_true == i) & (y_pred == j))[:, None]).sum(axis=0)
    return out[0] if single else out


def matrix_scores(matrix) -> dict:
    """
    Scores of a confusion matrix (or a stack of them): accuracy, balanced accuracy (the mean recall of the classes), macro F1 (the
    mean F1 of the classes) and the quadratic weighted kappa (agreement beyond chance, with a miss of two classes costing four times a
    miss of one: made for ordered classes such as Flash < Short < Sustained).
    """
    m = np.asarray(matrix, dtype=float)
    stacked = m.ndim == 3
    if not stacked:
        m = m[None]
    k = m.shape[1]
    total = m.sum(axis=(1, 2))
    diagonal = np.einsum("bii->bi", m)
    row = m.sum(axis=2)
    column = m.sum(axis=1)

    def ratio(a, b):
        return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b > 0)

    recall = ratio(diagonal, row)
    both_wrong = (row - diagonal) + (column - diagonal)                          # misses plus false alarms of every class
    f1 = np.where(2 * diagonal + both_wrong > 0, 2 * diagonal / np.where(2 * diagonal + both_wrong > 0, 2 * diagonal + both_wrong, 1.0), 0.0)
    # F1 = 2 TP / (2 TP + FP + FN): a class that is never forecast gets F1 = 0 (it is not left out of the average, which would flatter a model that ignores a class)
    idx = np.arange(k)
    cost = ((idx[:, None] - idx[None, :]) ** 2) / (k - 1) ** 2
    expected = row[:, :, None] * column[:, None, :] / np.where(total > 0, total, 1.0)[:, None, None]
    observed_cost = (cost[None] * m).sum(axis=(1, 2))
    expected_cost = (cost[None] * expected).sum(axis=(1, 2))
    kappa = 1 - ratio(observed_cost, expected_cost)
    scores = {"accuracy": ratio(diagonal.sum(axis=1), total), "balanced_accuracy": np.nanmean(recall, axis=1),
              "macro_f1": np.nanmean(f1, axis=1), "weighted_kappa": kappa}
    return scores if stacked else {key: float(value[0]) for key, value in scores.items()}


def multiclass_log_loss(y_true, probabilities, weights=None, eps: float = 1e-15):
    """Mean negative log of the probability given to the true class (lower is better)."""
    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float)[np.arange(len(y_true)), y_true], eps, 1.0)
    w, single = _weights(weights, len(y_true))
    return _out((w * (-np.log(p))[:, None]).sum(axis=0) / w.sum(axis=0), single)


# ================================================================================================ regression
def r_squared(y, prediction, weights=None):
    """1 - (squared error of the model) / (squared error of always predicting the mean of y). Can be negative: worse than the mean."""
    y = np.asarray(y, dtype=float)
    w, single = _weights(weights, len(y))
    prediction = np.asarray(prediction, dtype=float)
    mean = (w * y[:, None]).sum(axis=0) / w.sum(axis=0)
    sse = (w * ((y - prediction) ** 2)[:, None]).sum(axis=0)
    sst = (w * (y[:, None] - mean[None, :]) ** 2).sum(axis=0)
    return _out(1 - sse / sst, single)


def mean_absolute_error(y, prediction, weights=None):
    y = np.asarray(y, dtype=float)
    w, single = _weights(weights, len(y))
    return _out((w * np.abs(y - np.asarray(prediction, dtype=float))[:, None]).sum(axis=0) / w.sum(axis=0), single)


def root_mean_squared_error(y, prediction, weights=None):
    y = np.asarray(y, dtype=float)
    w, single = _weights(weights, len(y))
    mse = (w * ((y - np.asarray(prediction, dtype=float)) ** 2)[:, None]).sum(axis=0) / w.sum(axis=0)
    return _out(np.sqrt(mse), single)


# ================================================================================================ uncertainty
def artist_resample_weights(groups, n_boot: int = 1000, seed: int = 20240518) -> np.ndarray:
    """
    Weights (rows x n_boot) of a bootstrap that re-draws whole groups (artists). Each replicate draws as many groups as there are,
    with replacement; a row's weight is the number of times its group was drawn.
    """
    _, group_of_row = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = int(group_of_row.max()) + 1
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_boot)      # (n_boot, groups)
    return draws[:, group_of_row].T.astype(float)


def interval(replicates, confidence: float = 0.95):
    """Percentile interval of the finite replicates."""
    values = np.asarray(replicates, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    tail = (1 - confidence) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    return (float(low), float(high))


# ================================================================================================ decisions
def gains_table(y, score) -> pd.DataFrame:
    """
    Walk down the songs from the highest score to the lowest. For every share of songs looked at: the share of all positives found
    (gain) and how many times better than a random pick that is (lift = gain / share). Ties are kept together.
    """
    y = _binary(y)
    score = np.asarray(score, dtype=float)
    w = np.ones((len(y), 1))
    positive, negative = _tied_groups(score, y, w)
    positive, negative = positive[::-1, 0], negative[::-1, 0]
    looked = np.cumsum(positive + negative)
    found = np.cumsum(positive)
    share = looked / looked[-1]
    gain = found / found[-1]
    table = pd.DataFrame({"share_flagged": np.r_[0.0, share], "gain": np.r_[0.0, gain]})
    table["lift"] = np.where(table["share_flagged"] > 0, table["gain"] / table["share_flagged"].where(table["share_flagged"] > 0, 1.0), np.nan)
    return table


def decision_table(y, score, value_ratios, shares) -> pd.DataFrame:
    """
    The break-even view of a decision. Promoting one song costs 1 unit. A promoted song that does climb is worth `value_ratio` units.
    For every value ratio and every share of songs promoted (highest scores first):

        hits       promoted songs that did climb
        net        value_ratio * hits - promoted         (in units of one promotion)
        roi        net / promoted                        (return on the money spent; 0 = break-even)

    The break-even precision is 1 / value_ratio: a group of promoted songs pays back when its hit rate is above it.
    """
    y = _binary(y)
    rows = []
    for share in shares:
        flagged = top_share_flags(score, share)
        promoted = int(flagged.sum())
        hits = int((y & flagged).sum())
        for ratio in value_ratios:
            net = ratio * hits - promoted
            rows.append({"value_ratio": float(ratio), "share_promoted": float(share), "promoted": promoted, "hits": hits,
                         "precision": hits / promoted, "net": float(net), "roi": float(net / promoted), "break_even": 1.0 / ratio})
    return pd.DataFrame(rows)
