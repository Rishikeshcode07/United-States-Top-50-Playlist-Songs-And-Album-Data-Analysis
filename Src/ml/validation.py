"""
src/ml/validation.py  -  how a model is tested fairly: by artist, repeated, and forward in time.

WHAT THIS FILE IS
    Three tools.

        grouped_folds / repeated_grouped_folds   split the songs into folds so that ALL songs of one artist land in the same fold
        forward_split                            a split by date: train on earlier arrivals, test on later ones, with a gap
        cross_validated_predictions              train on the other folds, predict each held-out fold; collect every prediction

WHY WE NEED IT
    Songs of one artist look alike: the same voice, the same fan base, often the same weekday of release. If a random split puts
    some songs of an artist in the training part and others in the test part, the model can score well by recognising the artist
    instead of learning anything that carries over to a new one. Taylor Swift alone has 92 of the 894 songs. Splitting by artist
    removes that shortcut: the model is always tested on artists it has never seen.

    A second question is different: "does a model trained on the past work on the future?" `forward_split` answers it. The gap
    (the "purge") matters: a song that arrived 10 days before the cut-off has an outcome that is decided AFTER the cut-off
    (the first 30 days), so it must not be in the training part.

WHAT HAPPENS WHEN IT RUNS
    Folds are made without randomness inside sklearn: the groups are shuffled with a seed, the biggest groups are placed first,
    and every group goes to the currently smallest fold. So the folds are balanced in size and the same seed gives the same folds
    on every machine and every library version. (sklearn's own GroupKFold only learned to shuffle in version 1.6.)
    `cross_validated_predictions` clones the model for every fold, so nothing learned in one fold can leak into another.

EXPECTED OUTCOME
    Every row gets one prediction per repeat, made by a model that never saw that row or any song of the same artist.

MOTIVE
    A score is only worth reporting when it was earned on data the model could not have memorised.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone


# ================================================================================================ folds
def grouped_folds(groups, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """
    The fold number (0 ... n_splits-1) of every row, such that all rows of one group share a fold. Groups are shuffled with `seed`,
    the largest are placed first, and each goes to the fold that is smallest at that moment (ties are broken at random).
    """
    labels = np.asarray(groups)
    _, group_of_row, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    n_groups = len(sizes)
    if n_splits < 2 or n_splits > n_groups:
        raise ValueError(f"n_splits must be between 2 and the number of groups ({n_groups}), not {n_splits}")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_groups)
    order = shuffled[np.argsort(-sizes[shuffled], kind="stable")]
    fold_of_group = np.empty(n_groups, dtype=int)
    fold_sizes = np.zeros(n_splits)
    for group in order:
        smallest = np.flatnonzero(fold_sizes == fold_sizes.min())
        fold = int(rng.choice(smallest))
        fold_of_group[group] = fold
        fold_sizes[fold] += sizes[group]
    return fold_of_group[group_of_row]


def repeated_grouped_folds(groups, n_splits: int = 5, n_repeats: int = 3, seed: int = 0) -> list:
    """`n_repeats` different grouped fold assignments (seeds seed, seed+1, ...). Repeating shows how much the score depends on the split."""
    return [grouped_folds(groups, n_splits, seed + repeat) for repeat in range(n_repeats)]


def random_folds(n_rows: int, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Plain random folds that ignore the artist. Used only to SHOW how much a careless split overstates the score."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n_rows) % n_splits)


def forward_split(entry_snapshot, cutoff: int, gap: int):
    """
    Train rows: arrivals before `cutoff - gap`. Test rows: arrivals at or after `cutoff`. Rows in between (the gap) are used by
    neither part, because their outcome window reaches past the cut-off. Returns (train_rows, test_rows) as integer positions.
    """
    time = np.asarray(entry_snapshot)
    train = np.flatnonzero(time < cutoff - gap)
    test = np.flatnonzero(time >= cutoff)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("the cut-off leaves no rows in the training part or in the test part")
    return train, test


# ================================================================================================ predictions
def _predict(model, X, kind):
    if kind == "proba":
        probabilities = model.predict_proba(X)
        return probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities
    if kind == "value":
        return model.predict(X)
    raise ValueError("kind must be 'proba' or 'value'")


def cross_validated_predictions(make_model, X, y, fold_sets, kind: str = "proba"):
    """
    Out-of-fold predictions. For every repeat (one entry of `fold_sets`) and every fold: fit a fresh model on the other folds and
    predict the held-out fold. Returns an array of shape (repeats, rows), or (repeats, rows, classes) for a model with more than two
    classes. `kind` is "proba" (the probability of the positive class, or of every class) or "value" (a number, for regression).
    `X` is a DataFrame, `y` an array; `make_model` is a function without arguments that returns a new, unfitted model.
    """
    y = np.asarray(y)
    out = None
    for r, folds in enumerate(fold_sets):
        for fold in np.unique(folds):
            test = np.flatnonzero(folds == fold)
            train = np.flatnonzero(folds != fold)
            model = make_model()
            model.fit(X.iloc[train], y[train])
            part = _predict(model, X.iloc[test], kind)
            if out is None:
                out = np.full((len(fold_sets), len(y)) + part.shape[1:], np.nan)
            out[r, test] = part
    return out


def fit_predict(make_model, X, y, train_rows, test_rows, kind: str = "proba"):
    """One fit on `train_rows`, predictions for `test_rows` (used by the forward-in-time test)."""
    y = np.asarray(y)
    model = make_model()
    model.fit(X.iloc[train_rows], y[train_rows])
    return _predict(model, X.iloc[test_rows], kind)


def assert_no_group_overlap(groups, folds) -> None:
    """Raise when one group (artist) appears in two folds. The tests and the notebook call this on every fold assignment."""
    labels = np.asarray(groups)
    for group in np.unique(labels):
        if len(np.unique(folds[labels == group])) > 1:
            raise AssertionError(f"the group {group!r} is spread over several folds")
