"""
src/ml/explain.py  -  looking inside a model: which inputs does it lean on, and how does its answer change as one input changes?

WHAT THIS FILE IS
    Two tools that work for any model, however complicated.

        permutation_importance   scramble one input (or one GROUP of inputs) among the held-out songs and see how much the score falls
        partial_dependence       set one input to each value of a grid for ALL songs and average the model's answer

WHY WE NEED IT
    A gradient-boosting machine has no coefficients to read. Permutation importance asks a simple question of the finished model:
    "how much worse are your forecasts for songs you have not seen if the information in this input is destroyed?" The scores are
    measured on out-of-fold songs, so an input that only helps the model memorise the training artists does not look important.
    Inputs that carry the same information (an artist's total chart days and the number of the artist's songs) are scrambled TOGETHER
    as one group; otherwise the model simply uses the twin and the importance of each looks small.

    Partial dependence draws the shape of the relationship the model learned (for instance: the chance of a Top-10 debut against the
    length of the song).

    Both describe what the MODEL does, which is a statement about the data only as far as the model is any good. Neither says that
    changing the input would change the outcome: that needs a different kind of evidence.

WHAT HAPPENS WHEN IT RUNS
    `permutation_importance` repeats the grouped cross-validation of the notebook (the same folds), and inside every fold scrambles
    each group several times with a seeded generator. The scores of all folds are pooled before the score is calculated (a single fold
    holds only a handful of positives). Importance = score of the intact predictions minus score of the scrambled predictions.

EXPECTED OUTCOME
    An input the model ignores gets an importance near zero (the tests check that it does not exceed what chance alone produces).

MOTIVE
    A model that can be questioned can be trusted more, or distrusted for good reasons.

    `build_design` belongs to the same family: it turns a few chosen columns into the table of inputs for the regressions that EXPLAIN
    (odds ratios, coefficients with honest intervals): numbers put on one scale (one standard deviation), missing values filled and
    flagged, yes/no columns kept as they are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.validation import _predict


def permutation_importance(make_model, X, y, fold_sets, groups_of_columns: dict, metric, kind: str = "proba",
                           higher_is_better: bool = True, n_permutations: int = 5, seed: int = 20240518) -> pd.DataFrame:
    """
    `groups_of_columns` maps a name to the list of columns that are scrambled together. `metric(y, predictions)` scores pooled
    out-of-fold predictions. Returns one row per group: importance (mean fall of the score), the standard deviation over the repeats
    and permutations, and the score of the intact model in the last column `intact_score`.
    """
    y = np.asarray(y)
    names = list(groups_of_columns)
    n_rep = len(fold_sets)
    intact = None
    scrambled = {}
    for r, folds in enumerate(fold_sets):
        for fold in np.unique(folds):
            test = np.flatnonzero(folds == fold)
            train = np.flatnonzero(folds != fold)
            model = make_model()
            model.fit(X.iloc[train], y[train])
            base = _predict(model, X.iloc[test], kind)
            if intact is None:
                intact = np.full((n_rep, len(y)) + base.shape[1:], np.nan)
            intact[r, test] = base
            for g_index, name in enumerate(names):
                columns = groups_of_columns[name]
                for p in range(n_permutations):
                    rng = np.random.default_rng([seed, r, int(fold), g_index, p])
                    shuffled = X.iloc[test].copy()
                    order = rng.permutation(len(test))
                    shuffled[columns] = X.iloc[test][columns].to_numpy()[order]           # the columns move together
                    out = scrambled.setdefault((name, p), np.full((n_rep, len(y)) + base.shape[1:], np.nan))
                    out[r, test] = _predict(model, shuffled, kind)
    intact_scores = np.array([metric(y, intact[r]) for r in range(n_rep)])
    rows = []
    for name in names:
        falls = []
        for p in range(n_permutations):
            for r in range(n_rep):
                score = metric(y, scrambled[(name, p)][r])
                falls.append(intact_scores[r] - score if higher_is_better else score - intact_scores[r])
        rows.append({"group": name, "importance": float(np.mean(falls)), "importance_sd": float(np.std(falls, ddof=1)) if len(falls) > 1 else 0.0})
    table = pd.DataFrame(rows)
    table["intact_score"] = float(intact_scores.mean())
    return table


def partial_dependence(models, X, column: str, grid, kind: str = "proba", class_index: int = 1) -> np.ndarray:
    """
    Average prediction when `column` is set to every value of `grid` for all rows of X, one row of the result per model in `models`
    (fitted models). For a classifier the chance of class `class_index`, for a regressor the predicted value.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    out = np.zeros((len(models), len(grid)))
    for m_index, model in enumerate(models):
        for g_index, value in enumerate(grid):
            changed = X.copy()
            changed[column] = value
            prediction = _predict(model, changed, kind)
            if prediction.ndim == 2:
                prediction = prediction[:, class_index]
            out[m_index, g_index] = float(np.mean(prediction))
    return out


def build_design(frame, numeric: dict, binary: dict, flag_from: int = 1):
    """
    The input table of an explanatory regression. `numeric` maps a label to (column, transform) with transform "raw" or "log1p"; each such
    column is transformed, missing values are replaced by the median and flagged in an extra column "<label> (missing)", and the column is
    standardised (mean 0, standard deviation 1), so that its coefficient means "per one standard deviation". The flag is only added when
    at least `flag_from` values are missing: a flag for two or three songs would let the regression fit exactly those songs (and, for a
    yes/no outcome, can make it fail to converge), so for so few the median fill is used without a flag. `binary` maps a label to a
    0/1 (or True/False) column, which is kept as it is. A column of ones called "Intercept" comes first.
    Returns (design, scale): the design table and a table with the mean and the standard deviation of every numeric input on the
    transformed scale (to turn "per standard deviation" back into units).
    """
    out = {"Intercept": np.ones(len(frame))}
    flags = {}
    scale = []
    for label, (column, transform) in numeric.items():
        if transform not in ("raw", "log1p"):
            raise ValueError("transform must be 'raw' or 'log1p'")
        x = frame[column].astype(float).to_numpy()
        if transform == "log1p":
            x = np.log1p(x)
        missing = np.isnan(x)
        if missing.any():
            x = np.where(missing, np.nanmedian(x), x)
            if missing.sum() >= flag_from:
                flags[f"{label} (missing)"] = missing.astype(float)
        mean, sd = float(x.mean()), float(x.std(ddof=1))
        if sd == 0:
            raise ValueError(f"the column {column} is constant")
        out[label] = (x - mean) / sd
        scale.append({"input": label, "column": column, "transform": transform, "mean": mean, "sd": sd})
    for label, column in binary.items():
        out[label] = frame[column].astype(float).to_numpy()
    out.update(flags)
    return pd.DataFrame(out, index=frame.index), pd.DataFrame(scale)
