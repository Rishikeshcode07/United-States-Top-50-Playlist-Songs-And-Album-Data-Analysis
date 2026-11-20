"""
src/ml/models.py  -  the models of Phase 6A, with their settings fixed in advance, and the feature sets they are given.

WHAT THIS FILE IS
    Factory functions that return a NEW, unfitted model every time they are called:

        classifiers   baseline (always the share of the training songs), logistic regression, random forest, gradient boosting
        regressors    baseline (always the training mean), ridge regression, random forest, gradient boosting

    Every model is a small pipeline: (1) the skewed columns are put on a log scale, (2) missing values are filled with the training
    median and a "was missing" column is added, (3) the columns are standardised, (4) the model. All of it is learned from the
    training rows only, because it sits INSIDE the pipeline that is cloned for every fold.

    `feature_set(task, step)` lists the columns of a task at a step of the "ladder": release attributes only, then adding the
    timing, then the artist's record, then the debut-day chart information.

WHY WE NEED IT
    * The settings are chosen BEFORE looking at any score and are never tuned on the test folds: tuning on the data you test on
      makes every score optimistic. A sensitivity check in the notebook shows that other reasonable settings tell the same story.
    * Three model families with very different habits (a straight line, many deep-ish trees averaged, many shallow trees added up)
      show whether the answer depends on the model. If a simple logistic regression and a gradient-boosting machine agree, the
      signal is in the data and not in the method.
    * The baseline is the honest zero: a model that knows nothing but the base rate. A model that does not beat it has learned nothing.

WHAT HAPPENS WHEN IT RUNS
    Nothing by itself. The notebook calls `make_classifier("boosting")` and so on. All random choices use `SEED`, so the same data
    gives the same model.

EXPECTED OUTCOME
    Models that fit in well under a second each on 900 rows.

MOTIVE
    One place for every model setting, so that the notebook, the tests and a later Streamlit page can not disagree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.ml.dataset import ARTIST_FEATURES, DEBUT_FEATURES, EARLY_FEATURES, RELEASE_FEATURES, TIMING_FEATURES

SEED = 20240518

# Columns that are strongly right-skewed (a few huge values): they are put on a log scale for the straight-line models.
LOG_COLUMNS = ["duration_min", "total_tracks", "artist_prior_slot_days", "artist_slot_days_recent", "artist_prior_songs",
               "artist_prior_top10_days"]

CLASSIFIER_NAMES = ["baseline", "logistic", "forest", "boosting"]
REGRESSOR_NAMES = ["baseline", "ridge", "forest", "boosting"]
MODEL_LABELS = {"baseline": "Baseline", "logistic": "Logistic regression", "ridge": "Ridge regression", "forest": "Random forest",
                "boosting": "Gradient boosting"}

# The ladder: what the model is allowed to know, step by step.
LADDER = ["release", "timing", "artist", "debut", "early"]
LADDER_LABELS = {"release": "Release details", "timing": "+ Timing", "artist": "+ Artist's record", "debut": "+ Debut-day chart",
                 "early": "+ First three days"}
_BLOCK_COLUMNS = {"release": RELEASE_FEATURES, "timing": TIMING_FEATURES, "artist": ARTIST_FEATURES, "debut": DEBUT_FEATURES,
                  "early": EARLY_FEATURES}
# The first-week popularity is almost the same number as the popularity on the first day (rank correlation 0.9), so for that task the
# popularity of the first day is not an input. Every other task may use it.
EXCLUDED_BY_TASK = {"first_week_popularity": ["entry_pop"]}


def feature_set(step: str, task: str | None = None) -> list:
    """The columns a model may use at `step` of the ladder (a step includes all earlier steps)."""
    if step not in LADDER:
        raise ValueError(f"step must be one of {LADDER}")
    columns = [column for block in LADDER[: LADDER.index(step) + 1] for column in _BLOCK_COLUMNS[block]]
    return [column for column in columns if column not in EXCLUDED_BY_TASK.get(task, [])]


# ================================================================================================ the pipeline
def _log_scale(frame, columns):
    """log(1 + x) of the listed columns that are present (a module-level function, so that pipelines can be copied and saved)."""
    frame = pd.DataFrame(frame).copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = np.log1p(frame[column].astype(float))
    return frame


def _preprocess(scale: bool = True):
    steps = [("log", FunctionTransformer(_log_scale, kw_args={"columns": LOG_COLUMNS})),
             ("fill", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


# ================================================================================================ classifiers
BOOSTING_SETTINGS = {"max_depth": 3, "learning_rate": 0.05, "max_iter": 150, "min_samples_leaf": 20, "l2_regularization": 1.0}


def make_classifier(name: str, seed: int = SEED, logistic_c: float = 0.5, boosting: dict | None = None):
    """
    A new, unfitted classifier. The settings are fixed on purpose (see the top of the file). `logistic_c` and `boosting` (a dictionary that
    replaces some of BOOSTING_SETTINGS) exist only for the sensitivity check of the notebook: the real runs never pass them.
    """
    if name == "baseline":
        return DummyClassifier(strategy="prior")
    if name == "logistic":
        return Pipeline(_preprocess(True) + [("model", LogisticRegression(C=logistic_c, max_iter=5000))])
    if name == "forest":
        return Pipeline(_preprocess(False) + [("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                                                                random_state=seed, n_jobs=1))])
    if name == "boosting":
        return Pipeline(_preprocess(False) + [("model", HistGradientBoostingClassifier(**{**BOOSTING_SETTINGS, **(boosting or {})}, random_state=seed))])
    raise ValueError(f"unknown classifier {name!r}")


def make_regressor(name: str, seed: int = SEED, ridge_alpha: float = 10.0, boosting: dict | None = None):
    """A new, unfitted regressor (`ridge_alpha` and `boosting` as `logistic_c` and `boosting` of `make_classifier`: only for the sensitivity check)."""
    if name == "baseline":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Pipeline(_preprocess(True) + [("model", Ridge(alpha=ridge_alpha))])
    if name == "forest":
        return Pipeline(_preprocess(False) + [("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                                                               random_state=seed, n_jobs=1))])
    if name == "boosting":
        return Pipeline(_preprocess(False) + [("model", HistGradientBoostingRegressor(**{**BOOSTING_SETTINGS, **(boosting or {})}, random_state=seed))])
    raise ValueError(f"unknown regressor {name!r}")
