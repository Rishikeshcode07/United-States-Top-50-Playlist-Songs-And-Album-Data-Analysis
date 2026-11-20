"""
src/ml/benchmark.py  -  the seven questions of Phase 6A and the machine that scores every model on every question the same way.

WHAT THIS FILE IS
    `TASKS` lists the seven questions (what is predicted, for which songs, with which steps of the information ladder). `run_benchmark`
    answers one question: for every step and model it makes out-of-fold forecasts by artist-grouped cross-validation, scores them,
    and puts an artist-level bootstrap interval around every score. `run_forward` does the same with one forward-in-time split.

        question                                                         kind        songs   decided when
        C1  debut in the Top 10                                          yes/no      894     before the song arrives
        C2  climb into the Top 10 within 30 days (debut outside it)      yes/no      632     on the day it arrives
        C3  stay class in the first 30 days (Flash / Short / Sustained)  3 classes   832     on the day it arrives
        R1  popularity in the first week                                 number      878     on the day it arrives
        R2  chart days in the first 30                                   number      832     on the day it arrives
        E1  climb into the Top 10 after three quiet days outside it      yes/no      359     after three days
        E2  stay 15 or more days in the first 30, after three days       yes/no      359     after three days

WHY WE NEED IT
    Fair comparison means identical treatment. Every model gets the same folds (so the same songs are held out), the same repeats,
    the same scores and the same bootstrap draws. A difference between two models is then a difference of the models, not of the
    luck of the split. Because the draws are shared, the DIFFERENCE between two models also gets an interval (`compare`), which is
    much narrower than two separate intervals: when the data are unlucky for one model they are unlucky for the other too.

WHAT HAPPENS WHEN IT RUNS
    For each (step, model) the cross-validation is repeated `n_repeats` times with different artist splits; a song's forecast is the
    average of its out-of-fold forecasts, each made by a model that had never seen the song's artist. The scores are those of the
    averaged forecasts. The baseline model is scored as "no information": AUC 0.5 and average precision equal to the share of
    positives (scoring the baseline's fold-by-fold base rates directly would give an AUC below 0.5, an artefact of the folds).
    The interval of a score is the 2.5% and 97.5% points of the score over `n_boot` re-drawn sets of artists.

EXPECTED OUTCOME
    A table with one row per (step, model): the scores, their intervals and the number of songs. On the real file the full run takes
    about two minutes.

MOTIVE
    One fair procedure for every model and every question, so that the comparison tables can be trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.ml import metrics as M
from src.ml.dataset import STAY_CLASSES
from src.ml.models import SEED, feature_set, make_classifier, make_regressor
from src.ml.validation import (assert_no_group_overlap, cross_validated_predictions, fit_predict, forward_split,
                               repeated_grouped_folds)


# ================================================================================================ the questions
@dataclass(frozen=True)
class Task:
    key: str
    code: str                     # C1, C2 ... as used in the text
    title: str
    kind: str                     # "binary", "multiclass" or "regression"
    target: str                   # column of the modeling table
    eligible: str | None          # boolean column that says which songs can be used (None = all)
    steps: tuple                  # steps of the information ladder this question climbs
    stage: str                    # when the forecast is made
    unit: str = ""                # unit of a regression target


TASKS = {
    "debut_top10": Task("debut_top10", "C1", "Debut in the Top 10", "binary", "y_debut_top10", None,
                        ("release", "timing", "artist"), "before the song arrives"),
    "climb_top10": Task("climb_top10", "C2", "Climb into the Top 10 within 30 days", "binary", "y_climb_top10", "ok_climb",
                        ("release", "timing", "artist", "debut"), "on the day it arrives"),
    "stay_class": Task("stay_class", "C3", "Stay class in the first 30 days", "multiclass", "stay_class", "ok_stay",
                       ("release", "timing", "artist", "debut"), "on the day it arrives"),
    "first_week_popularity": Task("first_week_popularity", "R1", "Popularity in the first week", "regression", "pop_first_week", "ok_pop",
                                  ("release", "timing", "artist", "debut"), "on the day it arrives", "points"),
    "chart_days": Task("chart_days", "R2", "Chart days in the first 30", "regression", "chart_days_30", "ok_stay",
                       ("release", "timing", "artist", "debut"), "on the day it arrives", "days"),
    "climb_after3": Task("climb_after3", "E1", "Climb into the Top 10 after three days", "binary", "y_climb_after3", "ok_early",
                         ("release", "timing", "artist", "debut", "early"), "after three days"),
    "sustained_after3": Task("sustained_after3", "E2", "Stay 15 or more days, after three days", "binary", "y_sustained_after3", "ok_early",
                             ("release", "timing", "artist", "debut", "early"), "after three days"),
}

METRICS = {
    "binary": ["auc", "average_precision", "brier", "log_loss"],
    "multiclass": ["balanced_accuracy", "macro_f1", "weighted_kappa", "accuracy", "log_loss"],
    "regression": ["r2", "mae", "rmse"],
}
MAIN_METRIC = {"binary": "auc", "multiclass": "balanced_accuracy", "regression": "r2"}
HIGHER_IS_BETTER = {"auc": True, "average_precision": True, "brier": False, "log_loss": False, "balanced_accuracy": True, "macro_f1": True,
                    "weighted_kappa": True, "accuracy": True, "r2": True, "mae": False, "rmse": False}
CLASS_LISTS = {"stay_class": STAY_CLASSES}


def task_data(table: pd.DataFrame, task: Task):
    """The eligible rows of the modeling table and the outcome as numbers: 0/1 (binary), 0 ... K-1 (classes) or the value itself."""
    rows = table if task.eligible is None else table[table[task.eligible].astype(bool)]
    rows = rows.reset_index(drop=True)
    if task.kind == "multiclass":
        classes = CLASS_LISTS[task.key]
        y = rows[task.target].map({name: i for i, name in enumerate(classes)})
        if y.isna().any():
            raise ValueError(f"the outcome {task.target} has a class that is not in {classes}")
        return rows, y.to_numpy(dtype=int)
    y = rows[task.target].to_numpy(dtype=float)
    if np.isnan(y).any():
        raise ValueError(f"the outcome {task.target} has missing values among the eligible rows")
    return rows, (y.astype(int) if task.kind == "binary" else y)


def models_for(task: Task, steps=None, forest_steps=("last",)):
    """Which models are run at which step: baseline, the straight-line model and boosting at every step, the forest at the last one."""
    steps = list(task.steps if steps is None else steps)
    linear = "logistic" if task.kind != "regression" else "ridge"
    plan = {}
    for step in steps:
        names = ["baseline", linear, "boosting"]
        if step == steps[-1] and forest_steps:
            names.append("forest")
        plan[step] = names
    return plan


def _make(task: Task, name: str, seed: int):
    return make_regressor(name, seed=seed) if task.kind == "regression" else make_classifier(name, seed=seed)


# ================================================================================================ scoring
def baseline_overrides(kind: str, model: str, y):
    """
    The baseline is scored as "knows nothing": for a yes/no question every song gets the same score (AUC 0.5), for a class question the
    most common class is guessed for every song. Scoring the baseline's own fold-by-fold forecasts instead would give an AUC below 0.5
    and a balanced accuracy below one third, an artefact of the folds (a fold that holds more positives was trained on fewer).
    Returns (rank_pred, guess); both None for every model except the baseline.
    """
    if model != "baseline":
        return None, None
    y = np.asarray(y)
    if kind == "binary":
        return np.full(len(y), float(np.mean(y))), None
    if kind == "multiclass":
        return None, np.full(len(y), int(np.bincount(y).argmax()))
    return None, None


def score_predictions(kind: str, y, pred, weights=None, rank_pred=None, guess=None) -> dict:
    """
    All scores of one set of forecasts. `weights` is None (one number per score) or a matrix (one number per column).
    `pred` is the probability of "yes" (binary), the matrix of class probabilities (multiclass) or the predicted value (regression).
    `rank_pred` replaces `pred` in the scores that only use the ORDER of the forecasts (AUC, average precision); `guess` replaces the
    most probable class in the scores of a class question.
    """
    if kind == "binary":
        ranking = pred if rank_pred is None else rank_pred
        return {"auc": M.roc_auc(y, ranking, weights), "average_precision": M.average_precision(y, ranking, weights),
                "brier": M.brier_score(y, pred, weights), "log_loss": M.log_loss(y, pred, weights)}
    if kind == "multiclass":
        k = pred.shape[1]
        guess = pred.argmax(axis=1) if guess is None else guess
        matrix = M.confusion_matrix(y, guess, k, weights)
        scores = M.matrix_scores(matrix)
        scores["log_loss"] = M.multiclass_log_loss(y, pred, weights)
        return scores
    return {"r2": M.r_squared(y, pred, weights), "mae": M.mean_absolute_error(y, pred, weights),
            "rmse": M.root_mean_squared_error(y, pred, weights)}


@dataclass
class Benchmark:
    """The result of `run_benchmark`."""
    task: Task
    frame: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    folds: list
    weights: np.ndarray
    predictions: dict = field(default_factory=dict)          # (step, model) -> averaged out-of-fold forecasts
    replicates: dict = field(default_factory=dict)           # (step, model) -> {metric: array over bootstrap draws}
    table: pd.DataFrame | None = None

    def score(self, step: str, model: str, metric: str) -> float:
        row = self.table[(self.table["step"] == step) & (self.table["model"] == model)]
        return float(row[metric].iloc[0])

    def interval(self, step: str, model: str, metric: str):
        row = self.table[(self.table["step"] == step) & (self.table["model"] == model)]
        return float(row[f"{metric}_low"].iloc[0]), float(row[f"{metric}_high"].iloc[0])


def run_benchmark(table: pd.DataFrame, task_key: str, plan: dict | None = None, n_splits: int = 5, n_repeats: int = 3, n_boot: int = 1000,
                  seed: int = SEED, fold_seed: int = 0, exclude_artists=(), model_seed: int = SEED) -> Benchmark:
    """
    Run one question. `plan` maps a ladder step to the list of models to run there (default: `models_for(task)`). `exclude_artists` leaves
    out all songs of the named primary artists (used by the leave-out stress test). Returns a `Benchmark`.
    """
    task = TASKS[task_key]
    data = table[~table["primary_artist"].isin(list(exclude_artists))] if len(exclude_artists) else table
    frame, y = task_data(data, task)
    groups = frame["primary_artist"].to_numpy()
    folds = repeated_grouped_folds(groups, n_splits, n_repeats, seed=fold_seed)
    for fold_assignment in folds:
        assert_no_group_overlap(groups, fold_assignment)
    weights = M.artist_resample_weights(groups, n_boot, seed=seed)
    plan = models_for(task) if plan is None else plan
    bench = Benchmark(task, frame, y, groups, folds, weights)
    rows = []
    for step, names in plan.items():
        columns = feature_set(step, task.key)
        for name in names:
            raw = cross_validated_predictions(lambda n=name: _make(task, n, model_seed), frame[columns], y, folds,
                                              kind="value" if task.kind == "regression" else "proba")
            pred = raw.mean(axis=0)                                           # the average of the repeats: one forecast per song
            rank_pred, guess = baseline_overrides(task.kind, name, y)
            point = score_predictions(task.kind, y, pred, None, rank_pred, guess)
            boot = score_predictions(task.kind, y, pred, weights, rank_pred, guess)
            bench.predictions[(step, name)] = pred
            bench.replicates[(step, name)] = boot
            row = {"task": task.key, "step": step, "model": name, "n": len(y)}
            for metric in METRICS[task.kind]:
                low, high = M.interval(boot[metric])
                row[metric], row[f"{metric}_low"], row[f"{metric}_high"] = float(point[metric]), low, high
            rows.append(row)
    bench.table = pd.DataFrame(rows)
    return bench


def compare(bench: Benchmark, a: tuple, b: tuple, metric: str | None = None, confidence: float = 0.95):
    """
    Difference of a score between two (step, model) pairs, `a` minus `b`, with an interval from the SAME artist draws:
    returns (difference, low, high). For a "lower is better" score the sign is kept (a negative difference means that `a` is better).
    """
    metric = metric or MAIN_METRIC[bench.task.kind]
    diff = np.asarray(bench.replicates[a][metric]) - np.asarray(bench.replicates[b][metric])
    point = bench.score(*a, metric) - bench.score(*b, metric)
    low, high = M.interval(diff, confidence)
    return float(point), low, high


def score_rows(bench: Benchmark, key: tuple, rows, n_boot: int = 1000, seed: int = SEED, metric_set=None) -> dict:
    """Scores (with artist-bootstrap intervals) of the stored forecasts of `key` for a subset of rows (a boolean mask or positions)."""
    rows = np.asarray(rows)
    index = np.flatnonzero(rows) if rows.dtype == bool else rows
    pred = bench.predictions[key][index]
    y = bench.y[index]
    weights = M.artist_resample_weights(bench.groups[index], n_boot, seed=seed)
    rank_pred, guess = baseline_overrides(bench.task.kind, key[1], y)
    point = score_predictions(bench.task.kind, y, pred, None, rank_pred, guess)
    boot = score_predictions(bench.task.kind, y, pred, weights, rank_pred, guess)
    out = {}
    for metric in (metric_set or METRICS[bench.task.kind]):
        low, high = M.interval(boot[metric])
        out[metric] = (float(point[metric]), low, high)
    return out


def score_variant(bench: Benchmark, step: str, make_model, metric: str | None = None):
    """
    The main score of a model that is NOT in the benchmark table (a different setting, say), made with the benchmark's own folds and scored with
    the benchmark's own artist draws, so that it can be set beside the table. `make_model` returns a new, unfitted model. Returns (score, low, high).
    """
    task = bench.task
    metric = metric or MAIN_METRIC[task.kind]
    raw = cross_validated_predictions(make_model, bench.frame[feature_set(step, task.key)], bench.y, bench.folds,
                                      kind="value" if task.kind == "regression" else "proba")
    pred = raw.mean(axis=0)
    point = score_predictions(task.kind, bench.y, pred)[metric]
    low, high = M.interval(score_predictions(task.kind, bench.y, pred, bench.weights)[metric])
    return float(point), low, high


# ================================================================================================ forward in time
def run_forward(table: pd.DataFrame, task_key: str, cutoff: int, gap: int = 30, plan: dict | None = None, n_boot: int = 1000,
                seed: int = SEED, model_seed: int = SEED) -> pd.DataFrame:
    """
    Train on the songs that arrived before `cutoff - gap`, forecast the songs that arrived at or after `cutoff`. Returns one row per
    (step, model) with the scores on the later songs and their artist-bootstrap intervals, and the sizes of the two parts.
    """
    task = TASKS[task_key]
    frame, y = task_data(table, task)
    train, test = forward_split(frame["entry_snapshot"].to_numpy(), cutoff, gap)
    weights = M.artist_resample_weights(frame["primary_artist"].to_numpy()[test], n_boot, seed=seed)
    plan = models_for(task) if plan is None else plan
    rows = []
    for step, names in plan.items():
        columns = feature_set(step, task.key)
        for name in names:
            pred = fit_predict(lambda n=name: _make(task, n, model_seed), frame[columns], y, train, test,
                               kind="value" if task.kind == "regression" else "proba")
            rank_pred, guess = baseline_overrides(task.kind, name, y[test])
            point = score_predictions(task.kind, y[test], pred, None, rank_pred, guess)
            boot = score_predictions(task.kind, y[test], pred, weights, rank_pred, guess)
            row = {"task": task_key, "step": step, "model": name, "n_train": len(train), "n_test": len(test)}
            for metric in METRICS[task.kind]:
                low, high = M.interval(boot[metric])
                row[metric], row[f"{metric}_low"], row[f"{metric}_high"] = float(point[metric]), low, high
            rows.append(row)
    return pd.DataFrame(rows)
