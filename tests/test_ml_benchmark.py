"""
tests/test_ml_benchmark.py  -  checks of the benchmark machine (src/ml/benchmark.py) on a made-up modeling table with a planted signal.

WHAT THIS FILE IS
    A synthetic table with the same columns as the real modeling table, in which the debut position decides every outcome and the
    release attributes decide nothing. The machine must find the signal where it is (and only there), score the baseline as "knows
    nothing", give every score an interval that covers it, be repeatable, leave out the artists it is told to leave out, and compare two
    models on the same draws.

WHY WE NEED IT
    Every table and chart of the notebook comes out of this machine. With a planted signal we know what the right answer is: AUC well above
    one half where the signal is, near one half where it is not, a baseline of exactly one half, and a gap between two models that the
    interval around the difference confirms.

WHAT HAPPENS WHEN IT RUNS
    Small seeded data (240 songs, 80 artists), 3 repeats of 5 folds, a few hundred bootstrap draws: a few seconds.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Trust in the numbers of a model comparison starts with a comparison whose answer is known in advance.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.ml import metrics as M
from src.ml.benchmark import (MAIN_METRIC, METRICS, TASKS, baseline_overrides, compare, models_for, run_benchmark, run_forward,
                              score_predictions, score_rows, score_variant, task_data)
from src.ml.dataset import (ARTIST_FEATURES, DEBUT_FEATURES, EARLY_FEATURES, RELEASE_FEATURES, STAY_CLASSES, TIMING_FEATURES)
from src.ml.models import LOG_COLUMNS, make_classifier


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(0)
    n = 240
    artist = rng.integers(0, 80, n)
    frame = pd.DataFrame({"track_id": np.arange(n), "primary_artist": [f"artist {a:02d}" for a in artist], "entry_snapshot": np.sort(rng.integers(1, 500, n))})
    for column in RELEASE_FEATURES + TIMING_FEATURES + ARTIST_FEATURES + DEBUT_FEATURES + EARLY_FEATURES:
        frame[column] = np.abs(rng.normal(size=n)) if column in LOG_COLUMNS else rng.normal(size=n)       # log-scaled columns must not be negative
    frame["entry_rank"] = rng.integers(1, 51, n).astype(float)
    signal = -(frame["entry_rank"] - 25) / 10 + rng.normal(0, 0.8, n)              # a better debut position goes with every good outcome
    frame["y_debut_top10"] = rng.integers(0, 2, n)                                  # nothing predicts this one
    frame["ok_climb"] = True
    frame["y_climb_top10"] = (signal > 0.6).astype(float)
    frame["ok_stay"] = True
    days = np.clip(np.round(15 + 6 * signal + rng.normal(0, 2, n)), 1, 30)
    frame["chart_days_30"] = days
    frame["stay_class"] = np.where(days <= 3, STAY_CLASSES[0], np.where(days <= 14, STAY_CLASSES[1], STAY_CLASSES[2]))
    frame["ok_pop"] = True
    frame["pop_first_week"] = 80 + 2 * signal + rng.normal(0, 3, n)
    frame["ok_early"] = True
    frame["y_climb_after3"] = (signal > 0.6).astype(float)
    frame["y_sustained_after3"] = (signal > 0).astype(float)
    return frame


FAST = dict(n_repeats=2, n_boot=200)


# ================================================================================ the plan and the data
def test_every_task_names_columns_of_the_modeling_table_and_known_steps():
    for task in TASKS.values():
        assert task.kind in METRICS and task.steps[0] == "release" and task.code
    assert {task.code for task in TASKS.values()} == {"C1", "C2", "C3", "R1", "R2", "E1", "E2"}
    assert TASKS["climb_after3"].steps[-1] == "early" and "early" not in TASKS["climb_top10"].steps


def test_models_for_runs_the_baseline_everywhere_and_the_forest_only_at_the_last_step():
    plan = models_for(TASKS["climb_top10"])
    assert list(plan) == list(TASKS["climb_top10"].steps)
    assert all(names[0] == "baseline" and "boosting" in names for names in plan.values())
    assert [step for step, names in plan.items() if "forest" in names] == [TASKS["climb_top10"].steps[-1]]
    assert "logistic" in plan["release"] and "ridge" not in plan["release"]
    assert "ridge" in models_for(TASKS["chart_days"])["release"]


def test_task_data_returns_numbers_for_every_kind(table):
    rows, y = task_data(table, TASKS["climb_top10"])
    assert set(np.unique(y)) <= {0, 1} and y.dtype.kind == "i" and len(rows) == len(y)
    rows, y = task_data(table, TASKS["stay_class"])
    assert set(np.unique(y)) <= {0, 1, 2}
    rows, y = task_data(table, TASKS["chart_days"])
    assert y.dtype.kind == "f"


def test_task_data_keeps_only_the_eligible_songs_and_refuses_gaps(table):
    damaged = table.copy()
    damaged["ok_climb"] = damaged["track_id"] % 2 == 0
    rows, y = task_data(damaged, TASKS["climb_top10"])
    assert len(rows) == 120
    damaged.loc[damaged["ok_climb"], "y_climb_top10"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        task_data(damaged, TASKS["climb_top10"])
    odd = table.copy()
    odd.loc[0, "stay_class"] = "Forever"
    with pytest.raises(ValueError, match="class"):
        task_data(odd, TASKS["stay_class"])


# ================================================================================ the baseline is "knows nothing"
def test_baseline_overrides():
    y = np.array([1, 0, 0, 0])
    rank, guess = baseline_overrides("binary", "baseline", y)
    assert (rank == 0.25).all() and guess is None
    assert baseline_overrides("binary", "logistic", y) == (None, None)
    rank, guess = baseline_overrides("multiclass", "baseline", np.array([0, 1, 1, 2]))
    assert rank is None and (guess == 1).all()
    assert baseline_overrides("regression", "baseline", y) == (None, None)


# ================================================================================ finding the planted signal
@pytest.fixture(scope="module")
def climb(table):
    plan = {"release": ["baseline", "logistic"], "debut": ["baseline", "logistic", "boosting"]}
    return run_benchmark(table, "climb_top10", plan=plan, **FAST)


def test_the_table_has_one_row_per_step_and_model_with_scores_and_intervals(climb):
    t = climb.table
    assert len(t) == 5 and t["n"].eq(240).all()
    for metric in METRICS["binary"]:
        assert {metric, f"{metric}_low", f"{metric}_high"} <= set(t.columns)
        assert ((t[f"{metric}_low"] <= t[metric] + 1e-9) & (t[metric] <= t[f"{metric}_high"] + 1e-9)).all()


def test_the_baseline_scores_exactly_as_knowing_nothing(climb):
    for step in ("release", "debut"):
        row = climb.table[(climb.table["step"] == step) & (climb.table["model"] == "baseline")].iloc[0]
        assert row["auc"] == 0.5 and row["auc_low"] == 0.5 and row["auc_high"] == 0.5
        assert row["average_precision"] == pytest.approx(climb.y.mean())


def test_the_signal_is_found_where_it_is_and_not_where_it_is_not(climb):
    assert climb.score("debut", "logistic", "auc") > 0.8 and climb.interval("debut", "logistic", "auc")[0] > 0.7
    assert climb.score("debut", "boosting", "auc") > 0.75
    low, high = climb.interval("release", "logistic", "auc")
    assert low < 0.5 < high                                                       # the release attributes carry nothing: chance is inside the interval


def test_the_predictions_are_one_forecast_per_song(climb):
    pred = climb.predictions[("debut", "logistic")]
    assert pred.shape == (240,) and ((pred > 0) & (pred < 1)).all()


def test_comparing_two_models_uses_the_same_draws(climb):
    diff, low, high = compare(climb, ("debut", "logistic"), ("release", "logistic"))
    assert diff > 0.2 and low > 0.1                                               # the interval around the DIFFERENCE excludes zero
    separate = climb.interval("debut", "logistic", "auc")[1] - climb.interval("release", "logistic", "auc")[0]
    assert high - low < separate                                                 # narrower than the two separate intervals taken together
    same, low0, high0 = compare(climb, ("debut", "logistic"), ("debut", "logistic"))
    assert same == 0 and low0 == 0 and high0 == 0


def test_the_run_is_repeatable_and_the_fold_seed_changes_the_split(table):
    plan = {"debut": ["logistic"]}
    a = run_benchmark(table, "climb_top10", plan=plan, **FAST)
    b = run_benchmark(table, "climb_top10", plan=plan, **FAST)
    c = run_benchmark(table, "climb_top10", plan=plan, fold_seed=7, **FAST)
    assert a.table.equals(b.table)
    assert a.table["auc"].iloc[0] != c.table["auc"].iloc[0]
    assert abs(a.table["auc"].iloc[0] - c.table["auc"].iloc[0]) < 0.06


def test_excluded_artists_leave_the_data(table):
    gone = ["artist 03", "artist 07"]
    bench = run_benchmark(table, "climb_top10", plan={"debut": ["logistic"]}, exclude_artists=gone, **FAST)
    assert not set(gone) & set(bench.groups) and len(bench.y) == (~table["primary_artist"].isin(gone)).sum()
    assert bench.table["n"].iloc[0] == len(bench.y)


def test_scores_of_a_subset_of_songs(climb):
    early = climb.frame["entry_snapshot"].to_numpy() < 250
    scores = score_rows(climb, ("debut", "logistic"), early, n_boot=200)
    assert set(scores) == set(METRICS["binary"])
    point, low, high = scores["auc"]
    assert low <= point <= high and point > 0.7


# ================================================================================ the other kinds of questions
def test_class_question_baseline_has_balanced_accuracy_one_third_and_the_signal_is_found(table):
    bench = run_benchmark(table, "stay_class", plan={"debut": ["baseline", "logistic", "boosting"]}, **FAST)
    base = bench.table[bench.table["model"] == "baseline"].iloc[0]
    assert base["balanced_accuracy"] == pytest.approx(1 / 3) and base["weighted_kappa"] == pytest.approx(0.0)
    assert bench.score("debut", "logistic", "balanced_accuracy") > 0.5
    assert bench.predictions[("debut", "logistic")].shape == (240, 3)


def test_number_question_finds_the_signal_and_the_baseline_is_about_zero(table):
    bench = run_benchmark(table, "chart_days", plan={"debut": ["baseline", "ridge", "boosting"]}, **FAST)
    assert bench.score("debut", "ridge", "r2") > 0.4 and bench.score("debut", "boosting", "r2") > 0.3
    assert abs(bench.score("debut", "baseline", "r2")) < 0.1
    assert bench.score("debut", "ridge", "mae") < bench.score("debut", "baseline", "mae")


def test_the_popularity_question_never_uses_the_first_day_popularity(table):
    noisy = table.copy()
    noisy["entry_pop"] = noisy["pop_first_week"]                                    # a perfect copy of the answer ...
    bench = run_benchmark(noisy, "first_week_popularity", plan={"debut": ["ridge"]}, **FAST)
    assert bench.score("debut", "ridge", "r2") < 0.9                                 # ... must not be used (it would give R-squared near 1)
    leaky = run_benchmark(noisy, "chart_days", plan={"debut": ["ridge"]}, **FAST)
    assert leaky.table["n"].iloc[0] == 240


# ================================================================================ forward in time
def test_forward_run_trains_on_the_past_and_tests_on_the_future(table):
    out = run_forward(table, "climb_top10", cutoff=350, gap=30, plan={"debut": ["baseline", "logistic"]}, n_boot=200)
    entry = table["entry_snapshot"].to_numpy()
    assert out["n_train"].iloc[0] == (entry < 320).sum() and out["n_test"].iloc[0] == (entry >= 350).sum()
    assert out.loc[out["model"] == "logistic", "auc"].iloc[0] > 0.7
    assert out.loc[out["model"] == "baseline", "auc"].iloc[0] == 0.5
    with pytest.raises(ValueError):
        run_forward(table, "climb_top10", cutoff=0, gap=30)


def test_score_predictions_scores_a_perfect_forecast_perfectly():
    y = np.array([0, 1, 0, 1, 1, 0])
    scores = score_predictions("binary", y, y * 0.98 + 0.01)
    assert scores["auc"] == 1.0 and scores["average_precision"] == 1.0 and scores["brier"] < 0.001
    assert MAIN_METRIC == {"binary": "auc", "multiclass": "balanced_accuracy", "regression": "r2"}


# ================================================================================ a model that is not in the table
def test_a_variant_with_the_tables_own_settings_reproduces_the_tables_score_and_a_broken_one_does_not(climb):
    score, low, high = score_variant(climb, "debut", lambda: make_classifier("logistic"))
    assert score == pytest.approx(climb.score("debut", "logistic", "auc")) and low == pytest.approx(climb.interval("debut", "logistic", "auc")[0])
    assert high == pytest.approx(climb.interval("debut", "logistic", "auc")[1])
    heavy, _, _ = score_variant(climb, "debut", lambda: make_classifier("logistic", logistic_c=0.0001))       # almost no weight on any input
    assert heavy <= score + 1e-9 and low <= score <= high
    shallow, _, _ = score_variant(climb, "debut", lambda: make_classifier("boosting", boosting={"max_depth": 2}))
    assert 0.5 < shallow <= 1.0                                                                               # still finds the planted signal
