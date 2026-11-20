"""
tests/test_ml_models.py  -  checks of the model factories and the feature ladder (src/ml/models.py).

WHAT THIS FILE IS
    The ladder lists the right columns at every step, the excluded column stays out of the popularity task, each model can be built,
    fitted on data with missing values and asked for predictions, a baseline knows only the base rate, and the same seed gives the
    same model.

WHY WE NEED IT
    The notebook compares models fitted inside cross-validation. If a model silently saw different columns than the ladder says, or
    if a pipeline learned its medians from the whole data set instead of the training rows, the comparison would be unfair.

WHAT HAPPENS WHEN IT RUNS
    Small random data with a built-in signal and some missing values.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Same rules for every model, written down once and tested once.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.ml import metrics as M
from src.ml.dataset import ARTIST_FEATURES, DEBUT_FEATURES, EARLY_FEATURES, RELEASE_FEATURES, TIMING_FEATURES
from src.ml.models import (CLASSIFIER_NAMES, LADDER, LOG_COLUMNS, REGRESSOR_NAMES, feature_set, make_classifier, make_regressor)


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    n = 240
    X = pd.DataFrame({"duration_min": rng.gamma(9, 0.4, n), "total_tracks": rng.integers(1, 40, n).astype(float),
                      "artist_prior_slot_days": rng.integers(0, 500, n).astype(float), "entry_rank": rng.integers(1, 51, n).astype(float),
                      "artist_prior_mean_pop": rng.normal(85, 5, n), "is_explicit": rng.integers(0, 2, n).astype(float)})
    X.loc[rng.random(n) < 0.15, "artist_prior_mean_pop"] = np.nan                     # a fifth of the column is missing
    signal = -(X["entry_rank"] - 25) / 12 + 0.002 * X["artist_prior_slot_days"] + rng.normal(0, 1, n)
    return X, signal


# ================================================================================ the ladder
def test_each_step_adds_a_block_to_the_previous_ones():
    blocks = [RELEASE_FEATURES, TIMING_FEATURES, ARTIST_FEATURES, DEBUT_FEATURES, EARLY_FEATURES]
    for position, step in enumerate(LADDER):
        assert feature_set(step) == [c for block in blocks[: position + 1] for c in block]
    assert len(set(feature_set("early"))) == len(feature_set("early"))              # no column twice


def test_the_first_day_popularity_is_not_an_input_of_the_popularity_task_but_of_the_others():
    assert "entry_pop" not in feature_set("debut", "first_week_popularity")
    assert "entry_rank" in feature_set("debut", "first_week_popularity")
    assert "entry_pop" in feature_set("debut", "climb_top10")


def test_no_outcome_column_is_ever_an_input():
    outcome_like = {"days_on_chart", "avg_rank", "best_rank", "reached_top10", "y_debut_top10", "y_climb_top10", "chart_days_30",
                    "pop_first_week", "stay_class", "longest_run_days", "peak_popularity", "avg_popularity"}
    assert outcome_like.isdisjoint(feature_set("early"))


def test_unknown_step_is_refused():
    with pytest.raises(ValueError):
        feature_set("future")


# ================================================================================ classifiers and regressors
@pytest.mark.parametrize("name", CLASSIFIER_NAMES)
def test_every_classifier_fits_with_missing_values_and_returns_probabilities(frame, name):
    X, signal = frame
    y = (signal > 0).astype(int).to_numpy()
    model = make_classifier(name).fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (len(X), 2) and np.allclose(p.sum(axis=1), 1) and np.isfinite(p).all()
    if name != "baseline":
        assert M.roc_auc(y, p[:, 1]) > 0.7                                          # the signal is there, the model finds it


@pytest.mark.parametrize("name", REGRESSOR_NAMES)
def test_every_regressor_fits_with_missing_values(frame, name):
    X, signal = frame
    model = make_regressor(name).fit(X, signal.to_numpy())
    prediction = model.predict(X)
    assert prediction.shape == (len(X),) and np.isfinite(prediction).all()
    if name != "baseline":
        assert M.r_squared(signal.to_numpy(), prediction) > 0.3


def test_a_three_class_problem_gives_three_probabilities(frame):
    X, signal = frame
    y = np.digitize(signal.to_numpy(), [-0.5, 0.5])
    for name in ("logistic", "forest", "boosting"):
        p = make_classifier(name).fit(X, y).predict_proba(X)
        assert p.shape == (len(X), 3) and np.allclose(p.sum(axis=1), 1)


def test_the_baseline_knows_only_the_base_rate_and_the_mean(frame):
    X, signal = frame
    y = (signal > 0.5).astype(int).to_numpy()
    p = make_classifier("baseline").fit(X, y).predict_proba(X)[:, 1]
    assert np.allclose(p, y.mean())
    assert np.allclose(make_regressor("baseline").fit(X, signal.to_numpy()).predict(X), signal.mean())


def test_the_same_seed_gives_the_same_model_and_another_seed_a_different_forest(frame):
    X, signal = frame
    y = (signal > 0).astype(int).to_numpy()
    for name in ("forest", "boosting"):
        a = make_classifier(name, seed=1).fit(X, y).predict_proba(X)
        b = make_classifier(name, seed=1).fit(X, y).predict_proba(X)
        assert np.array_equal(a, b)
    assert not np.array_equal(make_classifier("forest", seed=1).fit(X, y).predict_proba(X), make_classifier("forest", seed=2).fit(X, y).predict_proba(X))


def test_the_pipeline_learns_its_medians_from_the_rows_it_is_fitted_on(frame):
    X, signal = frame
    y = (signal > 0).astype(int).to_numpy()
    model = make_classifier("logistic").fit(X.iloc[:120], y[:120])
    imputer = model.named_steps["fill"]
    column = list(X.columns).index("artist_prior_mean_pop")
    assert imputer.statistics_[column] == pytest.approx(np.nanmedian(X["artist_prior_mean_pop"].iloc[:120]))


def test_log_scaled_columns_are_only_those_present_and_do_not_change_the_input(frame):
    X, _ = frame
    before = X.copy()
    make_classifier("logistic").fit(X, (X["entry_rank"] < 25).astype(int))
    pd.testing.assert_frame_equal(X, before)                                        # fitting never edits the caller's table
    assert "duration_min" in LOG_COLUMNS


def test_unknown_model_names_are_refused():
    with pytest.raises(ValueError):
        make_classifier("neural")
    with pytest.raises(ValueError):
        make_regressor("neural")


def test_boosting_settings_can_be_replaced_for_a_sensitivity_check_and_default_to_the_fixed_ones(frame):
    from src.ml.models import BOOSTING_SETTINGS
    X, signal = frame
    y = (signal > 0).astype(int).to_numpy()
    default = make_classifier("boosting").named_steps["model"]
    assert {key: getattr(default, key) for key in BOOSTING_SETTINGS} == BOOSTING_SETTINGS
    deeper = make_classifier("boosting", boosting={"max_depth": 5}).named_steps["model"]
    assert deeper.max_depth == 5 and deeper.learning_rate == BOOSTING_SETTINGS["learning_rate"]       # only the named setting changes
    assert make_regressor("boosting", boosting={"max_iter": 40}).named_steps["model"].max_iter == 40
    assert make_classifier("logistic", logistic_c=5.0).named_steps["model"].C == 5.0
    a = make_classifier("boosting").fit(X, y).predict_proba(X)
    b = make_classifier("boosting", boosting={"max_depth": 2}).fit(X, y).predict_proba(X)
    assert not np.array_equal(a, b)
