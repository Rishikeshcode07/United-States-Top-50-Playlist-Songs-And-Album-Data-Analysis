"""
tests/test_ml_explain.py  -  checks of permutation importance and partial dependence (src/ml/explain.py).

WHAT THIS FILE IS
    Data where the answer depends on exactly one input and a second input is pure noise. The informative input must get a clearly
    positive importance, the noise input about zero; two inputs that carry the same information must be scrambled together; and
    partial dependence must reproduce a known curve.

WHY WE NEED IT
    An importance chart is read as a statement about the data. If the method gave a noise column a large importance, or hid an
    important input behind its twin, the statement would be wrong.

WHAT HAPPENS WHEN IT RUNS
    Seeded random data and small models.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Make sure the explanation of a model is not an artefact of the explaining method.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.ml import metrics as M
from src.ml.explain import build_design, partial_dependence, permutation_importance
from src.ml.validation import repeated_grouped_folds


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(1)
    n = 400
    signal = rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "twin": signal + rng.normal(0, 0.05, n), "noise": rng.normal(size=n)})
    y = (signal + rng.normal(0, 0.5, n) > 0).astype(int)
    folds = repeated_grouped_folds(np.arange(n) // 4, 5, 2, seed=0)
    return X, y, folds


def _forest():
    return RandomForestClassifier(n_estimators=60, min_samples_leaf=5, random_state=0)


def test_the_informative_input_matters_and_the_noise_does_not(setup):
    X, y, folds = setup
    table = permutation_importance(_forest, X, y, folds, {"signal+twin": ["signal", "twin"], "noise": ["noise"]}, M.roc_auc, n_permutations=4)
    importance = table.set_index("group")["importance"]
    assert importance["signal+twin"] > 0.2
    assert abs(importance["noise"]) < 0.03
    assert table["intact_score"].iloc[0] > 0.85


def test_twins_scrambled_separately_hide_each_other_but_together_do_not(setup):
    X, y, folds = setup
    separate = permutation_importance(_forest, X, y, folds, {"signal": ["signal"], "twin": ["twin"]}, M.roc_auc, n_permutations=4).set_index("group")["importance"]
    together = permutation_importance(_forest, X, y, folds, {"both": ["signal", "twin"]}, M.roc_auc, n_permutations=4).set_index("group")["importance"]
    assert together["both"] > 2 * max(separate["signal"], separate["twin"])


def test_the_columns_of_one_group_move_together(setup):
    seen = {}

    class Spy(LogisticRegression):
        def predict_proba(self, X):
            seen.setdefault("frames", []).append(X.copy())
            return super().predict_proba(X)

    X, y, folds = setup
    permutation_importance(lambda: Spy(max_iter=500), X, y, folds[:1], {"pair": ["signal", "twin"]}, M.roc_auc, n_permutations=1)
    scrambled = seen["frames"][1]                                                  # the first frame is the intact one
    intact_rows = X.loc[scrambled.index]
    gap_signal = scrambled["signal"].to_numpy() - intact_rows["signal"].to_numpy()
    assert not np.allclose(gap_signal, 0)
    # a scrambled row still has a signal and a twin that belong together (they came from the same original row)
    assert np.abs(scrambled["signal"] - scrambled["twin"]).max() < 0.5


def test_same_seed_same_importance(setup):
    X, y, folds = setup
    a = permutation_importance(_forest, X, y, folds[:1], {"noise": ["noise"]}, M.roc_auc, n_permutations=2, seed=5)
    b = permutation_importance(_forest, X, y, folds[:1], {"noise": ["noise"]}, M.roc_auc, n_permutations=2, seed=5)
    assert a.equals(b)


def test_loss_metrics_flip_the_sign(setup):
    X, y, folds = setup
    table = permutation_importance(_forest, X, y, folds[:1], {"signal": ["signal"]}, lambda t, p: M.log_loss(t, p), higher_is_better=False, n_permutations=3)
    assert table["importance"].iloc[0] > 0.1                                       # scrambling makes the loss go UP, importance is positive


def test_partial_dependence_reproduces_a_known_curve():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"x": rng.uniform(-3, 3, 600), "z": rng.normal(size=600)})
    y = (rng.random(600) < 1 / (1 + np.exp(-2 * X["x"]))).astype(int)
    model = LogisticRegression(C=100, max_iter=1000).fit(X, y)
    grid = np.array([-2.0, 0.0, 2.0])
    curve = partial_dependence(model, X, "x", grid)[0]
    assert curve[0] < 0.1 and curve[1] == pytest.approx(0.5, abs=0.12) and curve[2] > 0.9
    assert np.all(np.diff(curve) > 0)
    several = partial_dependence([model, model], X, "x", grid)
    assert several.shape == (2, 3) and np.array_equal(several[0], several[1])


# ================================================================================ the design of an explanatory regression
def _frame():
    return pd.DataFrame({"minutes": [2.0, 3.0, 4.0, 5.0, 6.0], "tracks": [1, 10, 12, 20, 100], "level": [80.0, np.nan, 90.0, 85.0, np.nan],
                         "single": [True, False, False, True, False]})


def test_design_standardises_transforms_and_keeps_the_binary_columns():
    design, scale = build_design(_frame(), {"Length": ("minutes", "raw"), "Album size": ("tracks", "log1p")}, {"Single": "single"})
    assert design.columns.tolist() == ["Intercept", "Length", "Album size", "Single"]
    assert (design["Intercept"] == 1).all()
    assert design["Length"].mean() == pytest.approx(0, abs=1e-12) and design["Length"].std(ddof=1) == pytest.approx(1)
    assert design["Album size"].mean() == pytest.approx(0, abs=1e-12) and design["Album size"].std(ddof=1) == pytest.approx(1)
    assert design["Single"].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    row = scale.set_index("input").loc["Album size"]
    assert row["mean"] == pytest.approx(np.log1p([1, 10, 12, 20, 100]).mean()) and row["sd"] == pytest.approx(np.log1p([1, 10, 12, 20, 100]).std(ddof=1))


def test_missing_values_are_filled_with_the_median_and_flagged():
    design, _ = build_design(_frame(), {"Level": ("level", "raw")}, {})
    assert design.columns.tolist() == ["Intercept", "Level", "Level (missing)"]
    assert design["Level (missing)"].tolist() == [0.0, 1.0, 0.0, 0.0, 1.0]
    filled = design["Level"].to_numpy() * 4.2720 + 85.0                                  # back to points: sd of (80, 85, 90, 85, 85) is about 3.536
    assert not np.isnan(design["Level"]).any()
    assert design["Level"].iloc[1] == design["Level"].iloc[3]                              # the filled value equals the median (85), as row 3 does


def test_a_handful_of_missing_values_is_filled_without_a_flag_when_flag_from_says_so():
    frame = pd.DataFrame({"level": [80.0, np.nan, 90.0, 85.0, 70.0, np.nan, 95.0]})
    flagged, _ = build_design(frame, {"Level": ("level", "raw")}, {}, flag_from=2)
    assert flagged.columns.tolist() == ["Intercept", "Level", "Level (missing)"]
    plain, _ = build_design(frame, {"Level": ("level", "raw")}, {}, flag_from=3)
    assert plain.columns.tolist() == ["Intercept", "Level"]                                  # only two values are missing: no flag
    assert not np.isnan(plain["Level"]).any() and plain["Level"].iloc[1] == plain["Level"].iloc[5]   # both filled with the same median


def test_design_refuses_unknown_transforms_and_constant_columns():
    with pytest.raises(ValueError):
        build_design(_frame(), {"Length": ("minutes", "square")}, {})
    with pytest.raises(ValueError):
        build_design(pd.DataFrame({"x": [1.0, 1.0, 1.0]}), {"X": ("x", "raw")}, {})
