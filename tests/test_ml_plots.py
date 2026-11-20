"""
tests/test_ml_plots.py  -  tests for the chart parts of Phase 6A (src/ml/plots.py).

WHAT THIS FILE IS
    Small drawings made with each panel function (ladder, ROC, precision-recall, calibration, confusion matrix, gains, break-even
    map, forecast against outcome) and checks on what ended up in the picture: how many dots and lines, which labels, whether the
    diagonal of a confusion matrix is outlined, whether a break-even map without a break-even line still draws.

WHY WE NEED IT
    The same panel function draws the same kind of chart for seven questions. A panel that fails only for an unusual case (a grid
    where the return never turns negative, a question with a single model) would break the notebook half way through a long run.

WHAT HAPPENS WHEN IT RUNS
    Everything is drawn with the non-interactive "Agg" backend (no window opens), inside a temporary matplotlib setting so that the
    project look does not leak into other tests. Pictures are closed after every test.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    A chart function is only reusable if it keeps working for the awkward cases.
"""
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib as mpl                                    # noqa: E402  (the backend has to be chosen before pyplot is used)
import matplotlib.pyplot as plt                             # noqa: E402
import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
import pytest                                               # noqa: E402

from matplotlib.collections import PathCollection           # noqa: E402  (what a scatter plot is made of; an interval drawn with vlines is not one)

from src.eda import style                                   # noqa: E402
from src.ml import metrics as M                             # noqa: E402
from src.ml import plots as MP                              # noqa: E402


@pytest.fixture(autouse=True)
def project_look():
    with mpl.rc_context():
        style.apply_style()
        yield
    plt.close("all")


def _dots(ax):
    """The number of dots (scatter points) in an axes."""
    return sum(len(c.get_offsets()) for c in ax.collections if isinstance(c, PathCollection))


def _ladder_table():
    rows = []
    for step_index, step in enumerate(["release", "timing", "artist"]):
        for model, shift in [("baseline", 0.0), ("logistic", 0.0), ("boosting", 0.01)]:
            auc = 0.5 if model == "baseline" else 0.55 + 0.05 * step_index + shift
            rows.append({"step": step, "model": model, "auc": auc, "auc_low": auc - 0.04, "auc_high": auc + 0.04})
    return pd.DataFrame(rows)


LABELS = {"release": "Release", "timing": "+ Timing", "artist": "+ Artist"}
MODELS = {"baseline": "Baseline", "logistic": "Logistic regression", "boosting": "Gradient boosting"}


# ================================================================================ colours
def test_step_colors_are_spread_over_the_whole_ramp_lightest_first():
    for n in range(1, 6):
        colors = MP.step_colors(n)
        assert len(colors) == n and len(set(colors)) == n
    five = MP.step_colors(5)
    assert five[0] == "#86b6ef" and five[-1] == "#104281"
    assert MP.step_colors(2) == [five[0], five[-1]]


@pytest.mark.parametrize("n", [0, 6, -1])
def test_step_colors_refuses_a_number_outside_the_ramp(n):
    with pytest.raises(ValueError):
        MP.step_colors(n)


# ================================================================================ the ladder
def test_ladder_draws_one_dot_per_model_and_step_and_leaves_out_the_baseline():
    fig, ax = plt.subplots()
    MP.draw_ladder(ax, _ladder_table(), ["release", "timing", "artist"], ["baseline", "logistic", "boosting"], "auc", LABELS, MODELS,
                   reference=0.5, reference_label="knows nothing", ylim=(0.4, 0.9))
    assert _dots(ax) == 3 * 2                                                             # three steps, two models drawn
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Release", "+ Timing", "+ Artist"]
    assert ax.get_ylim() == (0.4, 0.9)
    texts = [t.get_text() for t in ax.texts]
    assert any("Logistic regression" in t for t in texts) and any("Gradient boosting" in t for t in texts)     # labelled at the last step
    assert any("knows nothing" in t for t in texts)


def test_ladder_with_a_single_model_and_a_step_that_is_missing_still_draws():
    table = _ladder_table()
    table = table[~((table["step"] == "timing") & (table["model"] == "logistic"))]
    fig, ax = plt.subplots()
    MP.draw_ladder(ax, table, ["release", "timing", "artist"], ["logistic"], "auc", LABELS, MODELS)
    assert _dots(ax) == 2                                                                 # the missing (timing, logistic) row is skipped, not an error


def test_ladder_legend_lists_the_models_without_the_baseline_and_the_reference():
    fig, ax = plt.subplots()
    MP.draw_ladder(ax, _ladder_table(), ["release", "timing", "artist"], ["baseline", "logistic", "boosting"], "auc", LABELS, MODELS, label_last=False)
    MP.ladder_legend(ax, ["baseline", "logistic", "boosting"], MODELS, reference_label="knows nothing")
    fig.canvas.draw()
    texts = {t.get_text() for t in ax.texts} | {t.get_text() for child in ax.get_children() if hasattr(child, "get_texts") for t in child.get_texts()}
    assert "Logistic regression" in texts and "Gradient boosting" in texts and "knows nothing" in texts
    assert "Baseline" not in texts


# ================================================================================ curves
def _curves(n=300, seed=1):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    score = rng.normal(size=n) + 1.2 * y
    return y, score


def test_roc_panel_has_the_chance_diagonal_and_one_line_per_curve():
    y, score = _curves()
    fpr, tpr = M.roc_curve(y, score)[:2]
    fig, ax = plt.subplots()
    MP.draw_roc(ax, [{"label": "Model", "fpr": fpr, "tpr": tpr, "color": style.BLUE}])
    assert len(ax.lines) == 2                                                             # diagonal + one curve
    assert ax.get_xlim() == (0, 1) and ax.get_ylim() == (0, 1.0)
    assert ax.get_aspect() == 1.0


def test_precision_recall_panel_writes_the_base_rate():
    y, score = _curves()
    recall_precision = M.precision_recall_curve(y, score)
    recall, precision = recall_precision[0], recall_precision[1]
    fig, ax = plt.subplots()
    MP.draw_pr(ax, [{"label": "Model", "recall": recall, "precision": precision, "color": style.VIOLET}], float(y.mean()))
    assert any("base rate" in t.get_text() for t in ax.texts)
    assert len(ax.lines) == 2                                                             # base-rate line + one curve


def test_calibration_panel_draws_one_dot_per_group_on_a_square_axis():
    y, score = _curves()
    probability = 1 / (1 + np.exp(-(score - 1)))
    table = M.calibration_table(y, probability, n_bins=5)
    fig, ax = plt.subplots()
    MP.draw_calibration(ax, table)
    assert _dots(ax) == 5
    assert ax.get_xlim() == ax.get_ylim()


# ================================================================================ confusion matrix
def test_confusion_panel_shows_counts_and_shares_and_outlines_the_diagonal():
    matrix = np.array([[50, 8, 2], [10, 30, 10], [1, 9, 40]])
    fig, ax = plt.subplots()
    MP.draw_confusion(ax, matrix, ["a", "b", "c"], ["A", "B", "C"])
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 2 * 9                                                            # a count and a share in every cell
    assert "50" in texts and "83% of row" in texts                                        # 50 of 60 in the first row
    assert len(ax.patches) == 3                                                           # one outline per correct-forecast cell
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C"]


def test_confusion_panel_copes_with_a_row_that_has_no_songs():
    matrix = np.array([[5, 1], [0, 0]])
    fig, ax = plt.subplots()
    MP.draw_confusion(ax, matrix, ["yes", "no"], ["yes", "no"])
    assert "0% of row" in [t.get_text() for t in ax.texts]


# ================================================================================ decisions
def test_gains_panel_draws_the_random_line_the_best_possible_line_and_the_curves():
    y, score = _curves()
    table = M.gains_table(y, score)
    fig, ax = plt.subplots()
    MP.draw_gains(ax, [{"label": "Model", "share": table["share_flagged"], "gain": table["gain"], "color": style.BLUE}], float(y.mean()))
    assert len(ax.lines) == 3
    assert any("best possible" in t.get_text() for t in ax.texts)


def test_break_even_map_draws_the_zero_line_when_the_return_changes_sign():
    shares = np.array([0.05, 0.10, 0.20, 0.40])
    ratios = np.array([2, 4, 8, 16])
    roi = np.array([[-0.5, 0.2, 0.8, 1.5], [-0.6, 0.0, 0.7, 1.4], [-0.7, -0.2, 0.5, 1.2], [-0.8, -0.4, 0.2, 0.9]])
    fig, ax = plt.subplots()
    MP.draw_roi_map(ax, shares, ratios, roi)
    assert len(ax.collections) >= 2                                                       # the coloured grid and the contour of zero return
    assert [t.get_text() for t in ax.get_xticklabels()] == ["2", "4", "8", "16"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["5%", "10%", "20%", "40%"]


@pytest.mark.parametrize("sign", [1.0, -1.0, 0.0])
def test_break_even_map_still_draws_when_the_return_never_changes_sign(sign):
    """Every cell pays back (or none does, or all are exactly zero): there is no break-even line to draw, and that is not an error."""
    shares = np.array([0.1, 0.2])
    ratios = np.array([2, 4, 8])
    roi = sign * np.array([[0.3, 0.6, 0.9], [0.2, 0.5, 0.8]])
    fig, ax = plt.subplots()
    mesh = MP.draw_roi_map(ax, shares, ratios, roi)
    assert mesh is not None
    fig.canvas.draw()


# ================================================================================ numbers
def test_forecast_against_outcome_panel_draws_the_diagonal_and_the_binned_average():
    rng = np.random.default_rng(3)
    forecast = rng.normal(10, 3, 200)
    actual = forecast + rng.normal(0, 2, 200)
    fig, ax = plt.subplots()
    MP.draw_actual_vs_predicted(ax, actual, forecast, n_bins=10, unit="days")
    assert len(ax.lines) == 2                                                             # the 45-degree line and the binned averages
    assert len(ax.lines[1].get_xdata()) == 10
    assert ax.get_xlabel() == "forecast (days)" and ax.get_ylabel() == "what happened (days)"
    assert ax.get_xlim() == ax.get_ylim()


# ================================================================================ explanations
def test_forest_rows_are_open_dots_when_the_interval_includes_the_reference_value():
    fig, ax = plt.subplots()
    MP.draw_forest(ax, [0, 1, 2], [2.0, 0.5, 1.1], [1.5, 0.3, 0.8], [2.7, 0.8, 1.5], colors=[style.BLUE, style.RED, style.MUTED], log=True,
                   texts=["a", "b", "c"])
    dots = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(dots) == 3 and ax.get_xscale() == "log"
    faces = [tuple(np.round(c.get_facecolor()[0], 3)) for c in dots]
    open_face = tuple(np.round(mpl.colors.to_rgba(style.SURFACE), 3))
    assert faces[0] != open_face and faces[1] != open_face                       # the first two intervals exclude 1: filled dots
    assert faces[2] == open_face                                                 # 0.8 to 1.5 includes 1: an open dot
    assert [t.get_text() for t in ax.texts] == ["a", "b", "c"]


def test_forest_skips_a_row_with_a_missing_number_instead_of_failing():
    fig, ax = plt.subplots()
    MP.draw_forest(ax, [0, 1], [2.0, np.nan], [1.5, np.nan], [2.7, np.nan], colors=style.BLUE, texts=["a", "b"])
    assert len([c for c in ax.collections if isinstance(c, PathCollection)]) == 1
    assert [t.get_text() for t in ax.texts] == ["a"]


def test_partial_dependence_panel_draws_the_mean_line_the_band_and_ten_decile_ticks():
    grid = np.linspace(0, 10, 25)
    rng = np.random.default_rng(2)
    curves = 0.2 + 0.02 * grid[None, :] + rng.normal(0, 0.01, (15, 25))
    fig, ax = plt.subplots()
    MP.draw_partial(ax, grid, curves, np.linspace(1, 9, 10), color=style.VIOLET, xlabel="input", ylabel="chance", as_percent=True, ylim=(0, 0.6))
    assert len(ax.lines) == 1 and np.allclose(ax.lines[0].get_ydata(), curves.mean(axis=0))
    assert len(ax.collections) == 2                                              # the band and the decile ticks
    assert ax.get_ylim() == (0, 0.6) and ax.get_xlabel() == "input"
    ax.figure.canvas.draw()
    assert ax.get_yticklabels()[1].get_text().endswith("%")


def test_partial_dependence_panel_accepts_a_symlog_axis_for_an_input_with_zeros():
    grid = np.array([0.0, 2.0, 10.0, 100.0, 1000.0])
    curves = np.tile(np.linspace(0.2, 0.3, 5), (4, 1))
    fig, ax = plt.subplots()
    MP.draw_partial(ax, grid, curves, [0.0, 5.0, 50.0], color=style.BLUE, xlabel="days", ylabel="chance", x_scale="symlog")
    assert ax.get_xscale() == "symlog"
    fig.canvas.draw()                                                              # a zero on a log axis would fail here if the scale were plain "log"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "10", "100", "1,000"]


def test_break_even_map_can_write_the_return_in_every_cell():
    shares = np.array([0.1, 0.2])
    ratios = np.array([2, 4, 8])
    roi = np.array([[-0.5, 0.2, 1.4], [-0.6, 0.0, 0.9]])
    fig, ax = plt.subplots()
    MP.draw_roi_map(ax, shares, ratios, roi, annotate=True)
    assert sorted(t.get_text() for t in ax.texts) == sorted(["\u22120.5", "+0.2", "+1.4", "\u22120.6", "+0.0", "+0.9"])
    other, other_ax = plt.subplots()
    MP.draw_roi_map(other_ax, shares, ratios, roi)
    assert len(other_ax.texts) == 0                                                  # off by default


def test_gains_panel_can_mark_one_share_of_songs():
    y, score = _curves()
    table = M.gains_table(y, score)
    fig, ax = plt.subplots()
    MP.draw_gains(ax, [{"label": "Model", "share": table["share_flagged"], "gain": table["gain"], "color": style.BLUE}], float(y.mean()), mark_share=0.10)
    assert any(t.get_text() == "top 10%" for t in ax.texts) and len(ax.lines) == 4               # the guide is a fourth line
