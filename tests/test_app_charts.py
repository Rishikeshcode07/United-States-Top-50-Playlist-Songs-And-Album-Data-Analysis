"""
tests/test_app_charts.py  -  the chart builders of the dashboard (app/ui/charts.py), drawn without a browser.

WHAT THIS FILE IS
    Tests of every chart builder: it must return a Plotly figure for a small hand-made table, a calm "no data" figure for an empty
    table, and it must not change the table it was given. A few tests check the details that were once wrong (place 1 at the top,
    percentage ticks, the dotted 50 % line of a survival curve, the second axis of the Pareto chart, the log axis of a forest plot).

WHY WE NEED IT
    A chart that fails on an empty selection shows a red error box instead of a message, and the dashboard filters can easily
    produce an empty table. Testing the builders here (no Streamlit, no browser) is fast and finds this before a person does.

WHAT HAPPENS WHEN IT RUNS
    Plotly builds the figures in memory and the tests read their traces and axes. Nothing is drawn on a screen and nothing is
    written to disk. The Plotly template of the dashboard is registered for the module and the previous default is put back.

EXPECTED OUTCOME
    All tests pass with the oldest Plotly the project allows (5.20) and with the newest one.

MOTIVE
    Every page can only be as reliable as the charts it draws.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402

from app.ui import charts as ch  # noqa: E402
from app.ui import theme  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def dashboard_template():
    before = pio.templates.default
    theme.register_template()
    yield
    pio.templates.default = before


# ================================================================================================ small hand-made inputs
DATES = pd.date_range("2025-01-01", periods=6, freq="D")


def place_matrix() -> pd.DataFrame:
    """Two songs over six days; an empty cell is a day off the chart."""
    return pd.DataFrame({"s1": [1, 2, np.nan, 4, 5, 6], "s2": [10, 9, 8, 7, np.nan, np.nan]}, index=DATES)


def day_table() -> pd.DataFrame:
    return pd.DataFrame({"position": [1, 2, 3, 4], "song_clean": ["Song A", "Song B", "Song C", "Song D"], "artist_credit": ["X", "Y", "Z", "W"],
                         "movement": ["Same", "Up", "New entry", "Down"], "rank_change": [0, 3, np.nan, -2], "popularity_clean": [90, 80, 70, 60],
                         "selected": [True, True, True, False]})


def slope_table() -> pd.DataFrame:
    return pd.DataFrame({"label": ["rises", "falls", "stays", "arrives", "leaves", "far away"],
                         "place_a": [5, 3, 4, np.nan, 8, 30], "place_b": [2, 6, 4, 1, np.nan, 40]})


def calendar_cells() -> pd.DataFrame:
    days = pd.date_range("2025-01-01", periods=15)                      # a Wednesday to the Wednesday two weeks later
    return pd.DataFrame({"date": days, "weekday": days.day_of_week, "week_start": days - pd.to_timedelta(days.day_of_week, unit="D"), "value": np.arange(15.0)})


def survival_curves(groups=("A", "B")) -> pd.DataFrame:
    ages = np.arange(0, 6)
    parts = []
    for i, g in enumerate(groups):
        s = np.linspace(1.0, 0.6 - 0.1 * i, len(ages))
        parts.append(pd.DataFrame({"group": g, "age": ages, "survival": s, "low": s - 0.05, "high": np.minimum(s + 0.05, 1.0), "songs": 40}))
    return pd.concat(parts, ignore_index=True)


def song_daily() -> pd.DataFrame:
    return pd.DataFrame({"date": DATES, "position": [5, 4, np.nan, 3, 2, 2], "popularity_clean": [60, 62, np.nan, 66, 70, 71],
                         "popularity_trend_7d": [60, 61, 61, 62.5, 64, 65]})


def forest_table() -> pd.DataFrame:
    return pd.DataFrame({"name": ["Explicit", "Collaboration", "Album"], "hr": [0.8, 1.3, 2.4], "lo": [0.6, 1.0, 1.7], "hi": [1.05, 1.7, 3.4],
                         "reading": ["no clear difference", "leaves sooner", "leaves sooner"]})


def interval_table() -> pd.DataFrame:
    return pd.DataFrame({"day": [1, 2, 3], "share": [0.30, 0.20, 0.10], "lo": [0.25, 0.15, 0.06], "hi": [0.35, 0.26, 0.15], "kind": ["a", "a", "a"]})


# ================================================================================ every builder: a figure for real input
BUILDERS = {
    "hbar": lambda: ch.hbar(pd.DataFrame({"name": ["a", "b", "c"], "v": [3.0, 2.0, 1.0]}), "name", "v", "Title"),
    "vbar": lambda: ch.vbar(pd.DataFrame({"x": ["a", "b"], "y1": [1, 2], "y2": [3, 4]}), "x", ["y1", "y2"], "Title"),
    "line_chart": lambda: ch.line_chart(pd.DataFrame({"d": list(DATES) * 2, "v": range(12), "g": ["A"] * 6 + ["B"] * 6}), "d", "v", group="g", title="Title"),
    "heat": lambda: ch.heat(pd.DataFrame([[1, 2], [3, np.nan]], index=["r1", "r2"], columns=["c1", "c2"]), "Title"),
    "scatter_chart": lambda: ch.scatter_chart(pd.DataFrame({"x": [1, 2, 3], "y": [3, 1, 2], "g": ["a", "b", "a"], "s": [1, 5, 9]}), "x", "y", color="g", size="s", title="Title"),
    "histogram_chart": lambda: ch.histogram_chart(pd.DataFrame({"v": [1, 2, 2, 3, 3, 3], "g": list("aabbbb")}), "v", group="g", title="Title"),
    "rank_lines": lambda: ch.rank_lines(place_matrix(), names={"s1": "Song one"}, title="Title"),
    "rank_panels": lambda: ch.rank_panels(place_matrix(), names={"s1": "Song one"}, title="Title"),
    "day_bars": lambda: ch.day_bars(day_table(), "Title"),
    "slope_chart": lambda: ch.slope_chart(slope_table(), "Mon", "Tue", top_n=10, title="Title"),
    "calendar_heat": lambda: ch.calendar_heat(calendar_cells(), "Title"),
    "survival_chart": lambda: ch.survival_chart(survival_curves(), title="Title"),
    "lorenz_chart": lambda: ch.lorenz_chart([0, 0.5, 1], [0, 0.2, 1], 0.62, "Title"),
    "ribbon_chart": lambda: ch.ribbon_chart(pd.DataFrame({"x": [1, 2, 3], "m": [5, 6, 7], "l": [4, 5, 6], "h": [6, 7, 8]}), "x", "m", "l", "h", "Title"),
    "box_chart": lambda: ch.box_chart(pd.DataFrame({"v": [1, 2, 3, 4, 5, 6], "g": list("aaabbb")}), "v", "g", title="Title"),
    "span_chart": lambda: ch.span_chart(pd.DataFrame({"song": ["a", "b"], "first": pd.to_datetime(["2025-01-01", "2025-01-03"]),
                                                      "last": pd.to_datetime(["2025-01-05", "2025-01-03"])}), "song", "first", "last", title="Title"),
    "song_timeline": lambda: ch.song_timeline(song_daily(), "Title"),
    "pareto_chart": lambda: ch.pareto_chart(["a", "b", "c"], [50.0, 30.0, 20.0], [50.0, 80.0, 100.0], "Title"),
    "interval_bars": lambda: ch.interval_bars(interval_table(), "day", "share", "lo", "hi", "Title"),
    "forest_chart": lambda: ch.forest_chart(forest_table(), "name", "hr", "lo", "hi", "Title", color_col="reading"),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_a_builder_returns_a_figure_with_data_and_a_title(name):
    fig = BUILDERS[name]()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert fig.layout.title.text == "Title"


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_a_figure_can_be_sent_to_the_browser_as_json(name):
    """Streamlit sends the figure as JSON. A gap (NaN) must become null, not break the message."""
    payload = json.loads(BUILDERS[name]().to_json())
    assert "data" in payload and "layout" in payload


# ===================================================================================== every builder: calm on empty input
def empty(*columns) -> pd.DataFrame:
    return pd.DataFrame({c: [] for c in columns})


EMPTY_CASES = {
    "hbar": lambda: ch.hbar(empty("a", "b"), "a", "b"),
    "vbar": lambda: ch.vbar(empty("a", "b"), "a", "b"),
    "line_chart": lambda: ch.line_chart(empty("a", "b"), "a", "b"),
    "heat": lambda: ch.heat(pd.DataFrame()),
    "scatter_chart": lambda: ch.scatter_chart(empty("a", "b"), "a", "b"),
    "histogram_chart": lambda: ch.histogram_chart(empty("a"), "a"),
    "rank_lines": lambda: ch.rank_lines(pd.DataFrame()),
    "rank_lines (dates, no song)": lambda: ch.rank_lines(pd.DataFrame(index=DATES)),
    "rank_panels": lambda: ch.rank_panels(pd.DataFrame()),
    "day_bars": lambda: ch.day_bars(day_table().iloc[0:0]),
    "slope_chart": lambda: ch.slope_chart(slope_table().iloc[0:0], "a", "b"),
    "calendar_heat": lambda: ch.calendar_heat(calendar_cells().iloc[0:0]),
    "survival_chart": lambda: ch.survival_chart(survival_curves().iloc[0:0]),
    "lorenz_chart": lambda: ch.lorenz_chart([], [], 0.0),
    "ribbon_chart": lambda: ch.ribbon_chart(empty("x", "m", "l", "h"), "x", "m", "l", "h"),
    "box_chart": lambda: ch.box_chart(empty("v", "g"), "v", "g"),
    "span_chart": lambda: ch.span_chart(empty("a", "b", "c"), "a", "b", "c"),
    "song_timeline": lambda: ch.song_timeline(song_daily().iloc[0:0]),
    "pareto_chart": lambda: ch.pareto_chart([], [], []),
    "interval_bars": lambda: ch.interval_bars(interval_table().iloc[0:0], "day", "share", "lo", "hi"),
    "forest_chart": lambda: ch.forest_chart(forest_table().iloc[0:0], "name", "hr", "lo", "hi"),
}


@pytest.mark.parametrize("name", sorted(EMPTY_CASES))
def test_an_empty_table_gives_a_message_not_an_error(name):
    fig = EMPTY_CASES[name]()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1 and fig.layout.annotations[0].text
    assert fig.layout.xaxis.visible is False and fig.layout.yaxis.visible is False


def test_the_message_of_an_empty_figure_can_be_chosen():
    fig = ch.empty_figure("Nothing to compare", height=200)
    assert fig.layout.annotations[0].text == "Nothing to compare" and fig.layout.height == 200


def test_a_builder_whose_rows_are_all_nan_still_returns_a_figure():
    matrix = pd.DataFrame({"s1": [np.nan] * 6}, index=DATES)
    fig = ch.rank_lines(matrix)
    assert isinstance(fig, go.Figure) and len(fig.data) == 1
    json.loads(fig.to_json())


# ==================================================================================== builders never change their input
FRAME_BUILDERS = {
    "hbar": (lambda: pd.DataFrame({"name": ["a", "b"], "v": [2.0, 1.0]}), lambda d: ch.hbar(d, "name", "v")),
    "vbar": (lambda: pd.DataFrame({"x": ["a", "b"], "y": [2, 1]}), lambda d: ch.vbar(d, "x", "y")),
    "line_chart": (lambda: pd.DataFrame({"d": DATES, "v": range(6)}), lambda d: ch.line_chart(d, "d", "v")),
    "day_bars": (day_table, lambda d: ch.day_bars(d)),
    "slope_chart": (slope_table, lambda d: ch.slope_chart(d, "a", "b")),
    "survival_chart": (survival_curves, lambda d: ch.survival_chart(d)),
    "span_chart": (lambda: pd.DataFrame({"s": ["b", "a"], "f": pd.to_datetime(["2025-01-03", "2025-01-01"]), "l": pd.to_datetime(["2025-01-04", "2025-01-02"])}),
                   lambda d: ch.span_chart(d, "s", "f", "l")),
    "forest_chart": (forest_table, lambda d: ch.forest_chart(d, "name", "hr", "lo", "hi")),
    "interval_bars": (interval_table, lambda d: ch.interval_bars(d, "day", "share", "lo", "hi")),
    "calendar_heat": (calendar_cells, lambda d: ch.calendar_heat(d)),
    "rank_lines": (place_matrix, lambda d: ch.rank_lines(d)),
}


@pytest.mark.parametrize("name", sorted(FRAME_BUILDERS))
def test_a_builder_leaves_its_table_as_it_was(name):
    make, draw = FRAME_BUILDERS[name]
    table = make()
    before = table.copy(deep=True)
    draw(table)
    pd.testing.assert_frame_equal(table, before)


# ===================================================================================================== details that matter
def test_rank_lines_puts_place_one_at_the_top_and_labels_every_ten_places():
    fig = ch.rank_lines(place_matrix(), names={"s1": "Song one"})
    assert tuple(fig.layout.yaxis.range) == (50.5, 0.5)
    assert list(fig.layout.yaxis.tickvals) == [1, 10, 20, 30, 40, 50]
    assert [trace.name for trace in fig.data] == ["Song one", "s2"]


def test_a_day_off_the_chart_is_a_gap_in_the_line():
    fig = ch.rank_lines(place_matrix())
    assert fig.data[0].connectgaps is False
    assert np.isnan(np.asarray(fig.data[0].y, dtype=float)[2])


def test_rank_axis_uses_steps_of_five_for_a_short_range():
    fig = ch.rank_axis(go.Figure(), low=1, high=20)
    assert list(fig.layout.yaxis.tickvals) == [1, 5, 10, 15, 20]
    assert tuple(fig.layout.yaxis.range) == (20.5, 0.5)


def test_rank_panels_draws_one_panel_per_song_and_never_more_columns_than_songs():
    fig = ch.rank_panels(place_matrix(), columns=3)                          # two songs: the grid is 1 row x 2 columns, not 3 columns
    assert len(fig.data) == 2
    assert len(fig.layout.annotations) == 2                                   # one small title per panel
    assert fig.layout.xaxis.domain[1] < 0.6                                   # the first panel takes half of the width, not a third of an empty row
    assert fig.layout.height == 260


def test_rank_panels_adds_a_row_for_every_third_song():
    matrix = pd.DataFrame({f"s{i}": [1, 2, 3, 4, 5, 6] for i in range(5)}, index=DATES)
    fig = ch.rank_panels(matrix, columns=3, row_height=190)
    assert len(fig.data) == 5 and len(fig.layout.annotations) == 5
    assert fig.layout.height == 190 * 2 + 60
    assert tuple(fig.layout.yaxis.range) == (50.5, 0.5)


def test_percent_ticks_use_one_decimal_only_for_small_shares():
    fig = go.Figure(go.Bar(x=["a"], y=[0.05]))
    ch.percent_ticks(fig, [0.02, 0.05])
    assert fig.layout.yaxis.tickformat == ".1%"
    ch.percent_ticks(fig, [0.2, 0.4])
    assert fig.layout.yaxis.tickformat == ".0%"
    ch.percent_ticks(fig, [np.nan, np.nan])
    assert fig.layout.yaxis.tickformat == ".0%"
    ch.percent_ticks(fig, [0.3], axis="x")
    assert fig.layout.xaxis.tickformat == ".0%"


def test_bars_of_shares_get_percent_ticks():
    small = ch.vbar(pd.DataFrame({"x": ["a", "b"], "y": [0.02, 0.04]}), "x", "y", fmt=".0%")
    large = ch.hbar(pd.DataFrame({"n": ["a", "b"], "v": [0.2, 0.4]}), "n", "v", fmt=".0%")
    assert small.layout.yaxis.tickformat == ".1%"
    assert large.layout.xaxis.tickformat == ".0%"


def test_hbar_leaves_room_for_the_label_of_the_longest_bar():
    fig = ch.hbar(pd.DataFrame({"n": ["a", "b"], "v": [10.0, 5.0]}), "n", "v")
    assert fig.layout.xaxis.range[0] == 0 and fig.layout.xaxis.range[1] == pytest.approx(11.7)
    assert fig.layout.yaxis.autorange == "reversed"                       # the first row is on top
    assert list(fig.data[0].text) == ["10.0", "5.0"]


def test_hbar_with_a_negative_value_keeps_zero_in_view():
    fig = ch.hbar(pd.DataFrame({"n": ["a", "b"], "v": [10.0, -5.0]}), "n", "v")
    assert fig.layout.xaxis.rangemode == "tozero"


def test_vbar_draws_a_downward_series_below_the_axis_but_names_it_positive():
    df = pd.DataFrame({"x": ["a", "b"], "up": [3, 4], "down": [-2, -1]})
    fig = ch.vbar(df, "x", ["up", "down"], barmode="relative", downward=["down"])
    assert list(fig.data[1].y) == [-2, -1]
    assert list(fig.data[1].customdata) == [2, 1]


def test_survival_bands_are_fainter_when_there_are_many_groups():
    few = ch.survival_chart(survival_curves(("A", "B")))
    many = ch.survival_chart(survival_curves(("A", "B", "C", "D")))
    assert "0.15" in [t for t in few.data if t.fill == "tonexty"][0].fillcolor
    assert "0.07" in [t for t in many.data if t.fill == "tonexty"][0].fillcolor


def test_survival_chart_marks_the_median_and_can_hide_the_bands():
    fig = ch.survival_chart(survival_curves(), bands=False, max_age=3)
    assert len(fig.data) == 2                                              # one line per group, no band traces
    assert any(abs(shape.y0 - 0.5) < 1e-9 for shape in fig.layout.shapes)
    assert tuple(fig.layout.xaxis.range) == (0, 3)
    assert fig.layout.yaxis.tickformat == ".0%"


def test_survival_lines_are_steps_and_the_legend_names_the_number_of_songs():
    fig = ch.survival_chart(survival_curves(("A",)), bands=False)
    assert fig.data[0].line.shape == "hv"
    assert fig.data[0].name == "A (40 songs)"


def test_the_second_axis_of_the_pareto_chart_has_its_own_ticks():
    """Newer Plotly.js versions line up the ticks of a second axis with the first one (25.2, 50.5, 75.7). Linear ticks stop that."""
    fig = ch.pareto_chart(["a", "b", "c"], [50.0, 30.0, 20.0], [50.0, 80.0, 100.0])
    assert tuple(fig.layout.yaxis2.range) == (0, 100)
    assert fig.layout.yaxis2.tickmode == "linear" and fig.layout.yaxis2.dtick == 20 and fig.layout.yaxis2.tick0 == 0
    assert fig.layout.yaxis2.showgrid is False


def test_the_forest_chart_has_a_log_axis_and_a_line_at_no_difference():
    fig = ch.forest_chart(forest_table(), "name", "hr", "lo", "hi", color_col="reading")
    assert fig.layout.xaxis.type == "log"
    assert any(abs(shape.x0 - 1.0) < 1e-9 for shape in fig.layout.shapes)
    low, high = fig.layout.xaxis.range                                      # log10 units
    assert 10 ** low <= 0.6 and 10 ** high >= 3.4
    assert [trace.name for trace in fig.data] == ["no clear difference", "leaves sooner"]


def test_the_forest_chart_can_use_a_plain_axis_for_differences():
    table = pd.DataFrame({"name": ["a", "b"], "d": [0.1, -0.2], "lo": [-0.1, -0.4], "hi": [0.3, 0.0]})
    fig = ch.forest_chart(table, "name", "d", "lo", "hi", reference=0.0, log_x=False)
    assert fig.layout.xaxis.type != "log"
    assert any(abs(shape.x0) < 1e-9 for shape in fig.layout.shapes)


def test_interval_bars_never_draw_a_negative_whisker():
    """If a saved interval does not contain the estimate (a rounding quirk), the whisker is zero long instead of pointing the wrong way."""
    table = interval_table().assign(lo=[0.31, 0.21, 0.11], hi=[0.29, 0.19, 0.09])
    fig = ch.interval_bars(table, "day", "share", "lo", "hi")
    assert (np.asarray(fig.data[0].error_y.array) >= 0).all() and (np.asarray(fig.data[0].error_y.arrayminus) >= 0).all()


def test_interval_bars_write_the_axis_as_percent_and_start_at_zero():
    fig = ch.interval_bars(interval_table(), "day", "share", "lo", "hi")
    assert fig.layout.yaxis.tickformat == ".0%" and fig.layout.yaxis.rangemode == "tozero"


def test_day_bars_grey_out_the_songs_that_the_filters_leave_out():
    """Song D (place 4, 'Down') is outside the filters: it must not get a 'Down' bar but a grey one."""
    fig = ch.day_bars(day_table())
    assert [t.name for t in fig.data] == ["Up", "Same", "New entry", "Outside the filters"]
    assert fig.data[-1].marker.color == "#e2e1da"
    assert fig.layout.yaxis.autorange == "reversed"                           # place 1 on top
    assert len(fig.layout.yaxis.categoryarray) == 4 and fig.layout.yaxis.categoryarray[0].startswith("1")


def test_day_bars_can_leave_the_outside_songs_out_altogether():
    fig = ch.day_bars(day_table(), show_outside=False)
    assert [t.name for t in fig.data] == ["Up", "Same", "New entry"]
    assert len(fig.layout.yaxis.categoryarray) == 3


def test_day_bars_say_how_far_a_song_moved_in_the_hover_text():
    fig = ch.day_bars(day_table())
    up = [t for t in fig.data if t.name == "Up"][0]
    assert list(up.customdata) == ["up 3"]
    new = [t for t in fig.data if t.name == "New entry"][0]
    assert list(new.customdata) == ["new entry"]


def test_every_kind_of_movement_has_a_colour_and_can_be_drawn():
    kinds = ["Up", "Down", "Same", "New entry", "Re-entry", "Chart start"]
    assert set(kinds) <= set(theme.MOVE_COLORS)
    day = pd.DataFrame({"position": range(1, 7), "song_clean": list("ABCDEF"), "artist_credit": list("UVWXYZ"), "movement": kinds,
                        "rank_change": [2, -3, 0, np.nan, np.nan, np.nan], "popularity_clean": [50, 60, 70, 80, 90, 95], "selected": [True] * 6})
    fig = ch.day_bars(day)
    assert [t.name for t in fig.data] == kinds
    assert [t.marker.color for t in fig.data] == [theme.MOVE_COLORS[k] for k in kinds]


def test_slope_chart_draws_a_line_for_two_places_and_a_dot_for_one():
    fig = ch.slope_chart(slope_table(), "Mon", "Tue", top_n=10)
    named = [t.name for t in fig.data if t.name]
    assert named == ["Climbed", "Fell", "Same place", "Arrived in the top 10", "Left the top 10"]
    drawn = [t for t in fig.data if not t.name]
    assert len(drawn) == 5                                                 # the song far outside the top 10 is not drawn
    assert sum(1 for t in drawn if t.mode == "lines+markers+text") == 3 and sum(1 for t in drawn if t.mode == "markers+text") == 2
    assert tuple(fig.layout.yaxis.range) == (10.6, 0.4)


def test_calendar_heat_fills_the_days_that_exist_and_leaves_the_others_empty():
    fig = ch.calendar_heat(calendar_cells())
    z = np.asarray(fig.data[0].z, dtype=float)
    assert z.shape == (7, 3)
    assert np.isnan(z[0, 0]) and np.isnan(z[1, 0]) and z[2, 0] == 0.0         # the first day is a Wednesday
    assert list(fig.data[0].y) == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert fig.layout.yaxis.autorange == "reversed"                           # Monday on top


def test_heat_leaves_nan_cells_without_text():
    fig = ch.heat(pd.DataFrame([[1.0, 2.0], [3.0, np.nan]], index=["r1", "r2"], columns=["c1", "c2"]))
    assert np.asarray(fig.data[0].text).tolist() == [["1", "2"], ["3", ""]]


def test_scatter_with_reverse_y_puts_the_best_place_on_top():
    fig = ch.scatter_chart(pd.DataFrame({"x": [1, 2, 3], "y": [3, 20, 11]}), "x", "y", reverse_y=True)
    bottom_end, top_end = fig.layout.yaxis.range                              # the axis is turned round: the first number is the bottom
    assert bottom_end > top_end and top_end < 3 and bottom_end > 20


def test_scatter_sizes_the_bubbles_by_area():
    fig = ch.scatter_chart(pd.DataFrame({"x": [1, 2], "y": [1, 2], "s": [1, 4]}), "x", "y", size="s")
    assert fig.data[0].marker.sizemode == "area"


def test_histogram_with_a_percent_norm_says_so_on_the_axis():
    fig = ch.histogram_chart(pd.DataFrame({"v": [1, 2, 3, 4], "g": list("aabb")}), "v", group="g", histnorm="percent")
    assert fig.layout.yaxis.title.text == "Share of songs (%)"
    assert len(fig.data) == 2


def test_box_chart_skips_a_group_without_values():
    df = pd.DataFrame({"v": [1.0, 2.0, np.nan, np.nan], "g": list("aabb")})
    fig = ch.box_chart(df, "v", "g")
    assert [t.name for t in fig.data] == ["a"]


def test_a_one_day_stay_still_gets_a_visible_bar():
    df = pd.DataFrame({"s": ["a"], "f": pd.to_datetime(["2025-01-03"]), "l": pd.to_datetime(["2025-01-03"])})
    fig = ch.span_chart(df, "s", "f", "l")
    assert list(fig.data[0].x) == [86_400_000.0]


def test_span_chart_lists_the_songs_in_order_of_their_first_day():
    df = pd.DataFrame({"s": ["late", "early"], "f": pd.to_datetime(["2025-01-09", "2025-01-01"]), "l": pd.to_datetime(["2025-01-10", "2025-01-02"])})
    fig = ch.span_chart(df, "s", "f", "l")
    assert list(fig.layout.yaxis.categoryarray) == ["early", "late"]


def test_song_timeline_has_a_place_panel_and_a_popularity_panel():
    fig = ch.song_timeline(song_daily())
    assert [t.name for t in fig.data] == ["Place", "Popularity", "7-day average"]
    top = fig.layout.yaxis.range
    assert top[0] > top[1]                                                  # place 1 on top
    assert fig.layout.yaxis2 is not None


def test_lorenz_chart_names_the_gini_in_the_legend():
    fig = ch.lorenz_chart([0, 0.5, 1], [0, 0.2, 1], 0.62)
    assert fig.data[1].name == "Chart points (Gini 0.62)"
    assert fig.data[0].name == "Equal shares"


def test_ribbon_chart_draws_the_band_under_the_line():
    fig = ch.ribbon_chart(pd.DataFrame({"x": [1, 2], "m": [5, 6], "l": [4, 5], "h": [6, 7]}), "x", "m", "l", "h")
    assert fig.data[1].fill == "tonexty" and fig.data[2].name == "Median"


# ==================================================================================================================== helpers
def test_with_alpha_writes_an_rgba_colour():
    assert ch.with_alpha("#2a78d6", 0.5) == "rgba(42,120,214,0.5)"
    assert ch.with_alpha("2a78d6", 1) == "rgba(42,120,214,1)"


def test_shorten_cuts_long_labels_with_an_ellipsis():
    assert ch.shorten("short") == "short"
    cut = ch.shorten("x" * 60, 20)
    assert len(cut) == 20 and cut.endswith("…")
    assert ch.shorten(12345, 3) == "12…"


def test_log_axis_gets_readable_ticks():
    fig = ch.log_axis(go.Figure(go.Scatter(x=[1, 2], y=[1, 100])))
    assert fig.layout.yaxis.type == "log"
    assert list(fig.layout.yaxis.tickvals) == ch.LOG_TICKS
    assert list(fig.layout.yaxis.ticktext)[:3] == ["1", "2", "3"]
    other = ch.log_axis(go.Figure(go.Scatter(x=[1, 2], y=[1, 100])), axis="x", ticks=[1, 10, 100])
    assert other.layout.xaxis.type == "log" and list(other.layout.xaxis.ticktext) == ["1", "10", "100"]


def test_finish_places_the_legend_and_leaves_room_for_the_title():
    plain = ch.finish(go.Figure(go.Scatter(x=[1], y=[1])), legend="top")
    titled = ch.finish(go.Figure(go.Scatter(x=[1], y=[1])), title="A title", legend="top")
    assert plain.layout.legend.orientation == "h" and titled.layout.margin.t > plain.layout.margin.t
    assert ch.finish(go.Figure(), legend=None).layout.showlegend is False
    assert ch.finish(go.Figure(), legend="right").layout.legend.orientation == "v"
    assert ch.finish(go.Figure(), legend="bottom").layout.legend.y < 0
    axes = ch.finish(go.Figure(), x_title="X", y_title="Y")
    assert axes.layout.xaxis.title.text == "X" and axes.layout.yaxis.title.text == "Y"
