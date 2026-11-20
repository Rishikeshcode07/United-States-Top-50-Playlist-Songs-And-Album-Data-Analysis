"""
tests/test_eda_plots.py  -  tests for the chart building blocks (src/eda/plots.py).

WHAT THIS FILE IS
    Small drawings made with each helper (figure frame, bars, dots, legends, labels, KPI tile) and checks on what ended up in the
    picture: how many bars, which text, where the margins are, how big the PNG file is. It also covers the text helpers
    (protecting the dollar sign of A$AP Rocky, shortening a label, "1.2K", balanced wrapping of a headline) and the contrast rule
    that chooses white or dark text for numbers inside coloured cells.

WHY WE NEED IT
    A chart helper that is slightly wrong is wrong in thirty charts at once. Two of these checks were written after real mistakes:
    a dollar sign that turned a rapper's name into maths, and a headline whose last line was a single orphan word.

WHAT HAPPENS WHEN IT RUNS
    Everything is drawn with the non-interactive "Agg" backend (no window opens) inside a temporary matplotlib setting, so the
    project look does not leak into other tests. Pictures are closed after every test.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Make the right chart the easy chart, and keep it that way when somebody edits a helper.
"""
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib as mpl                                    # noqa: E402  (the backend has to be chosen before pyplot is used)
import matplotlib.pyplot as plt                             # noqa: E402
import numpy as np                                          # noqa: E402
import pytest                                               # noqa: E402
from matplotlib.axes import Axes                            # noqa: E402
from matplotlib.colors import to_rgba                       # noqa: E402

from src.eda import style                                   # noqa: E402
from src.eda.plots import (COMPACT, FIGURE_DPI, THOUSANDS, _wrap, clean_text, compact, contrast_ratio, direct_label, dot_row,  # noqa: E402
                           hbar, kpi_tile, make_fig, note, panel_title, percent_formatter, readable_ink, save_fig, short,
                           style_axes, swatch_legend)


@pytest.fixture(autouse=True)
def project_look():
    """Every test draws in the project look, and the look is taken away again afterwards."""
    with mpl.rc_context():
        style.apply_style()
        yield
    plt.close("all")


# ================================================================================ text helpers
def test_the_dollar_sign_is_protected_so_that_it_is_not_read_as_maths():
    assert clean_text("A$AP Rocky") == "A\\$AP Rocky"
    assert clean_text("$uicideboy$") == "\\$uicideboy\\$"
    assert clean_text("no dollars here") == "no dollars here"
    assert clean_text(12345) == "12345"


@pytest.mark.parametrize("text", ["A$AP Rocky", "$uicideboy$", "cost $5 and $6", "plain", "A\\$AP already protected"])
def test_protecting_twice_gives_the_same_text_as_protecting_once(text):
    assert clean_text(clean_text(text)) == clean_text(text)


def test_a_protected_name_draws_without_turning_into_maths():
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, clean_text("A$AP Rocky and $uicideboy$"))
    fig.canvas.draw()                                       # a dollar pair read as maths would be drawn in italics without the sign
    assert ax.texts[0].get_text() == "A\\$AP Rocky and \\$uicideboy\\$"


def test_short_cuts_long_labels_with_an_ellipsis_and_leaves_short_ones_alone():
    assert short("abcdef", 10) == "abcdef"
    assert short("abcdefghijk", 6) == "abcde…"
    assert short("ab cdefghi", 4) == "ab…"                  # no dangling space before the ellipsis
    assert short("A$AP Rocky is a rapper", 9) == "A\\$AP Roc…"


@pytest.mark.parametrize("value, decimals, expected", [
    (1234, 1, "1.2K"), (1_500_000, 1, "1.5M"), (85, 1, "85"), (0, 1, "0"), (999, 1, "999"), (1000, 1, "1K"), (2_000_000, 1, "2M"),
    (12_345, 1, "12.3K"), (258_819, 1, "258.8K"), (999_949, 1, "999.9K"), (999_999, 1, "1M"), (-1500, 1, "-1.5K"), (-999_999, 1, "-1M"),
    (1_234_567, 2, "1.23M"), (2000, 0, "2K"),
])
def test_compact_numbers(value, decimals, expected):
    assert compact(value, decimals) == expected


def test_the_axis_formatters():
    assert THOUSANDS(1_234_567, 0) == "1,234,567"
    assert COMPACT(2500, 0) == "2.5K"
    assert percent_formatter()(12, 0) == "12%" and percent_formatter(1)(12.34, 0) == "12.3%"


# ================================================================================ wrapping
HEADLINE = "The explicit share of the chart swung between 32% and 64% from month to month; singles averaged 29%, collaborations 11%"


def test_a_short_text_stays_on_one_line():
    assert _wrap("A short headline", 7.0, 15, 0.56) == "A short headline"


def test_a_long_text_is_wrapped_into_lines_of_similar_length():
    wrapped = _wrap(HEADLINE, 7.3, 15, 0.56)
    lines = wrapped.split("\n")
    assert len(lines) == 2
    assert len(lines[1]) >= 0.7 * len(lines[0]), f"orphan line: {lines!r}"           # not 'long line + two words'
    assert " ".join(lines) == HEADLINE                      # nothing lost, nothing added


def test_balancing_never_adds_a_line_and_never_exceeds_the_width():
    for width_in in (3.0, 4.2, 5.5, 7.3, 9.0):
        wrapped = _wrap(HEADLINE, width_in, 15, 0.56)
        limit = max(20, int(width_in * 72 / (15 * 0.56)))
        assert all(len(line) <= limit for line in wrapped.split("\n"))
        assert " ".join(wrapped.split("\n")) == HEADLINE


def test_a_tiny_width_is_raised_to_twenty_characters():
    assert all(len(line) <= 20 for line in _wrap("word " * 40, 0.1, 10, 0.5).split("\n"))


# ================================================================================ the figure frame
def test_the_shape_of_what_make_fig_returns():
    _, single = make_fig(8, 5)
    assert isinstance(single, Axes)
    _, row = make_fig(8, 5, 1, 3)
    assert row.shape == (3,)
    _, grid = make_fig(8, 5, 2, 2)
    assert grid.shape == (2, 2)


def test_headline_subtitle_and_note_are_drawn_on_the_figure():
    fig, _ = make_fig(9, 5, title="Half of all chart points belong to 52 songs", subtitle="Chart points by song", note_text="Source: Atlantic file")
    texts = [t.get_text() for t in fig.texts]
    assert texts == ["Half of all chart points belong to 52 songs", "Chart points by song", "Source: Atlantic file"]


def test_margins_are_given_in_inches():
    fig, ax = make_fig(8, 5, left=1.0, right=0.5, bottom=0.7)
    box = ax.get_position()
    assert box.x0 == pytest.approx(1.0 / 8) and box.x1 == pytest.approx(1 - 0.5 / 8) and box.y0 == pytest.approx(0.7 / 5)


def test_room_is_made_for_panel_titles_extra_space_and_a_footnote():
    _, plain = make_fig(8, 5, title="T", subtitle="S")
    _, with_titles = make_fig(8, 5, title="T", subtitle="S", panel_titles=True)
    _, with_extra = make_fig(8, 5, title="T", subtitle="S", top_extra=0.5)
    _, with_note = make_fig(8, 5, title="T", subtitle="S", note_text="n")
    assert plain.get_position().y1 - with_titles.get_position().y1 == pytest.approx(0.3 / 5)
    assert plain.get_position().y1 - with_extra.get_position().y1 == pytest.approx(0.5 / 5)
    assert with_note.get_position().y0 - plain.get_position().y0 == pytest.approx(0.14 / 5)


def test_a_headline_that_needs_two_lines_pushes_the_panels_down():
    _, one_line = make_fig(8, 5, title="Short")
    fig, two_lines = make_fig(8, 5, title=HEADLINE)
    assert fig.texts[0].get_text().count("\n") == 1
    assert one_line.get_position().y1 - two_lines.get_position().y1 == pytest.approx(0.27 / 5)


def test_shared_axes():
    _, axes = make_fig(8, 5, 1, 2, sharey=True)
    assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])


def test_save_fig_writes_a_png_of_the_expected_size_and_creates_the_folder(tmp_path):
    fig, ax = make_fig(6, 4)
    ax.plot([0, 1], [0, 1])
    path = save_fig(fig, "chart", tmp_path / "reports" / "figures")
    assert path == tmp_path / "reports" / "figures" / "chart.png" and path.exists()
    assert FIGURE_DPI == 150
    assert plt.imread(path).shape[:2] == (4 * 150, 6 * 150)        # no 'tight' cropping: always the same size


# ================================================================================ axes and marks
def test_style_axes_keeps_only_the_grid_on_the_value_axis():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    style_axes(ax, grid="y")
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    assert ax.spines["left"].get_visible() and ax.spines["bottom"].get_visible()
    assert ax.yaxis.get_gridlines()[0].get_visible() and not ax.xaxis.get_gridlines()[0].get_visible()
    style_axes(ax, grid=None, left_spine=False, bottom_spine=False)
    assert not ax.spines["left"].get_visible() and not ax.spines["bottom"].get_visible()


def test_panel_title_is_bold_left_aligned_and_protects_the_dollar_sign():
    fig, ax = plt.subplots()
    panel_title(ax, "A$AP Rocky")
    assert ax.get_title(loc="left") == "A\\$AP Rocky"
    assert ax.title.get_fontweight() in ("bold", 700) or ax._left_title.get_fontweight() in ("bold", 700)


def test_hbar_draws_one_bar_per_value_with_the_first_on_top_and_the_value_at_the_tip():
    fig, ax = plt.subplots()
    bars = hbar(ax, ["alpha", "beta", "gamma"], [3, 2, 1], fmt="{:.0f}")
    assert [b.get_width() for b in bars] == [3, 2, 1]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["alpha", "beta", "gamma"]
    assert ax.get_ylim()[0] > ax.get_ylim()[1]                              # inverted: the first label is on top
    assert [t.get_text() for t in ax.texts] == ["3", "2", "1"]
    assert all(t.get_position()[0] > b.get_width() for t, b in zip(ax.texts, bars))       # the number sits beyond the tip


def test_hbar_options():
    fig, ax = plt.subplots()
    bars = hbar(ax, ["A$AP Rocky", "a very long artist name indeed"], [5, 4], colors=[style.ORANGE, style.AQUA], fmt="{:.1f}%",
                label_chars=12, xlim=(0, 10))
    assert [t.get_text() for t in ax.get_yticklabels()] == ["A\\$AP Rocky", "a very long…"]
    assert [t.get_text() for t in ax.texts] == ["5.0%", "4.0%"]
    assert bars[0].get_facecolor() == to_rgba(style.ORANGE) and bars[1].get_facecolor() == to_rgba(style.AQUA)
    assert ax.get_xlim() == (0, 10)
    fig2, ax2 = plt.subplots()
    hbar(ax2, ["x"], [1], show_values=False)
    assert len(ax2.texts) == 0


def test_dot_row_draws_one_dot_per_value_with_its_number():
    fig, ax = plt.subplots()
    dot_row(ax, ["one", "two", "three"], [81.0, 85.5, 90.2], xlim=(78, 94))
    assert len(ax.collections[-1].get_offsets()) == 3
    assert [t.get_text() for t in ax.texts] == ["81.0", "85.5", "90.2"]
    assert ax.get_xlim() == (78, 94) and ax.get_ylim()[0] > ax.get_ylim()[1]


def test_direct_label_is_ink_coloured_and_can_be_offset():
    fig, ax = plt.subplots()
    plain = direct_label(ax, 1, 2, "A$AP Rocky")
    offset = direct_label(ax, 1, 2, "moved", dx=6, dy=-4)
    assert plain.get_text() == "A\\$AP Rocky" and to_rgba(plain.get_color()) == to_rgba(style.INK_2)
    assert offset.get_text() == "moved" and offset.xyann == (6, -4)


def test_note_has_a_leader_line_unless_it_is_switched_off():
    fig, ax = plt.subplots()
    assert note(ax, (1, 1), "with leader", (2, 2)).arrow_patch is not None
    assert note(ax, (1, 1), "without", (2, 2), leader=False).arrow_patch is None


def test_swatch_legend_entries_and_kinds():
    fig, ax = plt.subplots()
    legend = swatch_legend(ax, [("squares", style.BLUE), ("a line", style.ORANGE, "line"), ("dots", style.AQUA, "dot")])
    assert [t.get_text() for t in legend.get_texts()] == ["squares", "a line", "dots"]
    markers = [handle.get_marker() for handle in legend.legend_handles]
    assert markers == ["s", "None", "o"]


def test_swatch_legend_can_sit_flush_against_the_axes_and_can_be_opaque():
    fig, ax = plt.subplots()
    assert swatch_legend(ax, [("a", style.BLUE)]).borderaxespad == pytest.approx(0.2)
    flush = swatch_legend(ax, [("a", style.BLUE)], borderaxespad=0, frameon=True, facecolor=style.SURFACE, framealpha=1)
    assert flush.borderaxespad == 0 and flush.get_frame_on() and flush.get_frame().get_alpha() == 1


def test_kpi_tile_shows_label_value_and_context_and_an_optional_sparkline():
    fig, ax = plt.subplots(figsize=(3, 2))
    kpi_tile(ax, "Days on the chart", "7", "median stay of a song", spark=[1, 3, 2, 5, 4])
    assert not ax.axison
    assert [t.get_text() for t in ax.texts] == ["Days on the chart", "7", "median stay of a song"]
    assert len(ax.child_axes) == 1
    fig2, ax2 = plt.subplots(figsize=(3, 2))
    kpi_tile(ax2, "Songs", "944", "in the file", spark=[5])             # one number is not a line
    assert len(ax2.child_axes) == 0


# ================================================================================ text on coloured cells
def test_contrast_ratio_follows_the_wcag_scale():
    assert contrast_ratio("black", "white") == pytest.approx(21.0)
    assert contrast_ratio("white", "white") == pytest.approx(1.0)
    assert contrast_ratio("#2a78d6", "white") == pytest.approx(contrast_ratio("white", "#2a78d6"))
    assert contrast_ratio((0, 0, 0, 1.0), "#ffffff") == pytest.approx(21.0)            # RGBA tuples are accepted


def test_readable_ink_picks_white_on_dark_cells_and_ink_on_light_cells():
    assert readable_ink(style.BLUE_STEPS[650]) == "white"
    assert readable_ink(style.BLUE_STEPS[100]) == style.INK
    assert readable_ink("#000000", light="#ffffff", dark="#222222") == "#ffffff"
    assert readable_ink("#ffffff", light="#eeeeee", dark="#222222") == "#222222"


def test_the_chosen_ink_always_beats_the_other_choice_on_every_step_of_the_blue_scale():
    for step in style.BLUE_STEPS.values():
        chosen = readable_ink(step)
        other = style.INK if chosen == "white" else "white"
        assert contrast_ratio(step, chosen) >= contrast_ratio(step, other)
    for value in np.linspace(-1, 1, 41):                                                # and on the whole diverging scale
        cell = style.DIVERGING((value + 1) / 2)
        chosen = readable_ink(cell)
        assert contrast_ratio(cell, chosen) >= 4.0, f"value {value:.2f}"
