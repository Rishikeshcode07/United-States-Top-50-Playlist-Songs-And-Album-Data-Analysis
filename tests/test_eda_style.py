"""
tests/test_eda_style.py  -  tests for the colours and the matplotlib defaults of the Phase 4 charts (src/eda/style.py).

WHAT THIS FILE IS
    Checks of the claims that the colour file makes about itself: the eight hues are the published ones in the published order,
    an entity keeps its colour, ordered groups get darker one step at a time, the two arms of the diverging scale have the same
    lightness and colour strength, and small text can be read on the chart background.

WHY WE NEED IT
    The colour file says "all checks pass" and "the tests check that lightness matches step for step". A sentence like that is
    only worth something if a test would fail the day somebody edits one hex code. The tests measure lightness and chroma in the
    OKLab colour space (where equal numbers look equally different) and contrast with the WCAG formula.

WHAT HAPPENS WHEN IT RUNS
    Hex codes are turned into numbers and compared with the documented limits. `apply_style` is run inside a temporary
    matplotlib setting so that the other tests are not affected.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    A palette that stays validated after the next edit, and a dashboard that looks like the notebook.
"""
import re

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.colors import to_hex, to_rgb

from src.config import LONGEVITY_LABELS, RANK_TIER_LABELS
from src.eda import style
from src.eda.plots import contrast_ratio


def oklab(color):
    """(lightness L from 0 to 1, chroma) of a colour in the OKLab space."""
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(color)]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    lightness = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    b2 = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return float(lightness), float(np.hypot(a, b2))


def lightness(color):
    return oklab(color)[0]


# ================================================================================ the categorical hues
def test_eight_distinct_hex_colours_in_the_published_order():
    assert len(style.CATEGORICAL) == 8 == len(set(style.CATEGORICAL))
    assert all(re.fullmatch(r"#[0-9a-f]{6}", colour) for colour in style.CATEGORICAL)
    assert style.CATEGORICAL == ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    assert (style.BLUE, style.ORANGE, style.AQUA, style.YELLOW, style.MAGENTA, style.GREEN, style.VIOLET, style.RED) == tuple(style.CATEGORICAL)


def test_an_entity_keeps_its_colour_in_every_chart():
    assert style.EXPLICIT_COLOR == style.ORANGE and style.CLEAN_COLOR == style.BLUE
    assert style.SINGLE_COLOR == style.AQUA and style.ALBUM_COLOR == style.VIOLET
    assert len({style.EXPLICIT_COLOR, style.CLEAN_COLOR, style.SINGLE_COLOR, style.ALBUM_COLOR}) == 4


def test_exactly_aqua_yellow_and_magenta_are_below_three_to_one_on_the_chart_background():
    """The colour file discloses this validator warning; the rule that goes with it is 'label every mark directly'."""
    weak = {name for name in ("BLUE", "ORANGE", "AQUA", "YELLOW", "MAGENTA", "GREEN", "VIOLET", "RED")
            if contrast_ratio(getattr(style, name), style.SURFACE) < 3.0}
    assert weak == {"AQUA", "YELLOW", "MAGENTA"}


# ================================================================================ ordered groups
def test_the_five_step_ramp_gets_darker_by_a_clear_amount_at_every_step():
    steps = [lightness(colour) for colour in style._FIVE_STEPS]
    differences = -np.diff(steps)
    assert (differences >= 0.06).all(), "neighbouring steps must differ by at least 0.06 in lightness"
    assert contrast_ratio(style._FIVE_STEPS[0], style.SURFACE) >= 2.0          # even the lightest step can be seen on the background


def test_rank_tiers_and_longevity_classes_use_the_labels_of_the_project():
    assert list(style.TIER_COLORS) == RANK_TIER_LABELS
    assert list(style.LONGEVITY_COLORS) == LONGEVITY_LABELS
    assert len(style.AGE_BAND_COLORS) == 5


def test_the_more_important_or_longer_lived_group_is_the_darker_one():
    tiers = [lightness(style.TIER_COLORS[label]) for label in RANK_TIER_LABELS]
    assert tiers == sorted(tiers) and len(set(tiers)) == 3                      # Top 10 darkest, 21-50 lightest
    classes = [lightness(style.LONGEVITY_COLORS[label]) for label in LONGEVITY_LABELS]
    assert classes == sorted(classes, reverse=True) and len(set(classes)) == 5  # Flash lightest, Evergreen darkest
    ages = [lightness(colour) for colour in style.AGE_BAND_COLORS]
    assert ages == sorted(ages, reverse=True)


def test_the_blue_steps_go_from_light_to_dark_without_a_gap_or_a_reversal():
    steps = [lightness(style.BLUE_STEPS[key]) for key in sorted(style.BLUE_STEPS)]
    assert (np.diff(steps) < 0).all()
    samples = [lightness(style.SEQUENTIAL_BLUE(x)) for x in np.linspace(0, 1, 21)]
    assert (np.diff(samples) < 0).all()


# ================================================================================ the diverging scale
def test_the_two_arms_have_the_same_lightness_and_colour_strength_step_for_step():
    nodes = style.DIVERGING_NODES
    assert len(nodes) == 7 and nodes[3] == style.NEUTRAL_MID
    for step in range(3):
        blue_l, blue_c = oklab(nodes[step])
        red_l, red_c = oklab(nodes[6 - step])
        assert blue_l == pytest.approx(red_l, abs=0.005), f"step {step}: lightness"
        assert blue_c == pytest.approx(red_c, abs=0.005), f"step {step}: chroma"


def test_the_diverging_colormap_runs_from_red_through_grey_to_blue():
    colormap = style.DIVERGING
    assert to_hex(colormap(0.0)) == style.DIVERGING_NODES[-1]          # darkest red = the lowest value
    assert to_hex(colormap(1.0)) == style.DIVERGING_NODES[0]           # darkest blue = the highest value
    assert np.allclose(to_rgb(colormap(0.5)), to_rgb(style.NEUTRAL_MID), atol=0.005)      # the colormap is a 256-entry table: one step of rounding
    red, blue = to_rgb(colormap(0.0)), to_rgb(colormap(1.0))
    assert red[0] > red[2] and blue[2] > blue[0]


def test_climbed_and_fell_have_the_same_visual_weight():
    assert lightness(style.UP_COLOR) == pytest.approx(lightness(style.DOWN_COLOR), abs=0.005)
    assert style.UP_COLOR != style.DOWN_COLOR and style.SAME_COLOR == style.AXIS


# ================================================================================ text and background
def test_text_can_be_read_on_the_chart_background():
    assert contrast_ratio(style.INK, style.SURFACE) >= 15
    assert contrast_ratio(style.INK_2, style.SURFACE) >= 7
    assert contrast_ratio(style.MUTED, style.SURFACE) >= 4.5            # footnotes carry caveats, so they must be readable


def test_the_grid_is_quiet_and_the_axis_is_visible():
    assert contrast_ratio(style.GRID, style.SURFACE) < contrast_ratio(style.AXIS, style.SURFACE) < 3


# ================================================================================ matplotlib defaults
def test_apply_style_sets_the_project_defaults_and_can_be_repeated():
    with mpl.rc_context():
        style.apply_style()
        first = dict(mpl.rcParams)
        assert mpl.rcParams["figure.facecolor"] == style.SURFACE and mpl.rcParams["axes.facecolor"] == style.SURFACE
        assert list(mpl.rcParams["font.sans-serif"][:3]) == style.FONT_STACK
        assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == style.CATEGORICAL
        assert not mpl.rcParams["axes.spines.top"] and not mpl.rcParams["axes.spines.right"]
        assert not mpl.rcParams["axes.grid"] and not mpl.rcParams["legend.frameon"]
        assert mpl.rcParams["savefig.dpi"] == 150
        style.apply_style()
        assert dict(mpl.rcParams) == first


def test_apply_style_inside_a_context_leaves_the_outside_world_alone():
    before = mpl.rcParams["figure.facecolor"]
    with mpl.rc_context():
        style.apply_style()
    assert mpl.rcParams["figure.facecolor"] == before
