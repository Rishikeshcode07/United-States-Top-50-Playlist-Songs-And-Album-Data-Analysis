"""
src/eda/style.py  -  the look of every Phase 4 chart (colours, fonts, line widths) in ONE place.

WHAT THIS FILE IS
    The colours and the matplotlib settings of the project's charts:

        categorical colours   eight hues in a fixed order, used for "which series is this" (CATEGORICAL, BLUE, ORANGE ...)
        ordinal ramps         one blue in several steps, used for ordered groups (rank tiers, longevity classes, age bands)
        sequential colormap   one blue from light to dark, used for "how much" (heat maps)
        diverging colormap    blue on one side, red on the other, a neutral grey in the middle, used for "which side of zero"
        ink and surface       text, grid and background colours
        apply_style()         sets the matplotlib defaults (font, grid, spines, line width)

WHY WE NEED IT
    Colour is information. If "explicit" is orange in one chart and green in the next, the reader has to re-learn the chart
    every time. Here an entity keeps its colour in every figure of the project (explicit = orange, not explicit = blue,
    single = aqua, album = violet), and the same file can later feed the Streamlit theme and the Power BI theme.

HOW THE COLOURS WERE CHOSEN (so that you can trust them, not just like them)
    * The eight categorical hues and their order come from a published, validated palette. The order is the colour-blind
      safety mechanism, so it never changes. A validator checked them (lightness band, colour strength, distance between
      neighbours under red-green colour blindness and under normal vision, contrast against the chart background):
      all checks pass. Three of the hues (aqua, yellow, magenta) have a contrast below 3:1 on a light background, the
      validator's warning. The rule that goes with the warning is followed: those colours are never the only carrier of a
      value, every mark is labelled directly or listed in a table.
    * A chart that compares many series with many touching marks (scatter plots, small multiples) uses at most the first three
      hues, because only those three pass the stricter "any two can be neighbours" check.
    * The ordinal ramps are steps of one blue. The validator's ordinal mode confirmed: lightness only goes one way, every step
      differs from its neighbour by at least 0.06 in lightness, and the lightest step still has 2:1 contrast on the background.
    * The red arm of the diverging scale was derived by us (the palette names only "blue and red"): for every blue step we
      took the red hue of categorical slot 8 at the SAME lightness and the same colour strength, so that the two arms have
      the same visual weight. The tests check that lightness matches step for step.

WHAT HAPPENS WHEN IT RUNS
    Importing the file defines the colours. `apply_style()` changes matplotlib's default settings for the whole session.

EXPECTED OUTCOME
    Charts that look like one family: thin marks, hairline grid, no chart junk, the data is the loudest thing on the page.

MOTIVE
    Consistency across the notebook, the research paper, the Streamlit app and Power BI.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from src.config import LONGEVITY_LABELS, RANK_TIER_LABELS

# ---------------------------------------------------------------------------------------------- surface and ink
SURFACE = "#fcfcfb"        # the chart background
PAGE = "#f9f9f7"           # a slightly different plane for tiles
INK = "#0b0b0b"            # titles and values
INK_2 = "#52514e"          # axis titles, labels, secondary text
MUTED = "#6b6a65"          # footnotes and small labels: 5.3:1 on the chart background (4.5:1 is the usual minimum for small text)
GRID = "#e1e0d9"           # hairline grid
AXIS = "#c3c2b7"           # axis lines, and the grey of "context" marks in emphasis charts
NEUTRAL_MID = "#f0efec"    # the middle of the diverging scale

# Arial first on purpose: Liberation Sans (Linux) has exactly the same letter widths as Arial (Windows, macOS), so a chart
# checked on one machine has the same layout on the other. DejaVu Sans is matplotlib's own fallback.
FONT_STACK = ["Arial", "Liberation Sans", "DejaVu Sans"]

# ----------------------------------------------------------------------------------------------- categorical hues
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = CATEGORICAL

# An entity keeps its colour in every chart of the project.
EXPLICIT_COLOR, CLEAN_COLOR = ORANGE, BLUE        # explicit / not explicit
SINGLE_COLOR, ALBUM_COLOR = AQUA, VIOLET          # single / album

# Direction of a move: climbed = blue, fell = red (the two arms of the diverging scale), stayed = the neutral grey.
# (Defined below, after DIVERGING_NODES.)

# ------------------------------------------------------------------------------------------------- sequential blue
BLUE_STEPS = {
    100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 300: "#6da7ec", 350: "#5598e7", 400: "#3987e5",
    450: "#2a78d6", 500: "#256abf", 550: "#1c5cab", 600: "#184f95", 650: "#104281", 700: "#0d366b",
}

# ----------------------------------------------------------------------------------------------- ordinal ramps
# Ordered groups: the more important or the longer-lived the group, the darker the blue.
_TIER_STEPS = ["#1c5cab", "#3987e5", "#86b6ef"]                                           # darkest = best positions
TIER_COLORS = dict(zip(RANK_TIER_LABELS, _TIER_STEPS))                                    # Top 10, 11-20, 21-50

_FIVE_STEPS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]                     # lightest -> darkest
LONGEVITY_COLORS = dict(zip(LONGEVITY_LABELS, _FIVE_STEPS))                               # Flash ... Evergreen
AGE_BAND_COLORS = _FIVE_STEPS                                                              # newest ... oldest

# A six-step blue ramp does NOT pass the validator (neighbouring steps get too close in lightness), so ordered groups with six
# members (the album-size bands) are drawn in one blue and their order is carried by the axis.

# ----------------------------------------------------------------------------------------------- colormaps
# Diverging: blue (positive) <-> red (negative). The red arm has the same lightness as the blue arm, step for step.
DIVERGING_NODES = ["#104281", "#2a78d6", "#b7d3f6", NEUTRAL_MID, "#f4c2be", "#c74845", "#762221"]
BLUE_RED_POSITIVE_FIRST = DIVERGING_NODES          # documented name for the tests
UP_COLOR, SAME_COLOR, DOWN_COLOR = DIVERGING_NODES[1], AXIS, DIVERGING_NODES[5]     # climbed / stayed / fell

SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("atlantic_blue", [BLUE_STEPS[k] for k in sorted(BLUE_STEPS)])
DIVERGING = LinearSegmentedColormap.from_list("atlantic_diverging", list(reversed(DIVERGING_NODES)))
# DIVERGING runs from red (low) through grey (0) to blue (high): high = blue = "more of it"


def apply_style():
    """Set the matplotlib defaults for the project (call it once at the top of a notebook)."""
    mpl.rcParams.update({
        # surfaces
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 100, "savefig.dpi": 150,
        # text
        "font.family": "sans-serif", "font.sans-serif": FONT_STACK, "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.labelsize": 9.5, "axes.titlesize": 11, "axes.titlecolor": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.major.size": 3, "ytick.major.size": 3,
        # axes and grid: one hairline, never dashed
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        # marks
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "patch.linewidth": 0.0, "patch.edgecolor": SURFACE,
        # legends and layout
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "figure.constrained_layout.use": False,
    })
