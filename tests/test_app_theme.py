"""
tests/test_app_theme.py  -  the colours and the style sheet of the dashboard (app/ui/theme.py).

WHAT THIS FILE IS
    Tests that the dashboard palette is the same as the palette of the notebook charts (src/eda/style.py), that the colours in
    the CSS are the colours in Python, that small text is readable (contrast), that each meaning has its own colour, and that
    the style sheet contains no emoji.

WHY WE NEED IT
    The notebooks, the paper and the dashboard must look like one project. The palette lives in two places (matplotlib for the
    notebooks, Plotly and CSS for the dashboard); without a test the two would drift apart the first time someone changes a
    colour in only one of them.

WHAT HAPPENS WHEN IT RUNS
    Pure checks on constants and strings. The Plotly template is registered under a test name and the previous default is put
    back afterwards, so other tests are not affected.

EXPECTED OUTCOME
    All tests pass. The test of the notebook palette is skipped when matplotlib is not installed.

MOTIVE
    One palette, in every place that draws a colour.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("plotly")

from app.ui import theme  # noqa: E402  (after the importorskip on purpose: the module needs Plotly)

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _luminance(color: str) -> float:
    channels = []
    for i in (1, 3, 5):
        value = int(color[i:i + 2], 16) / 255
        channels.append(value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(foreground: str, background: str) -> float:
    a, b = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (a + 0.05) / (b + 0.05)


# ------------------------------------------------------------------------------------------- same palette everywhere
def test_the_palette_is_the_one_of_the_notebook_charts():
    pytest.importorskip("matplotlib")
    from src.eda import style
    assert theme.CATEGORICAL == list(style.CATEGORICAL)
    assert (theme.INK, theme.MUTED, theme.GRID, theme.AXIS, theme.SURFACE, theme.PAGE) == (style.INK, style.MUTED, style.GRID, style.AXIS, style.SURFACE, style.PAGE)
    assert theme.DIVERGING == list(style.DIVERGING_NODES)
    assert dict(theme.LONGEVITY_COLORS) == dict(style.LONGEVITY_COLORS)


def test_the_css_colour_tokens_are_the_python_colours():
    tokens = dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", theme.CSS.split("}")[0]))
    expected = {"blue": theme.BLUE, "blue-dark": theme.BLUE_DARK, "blue-soft": theme.BLUE_SOFT, "orange": theme.ORANGE, "aqua": theme.AQUA, "yellow": theme.YELLOW,
                "pink": theme.PINK, "green": theme.GREEN, "violet": theme.VIOLET, "red": theme.RED, "ink": theme.INK, "ink2": theme.INK_2, "muted": theme.MUTED,
                "grid": theme.GRID, "axis": theme.AXIS, "page": theme.PAGE, "card": theme.CARD}
    for name, value in expected.items():
        assert tokens.get(name, "").lower() == value.lower(), f"--{name} in the CSS differs from the Python colour"


def test_every_colour_is_a_six_digit_hex_code():
    palettes = [theme.CATEGORICAL, theme.SEQUENTIAL, theme.DIVERGING, list(theme.EXPLICIT_COLORS.values()), list(theme.ALBUM_COLORS.values()),
                list(theme.COLLAB_COLORS.values()), list(theme.MOVE_COLORS.values()), list(theme.LONGEVITY_COLORS.values()), list(theme.BAND_COLORS.values()),
                list(theme.ARCHETYPE_COLORS.values()), list(theme.SEGMENT_COLORS.values())]
    for palette in palettes:
        for color in palette:
            assert HEX.match(color), color


# ------------------------------------------------------------------------------------------------- meaning and contrast
def test_each_meaning_has_its_own_colour():
    assert theme.EXPLICIT_COLORS["Explicit"] != theme.EXPLICIT_COLORS["Not explicit"]
    assert theme.MOVE_COLORS["Up"] != theme.MOVE_COLORS["Down"]
    assert theme.COLLAB_COLORS["Solo"] != theme.COLLAB_COLORS["Collaboration"]
    assert len(set(theme.LONGEVITY_COLORS.values())) == len(theme.LONGEVITY_COLORS) == 5
    assert len(set(theme.BAND_COLORS.values())) == len(theme.BAND_COLORS) == 5
    assert len(set(theme.ARCHETYPE_COLORS.values())) == len(theme.ARCHETYPE_COLORS) == 5
    assert len(set(theme.SEGMENT_COLORS.values())) == len(theme.SEGMENT_COLORS) == 4
    assert len(set(theme.CATEGORICAL)) == len(theme.CATEGORICAL) == 8


def test_a_colour_keeps_its_meaning_on_every_page():
    assert theme.EXPLICIT_COLORS["Explicit"] == theme.ORANGE
    assert theme.MOVE_COLORS["Up"] == theme.GREEN and theme.MOVE_COLORS["Down"] == theme.RED
    assert theme.ALBUM_COLORS["single"] == theme.ALBUM_COLORS["Single"] == theme.AQUA
    assert theme.ALBUM_COLORS["album"] == theme.ALBUM_COLORS["Album"] == theme.VIOLET


@pytest.mark.parametrize("name, color", [("ink", theme.INK), ("ink 2", theme.INK_2), ("muted", theme.MUTED)])
def test_text_colours_are_readable_on_the_chart_background(name, color):
    """4.5 : 1 is the usual minimum for small text (WCAG AA)."""
    assert _contrast(color, theme.SURFACE) >= 4.5, name
    assert _contrast(color, theme.CARD) >= 4.5, name


def test_the_longevity_colours_get_darker_with_the_stay():
    shades = [_luminance(theme.LONGEVITY_COLORS[label]) for label in theme.LONGEVITY_COLORS]
    assert shades == sorted(shades, reverse=True)


# --------------------------------------------------------------------------------------------------------------- helpers
def test_colors_for_uses_the_meaning_first_and_the_palette_after():
    colors = theme.colors_for(["Explicit", "other", "another"], theme.EXPLICIT_COLORS)
    assert colors["Explicit"] == theme.ORANGE
    assert colors["other"] == theme.CATEGORICAL[0] and colors["another"] == theme.CATEGORICAL[1]


def test_colors_for_never_runs_out_of_colours():
    names = [f"group {i}" for i in range(25)]
    colors = theme.colors_for(names)
    assert len(colors) == 25 and all(HEX.match(c) for c in colors.values())


def test_the_template_is_registered_and_made_the_default():
    import plotly.io as pio
    before = pio.templates.default
    try:
        name = theme.register_template("atlantic_test")
        assert name == "atlantic_test" and pio.templates.default == "atlantic_test"
        layout = pio.templates["atlantic_test"].layout
        assert list(layout.colorway) == theme.CATEGORICAL
        assert layout.font.color == theme.INK_2
    finally:
        pio.templates.default = before


# ------------------------------------------------------------------------------------------------------------------ CSS
def test_the_style_sheet_has_no_emoji():
    emoji = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐⬆⬇]")
    assert not emoji.search(theme.CSS)


def test_the_style_sheet_has_balanced_braces():
    assert theme.CSS.count("{") == theme.CSS.count("}")


def test_the_style_sheet_needs_no_download():
    """System fonts only: the page must look the same offline (no @import, no remote font or image)."""
    assert "@import" not in theme.CSS and "http://" not in theme.CSS and "https://" not in theme.CSS
