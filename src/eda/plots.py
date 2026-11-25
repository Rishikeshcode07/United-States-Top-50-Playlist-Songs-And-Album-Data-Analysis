"""
src/eda/plots.py  -  the building blocks that every Phase 4 chart is made of.

WHAT THIS FILE IS
    A handful of small helpers around matplotlib:

        make_fig(...)        a figure with a left-aligned headline, a one-line subtitle, a footnote and a grid of panels. The margins
                             are calculated in INCHES from the amount of text, so a long headline can never overlap the chart
        style_axes(...)      hairline grid on the axis that carries the values, no top/right frame, quiet ticks
        hbar(...)            horizontal bars with the value written at the tip of every bar (the default "compare" chart)
        dot_row(...)         horizontal dots on a value axis that does not start at zero (for numbers that live in a narrow range)
        direct_label(...)    text next to a mark, in ink colour (text never wears the data colour)
        note(...)            a short annotation with a thin leader line
        swatch_legend(...)   a legend made of small squares or lines in the colours of the series
        kpi_tile(...)        one "stat tile": label, big value, one line of context and an optional sparkline
        readable_ink(...)    white or ink text, whichever reads better on a given background (numbers inside heat-map cells)
        save_fig(...)        write the PNG (always the same size, 150 dots per inch) and return its path
        short(...), compact(...), clean_text(...)   label helpers (shorten long titles, 1,200 -> 1.2K, protect "$")

WHY WE NEED IT
    Thirty-odd charts written with raw matplotlib calls would have thirty different margins, title sizes and label styles.
    Here the decisions are made once: a chart states its finding in the headline, names the measure in the subtitle, labels
    its marks directly and keeps the grid out of the way.

WHAT HAPPENS WHEN IT RUNS
    Nothing by itself; the notebook imports the helpers. `save_fig` writes a PNG file into reports/figures/.

EXPECTED OUTCOME
    Figures of identical size and typography. The tests draw each helper once and check the output file.

MOTIVE
    Make the right chart the easy chart.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

from src.eda import style

FIGURE_DPI = 150
TITLE_SIZE = 15
SUBTITLE_SIZE = 10.5
NOTE_SIZE = 8
HEADER_X_IN = 0.35                 # distance of the headline from the left edge of the picture, in inches


# =============================================================================================== text helpers
def clean_text(text):
    """
    matplotlib treats a pair of dollar signs as maths. Song and artist names such as A$AP Rocky need the sign protected.
    Safe to call twice: a dollar sign that is already protected is not protected again.
    """
    return str(text).replace("\\$", "$").replace("$", "\\$")


def short(text, max_chars):
    """Cut a long label to `max_chars` characters and end it with an ellipsis; the dollar sign is protected as well."""
    text = str(text)
    if len(text) > max_chars:
        text = text[: max(1, max_chars - 1)].rstrip() + "…"
    return clean_text(text)


def compact(value, decimals=1):
    """1234 -> '1.2K', 1500000 -> '1.5M', 85 -> '85'. A number that would round up to '1000K' is written as '1M'."""
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1_000_000 or round(magnitude / 1_000, decimals) >= 1_000:
        text, suffix = f"{magnitude / 1_000_000:.{decimals}f}", "M"
    elif magnitude >= 1_000:
        text, suffix = f"{magnitude / 1_000:.{decimals}f}", "K"
    else:
        return f"{value:,.0f}"
    if decimals and text.endswith("." + "0" * decimals):
        text = text[: -(decimals + 1)]                              # 2.0 -> 2
    return f"{sign}{text}{suffix}"


THOUSANDS = FuncFormatter(lambda value, _: f"{value:,.0f}")
COMPACT = FuncFormatter(lambda value, _: compact(value))


def percent_formatter(decimals=0):
    return FuncFormatter(lambda value, _: f"{value:.{decimals}f}%")


# =============================================================================================== the figure
def _wrap(text, width_in, font_size, factor):
    """
    Wrap a text so that it fits into `width_in` inches. `factor` = average letter width as a fraction of the font size.
    A text that needs more than one line is wrapped again with the SAME number of lines but a narrower width, so that the
    lines have similar lengths (no single orphan word on the last line).
    """
    text = str(text)
    characters = max(20, int(width_in * 72 / (font_size * factor)))
    lines = textwrap.wrap(text, width=characters)
    if len(lines) > 1:
        balanced = -(-len(text) // len(lines)) + 8              # ceiling of the average line length, plus a little slack
        candidate = textwrap.wrap(text, width=min(characters, balanced))
        if len(candidate) == len(lines):
            lines = candidate
    return "\n".join(lines)


def make_fig(width=11.0, height=6.0, nrows=1, ncols=1, *, title=None, subtitle=None, note_text=None, left=0.9, right=0.45,
             bottom=0.7, panel_titles=False, top_extra=0.0, hspace=0.4, wspace=0.28, width_ratios=None, height_ratios=None,
             sharex=False, sharey=False):
    """
    A figure with a headline (what the chart says), a subtitle (what is measured, in which unit), a footnote and
    nrows x ncols panels. Returns (fig, axes): a single Axes for a 1 x 1 grid, a 1-D array for one row or one column,
    otherwise a 2-D array. Sizes are in inches. `panel_titles=True` leaves room for a one-line title above every panel; `top_extra`
    adds more room (in inches) above the panels, for example for two-line panel titles.
    """
    fig = plt.figure(figsize=(width, height))
    text_width = width - 2 * HEADER_X_IN

    top = 0.2
    if title:
        wrapped = _wrap(title, text_width, TITLE_SIZE, 0.56)
        fig.text(HEADER_X_IN / width, 1 - 0.2 / height, wrapped, ha="left", va="top", fontsize=TITLE_SIZE,
                 fontweight="bold", color=style.INK, linespacing=1.15)
        top += wrapped.count("\n") * 0.27 + 0.27
    if subtitle:
        wrapped = _wrap(subtitle, text_width, SUBTITLE_SIZE, 0.5)
        fig.text(HEADER_X_IN / width, 1 - (top + 0.04) / height, wrapped, ha="left", va="top", fontsize=SUBTITLE_SIZE,
                 color=style.INK_2, linespacing=1.2)
        top += wrapped.count("\n") * 0.2 + 0.2 + 0.04
    top += 0.22 + (0.3 if panel_titles else 0.0) + top_extra

    bottom_margin = bottom
    if note_text:
        wrapped = _wrap(note_text, text_width, NOTE_SIZE, 0.5)
        fig.text(HEADER_X_IN / width, 0.12 / height, wrapped, ha="left", va="bottom", fontsize=NOTE_SIZE, color=style.MUTED,
                 linespacing=1.25)
        bottom_margin += wrapped.count("\n") * 0.14 + 0.14

    grid = fig.add_gridspec(nrows, ncols, left=left / width, right=1 - right / width, bottom=bottom_margin / height,
                            top=1 - top / height, hspace=hspace, wspace=wspace, width_ratios=width_ratios,
                            height_ratios=height_ratios)
    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            keywords = {}
            if sharex and (i, j) != (0, 0):
                keywords["sharex"] = axes[0, 0]
            if sharey and (i, j) != (0, 0):
                keywords["sharey"] = axes[0, 0]
            axes[i, j] = fig.add_subplot(grid[i, j], **keywords)
    if sharex or sharey:
        for ax in axes.ravel():
            ax.label_outer()
    return fig, (axes.item() if axes.size == 1 else axes.squeeze())


def save_fig(fig, name, directory):
    """Write <directory>/<name>.png at 150 dots per inch (same size every time: no 'tight' cropping) and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    return path


# =============================================================================================== axes and marks
def style_axes(ax, grid="y", left_spine=True, bottom_spine=True):
    """Hairline grid on the axis that carries the values ('y', 'x', 'both' or None), quiet frame, no top/right lines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left_spine)
    ax.spines["bottom"].set_visible(bottom_spine)
    ax.set_axisbelow(True)
    if grid in ("y", "both"):
        ax.grid(axis="y", color=style.GRID, linewidth=0.8, linestyle="-")
    if grid in ("x", "both"):
        ax.grid(axis="x", color=style.GRID, linewidth=0.8, linestyle="-")
    ax.tick_params(length=3, color=style.AXIS)
    return ax


def panel_title(ax, text):
    """A short bold title above one panel, left-aligned with the axes."""
    ax.set_title(clean_text(text), loc="left", fontsize=10.5, fontweight="bold", color=style.INK, pad=8)


def direct_label(ax, x, y, text, dx=0.0, dy=0.0, ha="left", va="center", size=8.5, color=None, weight="normal", clip=False):
    """Text next to a mark. Text is always written in ink (never in the colour of the data) so that it stays legible."""
    return ax.text(x, y, clean_text(text), ha=ha, va=va, fontsize=size, color=color or style.INK_2, fontweight=weight,
                   transform=ax.transData, clip_on=clip) if (dx == 0 and dy == 0) else ax.annotate(
        clean_text(text), (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=size,
        color=color or style.INK_2, fontweight=weight, annotation_clip=clip)


def note(ax, xy, text, xytext, ha="left", va="center", size=8.5, leader=True):
    """An annotation with a thin grey leader line from the text to the point `xy` (both in data coordinates)."""
    return ax.annotate(clean_text(text), xy=xy, xytext=xytext, ha=ha, va=va, fontsize=size, color=style.INK_2,
                       arrowprops=dict(arrowstyle="-", color=style.AXIS, linewidth=0.8, shrinkA=2, shrinkB=3) if leader else None)


def hbar(ax, labels, values, *, color=None, colors=None, fmt="{:,.0f}", label_chars=38, xlim=None, show_values=True,
         bar_height=0.62, value_size=8.5, value_pad_fraction=0.012, grid="x"):
    """
    Horizontal bars, first label on top, value written at the tip of every bar, bars grow from zero.
    `colors` (one per bar) lets a chart highlight some bars and grey the rest; otherwise all bars take `color`.
    """
    values = np.asarray(values, dtype=float)
    y = np.arange(len(values))
    bars = ax.barh(y, values, height=bar_height, color=colors if colors is not None else (color or style.BLUE),
                   edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([short(label, label_chars) if label_chars else clean_text(label) for label in labels])
    ax.invert_yaxis()
    style_axes(ax, grid=grid, bottom_spine=False)
    ax.tick_params(axis="y", length=0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    left, right = ax.get_xlim()
    if show_values:
        pad = (right - left) * value_pad_fraction
        for yi, value in zip(y, values):
            ax.text(value + pad, yi, fmt.format(value), ha="left", va="center", fontsize=value_size, color=style.INK_2)
    return bars


def dot_row(ax, labels, values, *, color=None, colors=None, fmt="{:.1f}", label_chars=38, xlim=None, size=48,
            value_size=8.5):
    """
    A dot plot for numbers that live in a narrow range far from zero (for example popularity 80 to 100). A thin line connects each
    dot to the left edge of the axis, so the eye can follow the row. The axis does NOT start at zero: say so in the subtitle.
    """
    values = np.asarray(values, dtype=float)
    y = np.arange(len(values))
    if xlim is not None:
        ax.set_xlim(*xlim)
    left, right = ax.get_xlim()
    ax.hlines(y, left, values, color=style.GRID, linewidth=1.0, zorder=1)
    ax.scatter(values, y, s=size, color=colors if colors is not None else (color or style.BLUE), zorder=3,
               edgecolor=style.SURFACE, linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([short(label, label_chars) if label_chars else clean_text(label) for label in labels])
    ax.invert_yaxis()
    style_axes(ax, grid=None, bottom_spine=False)
    ax.tick_params(axis="y", length=0)
    pad = (right - left) * 0.015
    for yi, value in zip(y, values):
        ax.text(value + pad, yi, fmt.format(value), ha="left", va="center", fontsize=value_size, color=style.INK_2)
    ax.set_ylim(len(values) - 0.5, -0.5)


def swatch_legend(ax, items, loc="upper right", ncol=1, kind="square", **keywords):
    """
    A legend of (label, colour) or (label, colour, kind) items. kind = 'square' for bars and areas, 'line' for lines, 'dot' for
    scatter plots; an item can carry its own kind, so one legend can mix lines and dots.
    """
    handles = []
    for item in items:
        label, colour = item[0], item[1]
        item_kind = item[2] if len(item) > 2 else kind
        if item_kind == "line":
            handles.append(Line2D([0], [0], color=colour, linewidth=2.2, label=label))
        elif item_kind == "dot":
            handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=6.5, markerfacecolor=colour,
                                  markeredgecolor=colour, label=label))
        else:
            handles.append(Line2D([0], [0], marker="s", linestyle="none", markersize=8, markerfacecolor=colour,
                                  markeredgecolor=colour, label=label))
    keywords.setdefault("borderaxespad", 0.2)
    return ax.legend(handles=handles, loc=loc, ncol=ncol, handletextpad=0.4, columnspacing=1.4, **keywords)


# =============================================================================================== text on coloured cells
def contrast_ratio(color_a, color_b):
    """WCAG contrast ratio of two colours: 1 for identical colours, 21 for black on white. Anything under 4.5 is hard to read."""
    def luminance(color):
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(color)]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    lighter, darker = sorted((luminance(color_a), luminance(color_b)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def readable_ink(background, light="white", dark=None):
    """The text colour (`light` or `dark`, default ink) with the higher contrast on `background`; for numbers inside heat-map cells."""
    dark = dark or style.INK
    return light if contrast_ratio(background, light) >= contrast_ratio(background, dark) else dark


# =============================================================================================== KPI tiles
def kpi_tile(ax, label, value, context, spark=None):
    """
    One stat tile drawn into an empty Axes: label (small, sentence case), value (big), one line of context and, if `spark` is
    given, a sparkline (a list of numbers) at the right. Nothing in a tile is interactive; the number IS the chart.
    """
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.figure.canvas.draw_idle()
    position = ax.get_position()
    width_in = position.width * ax.figure.get_figwidth()
    height_in = position.height * ax.figure.get_figheight()
    radius = 0.045
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle=f"round,pad=0,rounding_size={radius}",
                                mutation_aspect=width_in / height_in, facecolor=style.PAGE, edgecolor=style.GRID,
                                linewidth=1.0, transform=ax.transAxes, clip_on=False))
    ax.text(0.06, 0.86, label, ha="left", va="center", fontsize=10, color=style.INK_2, transform=ax.transAxes)
    ax.text(0.06, 0.58, value, ha="left", va="center", fontsize=27, fontweight="bold", color=style.INK, transform=ax.transAxes)
    wrapped = textwrap.fill(context, width=max(20, int(width_in * 72 / (8.5 * 0.5)) - 4))
    ax.text(0.06, 0.07, wrapped, ha="left", va="bottom", fontsize=8.5, color=style.INK_2, transform=ax.transAxes,
            linespacing=1.25)
    if spark is not None and len(spark) > 1:
        inner = ax.inset_axes([0.56, 0.36, 0.38, 0.3])
        inner.set_axis_off()
        series = np.asarray(spark, dtype=float)
        x = np.arange(len(series))
        inner.plot(x, series, color=style.BLUE, linewidth=1.8)
        inner.plot(x[-1], series[-1], marker="o", markersize=6, color=style.BLUE, markeredgecolor=style.SURFACE,
                   markeredgewidth=1.5, clip_on=False)
        pad = (series.max() - series.min()) * 0.12 or 1.0
        inner.set_ylim(series.min() - pad, series.max() + pad)
        inner.set_xlim(-0.5, len(series) - 0.5)
