"""
src/ml/plots.py  -  the chart parts of Phase 6A: ROC and precision-recall curves, calibration, confusion matrices, model ladders, gains, break-even maps.

WHAT THIS FILE IS
    Small functions that draw ONE kind of panel into an axes you give them, in the project's chart style (src/eda/style.py):

        draw_ladder        dots with intervals: a score for every model at every step of the information ladder
        draw_roc           ROC curves (true positive rate against false positive rate), with the chance diagonal
        draw_pr            precision-recall curves, with the base-rate line
        draw_calibration   the reliability diagram: predicted chance against the share that really happened
        draw_confusion     a confusion matrix as a heat map with counts and row shares
        draw_gains         the cumulative gains curve: share of the positives found against the share of songs looked at
        draw_roi_map       a grid of returns on promotion spend, with the break-even line
        draw_actual_vs_predicted   forecast against outcome for a number, with the average outcome in forecast bins
        draw_forest        one row per input: an estimate with its interval, open when the interval includes "no effect"
        draw_partial       the shape a model learned for one input (partial dependence), with the fold-to-fold band

WHY WE NEED IT
    The notebook calls the same kinds of panels for seven different questions. Writing each panel once keeps them identical (the same
    colour always means the same model, the same step of the ladder always has the same blue) and keeps the notebook cells short enough
    to read. The functions do no calculation of their own beyond what is needed to draw.

WHAT HAPPENS WHEN IT RUNS
    Each function draws into the axes it is given and returns nothing; the notebook builds the figure with `make_fig` and saves it.
    Colours: a model keeps one colour everywhere (straight-line model blue, forest green, boosting violet, baseline grey); the steps of
    the information ladder are the ordinal blue ramp, lightest = least information.

EXPECTED OUTCOME
    Panels that look like the charts of Phases 4 and 5: thin marks, hairline grid, direct labels instead of legends where possible.

MOTIVE
    One chart language for the whole project.
"""
from __future__ import annotations

import numpy as np
from matplotlib import patheffects
from matplotlib.patches import Rectangle

from src.eda import style
from src.eda.plots import direct_label, readable_ink, style_axes, swatch_legend

MODEL_COLORS = {"baseline": style.AXIS, "logistic": style.BLUE, "ridge": style.BLUE, "forest": style.GREEN, "boosting": style.VIOLET}
_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]          # lightest to darkest blue (the ordinal ramp of the style)


def step_colors(n: int) -> list:
    """`n` (2 to 5) colours from the ordinal blue ramp, lightest first, spread over the whole ramp."""
    if not 1 <= n <= 5:
        raise ValueError("n must be between 1 and 5")
    positions = np.round(np.linspace(0, 4, n)).astype(int)
    return [_RAMP[i] for i in positions]


def _ink_on(color):
    return readable_ink(color, light="white", dark=style.INK)


# ================================================================================================ the ladder
def draw_ladder(ax, table, steps, models, metric, step_labels, model_labels, *, reference=None, reference_label=None, ylim=None,
                fmt="{:.2f}", label_last=True, ylabel=None, spread=0.2):
    """
    A dot with its interval for every model at every step. `table` has the columns step, model, <metric>, <metric>_low, <metric>_high.
    The reference (the score of knowing nothing) is drawn as one thin line. Models are labelled at the last step.
    """
    models = [m for m in models if m != "baseline"]
    offsets = np.linspace(-spread, spread, len(models)) if len(models) > 1 else np.array([0.0])
    for step_index, step in enumerate(steps):
        for offset, model in zip(offsets, models):
            row = table[(table["step"] == step) & (table["model"] == model)]
            if row.empty:
                continue
            row = row.iloc[0]
            x = step_index + offset
            ax.vlines(x, row[f"{metric}_low"], row[f"{metric}_high"], color=MODEL_COLORS[model], linewidth=2.0, alpha=0.55, zorder=2)
            ax.scatter([x], [row[metric]], s=42, color=MODEL_COLORS[model], edgecolor=style.SURFACE, linewidth=1.2, zorder=3)
            if label_last and step_index == len(steps) - 1:
                direct_label(ax, x + 0.07, row[metric], f"{model_labels[model]} {fmt.format(row[metric])}", size=8.2, color=style.INK_2)
    if reference is not None:
        ax.axhline(reference, color=style.AXIS, linewidth=1.2, zorder=1)
        if reference_label:
            ax.text(-0.45, reference, f" {reference_label}", ha="left", va="bottom", fontsize=8, color=style.MUTED)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([step_labels[s] for s in steps], fontsize=8.6)
    ax.set_xlim(-0.5, len(steps) - 1 + (1.15 if label_last else 0.5))
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel)
    style_axes(ax, grid="y")


def ladder_legend(ax, models, model_labels, reference_label=None, loc="lower right", fontsize=8.4):
    """The legend of a ladder panel: one dot per model in its colour and, if given, one grey line for the reference (what knowing nothing scores)."""
    items = [(model_labels[m], MODEL_COLORS[m], "dot") for m in models if m != "baseline"]
    if reference_label:
        items.append((reference_label, style.AXIS, "line"))
    return swatch_legend(ax, items, loc=loc, fontsize=fontsize)


# ================================================================================================ curves
def draw_roc(ax, curves, *, legend_loc="lower right"):
    """
    `curves` = list of dicts with label, fpr, tpr, color (and optional width). Draws the chance diagonal and the curves, and a legend
    that carries the area under each curve.
    """
    ax.plot([0, 1], [0, 1], color=style.AXIS, linewidth=1.2, zorder=1)
    items = []
    for curve in curves:
        ax.plot(curve["fpr"], curve["tpr"], color=curve["color"], linewidth=curve.get("width", 2.0), zorder=3, drawstyle="steps-post")
        items.append((curve["label"], curve["color"], "line"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("false positive rate (songs wrongly flagged)")
    ax.set_ylabel("true positive rate (positives found)")
    style_axes(ax, grid="both")
    swatch_legend(ax, items, loc=legend_loc, fontsize=8.4)


def draw_pr(ax, curves, base_rate, *, legend_loc="upper right"):
    """`curves` = list of dicts with label, recall, precision, color. The base rate (precision of picking songs at random) is a thin line."""
    ax.axhline(base_rate, color=style.AXIS, linewidth=1.2, zorder=1)
    ax.text(0.03, base_rate - 0.012, f"base rate {base_rate:.0%}", ha="left", va="top", fontsize=8, color=style.MUTED)
    items = []
    for curve in curves:
        ax.plot(curve["recall"], curve["precision"], color=curve["color"], linewidth=2.0, zorder=3, drawstyle="steps-post")
        items.append((curve["label"], curve["color"], "line"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall (positives found)")
    ax.set_ylabel("precision (flagged songs that are positive)")
    style_axes(ax, grid="both")
    swatch_legend(ax, items, loc=legend_loc, fontsize=8.4)


def draw_calibration(ax, table, color=None, limit=None):
    """The reliability diagram from `metrics.calibration_table`: mean forecast on x, observed share (with its interval) on y."""
    color = color or style.BLUE
    top = limit or float(max(table["ci_high"].max(), table["mean_predicted"].max())) * 1.08
    ax.plot([0, top], [0, top], color=style.AXIS, linewidth=1.2, zorder=1)
    ax.vlines(table["mean_predicted"], table["ci_low"], table["ci_high"], color=color, linewidth=2.0, alpha=0.55, zorder=2)
    ax.scatter(table["mean_predicted"], table["observed"], s=46, color=color, edgecolor=style.SURFACE, linewidth=1.2, zorder=3)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("average forecast chance")
    ax.set_ylabel("share that really happened")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_axes(ax, grid="both")


def draw_confusion(ax, matrix, row_labels, col_labels, *, normalize=True, count_fmt="{:,.0f}"):
    """
    A confusion matrix: true class in the rows, forecast class in the columns. The colour is the share of the row (so that a rare class is
    as visible as a common one); each cell also shows the count. Correct forecasts are on the diagonal.
    """
    m = np.asarray(matrix, dtype=float)
    shares = m / np.where(m.sum(axis=1, keepdims=True) > 0, m.sum(axis=1, keepdims=True), 1.0)
    image = shares if normalize else m
    ax.imshow(image, cmap=style.SEQUENTIAL_BLUE, vmin=0, vmax=1 if normalize else m.max(), aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            rgba = ax.images[0].cmap(ax.images[0].norm(image[i, j]))
            ink = _ink_on(rgba[:3])
            ax.text(j, i - 0.12, count_fmt.format(m[i, j]), ha="center", va="center", fontsize=11, fontweight="bold", color=ink)
            ax.text(j, i + 0.2, f"{shares[i, j]:.0%} of row", ha="center", va="center", fontsize=8.2, color=ink)
    for i in range(min(m.shape)):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=style.INK, linewidth=1.8, zorder=5))     # the correct forecasts
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(col_labels, fontsize=8.6)
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=8.6)
    ax.set_xlabel("forecast")
    ax.set_ylabel("what happened")
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


# ================================================================================================ decisions
def draw_gains(ax, curves, base_rate, *, legend_loc="lower right", mark_share=None):
    """
    `curves` = list of dicts with label, share (x), gain (y), color. Draws the random line, the best possible curve (every positive
    first) and the model curves. Reading: look at the top 10% of songs by forecast; the curve says which share of all positives they hold.
    `mark_share` (for example 0.10) draws one thin vertical guide at that share of songs.
    """
    ax.plot([0, 1], [0, 1], color=style.AXIS, linewidth=1.2, zorder=1)
    if mark_share is not None:
        ax.axvline(mark_share, color=style.AXIS, linewidth=0.9, zorder=1)
        ax.text(mark_share + 0.012, 0.045, f"top {mark_share:.0%}", ha="left", va="bottom", fontsize=8, color=style.MUTED, rotation=90)
    ax.plot([0, base_rate, 1], [0, 1, 1], color=style.GRID, linewidth=1.6, zorder=1)
    ax.text(base_rate + 0.015, 0.985, "best possible", ha="left", va="top", fontsize=8, color=style.MUTED)
    items = []
    for curve in curves:
        ax.plot(curve["share"], curve["gain"], color=curve["color"], linewidth=2.0, zorder=3)
        items.append((curve["label"], curve["color"], "line"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("share of songs looked at (highest forecast first)")
    ax.set_ylabel("share of all positives found")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_axes(ax, grid="both")
    swatch_legend(ax, items, loc=legend_loc, fontsize=8.4)


def draw_roi_map(ax, shares, ratios, roi, *, vmax=None, annotate=False):
    """
    `roi` has one row per share promoted and one column per value ratio. Colour: blue = the money comes back, red = it does not, grey = break-even.
    The thick contour is where the return is exactly zero. With `annotate=True` every cell also shows its return (+0.5 = 50 cents back for every
    unit spent on top of the unit itself; -0.4 = 40 cents of every unit lost).
    """
    roi = np.asarray(roi, dtype=float)
    vmax = vmax or float(np.nanmax(np.abs(roi))) or 1.0                           # an all-zero grid still needs a colour range
    x = np.arange(len(ratios))
    y = np.arange(len(shares))
    mesh = ax.pcolormesh(np.arange(len(ratios) + 1) - 0.5, np.arange(len(shares) + 1) - 0.5, roi, cmap=style.DIVERGING, vmin=-vmax, vmax=vmax, shading="flat")
    if np.nanmin(roi) < 0 < np.nanmax(roi):                                       # the break-even line exists only where the return changes sign
        ax.contour(x, y, roi, levels=[0.0], colors=[style.INK], linewidths=1.6)
    if annotate:
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                cell = mesh.cmap(mesh.norm(roi[i, j]))[:3]
                label = ax.text(j, i, f"{roi[i, j]:+.1f}".replace("-", "\u2212"), ha="center", va="center", fontsize=8.2, color=_ink_on(cell), zorder=5)
                label.set_path_effects([patheffects.withStroke(linewidth=3.2, foreground=cell)])     # the number cuts a small gap in the break-even line instead of being struck through
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:g}" for r in ratios], fontsize=8.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s:.0%}" for s in shares], fontsize=8.4)
    ax.set_xlabel("what a hit is worth, in promotion costs")
    ax.set_ylabel("share of songs promoted")
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    return mesh


# ================================================================================================ explanations
def draw_forest(ax, y, estimate, low, high, *, colors, reference=1.0, log=False, size=46, texts=None, text_x=1.03, text_size=8.4):
    """
    One interval with a dot for every row of a "forest plot": `y` are the row positions, `estimate`, `low` and `high` the numbers.
    `colors` is one colour or a list with one colour per row. A row whose interval includes `reference` (1 for an odds ratio, 0 for a
    difference) is drawn as an OPEN dot: the data do not rule out "no association". `texts` (one string per row) are written to the right
    of the panel, outside the axes, in a column that starts at `text_x` (a fraction of the panel width). Rows with a missing number are skipped.
    """
    y = np.asarray(y, dtype=float)
    estimate, low, high = (np.asarray(v, dtype=float) for v in (estimate, low, high))
    colors = [colors] * len(y) if isinstance(colors, str) else list(colors)
    if log:
        ax.set_xscale("log")
    ax.axvline(reference, color=style.AXIS, linewidth=1.2, zorder=1)
    for i in range(len(y)):
        if not np.all(np.isfinite([estimate[i], low[i], high[i]])):
            continue
        ax.hlines(y[i], low[i], high[i], color=colors[i], linewidth=2.0, alpha=0.6, zorder=2)
        open_dot = low[i] <= reference <= high[i]
        ax.scatter([estimate[i]], [y[i]], s=size, facecolor=style.SURFACE if open_dot else colors[i], edgecolor=colors[i],
                   linewidth=1.6 if open_dot else 1.2, zorder=3)
        if texts is not None:
            ax.text(text_x, y[i], texts[i], transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=text_size, color=style.INK_2)
    style_axes(ax, grid="x", left_spine=False)
    ax.tick_params(axis="y", length=0)


def draw_partial(ax, grid, curves, deciles, *, color, xlabel, ylabel, as_percent=False, x_scale=None, ylim=None):
    """
    Partial dependence of one input. `curves` has one row per fitted model (one per fold) and one column per grid value. Line: the mean
    over the models. Band: the lowest and highest of the models, so that a wiggle which only one fold sees does not look like a finding.
    The little ticks at the bottom mark the deciles of the input among the songs (where the data really are). `x_scale` may be "log", or
    "symlog" for an input that has zeros and a long tail (chart days of an artist: a straight line up to 10, a log scale above).
    """
    grid = np.asarray(grid, dtype=float)
    curves = np.atleast_2d(np.asarray(curves, dtype=float))
    ax.fill_between(grid, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.22, linewidth=0, zorder=2)
    ax.plot(grid, curves.mean(axis=0), color=color, linewidth=2.2, zorder=3)
    if x_scale == "symlog":
        ax.set_xscale("symlog", linthresh=10)
        ticks = [t for t in (0, 10, 100, 1000, 10000) if t <= grid.max() * 1.5]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,}" for t in ticks])                                  # plain numbers (1,000), not 10 to the power 3
        ax.minorticks_off()
    elif x_scale:
        ax.set_xscale(x_scale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.vlines(np.asarray(deciles, dtype=float), 0, 0.045, transform=ax.get_xaxis_transform(), color=style.MUTED, linewidth=1.0, zorder=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if as_percent:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_axes(ax, grid="y")


# ================================================================================================ numbers
def draw_actual_vs_predicted(ax, actual, predicted, *, n_bins: int = 10, color=None, unit=""):
    """Forecast on x, outcome on y: every song as a faint dot, the average outcome in `n_bins` forecast groups as a line, and the 45-degree line."""
    color = color or style.BLUE
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    ax.plot([low, high], [low, high], color=style.AXIS, linewidth=1.2, zorder=1)
    ax.scatter(predicted, actual, s=14, color=color, alpha=0.28, edgecolor="none", zorder=2)
    order = np.argsort(predicted, kind="stable")
    parts = np.array_split(order, n_bins)
    ax.plot([predicted[p].mean() for p in parts], [actual[p].mean() for p in parts], color=style.INK, linewidth=2.0, marker="o", markersize=4.5, zorder=4)
    pad = (high - low) * 0.04
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel(f"forecast{(' (' + unit + ')') if unit else ''}")
    ax.set_ylabel(f"what happened{(' (' + unit + ')') if unit else ''}")
    style_axes(ax, grid="both")
