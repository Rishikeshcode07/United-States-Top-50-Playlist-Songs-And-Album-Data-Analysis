"""
src/survival/plots.py  -  the chart parts of Phase 6B: survival curves with bands and direct labels, and a label spacer.

WHAT THIS FILE IS
    Small functions that draw into an axes you give them, in the project's chart style (src/eda/style.py):

        draw_survival_curves   step curves of "share still on the chart" against chart days, with optional bands and a label at the end of every curve
        spread_apart           move labels apart so that no two are closer than a minimum gap, with the least total movement
        percent_axis           the axis formatter that writes 0.25 as 25%

WHY WE NEED IT
    Several charts of the notebook show curves that end close together (three groups that all fall to a few percent after 90 days). Labels
    written at the end of the lines would then be printed on top of each other. `spread_apart` finds the positions that are closest to
    where the labels want to be while keeping a minimum gap, so that every curve can be named directly, without a legend.

WHAT HAPPENS WHEN IT RUNS
    `draw_survival_curves` draws each curve as a step function (the survival value holds until the next chart day), the band as a
    lighter step area behind it, and the label next to the last point of the curve. Nothing is calculated except the label positions.

EXPECTED OUTCOME
    Curves that start at 100% on the left, fall quickly and then flatten, each named at its right end.

MOTIVE
    One chart language for the whole project: direct labels, thin marks, a hairline grid.
"""
from __future__ import annotations

import numpy as np
from matplotlib.ticker import FuncFormatter

from src.eda import style
from src.eda.plots import direct_label

percent_axis = FuncFormatter(lambda value, _: f"{value:.0%}")


def _pool_adjacent(values, weights=None):
    """Isotonic regression (non-decreasing) by pooling adjacent violators: the closest non-decreasing sequence to `values`."""
    blocks = []                                    # each block: [sum of weighted values, sum of weights, count]
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    for value, weight in zip(values, weights):
        blocks.append([value * weight, weight, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            top = blocks.pop()
            blocks[-1][0] += top[0]
            blocks[-1][1] += top[1]
            blocks[-1][2] += top[2]
    return np.concatenate([np.full(count, total / weight) for total, weight, count in blocks])


def spread_apart(positions, min_gap: float, low: float | None = None, high: float | None = None) -> np.ndarray:
    """
    Positions for labels, in the order they were given, such that any two are at least `min_gap` apart and the total squared movement
    from the wanted positions is as small as possible. If `low` and `high` are given, the result is shifted to stay inside them when it
    fits (when the labels cannot fit at all, they are spread evenly over the range).
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    if n == 0:
        return positions.copy()
    order = np.argsort(positions, kind="mergesort")
    wanted = positions[order]
    shifted = _pool_adjacent(wanted - np.arange(n) * min_gap) + np.arange(n) * min_gap        # y_k >= y_(k-1) + gap
    if low is not None and high is not None:
        if (n - 1) * min_gap >= high - low:
            shifted = np.linspace(low, high, n)
        else:
            if shifted[0] < low:
                shifted = shifted + (low - shifted[0])
            if shifted[-1] > high:
                shifted = shifted - (shifted[-1] - high)
                if shifted[0] < low:                                    # cannot happen when the labels fit, kept as a guard
                    shifted = shifted + (low - shifted[0])
    out = np.empty(n)
    out[order] = shifted
    return out


def draw_survival_curves(ax, curves, *, x_max: float, label_gap: float = 0.05, label_size: float = 8.8, x_label_pad: float = 1.2, band_alpha: float = 0.16):
    """
    `curves` is a list of dicts with `label`, `ages` (0, 1, 2 ...), `survival` and optionally `low`, `high` (a band), `color`, `width` and
    `linestyle`. Draws step curves up to `x_max` chart days and writes each label just right of the end of its curve (the labels are moved
    apart by `label_gap`, in units of the vertical axis, so that they do not touch). Pass label=None for a curve that needs no label.
    """
    ends = []
    for curve in curves:
        ages = np.asarray(curve["ages"], dtype=float)
        keep = ages <= x_max
        x = ages[keep]
        color = curve.get("color", style.BLUE)
        if curve.get("low") is not None:
            ax.fill_between(x, np.asarray(curve["low"])[keep], np.asarray(curve["high"])[keep], step="post", color=color, alpha=band_alpha, linewidth=0, zorder=1)
        ax.step(x, np.asarray(curve["survival"])[keep], where="post", color=color, linewidth=curve.get("width", 2.2),
                linestyle=curve.get("linestyle", "-"), zorder=3)
        ends.append((curve.get("label"), float(np.asarray(curve["survival"])[keep][-1]), color))
    labelled = [(i, e) for i, e in enumerate(ends) if e[0] is not None]
    if labelled:
        y = spread_apart([e[1] for _, e in labelled], label_gap, low=0.0, high=1.0)
        for (_, (label, _, _)), position in zip(labelled, y):
            direct_label(ax, x_max + x_label_pad, position, label, size=label_size, clip=False)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(percent_axis)
