"""
tests/test_survival_plots.py  -  checks of the chart helpers of Phase 6B (src/survival/plots.py).

WHAT THIS FILE IS
    The label spacer (no two labels closer than the gap, order kept, as little movement as possible, bounds respected), the isotonic helper
    it uses, and a drawing test that makes a figure with three curves and looks at what was drawn.

WHY WE NEED IT
    A label placed on top of another one is a chart that cannot be read.

EXPECTED OUTCOME
    All tests pass without a screen (the Agg backend).

MOTIVE
    Charts are part of the result, so their mechanics are tested like the numbers.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.survival.plots import _pool_adjacent, draw_survival_curves, percent_axis, spread_apart


def test_pool_adjacent_gives_the_closest_non_decreasing_sequence():
    assert list(_pool_adjacent([1.0, 3.0, 2.0, 4.0])) == [1.0, 2.5, 2.5, 4.0]
    assert list(_pool_adjacent([3.0, 2.0, 1.0])) == [2.0, 2.0, 2.0]
    assert list(_pool_adjacent([1.0, 2.0, 3.0])) == [1.0, 2.0, 3.0]


def test_spread_apart_keeps_the_gap_the_order_and_moves_as_little_as_it_can():
    out = spread_apart([0.10, 0.12, 0.15, 0.50], 0.05)
    assert np.all(np.diff(np.sort(out)) >= 0.05 - 1e-12)
    assert out[3] == 0.5                                                           # a label that is far from the others does not move
    assert out.mean() == pytest.approx(np.mean([0.10, 0.12, 0.15, 0.50]), abs=0.04)
    assert list(spread_apart([0.5, 0.1], 0.05)) == [0.5, 0.1]                      # already apart: unchanged, in the order given
    shuffled = spread_apart([0.15, 0.10, 0.12], 0.05)
    assert shuffled[1] < shuffled[2] < shuffled[0]


def test_spread_apart_respects_the_bounds():
    out = spread_apart([0.0, 0.01, 0.02], 0.05, low=0.0, high=1.0)
    assert out.min() >= 0.0 and np.all(np.diff(out) >= 0.05 - 1e-12)
    top = spread_apart([0.9, 0.95, 1.0], 0.1, low=0.0, high=1.0)
    assert top.max() <= 1.0 + 1e-12 and np.all(np.diff(top) >= 0.1 - 1e-12)
    crowded = spread_apart([0.5, 0.5, 0.5], 0.6, low=0.0, high=1.0)                # cannot fit with that gap: spread over the range
    assert list(crowded) == [0.0, 0.5, 1.0]
    assert len(spread_apart([], 0.1)) == 0


def test_draw_survival_curves_draws_curves_bands_and_labels():
    ages = np.arange(0, 31)
    curves = [{"label": name, "ages": ages, "survival": level ** (ages / 30), "low": np.clip(level ** (ages / 30) - 0.03, 0, 1),
               "high": np.clip(level ** (ages / 30) + 0.03, 0, 1)} for name, level in [("one", 0.20), ("two", 0.21), ("three", 0.22)]]
    fig, ax = plt.subplots()
    draw_survival_curves(ax, curves, x_max=30, label_gap=0.05)
    assert len(ax.lines) == 3 and len(ax.collections) == 3                          # three step lines, three bands
    labels = [t for t in ax.texts if t.get_text() in ("one", "two", "three")]
    assert len(labels) == 3
    y = sorted(t.get_position()[1] for t in labels)
    assert np.all(np.diff(y) >= 0.05 - 1e-9)                                        # nothing printed on top of anything else
    assert ax.get_xlim() == (0, 30) and ax.get_ylim() == (0, 1.0)
    plt.close(fig)


def test_percent_axis_formats_shares():
    assert percent_axis(0.25, 0) == "25%" and percent_axis(1.0, 0) == "100%"
