"""
tests/test_concentration.py  -  tests for the concentration measures (HHI, effective number, Gini, Lorenz curve, top-N share).

WHAT THIS FILE IS
    Small hand-calculated examples (four equal holders, one holder with everything, the numbers 1-2-3-4 ...) and one check
    on the real chart: the artist concentration that the EDA notebook quotes.

WHY WE NEED IT
    "Taylor Swift holds 7.5 % of the chart points" and "the Gini of the artists is 0.78" end up in the research paper and on the
    dashboards. If a formula were off by a small constant nobody would notice by eye, so every formula is tested against an
    answer that was worked out with pencil and paper first.

WHAT HAPPENS WHEN IT RUNS
    Tiny lists of numbers go in, the measures come out, and each one is compared with the value worked out by hand. The Gini is
    also checked against a second definition (one minus twice the area under the Lorenz curve). The last test runs the real file
    through the pipeline and is skipped when the real CSV is not in data/raw/.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    One definition of "concentration" for the notebook, the Streamlit app, the research paper and Power BI.
"""
import numpy as np
import pandas as pd
import pytest

from src.eda.concentration import (concentration_summary, cumulative_share, effective_number, gini, hhi, lorenz_curve,
                                   period_concentration, period_shares, shares, top_n_share)


# ================================================================================ hand-calculated examples
def test_four_equal_holders():
    assert hhi([1, 1, 1, 1]) == pytest.approx(2500.0)
    assert effective_number([1, 1, 1, 1]) == pytest.approx(4.0)
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-12)


def test_one_holder_owns_everything():
    assert hhi([10]) == pytest.approx(10_000.0)
    assert effective_number([10]) == pytest.approx(1.0)
    assert gini([0, 0, 0, 10]) == pytest.approx(0.75)                  # the largest Gini that four holders can reach: (n - 1) / n


def test_the_numbers_one_to_four():
    # sorted 1,2,3,4: sum 10, sum of rank x value = 1 + 4 + 9 + 16 = 30  ->  G = 2 x 30 / (4 x 10) - 5 / 4 = 0.25
    assert gini([4, 2, 1, 3]) == pytest.approx(0.25)
    # shares 10 %, 20 %, 30 %, 40 %  ->  HHI = 100 + 400 + 900 + 1600 = 3000
    assert hhi([1, 2, 3, 4]) == pytest.approx(3000.0)
    assert effective_number([1, 2, 3, 4]) == pytest.approx(10_000 / 3000)


def test_shares_add_up_to_one_and_follow_the_order_of_the_input():
    result = shares([2, 6, 2])
    assert result.sum() == pytest.approx(1.0)
    assert result.tolist() == pytest.approx([0.2, 0.6, 0.2])


def test_the_measures_do_not_depend_on_the_order_or_on_the_unit():
    values = [5, 1, 9, 3, 3, 12]
    for transformed in ([12, 9, 5, 3, 3, 1], [50, 10, 90, 30, 30, 120], np.array(values) / 7):
        assert hhi(transformed) == pytest.approx(hhi(values))
        assert gini(transformed) == pytest.approx(gini(values))
        assert effective_number(transformed) == pytest.approx(effective_number(values))


def test_a_holder_with_nothing_changes_the_gini_but_not_the_hhi():
    assert hhi([1, 1, 0, 0]) == pytest.approx(hhi([1, 1]))
    assert gini([0, 1]) == pytest.approx(0.5)                          # (n - 1) / n with n = 2
    assert gini([1, 1, 0, 0]) > gini([1, 1])


def test_input_can_be_a_list_an_array_or_a_series():
    values = [3, 1, 4, 1, 5]
    assert hhi(pd.Series(values)) == pytest.approx(hhi(values))
    assert hhi(np.array(values)) == pytest.approx(hhi(values))


# ================================================================================ the Lorenz curve
def test_the_lorenz_curve_starts_in_the_corner_and_ends_in_the_corner():
    x, y = lorenz_curve([1, 2, 3, 4])
    assert x.tolist() == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
    assert y.tolist() == pytest.approx([0, 0.1, 0.3, 0.6, 1.0])        # the poorest holder first: 1, 1+2, 1+2+3 ... out of 10


def test_the_lorenz_curve_never_goes_above_the_line_of_equality_and_never_goes_down():
    x, y = lorenz_curve([7, 1, 1, 30, 2, 2, 0, 5])
    assert (np.diff(y) >= 0).all()
    assert (y <= x + 1e-12).all()


@pytest.mark.parametrize("values", [[1, 2, 3, 4], [5, 5, 5], [0, 0, 0, 10], [3, 1, 4, 1, 5, 9, 2, 6], list(range(1, 50))])
def test_the_gini_equals_one_minus_twice_the_area_under_the_lorenz_curve(values):
    x, y = lorenz_curve(values)
    area = float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))            # trapezoid rule; exact for the polygon of a Lorenz curve
    assert gini(values) == pytest.approx(1 - 2 * area)


# ================================================================================ top-N and cumulative share
def test_top_n_share():
    values = [5, 3, 2]
    assert top_n_share(values, 1) == pytest.approx(0.5)
    assert top_n_share(values, 2) == pytest.approx(0.8)
    assert top_n_share(values, 3) == pytest.approx(1.0)
    assert top_n_share(values, 10) == pytest.approx(1.0)               # more holders asked for than there are


def test_top_n_share_needs_a_positive_n():
    with pytest.raises(ValueError):
        top_n_share([1, 2, 3], 0)


def test_cumulative_share_is_the_pareto_curve():
    curve = cumulative_share([2, 5, 3])
    assert curve.tolist() == pytest.approx([0.5, 0.8, 1.0])
    assert curve[-1] == pytest.approx(1.0)


def test_the_summary_collects_everything_and_counts_empty_holders_separately():
    summary = concentration_summary([5, 3, 2, 0, 0], top_ns=(1, 2))
    assert summary["holders"] == 5 and summary["holders_with_a_share"] == 3
    assert summary["hhi"] == pytest.approx(3800.0)                      # shares 50 %, 30 %, 20 %: 2,500 + 900 + 400
    assert summary["effective_number"] == pytest.approx(10_000 / 3800)
    assert summary["top_1_share"] == pytest.approx(0.5) and summary["top_2_share"] == pytest.approx(0.8)
    assert set(summary) == {"holders", "holders_with_a_share", "hhi", "effective_number", "gini", "top_1_share", "top_2_share"}


# ================================================================================ bad input is refused, not guessed
@pytest.mark.parametrize("bad", [[], [-1, 2], [1, np.nan], [0, 0, 0], [[1, 2], [3, 4]]])
def test_values_that_have_no_share_are_refused(bad):
    for function in (shares, hhi, effective_number, gini, lorenz_curve, cumulative_share):
        with pytest.raises(ValueError):
            function(bad)


# ================================================================================ shares per period
@pytest.fixture()
def points():
    """Two months. May: A 60 (in two rows), B 40. June: A 10, B 10, C 10 and D with nothing."""
    return pd.DataFrame({
        "month": ["2025-05", "2025-05", "2025-05", "2025-06", "2025-06", "2025-06", "2025-06"],
        "artist": ["A", "A", "B", "A", "B", "C", "D"],
        "points": [25, 35, 40, 10, 10, 10, 0],
    })


def test_period_shares_sum_the_rows_of_a_holder_and_skip_holders_with_nothing(points):
    table = period_shares(points, "month", "artist", "points")
    assert len(table) == 5                                              # A, B in May; A, B, C in June; D has no row
    may = table[table["month"] == "2025-05"].set_index("artist")
    assert may["value"].to_dict() == {"A": 60, "B": 40}
    assert may["share_pct"].to_dict() == pytest.approx({"A": 60.0, "B": 40.0})
    june = table[table["month"] == "2025-06"]
    assert june["share_pct"].sum() == pytest.approx(100.0)
    assert june["share_pct"].tolist() == pytest.approx([100 / 3] * 3)


def test_period_concentration_follows_the_hand_calculation(points):
    result = period_concentration(period_shares(points, "month", "artist", "points"), "month", "artist").set_index("month")
    assert result.loc["2025-05", "holders"] == 2
    assert result.loc["2025-05", "hhi"] == pytest.approx(60 ** 2 + 40 ** 2)                 # 5,200
    assert result.loc["2025-05", "effective_number"] == pytest.approx(10_000 / 5200)
    assert result.loc["2025-05", "top1_pct"] == pytest.approx(60.0)
    assert result.loc["2025-06", "holders"] == 3
    assert result.loc["2025-06", "hhi"] == pytest.approx(3 * (100 / 3) ** 2)                 # 3,333.3
    assert result.loc["2025-06", "effective_number"] == pytest.approx(3.0)


def test_a_period_with_one_holder_is_a_monopoly():
    table = period_shares(pd.DataFrame({"m": ["x", "x"], "a": ["A", "A"], "v": [1, 2]}), "m", "a", "v")
    result = period_concentration(table, "m", "a")
    assert result.loc[0, "hhi"] == pytest.approx(10_000.0) and result.loc[0, "effective_number"] == pytest.approx(1.0)


def test_unused_categories_do_not_create_empty_periods():
    frame = pd.DataFrame({"month": pd.Categorical(["a", "a"], categories=["a", "b"]), "artist": ["X", "Y"], "points": [1, 3]})
    table = period_shares(frame, "month", "artist", "points")
    assert set(table["month"].astype(str)) == {"a"}
    assert period_concentration(table, "month", "artist")["month"].astype(str).tolist() == ["a"]


# ================================================================================ the real chart
def test_artist_concentration_of_the_real_chart(real_result):
    """The numbers that the EDA notebook quotes for the artists (first-named artist of each song, chart points of every day)."""
    playlist = real_result.tables["playlist_daily"]
    points = playlist.groupby("primary_artist")["chart_points"].sum()
    summary = concentration_summary(points)
    assert summary["holders"] == 243
    assert summary["hhi"] == pytest.approx(254.4, abs=0.05)
    assert summary["effective_number"] == pytest.approx(39.3, abs=0.05)
    assert summary["gini"] == pytest.approx(0.782, abs=0.0005)
    assert summary["top_1_share"] == pytest.approx(0.0753, abs=0.00005)
    assert summary["top_5_share"] == pytest.approx(0.270, abs=0.0005)
    assert summary["top_10_share"] == pytest.approx(0.411, abs=0.0005)
    assert summary["top_20_share"] == pytest.approx(0.583, abs=0.0005)
    assert points.idxmax() == "Taylor Swift"
    # the two ways of writing the same fact have to agree
    assert summary["effective_number"] == pytest.approx(10_000 / summary["hhi"])
    assert points.sum() == playlist["chart_points"].sum()
