"""
tests/test_dash_metrics.py  -  the KPIs of the brief, recalculated for any selection of rows.

WHAT THIS FILE IS
    Tests of src/dash/metrics.py. Three kinds of check:
      1. On the whole file the song table must reproduce `dim_track` and the artist table must reproduce `dim_artist`
         (two independent routes to the same KPI, one written in the pipeline, one in the dashboard).
      2. On a small synthetic chart every value is recalculated with plain Python loops.
      3. Properties: the ADI of all primary artists adds up to 100, a filter must not change the denominator, an empty selection
         must give empty tables with the right columns.

WHY WE NEED IT
    The dashboard calculates the KPIs again for every filter combination. If the formula drifted from the one in the pipeline,
    the Power BI report, the SQL views and the dashboard would show three different "Average Rank" numbers.

WHAT HAPPENS WHEN IT RUNS
    The tests on the real tables are skipped (not failed) when data/processed/ has not been built.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    One formula for each KPI, checked against its first implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import LONGEVITY_LABELS
from src.dash import filters as F
from src.dash import metrics as M
from src.dash.load import chart_bounds
from src.eda import concentration as conc
from tests.helpers import same_values


@pytest.fixture(scope="module")
def play(dash_synthetic):
    return dash_synthetic.playlist


# ------------------------------------------------------------------------------ the song table against its first version
def test_song_table_reproduces_dim_track_on_the_real_file(dash_real):
    songs = M.song_table(dash_real.playlist).set_index("track_id")
    first = dash_real.dim_track.set_index("track_id").reindex(songs.index)
    exact = ["days_on_chart", "best_rank", "worst_rank", "chart_points", "days_in_top10", "days_in_top20", "days_at_number_one",
             "peak_popularity", "entry_rank", "exit_rank", "median_rank"]
    rounded = ["avg_rank", "rank_volatility_index", "avg_daily_rank_move", "avg_popularity"]       # dim_track rounds to 2 decimals
    for column in exact:
        assert same_values(songs[column], first[column]), column                  # two empty cells count as equal
    for column in rounded:
        assert np.allclose(songs[column].fillna(-1), first[column].fillna(-1), atol=0.0051), column
    assert (songs["longevity_class"] == first["longevity_class"]).all()
    assert len(songs) == 944


def test_song_table_reproduces_dim_track_on_the_synthetic_chart(dash_synthetic):
    songs = M.song_table(dash_synthetic.playlist).set_index("track_id")
    first = dash_synthetic.dim_track.set_index("track_id").reindex(songs.index)
    for column in ["days_on_chart", "best_rank", "worst_rank", "chart_points", "days_in_top10", "entry_rank", "exit_rank"]:
        assert (songs[column] == first[column]).all(), column
    assert np.allclose(songs["avg_rank"], first["avg_rank"], atol=0.0051)
    assert (songs["rank_volatility_index"].isna() == first["rank_volatility_index"].isna()).all()


def test_song_table_on_a_subset_matches_plain_python(play):
    days = sorted(play["date"].unique())
    rows = play[(play["date"] >= days[2]) & (play["date"] <= days[7]) & (play["position"] <= 30)]
    songs = M.song_table(rows).set_index("track_id")
    for track_id in list(songs.index)[:12]:
        mine = rows[rows["track_id"] == track_id].sort_values("date")
        places = mine["position"].tolist()
        row = songs.loc[track_id]
        assert row["days_on_chart"] == len(places)
        assert row["best_rank"] == min(places) and row["worst_rank"] == max(places)
        assert row["avg_rank"] == pytest.approx(sum(places) / len(places))
        assert row["entry_rank"] == places[0] and row["exit_rank"] == places[-1]
        assert row["chart_points"] == sum(51 - p for p in places)
        if len(places) > 1:
            assert row["rank_volatility_index"] == pytest.approx(float(np.std(places, ddof=1)))
        else:
            assert np.isnan(row["rank_volatility_index"])
        assert row["days_in_top10"] == sum(p <= 10 for p in places)
        assert row["first_date"] == mine["date"].min() and row["last_date"] == mine["date"].max()


def test_a_selection_changes_the_kpis_it_should(play):
    top = play[play["position"] <= 10]
    whole = M.song_table(play).set_index("track_id")
    part = M.song_table(top).set_index("track_id")
    common = part.index
    assert (part["days_on_chart"] <= whole.loc[common, "days_on_chart"]).all()
    assert (part["avg_rank"] <= 10).all() and (part["worst_rank"] <= 10).all()


def test_song_table_of_nothing_has_the_same_columns(play):
    empty = M.song_table(play.iloc[:0])
    assert empty.empty and list(empty.columns) == M.SONG_COLUMNS


def test_longevity_class_uses_the_project_bins():
    got = M.longevity_class([1, 3, 4, 14, 15, 60, 61, 180, 181, 536]).tolist()
    assert got == [LONGEVITY_LABELS[i] for i in (0, 0, 1, 1, 2, 2, 3, 3, 4, 4)]


def test_grouped_slope_matches_numpy_polyfit():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"k": np.repeat(["a", "b", "c"], [10, 2, 5]), "x": np.concatenate([np.arange(10), np.arange(2), np.arange(5)])})
    frame["y"] = frame["x"] * 2.5 + rng.normal(size=len(frame))
    slopes = M.grouped_slope(frame, "k", "x", "y")
    for key in ("a", "c"):
        part = frame[frame["k"] == key]
        assert slopes[key] == pytest.approx(np.polyfit(part["x"], part["y"], 1)[0])
    assert np.isnan(slopes["b"])                      # two points are not enough for a trend


# ------------------------------------------------------------------------------------------------------------ artists
def test_artist_table_reproduces_dim_artist_on_the_real_file(dash_real):
    play = dash_real.playlist
    total, days = M.total_chart_points(play), play["date"].nunique()
    artists = M.artist_table(play, total, days).set_index("artist")
    first = dash_real.dim_artist.set_index("artist_name")
    assert artists["adi"].sum() == pytest.approx(100.0)
    assert np.allclose(artists["adi"], first["dominance_index_primary"].reindex(artists.index), atol=0.0001)
    assert np.allclose(artists["share_of_slots_pct"], first["share_of_slots_primary_pct"].reindex(artists.index), atol=0.0001)
    credits = M.artist_table(play, total, days, "credits", dash_real.bridge).set_index("artist")
    assert np.allclose(credits["adi"], first["dominance_index_all_credits"].reindex(credits.index), atol=0.0001)
    assert credits["adi"].sum() > 100.0


def test_artist_points_are_the_sum_of_51_minus_place(play):
    total, days = M.total_chart_points(play), play["date"].nunique()
    artists = M.artist_table(play, total, days)
    for _, row in artists.head(5).iterrows():
        places = play.loc[play["primary_artist"] == row["artist"], "position"]
        assert row["chart_points"] == (51 - places).sum()
        assert row["chart_slots"] == len(places)
        assert row["adi"] == pytest.approx((51 - places).sum() / total * 100)
    assert artists["adi_rank"].tolist() == list(range(1, len(artists) + 1))
    assert artists["adi"].is_monotonic_decreasing


def test_a_filter_does_not_change_the_denominator_of_the_adi(play):
    total, days = M.total_chart_points(play), play["date"].nunique()
    explicit = play[play["is_explicit"]]
    part = M.artist_table(explicit, total, days)
    assert part["adi"].sum() == pytest.approx(explicit["chart_points"].sum() / total * 100)
    assert part["adi"].sum() < 100


def test_the_credits_view_needs_the_bridge(play):
    with pytest.raises(ValueError):
        M.artist_table(play, 1.0, 1, "credits")
    with pytest.raises(ValueError):
        M.artist_table(play, 1.0, 1, "sideways")


def test_artist_table_of_nothing_has_the_same_columns(play):
    empty = M.artist_table(play.iloc[:0], 1.0, 1)
    assert empty.empty and list(empty.columns) == M.ARTIST_COLUMNS


def test_period_adi_adds_up_to_100_for_all_artists(play):
    table = M.artist_period_adi(play, play, "day")
    sums = table.groupby("period")["adi"].sum()
    assert np.allclose(sums, 100.0)
    assert set(table["period"]) == set(play["date"].dt.normalize())


def test_period_adi_for_chosen_artists_is_filled_with_zeros(play):
    names = play["primary_artist"].value_counts().index[:2].tolist()
    table = M.artist_period_adi(play, play, "day", artists=names)
    days = play["date"].nunique()
    assert len(table) == days * 2
    assert (table["adi"] >= 0).all()
    assert set(table["artist"]) == set(names)


def test_period_adi_by_month_uses_the_points_of_the_month(play):
    window = play.assign(date=play["date"] + pd.Timedelta(days=0))
    table = M.artist_period_adi(play, window, "month", artists=[play["primary_artist"].iloc[0]])
    artist = play["primary_artist"].iloc[0]
    for _, row in table.iterrows():
        month = play[play["date"].dt.to_period("M").dt.to_timestamp() == row["period"]]
        assert row["points"] == month.loc[month["primary_artist"] == artist, "chart_points"].sum()
        assert row["total_points"] == month["chart_points"].sum()


def test_concentration_by_period_matches_the_concentration_module(play):
    table = M.concentration_by_period(play, "day")
    first_day = play["date"].min()
    values = play[play["date"] == first_day].groupby("primary_artist")["chart_points"].sum().to_numpy(float)
    row = table[table["period"] == first_day].iloc[0]
    assert row["hhi"] == pytest.approx(conc.hhi(values))
    assert row["effective_number"] == pytest.approx(conc.effective_number(values))
    assert row["top5_pct"] == pytest.approx(conc.top_n_share(values, 5) * 100)
    assert row["gini"] == pytest.approx(conc.gini(values))
    assert row["artists"] == len(values)


def test_concentration_of_one_artist_is_10000(play):
    one = play[play["primary_artist"] == play["primary_artist"].iloc[0]]
    table = M.concentration_by_period(one, "day")
    assert (table["hhi"] == 10_000).all() and (table["top1_pct"] == 100).all()


def test_add_period_gives_the_first_day_of_the_week_and_month(play):
    weekly = M.add_period(play, "week")
    assert (weekly["period"].dt.day_of_week == 0).all()
    assert ((weekly["date"] - weekly["period"]).dt.days.between(0, 6)).all()
    monthly = M.add_period(play, "month")
    assert (monthly["period"].dt.day == 1).all()
    assert (M.add_period(play, "day")["period"] == play["date"].dt.normalize()).all()
    with pytest.raises(ValueError):
        M.add_period(play, "decade")


def test_popularity_trend_compares_the_start_and_the_end(play):
    got = M.popularity_trend(play, days=2)
    daily = play.groupby("date")["popularity_trend_7d"].mean().dropna().sort_index()
    assert got["start"] == pytest.approx(daily.iloc[:2].mean())
    assert got["end"] == pytest.approx(daily.iloc[-2:].mean())
    assert got["change"] == pytest.approx(got["end"] - got["start"])
    assert np.isnan(M.popularity_trend(play.iloc[:0])["change"])


def test_headline_has_the_numbers_of_the_top_strip(dash_synthetic):
    play = dash_synthetic.playlist
    songs = M.song_table(play)
    artists = M.artist_table(play, M.total_chart_points(play), play["date"].nunique())
    head = M.headline(play, play, dash_synthetic.daily_summary, songs, artists)
    assert head["days"] == play["date"].nunique()
    assert head["song_days"] == len(play) and head["songs"] == play["track_id"].nunique()
    assert head["artists"] == play["primary_artist"].nunique()
    assert head["avg_rank"] == pytest.approx(25.5) and head["best_rank"] == 1
    assert head["explicit_share_rows"] == pytest.approx(play["is_explicit"].mean())
    assert head["explicit_share_chart"] == pytest.approx(dash_synthetic.daily_summary["explicit_share"].mean())
    assert head["adi_leader"] == artists.iloc[0]["artist"]
    assert head["adi_top10"] == pytest.approx(artists["adi"].head(10).sum())


def test_headline_of_nothing_is_empty_but_complete(dash_synthetic):
    play = dash_synthetic.playlist
    head = M.headline(play.iloc[:0], play, dash_synthetic.daily_summary)
    assert head["songs"] == 0 and head["song_days"] == 0
    assert np.isnan(head["avg_rank"]) and head["adi_leader"] is None
    assert head["longest_days_on_chart"] == 0


def test_the_real_leader_is_taylor_swift_with_about_7_5_percent(dash_real):
    play = dash_real.playlist
    bounds = chart_bounds(play)
    rows = F.apply_filters(play, F.default_state(bounds))
    artists = M.artist_table(rows, M.total_chart_points(play), play["date"].nunique())
    assert artists.iloc[0]["artist"] == "Taylor Swift"
    assert artists.iloc[0]["adi"] == pytest.approx(7.53, abs=0.01)


# ------------------------------------------------------------------------------------------------- the quiet median
def test_median_or_nan_is_quiet_when_there_is_nothing_to_take_the_median_of():
    """A one-day window has no volatility for any song: the median must be NaN without numpy's 'Mean of empty slice' warning."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(M.median_or_nan(pd.Series([], dtype=float)))
        assert np.isnan(M.median_or_nan(pd.Series([np.nan, np.nan])))
        assert M.median_or_nan(pd.Series([1.0, np.nan, 3.0])) == 2.0
        assert M.median_or_nan([4, 1, 2]) == 2.0


def test_the_headline_of_a_one_day_window_has_no_volatility_and_raises_no_warning(dash_synthetic):
    """On one day every song has one observation, so its Rank Volatility Index does not exist and the median volatility is NaN."""
    import warnings
    play = dash_synthetic.playlist
    one_day = play[play["date"] == play["date"].min()]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        head = M.headline(one_day, one_day, dash_synthetic.daily_summary.iloc[:1])
    assert head["songs"] == one_day["track_id"].nunique() and head["days"] == 1
    assert np.isnan(head["median_volatility"])
