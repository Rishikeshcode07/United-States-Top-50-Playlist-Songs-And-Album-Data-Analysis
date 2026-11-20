"""
tests/test_features.py  -  tests for the KPI and feature tables (step 5 of the pipeline).

WHAT THIS FILE IS
    The numbers of the dashboards, checked against an independent answer key. For the hand-made chart,
    tests/synthetic.py works out every KPI with plain Python loops (no pandas group-by, no shared code), and these tests
    compare the pipeline's tables with that key, column by column:

        playlist_daily   movement, previous position, rank change, 7-day gain, run numbers, chart day, age
        dim_track        Days on Chart, Average Rank, Best Rank, runs, entry/exit rank, Top-10 days, chart points ...
        dim_artist       unique songs, chart points, Artist Dominance Index
        daily_summary    new entries, re-entries, exits, explicit share, collab share ...
        dim_date         the complete calendar, including the missing day

    A few values are also worked out by hand for ONE song (song 50 climbs two places a day, so its 7-day average and
    volatility can be written as arithmetic).

WHY WE NEED IT
    A KPI that is calculated wrongly does not crash; it just shows a wrong number on a dashboard. Only a comparison
    with an independently calculated answer can catch that.

WHAT HAPPENS WHEN IT RUNS
    The session fixtures run the pipeline once on the synthetic chart; each test compares one table or one column.

EXPECTED OUTCOME
    All tests pass. (Averages are stored rounded to 2 decimals, so they are compared with a tolerance of 0.005.)

MOTIVE
    Calculate every KPI twice, in two different ways, and insist that the answers agree.
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.config import CHART_SIZE, DB_TABLE_ORDER, TABLE_COLUMNS
from tests import synthetic as syn
from tests.helpers import mismatches

ROUNDED_2 = 0.005 + 1e-9               # a value stored with 2 decimals differs from the exact one by at most 0.005


@pytest.fixture(scope="module")
def merged(synthetic_tables, plan):
    """playlist_daily joined 1-to-1 with the answer key (every row knows its true song_id)."""
    return synthetic_tables["playlist_daily"].merge(syn.expected_daily(plan), on=["date", "position"],
                                                    suffixes=("", "_exp"), validate="one_to_one")


@pytest.fixture(scope="module")
def songs(synthetic_tables, merged, plan):
    """dim_track joined with the answer key. The link between the two numbering systems is read from the daily rows."""
    link = merged[["song_id", "track_id"]].drop_duplicates().set_index("song_id")["track_id"]
    key = syn.expected_songs(plan)
    key["track_id"] = key["song_id"].map(link)
    return synthetic_tables["dim_track"].merge(key, on="track_id", suffixes=("", "_exp"), validate="one_to_one")


def rows_of(merged, song_id):
    return merged[merged["song_id"] == song_id].sort_values("snapshot_index").reset_index(drop=True)


# ================================================================================ playlist_daily against the answer key
@pytest.mark.parametrize("column", ["snapshot_index", "prev_position", "rank_change", "rank_gain_7d", "stint_no", "days_in_stint",
                                    "stint_length", "chart_day_number", "calendar_age_days", "movement"])
def test_daily_column_equals_the_answer_key(merged, column):
    assert mismatches(merged[column], merged[column + "_exp"]) == []


def test_rank_change_is_positive_when_a_song_climbs(merged):
    climbed = merged[merged["movement"] == "Up"]
    fell = merged[merged["movement"] == "Down"]
    assert (climbed["rank_change"] > 0).all() and (fell["rank_change"] < 0).all()
    assert (merged.loc[merged["movement"] == "Same", "rank_change"] == 0).all()


def test_the_seven_day_gain_needs_a_full_week_on_the_chart(merged):
    assert merged["rank_gain_7d"].notna().sum() == merged["rank_gain_7d_exp"].notna().sum() > 0
    first_week = merged[merged["days_in_stint"] <= 7]
    assert first_week["rank_gain_7d"].isna().all()                          # the 8th day of a run is the first with a gain


# ================================================================================ one song worked out by hand
@pytest.fixture(scope="module")
def song50(merged):
    """Song 50 enters the synthetic chart in position 50 and climbs exactly 2 places a day for 11 days: 50, 48, ... 30."""
    return rows_of(merged, 50)


def test_song50_positions_are_what_the_escalator_says(song50):
    assert song50["position"].tolist() == list(range(50, 29, -2))


def test_song50_seven_day_average_rank(song50):
    assert song50["avg_rank_7d"].iloc[0] == 50.0                            # first day: only itself
    assert song50["avg_rank_7d"].iloc[2] == 48.0                            # mean(50, 48, 46)
    assert song50["avg_rank_7d"].iloc[10] == 36.0                           # mean(42, 40, 38, 36, 34, 32, 30)


def test_song50_seven_day_rank_volatility(song50):
    assert song50["rank_volatility_7d"].iloc[:2].isna().all()               # fewer than 3 days of history: not reported
    assert song50["rank_volatility_7d"].iloc[2] == 2.0                      # sample std of 50, 48, 46
    assert song50["rank_volatility_7d"].iloc[10] == round(2 * math.sqrt(7 * 8 / 12), 2) == 4.32   # std of 7 equally spaced values


def test_song50_popularity_trend_is_a_rolling_mean_inside_the_run(song50):
    # popularity of song 50 on snapshot s (0-based) is 60 + s, so the rolling mean is easy to write down
    assert song50["popularity_trend_7d"].iloc[0] == 60.0
    assert song50["popularity_trend_7d"].iloc[2] == 61.0                    # mean(60, 61, 62)
    assert song50["popularity_trend_7d"].iloc[10] == 67.0                   # mean(64 ... 70): only the last 7 days count


def test_song50_weekly_gain(song50):
    assert song50["rank_gain_7d"].iloc[:7].isna().all()
    assert (song50["rank_gain_7d"].iloc[7:] == 14).all()                    # 7 days x 2 places


def test_the_popularity_trend_never_averages_across_an_absence(merged):
    rows = rows_of(merged, 30)                                              # song 30: run of 4 days, away, run of 5 days
    second_run_first_day = rows[rows["stint_no"] == 2].iloc[0]
    # on the first day of its second run, the trend is that day's own (repaired) value, not an average with the first run
    assert second_run_first_day["popularity_trend_7d"] == second_run_first_day["popularity_clean"] == 81.5


# ================================================================================ dim_track against the answer key
SONG_COLUMNS = ["song_clean", "artist_credit", "primary_artist", "artist_count", "days_on_chart", "best_rank", "worst_rank",
                "n_runs", "longest_run_days", "first_seen", "last_seen", "entry_rank", "exit_rank", "days_in_top10",
                "days_at_number_one", "chart_points", "span_snapshots", "on_chart_at_window_start", "on_chart_at_window_end"]


def test_the_answer_key_and_the_pipeline_describe_the_same_songs(songs):
    assert len(songs) == 71


@pytest.mark.parametrize("column", SONG_COLUMNS)
def test_song_column_equals_the_answer_key(songs, column):
    assert mismatches(songs[column], songs[column + "_exp"]) == []


def test_average_rank_equals_the_answer_key_up_to_rounding(songs):
    assert mismatches(songs["avg_rank"], songs["avg_rank_exp"], tolerance=ROUNDED_2) == []


def test_song50_row_of_dim_track(songs):
    row = songs[songs["song_id"] == 50].iloc[0]
    assert (row["days_on_chart"], row["best_rank"], row["worst_rank"], row["avg_rank"]) == (11, 30, 50, 40.0)
    assert row["chart_points"] == 121 and row["n_runs"] == 1 and row["presence_ratio"] == 1.0
    assert row["rank_volatility_index"] == round(2 * math.sqrt(11), 2) == 6.63
    assert row["avg_daily_rank_move"] == 2.0
    assert (row["entry_rank"], row["exit_rank"], row["days_in_top10"]) == (50, 30, 0)
    assert row["longevity_class"] == "Short (4-14)"
    assert row["run_status"] == "On chart the whole window"


def test_one_day_song_has_no_volatility_and_a_censored_status(songs):
    row = songs[songs["song_id"] == 70].iloc[0]                             # enters on the last snapshot only
    assert row["days_on_chart"] == 1 and pd.isna(row["rank_volatility_index"])
    assert row["run_status"] == "Still on chart at end"
    assert row["longevity_class"] == "Flash (1-3 days)"
    assert pd.isna(row["avg_popularity"])                                   # its only popularity value was an unfillable zero


def test_run_status_covers_every_song_exactly_once(synthetic_tables):
    dt = synthetic_tables["dim_track"]
    assert set(dt["run_status"]) <= {"On chart the whole window", "Already on chart at start", "Still on chart at end",
                                     "Complete history observed"}
    assert (dt["run_status"] == "Complete history observed").sum() == 1      # only the filler song: it enters and leaves inside the window
    assert dt["on_chart_at_window_start"].sum() == 50                        # the 50 songs of the first chart


def test_a_song_with_two_runs_reports_its_longest_run_and_presence(songs):
    row = songs[songs["song_id"] == 30].iloc[0]
    assert (row["n_runs"], row["longest_run_days"], row["days_on_chart"], row["span_snapshots"]) == (2, 5, 9, 11)
    assert row["presence_ratio"] == round(9 / 11, 3)


def test_the_first_chart_day_and_peak_dates(synthetic_tables):
    dt = synthetic_tables["dim_track"]
    assert (dt["peak_date"] >= dt["first_seen"]).all() and (dt["peak_date"] <= dt["last_seen"]).all()
    assert (dt["days_to_peak"] == (dt["peak_date"] - dt["first_seen"]).dt.days).all()


# ================================================================================ dim_artist and the bridge
@pytest.fixture(scope="module")
def artists(synthetic_tables, plan):
    key = syn.expected_artists(plan)
    return synthetic_tables["dim_artist"].merge(key, on="artist_name", suffixes=("", "_exp"), validate="one_to_one")


ARTIST_COLUMNS = ["unique_tracks", "unique_tracks_as_primary", "chart_slot_days", "chart_slot_days_primary", "days_present",
                  "first_seen", "last_seen", "best_rank", "days_in_top10", "days_at_number_one", "chart_points",
                  "chart_points_primary"]


def test_the_same_81_artists_in_both(synthetic_tables, artists):
    assert len(synthetic_tables["dim_artist"]) == len(artists) == 81


@pytest.mark.parametrize("column", ARTIST_COLUMNS)
def test_artist_column_equals_the_answer_key(artists, column):
    assert mismatches(artists[column], artists[column + "_exp"]) == []


def test_artist_average_rank_and_dominance_index_equal_the_answer_key(artists):
    assert mismatches(artists["avg_rank"], artists["avg_rank_exp"], tolerance=ROUNDED_2) == []
    assert mismatches(artists["dominance_index_primary"], artists["dominance_index_primary_exp"], tolerance=0.00005 + 1e-9) == []


def test_dominance_index_adds_up_to_one_hundred_percent(synthetic_tables):
    shares = synthetic_tables["dim_artist"]["dominance_index_primary"]
    assert shares.sum() == pytest.approx(100.0, abs=0.01)
    assert (shares >= 0).all() and (shares <= 100).all()


def test_all_credits_basis_counts_every_artist_of_a_song(synthetic_tables):
    da = synthetic_tables["dim_artist"]
    # a song with two artists counts for both of them, so this basis adds up to MORE than 100%
    assert da["dominance_index_all_credits"].sum() > 100
    assert (da["dominance_index_all_credits"] >= da["dominance_index_primary"] - 1e-9).all()


def test_artists_are_sorted_by_dominance_then_name(synthetic_tables):
    da = synthetic_tables["dim_artist"]
    expected = da.sort_values(["dominance_index_primary", "artist_name"], ascending=[False, True]).reset_index(drop=True)
    pd.testing.assert_frame_equal(da, expected)                              # ties never depend on the pandas version


def test_the_bridge_links_every_song_to_its_artists(synthetic_tables):
    bridge, dt = synthetic_tables["bridge_track_artist"], synthetic_tables["dim_track"]
    assert bridge.groupby("track_id")["is_primary"].sum().eq(1).all()
    per_song = bridge.groupby("track_id").size()
    assert (per_song.reindex(dt["track_id"]).to_numpy() == dt["artist_count"].to_numpy()).all()
    assert bridge.duplicated(["track_id", "artist_name"]).sum() == 0
    first = bridge[bridge["artist_order"] == 1]
    assert first["is_primary"].all() and not bridge.loc[bridge["artist_order"] > 1, "is_primary"].any()


# ================================================================================ daily_summary and dim_date
@pytest.fixture(scope="module")
def summary(synthetic_tables, plan):
    return synthetic_tables["daily_summary"].merge(syn.expected_daily_summary(plan), on="date", suffixes=("", "_exp"),
                                                   validate="one_to_one")


def test_entries_and_exits_equal_the_answer_key(summary, plan):
    key = syn.expected_entries_exits(plan)
    for _, row in summary.iterrows():
        assert (row["new_entries"], row["re_entries"], row["exits"]) == key[int(row["snapshot_index"])]


def test_turnover_rate_is_entries_over_chart_size(summary):
    entries = summary["new_entries"] + summary["re_entries"]
    assert mismatches(summary["turnover_rate_pct"], entries / CHART_SIZE * 100, tolerance=0.05 + 1e-9) == []
    assert summary.loc[summary["snapshot_index"] == 1, "turnover_rate_pct"].iloc[0] == 0.0     # nothing is "new" on day one


def test_the_shock_day_rule_flags_the_days_with_three_entries(summary):
    flagged = summary.loc[summary["is_high_turnover_day"], "snapshot_index"].tolist()
    assert flagged == [5, 7]                                                 # the filler day and the day song 30 returns


@pytest.mark.parametrize("column, tolerance", [("unique_tracks", 0), ("unique_primary_artists", 0), ("explicit_share", 0.00005 + 1e-9),
                                               ("collab_share", 0.00005 + 1e-9), ("single_share", 0.00005 + 1e-9),
                                               ("avg_duration_min", 0.0005 + 1e-9)])
def test_daily_summary_column_equals_the_answer_key(summary, column, tolerance):
    assert mismatches(summary[column], summary[column + "_exp"], tolerance=tolerance) == []


def test_daily_summary_popularity_uses_the_repaired_values(synthetic_tables):
    daily, ds = synthetic_tables["playlist_daily"], synthetic_tables["daily_summary"]
    for _, row in ds.iterrows():
        day = daily[daily["date"] == row["date"]]["popularity_clean"]
        assert row["avg_popularity"] == pytest.approx(day.mean(), abs=0.005)       # the empty value (song 70) is left out of the mean
        assert row["median_popularity"] == pytest.approx(day.median(), abs=0.005)


@pytest.fixture(scope="module")
def calendar(synthetic_tables, plan):
    return synthetic_tables["dim_date"].merge(syn.expected_calendar(plan), on="date", suffixes=("", "_exp"), validate="one_to_one")


@pytest.mark.parametrize("column", ["date_key", "has_snapshot", "year", "month", "month_name", "year_month", "quarter", "iso_week",
                                    "day_of_week", "day_name", "is_weekend", "season"])
def test_calendar_column_equals_python_dates(calendar, column):
    assert mismatches(calendar[column], calendar[column + "_exp"]) == []


def test_the_calendar_is_continuous(synthetic_tables):
    dd = synthetic_tables["dim_date"]
    assert len(dd) == 12 and dd["date"].diff().dropna().eq(pd.Timedelta(days=1)).all()


# ================================================================================ the shape of all seven tables
def test_every_table_has_exactly_the_promised_columns(synthetic_tables):
    assert set(synthetic_tables) == set(TABLE_COLUMNS)
    for name, columns in TABLE_COLUMNS.items():
        assert list(synthetic_tables[name].columns) == columns, name


def test_fact_table_is_the_daily_table_without_the_descriptive_columns(synthetic_tables):
    daily, fact = synthetic_tables["playlist_daily"], synthetic_tables["fact_chart_daily"]
    assert len(daily) == len(fact) == 550
    for column in fact.columns:
        assert mismatches(fact[column], daily[column]) == [], column


def test_database_tables_are_the_six_tables_the_loader_expects(synthetic_tables):
    assert all(name in synthetic_tables for name in DB_TABLE_ORDER)
    assert "playlist_daily" not in DB_TABLE_ORDER                            # the wide table stays in files


def test_no_key_column_is_empty(synthetic_tables):
    keys = {"fact_chart_daily": ["date", "track_id", "position"], "dim_track": ["track_id", "track_label", "song_clean"],
            "dim_artist": ["artist_name"], "bridge_track_artist": ["track_id", "artist_name", "artist_order"],
            "dim_date": ["date", "date_key"], "daily_summary": ["date", "snapshot_index"]}
    for name, columns in keys.items():
        assert synthetic_tables[name][columns].notna().all().all(), name


def test_integer_columns_stayed_integers(synthetic_tables):
    fact = synthetic_tables["fact_chart_daily"]
    assert pd.api.types.is_integer_dtype(fact["position"]) and pd.api.types.is_integer_dtype(fact["track_id"])
    assert pd.api.types.is_datetime64_any_dtype(fact["date"])
    assert fact["is_explicit"].dtype == bool and fact["in_top10"].dtype == bool
    first_chart = fact[fact["date"] == fact["date"].min()]
    assert first_chart["rank_change"].isna().all()                           # the first chart has no yesterday
