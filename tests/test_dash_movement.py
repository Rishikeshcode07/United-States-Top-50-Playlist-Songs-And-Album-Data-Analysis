"""
tests/test_dash_movement.py  -  risers, fallers, arrivals, departures and chart runs.

WHAT THIS FILE IS
    Tests of src/dash/movement.py. The synthetic chart is an "escalator" (every song climbs 2 places a day, two new songs enter at
    the bottom) with a missing calendar day, two swaps and a song that leaves and comes back, so every rule has a small case.
    Most checks re-calculate the answer with plain Python loops and compare.

WHY WE NEED IT
    "Positions gained" has a sign, a window, a notion of "the same song in the same run" and a notion of "complete run". Each of those
    is a place where a dashboard can quietly mislead, so each gets its own test.

WHAT HAPPENS WHEN IT RUNS
    Fast. The real-data tests are skipped when data/processed/ is not built.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Movement numbers that can be checked by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dash import filters as F
from src.dash import movement as V
from src.dash.load import chart_bounds


@pytest.fixture(scope="module")
def play(dash_synthetic):
    return dash_synthetic.playlist


@pytest.fixture(scope="module")
def calendar(dash_synthetic):
    return dash_synthetic.daily_summary[["date", "snapshot_index"]]


@pytest.fixture(scope="module")
def bounds(play):
    return chart_bounds(play)


def on_day(play, snapshot):
    return play[play["snapshot_index"] == snapshot].set_index("track_id")


# ---------------------------------------------------------------------------------------------- small helpers
def test_place_band_edges():
    got = V.place_band([1, 10, 11, 20, 21, 30, 31, 40, 41, 50]).tolist()
    assert got == ["1 to 10", "1 to 10", "11 to 20", "11 to 20", "21 to 30", "21 to 30", "31 to 40", "31 to 40", "41 to 50", "41 to 50"]


def test_day_chart_is_in_order_and_marks_the_selection(play):
    day = V.day_chart(play, "2024-01-03", keep_ids=[1, 2, 3])
    assert day["position"].tolist() == list(range(1, 51))
    assert set(day.loc[day["selected"], "track_id"]) <= {1, 2, 3}
    assert V.day_chart(play, "2024-01-03")["selected"].all()
    assert V.day_chart(play, "1999-01-01").empty


def test_a_date_without_a_snapshot_moves_back_to_the_day_before(calendar):
    assert V.snapshot_on_or_before(calendar, "2024-01-06") == pd.Timestamp("2024-01-05")
    assert V.snapshot_on_or_before(calendar, "2024-01-07") == pd.Timestamp("2024-01-07")
    assert V.snapshot_on_or_before(calendar, "2023-12-31") is None


# ------------------------------------------------------------------------------------------------- moves_between
@pytest.mark.parametrize("end_snapshot, window", [(2, 1), (4, 1), (8, 3), (11, 7), (11, 10)])
def test_moves_between_matches_plain_python(play, calendar, end_snapshot, window):
    end_date = calendar.loc[calendar["snapshot_index"] == end_snapshot, "date"].iloc[0]
    result = V.moves_between(play, calendar, end_date, window)
    end, start = on_day(play, end_snapshot), on_day(play, end_snapshot - window)
    moved = {}
    for track_id in end.index:
        if track_id in start.index and end.loc[track_id, "stint_no"] == start.loc[track_id, "stint_no"]:
            moved[track_id] = start.loc[track_id, "position"] - end.loc[track_id, "position"]
    assert dict(zip(result["moved"]["track_id"], result["moved"]["places_gained"])) == moved
    in_start = {(i, start.loc[i, "stint_no"]) for i in start.index}
    in_end = {(i, end.loc[i, "stint_no"]) for i in end.index}
    assert set(result["entered"]["track_id"]) == {i for i, s in in_end - in_start}
    assert set(result["left"]["track_id"]) == {i for i, s in in_start - in_end}
    assert len(result["moved"]) + len(result["entered"]) == 50


def test_a_climb_is_positive(play, calendar):
    result = V.moves_between(play, calendar, "2024-01-02", 1)
    moved = result["moved"]
    assert (moved["places_gained"] == 2).sum() >= 40                # the escalator
    assert moved["places_gained"].max() <= 50
    one = moved.iloc[0]
    assert one["position_start"] - one["position_end"] == one["places_gained"]
    assert moved["places_gained"].is_monotonic_decreasing


def test_a_window_counts_snapshots_so_the_missing_day_is_not_invented(play, calendar):
    result = V.moves_between(play, calendar, "2024-01-07", 1)
    assert result["start_date"] == pd.Timestamp("2024-01-05")      # 2024-01-06 has no snapshot
    assert result["end_date"] == pd.Timestamp("2024-01-07")
    again = V.moves_between(play, calendar, "2024-01-06", 1)       # a date without a snapshot is read as the day before it
    assert again["end_date"] == pd.Timestamp("2024-01-05")


def test_a_window_that_reaches_before_the_first_day_is_empty(play, calendar):
    result = V.moves_between(play, calendar, "2024-01-03", 5)
    assert result["start_date"] is None
    assert result["moved"].empty and result["entered"].empty and result["left"].empty
    assert list(result["moved"].columns) == V.MOVE_COLUMNS
    assert V.moves_between(play, calendar, "2024-01-03", 0)["moved"].empty


def test_a_returning_song_is_an_arrival_not_a_mover(play, calendar):
    runs_per_song = play.groupby("track_id")["stint_no"].nunique()
    track_id = runs_per_song[runs_per_song > 1].index[0]
    song = play[play["track_id"] == track_id].sort_values("snapshot_index")
    first_run_snapshot = song.loc[song["stint_no"] == song["stint_no"].min(), "snapshot_index"].iloc[0]
    second_run_snapshot = song.loc[song["stint_no"] == song["stint_no"].max(), "snapshot_index"].iloc[-1]
    end_date = calendar.loc[calendar["snapshot_index"] == second_run_snapshot, "date"].iloc[0]
    result = V.moves_between(play, calendar, end_date, second_run_snapshot - first_run_snapshot)
    assert track_id not in set(result["moved"]["track_id"])
    assert track_id in set(result["entered"]["track_id"])
    assert track_id in set(result["left"]["track_id"])


def test_moves_respect_the_selected_rows(play, calendar):
    explicit = play[play["is_explicit"]]
    result = V.moves_between(explicit, calendar, "2024-01-05", 2)
    ids = set(explicit["track_id"])
    assert set(result["moved"]["track_id"]) <= ids and set(result["entered"]["track_id"]) <= ids


# ---------------------------------------------------------------------------------------------------- runs
def test_runs_match_plain_python(play, bounds):
    runs = V.runs_table(play, bounds.first_snapshot, bounds.last_snapshot).set_index("stint_key")
    assert len(runs) == play["stint_key"].nunique()
    for key, group in list(play.groupby("stint_key"))[:25]:
        group = group.sort_values("snapshot_index")
        places = group["position"].tolist()
        snaps = group["snapshot_index"].tolist()
        row = runs.loc[key]
        assert row["days"] == len(places) and row["entry_position"] == places[0] and row["exit_position"] == places[-1]
        assert row["best_position"] == min(places)
        peak = snaps[places.index(min(places))]
        assert row["peak_snapshot"] == peak
        assert row["days_to_peak"] == peak - snaps[0] and row["days_after_peak"] == snaps[-1] - peak
        assert row["began_here"] == (bool(group["is_stint_start"].iloc[0]) and snaps[0] > bounds.first_snapshot)
        assert row["ended_here"] == (bool(group["is_stint_end"].iloc[-1]) and snaps[-1] < bounds.last_snapshot)
        if row["days_to_peak"] > 0:
            assert row["rise_per_day"] == pytest.approx((places[0] - min(places)) / row["days_to_peak"])
        else:
            assert np.isnan(row["rise_per_day"])


def test_runs_cut_by_the_edges_of_the_file_are_not_complete(play, bounds):
    runs = V.runs_table(play, bounds.first_snapshot, bounds.last_snapshot)
    first_day = runs[runs["start_snapshot"] == bounds.first_snapshot]
    last_day = runs[runs["end_snapshot"] == bounds.last_snapshot]
    assert len(first_day) == 50 and not first_day["began_here"].any() and not first_day["complete"].any()
    assert len(last_day) == 50 and not last_day["ended_here"].any() and not last_day["complete"].any()
    assert (runs["complete"] == (runs["began_here"] & runs["ended_here"])).all()


def test_a_date_filter_cuts_runs_and_they_are_flagged(play, bounds):
    middle = play[play["snapshot_index"].between(4, 8)]
    runs = V.runs_table(middle, bounds.first_snapshot, bounds.last_snapshot)
    really_started = on_day(play, 4)["is_stint_start"].astype(bool)
    at_the_cut = runs[runs["start_snapshot"] == 4].set_index("track_id")["began_here"]
    assert (at_the_cut == really_started.reindex(at_the_cut.index)).all()
    assert at_the_cut.any() and not at_the_cut.all()          # the two new songs began there, the other 48 were already running


def test_rise_and_decline_tables_respect_their_thresholds(play, bounds):
    runs = V.runs_table(play, bounds.first_snapshot, bounds.last_snapshot)
    tables = V.rise_decline_tables(runs, min_gain=10, min_after_peak=3, min_exit_position=21)
    risers, decliners = tables["risers"], tables["decliners"]
    assert (risers["entry_position"] - risers["best_position"] >= 10).all() and risers["began_here"].all()
    assert (risers["days_to_peak"] >= 1).all()
    assert risers["rise_per_week"].is_monotonic_decreasing
    assert (risers["places_gained"] == risers["entry_position"] - risers["best_position"]).all()
    assert decliners["complete"].all() and (decliners["days_after_peak"] >= 3).all() and (decliners["exit_position"] >= 21).all()
    assert decliners["fall_per_week"].is_monotonic_increasing
    assert (decliners["places_lost"] == decliners["exit_position"] - decliners["best_position"]).all()


def test_rise_and_decline_of_nothing(bounds, play):
    empty = V.runs_table(play.iloc[:0], bounds.first_snapshot, bounds.last_snapshot)
    assert empty.empty and list(empty.columns) == V.RUN_COLUMNS
    tables = V.rise_decline_tables(empty)
    assert tables["risers"].empty and tables["decliners"].empty


# ------------------------------------------------------------------------------------------ arrivals and exits
def test_entries_and_exits_agree_with_the_daily_summary(dash_synthetic):
    play, summary = dash_synthetic.playlist, dash_synthetic.daily_summary
    entries, exits = V.entries_table(play), V.exits_table(play)
    assert len(entries) == (summary["new_entries"] + summary["re_entries"]).sum()
    assert len(exits) == summary["exits"].sum()
    assert set(entries["movement"]) <= {"New entry", "Re-entry"}
    per_day = entries.groupby("date").size().reindex(summary["date"]).fillna(0)
    assert (per_day.to_numpy() == (summary["new_entries"] + summary["re_entries"]).to_numpy()).all()


def test_entries_and_exits_on_the_real_file(dash_real):
    play = dash_real.playlist
    entries, exits = V.entries_table(play), V.exits_table(play)
    assert len(entries) == 1_931 and (entries["movement"] == "New entry").sum() == 894
    assert len(exits) == dash_real.daily_summary["exits"].sum() == 1_931


def test_entry_exit_matrix_counts_every_complete_run(play, bounds):
    runs = V.runs_table(play, bounds.first_snapshot, bounds.last_snapshot)
    counts = V.entry_exit_matrix(runs)
    assert counts.shape == (5, 5) and list(counts.index) == V.BAND_LABELS == list(counts.columns)
    assert counts.to_numpy().sum() == runs["complete"].sum()
    days = V.entry_exit_matrix(runs, "median_days")
    assert days.shape == (5, 5)
    assert days.isna().to_numpy().sum() + (counts.to_numpy() > 0).sum() == 25
    assert V.entry_exit_matrix(runs.iloc[:0]).to_numpy().sum() == 0


def test_chart_flow_smooths_the_turnover(dash_synthetic):
    flow = V.chart_flow(dash_synthetic.daily_summary, smooth=3)
    assert (flow["arrivals"] == flow["new_entries"] + flow["re_entries"]).all()
    expected = flow["turnover_rate_pct"].rolling(3, min_periods=1).mean()
    assert np.allclose(flow["turnover_smooth"], expected)
    assert flow["date"].is_monotonic_increasing


def test_calendar_cells_put_every_day_in_a_week_and_a_weekday(dash_synthetic):
    cells = V.calendar_cells(dash_synthetic.daily_summary, "turnover_rate_pct")
    assert (cells["week_start"].dt.day_of_week == 0).all()
    assert (cells["date"] - cells["week_start"]).dt.days.tolist() == cells["weekday"].tolist()
    assert len(cells) == len(dash_synthetic.daily_summary)


# -------------------------------------------------------------------------------------------------- trend lines
def test_rank_matrix_has_one_column_per_song_and_gaps_where_it_was_absent(play):
    runs_per_song = play.groupby("track_id")["stint_no"].nunique()
    returning = runs_per_song[runs_per_song > 1].index[0]
    always = play.groupby("track_id").size().idxmax()
    matrix = V.rank_matrix(play, [returning, always, -1])
    assert list(matrix.columns) == [returning, always]
    for track_id in (returning, always):
        mine = play[play["track_id"] == track_id].set_index("date")["position"]
        assert matrix[track_id].notna().sum() == len(mine)
        assert np.allclose(matrix[track_id].dropna(), mine.reindex(matrix[track_id].dropna().index))
    assert matrix[returning].isna().sum() > 0                  # the song that left and came back has days off the chart
    assert V.rank_matrix(play, [-1, -2]).empty


def test_everything_works_for_a_filtered_selection(dash_synthetic, bounds, calendar):
    play = dash_synthetic.playlist
    state = F.FilterState.create(bounds.first_date, bounds.last_date, rank_low=1, rank_high=20)
    rows = F.apply_filters(play, state)
    runs = V.runs_table(rows, bounds.first_snapshot, bounds.last_snapshot)
    assert not runs.empty
    result = V.moves_between(rows, calendar, "2024-01-10", 2)
    assert (result["moved"]["position_end"] <= 20).all() and (result["moved"]["position_start"] <= 20).all()
