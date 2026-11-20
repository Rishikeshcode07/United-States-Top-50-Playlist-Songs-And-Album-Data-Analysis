"""
tests/test_eda_ranking.py  -  tests for the chart runs (stints), the decline speed and the position-to-position transitions.

WHAT THIS FILE IS
    A tiny hand-made chart (four runs of three songs on ten snapshot days) whose answers are worked out on paper, plus checks on
    the real chart: 1,981 runs, 50 cut off at the start, 50 cut off at the end, and every row of the file accounted for.

WHY WE NEED IT
    Every statement about "how long songs stay" and "how fast they fall" rests on the run table. The most dangerous mistake
    would be silent: a run that is still alive on the last day treated as finished makes songs look short-lived. The tests make
    sure the cut-off runs are flagged, the first best place wins a tie, a gap inside a run is refused, and no song-day gets lost
    on its way into the transition table.

WHAT HAPPENS WHEN IT RUNS
    Small tables go in, the expected tables come out. The real-data tests run the whole pipeline once on the real CSV and are
    skipped when data/raw/Atlantic_United_States.csv is missing.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Longevity numbers that can be defended: complete runs only where it matters, honest flags everywhere else.
"""
import numpy as np
import pandas as pd
import pytest

from src.eda.ranking import PositionTransitions, decline_speed, position_transitions, stint_table


# ================================================================================ a hand-made chart
def _rows(stint_key, track_id, label, snapshots_and_positions):
    start = pd.Timestamp("2025-01-01")
    return [{"stint_key": stint_key, "track_id": track_id, "track_label": label, "snapshot_index": snapshot,
             "date": start + pd.Timedelta(days=snapshot - 1), "position": position}
            for snapshot, position in snapshots_and_positions]


@pytest.fixture()
def toy():
    """Ten snapshots. A starts on the first one, C is still there on the last one, B leaves and comes back for one day."""
    rows = (_rows("A#1", "A", "Song A", [(1, 10), (2, 5), (3, 8)])
            + _rows("B#1", "B", "Song B", [(4, 30), (5, 20), (6, 20), (7, 25)])      # its best place (20) is reached twice: first on 5
            + _rows("B#2", "B", "Song B", [(9, 40)])
            + _rows("C#1", "C", "Song C", [(8, 12), (9, 11), (10, 15)]))
    return pd.DataFrame(rows)


# ================================================================================ stint_table
def test_one_row_per_run_in_the_order_track_then_start(toy):
    stints = stint_table(toy)
    assert stints["stint_key"].tolist() == ["A#1", "B#1", "B#2", "C#1"]


def test_every_number_of_every_run(toy):
    stints = stint_table(toy).set_index("stint_key")
    expected = pd.DataFrame({
        "start_snapshot": [1, 4, 9, 8], "end_snapshot": [3, 7, 9, 10], "length": [3, 4, 1, 3],
        "entry_position": [10, 30, 40, 12], "exit_position": [8, 25, 40, 15], "best_position": [5, 20, 40, 11],
        "peak_snapshot": [2, 5, 9, 9], "days_to_peak": [1, 1, 0, 1], "days_after_peak": [1, 2, 0, 1],
    }, index=["A#1", "B#1", "B#2", "C#1"])
    pd.testing.assert_frame_equal(stints[expected.columns], expected, check_dtype=False, check_names=False)
    assert stints.loc["B#1", "start_date"] == pd.Timestamp("2025-01-04") and stints.loc["B#1", "end_date"] == pd.Timestamp("2025-01-07")


def test_the_edge_of_the_window_decides_which_runs_are_complete(toy):
    stints = stint_table(toy).set_index("stint_key")
    assert stints["left_censored"].to_dict() == {"A#1": True, "B#1": False, "B#2": False, "C#1": False}
    assert stints["right_censored"].to_dict() == {"A#1": False, "B#1": False, "B#2": False, "C#1": True}
    assert stints["complete"].to_dict() == {"A#1": False, "B#1": True, "B#2": True, "C#1": False}


def test_a_tie_for_the_best_place_goes_to_the_first_day(toy):
    stints = stint_table(toy).set_index("stint_key")
    assert stints.loc["B#1", "peak_snapshot"] == 5 and stints.loc["B#1", "days_to_peak"] == 1


def test_the_order_of_the_rows_does_not_matter(toy):
    shuffled = toy.sample(frac=1.0, random_state=11)
    pd.testing.assert_frame_equal(stint_table(shuffled), stint_table(toy))


def test_the_input_table_is_left_alone(toy):
    before = toy.copy()
    stint_table(toy)
    pd.testing.assert_frame_equal(toy, before)


def test_a_run_with_a_gap_inside_is_refused(toy):
    broken = pd.concat([toy, pd.DataFrame(_rows("G#1", "G", "Song G", [(1, 1), (2, 2), (4, 3)]))], ignore_index=True)
    with pytest.raises(ValueError, match="gap"):
        stint_table(broken)


def test_a_missing_column_is_named(toy):
    with pytest.raises(ValueError, match="track_label"):
        stint_table(toy.drop(columns=["track_label"]))


# ================================================================================ decline_speed
@pytest.fixture()
def stints_for_decline():
    return pd.DataFrame({
        "stint_key": ["X#1", "Y#1", "Z#1", "W#1", "V#1"],
        "complete": [True, True, False, True, True],
        "days_after_peak": [28, 56, 90, 27, 60],
        "exit_position": [41, 36, 50, 45, 15],
        "best_position": [1, 6, 2, 3, 5],
    })


def test_decline_speed_is_positions_lost_per_week_and_the_slowest_comes_first(stints_for_decline):
    result = decline_speed(stints_for_decline)
    # X: (41 - 1) / 28 x 7 = 10.0      Y: (36 - 6) / 56 x 7 = 3.75
    assert result["stint_key"].tolist() == ["Y#1", "X#1"]
    assert result["positions_lost_per_week"].tolist() == pytest.approx([3.75, 10.0])


def test_decline_speed_leaves_out_unfinished_runs_short_fades_and_removals(stints_for_decline):
    kept = set(decline_speed(stints_for_decline)["stint_key"])
    assert "Z#1" not in kept            # not complete
    assert "W#1" not in kept            # only 27 days after the peak (limit 28)
    assert "V#1" not in kept            # left the chart at position 15: removed, not faded


def test_the_limits_can_be_changed(stints_for_decline):
    assert decline_speed(stints_for_decline, min_days_after_peak=27)["stint_key"].tolist() == ["Y#1", "X#1", "W#1"]
    assert decline_speed(stints_for_decline, min_exit_position=10)["stint_key"].tolist()[0] == "V#1"        # (15 - 5) / 60 x 7 = 1.17


def test_decline_speed_does_not_add_a_column_to_the_input(stints_for_decline):
    decline_speed(stints_for_decline)
    assert "positions_lost_per_week" not in stints_for_decline.columns


# ================================================================================ position_transitions
@pytest.fixture()
def moves_input():
    """Snapshot 1: a=1, b=2, c=3.  Snapshot 2: a=2, b=1 (c is gone).  Snapshot 3: b=1, d=2 (a is gone)."""
    return pd.DataFrame([(1, "a", 1), (1, "b", 2), (1, "c", 3), (2, "a", 2), (2, "b", 1), (3, "b", 1), (3, "d", 2)],
                        columns=["snapshot_index", "track_id", "position"])


def test_transition_counts_by_hand(moves_input):
    result = position_transitions(moves_input, chart_size=3)
    assert isinstance(result, PositionTransitions)
    expected = pd.DataFrame([[1, 1, 0], [1, 0, 0], [0, 0, 0]], index=[1, 2, 3], columns=[1, 2, 3])      # row = position today
    assert result.counts.to_numpy().tolist() == expected.to_numpy().tolist()
    assert result.left_chart.to_dict() == {1: 0, 2: 1, 3: 1}                                             # a left from 2, c left from 3


def test_rates_by_hand(moves_input):
    result = position_transitions(moves_input, chart_size=3)
    assert result.n_today.to_dict() == {1: 2, 2: 2, 3: 1}
    assert result.stay_rate().to_dict() == pytest.approx({1: 0.5, 2: 0.0, 3: 0.0})
    assert result.exit_rate().to_dict() == pytest.approx({1: 0.0, 2: 0.5, 3: 1.0})
    probabilities = result.probabilities()
    assert probabilities.loc[1].tolist() == pytest.approx([0.5, 0.5, 0.0])
    assert probabilities.loc[2].tolist() == pytest.approx([0.5, 0.0, 0.0])


def test_every_song_day_with_a_following_snapshot_is_counted_exactly_once(moves_input):
    result = position_transitions(moves_input, chart_size=3)
    followed = int((moves_input["snapshot_index"] < moves_input["snapshot_index"].max()).sum())          # the last snapshot has no successor
    assert int(result.counts.to_numpy().sum() + result.left_chart.sum()) == followed == 5


def test_a_position_nobody_stood_on_has_no_rate_instead_of_an_error(moves_input):
    result = position_transitions(moves_input, chart_size=4)
    assert np.isnan(result.probabilities().loc[4]).all()
    assert np.isnan(result.exit_rate().loc[4]) and np.isnan(result.stay_rate().loc[4])


def test_the_same_song_twice_on_one_day_is_refused(moves_input):
    duplicated = pd.concat([moves_input, moves_input.iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        position_transitions(duplicated, chart_size=3)


# ================================================================================ the real chart
def test_runs_of_the_real_chart(real_result):
    playlist = real_result.tables["playlist_daily"]
    dim_track = real_result.tables["dim_track"].set_index("track_id")
    stints = stint_table(playlist)

    assert len(stints) == 1981 and stints["track_id"].nunique() == 944
    assert int(stints["left_censored"].sum()) == 50              # the 50 songs that were on the chart on the first day
    assert int(stints["right_censored"].sum()) == 50             # the 50 songs that were still there on the last day
    assert int(stints["complete"].sum()) == 1881
    assert int(stints["length"].sum()) == len(playlist) == 27_750

    per_song = stints.groupby("track_id")
    pd.testing.assert_series_equal(per_song.size().rename("n_runs"), dim_track["n_runs"].sort_index(), check_dtype=False)
    pd.testing.assert_series_equal(per_song["length"].sum().rename("days_on_chart"), dim_track["days_on_chart"].sort_index(), check_dtype=False)
    pd.testing.assert_series_equal(per_song["best_position"].min().rename("best_rank"), dim_track["best_rank"].sort_index(), check_dtype=False)


def test_decline_speed_on_the_real_chart_follows_an_independent_calculation(real_result):
    stints = stint_table(real_result.tables["playlist_daily"])
    result = decline_speed(stints)
    assert len(result) > 0
    assert result["complete"].all() and (result["days_after_peak"] >= 28).all() and (result["exit_position"] >= 21).all()
    assert result["positions_lost_per_week"].is_monotonic_increasing
    for row in result.head(25).itertuples():                    # a plain-Python re-calculation for the first rows
        assert row.positions_lost_per_week == pytest.approx((row.exit_position - row.best_position) * 7.0 / row.days_after_peak)


def test_transitions_of_the_real_chart_lose_no_row(real_result):
    playlist = real_result.tables["playlist_daily"]
    result = position_transitions(playlist)
    followed = int((playlist["snapshot_index"] < playlist["snapshot_index"].max()).sum())
    assert followed == 27_700                                    # 27,750 song-days minus the 50 songs of the last day
    assert int(result.counts.to_numpy().sum() + result.left_chart.sum()) == followed
    assert result.counts.shape == (50, 50)
    assert np.allclose(result.probabilities().sum(axis=1) + result.exit_rate(), 1.0)
