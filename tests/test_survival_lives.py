"""
tests/test_survival_lives.py  -  checks of the definition of a stay (src/survival/lives.py).

WHAT THIS FILE IS
    Small songs whose answer can be worked out by hand: where a stay ends, when its end counts as "seen" and when it is censored, how the
    boundary of the rule behaves (an absence of exactly 7 snapshots ends a stay, 6 does not), the chart-day table, the album-drop size, and
    one check on the real file (894 stays, 835 finished, 59 censored, 19,404 chart days).

WHY WE NEED IT
    Every number of Phase 6B depends on this definition. A stay that ends one day too early or too late moves every curve.

EXPECTED OUTCOME
    All tests pass; the real-file test is skipped when data/raw is not there.

MOTIVE
    The definition is simple enough to be checked by hand, so it is.
"""
import numpy as np
import pandas as pd
import pytest

from src.survival.lives import GAP, LIVES_COLUMNS, build_lives, chart_with_snapshots, drop_sizes, person_days, snapshot_numbers


def chart_of(**songs):
    rows = [(name, s, 10 + k) for name, snaps in songs.items() for k, s in enumerate(snaps)]
    return pd.DataFrame(rows, columns=["track_id", "s", "position"])


N = 20                                    # snapshots 0 ... 19


def one(snaps, gap=GAP, n=N):
    chart = chart_of(song=snaps)
    return build_lives(chart, ["song"], n, gap).iloc[0]


def test_default_rule_is_seven_snapshots():
    assert GAP == 7


def test_unbroken_run_in_the_middle_is_a_finished_stay():
    life = one([3, 4, 5, 6, 7])
    assert (life.stay, life.exit_seen, life.last_snapshot, life.short_breaks, life.first_run_days, life.returned) == (5, True, 7, 0, 5, False)


def test_short_break_is_inside_the_stay_and_not_counted():
    life = one([3, 4, 5, 9, 10])                          # three snapshots missing between 5 and 9
    assert life.stay == 5 and life.short_breaks == 1 and life.first_run_days == 3
    assert life.exit_seen and life.last_snapshot == 10 and not life.returned


def test_a_break_of_exactly_seven_ends_the_stay_and_six_does_not():
    ends = one([0, 8])                                    # snapshots 1 ... 7 are missing: seven of them
    assert (ends.stay, ends.exit_seen, ends.returned, ends.last_snapshot) == (1, True, True, 0)
    goes_on = one([0, 7])                                 # six missing
    assert (goes_on.stay, goes_on.returned, goes_on.short_breaks) == (2, False, 1)


def test_song_that_comes_back_after_a_long_absence_keeps_only_its_first_stay():
    life = one([0, 1, 12, 13])
    assert (life.stay, life.exit_seen, life.returned, life.last_snapshot) == (2, True, True, 1)


def test_censoring_depends_on_how_long_the_song_has_been_absent_at_the_end_of_the_file():
    assert one([17, 18, 19]).exit_seen is False or not one([17, 18, 19]).exit_seen         # still on the chart on the last day
    assert not one([13, 14]).exit_seen                                                      # absent for 5: it may come back
    assert not one([12, 13]).exit_seen                                                      # absent for 6
    assert one([12]).exit_seen                                                              # absent for 7 (snapshots 13 ... 19): seen to end
    assert one([10, 11]).exit_seen


def test_rule_with_gap_one_is_the_first_unbroken_run():
    life = one([0, 1, 3, 4], gap=1)
    assert (life.stay, life.first_run_days, life.returned) == (2, 2, True)


def test_first_run_days_does_not_depend_on_the_gap():
    for gap in (1, 3, 7, 30):
        assert one([0, 1, 2, 6, 7], gap=gap).first_run_days == 3


def test_many_songs_come_back_in_the_order_asked_for():
    chart = chart_of(b=[0, 1, 2], a=[5, 6], c=[2])
    lives = build_lives(chart, ["c", "a", "b"], N)
    assert list(lives["track_id"]) == ["c", "a", "b"] and list(lives.columns) == LIVES_COLUMNS
    assert list(lives["stay"]) == [1, 2, 3]


def test_songs_that_are_not_asked_for_are_ignored_and_missing_songs_are_refused():
    chart = chart_of(a=[0, 1], b=[3])
    assert list(build_lives(chart, ["a"], N)["track_id"]) == ["a"]
    with pytest.raises(ValueError, match="no chart rows"):
        build_lives(chart, ["a", "zzz"], N)


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        build_lives(chart_of(a=[0]), ["a"], N, gap=0)
    with pytest.raises(ValueError, match="two chart rows"):
        build_lives(chart_of(a=[0, 0, 1]), ["a"], N)
    with pytest.raises(ValueError, match="beyond the end"):
        build_lives(chart_of(a=[0, 25]), ["a"], N)


def test_person_days_has_one_row_per_chart_day_of_the_stay():
    chart = chart_of(a=[0, 1, 2, 3], b=[2, 3, 14, 15], c=[18, 19])
    lives = build_lives(chart, ["a", "b", "c"], N)
    days = person_days(chart, lives)
    assert len(days) == lives["stay"].sum() == 4 + 2 + 2
    assert list(days[days.track_id == "a"]["age"]) == [1, 2, 3, 4]
    assert list(days[days.track_id == "b"]["s"]) == [2, 3]                # the days after the long absence are not part of the stay
    assert list(days[days.track_id == "b"]["event"]) == [0, 1]            # finished: the last day is the event
    assert list(days[days.track_id == "c"]["event"]) == [0, 0]            # censored: no event
    assert days["event"].sum() == lives["exit_seen"].sum()


def test_person_days_carries_the_place_of_each_day():
    chart = chart_of(a=[0, 1, 2])
    days = person_days(chart, build_lives(chart, ["a"], N))
    assert list(days["position"]) == [10, 11, 12]


def test_drop_sizes_counts_the_songs_of_an_artist_that_arrive_together():
    table = pd.DataFrame({"track_id": list("abcdef"), "primary_artist": ["X", "X", "X", "Y", "X", "Y"], "entry_snapshot": [5, 5, 5, 5, 9, 12]})
    assert list(drop_sizes(table)) == [3, 3, 3, 1, 1, 1]


def test_snapshot_numbers_skip_the_days_without_a_snapshot():
    dim_date = pd.DataFrame({"date": pd.to_datetime(["2024-05-18", "2024-05-19", "2024-05-20", "2024-05-21"]),
                             "has_snapshot": [True, False, True, True], "snapshot_index": [7, 8, 8, 9]})
    numbers = snapshot_numbers(dim_date)
    assert list(numbers["s"]) == [0, 1, 2]
    fact = pd.DataFrame({"date": pd.to_datetime(["2024-05-20", "2024-05-18"]), "track_id": ["a", "a"], "position": [4, 3]})
    chart = chart_with_snapshots(fact, dim_date)
    assert list(chart["s"]) == [0, 1] and list(chart["position"]) == [3, 4]
    with pytest.raises(ValueError, match="without a snapshot"):
        chart_with_snapshots(pd.DataFrame({"date": pd.to_datetime(["2024-05-19"]), "track_id": ["a"], "position": [1]}), dim_date)


@pytest.mark.realdata
def test_real_file_has_the_counts_the_notebook_reports():
    from src.config import default_paths
    from src.ml.dataset import build_modeling_table

    processed = default_paths().processed_dir
    needed = ["dim_track", "fact_chart_daily", "bridge_track_artist", "dim_date"]
    if not all((processed / f"{name}.parquet").exists() for name in needed):
        pytest.skip("data/processed is not built")
    tables = {name: pd.read_parquet(processed / f"{name}.parquet") for name in needed}
    table = build_modeling_table(tables["dim_track"], tables["fact_chart_daily"], tables["bridge_track_artist"], tables["dim_date"])
    chart = chart_with_snapshots(tables["fact_chart_daily"], tables["dim_date"])
    lives = build_lives(chart, table["track_id"], int(tables["dim_date"]["has_snapshot"].sum()))
    days = person_days(chart, lives)
    assert len(lives) == 894 and int(lives["exit_seen"].sum()) == 835 and int((~lives["exit_seen"]).sum()) == 59
    assert len(days) == 19_404 and int(days["event"].sum()) == 835
    assert (lives["entry_snapshot"].to_numpy() == table["entry_snapshot"].to_numpy()).all()
