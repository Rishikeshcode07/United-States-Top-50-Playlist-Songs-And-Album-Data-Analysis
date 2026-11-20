"""
tests/test_quality_gate.py  -  tests for the 25-test quality gate (step 6 of the pipeline).

WHAT THIS FILE IS
    The gate is the pipeline's last line of defence: nothing is saved or loaded unless all 25 of its tests pass.
    Here we test the gate itself. First it must PASS on good data. Then we damage a copy of the tables in 25 different
    ways, one defect per test, and check that the test that is responsible for that defect turns to FAIL.

WHY WE NEED IT
    A guard that has never been seen to say "no" can not be trusted. These tests are the proof that every gate test can
    really fail, and that a failed gate stops the pipeline with a message that names the broken rule.

WHAT HAPPENS WHEN IT RUNS
    Each test takes a private copy of the synthetic tables (fixture `tables`), breaks it in one way, runs the gate and
    inspects the result. The shared tables used by the other tests are never touched.

EXPECTED OUTCOME
    All tests pass: 25 of 25 on good data, and the expected FAIL for each of the damaged copies.

MOTIVE
    Test the tester: a safety net is only as good as the falls it has been shown to catch.
"""
import pandas as pd
import pytest

from src.etl.errors import DataIntegrityError
from src.etl.quality import raise_if_gate_failed, run_quality_gate


def failed(tables):
    gate = run_quality_gate(tables)
    return set(gate.loc[gate["status"] == "FAIL", "id"]), gate


# ================================================================================ good data passes
def test_the_gate_has_25_tests_numbered_g01_to_g25(synthetic_result):
    gate = synthetic_result.gate
    assert list(gate["id"]) == [f"G{i:02d}" for i in range(1, 26)]
    assert list(gate.columns) == ["id", "test", "status", "detail"]


def test_every_gate_test_passes_on_the_synthetic_chart(synthetic_result):
    assert (synthetic_result.gate["status"] == "PASS").all()


def test_gate_names_describe_the_data_in_front_of_them(synthetic_result):
    names = synthetic_result.gate.set_index("id")["test"]
    assert names["G01"] == "11 dates x 50 rows = 550 rows"
    assert names["G14"] == "dim_track: one row per song, 71 songs"
    assert names["G20"] == "dim_date is continuous: 12 days, 11 with a snapshot"


def test_every_gate_test_passes_on_the_real_data(real_result):
    gate = real_result.gate
    assert len(gate) == 25 and (gate["status"] == "PASS").all()
    names = gate.set_index("id")["test"]
    assert names["G01"] == "555 dates x 50 rows = 27,750 rows"
    assert names["G14"] == "dim_track: one row per song, 944 songs"


# ================================================================================ damaged tables fail
def duplicate_a_daily_row(t):
    t["playlist_daily"] = pd.concat([t["playlist_daily"], t["playlist_daily"].iloc[[0]]], ignore_index=True)


def drop_a_daily_row(t):
    t["playlist_daily"] = t["playlist_daily"].iloc[1:].reset_index(drop=True)


def punch_a_hole_in_snapshot_numbers(t):
    d = t["playlist_daily"]
    d["snapshot_index"] = d["snapshot_index"].where(d["snapshot_index"] < 6, d["snapshot_index"] + 100)


def blank_a_key_value(t):
    d = t["playlist_daily"]
    d["track_label"] = d["track_label"].mask(d.index == 0)


def popularity_above_100(t):
    d = t["playlist_daily"]
    d["popularity_clean"] = d["popularity_clean"].mask(d.index == 0, 150.0)


def an_unexplained_empty_popularity(t):
    d = t["playlist_daily"]
    d["popularity_clean"] = d["popularity_clean"].mask(d.index == 0)


def change_a_real_popularity_value(t):
    d = t["playlist_daily"]
    d["popularity_clean"] = d["popularity_clean"] + (d.index == 0)


def break_prev_position(t):
    d = t["playlist_daily"]
    d["prev_position"] = d["prev_position"] + ((d["snapshot_index"] == 2) & (d["position"] == 1))


def break_rank_change(t):
    d = t["playlist_daily"]
    d["rank_change"] = d["rank_change"] + ((d["snapshot_index"] == 2) & (d["position"] == 1))


def invent_a_movement_label(t):
    d = t["playlist_daily"]
    d["movement"] = d["movement"].mask(d.index == 0, "Sideways")


def exits_that_do_not_match_entries(t):
    s = t["daily_summary"]
    s["exits"] = s["exits"] + (s.index == 3)


def lose_a_song(t):
    t["dim_track"] = t["dim_track"].iloc[1:].reset_index(drop=True)


def inflate_days_on_chart(t):
    d = t["dim_track"]
    d["days_on_chart"] = d["days_on_chart"] + (d.index == 0)


def average_rank_worse_than_the_worst(t):
    d = t["dim_track"]
    d["avg_rank"] = d["worst_rank"] + 5


def negative_volatility(t):
    d = t["dim_track"]
    d["rank_volatility_index"] = d["rank_volatility_index"].mask(d.index == 0, -1.0)


def a_song_without_a_primary_artist(t):
    t["bridge_track_artist"]["is_primary"] = False


def double_every_dominance_share(t):
    d = t["dim_artist"]
    d["dominance_index_primary"] = d["dominance_index_primary"] * 2


def lose_a_calendar_day(t):
    t["dim_date"] = t["dim_date"].iloc[1:].reset_index(drop=True)


def a_fact_row_for_a_song_that_does_not_exist(t):
    f = t["fact_chart_daily"]
    f["track_id"] = f["track_id"] + (f.index == 0) * 1000


def a_bridge_row_for_an_unknown_artist(t):
    b = t["bridge_track_artist"]
    b["artist_name"] = b["artist_name"].mask(b.index == 0, "Nobody At All")


def a_summary_for_a_date_that_is_not_a_snapshot(t):
    s = t["daily_summary"]
    s["date"] = s["date"].mask(s.index == 0, pd.Timestamp("2030-01-01"))


def lose_a_fact_row_of_a_long_song(t):
    f = t["fact_chart_daily"]
    t["fact_chart_daily"] = f.drop(index=f.index[f["position"] == 50][0])        # a row of song 50, which has 11 rows


def change_chart_points(t):
    f = t["fact_chart_daily"]
    f["chart_points"] = f["chart_points"] + (f.index == 0)


DEFECTS = [
    pytest.param(duplicate_a_daily_row, {"G01", "G02", "G03"}, id="G01-G03 duplicated daily row"),
    pytest.param(drop_a_daily_row, {"G01", "G04"}, id="G01+G04 a daily row is missing"),
    pytest.param(punch_a_hole_in_snapshot_numbers, {"G05"}, id="G05 snapshot numbers have a hole"),
    pytest.param(blank_a_key_value, {"G06"}, id="G06 empty key value"),
    pytest.param(popularity_above_100, {"G07"}, id="G07 popularity above 100"),
    pytest.param(an_unexplained_empty_popularity, {"G08"}, id="G08 unexplained empty popularity"),
    pytest.param(change_a_real_popularity_value, {"G09"}, id="G09 a real popularity value was changed"),
    pytest.param(break_prev_position, {"G10", "G11"}, id="G10 wrong previous position"),
    pytest.param(break_rank_change, {"G11"}, id="G11 wrong rank change"),
    pytest.param(invent_a_movement_label, {"G12"}, id="G12 unknown movement label"),
    pytest.param(exits_that_do_not_match_entries, {"G13"}, id="G13 exits differ from entries"),
    pytest.param(lose_a_song, {"G14"}, id="G14 a song is missing from dim_track"),
    pytest.param(inflate_days_on_chart, {"G15", "G24"}, id="G15 days_on_chart too high"),
    pytest.param(average_rank_worse_than_the_worst, {"G16"}, id="G16 avg_rank above worst_rank"),
    pytest.param(negative_volatility, {"G17"}, id="G17 negative volatility"),
    pytest.param(a_song_without_a_primary_artist, {"G18"}, id="G18 no primary artist"),
    pytest.param(double_every_dominance_share, {"G19"}, id="G19 dominance does not add up to 100"),
    pytest.param(lose_a_calendar_day, {"G20"}, id="G20 calendar has a hole"),
    pytest.param(a_fact_row_for_a_song_that_does_not_exist, {"G21"}, id="G21 fact row without a song"),
    pytest.param(a_bridge_row_for_an_unknown_artist, {"G22"}, id="G22 bridge row without an artist"),
    pytest.param(a_summary_for_a_date_that_is_not_a_snapshot, {"G23"}, id="G23 summary date outside the calendar"),
    pytest.param(lose_a_fact_row_of_a_long_song, {"G24"}, id="G24 days_on_chart differs from fact rows"),
    pytest.param(change_chart_points, {"G25"}, id="G25 chart points do not add up"),
]


@pytest.mark.parametrize("damage, must_fail", DEFECTS)
def test_each_defect_turns_the_responsible_test_red(tables, damage, must_fail):
    damage(tables)
    failed_ids, gate = failed(tables)
    assert must_fail <= failed_ids, f"expected {sorted(must_fail)} to fail, but only {sorted(failed_ids)} did"
    assert len(gate) == 25                                          # a failing test never stops the other tests from running


def test_a_test_that_crashes_counts_as_a_failure_and_says_so(tables):
    tables["playlist_daily"] = tables["playlist_daily"].drop(columns=["prev_position"])      # the test can not even look at the column
    failed_ids, gate = failed(tables)
    assert "G10" in failed_ids
    assert "test crashed" in gate.set_index("id").loc["G10", "detail"]
    assert len(gate) == 25


def test_a_failed_gate_raises_and_names_the_broken_rule(tables):
    break_rank_change(tables)
    gate = run_quality_gate(tables)
    with pytest.raises(DataIntegrityError) as error:
        raise_if_gate_failed(gate)
    message = str(error.value)
    assert "G11" in message and "rank_change = prev_position - position" in message
    assert "Nothing was saved or loaded" in message


def test_a_clean_gate_does_not_raise(synthetic_result):
    raise_if_gate_failed(synthetic_result.gate)


def test_the_gate_never_modifies_the_tables(tables, synthetic_tables):
    run_quality_gate(tables)
    for name, table in synthetic_tables.items():
        pd.testing.assert_frame_equal(tables[name], table)
