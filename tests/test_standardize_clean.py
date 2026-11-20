"""
tests/test_standardize_clean.py  -  tests for the cleaning rules (artists, song identity, snapshots, popularity repair, runs).

WHAT THIS FILE IS
    Every rule that changes or interprets the data has at least one test here, each on a situation of the hand-made chart
    (tests/synthetic.py) whose right answer a human can check with a pencil:

        song identity       credit change = same song | same title, different artists = two songs | curly quote = same title
        artists             "Earth, Wind & Fire" and "Tyler, The Creator" stay whole | accents are real differences
        snapshots           the stale block of a double date is dropped | a missing day stays a gap
        popularity repair   interpolate inside the run -> song median -> honestly empty
        chart runs          a song that leaves and returns has two runs, and its comeback is a "Re-entry"
        release format      "compilation" is grouped with Album, the original column is kept

WHY WE NEED IT
    These are the decisions that change the numbers in the dashboards. If one of them silently breaks, every KPI after
    it is wrong and nobody sees an error.

WHAT HAPPENS WHEN IT RUNS
    The pipeline runs once on the synthetic chart (a pytest "session fixture") and the tests look at its output.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Each cleaning decision of the project log is backed by an executable example.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import TABLE_COLUMNS
from tests import synthetic as syn

NUMBER_OF_SONGS = 71          # 70 escalator songs (1..70) + the filler (71); songs 40 and 41 share a title but are two songs


@pytest.fixture(scope="module")
def daily(synthetic_tables):
    return synthetic_tables["playlist_daily"]


@pytest.fixture(scope="module")
def merged(daily, plan):
    """The pipeline's daily table joined 1-to-1 with the answer key, so every row knows its true song_id."""
    return daily.merge(syn.expected_daily(plan), on=["date", "position"], suffixes=("", "_exp"), validate="one_to_one")


def rows_of(merged, song_id):
    return merged[merged["song_id"] == song_id].sort_values("snapshot_index")


def row_at(merged, song_id, s):
    """The row of a song at snapshot number s (0-based, as in tests/synthetic.py)."""
    found = merged[(merged["song_id"] == song_id) & (merged["snapshot_index"] == s + 1)]
    assert len(found) == 1, f"song {song_id} should have exactly one row at snapshot {s}"
    return found.iloc[0]


# ================================================================================ song identity
def test_every_song_gets_exactly_one_track_id_and_every_track_id_is_one_song(merged):
    assert merged["song_id"].nunique() == merged["track_id"].nunique() == NUMBER_OF_SONGS
    assert merged.groupby("song_id")["track_id"].nunique().eq(1).all()
    assert merged.groupby("track_id")["song_id"].nunique().eq(1).all()


def test_a_changed_credit_does_not_split_a_song(merged):
    rows = rows_of(merged, 12)                      # credited "Artist 12 & Artist 99" for 3 days, then "Artist 12"
    assert len(rows) == 6 and rows["track_id"].nunique() == 1
    assert set(rows["artist_credit"]) == {"Artist 12 & Artist 99"}          # the fullest credit wins everywhere
    assert set(rows["artist_count"]) == {2}


def test_two_different_songs_with_one_title_stay_two_songs(merged):
    first, second = rows_of(merged, 40), rows_of(merged, 41)
    assert set(first["song_clean"]) == set(second["song_clean"]) == {"Angel"}
    assert first["track_id"].iloc[0] != second["track_id"].iloc[0]
    assert set(first["track_label"]) == {"Angel by Artist 40"}
    assert set(second["track_label"]) == {"Angel by Artist 41"}


def test_curly_and_straight_apostrophes_are_one_song_with_one_spelling(merged):
    rows = rows_of(merged, 28)
    assert len(rows) == 11 and rows["track_id"].nunique() == 1
    assert set(rows["song_clean"]) == {syn.final_title(28)}                  # the curly spelling is the most frequent one (6 of 11)


def test_a_track_label_identifies_one_song(daily):
    assert daily.groupby("track_label")["track_id"].nunique().max() == 1
    assert daily.groupby("track_id")["track_label"].nunique().max() == 1


def test_track_ids_are_numbered_in_order_of_first_appearance(daily):
    first_rows = daily.sort_values(["date", "position"]).drop_duplicates("track_id")
    first_day = first_rows[first_rows["date"] == first_rows["date"].min()]
    assert sorted(first_day["track_id"]) == list(range(1, 51))              # the 50 songs of the first chart


# ================================================================================ artists
def test_band_names_with_a_comma_or_an_ampersand_stay_whole(synthetic_tables, merged):
    bridge = synthetic_tables["bridge_track_artist"]

    def artists_of(song_id):
        track_id = int(rows_of(merged, song_id)["track_id"].iloc[0])
        return bridge[bridge["track_id"] == track_id].sort_values("artist_order")["artist_name"].tolist()

    assert artists_of(25) == ["Earth, Wind & Fire"]
    assert artists_of(26) == ["Earth, Wind & Fire", "Artist 26"]
    assert artists_of(27) == ["Tyler, The Creator"]
    assert artists_of(7) == ["Artist 7", "Artist 107"]


def test_an_accent_is_a_real_difference_between_two_artists(synthetic_tables):
    names = set(synthetic_tables["dim_artist"]["artist_name"])
    assert {"Beyoncé", "Beyonce"} <= names


def test_collaboration_columns(merged):
    solo, duo = rows_of(merged, 8).iloc[0], rows_of(merged, 7).iloc[0]
    assert (solo["artist_count"], solo["is_collab"], solo["primary_artist"]) == (1, False, "Artist 8")
    assert (duo["artist_count"], duo["is_collab"], duo["primary_artist"]) == (2, True, "Artist 7")
    assert duo["artists_all"] == "Artist 7 | Artist 107"


# ================================================================================ snapshots and the calendar
def test_the_stale_block_of_the_double_date_is_dropped(daily, plan):
    assert len(daily) == 550
    on_double_day = daily[daily["date"] == pd.Timestamp(syn.DOUBLE_DAY)]
    assert len(on_double_day) == 50
    assert (on_double_day["popularity"] != 1).all()                          # the stale rows were all written with popularity 1
    truth = {(pd.Timestamp(r["date"]), r["position"]): r["popularity"] for r in plan.true_rows}
    actual = {(d, p): v for d, p, v in zip(daily["date"], daily["position"], daily["popularity"])}
    assert actual == truth                                                   # exactly the rows of the real block, nothing else


def test_a_missing_day_stays_a_gap(daily, synthetic_tables):
    assert pd.Timestamp(syn.MISSING_DAY) not in set(daily["date"])           # no invented row
    assert sorted(daily["snapshot_index"].unique()) == list(range(1, 12))    # 11 snapshots, numbered without a hole

    gaps = daily.drop_duplicates("date").set_index("date")["days_since_prev_snapshot"]
    assert np.isnan(gaps[pd.Timestamp(syn.START)])                          # nothing before the first snapshot
    assert gaps[pd.Timestamp("2024-01-07")] == 2                             # the day after the gap
    assert (gaps.drop([pd.Timestamp(syn.START), pd.Timestamp("2024-01-07")]) == 1).all()

    calendar = synthetic_tables["dim_date"].set_index("date")
    assert len(calendar) == 12                                               # the calendar still has the missing day
    assert not calendar.loc[pd.Timestamp(syn.MISSING_DAY), "has_snapshot"]
    assert pd.isna(calendar.loc[pd.Timestamp(syn.MISSING_DAY), "snapshot_index"])
    assert calendar["has_snapshot"].sum() == 11


def test_every_snapshot_has_fifty_rows_and_all_fifty_ranks(daily):
    assert daily.groupby("date").size().eq(50).all()
    assert daily.groupby("date")["position"].nunique().eq(50).all()


def test_the_calendar_check_without_reference_songs_does_nothing(synthetic_result):
    assert synthetic_result.calendar_reality.empty
    step = synthetic_result.cleaning_log.set_index("step").loc["1.6"]
    assert step["rows_affected"] == 0 and step["rows_before"] == step["rows_after"]


# ================================================================================ popularity repair
def test_popularity_repair_counts(daily):
    assert daily["popularity_fill_method"].value_counts().to_dict() == {
        "original": 542, "song_median": 5, "interpolated_in_run": 2, "unfilled": 1}
    assert int(daily["popularity_was_zero"].sum()) == 8


def test_a_zero_on_a_debut_day_takes_the_next_real_value_of_the_run(merged):
    row = row_at(merged, 51, 1)                                              # song 51 enters on snapshot 1 with a 0
    assert row["popularity"] == 0 and row["popularity_fill_method"] == "interpolated_in_run"
    assert row["popularity_clean"] == row_at(merged, 51, 2)["popularity"] == 63.0


def test_a_zero_in_the_middle_of_a_run_is_interpolated(merged):
    row = row_at(merged, 10, 2)
    before, after = row_at(merged, 10, 1)["popularity"], row_at(merged, 10, 3)["popularity"]
    assert (before, after) == (61, 63)
    assert row["popularity_fill_method"] == "interpolated_in_run" and row["popularity_clean"] == 62.0


def test_a_run_made_only_of_zeros_takes_the_median_of_the_song(merged):
    for s in range(6, 11):                                                   # song 30 returns for 5 days, all with popularity 0
        row = row_at(merged, 30, s)
        assert row["popularity_fill_method"] == "song_median"
        assert row["popularity_clean"] == 81.5                               # median of its real values 80, 81, 82, 83


def test_a_one_day_song_with_a_zero_stays_honestly_empty(merged):
    row = row_at(merged, 70, 10)
    assert row["popularity_fill_method"] == "unfilled" and pd.isna(row["popularity_clean"])


def test_real_values_are_never_changed_and_the_original_column_is_kept(daily):
    real = daily[~daily["popularity_was_zero"]]
    assert (real["popularity_clean"] == real["popularity"]).all()
    assert (real["popularity_fill_method"] == "original").all()
    assert (daily.loc[daily["popularity_was_zero"], "popularity"] == 0).all()   # the zeros are still visible in the original column


# ================================================================================ chart runs and movement
def test_a_song_that_leaves_and_returns_has_two_runs(merged):
    rows = rows_of(merged, 30)
    assert rows["snapshot_index"].tolist() == [1, 2, 3, 4, 7, 8, 9, 10, 11]  # away on snapshots 5 and 6
    assert rows["stint_no"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert rows["stint_length"].tolist() == [4, 4, 4, 4, 5, 5, 5, 5, 5]
    assert rows["days_in_stint"].tolist() == [1, 2, 3, 4, 1, 2, 3, 4, 5]
    assert rows["chart_day_number"].tolist() == list(range(1, 10))           # the song's own day count does not restart
    assert rows["movement"].iloc[4] == "Re-entry"
    assert pd.isna(rows["rank_change"].iloc[4])                              # no "yesterday" after an absence


def test_the_filler_song_is_a_two_day_run_that_climbs(merged):
    rows = rows_of(merged, syn.FILLER_ID)
    assert rows["snapshot_index"].tolist() == [5, 6]
    assert rows["movement"].tolist() == ["New entry", "Up"]
    assert rows["rank_change"].iloc[1] == 2
    assert rows["is_stint_start"].tolist() == [True, False] and rows["is_stint_end"].tolist() == [False, True]


def test_the_first_chart_is_labelled_chart_start_not_new_entry(daily):
    first = daily[daily["snapshot_index"] == 1]
    assert (first["movement"] == "Chart start").all() and len(first) == 50
    assert (daily["movement"] == "Chart start").sum() == 50


def test_the_hand_made_chart_exercises_all_six_movement_labels(plan):
    labels = set(syn.expected_daily(plan)["movement"])
    assert labels == {"Chart start", "New entry", "Re-entry", "Up", "Down", "Same"}


def test_a_song_that_falls_or_stays_is_labelled_down_or_same(merged):
    stays = row_at(merged, 22, 3)                      # song 22 trades places with song 24 on snapshot 4 and keeps yesterday's position
    assert (stays["prev_position"], stays["position"]) == (18, 18)
    assert (stays["movement"], stays["rank_change"]) == ("Same", 0)

    falls = row_at(merged, 18, 6)                      # song 18 trades places with song 23 on snapshot 7 and drops from 8 to 11
    assert (falls["prev_position"], falls["position"]) == (8, 11)
    assert (falls["movement"], falls["rank_change"]) == ("Down", -3)

    climbs = row_at(merged, 23, 6)                     # ... and song 23 climbs 7 places instead of 2
    assert (climbs["prev_position"], climbs["position"]) == (13, 6)
    assert (climbs["movement"], climbs["rank_change"]) == ("Up", 7)


def test_every_row_has_one_label_and_the_counts_match_the_answer_key(merged):
    assert merged["movement"].notna().all()
    assert merged["movement"].value_counts().to_dict() == merged["movement_exp"].value_counts().to_dict()
    later = merged[merged["snapshot_index"] > 1]
    assert (later["movement"] == "New entry").sum() == 21      # 2 on each of 10 snapshots, +1 extra on snapshot 5 (the filler)
    assert (later["movement"] == "Re-entry").sum() == 1        # song 30 coming back


# ================================================================================ release format
def test_a_compilation_is_grouped_with_album_but_stays_visible(merged):
    rows = rows_of(merged, 33)
    assert set(rows["album_type"]) == {"compilation"}                        # the original value is kept
    assert set(rows["release_format"]) == {"Album"}                          # only the analysis column groups it
    assert set(rows["release_size_band"]) == {"16-40"}                       # 20 tracks
    assert set(rows_of(merged, 8)["release_format"]) == {"Single"}
    assert set(rows_of(merged, 8)["release_size_band"]) == {"1 track"}
    assert set(rows_of(merged, 9)["release_size_band"]) == {"8-15"}          # an album with 12 tracks


# ================================================================================ the cleaning log
def test_the_cleaning_log_lists_every_step_with_its_numbers(synthetic_result):
    log = synthetic_result.cleaning_log.set_index("step")
    assert list(log.index) == ["1.6", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "3.x"]
    assert (log.loc["2.1", "rows_before"], log.loc["2.1", "rows_after"], log.loc["2.1", "rows_affected"]) == (600, 550, 50)
    assert log.loc["2.4", "rows_affected"] == 6                              # the 6 rows of song 12
    assert log.loc["2.5", "rows_affected"] == 8                              # the 8 zeros
    assert "2024-01-09" in log.loc["2.1", "rule"]                            # the log names the date it repaired


def test_the_clean_table_keeps_the_promised_columns_in_the_promised_order(synthetic_tables):
    for name, columns in TABLE_COLUMNS.items():
        assert list(synthetic_tables[name].columns) == columns, name
