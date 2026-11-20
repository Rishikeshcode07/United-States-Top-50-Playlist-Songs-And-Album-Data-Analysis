"""
tests/test_ml_dataset.py  -  checks of the modeling table (src/ml/dataset.py): right numbers, and no peeking at the future.

WHAT THIS FILE IS
    A tiny hand-made chart (six songs, five artists, ten snapshot days) whose modeling table was worked out on paper, row by row.
    On top of that, "peeking" tests: the chart is damaged AFTER a song's first day (other songs' positions scrambled, rows deleted)
    and the inputs of that song must not move. The same tests run on the real file (skipped when it is not there), together with an
    independent re-calculation in plain Python loops of a sample of songs.

WHY WE NEED IT
    A model that is allowed to see the future gets a great score and is worth nothing. The only way to be sure that no input leaks the
    outcome is to change the future and watch the inputs: if they stay put, they cannot depend on it. The hand-made chart also
    contains the awkward cases: a song that climbs into the Top 10 only on its fifth day (outside the 4-day window, so it must NOT
    count), a collaboration whose credited artist with the longer record is not the primary one, a song that arrives too late
    to have been followed for the whole window (not eligible, rather than a false "no"), and a missing popularity value.

WHAT HAPPENS WHEN IT RUNS
    The toy tests build the table with horizon = 4 snapshots and lookback = 3 snapshots. The real-data tests use the project settings
    (30 and 90) and run the whole pipeline once on the real CSV.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Every input of every model can be defended with the sentence "we knew this on the day the song arrived".
"""
import numpy as np
import pandas as pd
import pytest

from src.ml.dataset import (ARTIST_FEATURES, DEBUT_FEATURES, EARLY_FEATURES, HORIZON, LOOKBACK, NEVER_CHARTED_RANK, OUTCOME_COLUMNS,
                            RELEASE_FEATURES, STAY_CLASSES, TIMING_FEATURES, build_modeling_table, stay_class)

START = pd.Timestamp("2024-01-01")                      # a Monday
NAN = float("nan")

# track -> (credited artists in order, release facts, rows {snapshot: (position, popularity)})
CHART = {
    1: (["A"], dict(duration_min=3.0, total_tracks=12, release_format="Album", is_explicit=True, is_collab=False),
        {0: (5, 90), 1: (5, 92), 2: (6, 93), 3: (7, 94), 4: (8, 95), 5: (9, 96), 6: (10, 97), 7: (11, 98), 8: (12, 99), 9: (13, 99)}),
    2: (["A"], dict(duration_min=4.0, total_tracks=1, release_format="Single", is_explicit=False, is_collab=False),
        {2: (8, 80), 3: (6, 82), 4: (4, 85), 5: (3, 86)}),
    3: (["B"], dict(duration_min=2.5, total_tracks=20, release_format="Album", is_explicit=True, is_collab=False),
        {3: (30, 60), 4: (25, 62), 5: (20, 64), 6: (12, 66), 7: (9, 68), 8: (15, 70)}),                   # reaches the Top 10 on its FIFTH day
    4: (["C", "A"], dict(duration_min=3.5, total_tracks=1, release_format="Single", is_explicit=False, is_collab=True),
        {4: (15, 70), 5: (18, 75), 6: (9, NAN), 7: (7, 78), 8: (6, 79), 9: (20, 80)}),                    # a missing popularity on its third day
    5: (["B"], dict(duration_min=3.0, total_tracks=10, release_format="Album", is_explicit=False, is_collab=False),
        {8: (40, 55), 9: (38, 56)}),                                                                       # arrives too late to be followed for 4 snapshots
    6: (["D"], dict(duration_min=3.2, total_tracks=15, release_format="Album", is_explicit=True, is_collab=False),
        {3: (22, 65), 4: (21, 66), 5: (20, 67), 6: (10, 68), 7: (18, 69), 8: (25, 70), 9: (30, 71)}),     # reaches the Top 10 on its FOURTH day
}
SMALL = dict(horizon=4, lookback=3)


def _tables(chart=CHART, n_snapshots=10):
    dates = [START + pd.Timedelta(days=s) for s in range(n_snapshots)]
    dim_date = pd.DataFrame({"date": dates, "has_snapshot": True, "snapshot_index": np.arange(1, n_snapshots + 1)})
    facts, credits, songs = [], [], []
    for track, (artists, release, rows) in chart.items():
        for s, (position, popularity) in rows.items():
            facts.append({"date": dates[s], "track_id": track, "position": position, "popularity_clean": popularity,
                          "in_top10": position <= 10})
        for order, artist in enumerate(artists):
            credits.append({"track_id": track, "artist_name": artist, "artist_order": order})
        first = min(rows)
        songs.append({"track_id": track, "track_label": f"Song {track}", "primary_artist": artists[0], "first_seen": dates[first], **release})
    return (pd.DataFrame(songs), pd.DataFrame(facts), pd.DataFrame(credits), dim_date)


@pytest.fixture(scope="module")
def toy():
    return build_modeling_table(*_tables(), **SMALL).set_index("track_id")


# ================================================================================ which songs, in which order
def test_only_songs_that_arrived_inside_the_window_are_kept_in_order_of_arrival(toy):
    assert toy.index.tolist() == [2, 3, 6, 4, 5]                 # song 1 was already on the chart on the first day
    assert toy["entry_snapshot"].tolist() == [2, 3, 3, 4, 8]
    assert toy["follow_up_days"].tolist() == [7, 6, 6, 5, 1]


def test_columns_are_the_documented_groups_in_the_documented_order():
    table = build_modeling_table(*_tables(), **SMALL)
    expected = (["track_id", "track_label", "primary_artist", "record_artist", "entry_date", "entry_snapshot"] + RELEASE_FEATURES
                + TIMING_FEATURES + ARTIST_FEATURES + DEBUT_FEATURES + EARLY_FEATURES + OUTCOME_COLUMNS)
    assert table.columns.tolist() == expected


# ================================================================================ song 2: a Top-10 debut by an artist with a record
def test_song_2_release_timing_and_debut(toy):
    row = toy.loc[2]
    assert (row["duration_min"], row["total_tracks"], row["is_single"], row["is_explicit"], row["is_collab"]) == (4.0, 1, 1, 0, 0)
    assert row["weekday_wed"] == 1 and row[[c for c in TIMING_FEATURES if c.startswith("weekday_")]].sum() == 1   # 3 January 2024 is a Wednesday
    assert row["history_days"] == 2                               # two snapshots of history, fewer than the lookback of 3
    assert (row["entry_rank"], row["entry_pop"]) == (8, 80)


def test_song_2_artist_record_counts_only_the_two_days_before(toy):
    row = toy.loc[2]
    assert row["record_artist"] == "A"
    assert row["artist_prior_slot_days"] == 2                     # song 1 on snapshots 0 and 1
    assert row["artist_slot_days_recent"] == 2
    assert row["artist_prior_songs"] == 1
    assert row["artist_prior_top10_days"] == 2
    assert row["artist_prior_best_rank"] == 5
    assert row["artist_prior_mean_pop"] == pytest.approx(91.0)    # (90 + 92) / 2
    assert row["artist_on_chart_prev_day"] == 1


def test_song_2_outcomes(toy):
    row = toy.loc[2]
    assert row["y_debut_top10"] == 1
    assert not row["ok_climb"] and np.isnan(row["y_climb_top10"])          # it debuted inside the Top 10: nothing to climb
    assert row["ok_stay"] and row["chart_days_30"] == 4 and row["stay_class"] == STAY_CLASSES[1]
    assert row["pop_first_week"] == pytest.approx(83.25) and row["ok_pop"]
    assert not row["ok_early"]                                              # already in the Top 10 during its first three days
    assert (row["pos_day2"], row["move_first_3d"], row["best_rank_first_3d"], row["pop_day3"], row["pop_move_first_3d"]) == (4, 4, 4, 85, 5)


# ================================================================================ song 3: climbs on day five, outside the window
def test_song_3_has_no_record_and_a_climb_on_day_five_does_not_count(toy):
    row = toy.loc[3]
    assert row["record_artist"] == "B"
    assert row["artist_prior_slot_days"] == 0 and row["artist_slot_days_recent"] == 0 and row["artist_prior_songs"] == 0
    assert row["artist_prior_top10_days"] == 0 and row["artist_on_chart_prev_day"] == 0
    assert row["artist_prior_best_rank"] == NEVER_CHARTED_RANK
    assert np.isnan(row["artist_prior_mean_pop"])
    assert row["weekday_thu"] == 1 and row["history_days"] == 3
    assert row["ok_climb"] and row["y_climb_top10"] == 0                    # position 9 comes on its fifth day, the window has four days
    assert row["pop_first_week"] == pytest.approx(65.0)


def test_song_3_early_warning_population_and_outcomes(toy):
    row = toy.loc[3]
    assert row["ok_early"]
    assert (row["pos_day2"], row["move_first_3d"], row["best_rank_first_3d"], row["pop_day3"], row["pop_move_first_3d"]) == (20, 10, 20, 64, 4)
    assert row["y_climb_after3"] == 0 and row["y_sustained_after3"] == 0


# ================================================================================ song 6: climbs on day four, inside the window
def test_song_6_climbs_inside_the_window(toy):
    row = toy.loc[6]
    assert row["ok_climb"] and row["y_climb_top10"] == 1
    assert row["ok_early"] and row["y_climb_after3"] == 1
    assert (row["pos_day2"], row["move_first_3d"], row["best_rank_first_3d"], row["pop_day3"], row["pop_move_first_3d"]) == (20, 2, 20, 67, 2)
    assert row["pop_first_week"] == pytest.approx(68.0)


# ================================================================================ song 4: a collaboration and a missing popularity
def test_collaboration_takes_the_record_of_the_credited_artist_with_the_longer_history(toy):
    row = toy.loc[4]
    assert row["primary_artist"] == "C" and row["record_artist"] == "A"
    assert row["is_collab"] == 1
    assert row["artist_prior_slot_days"] == 6                               # song 1 on snapshots 0-3 and song 2 on 2-3
    assert row["artist_slot_days_recent"] == 5                              # lookback of 3: snapshots 1, 2, 3
    assert row["artist_prior_songs"] == 2
    assert row["artist_prior_top10_days"] == 6
    assert row["artist_prior_best_rank"] == 5
    assert row["artist_prior_mean_pop"] == pytest.approx(88.5)              # (90 + 92 + 93 + 94 + 80 + 82) / 6
    assert row["weekday_fri"] == 1 and row["history_days"] == 3


def test_a_missing_popularity_stays_missing_and_is_skipped_in_the_first_week_mean(toy):
    row = toy.loc[4]
    assert row["y_climb_top10"] == 1                                        # position 9 on its third day
    assert row["pop_first_week"] == pytest.approx((70 + 75 + 78 + 79 + 80) / 5)
    assert np.isnan(row["pop_day3"]) and np.isnan(row["pop_move_first_3d"])
    assert not row["ok_early"]


# ================================================================================ song 5: too late to be followed
def test_a_song_that_arrives_too_late_is_not_eligible_instead_of_counted_as_no(toy):
    row = toy.loc[5]
    assert not row["ok_climb"] and np.isnan(row["y_climb_top10"])
    assert not row["ok_stay"]
    assert np.isnan(row["chart_days_30"])
    assert pd.isna(row["stay_class"])
    assert not row["ok_early"] and np.isnan(row["pos_day2"])                # the third day is after the end of the file
    assert row["pop_first_week"] == pytest.approx(55.5) and row["ok_pop"]
    assert row["artist_prior_slot_days"] == 5 and row["artist_slot_days_recent"] == 3
    assert row["artist_prior_top10_days"] == 1 and row["artist_prior_best_rank"] == 9


def test_stay_classes_cut_at_three_and_fourteen_chart_days():
    labels = stay_class([1, 3, 4, 14, 15, 30, np.nan])
    assert labels.tolist() == [STAY_CLASSES[0], STAY_CLASSES[0], STAY_CLASSES[1], STAY_CLASSES[1], STAY_CLASSES[2], STAY_CLASSES[2], None]


# ================================================================================ no peeking at the future (toy chart)
def _damage_future(tables, track, entry, rng, own_rows_from=None):
    """Scramble the chart from the snapshot `entry` on for every OTHER song; with own_rows_from, also the rows of `track` from that snapshot."""
    songs, facts, credits, dim_date = (frame.copy() for frame in tables)
    days = dim_date[dim_date["has_snapshot"].astype(bool)]                         # only days with a snapshot count (the real file has four gaps)
    first = int(days["snapshot_index"].min())
    snapshot = facts["date"].map(dict(zip(days["date"], days["snapshot_index"].astype(int) - first)))
    mine = facts["track_id"] == track
    hit = (snapshot >= entry) & ~mine
    if own_rows_from is not None:
        hit = hit | (mine & (snapshot >= own_rows_from))
    facts.loc[hit, "position"] = rng.integers(1, 51, hit.sum())
    facts.loc[hit, "popularity_clean"] = rng.integers(20, 100, hit.sum()).astype(float)
    facts["in_top10"] = facts["position"] <= 10
    return songs, facts, credits, dim_date


def test_changing_the_future_of_other_songs_never_moves_the_inputs_of_a_song():
    tables = _tables()
    base = build_modeling_table(*tables, **SMALL).set_index("track_id")
    inputs = RELEASE_FEATURES + TIMING_FEATURES + ARTIST_FEATURES + DEBUT_FEATURES + EARLY_FEATURES
    rng = np.random.default_rng(5)
    for track in base.index:
        damaged = build_modeling_table(*_damage_future(tables, track, int(base.loc[track, "entry_snapshot"]), rng), **SMALL).set_index("track_id")
        pd.testing.assert_series_equal(damaged.loc[track, inputs].astype(float), base.loc[track, inputs].astype(float), check_names=False)


def test_changing_a_songs_own_life_after_day_three_moves_no_input_but_may_move_the_outcome():
    tables = _tables()
    base = build_modeling_table(*tables, **SMALL).set_index("track_id")
    inputs = RELEASE_FEATURES + TIMING_FEATURES + ARTIST_FEATURES + DEBUT_FEATURES + EARLY_FEATURES
    rng = np.random.default_rng(6)
    moved = False
    for track in (3, 6):
        entry = int(base.loc[track, "entry_snapshot"])
        damaged = build_modeling_table(*_damage_future(tables, track, 99, rng, own_rows_from=entry + 3), **SMALL).set_index("track_id")
        pd.testing.assert_series_equal(damaged.loc[track, inputs].astype(float), base.loc[track, inputs].astype(float), check_names=False)
        moved |= not damaged.loc[track, ["y_climb_top10", "chart_days_30", "pop_first_week"]].equals(base.loc[track, ["y_climb_top10", "chart_days_30", "pop_first_week"]])
    assert moved, "the test has no teeth: damaging the life of the song did not change any outcome"


def test_a_missing_column_is_refused_with_its_name():
    songs, facts, credits, dim_date = _tables()
    with pytest.raises(ValueError, match="popularity_clean"):
        build_modeling_table(songs, facts.drop(columns=["popularity_clean"]), credits, dim_date)


# ================================================================================ the real file
@pytest.fixture(scope="module")
def real_table(real_result):
    t = real_result.tables
    return build_modeling_table(t["dim_track"], t["fact_chart_daily"], t["bridge_track_artist"], t["dim_date"])


def test_real_file_has_the_documented_sizes(real_table):
    assert len(real_table) == 894 and real_table["primary_artist"].nunique() == 229
    assert int(real_table["y_debut_top10"].sum()) == 222
    assert int(real_table["ok_climb"].sum()) == 632 and int(real_table["ok_stay"].sum()) == 832 and int(real_table["ok_pop"].sum()) == 878
    assert real_table["stay_class"].value_counts().to_dict() == {STAY_CLASSES[0]: 289, STAY_CLASSES[1]: 289, STAY_CLASSES[2]: 254}


def test_real_file_the_debut_columns_agree_with_dim_track(real_table, real_result):
    songs = real_result.tables["dim_track"].set_index("track_id")
    mine = real_table.set_index("track_id")
    assert (mine["entry_rank"] == songs.loc[mine.index, "entry_rank"]).all()
    assert (mine["entry_date"] == songs.loc[mine.index, "first_seen"]).all()
    assert not songs.loc[mine.index, "on_chart_at_window_start"].any()
    assert songs["on_chart_at_window_start"].sum() == 50 and len(songs) - len(mine) == 50


def test_real_file_no_input_depends_on_the_future(real_table, real_result):
    t = real_result.tables
    tables = (t["dim_track"], t["fact_chart_daily"], t["bridge_track_artist"], t["dim_date"])
    base = real_table.set_index("track_id")
    inputs = RELEASE_FEATURES + TIMING_FEATURES + ARTIST_FEATURES + DEBUT_FEATURES + EARLY_FEATURES
    rng = np.random.default_rng(11)
    late = base.index[base["entry_snapshot"] > 460].to_numpy()                       # after all four gaps in the calendar: the snapshot index and the calendar index differ most
    chosen = list(rng.choice(base.index.to_numpy(), 5, replace=False)) + list(rng.choice(late, 3, replace=False))
    moved_any_outcome = False
    for track in chosen:
        entry = int(base.loc[track, "entry_snapshot"])
        damaged = _damage_future(tables, int(track), entry, rng, own_rows_from=entry + 3)
        again = build_modeling_table(*damaged).set_index("track_id")
        pd.testing.assert_series_equal(again.loc[track, inputs].astype(float), base.loc[track, inputs].astype(float), check_names=False)
        outcomes = ["y_climb_top10", "chart_days_30", "pop_first_week", "y_climb_after3", "y_sustained_after3"]
        moved_any_outcome |= not again.loc[track, outcomes].equals(base.loc[track, outcomes])
    assert moved_any_outcome, "the test has no teeth: scrambling the future changed no outcome"


def _oracle(track, facts, credits, last_snapshot, horizon=HORIZON, lookback=LOOKBACK):
    """The inputs and outcomes of one song, worked out with plain loops (no groupby, no cumulative sums)."""
    mine = sorted((s, p, q, t10) for s, tr, p, q, t10 in facts if tr == track)
    entry, entry_rank, entry_pop = mine[0][0], mine[0][1], mine[0][2]
    artists = [name for name, order, tr in sorted(credits, key=lambda c: c[1]) if tr == track]
    best_record = None
    for artist in artists:
        songs_of = {tr for name, order, tr in credits if name == artist}
        before = [(s, p, q, t10) for s, tr, p, q, t10 in facts if tr in songs_of and s < entry]
        if best_record is None or len(before) > len(best_record[1]):
            best_record = (artist, before, songs_of)
    artist, before, songs_of = best_record
    first_seen = {tr: min(s for s, trr, *_ in facts if trr == tr) for tr in songs_of}
    pops = [q for s, p, q, t10 in before if not np.isnan(q)]
    window = [row for row in mine if row[0] - entry < horizon]
    return {
        "record_artist": artist, "artist_prior_slot_days": len(before),
        "artist_slot_days_recent": len([1 for s, *_ in before if s >= entry - lookback]),
        "artist_prior_songs": len([1 for tr in songs_of if first_seen[tr] < entry]),
        "artist_prior_top10_days": len([1 for s, p, q, t10 in before if t10]),
        "artist_prior_best_rank": min([p for s, p, q, t10 in before], default=NEVER_CHARTED_RANK),
        "artist_prior_mean_pop": float(np.mean(pops)) if pops else np.nan,
        "entry_rank": entry_rank, "entry_pop": entry_pop, "chart_days_30": len(window),
        "y_climb_top10": float(any(t10 for s, p, q, t10 in window)),
    }


def test_real_file_sample_of_songs_against_an_independent_calculation(real_table, real_result):
    t = real_result.tables
    calendar = t["dim_date"].loc[t["dim_date"]["has_snapshot"], ["date", "snapshot_index"]]
    to_s = dict(zip(calendar["date"], calendar["snapshot_index"] - calendar["snapshot_index"].min()))
    chart = t["fact_chart_daily"]
    facts = [(to_s[d], tr, p, q, bool(t10)) for d, tr, p, q, t10 in zip(chart["date"], chart["track_id"], chart["position"],
                                                                         chart["popularity_clean"], chart["in_top10"])]
    credits = list(zip(t["bridge_track_artist"]["artist_name"], t["bridge_track_artist"]["artist_order"], t["bridge_track_artist"]["track_id"]))
    last = max(to_s.values())
    table = real_table.set_index("track_id")
    rng = np.random.default_rng(3)
    for track in rng.choice(table.index.to_numpy(), 25, replace=False):
        expected = _oracle(int(track), facts, credits, last)
        row = table.loc[track]
        for key, value in expected.items():
            if key == "chart_days_30" and not row["ok_stay"]:
                continue
            if key == "y_climb_top10" and not row["ok_climb"]:
                continue
            if isinstance(value, str):
                assert row[key] == value, (track, key)
            else:
                assert row[key] == pytest.approx(value, nan_ok=True), (track, key)
