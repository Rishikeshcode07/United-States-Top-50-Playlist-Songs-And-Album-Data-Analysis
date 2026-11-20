"""
tests/test_dash_filters.py  -  the sidebar choices and how they cut the table.

WHAT THIS FILE IS
    Tests of src/dash/filters.py on the synthetic chart (a small chart with known features) and, where it adds something, on the
    real tables. Every filter is compared with a plain, independent re-statement of the rule.

WHY WE NEED IT
    The same filter state drives sixteen pages. If "places 1 to 10" were off by one, or if a featured artist were missed by the artist
    filter, every page would be wrong in the same quiet way.

WHAT HAPPENS WHEN IT RUNS
    Fast: no files are written.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Check the one rule that all pages share.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.dash import filters as F
from src.dash.load import chart_bounds


@pytest.fixture(scope="module")
def play(dash_synthetic):
    return dash_synthetic.playlist


@pytest.fixture(scope="module")
def bounds(play):
    return chart_bounds(play)


def state(bounds, **changes):
    base = F.default_state(bounds)
    values = {"start": base.start, "end": base.end, "rank_low": 1, "rank_high": 50, "album_types": (), "explicit": "all",
              "collab": "all", "artists": (), "track_ids": ()}
    values.update(changes)
    return F.FilterState.create(**values)


# ---------------------------------------------------------------------------------------------- the state object
def test_create_swaps_dates_and_places_that_are_the_wrong_way_round(bounds):
    s = F.FilterState.create("2024-02-01", "2024-01-10", rank_low=30, rank_high=5)
    assert s.start == date(2024, 1, 10) and s.end == date(2024, 2, 1)
    assert (s.rank_low, s.rank_high) == (5, 30)


def test_create_keeps_places_inside_1_to_50():
    s = F.FilterState.create("2024-01-01", "2024-01-02", rank_low=-5, rank_high=900)
    assert (s.rank_low, s.rank_high) == (1, 50)


def test_create_ignores_unknown_words_and_repeats():
    s = F.FilterState.create("2024-01-01", "2024-01-02", album_types=["album", "album", "vinyl"], explicit="maybe", collab="yes",
                             artists=["B", "A", "A"], track_ids=[3, 1, 3, "2"])
    assert s.album_types == ("album",)
    assert s.explicit == "all" and s.collab == "all"
    assert s.artists == ("A", "B") and s.track_ids == (1, 2, 3)


def test_choosing_every_album_type_means_no_restriction():
    s = F.FilterState.create("2024-01-01", "2024-01-02", album_types=["single", "album", "compilation"])
    assert s.album_types == ()


def test_the_key_round_trips(bounds):
    s = state(bounds, rank_high=20, album_types=("single",), explicit="explicit", artists=("A",), track_ids=(5, 4))
    assert F.FilterState.from_key(s.key()) == s
    assert hash(s) == hash(F.FilterState.from_key(s.key()))


def test_to_date_accepts_every_kind_of_date():
    assert F.to_date("2024-03-05") == date(2024, 3, 5)
    assert F.to_date(pd.Timestamp("2024-03-05 13:00")) == date(2024, 3, 5)
    assert F.to_date(date(2024, 3, 5)) == date(2024, 3, 5)


def test_with_dates_keeps_the_other_choices(bounds):
    s = state(bounds, explicit="clean", rank_high=10)
    t = s.with_dates("2024-01-03", "2024-01-05")
    assert (t.explicit, t.rank_high) == ("clean", 10)
    assert (t.start, t.end) == (date(2024, 1, 3), date(2024, 1, 5))


# ----------------------------------------------------------------------------------------------- the filters
def test_the_default_state_keeps_every_row(play, bounds, dash_synthetic):
    s = F.default_state(bounds)
    assert s.is_whole_chart(bounds)
    assert len(F.apply_filters(play, s, dash_synthetic.bridge)) == len(play)


def test_dates_are_inclusive(play, bounds):
    days = sorted(play["date"].unique())
    s = state(bounds, start=days[2], end=days[4])
    got = F.apply_filters(play, s)
    assert set(got["date"]) == set(days[2:5])
    assert len(got) == int(play["date"].isin(days[2:5]).sum())


def test_place_range_is_inclusive(play, bounds):
    got = F.apply_filters(play, state(bounds, rank_low=3, rank_high=7))
    assert got["position"].min() == 3 and got["position"].max() == 7
    assert len(got) == int(play["position"].between(3, 7).sum())


def test_album_type_filter(play, bounds):
    kinds = play["album_type"].unique().tolist()
    for kind in kinds:
        got = F.apply_filters(play, state(bounds, album_types=(kind,)))
        assert set(got["album_type"]) == {kind}
        assert len(got) == int((play["album_type"] == kind).sum())


def test_explicit_filter(play, bounds):
    only = F.apply_filters(play, state(bounds, explicit="explicit"))
    clean = F.apply_filters(play, state(bounds, explicit="clean"))
    assert only["is_explicit"].all() and not clean["is_explicit"].any()
    assert len(only) + len(clean) == len(play)


def test_collaboration_filter(play, bounds):
    collab = F.apply_filters(play, state(bounds, collab="collab"))
    solo = F.apply_filters(play, state(bounds, collab="solo"))
    assert collab["is_collab"].all() and not solo["is_collab"].any()
    assert len(collab) + len(solo) == len(play)


def test_artist_filter_counts_a_feature_credit(play, bounds, dash_synthetic):
    bridge = dash_synthetic.bridge
    featured = bridge.loc[~bridge["is_primary"], "artist_name"]
    if featured.empty:
        pytest.skip("the synthetic chart has no featured artist")
    name = featured.iloc[0]
    got = F.apply_filters(play, state(bounds, artists=(name,)), bridge)
    expected_ids = set(bridge.loc[bridge["artist_name"] == name, "track_id"])
    assert set(got["track_id"]) == expected_ids & set(play["track_id"])
    assert set(got["primary_artist"]) - {name}                 # a feature credit brings in songs whose main artist is somebody else
    primary_only = F.apply_filters(play, state(bounds, artists=(name,)))             # no bridge: primary artist only
    assert set(primary_only["primary_artist"]) <= {name}
    assert len(got) >= len(primary_only)


def test_song_filter(play, bounds):
    ids = tuple(play["track_id"].unique()[:3])
    got = F.apply_filters(play, state(bounds, track_ids=ids))
    assert set(got["track_id"]) == set(ids)


def test_filters_combine_with_and(play, bounds):
    s = state(bounds, rank_high=25, explicit="explicit", collab="solo")
    got = F.apply_filters(play, s)
    expected = play[(play["position"] <= 25) & play["is_explicit"] & ~play["is_collab"]]
    assert got.index.equals(expected.index)


def test_contradictory_choices_give_an_empty_table_with_the_same_columns(play, bounds):
    got = F.apply_filters(play, state(bounds, rank_low=40, rank_high=45, track_ids=(10 ** 9,)))
    assert got.empty and list(got.columns) == list(play.columns)


def test_keep_dates_ignores_every_other_choice(play, bounds, dash_synthetic):
    days = sorted(play["date"].unique())
    s = state(bounds, start=days[1], end=days[3], rank_high=5, explicit="explicit")
    got = F.keep_dates(dash_synthetic.daily_summary, s)
    assert set(got["date"]) == set(days[1:4])


def test_filtering_does_not_change_the_original(play, bounds):
    before = play.copy()
    F.apply_filters(play, state(bounds, rank_high=3))
    assert play.equals(before)


# ---------------------------------------------------------------------------------------------- the text and lists
def test_describe_lists_only_what_is_chosen(bounds):
    assert F.describe(F.default_state(bounds), bounds) == ["whole period"]
    days = [pd.Timestamp(bounds.first_date) + pd.Timedelta(days=1), pd.Timestamp(bounds.last_date)]
    s = state(bounds, start=days[0], end=days[1], rank_high=10, album_types=("single",), explicit="explicit", collab="collab",
              artists=("A", "B"), track_ids=(1,))
    phrases = F.describe(s, bounds)
    assert "to" in phrases[0] and phrases[0] != "whole period"
    assert "places 1 to 10" in phrases
    assert "release type: single" in phrases
    assert "explicit songs only" in phrases and "collaborations only" in phrases
    assert "2 artists" in phrases and "1 song" in phrases


def test_artist_options_put_the_busiest_artist_first(dash_synthetic):
    names = F.artist_options(dash_synthetic.bridge, dash_synthetic.dim_artist)
    slots = dash_synthetic.dim_artist.set_index("artist_name")["chart_slot_days"]
    assert slots[names[0]] == slots.max()
    assert sorted(names) == sorted(dash_synthetic.dim_artist["artist_name"])
    assert F.artist_options(dash_synthetic.bridge) == sorted(dash_synthetic.bridge["artist_name"].unique())


def test_song_options_map_ids_to_labels_longest_first(dash_synthetic):
    options = F.song_options(dash_synthetic.dim_track)
    first_id = next(iter(options))
    assert dash_synthetic.dim_track.set_index("track_id").loc[first_id, "days_on_chart"] == dash_synthetic.dim_track["days_on_chart"].max()
    assert len(options) == len(dash_synthetic.dim_track)


def test_on_the_real_file_the_default_keeps_all_27750_rows(dash_real):
    bounds = chart_bounds(dash_real.playlist)
    got = F.apply_filters(dash_real.playlist, F.default_state(bounds), dash_real.bridge)
    assert len(got) == 27_750
    top10 = F.apply_filters(dash_real.playlist, F.FilterState.create(bounds.first_date, bounds.last_date, 1, 10))
    assert len(top10) == 555 * 10
