"""
tests/test_dash_survival_live.py  -  the survival curves of the dashboard against the curves of Phase 6B.

WHAT THIS FILE IS
    Tests of src/dash/survival_live.py. The main test is a cross-check: with no filters on, the live curve of every grouping must
    equal the curve that notebook 06b saved in reports/p6b_survival_curves.csv. Other tests use a small made-up table of stays where
    the answer can be written by hand.

WHY WE NEED IT
    The dashboard re-runs the survival analysis for any selection. If it ever disagreed with the notebook for the whole file, a
    reader would not know which to believe. The hand-made cases also check the rules: a song that is still on the chart is
    "censored" and a group that is too small gets no curve.

WHAT HAPPENS WHEN IT RUNS
    The cross-check is skipped when reports/p6b_stays.csv or data/processed/ is missing.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    One answer to "how long do new songs stay", in the notebook and in the dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dash import filters as F
from src.dash import survival_live as SL
from src.dash.load import chart_bounds, read_report
from src.survival.estimators import logrank_k, survival_grid


@pytest.fixture(scope="module")
def stays_real(dash_real):
    raw = read_report("p6b_stays")
    if raw is None:
        pytest.skip("reports/p6b_stays.csv is not built (run notebook 06b)")
    return SL.prepare_stays(raw, dash_real.dim_track)


def toy_stays():
    """12 songs: six that leave early and six that stay, from 4 artists, 2 of them still on the chart on the last day."""
    n = 12
    return pd.DataFrame({
        "track_id": np.arange(1, n + 1), "primary_artist": list("AAABBBCCCDDD"),
        "entry_date": pd.date_range("2024-01-01", periods=n, freq="7D"), "entry_rank": [45, 48, 30, 44, 12, 9, 40, 41, 20, 25, 3, 7],
        "debut_band": ["41 to 50", "41 to 50", "21 to 30", "41 to 50", "11 to 20", "1 to 10", "31 to 40", "41 to 50", "11 to 20",
                       "21 to 30", "1 to 10", "1 to 10"],
        "is_single": [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0], "is_explicit": [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
        "is_collab": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], "drop_size": [3, 3, 3, 1, 1, 1, 6, 6, 6, 2, 2, 2],
        "stay": [2, 3, 1, 4, 20, 30, 2, 2, 5, 3, 40, 12], "exit_seen": [True] * 10 + [False, False],
        "album_type": ["single", "album", "album", "single", "single", "album", "compilation", "album", "single", "album", "single", "album"],
    })


# --------------------------------------------------------------------------------------- the cross-check with 6B
@pytest.mark.parametrize("grouping, split", [("none", "All songs"), ("debut", "Debut place"), ("release", "Release type"),
                                              ("explicit", "Explicit content"), ("collab", "Collaboration"),
                                              ("same_day", "Songs by the artist arriving the same day")])
def test_the_live_curves_equal_the_saved_curves_of_phase_6b(stays_real, grouping, split):
    saved = read_report("p6b_survival_curves")
    saved = saved[saved["split"] == split]
    live = SL.live_survival(stays_real, grouping)["curves"]
    assert set(live["group"]) == set(saved["group"])
    for group in live["group"].unique():
        a = live[live["group"] == group].set_index("age")["survival"]
        b = saved[saved["group"] == group].set_index("age")["survival"]
        ages = b.index[b.index <= a.index.max()]
        assert len(ages) > 50
        assert np.allclose(a.reindex(ages), b.loc[ages], atol=1e-9)


def test_the_headline_numbers_of_the_whole_file(stays_real):
    summary = SL.live_survival(stays_real, "none")["summary"].iloc[0]
    assert summary["songs"] == 894 and summary["artists"] == 229 and summary["exits_seen"] == 835
    assert summary["still_on_chart"] == 59 and summary["median_stay"] == 6
    assert summary["s3"] == pytest.approx(0.634, abs=0.001) and summary["s7"] == pytest.approx(0.432, abs=0.001)
    assert summary["s30"] == pytest.approx(0.192, abs=0.001)


def test_the_log_rank_test_equals_the_estimator_it_wraps(stays_real):
    result = SL.live_survival(stays_real, "release")["logrank"]
    labels = SL.group_labels(stays_real, "release").to_numpy()
    direct = logrank_k(stays_real["stay"].to_numpy(int), stays_real["exit_seen"].to_numpy(bool), labels,
                       clusters=stays_real["primary_artist"].to_numpy())
    assert result["p"] == pytest.approx(direct["p"]) and result["p_artist"] == pytest.approx(direct["p_artist"])
    assert result["n_artists"] == 229 and set(result["groups"]) == {"Single", "Album or compilation"}


# ------------------------------------------------------------------------------------------------ the rules
def test_a_curve_on_hand_made_stays_is_the_kaplan_meier_curve():
    stays = toy_stays()
    curve = SL.live_survival(stays, "none", max_age=45, min_songs=5)["curves"].set_index("age")["survival"]
    expected = survival_grid(stays["stay"].to_numpy(int), stays["exit_seen"].to_numpy(bool), np.arange(0, 46))
    assert np.allclose(curve, expected)
    assert curve[0] == 1.0 and curve.is_monotonic_decreasing
    # day 1: 12 songs are at risk and 1 leaves -> 11/12
    assert curve[1] == pytest.approx(11 / 12)


def test_songs_still_on_the_chart_are_censored_not_counted_as_exits():
    stays = toy_stays()
    result = SL.live_survival(stays, "none", min_songs=5)
    row = result["summary"].iloc[0]
    assert row["exits_seen"] == 10 and row["still_on_chart"] == 2
    # the longest stays (30 and 12 days) belong to songs that were still there: the curve must not fall to zero at day 40
    assert result["curves"].set_index("age").loc[45, "survival"] > 0


def test_a_group_with_too_few_songs_gets_a_note_and_no_curve():
    stays = toy_stays()
    result = SL.live_survival(stays, "release", min_songs=8)
    summary = result["summary"].set_index("group")
    assert "no curve" in summary.loc["Single", "note"] or "no curve" in summary.loc["Album or compilation", "note"]
    assert result["curves"].empty or set(result["curves"]["group"]) < {"Single", "Album or compilation"}
    assert result["logrank"] is None
    for group in set(result["curves"]["group"]):
        assert summary.loc[group, "songs"] >= 8


def test_every_grouping_labels_every_song():
    stays = toy_stays()
    for grouping in SL.GROUPINGS:
        labels = SL.group_labels(stays, grouping)
        assert len(labels) == len(stays) and set(labels) <= set(SL.GROUP_ORDER[grouping])
    assert SL.group_labels(stays, "release").tolist()[:3] == ["Single", "Album or compilation", "Album or compilation"]
    assert SL.group_labels(stays, "same_day").tolist()[:4] == ["2 to 4 songs", "2 to 4 songs", "2 to 4 songs", "1 song (alone)"]
    with pytest.raises(ValueError):
        SL.group_labels(stays, "zodiac sign")


def random_stays(n=90, n_artists=30, seed=0):
    """A larger made-up table (many artists, so that two bootstrap seeds really give different bands)."""
    rng = np.random.default_rng(seed)
    stays = toy_stays().iloc[[0] * n].reset_index(drop=True)
    stays["track_id"] = np.arange(1, n + 1)
    stays["primary_artist"] = [f"artist {i}" for i in rng.integers(0, n_artists, n)]
    stays["stay"] = np.maximum(1, rng.geometric(0.12, n))
    stays["exit_seen"] = rng.random(n) > 0.1
    return stays


def test_the_bands_are_reproducible_and_contain_the_curve():
    stays = random_stays()
    a = SL.live_survival(stays, "none", n_boot=200, seed=5)["curves"]
    b = SL.live_survival(stays, "none", n_boot=200, seed=5)["curves"]
    c = SL.live_survival(stays, "none", n_boot=200, seed=6)["curves"]
    assert a.equals(b) and not a["low"].equals(c["low"])
    assert (a["low"] <= a["survival"] + 1e-9).mean() > 0.9 and (a["high"] >= a["survival"] - 1e-9).mean() > 0.9
    assert (a["low"] <= a["high"]).all() and a["low"].between(0, 1).all() and a["high"].between(0, 1).all()


def test_empty_stays_give_empty_tables():
    empty = SL.live_survival(toy_stays().iloc[:0], "release")
    assert empty["curves"].empty and empty["summary"].empty and empty["logrank"] is None
    assert SL.live_hazard(toy_stays().iloc[:0]).empty


def test_the_hazard_windows_cover_every_chart_day():
    stays = toy_stays()
    table = SL.live_hazard(stays, "none", min_songs=5)
    assert table["exits"].sum() == int(stays["exit_seen"].sum())
    assert table["chart_days"].sum() == int(stays["stay"].sum())
    assert np.allclose(table["hazard"], table["exits"] / table["chart_days"], equal_nan=True)     # a window nobody reached has no hazard
    assert table.iloc[0]["from_day"] == 1 and table.iloc[0]["songs_reaching"] == len(stays)


# ------------------------------------------------------------------------------------------------ the filters
def test_prepare_stays_adds_the_album_type_and_real_dates(dash_real):
    raw = read_report("p6b_stays")
    if raw is None:
        pytest.skip("reports/p6b_stays.csv is not built")
    stays = SL.prepare_stays(raw, dash_real.dim_track)
    assert str(stays["entry_date"].dtype).startswith("datetime64") and stays["album_type"].notna().all()
    assert stays["exit_seen"].dtype == bool and len(stays) == len(raw)


def state_for(stays, **changes):
    values = {"start": stays["entry_date"].min(), "end": stays["entry_date"].max(), "rank_low": 1, "rank_high": 50}
    values.update(changes)
    return F.FilterState.create(**values)


def test_filter_stays_applies_every_choice_to_the_arrival():
    stays = toy_stays()
    got = SL.filter_stays(stays, state_for(stays, start="2024-01-15", end="2024-03-01"))
    assert got["entry_date"].between("2024-01-15", "2024-03-01").all() and len(got) == int(stays["entry_date"].between("2024-01-15", "2024-03-01").sum())
    assert (SL.filter_stays(stays, state_for(stays, rank_high=20))["entry_rank"] <= 20).all()          # the place range = the debut place
    assert set(SL.filter_stays(stays, state_for(stays, album_types=("single",)))["album_type"]) == {"single"}
    assert SL.filter_stays(stays, state_for(stays, explicit="explicit"))["is_explicit"].astype(bool).all()
    assert not SL.filter_stays(stays, state_for(stays, explicit="clean"))["is_explicit"].astype(bool).any()
    assert SL.filter_stays(stays, state_for(stays, collab="collab"))["is_collab"].astype(bool).all()
    assert not SL.filter_stays(stays, state_for(stays, collab="solo"))["is_collab"].astype(bool).any()
    assert set(SL.filter_stays(stays, state_for(stays, artists=("B",)))["primary_artist"]) == {"B"}
    assert SL.filter_stays(stays, state_for(stays, track_ids=(2, 5)))["track_id"].tolist() == [2, 5]
    assert len(SL.filter_stays(stays, state_for(stays))) == len(stays)


def test_the_artist_filter_follows_the_bridge_when_it_is_given():
    stays = toy_stays()
    bridge = pd.DataFrame({"track_id": [1, 2, 4], "artist_name": ["Z", "Z", "Z"], "is_primary": [False, False, False]})
    got = SL.filter_stays(stays, state_for(stays, artists=("Z",)), bridge)
    assert got["track_id"].tolist() == [1, 2, 4]


def test_a_filtered_selection_changes_the_curve(stays_real, dash_real):
    bounds = chart_bounds(dash_real.playlist)
    whole = SL.live_survival(stays_real, "none")["summary"].iloc[0]
    state = F.FilterState.create(bounds.first_date, bounds.last_date, rank_high=10)
    top = SL.live_survival(SL.filter_stays(stays_real, state, dash_real.bridge), "none")["summary"].iloc[0]
    assert top["songs"] < whole["songs"] and top["median_stay"] > whole["median_stay"]       # songs that debut high stay longer
