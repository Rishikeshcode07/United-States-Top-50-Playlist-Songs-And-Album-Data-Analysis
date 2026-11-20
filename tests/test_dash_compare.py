"""
tests/test_dash_compare.py  -  comparing groups of songs, and popularity against place.

WHAT THIS FILE IS
    Tests of src/dash/compare.py: effect sizes (Cliff's delta, Spearman), group profiles, popularity by place, the daily rank
    correlation, the 7-day popularity change, and shares by period and by band of places.

WHY WE NEED IT
    A comparison that sounds careful ("singles stay longer, a small effect") is only as good as the arithmetic under it. Known small
    cases (where the answer can be written by hand) and independent routes (scipy, loops) are used here.

WHAT HAPPENS WHEN IT RUNS
    Fast; real-data checks are skipped when data/processed/ is not built.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Effect sizes that are right, and plain statements of what they measure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.dash import compare as C
from src.dash import metrics as M


# ---------------------------------------------------------------------------------------------- effect sizes
def test_cliffs_delta_on_hand_made_cases():
    assert C.cliffs_delta([4, 5, 6], [1, 2, 3]) == pytest.approx(1.0)       # every member of a is larger
    assert C.cliffs_delta([1, 2, 3], [4, 5, 6]) == pytest.approx(-1.0)
    assert C.cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert C.cliffs_delta([1, 2], [2, 3]) == pytest.approx(-0.75)           # four pairs: three are smaller, one is a tie -> (0 - 3) / 4
    assert np.isnan(C.cliffs_delta([], [1, 2]))


def test_cliffs_delta_matches_the_definition_with_ties():
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 8, 40), rng.integers(2, 10, 55)
    greater = sum(x > y for x in a for y in b)
    smaller = sum(x < y for x in a for y in b)
    assert C.cliffs_delta(a, b) == pytest.approx((greater - smaller) / (len(a) * len(b)))


def test_two_group_summary_agrees_with_scipy():
    rng = np.random.default_rng(4)
    a, b = rng.exponential(20, 60), rng.exponential(14, 80)
    got = C.two_group_summary(a, b)
    assert got["n_a"] == 60 and got["n_b"] == 80
    assert got["median_a"] == pytest.approx(np.median(a)) and got["median_difference"] == pytest.approx(np.median(a) - np.median(b))
    assert got["p"] == pytest.approx(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    assert got["delta"] == pytest.approx(C.cliffs_delta(a, b))
    assert got["size"] in {"negligible", "small", "medium", "large"}


def test_two_group_summary_with_a_tiny_group_gives_no_test():
    got = C.two_group_summary([1.0], [2.0, 3.0, 4.0])
    assert got["n_a"] == 1 and np.isnan(got["p"]) and np.isnan(got["delta"]) and got["size"] == ""
    assert np.isnan(C.two_group_summary([], [])["median_a"])


def test_spearman_known_cases():
    assert C.spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])["rho"] == pytest.approx(1.0)
    assert C.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])["rho"] == pytest.approx(-1.0)
    flat = C.spearman([1, 1, 1, 1], [1, 2, 3, 4])
    assert np.isnan(flat["rho"]) and flat["n"] == 4
    short = C.spearman([1, 2], [3, 4])
    assert np.isnan(short["rho"]) and short["n"] == 2
    gap = C.spearman([1, 2, 3, np.nan, 5], [2, 4, 6, 8, np.nan])
    assert gap["n"] == 3


# ------------------------------------------------------------------------------------------------ group profile
def test_group_profile_matches_plain_python(dash_synthetic):
    songs = M.song_table(dash_synthetic.playlist)
    profile = C.group_profile(songs, "is_explicit", labels={True: "Explicit", False: "Not explicit"}, order=["Not explicit", "Explicit"])
    assert list(profile["group"]) == ["Not explicit", "Explicit"]
    for _, row in profile.iterrows():
        flag = row["group"] == "Explicit"
        mine = songs[songs["is_explicit"] == flag]
        assert row["songs"] == len(mine)
        assert row["median_days"] == mine["days_on_chart"].median() and row["mean_days"] == pytest.approx(mine["days_on_chart"].mean())
        assert row["share_top10"] == pytest.approx((mine["best_rank"] <= 10).mean())
        assert row["share_number_one"] == pytest.approx((mine["days_at_number_one"] > 0).mean())
        assert row["share_of_points"] == pytest.approx(mine["chart_points"].sum() / songs["chart_points"].sum())
    assert profile["share_of_songs"].sum() == pytest.approx(1.0) and profile["share_of_points"].sum() == pytest.approx(1.0)


def test_group_profile_accepts_a_series_and_keeps_the_order(dash_synthetic):
    songs = M.song_table(dash_synthetic.playlist)
    band = C.duration_band(songs["duration_min"])
    profile = C.group_profile(songs, band, order=C.DURATION_LABELS)
    assert list(profile["group"]) == [g for g in C.DURATION_LABELS if g in set(band)]
    assert profile["songs"].sum() == len(songs)


def test_group_profile_of_nothing(dash_synthetic):
    empty = C.group_profile(M.song_table(dash_synthetic.playlist.iloc[:0]), "is_explicit")
    assert empty.empty and "median_days" in empty.columns


def test_singles_stay_longer_on_the_real_file_as_hypothesis_h2_says(dash_real):
    songs = M.song_table(dash_real.playlist)
    singles = songs.loc[songs["album_type"] == "single", "days_on_chart"]
    others = songs.loc[songs["album_type"] != "single", "days_on_chart"]
    got = C.two_group_summary(singles, others)
    assert got["n_a"] == 238 and got["n_b"] == 706
    assert got["delta"] == pytest.approx(0.2328, abs=0.001) and got["size"] == "small"


def test_duration_band_edges():
    got = C.duration_band([1.0, 2.49, 2.5, 3.49, 3.5, 4.49, 4.5, 9.0]).tolist()
    assert got == ["under 2.5 min", "under 2.5 min", "2.5 to 3.5 min", "2.5 to 3.5 min", "3.5 to 4.5 min", "3.5 to 4.5 min",
                   "4.5 min or more", "4.5 min or more"]


# ------------------------------------------------------------------------------------ popularity against place
def test_position_profile(dash_synthetic):
    play = dash_synthetic.playlist
    profile = C.position_profile(play)
    assert profile["position"].tolist() == list(range(1, 51))
    assert (profile["n"] == play["date"].nunique()).all()
    one = play[play["position"] == 7]
    row = profile[profile["position"] == 7].iloc[0]
    assert row["mean"] == pytest.approx(one["popularity_clean"].mean()) and row["median"] == one["popularity_clean"].median()
    assert row["p25"] <= row["median"] <= row["p75"]
    assert row["explicit_share"] == pytest.approx(one["is_explicit"].mean())


def test_daily_correlation_is_positive_when_popular_songs_sit_higher():
    days = pd.date_range("2024-01-01", periods=3)
    frame = pd.DataFrame({"date": np.repeat(days, 10), "position": np.tile(np.arange(1, 11), 3)})
    frame["popularity_clean"] = 100 - frame["position"] * 3            # place 1 is the most popular
    assert np.allclose(C.daily_rank_popularity_corr(frame)["rho"], 1.0)
    frame["popularity_clean"] = frame["position"] * 3                  # place 1 is the least popular
    assert np.allclose(C.daily_rank_popularity_corr(frame)["rho"], -1.0)


def test_daily_correlation_matches_scipy_for_single_days(dash_real):
    play = dash_real.playlist
    table = C.daily_rank_popularity_corr(play).set_index("date")
    for day in ("2024-07-04", "2025-02-14", "2025-10-31"):
        one = play[play["date"] == day].dropna(subset=["popularity_clean"])
        expected = stats.spearmanr(51 - one["position"], one["popularity_clean"]).statistic
        assert table.loc[pd.Timestamp(day), "rho"] == pytest.approx(expected, abs=1e-9)
    assert (table["n"] >= 5).all()


def test_daily_correlation_skips_days_with_too_few_songs():
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 3), "position": [1, 2, 3], "popularity_clean": [3.0, 2.0, 1.0]})
    assert C.daily_rank_popularity_corr(frame).empty
    assert len(C.daily_rank_popularity_corr(frame, min_songs=3)) == 1


def test_pop_change_is_the_popularity_now_minus_the_popularity_7_snapshots_ago(dash_real):
    play = dash_real.playlist
    out = C.add_pop_change(play, 7)
    assert out.index.equals(play.index) and len(out) == len(play)
    sample = out[out["pop_change_lag"].notna()].sample(40, random_state=1)
    for _, row in sample.iterrows():
        earlier = play[(play["track_id"] == row["track_id"]) & (play["snapshot_index"] == row["snapshot_index"] - 7)]
        assert len(earlier) == 1
        assert row["pop_change_lag"] == pytest.approx(row["popularity_clean"] - earlier["popularity_clean"].iloc[0])
    assert out.loc[out["snapshot_index"] <= 7, "pop_change_lag"].isna().all()
    assert "_pop_before" not in out.columns


# --------------------------------------------------------------------------------------------------- shares
def test_share_by_period_matches_plain_python(dash_synthetic):
    play = dash_synthetic.playlist
    got = C.share_by_period(play, "is_explicit", "day").set_index("period")
    for day, group in list(play.groupby(play["date"].dt.normalize()))[:5]:
        assert got.loc[day, "share"] == pytest.approx(group["is_explicit"].mean()) and got.loc[day, "rows"] == len(group)
    assert C.share_by_period(play.iloc[:0], "is_explicit").empty


def test_share_by_band_has_one_row_per_band(dash_synthetic):
    play = dash_synthetic.playlist
    got = C.share_by_band(play, "is_collab")
    assert got["band"].tolist() == ["1 to 10", "11 to 20", "21 to 30", "31 to 40", "41 to 50"]
    first = play[play["position"] <= 10]
    assert got.iloc[0]["share"] == pytest.approx(first["is_collab"].mean()) and got.iloc[0]["rows"] == len(first)
    assert C.share_by_band(play.iloc[:0], "is_collab").empty
