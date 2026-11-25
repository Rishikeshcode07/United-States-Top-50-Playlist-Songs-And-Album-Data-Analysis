"""
src/ml/dataset.py  -  the modeling table: one row per song, built ONLY from what is known on the day the song arrives.

WHAT THIS FILE IS
    One function, `build_modeling_table`, and the lists of column names that go with it. It turns the Phase 3 tables into a table
    that a model can learn from. Each row is a song that ENTERED the chart during our window (the 50 songs that were already on
    the chart on the first day are left out: we never saw their arrival). The columns come in five groups:

        identity     track_id, label, primary artist, entry date (never used as a model input)
        release      duration, album size, single or album, explicit, collaboration          (what the song IS)
        timing       the weekday of the first appearance, and how much history we have       (WHEN it arrives)
        artist       the chart record of the strongest credited artist BEFORE the arrival     (WHO it is by)
        debut        the position and the popularity on the first day                         (HOW it arrives)
        outcomes     what happened next: debut in the Top 10, climb into the Top 10 within 30 days, chart days in the first 30
                     days (and its three classes), popularity in the first week

WHY WE NEED IT
    The two ways to fool yourself with a model are to feed it the answer and to test it on the same things you trained it on.
    This file takes care of the first. A column such as "days on chart" or "average rank" is calculated from the whole life of
    a song, so it contains the outcome; it must never be a model input. Here every input is calculated from data that is
    strictly older than the song's first day on the chart. The artist record, for example, counts the artist's chart days
    BEFORE the arrival, never after it. The tests prove this by changing the future of the chart and checking that no input moves.

    Outcomes need the same care. "Does the song climb into the Top 10 within 30 days?" can only be answered for songs that we
    followed for 30 days; a song that arrived on the last week of the file has had no time to climb. Those songs are marked as
    not eligible instead of being counted as "no".

WHAT HAPPENS WHEN IT RUNS
    Days are counted in SNAPSHOTS (the project's rule since Phase 4: the file has 555 snapshots over 559 calendar days).
    `build_modeling_table(dim_track, fact_chart_daily, bridge_track_artist, dim_date)` returns a DataFrame sorted by entry date.
    Songs with several credited artists take the record of the credited artist with the longest earlier chart history.

EXPECTED OUTCOME
    On the real file: 894 songs, 229 primary artists. 894 have a debut outcome, 632 an eligible climb outcome (a debut outside
    the Top 10 and at least 30 snapshots of follow-up), 832 an eligible stay outcome, 878 a first-week popularity.

MOTIVE
    Models are only as honest as their inputs. Every column here can be explained with the sentence "we knew this on the day the
    song arrived".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------ settings
TOP_N = 10                       # "the Top 10"
HORIZON = 30                     # snapshots followed after the arrival (the first 30 days on the chart: ages 0 ... 29)
LOOKBACK = 90                    # snapshots used for the "recent record" of the artist
POPULARITY_DAYS = 7              # the first week: ages 0 ... 6
STAY_EDGES = (3, 14)             # 1-3 chart days = Flash, 4-14 = Short, 15 or more = Sustained
STAY_CLASSES = ["Flash (1-3 days)", "Short (4-14)", "Sustained (15+)"]
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]      # as recorded in the file (the calendar sits about a year ahead)
NEVER_CHARTED_RANK = 51          # "best earlier rank" of an artist with no earlier chart day: one place below the chart

# ------------------------------------------------------------------------------------------------ column groups
IDENTITY_COLUMNS = ["track_id", "track_label", "primary_artist", "record_artist", "entry_date", "entry_snapshot"]
RELEASE_FEATURES = ["duration_min", "total_tracks", "is_single", "is_explicit", "is_collab"]
TIMING_FEATURES = [f"weekday_{name}" for name in WEEKDAYS] + ["history_days"]
ARTIST_FEATURES = ["artist_prior_slot_days", "artist_slot_days_recent", "artist_prior_songs", "artist_prior_top10_days",
                   "artist_prior_best_rank", "artist_prior_mean_pop", "artist_on_chart_prev_day"]
DEBUT_FEATURES = ["entry_rank", "entry_pop"]
EARLY_FEATURES = ["pos_day2", "move_first_3d", "best_rank_first_3d", "pop_day3", "pop_move_first_3d"]    # the first three days on the chart
OUTCOME_COLUMNS = ["follow_up_days", "y_debut_top10", "ok_climb", "y_climb_top10", "ok_stay", "chart_days_30", "stay_class",
                   "ok_pop", "pop_first_week", "ok_early", "y_climb_after3", "y_sustained_after3"]

FEATURE_BLOCKS = {                # block name -> columns
    "release": RELEASE_FEATURES,
    "timing": TIMING_FEATURES,
    "artist": ARTIST_FEATURES,
    "debut": DEBUT_FEATURES,
    "early": EARLY_FEATURES,
}

REQUIRED_COLUMNS = {
    "dim_track": ["track_id", "track_label", "primary_artist", "duration_min", "total_tracks", "release_format", "is_explicit",
                  "is_collab", "first_seen"],
    "fact_chart_daily": ["date", "track_id", "position", "popularity_clean", "in_top10"],
    "bridge_track_artist": ["track_id", "artist_name", "artist_order"],
    "dim_date": ["date", "has_snapshot", "snapshot_index"],
}


def _check_columns(tables):
    for name, needed in REQUIRED_COLUMNS.items():
        missing = [column for column in needed if column not in tables[name].columns]
        if missing:
            raise ValueError(f"the table {name} is missing the column(s) {missing}")


def stay_class(chart_days):
    """The three classes of the first 30 days: 1-3 chart days, 4-14, 15 or more (NaN stays NaN)."""
    days = np.asarray(chart_days, dtype=float)
    labels = np.full(days.shape, None, dtype=object)
    known = np.isfinite(days)
    labels[known & (days <= STAY_EDGES[0])] = STAY_CLASSES[0]
    labels[known & (days > STAY_EDGES[0]) & (days <= STAY_EDGES[1])] = STAY_CLASSES[1]
    labels[known & (days > STAY_EDGES[1])] = STAY_CLASSES[2]
    return labels


# =============================================================================================== the artist record
def _artist_arrays(chart_rows, n_artists, n_snapshots):
    """
    Cumulative chart record of every artist, snapshot by snapshot. `chart_rows` has one row per chart slot and credited artist, with the
    integer columns `a` (artist number) and `s` (snapshot number) and the columns position, popularity_clean, in_top10.
    Returns arrays of shape (artists, snapshots): the value at snapshot s is the total over snapshots 0 ... s.
    """
    a = chart_rows["a"].to_numpy()
    s = chart_rows["s"].to_numpy()
    shape = (n_artists, n_snapshots)

    slots = np.zeros(shape)
    np.add.at(slots, (a, s), 1.0)
    top10 = np.zeros(shape)
    np.add.at(top10, (a, s), chart_rows["in_top10"].to_numpy(dtype=float))
    popularity = chart_rows["popularity_clean"].to_numpy(dtype=float)
    known = np.isfinite(popularity)
    pop_sum = np.zeros(shape)
    np.add.at(pop_sum, (a[known], s[known]), popularity[known])
    pop_n = np.zeros(shape)
    np.add.at(pop_n, (a[known], s[known]), 1.0)
    best = np.full(shape, np.inf)
    np.minimum.at(best, (a, s), chart_rows["position"].to_numpy(dtype=float))

    return {"slots": slots, "cum_slots": np.cumsum(slots, axis=1), "cum_top10": np.cumsum(top10, axis=1),
            "cum_pop_sum": np.cumsum(pop_sum, axis=1), "cum_pop_n": np.cumsum(pop_n, axis=1),
            "best_so_far": np.minimum.accumulate(best, axis=1)}


def _before(cumulative, artist, snapshot):
    """The value of a cumulative array just BEFORE `snapshot` (0 for the first snapshot)."""
    return 0.0 if snapshot <= 0 else cumulative[artist, snapshot - 1]


# =============================================================================================== the table
def build_modeling_table(dim_track, fact_chart_daily, bridge_track_artist, dim_date, horizon: int = HORIZON, lookback: int = LOOKBACK):
    """
    The modeling table described at the top of this file. The four arguments are the Phase 3 tables of the same names.
    `horizon` is the number of snapshots followed after the arrival; `lookback` the length of the artist's "recent record".
    """
    tables = {"dim_track": dim_track, "fact_chart_daily": fact_chart_daily, "bridge_track_artist": bridge_track_artist, "dim_date": dim_date}
    _check_columns(tables)

    # ---- every chart row gets the number of its snapshot (0 = the first snapshot)
    calendar = dim_date.loc[dim_date["has_snapshot"].astype(bool), ["date", "snapshot_index"]].copy()
    calendar["s"] = calendar["snapshot_index"].astype(int) - int(calendar["snapshot_index"].min())
    chart = fact_chart_daily[["date", "track_id", "position", "popularity_clean", "in_top10"]].merge(calendar[["date", "s"]], on="date", how="left")
    if chart["s"].isna().any():
        raise ValueError("some chart rows are on a date without a snapshot in dim_date")
    chart["s"] = chart["s"].astype(int)
    chart = chart.sort_values(["track_id", "s"], kind="mergesort").reset_index(drop=True)
    n_snapshots = int(calendar["s"].max()) + 1

    # ---- the arrival of every song: first snapshot, position and popularity on that day
    first = chart.drop_duplicates("track_id", keep="first").set_index("track_id")       # the FIRST row of each song (a missing popularity stays missing)
    arrivals = pd.DataFrame({"entry_snapshot": first["s"], "entry_date": first["date"], "entry_rank": first["position"],
                             "entry_pop": first["popularity_clean"]})
    songs = dim_track[REQUIRED_COLUMNS["dim_track"]]                  # only the song attributes: outcome columns of dim_track are never copied
    entrants = songs.merge(arrivals, left_on="track_id", right_index=True, how="inner")
    entrants = entrants[entrants["entry_snapshot"] > 0].copy()                    # arrivals we actually saw
    entrants["follow_up_days"] = (n_snapshots - 1) - entrants["entry_snapshot"]

    # ---- outcomes: what happened in the first `horizon` snapshots (age 0 ... horizon-1)
    aged = chart.merge(entrants[["track_id", "entry_snapshot"]], on="track_id", how="inner")
    aged["age"] = aged["s"] - aged["entry_snapshot"]
    inside = aged[aged["age"] < horizon]
    chart_days = inside.groupby("track_id").size().rename("chart_days_30")
    climbed = inside[inside["in_top10"].astype(bool)].groupby("track_id").size().rename("top10_days_30")
    first_week = aged[aged["age"] < POPULARITY_DAYS].groupby("track_id")["popularity_clean"].mean().rename("pop_first_week")
    entrants = entrants.merge(chart_days, left_on="track_id", right_index=True, how="left")
    entrants = entrants.merge(climbed, left_on="track_id", right_index=True, how="left")
    entrants = entrants.merge(first_week, left_on="track_id", right_index=True, how="left")

    entrants["y_debut_top10"] = (entrants["entry_rank"] <= TOP_N).astype(int)
    full_follow_up = entrants["follow_up_days"] >= horizon - 1
    entrants["ok_climb"] = (entrants["entry_rank"] > TOP_N) & full_follow_up
    entrants["y_climb_top10"] = np.where(entrants["ok_climb"], (entrants["top10_days_30"].fillna(0) > 0).astype(float), np.nan)
    entrants["ok_stay"] = full_follow_up
    entrants["chart_days_30"] = np.where(entrants["ok_stay"], entrants["chart_days_30"], np.nan)
    entrants["stay_class"] = stay_class(entrants["chart_days_30"].to_numpy())
    entrants["ok_pop"] = entrants["pop_first_week"].notna()

    # ---- the first three days (ages 0, 1, 2): what a label sees at the end of the third day on the chart
    first_days = aged[aged["age"] <= 2]
    position_by_age = first_days.pivot_table(index="track_id", columns="age", values="position", aggfunc="first").reindex(columns=[0, 1, 2])
    popularity_by_age = first_days.pivot_table(index="track_id", columns="age", values="popularity_clean", aggfunc="first").reindex(columns=[0, 1, 2])
    position_by_age = position_by_age.reindex(entrants["track_id"])
    popularity_by_age = popularity_by_age.reindex(entrants["track_id"])
    on_all_three = position_by_age.notna().all(axis=1).to_numpy()
    entrants["pos_day2"] = position_by_age[2].to_numpy()
    entrants["move_first_3d"] = (position_by_age[0] - position_by_age[2]).to_numpy()                   # places climbed in two days (negative = fell)
    entrants["best_rank_first_3d"] = position_by_age.min(axis=1).to_numpy()
    entrants["pop_day3"] = popularity_by_age[2].to_numpy()
    entrants["pop_move_first_3d"] = (popularity_by_age[2] - popularity_by_age[0]).to_numpy()
    top10_by_day2 = (position_by_age <= TOP_N).any(axis=1).to_numpy()
    top10_days_after = aged[(aged["age"] >= 3) & (aged["age"] < horizon) & aged["in_top10"].astype(bool)].groupby("track_id").size()
    later_top10 = entrants["track_id"].map(top10_days_after).fillna(0).to_numpy() > 0
    # early-warning population: still on the chart on all of the first three days, never in the Top 10 during them, followed for 30 snapshots
    entrants["ok_early"] = on_all_three & ~top10_by_day2 & full_follow_up.to_numpy()
    entrants["y_climb_after3"] = np.where(entrants["ok_early"], later_top10.astype(float), np.nan)
    sustained = (entrants["chart_days_30"] > STAY_EDGES[1]).astype(float)
    entrants["y_sustained_after3"] = np.where(entrants["ok_early"], sustained, np.nan)

    # ---- release and timing inputs
    entrants["is_single"] = (entrants["release_format"] == "Single").astype(int)
    entrants["is_explicit"] = entrants["is_explicit"].astype(int)
    entrants["is_collab"] = entrants["is_collab"].astype(int)
    weekday = pd.to_datetime(entrants["entry_date"]).dt.day_of_week.to_numpy()
    for number, name in enumerate(WEEKDAYS):
        entrants[f"weekday_{name}"] = (weekday == number).astype(int)
    entrants["history_days"] = np.minimum(entrants["entry_snapshot"], lookback).astype(int)

    # ---- the artist record before the arrival
    credits = bridge_track_artist[["track_id", "artist_name", "artist_order"]]
    names = sorted(credits["artist_name"].unique())
    number_of = {name: i for i, name in enumerate(names)}
    rows = chart.merge(credits[["track_id", "artist_name"]], on="track_id", how="inner")
    rows["a"] = rows["artist_name"].map(number_of).astype(int)
    arrays = _artist_arrays(rows, len(names), n_snapshots)

    # distinct songs of every artist by the snapshot of their own first appearance
    song_first = arrivals["entry_snapshot"]
    song_credits = credits.assign(a=credits["artist_name"].map(number_of), first_s=credits["track_id"].map(song_first))
    new_songs = np.zeros((len(names), n_snapshots))
    np.add.at(new_songs, (song_credits["a"].to_numpy(dtype=int), song_credits["first_s"].to_numpy(dtype=int)), 1.0)
    cum_new_songs = np.cumsum(new_songs, axis=1)

    credited = {track: group.sort_values("artist_order")["artist_name"].tolist() for track, group in credits.groupby("track_id")}
    record = []
    for track, snap in zip(entrants["track_id"], entrants["entry_snapshot"]):
        snap = int(snap)
        candidates = [number_of[name] for name in credited[track]]
        prior = [_before(arrays["cum_slots"], a, snap) for a in candidates]
        a = candidates[int(np.argmax(prior))]                       # first listed wins a tie
        recent = _before(arrays["cum_slots"], a, snap) - _before(arrays["cum_slots"], a, snap - lookback)
        n_pop = _before(arrays["cum_pop_n"], a, snap)
        best = arrays["best_so_far"][a, snap - 1]
        record.append({
            "record_artist": names[a],
            "artist_prior_slot_days": _before(arrays["cum_slots"], a, snap),
            "artist_slot_days_recent": recent,
            "artist_prior_songs": _before(cum_new_songs, a, snap),
            "artist_prior_top10_days": _before(arrays["cum_top10"], a, snap),
            "artist_prior_best_rank": float(best) if np.isfinite(best) else float(NEVER_CHARTED_RANK),
            "artist_prior_mean_pop": _before(arrays["cum_pop_sum"], a, snap) / n_pop if n_pop > 0 else np.nan,
            "artist_on_chart_prev_day": int(arrays["slots"][a, snap - 1] > 0),
        })
    entrants = pd.concat([entrants.reset_index(drop=True), pd.DataFrame(record)], axis=1)

    columns = IDENTITY_COLUMNS + RELEASE_FEATURES + TIMING_FEATURES + ARTIST_FEATURES + DEBUT_FEATURES + EARLY_FEATURES + OUTCOME_COLUMNS
    table = entrants[columns].sort_values(["entry_snapshot", "track_id"], kind="mergesort").reset_index(drop=True)
    table["entry_date"] = pd.to_datetime(table["entry_date"])
    return table
