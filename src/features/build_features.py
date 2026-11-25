"""
src/features/build_features.py  -  step 5 of the pipeline: turn the clean daily table into the analysis tables.

WHAT THIS FILE IS
    The feature-engineering logic of Phase 2 (Steps 3 - 7) as functions. It adds day-level features (calendar, rank tier,
    movement, rolling trend), then builds the seven output tables:

        playlist_daily       one row per date x song (wide; the table most notebooks and the Streamlit app read)
        fact_chart_daily     the same facts, slimmed down for the database (21 columns)
        dim_track            one row per song: the song-level KPIs (Days on Chart, Average Rank, Best Rank, ...)
        dim_artist           one row per artist: unique songs, Artist Dominance Index, ...
        bridge_track_artist  which artists are on which song (a song can have several artists)
        dim_date             one row per calendar day, including the days without a snapshot
        daily_summary        one row per snapshot: turnover, explicit share, average popularity, ...

WHY WE NEED IT
    The six KPIs of the brief (Days on Chart, Average Rank, Best Rank, Rank Volatility Index, Popularity Score Trend,
    Artist Dominance Index) and the Explicit Content Share do not exist in the raw file. They are calculated here, once,
    so that the notebooks, the SQL views, the Streamlit app and Power BI all show the SAME numbers.
    The split into fact and dimension tables is the "star schema" that Power BI and SQL like best.

WHAT HAPPENS WHEN IT RUNS
    `build_all(clean, clog)` returns a dictionary {table name: DataFrame}. Nothing is written to disk here;
    saving is the job of run_pipeline.py, and only after the quality gate has passed.

EXPECTED OUTCOME
    playlist_daily 27,750 x 54 | fact_chart_daily 27,750 x 21 | dim_track 944 x 47 | dim_artist 283 x 17 |
    bridge_track_artist 1,105 x 4 | dim_date 559 x 15 | daily_summary 555 x 15   (for the project file).

MOTIVE
    Calculate every KPI in exactly one place. A number that is defined twice will, one day, be defined differently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (CHART_SIZE, LONGEVITY_BINS, LONGEVITY_LABELS, RANK_TIER_BINS, RANK_TIER_LABELS, ROLLING_WINDOW,
                        SIZE_BINS, SIZE_LABELS, TABLE_COLUMNS, TURNOVER_FENCE_MULTIPLIER, VOLATILITY_MIN_DAYS)
from src.etl.errors import DataIntegrityError
from src.etl.numeric_utils import round_half_up
from src.logger import get_logger

log = get_logger(__name__)


def build_all(clean, clog):
    """Add the daily features and build the seven tables. `clean` is the output of clean.apply_cleaning."""
    df = add_daily_features(clean.copy())
    songs = build_dim_track(df)
    bridge = build_bridge(songs)
    artists = build_dim_artist(df, bridge)
    calendar = build_dim_date(df)
    daily = build_daily_summary(df)
    clog.add("3.x", "Added daily features (calendar, rank tiers, movement, chart life, rolling trend)",
             len(df), len(df), len(df), note="No rows removed; only new columns")
    return assemble_tables(df, songs, artists, bridge, calendar, daily)


# ====================================================================================== day-level features
def add_daily_features(df):
    """Add the columns that describe one row (one song on one day). `df` must be sorted by track_id, then date."""
    # ---- calendar features (so that nobody has to compute weekdays again in Power BI, SQL and Python) ----
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["year_month"] = df["date"].dt.strftime("%Y-%m")                  # text like 2024-05, sorts correctly
    df["quarter"] = df["date"].dt.quarter
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.day_of_week + 1                      # Monday = 1 ... Sunday = 7
    df["day_name"] = df["date"].dt.day_name()
    df["is_weekend"] = df["day_of_week"] >= 6

    # ---- rank features ----
    df["rank_tier"] = pd.cut(df["position"], bins=RANK_TIER_BINS, labels=RANK_TIER_LABELS).astype(str)
    df["in_top10"] = df["position"] <= 10
    df["in_top20"] = df["position"] <= 20
    df["is_number_one"] = df["position"] == 1
    df["chart_points"] = (CHART_SIZE + 1) - df["position"]               # rank 1 = 50 points ... rank 50 = 1 point
    df["duration_min"] = round_half_up(df["duration_ms"] / 60_000, 3)    # the brief's "Duration in minutes"
    df["is_collab"] = df["artist_count"] > 1                             # two or more artists on the song

    # ---- rank movement. Only a REAL "yesterday" counts: a song that was off the chart has no previous position. ----
    by_song = df.groupby("track_id")
    previous_position = by_song["position"].shift(1)                     # position on the song's previous appearance
    previous_snapshot = by_song["snapshot_index"].shift(1)
    was_on_previous_snapshot = (df["snapshot_index"] - previous_snapshot) == 1
    df["prev_position"] = previous_position.where(was_on_previous_snapshot)
    df["rank_change"] = df["prev_position"] - df["position"]             # positive = climbed, negative = fell

    position_7_ago = by_song["position"].shift(ROLLING_WINDOW)           # change over one week, only if on all 7 snapshots
    snapshot_7_ago = by_song["snapshot_index"].shift(ROLLING_WINDOW)
    df["rank_gain_7d"] = (position_7_ago - df["position"]).where((df["snapshot_index"] - snapshot_7_ago) == ROLLING_WINDOW)

    conditions = [df["is_stint_start"] & (df["snapshot_index"] == 1),    # already on the chart when our data begins
                  df["is_stint_start"] & (df["stint_no"] == 1),          # first appearance in our data
                  df["is_stint_start"],                                  # came back after being away
                  df["rank_change"] > 0, df["rank_change"] < 0, df["rank_change"] == 0]
    labels = ["Chart start", "New entry", "Re-entry", "Up", "Down", "Same"]
    df["movement"] = np.select([c.to_numpy() for c in conditions], labels, default="Unknown")
    if (df["movement"] == "Unknown").any():
        raise DataIntegrityError("Some rows could not be given a movement label.")

    # ---- life on the chart ----
    last_snapshot = df["snapshot_index"].max()
    df["exits_next_snapshot"] = df["is_stint_end"] & (df["snapshot_index"] < last_snapshot)   # a real exit, not the end of our data
    df["chart_day_number"] = df.groupby("track_id").cumcount() + 1                              # 1 = debut day, 2 = second chart day ...
    first_seen_per_song = df.groupby("track_id")["date"].transform("min")
    df["calendar_age_days"] = (df["date"] - first_seen_per_song).dt.days                        # days since the first appearance
    n_entries = int(df["movement"].isin(["New entry", "Re-entry"]).sum())
    n_exits = int(df["exits_next_snapshot"].sum())
    if n_entries != n_exits:
        raise DataIntegrityError(f"Entries ({n_entries}) and exits ({n_exits}) must match when every date has exactly "
                                 f"{CHART_SIZE} rows.")

    # ---- rolling trend inside ONE run, so nothing is averaged across a period when the song was off the chart ----
    by_run = df.groupby("stint_key")
    df["popularity_trend_7d"] = round_half_up(by_run["popularity_clean"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean()), 2)              # KPI: Popularity Score Trend
    df["avg_rank_7d"] = round_half_up(by_run["position"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean()), 2)
    df["rank_volatility_7d"] = round_half_up(by_run["position"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=VOLATILITY_MIN_DAYS).std()), 2)
    return df


# ====================================================================================== dim_track (one row per song)
def build_dim_track(df):
    """One row per song with the song-level KPIs. Returned with track_id as the index."""
    by_song = df.groupby("track_id")                                          # df is sorted by song, then date

    songs = by_song.agg(
        track_label=("track_label", "first"),
        song_clean=("song_clean", "first"),
        artist_credit=("artist_credit", "first"),
        primary_artist=("primary_artist", "first"),
        artist_count=("artist_count", "first"),
        artists_all=("artists_all", "first"),
        days_on_chart=("date", "size"),                                       # KPI: Days on Chart
        avg_rank=("position", "mean"),                                        # KPI: Average Rank
        median_rank=("position", "median"),
        best_rank=("position", "min"),                                        # KPI: Best Rank (best = smallest number)
        worst_rank=("position", "max"),
        rank_volatility_index=("position", "std"),                            # KPI: Rank Volatility Index
        avg_daily_rank_move=("rank_change", lambda s: s.abs().mean()),        # average size of the day-to-day change
        days_in_top10=("in_top10", "sum"),
        days_in_top20=("in_top20", "sum"),
        days_at_number_one=("is_number_one", "sum"),
        chart_points=("chart_points", "sum"),
        avg_popularity=("popularity_clean", "mean"),
        peak_popularity=("popularity_clean", "max"),
        popularity_std=("popularity_clean", "std"),                           # lower = more stable popularity
        n_runs=("stint_no", "max"),
    )
    songs["is_collab"] = songs["artist_count"] > 1
    for col in ["avg_rank", "avg_daily_rank_move", "rank_volatility_index", "avg_popularity", "popularity_std"]:
        songs[col] = round_half_up(songs[col], 2)

    # ---- the song's timeline ----
    songs["first_seen"] = by_song["date"].min()
    songs["last_seen"] = by_song["date"].max()
    songs["span_days"] = (songs["last_seen"] - songs["first_seen"]).dt.days + 1          # calendar days, first to last
    first_idx, last_idx = by_song["snapshot_index"].min(), by_song["snapshot_index"].max()
    songs["span_snapshots"] = last_idx - first_idx + 1                                    # snapshots, first to last
    songs["presence_ratio"] = round_half_up(songs["days_on_chart"] / songs["span_snapshots"], 3)   # 1.0 = never left the chart
    songs["longest_run_days"] = df.groupby(["track_id", "stint_key"]).size().groupby("track_id").max()
    songs["entry_rank"] = by_song["position"].first()                                     # position on the first day
    songs["exit_rank"] = by_song["position"].last()                                       # position on the last day
    peak_rows = by_song["position"].idxmin()                                              # first row with the best rank
    songs["peak_date"] = df.loc[peak_rows, ["track_id", "date"]].set_index("track_id")["date"]
    songs["days_to_peak"] = (songs["peak_date"] - songs["first_seen"]).dt.days

    # ---- censoring: what do we NOT know? A song already on the chart when our data begins has an unknown past; a
    # song still on the chart when it ends has an unfinished run. Longevity comparisons must respect that. ----
    window_start, window_end = df["date"].min(), df["date"].max()
    songs["on_chart_at_window_start"] = songs["first_seen"] == window_start
    songs["on_chart_at_window_end"] = songs["last_seen"] == window_end
    songs["run_status"] = np.select(
        [(songs["on_chart_at_window_start"] & songs["on_chart_at_window_end"]).to_numpy(),
         songs["on_chart_at_window_start"].to_numpy(),
         songs["on_chart_at_window_end"].to_numpy()],
        ["On chart the whole window", "Already on chart at start", "Still on chart at end"],
        default="Complete history observed")

    # ---- song attributes that can change over time: keep the MOST FREQUENT value (ties: the most recent one) ----
    descriptor_cols = ["duration_ms", "album_type", "total_tracks", "is_explicit"]
    rule_b_frequent = by_song[descriptor_cols].agg(_most_frequent)
    for col in descriptor_cols:
        songs[col] = rule_b_frequent[col]
    songs["duration_min"] = round_half_up(songs["duration_ms"] / 60_000, 3)
    songs["release_format"] = np.where(songs["album_type"] == "single", "Single", "Album")
    songs["release_size_band"] = pd.cut(songs["total_tracks"], bins=SIZE_BINS, labels=SIZE_LABELS).astype(str)
    songs["album_cover_url"] = by_song["album_cover_url"].last()                          # cosmetic: the latest artwork

    # ---- segmentation ----
    songs["longevity_class"] = pd.cut(songs["days_on_chart"], bins=LONGEVITY_BINS, labels=LONGEVITY_LABELS).astype(str)
    songs["reached_top10"] = songs["best_rank"] <= 10
    songs["reached_number_one"] = songs["best_rank"] == 1
    return songs


def _most_frequent(values):
    """Return the most common value; if several tie, the one that appeared most recently."""
    counts = values.value_counts()
    tied = counts[counts == counts.max()].index
    return values[values.isin(tied)].iloc[-1]


# ====================================================================================== bridge and dim_artist
def build_bridge(songs):
    """
    A song can have several artists and an artist can have several songs (many-to-many). A "bridge" table solves it:
    one row per (song, artist). `is_primary` marks the lead artist, which is what the Artist Dominance Index uses.
    """
    bridge = (songs[["artists_all"]].reset_index()
              .assign(artist_name=lambda t: t["artists_all"].str.split(" | ", regex=False))     # list of artists per song
              .explode("artist_name")                                                           # one row per artist
              .drop(columns="artists_all")
              .reset_index(drop=True))
    bridge["artist_order"] = bridge.groupby("track_id").cumcount() + 1                          # 1 = first artist in the credit
    bridge["is_primary"] = bridge["artist_order"] == 1
    if not bridge.groupby("track_id")["is_primary"].sum().eq(1).all():
        raise DataIntegrityError("Each song must have exactly one primary artist.")
    if bridge["track_id"].nunique() != len(songs):
        raise DataIntegrityError("The bridge table does not cover every song.")
    return bridge


def build_dim_artist(df, bridge):
    """One row per individual artist, with the Artist Dominance Index (share of all chart points)."""
    # Every daily row is counted once for EACH artist on the song
    artist_rows = df[["date", "track_id", "position", "chart_points", "in_top10", "is_number_one"]].merge(
        bridge[["track_id", "artist_name", "is_primary"]], on="track_id", how="inner")

    artists = artist_rows.groupby("artist_name").agg(
        unique_tracks=("track_id", "nunique"),                       # KPI: number of unique songs
        chart_slot_days=("date", "size"),                            # song-days (3 songs on one day = 3)
        days_present=("date", "nunique"),                            # distinct dates with at least one song
        first_seen=("date", "min"),
        last_seen=("date", "max"),
        avg_rank=("position", "mean"),
        best_rank=("position", "min"),
        days_in_top10=("in_top10", "sum"),
        days_at_number_one=("is_number_one", "sum"),
        chart_points=("chart_points", "sum"))

    as_primary = artist_rows[artist_rows["is_primary"]].groupby("artist_name").agg(
        unique_tracks_as_primary=("track_id", "nunique"),
        chart_slot_days_primary=("date", "size"),
        chart_points_primary=("chart_points", "sum"))
    artists = artists.join(as_primary)
    for col in as_primary.columns:
        artists[col] = artists[col].fillna(0).astype(int)            # artists who are never the lead get 0

    total_points, total_slots = df["chart_points"].sum(), len(df)
    adi_primary = artists["chart_points_primary"] / total_points * 100      # shares (%) that add up to exactly 100
    adi_all = artists["chart_points"] / total_points * 100
    if not np.isclose(adi_primary.sum(), 100.0):
        raise DataIntegrityError("Primary-basis dominance shares must add up to 100%.")

    artists["dominance_index_primary"] = round_half_up(adi_primary, 4)               # KPI: Artist Dominance Index
    artists["dominance_index_all_credits"] = round_half_up(adi_all, 4)
    artists["share_of_slots_primary_pct"] = round_half_up(artists["chart_slot_days_primary"] / total_slots * 100, 4)
    artists["avg_rank"] = round_half_up(artists["avg_rank"], 2)
    log.info("Artists: %s | top-10 artists hold %.1f%% of all chart points | HHI = %.0f", f"{len(artists):,}",
             adi_primary.nlargest(10).sum(), float((adi_primary ** 2).sum()))
    return artists.reset_index()


# ====================================================================================== dim_date and daily_summary
def build_dim_date(df):
    """
    A complete calendar: EVERY day between the first and the last date, including the days without a snapshot.
    A continuous date table is what Power BI's time intelligence (month-to-date, year-over-year ...) requires.
    """
    snapshots = (df[["date", "snapshot_index", "days_since_prev_snapshot"]].drop_duplicates("date")
                   .sort_values("snapshot_index").reset_index(drop=True))
    calendar = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")})
    calendar = calendar.merge(snapshots, on="date", how="left")
    calendar["has_snapshot"] = calendar["snapshot_index"].notna()
    calendar["snapshot_index"] = calendar["snapshot_index"].astype("Int64")
    calendar["days_since_prev_snapshot"] = calendar["days_since_prev_snapshot"].astype("Int64")

    calendar["date_key"] = calendar["date"].dt.strftime("%Y%m%d").astype(int)
    calendar["year"] = calendar["date"].dt.year
    calendar["month"] = calendar["date"].dt.month
    calendar["month_name"] = calendar["date"].dt.month_name()
    calendar["year_month"] = calendar["date"].dt.strftime("%Y-%m")
    calendar["quarter"] = calendar["date"].dt.quarter
    calendar["iso_week"] = calendar["date"].dt.isocalendar().week.astype(int)
    calendar["day_of_week"] = calendar["date"].dt.day_of_week + 1
    calendar["day_name"] = calendar["date"].dt.day_name()
    calendar["is_weekend"] = calendar["day_of_week"] >= 6
    calendar["season"] = np.select([calendar["month"].isin([12, 1, 2]), calendar["month"].isin([3, 4, 5]),
                                    calendar["month"].isin([6, 7, 8])], ["Winter", "Spring", "Summer"], default="Fall")
    return calendar


def build_daily_summary(df):
    """One row per snapshot: how much the chart changed that day and what its overall character was."""
    daily = df.groupby("date").agg(
        snapshot_index=("snapshot_index", "first"),
        unique_tracks=("track_id", "nunique"),
        avg_popularity=("popularity_clean", "mean"),
        median_popularity=("popularity_clean", "median"),
        explicit_share=("is_explicit", "mean"),                      # KPI: Explicit Content Share (0 to 1)
        collab_share=("is_collab", "mean"),
        avg_duration_min=("duration_min", "mean"),
        unique_primary_artists=("primary_artist", "nunique"))
    daily["single_share"] = df.assign(is_single=df["release_format"].eq("Single")).groupby("date")["is_single"].mean()

    daily["new_entries"] = df[df["movement"] == "New entry"].groupby("date").size().reindex(daily.index, fill_value=0)
    daily["re_entries"] = df[df["movement"] == "Re-entry"].groupby("date").size().reindex(daily.index, fill_value=0)

    exits_by_snapshot = df[df["exits_next_snapshot"]].groupby("snapshot_index").size()
    exits_by_snapshot.index = exits_by_snapshot.index + 1            # the exit is visible on the NEXT snapshot
    daily["exits"] = daily["snapshot_index"].map(exits_by_snapshot).fillna(0).astype(int)

    entries = daily["new_entries"] + daily["re_entries"]
    if not (entries.iloc[1:] == daily["exits"].iloc[1:]).all():
        raise DataIntegrityError("Entries and exits must match on every date.")
    daily["turnover_rate_pct"] = round_half_up(entries / CHART_SIZE * 100, 1)

    # A "shock day" is a day with unusually many entries: Tukey's far-out fence = Q3 + 3 x IQR.
    # Day 1 can not have entries (there is no yesterday), so it is left out of the quartiles.
    q1, q3 = entries.iloc[1:].quantile([0.25, 0.75])
    fence = q3 + TURNOVER_FENCE_MULTIPLIER * (q3 - q1)
    daily["is_high_turnover_day"] = entries > fence
    for column, decimals in {"avg_popularity": 2, "median_popularity": 2, "explicit_share": 4, "collab_share": 4,
                             "single_share": 4, "avg_duration_min": 3}.items():
        daily[column] = round_half_up(daily[column], decimals)
    log.info("High-turnover days (more than %.0f entries): %s of %s", fence, int(daily["is_high_turnover_day"].sum()), len(daily))
    return daily.reset_index()


# ====================================================================================== the seven output tables
def assemble_tables(df, songs, artists, bridge, calendar, daily):
    """Select the exact columns (and order) promised by config.TABLE_COLUMNS for each table."""
    cols = TABLE_COLUMNS
    tables = {
        "playlist_daily":      df.sort_values(["date", "position"])[cols["playlist_daily"]].reset_index(drop=True),
        "fact_chart_daily":    df.sort_values(["date", "position"])[cols["fact_chart_daily"]].reset_index(drop=True),
        "dim_track":           songs.reset_index()[cols["dim_track"]].sort_values("track_id").reset_index(drop=True),
        "dim_artist":          (artists[cols["dim_artist"]]
                                .sort_values(["dominance_index_primary", "artist_name"], ascending=[False, True])
                                .reset_index(drop=True)),
        "bridge_track_artist": bridge[cols["bridge_track_artist"]],
        "dim_date":            calendar[cols["dim_date"]],
        "daily_summary":       daily[cols["daily_summary"]],
    }
    for name, table in tables.items():
        log.info("%-22s %7s rows x %2d columns", name, f"{table.shape[0]:,}", table.shape[1])
    return tables
