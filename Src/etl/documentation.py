"""
src/etl/documentation.py  -  step 7 of the pipeline: write the data dictionary and the KPI dictionary.

WHAT THIS FILE IS
    Two tables that explain the project to a human:
        data dictionary : one row per column of every output table (plain type, plain-English definition, an example value)
        KPI dictionary  : one row per KPI (where to find it, the formula, how to read it)

WHY WE NEED IT
    A column called `rank_volatility_index` means nothing to a new colleague, a reviewer or the person who opens the
    Power BI file in six months. The dictionaries are the "user manual" of the data. They are also checked by the
    pipeline: if somebody adds a column and forgets to describe it, the run stops.

WHAT HAPPENS WHEN IT RUNS
    `build_data_dictionary(tables)` looks up a definition for every column (table-specific wording first, then the
    general wording) and adds the first non-empty value of the column as an example.
    Numbers that depend on the data (number of dates, missing days, chart slots) are filled in from the tables,
    so the text stays true when the CSV is refreshed.

EXPECTED OUTCOME
    A data dictionary with 173 rows for the project file and a KPI dictionary with 11 rows.

MOTIVE
    A number without a definition is an opinion. Documentation is part of the product, not an afterthought.
"""
from __future__ import annotations

import pandas as pd

from src.config import CHART_SIZE
from src.etl.errors import DataIntegrityError

COMMON = {
    # ---- time ----
    "date": "Calendar date of the playlist snapshot, kept exactly as provided in the source file.",
    "snapshot_index": "Order number of the snapshot: 1 for the first date, {n_dates} for the last. Missing calendar days get no number.",
    "days_since_prev_snapshot": "Calendar days since the previous snapshot (1 normally, 2 right after a missing day; empty for the first).",
    "year": "Calendar year.", "month": "Calendar month number (1-12).", "month_name": "Calendar month name.",
    "year_month": "Year and month as text, e.g. 2024-05 (sorts correctly).", "quarter": "Calendar quarter (1-4).",
    "iso_week": "ISO week number of the year.", "day_of_week": "Day of the week, Monday = 1 ... Sunday = 7.",
    "day_name": "Name of the weekday.", "is_weekend": "True for Saturday and Sunday.",
    "date_key": "Date as a whole number yyyymmdd, e.g. 20240518 (handy join key).",
    "has_snapshot": "True if the playlist was captured on this calendar day; False for the {n_missing_days} missing days.",
    "season": "Northern-hemisphere season: Winter (Dec-Feb), Spring (Mar-May), Summer (Jun-Aug), Fall (Sep-Nov).",
    # ---- song identity ----
    "track_id": "Unique id of a song after resolving title and artist-credit variants (Phase 1, Step 6.5).",
    "track_label": "Readable name: song title + 'by' + artist credit. Unique per song.",
    "song_clean": "Cleaned song title (single spaces, one spelling per song).",
    "artist_credit": "Fullest artist credit seen for the song; artists joined with ' & '.",
    "primary_artist": "First (lead) artist of the credit.",
    "artist_count": "Number of artists in the credit.",
    "artists_all": "All artists of the credit joined with ' | ' (easy to split into a list).",
    "is_collab": "True when the song has two or more artists.",
    # ---- chart position ----
    "position": "Playlist rank on that day: 1 = top, 50 = bottom.",
    "rank_tier": "Rank group of the row: Top 10, 11-20 or 21-50.",
    "in_top10": "True when position is 10 or better.", "in_top20": "True when position is 20 or better.",
    "is_number_one": "True when position = 1.",
    "chart_points": "51 minus position (rank 1 = 50 points ... rank 50 = 1 point); the basis of the Artist Dominance Index.",
    # ---- popularity ----
    "popularity": "Popularity score from the source file (0-100). A 0 means 'not yet scored'; see popularity_clean.",
    "popularity_clean": "Popularity after repairing zeros (run interpolation, then song median; empty if no real value exists).",
    "popularity_was_zero": "True when the source popularity value was 0.",
    "popularity_fill_method": "How popularity_clean was obtained: original, interpolated_in_run, song_median or unfilled.",
    "popularity_trend_7d": "Popularity Trend Score: 7-day rolling average of popularity_clean inside the same chart run.",
    # ---- movement ----
    "prev_position": "Position on the immediately preceding snapshot (empty on the first day of a run).",
    "rank_change": "prev_position minus position. Positive = climbed, negative = fell, 0 = unchanged.",
    "movement": "Up, Down, Same, New entry, Re-entry or Chart start.",
    "rank_gain_7d": "Position 7 snapshots ago minus position now (positive = climbed over the week); empty unless present on all 7.",
    "avg_rank_7d": "7-day rolling average of position inside the same run.",
    "rank_volatility_7d": "7-day rolling standard deviation of position inside the same run (needs at least 3 days).",
    # ---- life on the chart ----
    "stint_no": "Run number of the song: 1 for its first unbroken streak on the chart, 2 for the next, and so on.",
    "stint_key": "Unique id of one run: track_id-run number.",
    "days_in_stint": "Day number inside the current run (1 = first day of the run).",
    "stint_length": "Total length in days of the run this row belongs to.",
    "is_stint_start": "True on the first day of a run (an entry or re-entry).",
    "is_stint_end": "True on the last day of a run.",
    "exits_next_snapshot": "True when the song is on the chart today but absent on the next snapshot (a real exit).",
    "chart_day_number": "Chart days accumulated so far by the song (1 = debut day).",
    "calendar_age_days": "Calendar days since the song's first appearance.",
    # ---- attributes ----
    "duration_ms": "Song duration in milliseconds.", "duration_min": "Song duration in minutes (the brief's 'Duration in minutes').",
    "album_type": "Album type from the source: single, album or compilation.",
    "release_format": "Single, or Album (album and compilation together).",
    "total_tracks": "Number of tracks on the release.",
    "release_size_band": "Release size group: 1 track, 2-3, 4-7, 8-15, 16-40 or 41+.",
    "is_explicit": "Explicit-content flag.", "album_cover_url": "Link to the artwork image.",
    # ---- song KPIs ----
    "days_on_chart": "KPI Days on Chart: number of snapshots on which the song appeared.",
    "first_seen": "First date the song appeared.", "last_seen": "Last date the song appeared.",
    "span_days": "Calendar days from first to last appearance, inclusive.",
    "span_snapshots": "Snapshots from first to last appearance, inclusive.",
    "presence_ratio": "days_on_chart divided by span_snapshots: 1.0 = never left the chart, near 0 = many short visits.",
    "n_runs": "Number of separate chart runs.", "longest_run_days": "Length of the longest unbroken run in days.",
    "avg_rank": "KPI Average Rank: mean position over the song's chart days (lower is better).",
    "median_rank": "Median position.", "best_rank": "KPI Best Rank Achieved: the smallest position reached.",
    "worst_rank": "Largest (worst) position reached.",
    "rank_volatility_index": "KPI Rank Volatility Index: standard deviation of position (empty for songs with a single day).",
    "avg_daily_rank_move": "Average absolute day-to-day rank change (day-to-day jumpiness).",
    "days_in_top10": "Days in the Top 10.", "days_in_top20": "Days in the Top 20.", "days_at_number_one": "Days at #1.",
    "entry_rank": "Position on the first chart day.", "exit_rank": "Position on the last chart day.",
    "peak_date": "First date the best rank was reached.", "days_to_peak": "Calendar days from first appearance to peak_date.",
    "avg_popularity": "Average of popularity_clean.", "peak_popularity": "Highest popularity_clean.",
    "popularity_std": "Standard deviation of popularity_clean (lower = more stable).",
    "on_chart_at_window_start": "True when the song was already on the chart on the first snapshot (history before our data is unknown).",
    "on_chart_at_window_end": "True when the song is still on the chart on the last snapshot (its run is not finished).",
    "run_status": "Complete history observed, Already on chart at start, Still on chart at end, or On chart the whole window.",
    "longevity_class": "Flash (1-3 days), Short (4-14), Medium (15-60), Long (61-180) or Evergreen (181+).",
    "reached_top10": "True when the song reached position 10 or better.", "reached_number_one": "True when the song reached #1.",
    # ---- bridge ----
    "artist_name": "Individual artist name (standardized spelling).",
    "artist_order": "Position of the artist inside the credit (1 = lead).", "is_primary": "True for the lead artist.",
    # ---- daily summary ----
    "unique_tracks": "Distinct songs on the date (always {chart_size}).",
    "new_entries": "Songs appearing in our data for the first time on this date.",
    "re_entries": "Songs returning after being off the chart.",
    "exits": "Songs that were on the previous snapshot but are not on this one.",
    "turnover_rate_pct": "(new_entries + re_entries) / {chart_size} x 100.",
    "is_high_turnover_day": "True when entries exceed Tukey's upper far-out fence (Q3 + 3 x IQR).",
    "median_popularity": "Median popularity_clean of the {chart_size} songs.",
    "explicit_share": "KPI Explicit Content Share: share of the {chart_size} rows flagged explicit (0-1).",
    "collab_share": "Share of the {chart_size} rows that are collaborations (0-1).",
    "single_share": "Share of the {chart_size} rows that are singles (0-1).",
    "avg_duration_min": "Average song duration in minutes.",
    "unique_primary_artists": "Distinct lead artists on the date.",
}

SONG_LEVEL = " Song level: the most frequent value across the song's chart days (ties: most recent)."
OVERRIDES = {
    ("dim_track", "duration_ms"): COMMON["duration_ms"] + SONG_LEVEL,
    ("dim_track", "duration_min"): COMMON["duration_min"] + SONG_LEVEL,
    ("dim_track", "album_type"): COMMON["album_type"] + SONG_LEVEL,
    ("dim_track", "release_format"): COMMON["release_format"] + SONG_LEVEL,
    ("dim_track", "total_tracks"): COMMON["total_tracks"] + SONG_LEVEL,
    ("dim_track", "release_size_band"): COMMON["release_size_band"] + SONG_LEVEL,
    ("dim_track", "is_explicit"): COMMON["is_explicit"] + SONG_LEVEL,
    ("dim_track", "album_cover_url"): "Link to the song's latest artwork image.",
    ("dim_track", "chart_points"): "Total chart points earned by the song (sum of 51 - position).",
    ("dim_artist", "unique_tracks"): "KPI: number of unique songs the artist is credited on.",
    ("dim_artist", "first_seen"): "First date the artist had a song on the chart.",
    ("dim_artist", "last_seen"): "Last date the artist had a song on the chart.",
    ("dim_artist", "avg_rank"): "Mean position over all of the artist's song-days (every credited artist counts).",
    ("dim_artist", "best_rank"): "Best position reached by any of the artist's songs.",
    ("dim_artist", "days_in_top10"): "Song-days the artist had in the Top 10.",
    ("dim_artist", "days_at_number_one"): "Song-days the artist had at #1.",
    ("dim_artist", "chart_points"): "Total chart points of all songs the artist is credited on.",
    ("dim_artist", "unique_tracks_as_primary"): "Songs on which the artist is the lead artist.",
    ("dim_artist", "chart_slot_days"): "Song-days on the chart (three songs on one day count as three).",
    ("dim_artist", "chart_slot_days_primary"): "Song-days where the artist is the lead artist.",
    ("dim_artist", "days_present"): "KPI: distinct dates on which the artist had at least one song on the chart.",
    ("dim_artist", "chart_points_primary"): "Chart points of songs where the artist is the lead artist.",
    ("dim_artist", "dominance_index_primary"): "KPI Artist Dominance Index (%): share of all chart points earned as lead artist; adds up to 100.",
    ("dim_artist", "dominance_index_all_credits"): "Artist Dominance Index (%) when every credited artist receives the song's full points; adds up to more than 100.",
    ("dim_artist", "share_of_slots_primary_pct"): "Share (%) of all {total_slots:,} chart slots held as lead artist.",
    ("daily_summary", "avg_popularity"): "Average popularity_clean of the {chart_size} songs on the date.",
    ("daily_summary", "snapshot_index"): COMMON["snapshot_index"],
}


def _definition(table, column, facts):
    """Table-specific wording first, then the general one. Placeholders such as {n_dates} are filled from the data."""
    text = OVERRIDES.get((table, column), COMMON.get(column))
    return text.format(**facts) if text and "{" in text else text


def _friendly_type(series):
    """A plain label for a column's type. pandas names the same thing differently in different versions ("object" or "str")."""
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "true/false"
    if pd.api.types.is_integer_dtype(dtype):
        return "whole number"
    if pd.api.types.is_float_dtype(dtype):
        return "decimal number"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "date"
    return "text"


def build_data_dictionary(tables):
    """One row per column of every table: table, column, data_type, definition, example. Stops if a column has no definition."""
    facts = {"n_dates": int(tables["daily_summary"]["date"].nunique()),
             "n_missing_days": int((~tables["dim_date"]["has_snapshot"]).sum()),
             "total_slots": int(len(tables["playlist_daily"])),
             "chart_size": CHART_SIZE}
    rows = []
    for table_name, table in tables.items():
        for col in table.columns:
            example = table[col].dropna()
            rows.append({"table": table_name, "column": col, "data_type": _friendly_type(table[col]),
                         "definition": _definition(table_name, col, facts),
                         "example": str(example.iloc[0])[:45] if len(example) else ""})
    dictionary = pd.DataFrame(rows)
    undocumented = dictionary[dictionary["definition"].isna()]
    if not undocumented.empty:
        raise DataIntegrityError(f"Columns without a definition: {undocumented[['table', 'column']].values.tolist()}. "
                                 "Add them to COMMON or OVERRIDES in src/etl/documentation.py.")
    return dictionary


def build_kpi_dictionary():
    """The KPIs of the project brief (and a few extra) with formula and interpretation."""
    return pd.DataFrame([
        ("Days on Chart", "dim_track.days_on_chart", "Count of snapshots on which the song appears", "Longevity indicator"),
        ("Average Rank", "dim_track.avg_rank", "Mean of position over the song's chart days", "Overall performance; lower is better"),
        ("Best Rank Achieved", "dim_track.best_rank", "Minimum of position", "Peak performance"),
        ("Rank Volatility Index", "dim_track.rank_volatility_index", "Standard deviation of position over the song's chart days", "Stability: higher = less stable"),
        ("Average Daily Rank Move", "dim_track.avg_daily_rank_move", "Mean of |rank_change| from one snapshot to the next", "Day-to-day jumpiness"),
        ("Popularity Score Trend", "playlist_daily.popularity_trend_7d", "7-day rolling mean of popularity_clean inside a run", "Listener engagement over time"),
        ("Artist Dominance Index", "dim_artist.dominance_index_primary", "Artist's chart points / all chart points x 100; points = 51 - position", "Market presence; adds up to 100%"),
        ("Explicit Content Share", "daily_summary.explicit_share", "Share of the 50 daily rows with is_explicit = True", "Content strategy"),
        ("Presence Ratio", "dim_track.presence_ratio", "days_on_chart / snapshots between first and last appearance", "1.0 = never left the chart"),
        ("Turnover Rate", "daily_summary.turnover_rate_pct", "(new entries + re-entries) / 50 x 100", "How much of the chart changed that day"),
        ("Longevity Class", "dim_track.longevity_class", "Flash / Short / Medium / Long / Evergreen by days_on_chart", "Segmentation of songs"),
    ], columns=["kpi", "where_to_find_it", "formula", "interpretation"])
