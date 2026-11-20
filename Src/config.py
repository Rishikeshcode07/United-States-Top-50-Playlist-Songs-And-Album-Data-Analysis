"""
src/config.py  -  every folder, file name and constant of the project in ONE place.

WHAT THIS FILE IS
    The "settings" file. It lists where the data lives, what the raw file must look like (the data
    contract) and every fixed number the pipeline uses (chart size, bins, protected artist names ...).

WHY WE NEED IT
    If the same number (for example 50 = size of the chart) were typed in ten files, one of them would be
    forgotten the day something changes. Here every value lives in exactly one place.

WHAT HAPPENS WHEN IT RUNS
    Nothing runs. Other files do `from src.config import ...` and read these values.

EXPECTED OUTCOME
    No output. A typo here makes the other files fail immediately and loudly, which is what we want.

MOTIVE
    One source of truth, so the pipeline, the tests and the notebooks can never disagree about a setting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PIPELINE_VERSION = "1.0.0"

# ------------------------------------------------------------------------------------------ folders
# This file sits in <project>/src/config.py, so the project root is one level above "src".
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                  # the original CSV: never edited
INTERIM_DIR = DATA_DIR / "interim"          # checkpoints between stages
PROCESSED_DIR = DATA_DIR / "processed"      # analysis-ready tables
WAREHOUSE_DIR = DATA_DIR / "warehouse"      # the SQLite database file lives here
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
SQL_DIR = PROJECT_ROOT / "sql"

RAW_FILE = RAW_DIR / "Atlantic_United_States.csv"
SQLITE_FILE = WAREHOUSE_DIR / "atlantic_top50.db"

@dataclass(frozen=True)
class ProjectPaths:
    """
    Every folder and file the pipeline touches, bundled in one object.

    Normal runs use `default_paths()`. The automatic tests build a throw-away copy with `ProjectPaths.under(tmp_folder)`
    so that testing never touches your real data folders.
    """
    root: Path
    raw_file: Path
    interim_dir: Path
    processed_dir: Path
    warehouse_dir: Path
    reports_dir: Path
    logs_dir: Path
    sql_dir: Path

    @property
    def sqlite_file(self) -> Path:
        return self.warehouse_dir / SQLITE_FILE.name

    @classmethod
    def under(cls, root, raw_file=None, sql_dir=None):
        """All folders below `root`. The SQL scripts are shipped code, so they default to the project's own sql folder."""
        root = Path(root)
        return cls(root=root,
                   raw_file=Path(raw_file) if raw_file else root / "data" / "raw" / RAW_FILE.name,
                   interim_dir=root / "data" / "interim", processed_dir=root / "data" / "processed",
                   warehouse_dir=root / "data" / "warehouse", reports_dir=root / "reports", logs_dir=root / "logs",
                   sql_dir=Path(sql_dir) if sql_dir else SQL_DIR)


def default_paths():
    """The folders of this project (what you get when you simply run the pipeline)."""
    return ProjectPaths(root=PROJECT_ROOT, raw_file=RAW_FILE, interim_dir=INTERIM_DIR, processed_dir=PROCESSED_DIR,
                        warehouse_dir=WAREHOUSE_DIR, reports_dir=REPORTS_DIR, logs_dir=LOGS_DIR, sql_dir=SQL_DIR)


# ------------------------------------------------------------------------------------ data contract
# Columns promised by the project brief (name -> meaning)
EXPECTED_COLUMNS = {
    "date":            "Date of the playlist snapshot",
    "position":        "Playlist rank (1-50)",
    "song":            "Song title",
    "artist":          "Artist(s)",
    "popularity":      "Popularity score from the Atlantic API",
    "duration_ms":     "Song duration in milliseconds",
    "album_type":      "Single / Album",
    "total_tracks":    "Number of tracks in the album",
    "is_explicit":     "Explicit content flag",
    "album_cover_url": "Album artwork URL",
}

DATE_FORMAT = "%d-%m-%Y"          # the CSV writes dates like 18-05-2024 (day-month-year)
CHART_SIZE = 50                   # a Top 50 playlist has exactly 50 rows per day
POPULARITY_RANGE = (0, 100)
DURATION_RANGE_MS = (30_000, 20 * 60_000)        # 30 seconds ... 20 minutes
ALLOWED_ALBUM_TYPES = {"single", "album"}        # exactly what the brief lists ("compilation" will raise a WARN)

# ---------------------------------------------------------------------------------- text standardization
SEPARATOR = " & "                 # the data joins collaborating artists with " & " (never with a comma)
PROTECTED_NAMES = [               # real act names that contain " & " and must NOT be split
    "Earth, Wind & Fire",
    "Richy Mitch & The Coal Miners",
    'Bobby "Boris" Pickett & The Crypt-Kickers',
]

# --------------------------------------------------------------------------------------- feature settings
ROLLING_WINDOW = 7                # days used for the Popularity Trend Score and the rank smoothing (the column names say "7d")
VOLATILITY_MIN_DAYS = 3           # a 7-day volatility needs at least this many days of history
TURNOVER_FENCE_MULTIPLIER = 3     # Tukey "far-out" fence: Q3 + 3 x IQR marks a shock day
RANK_TIER_BINS = [0, 10, 20, 50]
RANK_TIER_LABELS = ["Top 10", "11-20", "21-50"]
SIZE_BINS = [0, 1, 3, 7, 15, 40, 1000]
SIZE_LABELS = ["1 track", "2-3", "4-7", "8-15", "16-40", "41+"]
LONGEVITY_BINS = [0, 3, 14, 60, 180, float("inf")]
LONGEVITY_LABELS = ["Flash (1-3 days)", "Short (4-14)", "Medium (15-60)", "Long (61-180)", "Evergreen (181+)"]

# ------------------------------------------------------------------------------------ reference data
# (song title starts with, artist name contains, real-world release date) - typed from public knowledge.
# Used ONLY to test whether the calendar of the file lines up with the real world (clean.calendar_reality_check).
KNOWN_RELEASES = [
    ("Beautiful Things",       "benson boone",      "2024-01-18"),
    ("Fortnight",              "taylor swift",      "2024-04-19"),
    ("Espresso",               "sabrina carpenter", "2024-04-12"),
    ("Not Like Us",            "kendrick lamar",    "2024-05-04"),
    ("Please Please Please",   "sabrina carpenter", "2024-06-06"),
    ("Houdini",                "eminem",            "2024-05-31"),
    ("Good Luck, Babe!",       "chappell roan",     "2024-04-05"),
    ("A Bar Song (Tipsy)",     "shaboozey",         "2024-04-12"),
    ("Die With A Smile",       "lady gaga",         "2024-08-16"),
    ("APT.",                   "bruno mars",        "2024-10-18"),
    ("Luther",                 "kendrick lamar",    "2024-11-22"),
    ("Too Sweet",              "hozier",            "2024-03-22"),
    ("Texas Hold",             "beyonc",            "2024-02-11"),
    ("Yes, And?",              "ariana grande",     "2024-01-12"),
    ("greedy",                 "tate mcrae",        "2023-09-15"),
    ("Lovin On Me",            "jack harlow",       "2023-11-10"),
    ("Paint The Town Red",     "doja cat",          "2023-08-04"),
    ("FE!N",                   "travis scott",      "2023-07-28"),
    ("Lunch",                  "billie eilish",     "2024-05-17"),
    ("Taste",                  "sabrina carpenter", "2024-08-23"),
]

# --------------------------------------------------------------------------------------- output tables
# The exact columns (and their order) of every table the pipeline writes. These lists are the OUTPUT
# CONTRACT: the SQL schema in sql/01_schema.sql and the tests are checked against them.
TABLE_COLUMNS = {
    "playlist_daily": [
        # when
        "date", "snapshot_index", "days_since_prev_snapshot", "year", "month", "month_name", "year_month", "quarter",
        "iso_week", "day_of_week", "day_name", "is_weekend",
        # which song
        "track_id", "track_label", "song_clean", "artist_credit", "primary_artist", "artist_count", "artists_all", "is_collab",
        # chart position
        "position", "rank_tier", "in_top10", "in_top20", "is_number_one", "chart_points",
        # popularity
        "popularity", "popularity_clean", "popularity_was_zero", "popularity_fill_method", "popularity_trend_7d",
        # rank movement
        "prev_position", "rank_change", "movement", "rank_gain_7d", "avg_rank_7d", "rank_volatility_7d",
        # life on the chart
        "stint_no", "stint_key", "days_in_stint", "stint_length", "is_stint_start", "is_stint_end", "exits_next_snapshot",
        "chart_day_number", "calendar_age_days",
        # song attributes as they were on that day
        "duration_ms", "duration_min", "album_type", "release_format", "total_tracks", "release_size_band", "is_explicit",
        "album_cover_url",
    ],
    "fact_chart_daily": [
        "date", "track_id", "position", "chart_points", "popularity_clean", "popularity_trend_7d", "popularity_fill_method",
        "rank_change", "rank_gain_7d", "avg_rank_7d", "rank_volatility_7d", "movement", "rank_tier", "in_top10", "in_top20",
        "is_number_one", "chart_day_number", "days_in_stint", "is_stint_start", "is_stint_end", "is_explicit",
    ],
    "dim_track": [
        "track_id", "track_label", "song_clean", "artist_credit", "primary_artist", "artist_count", "is_collab", "artists_all",
        "days_on_chart", "first_seen", "last_seen", "span_days", "span_snapshots", "presence_ratio", "n_runs", "longest_run_days",
        "avg_rank", "median_rank", "best_rank", "worst_rank", "rank_volatility_index", "avg_daily_rank_move",
        "days_in_top10", "days_in_top20", "days_at_number_one", "chart_points", "entry_rank", "exit_rank", "peak_date",
        "days_to_peak", "avg_popularity", "peak_popularity", "popularity_std",
        "on_chart_at_window_start", "on_chart_at_window_end", "run_status", "longevity_class", "reached_top10",
        "reached_number_one", "duration_ms", "duration_min", "album_type", "release_format", "total_tracks",
        "release_size_band", "is_explicit", "album_cover_url",
    ],
    "dim_artist": [
        "artist_name", "unique_tracks", "unique_tracks_as_primary", "chart_slot_days", "chart_slot_days_primary", "days_present",
        "first_seen", "last_seen", "avg_rank", "best_rank", "days_in_top10", "days_at_number_one", "chart_points",
        "chart_points_primary", "dominance_index_primary", "dominance_index_all_credits", "share_of_slots_primary_pct",
    ],
    "bridge_track_artist": ["track_id", "artist_name", "artist_order", "is_primary"],
    "dim_date": [
        "date", "date_key", "has_snapshot", "snapshot_index", "days_since_prev_snapshot", "year", "month", "month_name",
        "year_month", "quarter", "iso_week", "day_of_week", "day_name", "is_weekend", "season",
    ],
    "daily_summary": [
        "date", "snapshot_index", "unique_tracks", "new_entries", "re_entries", "exits", "turnover_rate_pct",
        "is_high_turnover_day", "avg_popularity", "median_popularity", "explicit_share", "collab_share", "single_share",
        "avg_duration_min", "unique_primary_artists",
    ],
}

# Tables that go into the database (the star schema). `playlist_daily` is the wide analysis table and stays in files.
# The ORDER matters: parents first, so that foreign keys are satisfied while loading.
DB_TABLE_ORDER = ["dim_date", "dim_track", "dim_artist", "bridge_track_artist", "fact_chart_daily", "daily_summary"]

# ----------------------------------------------------------------------------------------- run settings
