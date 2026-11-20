"""
src/dash/load.py  -  read the cleaned tables and the result files, and say clearly what is missing.

WHAT THIS FILE IS
    load_tables     the six tables the dashboard needs, from data/processed/ (playlist_daily, dim_track, dim_artist, daily_summary,
                    dim_date, bridge_track_artist), bundled in one `Tables` object
    missing_tables  the names of the table files that are not there
    read_report     one result file of Phases 4 to 6C (a CSV in reports/), or None when it has not been created yet
    read_facts      the "facts" file of a phase (a JSON dictionary of every number quoted in the notebook text)
    chart_bounds    first and last date and snapshot of the file, used by the filters and by the page widgets

WHY WE NEED IT
    The dashboard must never crash with a long red error on a person's first run. If data/processed/ is empty, the app should say
    in one calm message which command creates it. If one result file of a research page is missing, only that page says so and
    all the other pages keep working. So loading is separated from drawing, and every "not there" is an ordinary value or a
    specific, friendly exception (`DataNotReady`).

WHAT HAPPENS WHEN IT RUNS
    Parquet files are read with pyarrow (listed in requirements.txt). Dates become real dates. Nothing is written to disk.

EXPECTED OUTCOME
    On the project data: playlist_daily 27,750 x 54, dim_track 944 x 47, dim_artist 283 x 17, daily_summary 555 x 15,
    dim_date 559 x 15, bridge_track_artist 1,105 x 4.

MOTIVE
    A first run that fails politely, and tests that can point the loader at a temporary folder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import default_paths

# the name used in the code  ->  the file name in data/processed/
TABLE_FILES = {
    "playlist": "playlist_daily",
    "dim_track": "dim_track",
    "dim_artist": "dim_artist",
    "daily_summary": "daily_summary",
    "dim_date": "dim_date",
    "bridge": "bridge_track_artist",
}

PIPELINE_COMMAND = "python -m src.etl.run_pipeline"


class DataNotReady(RuntimeError):
    """The cleaned tables are not in data/processed/. The message says what to run."""

    def __init__(self, missing, folder):
        self.missing = list(missing)
        self.folder = Path(folder)
        names = ", ".join(f"{name}.parquet" for name in self.missing)
        super().__init__(
            f"The dashboard needs the cleaned tables, but these files are not in {self.folder}: {names}. "
            f"Copy Atlantic_United_States.csv into data/raw/ and run `{PIPELINE_COMMAND}` once; then start the dashboard again."
        )


@dataclass(frozen=True)
class Tables:
    """The six tables of the dashboard. They are shared by every page: read them, never change them in place."""
    playlist: pd.DataFrame
    dim_track: pd.DataFrame
    dim_artist: pd.DataFrame
    daily_summary: pd.DataFrame
    dim_date: pd.DataFrame
    bridge: pd.DataFrame


@dataclass(frozen=True)
class Bounds:
    """First and last date and snapshot of the file (the edges of every filter)."""
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    first_snapshot: int
    last_snapshot: int
    snapshots: int


def _folder(folder, kind):
    paths = default_paths()
    if folder is not None:
        return Path(folder)
    return paths.processed_dir if kind == "processed" else paths.reports_dir


def missing_tables(folder=None) -> list:
    """The table names (file stems) that are not in the processed folder."""
    root = _folder(folder, "processed")
    return [stem for stem in TABLE_FILES.values() if not (root / f"{stem}.parquet").exists()]


def load_tables(folder=None) -> Tables:
    """Read the six tables. Raises DataNotReady (with the command to run) when a file is missing."""
    root = _folder(folder, "processed")
    missing = missing_tables(root)
    if missing:
        raise DataNotReady(missing, root)
    frames = {name: pd.read_parquet(root / f"{stem}.parquet") for name, stem in TABLE_FILES.items()}
    frames["playlist"] = frames["playlist"].sort_values(["date", "position"]).reset_index(drop=True)
    return Tables(**frames)


def chart_bounds(playlist: pd.DataFrame) -> Bounds:
    """First and last date and snapshot of the playlist table."""
    return Bounds(first_date=pd.Timestamp(playlist["date"].min()), last_date=pd.Timestamp(playlist["date"].max()),
                  first_snapshot=int(playlist["snapshot_index"].min()), last_snapshot=int(playlist["snapshot_index"].max()),
                  snapshots=int(playlist["date"].nunique()))


def read_report(name: str, folder=None, dates=()) -> "pd.DataFrame | None":
    """
    One result file, reports/<name>.csv, or None when it does not exist (or cannot be read).
    `dates` lists the columns that hold dates, so they come back as real dates.
    """
    path = _folder(folder, "reports") / f"{name}.csv"
    if not path.exists():
        return None
    try:
        table = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    for column in dates:
        if column in table.columns:
            table[column] = pd.to_datetime(table[column], errors="coerce")
    return table


def read_facts(name: str, folder=None) -> dict:
    """The facts file of a phase, reports/<name>.json (for example "p6c_facts"), or an empty dictionary when it is not there."""
    path = _folder(folder, "reports") / f"{name}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def reports_available(names, folder=None) -> dict:
    """{name: True/False}: which of the result files exist. Used by the 'Data and methods' page."""
    root = _folder(folder, "reports")
    return {name: (root / f"{name}.csv").exists() or (root / f"{name}.json").exists() for name in names}
