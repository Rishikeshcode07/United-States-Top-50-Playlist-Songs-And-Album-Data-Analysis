"""
src/survival/lives.py  -  from chart rows to "stays": how many chart days does a song spend on the chart after it arrives?

WHAT THIS FILE IS
    Three small functions that turn the chart (one row per song per snapshot day) into the two tables that every survival method
    needs.

        build_lives   one row per song: how long its first stay lasted, and whether the end of the stay was seen
        person_days   one row per chart day of a stay (the table of the day-by-day hazard model)
        drop_sizes    how many songs of the same artist arrived on the same day (an album drop has many, a single has one)

WHY WE NEED IT
    A song does not always leave the chart for good the first time it falls off. Of the songs that arrived after the first
    snapshot, 627 have one unbroken run, and the rest leave and come back: a day missing in the file, a week away and a return.
    Calling every absence an exit would count a one-day gap as the end of a song; calling nothing an exit would count a song that
    came back six months later as one long stay. The rule of this project is in between and it is written down in one place:

        a STAY ends at the last chart day before the first absence of `gap` snapshots or more (the default is 7).

    Absences shorter than that are "short breaks": the song is treated as still being on the chart, and the days of the break are
    not counted. The clock of a stay is CHART DAYS (the days the song is actually on the chart), counted from 1 on the arrival day.

    A stay is "finished" (`exit_seen`) when we saw it end: either the song came back after an absence of at least `gap`
    snapshots, or it has not been back for `gap` snapshots up to the end of the file. Otherwise the song may still be on the
    chart, or may have just stepped away for a day, so the stay is "censored": all we know is that it lasted AT LEAST this long.

WHAT HAPPENS WHEN IT RUNS
    `build_lives(chart, track_ids, n_snapshots, gap)` reads the snapshot numbers of every song and returns the table described above.
    `person_days(chart, lives)` lists the chart days of each stay with their age (1, 2, 3 ...), the place on the chart that day and
    an `event` column that is 1 on the last day of a finished stay and 0 on every other day.

EXPECTED OUTCOME
    On the real file with the default rule: 894 stays (the songs that arrived after the first snapshot), 835 of them finished and
    59 censored, and 19,404 chart days in all.

MOTIVE
    Every later number depends on this definition, so it is the part of the project that is tested most closely, with small
    songs whose answer can be worked out by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAP = 7                                  # an absence of 7 snapshots or more ends a stay
LIVES_COLUMNS = ["track_id", "entry_snapshot", "stay", "exit_seen", "last_snapshot", "short_breaks", "first_run_days", "returned"]


def snapshot_numbers(dim_date: pd.DataFrame) -> pd.DataFrame:
    """The number of every snapshot day (0 = the first one): the table (date, s) that the chart rows are matched with."""
    calendar = dim_date.loc[dim_date["has_snapshot"].astype(bool), ["date", "snapshot_index"]].copy()
    calendar["s"] = calendar["snapshot_index"].astype(int) - int(calendar["snapshot_index"].min())
    return calendar[["date", "s"]].reset_index(drop=True)


def chart_with_snapshots(fact_chart_daily: pd.DataFrame, dim_date: pd.DataFrame) -> pd.DataFrame:
    """The chart rows (track_id, s, position) with the number of the snapshot day, sorted by song and day."""
    chart = fact_chart_daily[["date", "track_id", "position"]].merge(snapshot_numbers(dim_date), on="date", how="left")
    if chart["s"].isna().any():
        raise ValueError("some chart rows are on a date without a snapshot in dim_date")
    chart["s"] = chart["s"].astype(int)
    return chart[["track_id", "s", "position"]].sort_values(["track_id", "s"], kind="mergesort").reset_index(drop=True)


def build_lives(chart: pd.DataFrame, track_ids, n_snapshots: int, gap: int = GAP) -> pd.DataFrame:
    """
    One row per song of `track_ids` with the columns of LIVES_COLUMNS:

        entry_snapshot   the snapshot of the first chart day
        stay             chart days of the first stay (1 or more)
        exit_seen        True when the end of the stay was seen, False when the stay is censored
        last_snapshot    the snapshot of the last chart day of the stay
        short_breaks     absences shorter than `gap` inside the stay
        first_run_days   chart days of the first unbroken run (the rule "any absence ends the run", the rule of Phase 5)
        returned         True when the song has chart days after its stay (it came back later)

    `chart` needs the columns track_id and s (the snapshot number); `n_snapshots` is the number of snapshots in the file.
    """
    if gap < 1:
        raise ValueError("gap must be at least 1")
    track_ids = list(track_ids)
    rows = chart[chart["track_id"].isin(set(track_ids))].sort_values(["track_id", "s"], kind="mergesort")
    out = []
    for track, group in rows.groupby("track_id", sort=False):
        s = group["s"].to_numpy(dtype=int)
        if len(s) > 1 and np.any(np.diff(s) <= 0):
            raise ValueError(f"song {track} has two chart rows on one snapshot day")
        if s[-1] >= n_snapshots:
            raise ValueError(f"song {track} has a snapshot number beyond the end of the file")
        absent = np.diff(s) - 1                                   # snapshots missing between two chart days
        long_breaks = np.flatnonzero(absent >= gap)
        if long_breaks.size:
            stay, exit_seen = int(long_breaks[0]) + 1, True       # the stay is over, and we saw it end
        else:
            stay = len(s)
            exit_seen = (n_snapshots - 1 - int(s[-1])) >= gap     # absent for `gap` snapshots up to the end of the file
        any_break = np.flatnonzero(absent > 0)
        out.append({"track_id": track, "entry_snapshot": int(s[0]), "stay": stay, "exit_seen": bool(exit_seen),
                    "last_snapshot": int(s[stay - 1]), "short_breaks": int((absent[: stay - 1] > 0).sum()),
                    "first_run_days": int(any_break[0]) + 1 if any_break.size else len(s), "returned": bool(stay < len(s))})
    lives = pd.DataFrame(out, columns=LIVES_COLUMNS)
    missing = set(track_ids) - set(lives["track_id"])
    if missing:
        raise ValueError(f"{len(missing)} song(s) have no chart rows, for example {sorted(missing)[:3]}")
    return lives.set_index("track_id").loc[track_ids].reset_index()


def person_days(chart: pd.DataFrame, lives: pd.DataFrame) -> pd.DataFrame:
    """
    One row per chart day inside a stay: track_id, s (snapshot), age (1 on the arrival day, 2 on the next chart day ...), position
    (the place on the chart that day) and event (1 on the last day of a finished stay, 0 on every other day).
    """
    rows = chart.merge(lives[["track_id", "stay", "exit_seen"]], on="track_id", how="inner").sort_values(["track_id", "s"], kind="mergesort")
    rows["age"] = rows.groupby("track_id").cumcount() + 1
    rows = rows[rows["age"] <= rows["stay"]].copy()
    rows["event"] = ((rows["age"] == rows["stay"]) & rows["exit_seen"]).astype(int)
    return rows[["track_id", "s", "age", "position", "event"]].reset_index(drop=True)


def drop_sizes(table: pd.DataFrame) -> pd.Series:
    """
    For every song: how many songs of the same primary artist arrived on the same snapshot day, the song itself included
    (1 = it arrived alone; 15 = a whole album came in on one day). `table` needs primary_artist and entry_snapshot.
    """
    return table.groupby(["primary_artist", "entry_snapshot"])["track_id"].transform("size").astype(int)
