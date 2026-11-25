"""
src/cluster/paths.py  -  the first month of every song as one row of numbers.

WHAT THIS FILE IS
    first_month_matrix   (songs x days) table of the CHART SCORE of a song on each of its first `horizon` days
    chart_score          51 - place on the chart (place 1 = score 50, place 50 = score 1), and 0 for a day on which the song is not on the chart

WHY WE NEED IT
    To group songs by the SHAPE of their first month we need every song as the same list of numbers. The chart score turns "high on
    the chart" into "a big number" and "off the chart" into zero, so the distance between two songs is large when one is high and the
    other is away, and small when both are away. Day 1 is the arrival day and day d is d - 1 snapshot days later.

WHAT HAPPENS WHEN IT RUNS
    Only songs whose whole first month is inside the file are kept (a song that arrived in the last `horizon - 1` snapshot days has an
    unfinished month and would look like a song that left).

EXPECTED OUTCOME
    On the real file: 832 songs x 30 days; the first column is never 0 (the arrival day is a chart day).

MOTIVE
    One row per song, the same columns for every song: the table every grouping method needs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOP = 50


def chart_score(position: np.ndarray, top: int = TOP) -> np.ndarray:
    """51 - place (place 1 = 50 ... place 50 = 1)."""
    return (top + 1) - np.asarray(position)


def first_month_matrix(chart: pd.DataFrame, songs: pd.DataFrame, n_snapshots: int, horizon: int = 30, top: int = TOP):
    """
    chart: one row per song per snapshot day (track_id, s, position); songs: track_id and entry_snapshot (the arrival day of every song).
    Returns (matrix, kept) where kept is a boolean array over `songs` (True = the whole first month is inside the file) and
    matrix has one row per kept song, in the order of `songs`, and `horizon` columns.
    """
    entry = songs["entry_snapshot"].to_numpy()
    kept = entry + horizon <= n_snapshots
    kept_songs = songs.loc[kept, ["track_id", "entry_snapshot"]].reset_index(drop=True)
    row_of = pd.Series(np.arange(len(kept_songs)), index=kept_songs["track_id"].to_numpy())
    rows = chart[chart["track_id"].isin(row_of.index)]
    day = rows["s"].to_numpy() - kept_songs["entry_snapshot"].to_numpy()[row_of.reindex(rows["track_id"]).to_numpy()]
    inside = (day >= 0) & (day < horizon)
    matrix = np.zeros((len(kept_songs), horizon))
    matrix[row_of.reindex(rows["track_id"]).to_numpy()[inside], day[inside]] = chart_score(rows["position"].to_numpy()[inside], top)
    return matrix, kept
