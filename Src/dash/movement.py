"""
src/dash/movement.py  -  how songs move: up and down the chart, in and out of it.

WHAT THIS FILE IS
    day_chart            the Top 50 of one day, in order, with every song that is not in the current selection marked
    moves_between        risers, fallers, new arrivals and leavers between two days (the end day and `window` snapshots earlier)
    runs_table           one row per chart RUN (a stay without a break): entry place, best place, exit place, days to the peak ...
    rise_decline_tables  the fastest risers and the slowest decliners among the runs
    entries_table / exits_table   every arrival (new entry or re-entry) and every departure, with the place it happened at
    entry_exit_matrix    from which band of places songs arrive against from which band they leave
    chart_flow           new entries, re-entries, exits and the turnover rate day by day (chart-wide, from daily_summary)
    rank_matrix          the place of chosen songs on every date (the table behind the ranking trend lines)
    place_band           1 to 10, 11 to 20 ... 41 to 50

WHY WE NEED IT
    The brief asks for "fast risers and slow decliners", for "how songs enter and leave" and for ranking trend charts. These are
    questions about change over time, so they need the same careful rules everywhere:
      * "Positions gained" is positive when a song climbs (place 30 -> place 20 = +10).
      * A song is compared with ITSELF in the same chart run. A song that left and came back is an arrival, not a mover.
      * A run is "complete" only when we saw both its start and its end (the first 50 songs of the file and the 50 songs of the
        last day are cut off by the edges of the data and are flagged, not trusted).
      * Entries and exits of the whole chart come from daily_summary and use the dates only, because "turnover of the explicit
        songs" has no meaning.

WHAT HAPPENS WHEN IT RUNS
    Every function returns new tables; an empty selection gives empty tables with the right columns.
    `moves_between` needs the calendar (date and snapshot number) so that "7 days earlier" means "7 snapshots earlier" and the
    four calendar days without a snapshot are never invented.

EXPECTED OUTCOME
    On the whole file: 894 new entries and 1,037 re-entries (`movement` column), 1,981 runs of which 1,881 are complete runs (the 50
    runs that were already going on the first day and the 50 that are still going on the last day are cut off by the file).

MOTIVE
    One careful definition of every kind of movement, behind every page that shows one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CHART_SIZE

BAND_EDGES = [0, 10, 20, 30, 40, CHART_SIZE]
BAND_LABELS = ["1 to 10", "11 to 20", "21 to 30", "31 to 40", "41 to 50"]
MOVE_COLUMNS = ["track_id", "track_label", "primary_artist", "position_start", "position_end", "places_gained",
                "popularity_start", "popularity_end", "popularity_change", "days_in_run"]
ENTRY_COLUMNS = ["track_id", "track_label", "primary_artist", "position", "days_in_run"]
RUN_COLUMNS = ["stint_key", "track_id", "track_label", "primary_artist", "album_type", "is_explicit", "start_date", "end_date",
               "days", "start_snapshot", "end_snapshot", "entry_position", "exit_position", "best_position", "peak_snapshot",
               "days_to_peak", "days_after_peak", "began_here", "ended_here", "complete", "entry_kind", "rise_per_day",
               "fall_per_day"]


def place_band(position) -> pd.Series:
    """'1 to 10', '11 to 20', ... '41 to 50' for every place."""
    return pd.cut(pd.Series(position), bins=BAND_EDGES, labels=BAND_LABELS).astype("object")


# ====================================================================================================== one day
def day_chart(playlist: pd.DataFrame, date, keep_ids=None) -> pd.DataFrame:
    """
    The Top 50 of one date in order of place. `selected` says whether the song is in the current selection (keep_ids = the
    track ids of the selected rows; None means every song counts as selected).
    """
    day = playlist.loc[playlist["date"] == pd.Timestamp(date)].sort_values("position")
    selected = np.ones(len(day), dtype=bool) if keep_ids is None else day["track_id"].isin(set(keep_ids)).to_numpy()
    return day.assign(selected=selected).reset_index(drop=True)


def snapshot_on_or_before(calendar: pd.DataFrame, date) -> "pd.Timestamp | None":
    """The last snapshot date that is on or before `date` (a date without a snapshot is moved back to the one before it)."""
    dates = calendar["date"]
    earlier = dates[dates <= pd.Timestamp(date)]
    return None if earlier.empty else pd.Timestamp(earlier.max())


# ================================================================================================== two days
def moves_between(rows: pd.DataFrame, calendar: pd.DataFrame, end_date, window: int) -> dict:
    """
    Compare the chart on `end_date` with the chart `window` snapshots earlier, for the selected rows.
        moved     songs on the chart on both days in the SAME run; places_gained > 0 = climbed, < 0 = fell
        entered   songs on the chart at the end that were not there at the start (new songs and returning songs)
        left      songs on the chart at the start that are gone at the end
    Returns {"end_date", "start_date", "moved", "entered", "left"}; start_date is None when the window reaches before the first day.
    """
    cal = calendar.drop_duplicates("date").set_index("date")["snapshot_index"].sort_index()
    empty = {"end_date": None, "start_date": None, "moved": pd.DataFrame(columns=MOVE_COLUMNS),
             "entered": pd.DataFrame(columns=ENTRY_COLUMNS), "left": pd.DataFrame(columns=ENTRY_COLUMNS)}
    end = snapshot_on_or_before(calendar, end_date)
    if end is None or window < 1:
        return empty
    s_end = int(cal.loc[end])
    s_start = s_end - int(window)
    start_dates = cal.index[cal == s_start]
    result = dict(empty, end_date=end)
    if len(start_dates) == 0:
        return result
    result["start_date"] = pd.Timestamp(start_dates[0])
    fields = ["track_id", "stint_no", "track_label", "primary_artist", "position", "popularity_clean", "days_in_stint"]
    a = rows.loc[rows["snapshot_index"] == s_end, fields]
    b = rows.loc[rows["snapshot_index"] == s_start, fields]
    both = a.merge(b[["track_id", "stint_no", "position", "popularity_clean"]], on=["track_id", "stint_no"], suffixes=("_end", "_start"))
    both["places_gained"] = both["position_start"] - both["position_end"]
    both["popularity_change"] = both["popularity_clean_end"] - both["popularity_clean_start"]
    moved = both.rename(columns={"popularity_clean_start": "popularity_start", "popularity_clean_end": "popularity_end",
                                 "days_in_stint": "days_in_run"})[MOVE_COLUMNS]
    entered = a.merge(b[["track_id", "stint_no"]], on=["track_id", "stint_no"], how="left", indicator=True)
    entered = entered.loc[entered["_merge"] == "left_only"].rename(columns={"days_in_stint": "days_in_run"})[ENTRY_COLUMNS]
    left = b.merge(a[["track_id", "stint_no"]], on=["track_id", "stint_no"], how="left", indicator=True)
    left = left.loc[left["_merge"] == "left_only"].rename(columns={"days_in_stint": "days_in_run"})[ENTRY_COLUMNS]
    result.update(moved=moved.sort_values("places_gained", ascending=False).reset_index(drop=True),
                  entered=entered.sort_values("position").reset_index(drop=True),
                  left=left.sort_values("position").reset_index(drop=True))
    return result


# ====================================================================================================== runs
def runs_table(rows: pd.DataFrame, first_snapshot: int, last_snapshot: int) -> pd.DataFrame:
    """
    One row per chart run in the selected rows. `began_here` / `ended_here` say that the START / END of the run was seen (it is not
    cut off by the first or last snapshot of the file or by the chosen dates); `complete` = both. rise_per_day = places gained
    from entry to the peak / snapshots it took; fall_per_day = places lost from the peak to the exit / snapshots it took.
    """
    if rows.empty:
        return pd.DataFrame(columns=RUN_COLUMNS)
    r = rows.sort_values(["stint_key", "snapshot_index"])
    grouped = r.groupby("stint_key", sort=False)
    table = grouped.agg(track_id=("track_id", "first"), track_label=("track_label", "first"),
                        primary_artist=("primary_artist", "first"), album_type=("album_type", "first"),
                        is_explicit=("is_explicit", "first"), start_date=("date", "min"), end_date=("date", "max"),
                        days=("position", "size"), start_snapshot=("snapshot_index", "min"), end_snapshot=("snapshot_index", "max"),
                        entry_position=("position", "first"), exit_position=("position", "last"), best_position=("position", "min"),
                        _first_flag=("is_stint_start", "first"), _last_flag=("is_stint_end", "last"), entry_kind=("movement", "first"))
    is_best = r["position"].to_numpy() == grouped["position"].transform("min").to_numpy()
    table["peak_snapshot"] = r.loc[is_best].groupby("stint_key", sort=False)["snapshot_index"].min()
    table["days_to_peak"] = table["peak_snapshot"] - table["start_snapshot"]
    table["days_after_peak"] = table["end_snapshot"] - table["peak_snapshot"]
    table["began_here"] = table["_first_flag"].astype(bool) & (table["start_snapshot"] > first_snapshot)
    table["ended_here"] = table["_last_flag"].astype(bool) & (table["end_snapshot"] < last_snapshot)
    table["complete"] = table["began_here"] & table["ended_here"]
    table["rise_per_day"] = (table["entry_position"] - table["best_position"]) / table["days_to_peak"].where(table["days_to_peak"] > 0)
    table["fall_per_day"] = (table["exit_position"] - table["best_position"]) / table["days_after_peak"].where(table["days_after_peak"] > 0)
    return table.reset_index()[RUN_COLUMNS]


def rise_decline_tables(runs: pd.DataFrame, min_gain: int = 10, min_after_peak: int = 14, min_exit_position: int = 21) -> dict:
    """
    {"risers": ..., "decliners": ...}
      risers      runs whose start was seen, that gained at least `min_gain` places on the way to their peak, fastest first
                  (places per week = rise_per_day x 7)
      decliners   complete runs that lasted at least `min_after_peak` snapshots after the peak and ended at `min_exit_position` or
                  worse (a song that vanishes from the top 20 was removed, it did not fade), slowest fall first
    """
    if runs.empty:
        return {"risers": runs.assign(rise_per_week=np.nan), "decliners": runs.assign(fall_per_week=np.nan)}
    gain = runs["entry_position"] - runs["best_position"]
    risers = runs.loc[runs["began_here"] & (runs["days_to_peak"] >= 1) & (gain >= min_gain)].copy()
    risers["places_gained"] = (risers["entry_position"] - risers["best_position"]).astype(int)
    risers["rise_per_week"] = risers["rise_per_day"] * 7.0
    risers = risers.sort_values(["rise_per_week", "track_label"], ascending=[False, True]).reset_index(drop=True)
    fallen = runs.loc[runs["complete"] & (runs["days_after_peak"] >= min_after_peak) & (runs["exit_position"] >= min_exit_position)].copy()
    fallen["places_lost"] = (fallen["exit_position"] - fallen["best_position"]).astype(int)
    fallen["fall_per_week"] = fallen["fall_per_day"] * 7.0
    decliners = fallen.sort_values(["fall_per_week", "track_label"], ascending=[True, True]).reset_index(drop=True)
    return {"risers": risers, "decliners": decliners}


# ================================================================================================ arrivals, exits
def entries_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Every arrival (a new entry or a re-entry) in the selected rows, with the place it arrived at."""
    keep = rows["movement"].isin(["New entry", "Re-entry"])
    return rows.loc[keep, ["date", "track_id", "track_label", "primary_artist", "position", "movement", "album_type", "is_explicit"]].reset_index(drop=True)


def exits_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Every last chart day before a departure, with the place the song left from."""
    keep = rows["exits_next_snapshot"].astype(bool)
    return rows.loc[keep, ["date", "track_id", "track_label", "primary_artist", "position", "album_type", "is_explicit"]].reset_index(drop=True)


def entry_exit_matrix(runs: pd.DataFrame, value: str = "count") -> pd.DataFrame:
    """
    Complete runs: rows = the band of places the song ARRIVED at, columns = the band it LEFT from. value = "count" (number of
    runs) or "median_days" (median number of days on the chart).
    """
    grid = pd.DataFrame(0 if value == "count" else np.nan, index=BAND_LABELS, columns=BAND_LABELS, dtype="float64")
    done = runs.loc[runs["complete"]] if not runs.empty else runs
    if done.empty:
        return grid
    frame = pd.DataFrame({"arrived": place_band(done["entry_position"]).to_numpy(), "left": place_band(done["exit_position"]).to_numpy(),
                          "days": done["days"].to_numpy()})
    if value == "count":
        table = frame.groupby(["arrived", "left"]).size().unstack("left")
    else:
        table = frame.groupby(["arrived", "left"])["days"].median().unstack("left")
    grid.loc[table.index, table.columns] = table.to_numpy()
    return grid.fillna(0.0) if value == "count" else grid


# ============================================================================================ chart-wide flow
def chart_flow(summary: pd.DataFrame, smooth: int = 7) -> pd.DataFrame:
    """
    New entries, re-entries, exits and the turnover rate of every day (chart-wide), with the turnover smoothed over `smooth` days.
    `summary` is daily_summary for the chosen dates.
    """
    flow = summary[["date", "snapshot_index", "new_entries", "re_entries", "exits", "turnover_rate_pct", "is_high_turnover_day"]].sort_values("date").copy()
    flow["arrivals"] = flow["new_entries"] + flow["re_entries"]
    flow["turnover_smooth"] = flow["turnover_rate_pct"].rolling(max(int(smooth), 1), min_periods=1).mean()
    return flow.reset_index(drop=True)


def calendar_cells(summary: pd.DataFrame, value: str) -> pd.DataFrame:
    """The days as cells of a calendar: week_start (Monday), weekday (0 = Monday) and the value of the column `value`."""
    cells = summary[["date", value]].copy()
    cells["weekday"] = cells["date"].dt.day_of_week
    cells["week_start"] = cells["date"].dt.normalize() - pd.to_timedelta(cells["weekday"], unit="D")
    return cells.rename(columns={value: "value"})


# ================================================================================================== trend lines
def rank_matrix(rows: pd.DataFrame, track_ids) -> pd.DataFrame:
    """The place of the chosen songs on every date: one row per date, one column per track_id (empty where the song was not on the chart)."""
    ids = list(track_ids)
    chosen = rows.loc[rows["track_id"].isin(ids)]
    if chosen.empty:
        return pd.DataFrame(columns=ids)
    return chosen.pivot(index="date", columns="track_id", values="position").reindex(columns=[i for i in ids if i in set(chosen["track_id"])])
