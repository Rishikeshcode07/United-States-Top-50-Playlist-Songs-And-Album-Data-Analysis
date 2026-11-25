"""
src/eda/ranking.py  -  the "life" of a song on the chart: chart runs (stints), position-to-position moves and decline speed.

WHAT THIS FILE IS
    Three tested building blocks for the ranking analysis of Phase 4:

        stint_table(playlist)            one row per CHART RUN ("stint": the days in a row that a song stayed on the chart),
                                         with entry position, exit position, best position, days to the peak, days after the
                                         peak and two flags that say whether the start or the end of the run was cut off by
                                         the edge of our observation window
        position_transitions(playlist)   how many songs moved from position A today to position B at the next snapshot (or
                                         left the chart): the raw material of the transition heat map
        decline_speed(stints)            positions lost per week between the peak and the last day on the chart, for runs that
                                         faded out gradually (the "slow decliners" of the brief)

WHY WE NEED IT
    Averages hide the story of a single song. A song that enters at 45, climbs to 3 and fades for two months has a very
    different life from a song that debuts at 3 and is gone in four days, even if both have the same average rank. A run
    table is the smallest structure that keeps this life together. The flags matter too: a run that begins on the very first
    day of the file really began earlier, and a run that is still alive on the last day has not finished yet. Treating those
    runs as complete would make songs look shorter-lived than they are.

WHAT HAPPENS WHEN IT RUNS
    Nothing by itself; the notebook imports the functions. Day counts are counts of SNAPSHOTS (`snapshot_index`), the same
    convention the pipeline uses for "was on the previous snapshot". The file has 555 snapshots over 559 calendar days
    (four calendar days have no snapshot), so one snapshot step is one day except across those four gaps.

EXPECTED OUTCOME
    On the real file: 1,981 runs of 944 songs. Exactly 50 are cut off at the start (the 50 songs on the first day), exactly
    50 are cut off at the end (the 50 songs on the last day), 1,881 are complete.

MOTIVE
    Every longevity statement in Phase 4 says whether it uses complete runs only or also counts the cut-off ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CHART_SIZE = 50

STINT_REQUIRED_COLUMNS = ["stint_key", "track_id", "track_label", "snapshot_index", "date", "position"]


# =============================================================================================== chart runs (stints)
def stint_table(playlist):
    """
    One row per chart run. Columns:

        stint_key, track_id, track_label            who and which run
        start_snapshot, end_snapshot, start_date, end_date, length     when, and how many snapshots in a row
        entry_position, exit_position, best_position                  where the run began, where it ended, its best place
        peak_snapshot                                                  the FIRST snapshot on which the best place was reached
        days_to_peak, days_after_peak                                  snapshots from the start to the peak, from the peak to the end
        left_censored    the run starts on the first snapshot of the file (it really began earlier)
        right_censored   the run ends on the last snapshot of the file (it has not finished yet)
        complete         neither of the two: the whole run is inside the window
    """
    missing = [c for c in STINT_REQUIRED_COLUMNS if c not in playlist.columns]
    if missing:
        raise ValueError(f"the playlist table has no column(s) {missing}")

    frame = playlist.sort_values(["stint_key", "snapshot_index"])
    first_snapshot = int(frame["snapshot_index"].min())
    last_snapshot = int(frame["snapshot_index"].max())
    grouped = frame.groupby("stint_key", sort=False)

    stints = grouped.agg(
        track_id=("track_id", "first"), track_label=("track_label", "first"),
        start_snapshot=("snapshot_index", "min"), end_snapshot=("snapshot_index", "max"),
        start_date=("date", "min"), end_date=("date", "max"), length=("snapshot_index", "size"),
        entry_position=("position", "first"), exit_position=("position", "last"), best_position=("position", "min"),
    )

    not_in_a_row = stints["length"] != (stints["end_snapshot"] - stints["start_snapshot"] + 1)
    if not_in_a_row.any():
        raise ValueError(f"{int(not_in_a_row.sum())} run(s) have a gap inside them, so they are not runs of consecutive snapshots")

    is_best = frame["position"].to_numpy() == grouped["position"].transform("min").to_numpy()
    stints["peak_snapshot"] = frame.loc[is_best].groupby("stint_key", sort=False)["snapshot_index"].min()
    stints["days_to_peak"] = stints["peak_snapshot"] - stints["start_snapshot"]
    stints["days_after_peak"] = stints["end_snapshot"] - stints["peak_snapshot"]
    stints["left_censored"] = stints["start_snapshot"] == first_snapshot
    stints["right_censored"] = stints["end_snapshot"] == last_snapshot
    stints["complete"] = ~(stints["left_censored"] | stints["right_censored"])

    return stints.reset_index().sort_values(["track_id", "start_snapshot"]).reset_index(drop=True)


def decline_speed(stints, min_days_after_peak=28, min_exit_position=21):
    """
    How fast did a song lose its places after its peak? Positions lost per week = (exit position - best position) / days after
    the peak x 7. Only COMPLETE runs count, and only runs that

        * lasted at least `min_days_after_peak` snapshots after the peak (a rate over three days is just noise), and
        * ended at `min_exit_position` or worse. A song that vanishes while it is still in the top 20 did not fade, it was
          removed (the Christmas songs after the holidays are the typical case) and is a different story.

    Returns the matching rows of `stints` plus the column `positions_lost_per_week`, slowest decliner first.
    """
    eligible = stints[stints["complete"]
                      & (stints["days_after_peak"] >= min_days_after_peak)
                      & (stints["exit_position"] >= min_exit_position)].copy()
    eligible["positions_lost_per_week"] = ((eligible["exit_position"] - eligible["best_position"])
                                           / eligible["days_after_peak"] * 7.0)
    return eligible.sort_values(["positions_lost_per_week", "stint_key"]).reset_index(drop=True)


# =============================================================================================== position transitions
@dataclass(frozen=True)
class PositionTransitions:
    """
    counts      50 x 50 table of whole numbers. Row = position at one snapshot, column = position of the same song at the
                NEXT snapshot. Songs that left the chart are not in this table, they are in `left_chart`.
    left_chart  for every position (1..50) the number of songs that were NOT on the chart at the next snapshot.
    """
    counts: pd.DataFrame
    left_chart: pd.Series

    @property
    def n_today(self):
        """How many songs stood on each position (only snapshots that have a following snapshot are counted)."""
        return self.counts.sum(axis=1) + self.left_chart

    def probabilities(self):
        """Counts divided by the number of songs on that position: every row of the 50 x 50 table plus `left_chart` adds to 1."""
        return self.counts.div(self.n_today, axis=0)

    def exit_rate(self):
        """Share of the songs of each position that were off the chart at the next snapshot."""
        return self.left_chart / self.n_today

    def stay_rate(self):
        """Share of the songs of each position that were on the SAME position at the next snapshot."""
        diagonal = pd.Series(np.diag(self.counts.to_numpy()), index=self.counts.index)
        return diagonal / self.n_today


def position_transitions(playlist, chart_size=CHART_SIZE):
    """
    Count the moves from one snapshot to the next. The LAST snapshot is skipped because nothing follows it (counting it
    would make every song look as if it had left the chart).
    """
    today = playlist[["snapshot_index", "track_id", "position"]]
    tomorrow = today.assign(snapshot_index=today["snapshot_index"] - 1).rename(columns={"position": "next_position"})
    pairs = today.merge(tomorrow, on=["snapshot_index", "track_id"], how="left", validate="one_to_one")
    pairs = pairs[pairs["snapshot_index"] < today["snapshot_index"].max()]

    positions = list(range(1, chart_size + 1))
    on_chart = pairs.dropna(subset=["next_position"])
    counts = (pd.crosstab(on_chart["position"], on_chart["next_position"].astype(int))
              .reindex(index=positions, columns=positions, fill_value=0).astype(int))
    counts.index.name, counts.columns.name = "position", "next_position"

    left = pairs[pairs["next_position"].isna()].groupby("position").size()
    left_chart = left.reindex(positions, fill_value=0).astype(int).rename("left_chart")
    return PositionTransitions(counts=counts, left_chart=left_chart)
