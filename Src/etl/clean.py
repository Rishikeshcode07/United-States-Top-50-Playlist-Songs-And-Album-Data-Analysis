"""
src/etl/clean.py  -  step 4 of the pipeline: repair the data and add the "life on the chart" columns.

WHAT THIS FILE IS
    The cleaning logic of Phase 2 (Steps 1.6 and 2.1 - 2.6) as functions. Every function does ONE job and writes ONE
    line into the cleaning log, so that anybody can read what was changed, how many rows were touched, and why.

WHY WE NEED IT
    The validated table still has known problems: a date with two snapshots, popularity scores of 0 that really mean
    "not scored yet", song titles and credits that differ slightly from day to day, and no notion of time on the chart.
    Each problem has a decision behind it. Decisions that are not written down are decisions nobody can defend.

WHAT HAPPENS WHEN IT RUNS
    `apply_cleaning(validated, clog)` runs these steps in this order:
        1.6  calendar reality check (only measures, changes nothing)
        2.1  keep the latest block of a date that has two snapshots
        2.2  number the snapshots (1, 2, 3 ...) and measure the gap between them; missing days are never invented
        2.3  cut every song's history into runs ("stints"): consecutive snapshots on the chart = one run
        2.4  one title, one fullest artist credit and one readable label per song
        2.5  repair popularity zeros: interpolate inside the run, else use the song's median, else leave empty
        2.6  add release_format (Single / Album) and release_size_band

EXPECTED OUTCOME
    The project file goes from 27,800 to 27,750 rows (50 superseded rows removed), gets 21 new columns,
    and the cleaning log has 7 rows (the 8th, step 3.x, is added by build_features.py).

MOTIVE
    Repair what is clearly broken, keep the original column next to every repaired one, never invent rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CHART_SIZE, KNOWN_RELEASES, SIZE_BINS, SIZE_LABELS
from src.etl.errors import DataContractError, DataIntegrityError
from src.etl.numeric_utils import round_half_up
from src.logger import get_logger

log = get_logger(__name__)

LOG_COLUMNS = ["step", "rule", "rows_before", "rows_after", "rows_affected", "note"]


class CleaningLog:
    """The decision log: one row per cleaning action."""

    def __init__(self):
        self._rows = []

    def add(self, step, rule, rows_before, rows_after, affected, note=""):
        """Record what we did, how many rows went in and out, and how many rows the action touched."""
        self._rows = [r for r in self._rows if r["step"] != step]            # a re-run replaces the old entry
        self._rows.append({"step": step, "rule": rule, "rows_before": int(rows_before),
                           "rows_after": int(rows_after), "rows_affected": int(affected), "note": note})
        log.info("[Step %s] %s: rows %s -> %s | rows affected: %s", step, rule,
                 f"{int(rows_before):,}", f"{int(rows_after):,}", f"{int(affected):,}")

    def to_frame(self):
        return pd.DataFrame(self._rows, columns=LOG_COLUMNS)


def apply_cleaning(validated, clog):
    """Run all cleaning steps. Returns (clean_table, calendar_reality_table)."""
    df = validated.copy()
    reality = calendar_reality_check(df, clog)             # step 1.6: measure only
    superseded_dates = (df.loc[df["flag_superseded_snapshot"], "date"].dt.strftime("%Y-%m-%d").unique().tolist())
    df = _keep_latest_snapshot(df, clog, superseded_dates)
    df = _add_snapshot_index(df, clog)
    df = _add_chart_runs(df, clog)
    df = _unify_titles_and_credits(df, clog)
    df = _repair_popularity(df, clog)
    df = _add_release_format(df, clog)
    return df, reality


# ====================================================================================== step 1.6
def calendar_reality_check(df, clog):
    """
    Does the calendar of the file match the real world? We look up songs whose real release date we know
    and compare it with the first day the song shows up in the data. A tiny gap would mean "calendar is right";
    a gap of about one year means the dates in the file are shifted.
    """
    first_seen = df.groupby("track_id").agg(song=("song_clean", "first"), credit=("track_credit", "first"),
                                            first_seen=("date", "min"), days=("date", "size"))
    rows = []
    for title, artist, released in KNOWN_RELEASES:
        hit = first_seen[first_seen["song"].str.lower().str.startswith(title.lower())
                         & first_seen["credit"].str.lower().str.contains(artist, regex=False)]
        if hit.empty:
            continue
        best = hit.sort_values("days", ascending=False, kind="stable").iloc[0]   # several matches: the longest-charting one
        rows.append({"song": title, "real_release": pd.Timestamp(released), "first_seen_in_data": best["first_seen"]})

    rows_loaded = len(df)
    if not rows:                                                                    # a different dataset: nothing to compare
        clog.add("1.6", "Calendar kept exactly as provided; no known song matched, offset not assessed",
                 rows_loaded, rows_loaded, 0, note="None of the reference songs was found in the data")
        return pd.DataFrame(columns=["song", "real_release", "first_seen_in_data", "gap_days"])

    reality = pd.DataFrame(rows)
    reality["gap_days"] = (reality["first_seen_in_data"] - reality["real_release"]).dt.days
    reality["real_release"] = reality["real_release"].dt.date
    reality["first_seen_in_data"] = reality["first_seen_in_data"].dt.date
    median_gap = float(reality["gap_days"].median())
    if 360 <= median_gap <= 380:
        rule = "Calendar kept exactly as provided; evidence suggests it is one year ahead of real-world dates"
    else:
        rule = "Calendar kept exactly as provided; no one-year offset found against known release dates"
    clog.add("1.6", rule, rows_loaded, rows_loaded, 0,
             note=f"Median gap between real release and first chart appearance is about {median_gap:.0f} days; "
                  "relative analysis is unaffected")
    return reality


# ====================================================================================== step 2.1
def _keep_latest_snapshot(df, clog, superseded_dates):
    """A date captured twice keeps only its latest block. Afterwards every date must hold exactly 50 distinct rows."""
    n_before = len(df)
    df = df[~df["flag_superseded_snapshot"]].copy()                          # keep only the latest block of each date
    df = df.drop(columns=["snapshot_seq", "flag_superseded_snapshot"])       # helper columns no longer needed

    rows_per_date = df.groupby("date").size()
    broken_days = rows_per_date[rows_per_date != CHART_SIZE]
    if len(broken_days):
        sample = {d.strftime("%Y-%m-%d"): int(n) for d, n in broken_days.head(5).items()}
        raise DataContractError(f"{len(broken_days)} date(s) do not hold exactly {CHART_SIZE} rows after keeping the "
                                f"latest snapshot, for example {sample}. Fix or remove those dates in the raw file.")
    if df.duplicated(["date", "position"]).any():
        raise DataContractError("A (date, position) pair is repeated after keeping the latest snapshot.")
    if df.duplicated(["date", "track_id"]).any():
        raise DataContractError("A song appears twice on the same date after keeping the latest snapshot.")

    where = f"({', '.join(superseded_dates)})" if superseded_dates else "(none found in this file)"
    clog.add("2.1", f"Keep the latest block of a double-snapshot date {where}", n_before, len(df), n_before - len(df),
             note="Decision #2 of the Phase 1 log")
    return df


# ====================================================================================== step 2.2
def _add_snapshot_index(df, clog):
    """Count time in SNAPSHOTS (1, 2, 3 ...). A missing calendar day gets no number: we never invent a snapshot."""
    df["snapshot_index"] = df["date"].rank(method="dense").astype(int)         # 1 for the first date, 2 for the next, ...
    snapshots = (df[["date", "snapshot_index"]].drop_duplicates()
                   .sort_values("snapshot_index").reset_index(drop=True))
    snapshots["days_since_prev_snapshot"] = snapshots["date"].diff().dt.days   # empty for the very first snapshot
    df = df.merge(snapshots[["date", "days_since_prev_snapshot"]], on="date", how="left")
    missing_days = int((snapshots["days_since_prev_snapshot"].dropna() - 1).sum())
    clog.add("2.2", "Added snapshot_index and days_since_prev_snapshot (time counted in snapshots, no invented rows)",
             len(df), len(df), 0, note=f"Decision #3: {missing_days} missing calendar days are kept as gaps, never filled")
    return df


# ====================================================================================== step 2.3
def _add_chart_runs(df, clog):
    """
    Cut each song's history into RUNS ("stints"). A run is a streak of consecutive snapshots on the chart.
    If the song disappears for even one snapshot, the next appearance starts run number 2.
    """
    df = df.sort_values(["track_id", "date"]).reset_index(drop=True)          # all later steps rely on this order
    distance = df.groupby("track_id")["snapshot_index"].diff()               # snapshots since this song's previous appearance
    starts_new_run = distance.isna() | (distance != 1)                       # first appearance, or a break in the streak
    df["stint_no"] = starts_new_run.astype(int).groupby(df["track_id"]).cumsum()               # run number: 1, 2, 3 ... per song
    df["stint_key"] = df["track_id"].astype(str) + "-" + df["stint_no"].astype(str)            # unique id of one run
    df["days_in_stint"] = df.groupby("stint_key").cumcount() + 1                               # day 1, 2, 3 ... inside the run
    df["stint_length"] = df.groupby("stint_key")["date"].transform("size")                     # total length of that run
    df["is_stint_start"] = df["days_in_stint"] == 1
    df["is_stint_end"] = df["days_in_stint"] == df["stint_length"]
    clog.add("2.3", "Added chart-run columns (stint_no, stint_key, days_in_stint, stint_length, is_stint_start, is_stint_end)",
             len(df), len(df), 0, note="A run = consecutive snapshots; any absence starts a new run")
    return df


# ====================================================================================== step 2.4
def _unify_titles_and_credits(df, clog):
    """One song = one title spelling, one (fullest) artist credit, one readable label."""
    titles_before = df.groupby("track_id")["song_clean"].nunique()
    df["song_clean"] = df.groupby("track_id")["song_clean"].transform(lambda s: s.mode().iloc[0])

    # For every song keep the credit with the most artists (ties: more days, then the most recent one).
    credits = (df.groupby(["track_id", "artists_all"])
                 .agg(artist_credit=("artist_clean", "first"), artist_count=("artist_count", "first"),
                      days=("date", "size"), last_seen=("date", "max"))
                 .reset_index())
    credits_per_song = credits.groupby("track_id").size()
    fullest = (credits.sort_values(["track_id", "artist_count", "days", "last_seen", "artists_all"],
                                   ascending=[True, False, False, False, True])
                      .drop_duplicates("track_id").set_index("track_id"))

    df = df.drop(columns=["artist_clean", "primary_artist", "artist_count", "artists_all", "track_credit"])
    df["artist_credit"] = df["track_id"].map(fullest["artist_credit"])
    df["artists_all"] = df["track_id"].map(fullest["artists_all"])
    df["artist_count"] = df["track_id"].map(fullest["artist_count"]).astype(int)
    df["primary_artist"] = df["artists_all"].str.split(" | ", regex=False).str[0]          # the first (lead) artist
    df["track_label"] = df["song_clean"] + " by " + df["artist_credit"]                    # e.g. "Last Night by Morgan Wallen"

    if df.groupby("track_label")["track_id"].nunique().max() != 1:
        raise DataIntegrityError("Two different songs share one track_label; the label would not be a safe display key.")
    if df.groupby("track_id")["artist_credit"].nunique().max() != 1:
        raise DataIntegrityError("A song still carries more than one artist credit after unification.")

    affected = int(df["track_id"].isin(credits_per_song[credits_per_song > 1].index).sum())
    log.debug("Songs with more than one title spelling before: %s", int((titles_before > 1).sum()))
    clog.add("2.4", "One title, one fullest artist credit and one label per song", len(df), len(df), affected,
             note="Rows whose song had more than one credit before the unification")
    return df


# ====================================================================================== step 2.5
def _repair_popularity(df, clog):
    """
    A popularity score of 0 for a song inside the Top 50 is not real: it is a "not scored yet" placeholder,
    and it is concentrated on a song's first chart day. We therefore treat 0 as MISSING and repair it in two levels:
        level 1: straight-line interpolation between the nearest real scores of the SAME run
        level 2: the median of the song's real scores (if the run has no real score at all)
        else   : leave it empty ("unfilled"), because inventing a number would be worse than an honest gap
    The original column is kept; `popularity_fill_method` says how each value was obtained.
    """
    zero = df["popularity"] == 0
    real_only = df["popularity"].where(~zero)                                         # zeros become NaN ("missing")
    level1_values = real_only.groupby(df["stint_key"]).transform(
        lambda s: s.interpolate(method="linear", limit_direction="both"))             # nearest real values inside the run
    song_median = real_only.groupby(df["track_id"]).transform("median")               # median of the song's real values

    df["popularity_clean"] = level1_values
    use_level2 = zero & level1_values.isna() & song_median.notna()
    df.loc[use_level2, "popularity_clean"] = song_median[use_level2]
    df["popularity_clean"] = round_half_up(df["popularity_clean"], 2)

    df["popularity_was_zero"] = zero
    df["popularity_fill_method"] = np.select(
        [(zero & level1_values.notna()).to_numpy(), use_level2.to_numpy(), zero.to_numpy()],
        ["interpolated_in_run", "song_median", "unfilled"], default="original")
    log.info("Popularity values by method: %s", df["popularity_fill_method"].value_counts().to_dict())
    clog.add("2.5", "Popularity zeros treated as missing and repaired (run interpolation -> song median -> unfilled)",
             len(df), len(df), int(zero.sum()),
             note="original column kept; method recorded in popularity_fill_method; unfilled rows stay empty")
    return df


# ====================================================================================== step 2.6
def _add_release_format(df, clog):
    """Two simple descriptors of the release: Single or Album (compilation joins Album), and a size band."""
    df["release_format"] = np.where(df["album_type"] == "single", "Single", "Album")
    df["release_size_band"] = pd.cut(df["total_tracks"], bins=SIZE_BINS, labels=SIZE_LABELS).astype(str)
    clog.add("2.6", "Added release_format (Single/Album) and release_size_band", len(df), len(df), len(df),
             note="album_type kept unchanged; compilation merged into Album only inside release_format")
    return df
