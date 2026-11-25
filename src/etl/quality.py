"""
src/etl/quality.py  -  step 6 of the pipeline: the QUALITY GATE. Nothing is saved or loaded unless it passes.

WHAT THIS FILE IS
    25 automatic tests that run on the finished tables, plus one function that turns a failed test into a stopped pipeline.
    Tests 1-20 are the checks of Phase 2 (counts, uniqueness, rank rules, popularity rules, movement logic, KPI sanity).
    Tests 21-25 are new: they prove that the tables fit together (every key in the fact table exists in its dimension),
    so that the database load can never fail on a foreign key.

WHY WE NEED IT
    Phase 1 checked the raw data; this gate checks OUR work. A bug in a rolling calculation or in a merge would not
    show up as an error: it would show up as a wrong number on a dashboard weeks later. The gate recomputes key facts
    in a second, independent way and compares: for example it rebuilds "yesterday's position" with a merge,
    while the pipeline used a shift.

WHAT HAPPENS WHEN IT RUNS
    `run_quality_gate(tables)` returns a table with one row per test (id, test, status, detail).
    `raise_if_gate_failed(gate)` raises DataIntegrityError if any test has the status FAIL.
    A test that crashes (for example because a column is missing) is recorded as FAIL with the error message.

EXPECTED OUTCOME
    "25 of 25 tests passed" for the project file.

MOTIVE
    Fail fast and loudly: a stopped pipeline is a small problem, a silently wrong dashboard is a big one.
"""
from __future__ import annotations

import pandas as pd

from src.config import CHART_SIZE
from src.etl.errors import DataIntegrityError
from src.logger import get_logger

log = get_logger(__name__)

MOVEMENT_LABELS = ["Up", "Down", "Same", "New entry", "Re-entry", "Chart start"]
KEY_COLUMNS = ["date", "track_id", "position", "track_label", "stint_key", "movement"]


def run_quality_gate(tables):
    """Run all tests on the dictionary of tables and return the results as a table."""
    pdl = tables["playlist_daily"]
    fact = tables["fact_chart_daily"]
    dt = tables["dim_track"]
    da = tables["dim_artist"]
    br = tables["bridge_track_artist"]
    dd = tables["dim_date"]
    ds = tables["daily_summary"]

    results = []

    def gate(name, test):
        """Run one test. `test` returns True/False, or (True/False, detail text). A crash counts as FAIL."""
        try:
            outcome = test()
            passed, detail = outcome if isinstance(outcome, tuple) else (outcome, "")
            status = "PASS" if bool(passed) else "FAIL"
        except Exception as error:                      # a broken test must never stop the other tests from running
            status, detail = "FAIL", f"test crashed: {type(error).__name__}: {error}"
        results.append({"id": f"G{len(results) + 1:02d}", "test": name, "status": status, "detail": detail})

    n_dates = pdl["date"].nunique()
    n_songs = pdl["track_id"].nunique()
    n_days = (pdl["date"].max() - pdl["date"].min()).days + 1

    # ---------------------------------------------------------------- the daily table: shape and uniqueness (G01 - G06)
    gate(f"{n_dates} dates x {CHART_SIZE} rows = {n_dates * CHART_SIZE:,} rows",
         lambda: (len(pdl) == n_dates * CHART_SIZE and pdl.groupby("date").size().eq(CHART_SIZE).all(), f"{len(pdl):,} rows"))
    gate("(date, position) is unique", lambda: not pdl.duplicated(["date", "position"]).any())
    gate("(date, track_id) is unique: no song twice on one day", lambda: not pdl.duplicated(["date", "track_id"]).any())
    gate(f"Every date holds all positions 1-{CHART_SIZE}", lambda: pdl.groupby("date")["position"].nunique().eq(CHART_SIZE).all())
    gate(f"snapshot_index runs 1..{n_dates} without holes",
         lambda: sorted(pdl["snapshot_index"].unique()) == list(range(1, n_dates + 1)))
    gate("No missing values in the key columns", lambda: pdl[KEY_COLUMNS].notna().all().all())

    # ---------------------------------------------------------------- popularity repair (G07 - G09)
    gate("popularity_clean stays inside 0-100", lambda: pdl["popularity_clean"].dropna().between(1, 100).all())
    gate("Only the 'unfilled' rows have no clean popularity",
         lambda: (pdl["popularity_clean"].isna().sum() == (pdl["popularity_fill_method"] == "unfilled").sum(),
                  f"{int(pdl['popularity_clean'].isna().sum())} empty values"))
    gate("popularity_clean equals the original wherever the original was not zero",
         lambda: (pdl.loc[~pdl["popularity_was_zero"], "popularity_clean"]
                  == pdl.loc[~pdl["popularity_was_zero"], "popularity"]).all())

    # ---------------------------------------------------------------- movement logic (G10 - G13)
    def previous_position_matches():
        # independent recalculation of yesterday's position: shift every row one snapshot FORWARD and merge it back
        yesterday = pdl[["track_id", "snapshot_index", "position"]].copy()
        yesterday["snapshot_index"] += 1
        yesterday = yesterday.rename(columns={"position": "prev_position_check"})
        check = pdl[["track_id", "snapshot_index", "prev_position"]].merge(yesterday, on=["track_id", "snapshot_index"], how="left")
        return (check["prev_position"].fillna(-1) == check["prev_position_check"].fillna(-1)).all()

    gate("prev_position equals an independent recalculation", previous_position_matches)
    gate("rank_change = prev_position - position",
         lambda: ((pdl["prev_position"] - pdl["position"]).fillna(-999) == pdl["rank_change"].fillna(-999)).all())
    gate("Every row has exactly one movement label", lambda: pdl["movement"].isin(MOVEMENT_LABELS).all())
    gate("Entries equal exits on every day",
         lambda: (ds["new_entries"] + ds["re_entries"]).iloc[1:].eq(ds["exits"].iloc[1:]).all())

    # ---------------------------------------------------------------- song-level KPIs (G14 - G17)
    gate(f"dim_track: one row per song, {n_songs} songs", lambda: dt["track_id"].is_unique and len(dt) == n_songs)
    gate("Sum of days_on_chart equals the number of daily rows", lambda: dt["days_on_chart"].sum() == len(pdl))
    gate("best_rank <= avg_rank <= worst_rank for every song",
         lambda: ((dt["best_rank"] <= dt["avg_rank"]) & (dt["avg_rank"] <= dt["worst_rank"])).all())
    gate("Rank Volatility Index is never negative", lambda: (dt["rank_volatility_index"].dropna() >= 0).all())

    # ---------------------------------------------------------------- artists and calendar (G18 - G20)
    gate("Every song has exactly one primary artist in the bridge", lambda: br.groupby("track_id")["is_primary"].sum().eq(1).all())
    gate("Artist Dominance Index (primary basis) adds up to 100%",
         lambda: (abs(da["dominance_index_primary"].sum() - 100) < 0.05, f"{da['dominance_index_primary'].sum():.3f}%"))
    gate(f"dim_date is continuous: {n_days} days, {n_dates} with a snapshot",
         lambda: len(dd) == n_days and dd["date"].diff().dropna().eq(pd.Timedelta(days=1)).all()
                 and int(dd["has_snapshot"].sum()) == n_dates)

    # ---------------------------------------------------------------- NEW: do the tables fit together? (G21 - G25)
    gate("Every fact row points to an existing song and an existing date",
         lambda: fact["track_id"].isin(dt["track_id"]).all() and fact["date"].isin(dd["date"]).all())
    gate("Every bridge row points to an existing song and an existing artist; every artist has a song",
         lambda: br["track_id"].isin(dt["track_id"]).all() and br["artist_name"].isin(da["artist_name"]).all()
                 and da["artist_name"].isin(br["artist_name"]).all())
    gate("Every daily_summary date is a snapshot date of dim_date",
         lambda: ds["date"].isin(dd.loc[dd["has_snapshot"], "date"]).all() and ds["date"].is_unique)

    def days_match_fact():
        counted = fact.groupby("track_id").size()
        return (dt.set_index("track_id")["days_on_chart"] == counted).all()

    gate("Every song's days_on_chart equals its number of fact rows", days_match_fact)
    gate("Total chart points = number of dates x points per day",
         lambda: (fact["chart_points"].sum() == n_dates * CHART_SIZE * (CHART_SIZE + 1) // 2,
                  f"{int(fact['chart_points'].sum()):,}"))

    gate_df = pd.DataFrame(results, columns=["id", "test", "status", "detail"])
    passed = int((gate_df["status"] == "PASS").sum())
    log.info("Quality gate: %s of %s tests passed", passed, len(gate_df))
    for _, row in gate_df[gate_df["status"] == "FAIL"].iterrows():
        log.error("Gate test %s FAILED: %s %s", row["id"], row["test"], row["detail"])
    return gate_df


def raise_if_gate_failed(gate_df):
    """Stop the pipeline if at least one test has the status FAIL."""
    failed = gate_df[gate_df["status"] == "FAIL"]
    if len(failed):
        lines = "\n".join(f"  - {r.id}: {r.test} {r.detail}".rstrip() for r in failed.itertuples())
        raise DataIntegrityError(f"The quality gate failed ({len(failed)} of {len(gate_df)} tests):\n{lines}\n"
                                 "Nothing was saved or loaded. See reports/quality_gate.csv.")
