"""
src/etl/validate.py  -  step 2 of the pipeline: test the raw data against the data contract.

WHAT THIS FILE IS
    Two things. `QualityLog` is the scorecard: every check adds one row (PASS / WARN / INFO / FAIL).
    `validate_raw` runs the checks that only need the ORIGINAL columns: structure, data types, missing values,
    calendar coverage, rank rules, duplicates, and valid ranges for popularity, duration, album and explicit flag.
    (Checks that need song identities live in clean.py, because identities do not exist yet at this point.)

WHY WE NEED IT
    Bad data must be stopped at the door. A rank of 57 or a date that can not be read would silently produce
    wrong charts. We classify every problem by severity:
        FAIL  = the data can not be trusted, the pipeline stops
        WARN  = a real problem that a later step repairs (and the repair is logged)
        INFO  = just worth knowing
        PASS  = rule respected

WHAT HAPPENS WHEN IT RUNS
    1. Structure and data types are checked first. If a column is missing or a date can not be read,
       we stop immediately, because every later check would be meaningless.
    2. The remaining checks run on the typed table. Nothing is changed or dropped here; we only measure.
    3. The caller decides what to do with the result: `log.raise_if_failed()` stops the pipeline on any FAIL.

EXPECTED OUTCOME
    For the project file: 24 rows in the scorecard from this file (the other 7 come from clean.py),
    with these WARN rules: missing calendar days, rows per date, repeated rank/song/rows (the double snapshot),
    album type 'compilation' and 189 zero-popularity rows. No FAIL.

MOTIVE
    Measure first, repair second, and write down both. This is the "trust but verify" part of the pipeline.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import (ALLOWED_ALBUM_TYPES, CHART_SIZE, DATE_FORMAT, DURATION_RANGE_MS, EXPECTED_COLUMNS,
                        POPULARITY_RANGE)
from src.etl.errors import DataContractError
from src.logger import get_logger

log = get_logger(__name__)

STATUS_ORDER = {"FAIL": 0, "WARN": 1, "INFO": 2, "PASS": 3}
NUMERIC_COLUMNS = ["position", "popularity", "duration_ms", "total_tracks"]
TEXT_COLUMNS = ["song", "artist", "album_type", "album_cover_url"]
PLACEHOLDERS = {"", "n/a", "na", "null", "none", "unknown", "nan", "-", "?"}   # text that pretends to be a value
SCORECARD_COLUMNS = ["dimension", "check", "rule", "affected", "total", "unit", "pct_affected", "status", "action"]


class QualityLog:
    """The data-quality scorecard. One row per check; re-adding a check with the same name replaces the old row."""

    def __init__(self):
        self._rows = []

    def add(self, dimension, check, rule, affected, total, unit="rows", severity="WARN", action=""):
        """
        Record one check.

        dimension : Structure, Completeness, Validity, Uniqueness, Consistency or Coverage
        check     : short, unique name of the check
        rule      : the rule in plain words
        affected  : how many items broke the rule (0 = the check passed)
        total     : how many items were examined
        unit      : what is counted: rows, days, tracks, columns ...
        severity  : the status used when the rule is broken: FAIL, WARN or INFO
        action    : what we do about it (shown in the scorecard)
        """
        status = "PASS" if affected == 0 else severity
        pct = (affected / total * 100) if total else 0.0
        self._rows = [row for row in self._rows if row["check"] != check]      # a re-run replaces the old result
        self._rows.append({"dimension": dimension, "check": check, "rule": rule,
                           "affected": int(affected), "total": int(total), "unit": unit,
                           "pct_affected": round(pct, 2), "status": status, "action": action})
        level = {"FAIL": logging.ERROR, "WARN": logging.WARNING}.get(status, logging.DEBUG)   # PASS/INFO stay quiet on screen
        log.log(level, "[%-4s] %s: %s of %s %s (%.2f%%)", status, check, f"{int(affected):,}", f"{int(total):,}", unit, pct)
        return status

    def to_frame(self):
        """The scorecard as a table, worst problems first (FAIL, WARN, INFO, PASS), then by dimension and check name."""
        scorecard = pd.DataFrame(self._rows, columns=SCORECARD_COLUMNS)
        scorecard["_order"] = scorecard["status"].map(STATUS_ORDER)
        return (scorecard.sort_values(["_order", "dimension", "check"])
                         .drop(columns="_order").reset_index(drop=True))

    def summary(self):
        """How many checks have each status, e.g. {'PASS': 12, 'WARN': 9, 'INFO': 7, 'FAIL': 0}."""
        counts = self.to_frame()["status"].value_counts()
        return {status: int(counts.get(status, 0)) for status in ("PASS", "WARN", "INFO", "FAIL")}

    def failures(self):
        return [row for row in self._rows if row["status"] == "FAIL"]

    def raise_if_failed(self, stage=""):
        """Stop the pipeline when at least one check has the status FAIL, and say exactly which ones."""
        failed = self.failures()
        if failed:
            lines = "\n".join(f"  - {r['check']}: {r['affected']:,} of {r['total']:,} {r['unit']} "
                              f"({r['pct_affected']}%)  ->  {r['action']}" for r in failed)
            raise DataContractError(f"{len(failed)} data-quality check(s) FAILED{' ' + stage if stage else ''}:\n{lines}")


def validate_raw(df_raw, qlog):
    """
    Run all checks that need only the original 10 columns and return a copy with proper data types
    (dates as dates, numeric columns as numbers). The input table is never modified.
    """
    df = df_raw.copy()

    # ------------------------------------------------------------------ 1. structure: are the promised columns there?
    expected_cols, received_cols = list(EXPECTED_COLUMNS), list(df.columns)
    missing_cols = [c for c in expected_cols if c not in received_cols]       # promised but not delivered
    extra_cols = [c for c in received_cols if c not in expected_cols]         # delivered but not promised
    qlog.add("Structure", "No expected column is missing", "All 10 columns from the brief exist",
             len(missing_cols), len(expected_cols), unit="columns", severity="FAIL", action="Stop and re-export the data")
    qlog.add("Structure", "No unexpected extra column", "Only the 10 columns from the brief exist",
             len(extra_cols), len(received_cols), unit="columns", severity="WARN", action="Review the extra columns")
    if missing_cols:
        qlog.raise_if_failed("(missing columns: " + ", ".join(missing_cols) + ")")     # nothing else can be checked

    # ------------------------------------------------------------------ 2. data types: dates and numbers
    # The date format is stated explicitly. Letting pandas guess would read 03-04-2025 as 4 March (month first).
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT, errors="coerce")
    qlog.add("Validity", "Dates parse with DD-MM-YYYY", "Every date matches the DD-MM-YYYY pattern",
             int(df["date"].isna().sum()), len(df), severity="FAIL", action="Inspect the raw text of the failing rows")
    for col in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[col], errors="coerce")                   # text such as "abc" becomes NaN
        newly_missing = int(converted.isna().sum() - df[col].isna().sum())
        qlog.add("Validity", f"{col} is numeric", "Every value can be read as a number",
                 newly_missing, len(df), severity="FAIL", action="Inspect the raw text of the failing rows")
        df[col] = converted
    qlog.raise_if_failed("(unreadable dates or numbers)")                      # later checks need valid dates and numbers

    # ------------------------------------------------------------------ 3. completeness: missing and disguised-missing values
    qlog.add("Completeness", "No missing values (NaN) in any column", "Every cell is filled",
             int(df.isna().any(axis=1).sum()), len(df), severity="FAIL", action="Decide per column: drop, fill or flag")

    disguised_rows = pd.Series(False, index=df.index)
    for col in TEXT_COLUMNS:
        cleaned = df[col].astype("string").str.strip().str.casefold()          # trim spaces + ignore upper/lower case
        disguised_rows = disguised_rows | cleaned.isin(PLACEHOLDERS).fillna(False).astype(bool)
    qlog.add("Completeness", "No blank or placeholder text", "No empty strings or N/A-style values in text columns",
             int(disguised_rows.sum()), len(df), severity="FAIL", action="Treat as missing")

    # ------------------------------------------------------------------ 4. coverage: does every calendar day have a snapshot?
    calendar = pd.date_range(df["date"].min(), df["date"].max(), freq="D")         # every calendar day in the window
    rows_per_day = df.groupby("date").size().reindex(calendar, fill_value=0)       # 0 rows for days without data
    missing_days = rows_per_day[rows_per_day == 0].index
    odd_days = rows_per_day[(rows_per_day != CHART_SIZE) & (rows_per_day > 0)]
    qlog.add("Coverage", "No missing calendar days", "Every day between first and last date has a snapshot",
             len(missing_days), len(calendar), unit="days", severity="WARN",
             action="Do not invent rows; make time features gap-aware")
    qlog.add("Coverage", f"Exactly {CHART_SIZE} rows per date", f"Each snapshot has {CHART_SIZE} rows",
             len(odd_days), int((rows_per_day > 0).sum()), unit="days", severity="WARN",
             action="Investigate (Step 5.2) and keep one snapshot per date")
    log.debug("Missing days: %s | days with an unusual row count: %s",
              [d.strftime("%Y-%m-%d") for d in missing_days],
              {d.strftime("%Y-%m-%d"): int(n) for d, n in odd_days.items()})

    # ------------------------------------------------------------------ 5. ranks
    rank = df["position"]
    lo, hi = 1, CHART_SIZE
    qlog.add("Validity", f"Rank inside {lo}-{hi}", f"{lo} <= position <= {hi}",
             int((~rank.between(lo, hi)).sum()), len(df), severity="FAIL", action="Drop or correct the rows")
    qlog.add("Validity", "Rank is a whole number", "position has no decimals",
             int((rank % 1 != 0).sum()), len(df), severity="FAIL", action="Drop or correct the rows")
    per_day = df.groupby("date")["position"].agg(rows="size", distinct_ranks="nunique")
    qlog.add("Completeness", f"Every date has all ranks {lo}-{hi}", f"Each date contains {hi} distinct ranks",
             int((per_day["distinct_ranks"] != hi).sum()), len(per_day), unit="days", severity="FAIL",
             action="Investigate the day")
    qlog.add("Uniqueness", "No rank repeated within a date", "Each (date, position) pair is unique",
             int(df.duplicated(["date", "position"]).sum()), len(df), severity="WARN",
             action="Keep one snapshot per date (Step 5.2)")

    # ------------------------------------------------------------------ 6. duplicates
    original_cols = list(EXPECTED_COLUMNS)                                     # compare only the 10 original columns
    qlog.add("Uniqueness", "No identical rows", "No row is an exact copy of another row",
             int(df.duplicated(subset=original_cols, keep="first").sum()), len(df), severity="WARN",
             action="Resolved by keeping one snapshot per date")
    qlog.add("Uniqueness", "No song repeated within a date", "Each (date, song, artist) appears once per date",
             int(df.duplicated(["date", "song", "artist"], keep="first").sum()), len(df), severity="WARN",
             action="Resolved by keeping one snapshot per date")

    # ------------------------------------------------------------------ 7. valid ranges of the descriptive columns
    n = len(df)
    qlog.add("Validity", "Popularity inside 0-100", "0 <= popularity <= 100",
             int((~df["popularity"].between(*POPULARITY_RANGE)).sum()), n, severity="FAIL",
             action="Drop or correct the rows")
    qlog.add("Validity", "Duration between 30 s and 20 min", "30,000 <= duration_ms <= 1,200,000",
             int((~df["duration_ms"].between(*DURATION_RANGE_MS)).sum()), n, severity="FAIL",
             action="Check units (seconds vs milliseconds)")
    bad_tracks = (df["total_tracks"] < 1) | (df["total_tracks"] % 1 != 0)
    qlog.add("Validity", "total_tracks is a whole number >= 1", "total_tracks >= 1",
             int(bad_tracks.sum()), n, severity="FAIL", action="Drop or correct the rows")
    album_clean = df["album_type"].astype("string").str.strip().str.casefold()
    bad_album = ~album_clean.isin(ALLOWED_ALBUM_TYPES).fillna(False).astype(bool)
    qlog.add("Validity", "album_type is Single or Album", "album_type in {single, album}",
             int(bad_album.sum()), n, severity="WARN",
             action="Keep as its own category; group with 'album' only where a test needs bigger groups")
    qlog.add("Validity", "is_explicit is True/False", "boolean",
             int((~df["is_explicit"].isin([True, False])).sum()), n, severity="FAIL", action="Map text values to True/False")
    bad_url = ~df["album_cover_url"].astype("string").str.match(r"^https?://", na=False)
    qlog.add("Validity", "album_cover_url looks like a link", "starts with http:// or https://",
             int(bad_url.sum()), n, severity="WARN", action="Treat as missing")

    # ------------------------------------------------------------------ 8. popularity = 0 (a hidden "not scored yet")
    qlog.add("Validity", "No disguised-missing popularity (0)", "A song inside the Top 50 should not have popularity 0",
             int((df["popularity"] == 0).sum()), len(df), severity="WARN",
             action="Flag now; Phase 2 treats as missing and fills from the track's own history")

    return df
