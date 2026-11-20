"""
tests/synthetic.py  -  a small, hand-designed chart that exercises every rule of the pipeline, plus an independent "oracle".

WHAT THIS FILE IS
    `build_plan()` creates 11 daily snapshots of a 50-song chart. It is an "escalator": every day each song climbs 2 places
    and two new songs enter at the bottom. On top of that escalator we plant the situations that make real data hard:

        * a calendar day without a snapshot (2024-01-06)                      -> gap-aware time features
        * a date captured twice (2024-01-09): a stale block, then the real one -> keep the latest block
        * two swaps of songs (snapshots 4 and 7)                               -> songs that stay in place or fall ("Same", "Down")
        * song 30 leaves for two snapshots and comes back                      -> re-entry, two chart runs
        * popularity 0 on a debut day, in the middle of a run, on a one-day song, and for a whole run
                                                                               -> interpolation, song median, "unfilled"
        * one song credited "A & B" for three days and "A" afterwards          -> same song, one track_id
        * two different songs both called "Angel"                              -> must stay two songs
        * "Earth, Wind & Fire" (contains &) and "Tyler, The Creator" (contains a comma) -> artist splitting rules
        * a curly apostrophe on even days and a straight one on odd days       -> one title
        * "Beyonce" and "Beyonce with an accent"                               -> two artists (accents are real differences)
        * an album_type of "compilation"                                       -> WARN, grouped with Album

    `expected_daily()` and `expected_songs()` then compute the right answers with plain Python loops (no pandas groupby),
    so the tests compare the pipeline against an implementation that shares none of its code.

WHY WE NEED IT
    A test that re-uses the code under test only proves that the code agrees with itself. This oracle is written
    differently on purpose.

WHAT HAPPENS WHEN IT RUNS
    Nothing by itself; the test files import it.

EXPECTED OUTCOME
    A raw table (and CSV) of 11 dates + 1 stale block = 600 rows, in the same format as the real file.

MOTIVE
    Every rule of the pipeline has one small example whose answer a human can verify.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

START = date(2024, 1, 1)
MISSING_DAY = date(2024, 1, 6)
DOUBLE_DAY = date(2024, 1, 9)
CALENDAR_DAYS = 12
FILLER_ID = 71
SWAPS = {3: (16, 18), 6: (6, 11)}       # snapshot s -> two chart positions whose songs trade places (a song that stays, a song that falls)


def snapshot_dates():
    """The 11 dates that have a snapshot: 12 calendar days minus the missing one."""
    days = [START + timedelta(days=i) for i in range(CALENDAR_DAYS)]
    return [d for d in days if d != MISSING_DAY]


# ---------------------------------------------------------------------------------------------- song attributes
def title_as_written(song_id, s):
    if song_id in (40, 41):
        return "Angel"
    if song_id == 28:
        return "Thinkin’ Bout You" if s % 2 == 0 else "Thinkin' Bout You"    # curly on even snapshots, straight on odd
    if song_id == FILLER_ID:
        return "Filler"
    return f"Song {song_id:02d}"


def artist_as_written(song_id, s):
    special = {25: "Earth, Wind & Fire", 26: "Earth, Wind & Fire & Artist 26", 27: "Tyler, The Creator",
               29: "Beyoncé", 31: "Beyonce", 28: "Artist 28"}
    if song_id in special:
        return special[song_id]
    if song_id == 12:
        return "Artist 12 & Artist 99" if s <= 2 else "Artist 12"
    if song_id % 7 == 0:
        return f"Artist {song_id} & Artist {song_id + 100}"
    return f"Artist {song_id}"


def popularity_as_written(song_id, s):
    zero_cases = {(51, 1), (10, 2), (70, 10)} | {(30, k) for k in range(6, 11)}
    if (song_id, s) in zero_cases:
        return 0
    return 70 if song_id == FILLER_ID else 50 + (song_id % 40) + s


def album_type_of(song_id):
    if song_id == 33:
        return "compilation"
    return "single" if song_id % 2 == 0 else "album"


def total_tracks_of(song_id):
    kind = album_type_of(song_id)
    return {"single": 1, "album": 12, "compilation": 20}[kind]


def chart_at(s):
    """{position: song_id} for snapshot s: the escalator, two swaps, and the absence of song 30 on snapshots 5 and 6."""
    chart = {pos: pos + 2 * s for pos in range(1, 51)}
    if s in SWAPS:                                      # two songs trade places: one of them stays or falls instead of climbing
        first, second = SWAPS[s]
        chart[first], chart[second] = chart[second], chart[first]
    if s in (4, 5):                                     # song 30 is away for two snapshots; a filler takes its place
        position_of_30 = 30 - 2 * s
        chart[position_of_30] = FILLER_ID
    return chart


# ---------------------------------------------------------------------------------------------- the plan
@dataclass
class Plan:
    dates: list
    raw_rows: list = field(default_factory=list)      # every row of the file, in file order (stale block included)
    true_rows: list = field(default_factory=list)     # the rows the pipeline must keep (latest block of every date)


def build_plan():
    dates = snapshot_dates()
    plan = Plan(dates=dates)
    for s, day in enumerate(dates):
        if day == DOUBLE_DAY:                           # a STALE block first: yesterday's chart, wrongly stamped with today's date
            for pos, song_id in chart_at(s - 1).items():
                plan.raw_rows.append(_row(s, day, pos, song_id, stale=True))
        for pos, song_id in chart_at(s).items():
            row = _row(s, day, pos, song_id)
            plan.raw_rows.append(row)
            plan.true_rows.append(row)
    return plan


def _row(s, day, pos, song_id, stale=False):
    s_used = s - 1 if stale else s                      # a stale row describes the previous snapshot
    return {"s": s, "date": day, "position": pos, "song_id": song_id, "stale": stale,
            "song": title_as_written(song_id, s_used), "artist": artist_as_written(song_id, s_used),
            "popularity": 1 if stale else popularity_as_written(song_id, s),
            "duration_ms": 150_000 + 1_000 * song_id, "album_type": album_type_of(song_id),
            "total_tracks": total_tracks_of(song_id), "is_explicit": song_id % 3 == 0,
            "album_cover_url": f"https://img.example.com/cover_{song_id}.jpg"}


def raw_frame(plan):
    """The raw table exactly as the CSV would show it: dates written day-month-year."""
    frame = pd.DataFrame(plan.raw_rows)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%d-%m-%Y")
    return frame[["date", "position", "song", "artist", "popularity", "duration_ms", "album_type", "total_tracks",
                  "is_explicit", "album_cover_url"]]


def write_csv(plan, path):
    raw_frame(plan).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------------------------- the oracle (plain loops)
def _presence(plan):
    """song_id -> {snapshot s: position} using only the true rows."""
    presence = defaultdict(dict)
    for row in plan.true_rows:
        presence[row["song_id"]][row["s"]] = row["position"]
    return presence


def _runs(snapshots):
    """Split a sorted list of snapshot numbers into runs of consecutive numbers."""
    runs, current = [], []
    for s in sorted(snapshots):
        if current and s == current[-1] + 1:
            current.append(s)
        else:
            if current:
                runs.append(current)
            current = [s]
    if current:
        runs.append(current)
    return runs


def expected_daily(plan):
    """One dictionary per true row, with the movement columns computed by hand-written loops."""
    presence = _presence(plan)
    out = []
    for row in plan.true_rows:
        sid, s, pos = row["song_id"], row["s"], row["position"]
        seen = presence[sid]
        prev = seen.get(s - 1)
        earlier = [k for k in seen if k < s]
        if s == 0:
            movement = "Chart start"
        elif prev is None:
            movement = "Re-entry" if earlier else "New entry"
        else:
            movement = "Up" if prev > pos else ("Down" if prev < pos else "Same")
        gain = (seen[s - 7] - pos) if all((s - k) in seen for k in range(0, 8)) else np.nan
        runs = _runs(seen.keys())
        run = next(r for r in runs if s in r)
        out.append({"date": pd.Timestamp(row["date"]), "position": pos, "song_id": sid, "snapshot_index": s + 1,
                    "movement": movement, "prev_position": float(prev) if prev is not None else np.nan,
                    "rank_change": float(prev - pos) if prev is not None else np.nan,
                    "rank_gain_7d": float(gain),
                    "stint_no": runs.index(run) + 1, "days_in_stint": run.index(s) + 1, "stint_length": len(run),
                    "chart_day_number": sorted(seen).index(s) + 1,
                    "calendar_age_days": (row["date"] - plan.dates[min(seen)]).days})
    return pd.DataFrame(out)


def final_title(song_id):
    return "Thinkin’ Bout You" if song_id == 28 else title_as_written(song_id, 0)


def final_credit(song_id):
    """The fullest credit of a song (the version with the most artists)."""
    return {12: "Artist 12 & Artist 99"}.get(song_id, artist_as_written(song_id, 5))


def final_artists(song_id):
    credit = final_credit(song_id)
    for protected in ("Earth, Wind & Fire",):
        if protected in credit:
            rest = credit.replace(protected, "").replace(" & ", "", 1) if credit != protected else ""
            return [protected] + ([rest] if rest else [])
    return credit.split(" & ")


def expected_songs(plan):
    """One dictionary per song: the song-level KPIs computed by hand-written loops."""
    presence = _presence(plan)
    dates = plan.dates
    out = []
    for sid, seen in presence.items():
        snaps = sorted(seen)
        positions = [seen[k] for k in snaps]
        runs = _runs(snaps)
        out.append({"song_id": sid, "song_clean": final_title(sid), "artist_credit": final_credit(sid),
                    "primary_artist": final_artists(sid)[0], "artist_count": len(final_artists(sid)),
                    "days_on_chart": len(snaps), "best_rank": min(positions), "worst_rank": max(positions),
                    "avg_rank": sum(positions) / len(positions), "n_runs": len(runs),
                    "longest_run_days": max(len(r) for r in runs),
                    "first_seen": pd.Timestamp(dates[snaps[0]]), "last_seen": pd.Timestamp(dates[snaps[-1]]),
                    "entry_rank": positions[0], "exit_rank": positions[-1],
                    "days_in_top10": sum(p <= 10 for p in positions), "days_at_number_one": sum(p == 1 for p in positions),
                    "chart_points": sum(51 - p for p in positions),
                    "span_snapshots": snaps[-1] - snaps[0] + 1,
                    "on_chart_at_window_start": snaps[0] == 0, "on_chart_at_window_end": snaps[-1] == len(dates) - 1})
    return pd.DataFrame(out)


def expected_entries_exits(plan):
    """{snapshot s (1-based index): (new_entries, re_entries, exits)} from plain set arithmetic."""
    presence = _presence(plan)
    at = defaultdict(set)
    for sid, seen in presence.items():
        for s in seen:
            at[s].add(sid)
    result = {}
    ever_before = set()
    for s in range(len(plan.dates)):
        if s == 0:
            ever_before |= at[0]
            result[1] = (0, 0, 0)
            continue
        entries = at[s] - at[s - 1]
        new = {sid for sid in entries if sid not in ever_before}
        result[s + 1] = (len(new), len(entries) - len(new), len(at[s - 1] - at[s]))
        ever_before |= at[s]
    return result


def expected_artists(plan):
    """One dictionary per individual artist: songs, chart days and chart points, from plain loops over the songs."""
    presence = _presence(plan)
    dates = plan.dates
    total_points = sum(51 - p for seen in presence.values() for p in seen.values())
    blank = {"unique_tracks": 0, "unique_tracks_as_primary": 0, "chart_slot_days": 0, "chart_slot_days_primary": 0,
             "chart_points": 0, "chart_points_primary": 0, "position_sum": 0, "best_rank": 99, "days_in_top10": 0,
             "days_at_number_one": 0, "snapshots": set()}
    artists = {}
    for song_id, seen in presence.items():
        points = sum(51 - p for p in seen.values())
        for order, name in enumerate(final_artists(song_id)):
            row = artists.setdefault(name, {**blank, "snapshots": set()})
            row["unique_tracks"] += 1
            row["chart_slot_days"] += len(seen)
            row["chart_points"] += points
            row["position_sum"] += sum(seen.values())
            row["best_rank"] = min(row["best_rank"], min(seen.values()))
            row["days_in_top10"] += sum(p <= 10 for p in seen.values())
            row["days_at_number_one"] += sum(p == 1 for p in seen.values())
            row["snapshots"] |= set(seen)
            if order == 0:
                row["unique_tracks_as_primary"] += 1
                row["chart_slot_days_primary"] += len(seen)
                row["chart_points_primary"] += points
    out = []
    for name, row in artists.items():
        snaps = sorted(row["snapshots"])
        out.append({"artist_name": name, **{k: v for k, v in row.items() if k not in ("snapshots", "position_sum")},
                    "avg_rank": row["position_sum"] / row["chart_slot_days"], "days_present": len(snaps),
                    "first_seen": pd.Timestamp(dates[snaps[0]]), "last_seen": pd.Timestamp(dates[snaps[-1]]),
                    "dominance_index_primary": 100.0 * row["chart_points_primary"] / total_points})
    return pd.DataFrame(out)


def expected_daily_summary(plan):
    """One dictionary per snapshot: the shares and averages of that day's 50 songs, from plain loops."""
    rows_of_day = defaultdict(list)
    for row in plan.true_rows:
        rows_of_day[row["s"]].append(row)
    out = []
    for s in range(len(plan.dates)):
        rows = rows_of_day[s]
        n = len(rows)
        out.append({"date": pd.Timestamp(plan.dates[s]), "snapshot_index": s + 1, "unique_tracks": len({r["song_id"] for r in rows}),
                    "explicit_share": sum(bool(r["is_explicit"]) for r in rows) / n,
                    "collab_share": sum(len(final_artists(r["song_id"])) > 1 for r in rows) / n,
                    "single_share": sum(r["album_type"] == "single" for r in rows) / n,
                    "avg_duration_min": sum(r["duration_ms"] for r in rows) / n / 60_000,
                    "unique_primary_artists": len({final_artists(r["song_id"])[0] for r in rows})})
    return pd.DataFrame(out)


def expected_calendar(plan):
    """One dictionary per CALENDAR day (including the missing one), from Python's own date functions."""
    have = set(plan.dates)
    seasons = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
               6: "Summer", 7: "Summer", 8: "Summer"}
    out = []
    for i in range(CALENDAR_DAYS):
        day = START + timedelta(days=i)
        out.append({"date": pd.Timestamp(day), "date_key": int(day.strftime("%Y%m%d")), "has_snapshot": day in have,
                    "year": day.year, "month": day.month, "month_name": day.strftime("%B"), "year_month": day.strftime("%Y-%m"),
                    "quarter": (day.month - 1) // 3 + 1, "iso_week": day.isocalendar()[1], "day_of_week": day.weekday() + 1,
                    "day_name": day.strftime("%A"), "is_weekend": day.weekday() >= 5, "season": seasons.get(day.month, "Fall")})
    return pd.DataFrame(out)
