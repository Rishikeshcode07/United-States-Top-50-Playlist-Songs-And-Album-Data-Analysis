"""
src/dash/metrics.py  -  the KPIs of the brief, one table per song and one table per artist, for whatever rows are selected.

WHAT THIS FILE IS
    headline             the numbers of the top strip of the dashboard (songs, artists, average place, volatility, popularity trend ...)
    song_table           one row per song: Days on Chart, Average Rank, Best Rank, Rank Volatility Index, Average Daily Rank Move,
                         chart points, days in the top 10, popularity, entry and exit place ... computed from the SELECTED rows
    artist_table         one row per artist: songs, chart slots, chart points and the Artist Dominance Index (ADI)
    artist_period_adi    the ADI of chosen artists month by month (or week by week, day by day)
    concentration_by_period   HHI, effective number of artists, top-1/5/10 share and Gini for every month (week, day)
    explode_credits      one row per (song-day, credited artist), for the "all credits" view of artist dominance
    popularity_trend     the Popularity Score Trend: the average 7-day popularity at the start and at the end of the selection
    add_period           a month / week / day start column for grouping by time

WHY WE NEED IT
    The processed tables (`dim_track`, `dim_artist`) hold each KPI for the WHOLE file. When a person narrows the dates to
    one quarter or picks only explicit songs, the same KPI must be calculated again from the rows that are left. This file is
    the one place where each KPI is written as a formula, and the tests prove that on the whole file it gives back exactly the
    numbers of `dim_track` and `dim_artist`.

    Artist Dominance Index (ADI) = the artist's chart points / ALL chart points of the chosen dates x 100, where a song-day earns
    51 - place points (place 1 = 50 points, place 50 = 1). The denominator is always the whole Top 50 of the chosen dates, even
    when other filters are on, so an ADI means "this share of the whole chart". Two views exist: by PRIMARY artist (the shares
    add up to 100 %) and by ALL CREDITS (a featured artist also earns the points; the shares add up to more than 100 %).

WHAT HAPPENS WHEN IT RUNS
    All functions take the rows that `filters.apply_filters` returned and give back new tables; nothing is changed in place. An
    empty selection returns empty tables with the same columns, so a page can show "no songs match" instead of failing.
    Rank Volatility Index is the standard deviation of a song's places, so it needs two or more days; a one-day song gets an
    empty value, exactly as in `dim_track`.

EXPECTED OUTCOME
    On the whole file: 944 songs, Days on Chart / Average Rank / Best Rank / Rank Volatility Index equal `dim_track`
    (to the rounding of that table), the ADI of 243 primary artists adds up to 100.0 and equals `dim_artist.dominance_index_primary`.

MOTIVE
    The same KPI, from the same formula, whatever the person has filtered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CHART_SIZE, LONGEVITY_BINS, LONGEVITY_LABELS
from src.eda import concentration as conc

POINTS_PER_DAY = CHART_SIZE * (CHART_SIZE + 1) // 2          # 50 + 49 + ... + 1 = 1,275 chart points a day
PERIODS = ("day", "week", "month")

SONG_COLUMNS = ["track_id", "track_label", "song", "primary_artist", "artist_credit", "is_collab", "is_explicit", "album_type",
                "total_tracks", "release_size_band", "duration_min", "album_cover_url", "days_on_chart", "first_date", "last_date",
                "avg_rank", "median_rank", "best_rank", "worst_rank", "rank_volatility_index", "avg_daily_rank_move",
                "chart_points", "days_in_top10", "days_in_top20", "days_at_number_one", "share_in_top10", "avg_popularity",
                "peak_popularity", "popularity_first", "popularity_last", "popularity_change", "popularity_slope_week",
                "entry_rank", "exit_rank", "longevity_class"]
ARTIST_COLUMNS = ["artist", "songs", "chart_slots", "days_present", "chart_points", "adi", "adi_rank", "share_of_slots_pct",
                  "avg_rank", "best_rank", "slots_in_top10", "days_at_number_one", "first_date", "last_date", "points_per_day"]


# ==================================================================================================== small helpers
def add_period(frame: pd.DataFrame, period: str, column: str = "period") -> pd.DataFrame:
    """A copy with a column that holds the first day of the day / week (Monday) / month of each row."""
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}")
    dates = frame["date"].dt.normalize()
    if period == "week":
        dates = dates - pd.to_timedelta(frame["date"].dt.day_of_week, unit="D")
    elif period == "month":
        dates = frame["date"].dt.to_period("M").dt.to_timestamp()
    return frame.assign(**{column: dates})


def median_or_nan(values) -> float:
    """
    The median of a column, or NaN when it has no value at all (an empty selection, or a volatility column on a one-day window).
    pandas would return NaN too, but only after numpy prints a "Mean of empty slice" warning; this helper stays quiet.
    """
    series = pd.Series(values).dropna()
    return float(series.median()) if len(series) else float("nan")


def total_chart_points(window: pd.DataFrame) -> float:
    """All chart points of the chosen dates (the denominator of the ADI). `window` = every row of those dates."""
    return float(window["chart_points"].sum())


def grouped_slope(frame: pd.DataFrame, key: str, x: str, y: str, min_points: int = 3) -> pd.Series:
    """The least-squares slope of y on x inside each group (empty when a group has fewer than `min_points` points)."""
    pairs = frame[[key, x, y]].dropna()
    if pairs.empty:
        return pd.Series(dtype="float64")
    pairs = pairs.assign(_xy=pairs[x] * pairs[y], _xx=pairs[x] ** 2)
    grouped = pairs.groupby(key)
    n = grouped.size().astype(float)
    sx, sy, sxy, sxx = grouped[x].sum(), grouped[y].sum(), grouped["_xy"].sum(), grouped["_xx"].sum()
    denominator = n * sxx - sx ** 2
    slope = (n * sxy - sx * sy) / denominator.where(denominator > 0)
    return slope.where(n >= min_points)


def longevity_class(days) -> pd.Series:
    """Flash / Short / Medium / Long / Evergreen from the number of chart days (the classes of dim_track)."""
    return pd.cut(pd.Series(days), bins=LONGEVITY_BINS, labels=LONGEVITY_LABELS).astype("object")


# ======================================================================================================== songs
def song_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per song, calculated from the selected rows only. Empty selection -> empty table with the same columns."""
    if rows.empty:
        return pd.DataFrame(columns=SONG_COLUMNS)
    r = rows.sort_values(["track_id", "date"]).assign(_abs_move=lambda d: d["rank_change"].abs())
    grouped = r.groupby("track_id", sort=True)
    table = grouped.agg(
        track_label=("track_label", "first"), song=("song_clean", "first"), primary_artist=("primary_artist", "first"),
        artist_credit=("artist_credit", "first"), is_collab=("is_collab", "first"), is_explicit=("is_explicit", "first"),
        album_type=("album_type", "first"), total_tracks=("total_tracks", "first"),
        release_size_band=("release_size_band", "first"), duration_min=("duration_min", "first"),
        album_cover_url=("album_cover_url", "first"),
        days_on_chart=("position", "size"), first_date=("date", "min"), last_date=("date", "max"),
        avg_rank=("position", "mean"), median_rank=("position", "median"), best_rank=("position", "min"),
        worst_rank=("position", "max"), rank_volatility_index=("position", "std"), avg_daily_rank_move=("_abs_move", "mean"),
        chart_points=("chart_points", "sum"), days_in_top10=("in_top10", "sum"), days_in_top20=("in_top20", "sum"),
        days_at_number_one=("is_number_one", "sum"), avg_popularity=("popularity_clean", "mean"),
        peak_popularity=("popularity_clean", "max"), popularity_first=("popularity_clean", "first"),
        popularity_last=("popularity_clean", "last"), entry_rank=("position", "first"), exit_rank=("position", "last"),
    )
    table["share_in_top10"] = table["days_in_top10"] / table["days_on_chart"]
    table["popularity_change"] = table["popularity_last"] - table["popularity_first"]
    day_number = (r["date"] - r.groupby("track_id")["date"].transform("min")).dt.days.astype(float)
    slope = grouped_slope(r.assign(_day=day_number), "track_id", "_day", "popularity_clean")
    table["popularity_slope_week"] = (slope * 7.0).reindex(table.index)
    table["longevity_class"] = longevity_class(table["days_on_chart"]).to_numpy()
    for column in ("days_in_top10", "days_in_top20", "days_at_number_one"):
        table[column] = table[column].astype(int)
    return table.reset_index()[SONG_COLUMNS]


# ======================================================================================================= artists
def explode_credits(rows: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    """One row per (song-day, credited artist); the column `artist_name` says which artist."""
    pairs = bridge[["track_id", "artist_name"]].drop_duplicates()
    return rows.merge(pairs, on="track_id", how="inner")


def artist_table(rows: pd.DataFrame, total_points: float, n_days: int, basis: str = "primary",
                 bridge: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """
    One row per artist, biggest ADI first. `total_points` = all chart points of the chosen dates (see `total_chart_points`),
    `n_days` = the number of chart days of those dates. basis = "primary" (the shares add up to 100) or "credits" (every credited
    artist earns the points of the song, so the shares add up to more than 100) and needs the bridge table.
    """
    if basis not in ("primary", "credits"):
        raise ValueError("basis must be 'primary' or 'credits'")
    if rows.empty:
        return pd.DataFrame(columns=ARTIST_COLUMNS)
    if basis == "credits":
        if bridge is None:
            raise ValueError("the 'credits' view needs the bridge table")
        frame, key = explode_credits(rows, bridge), "artist_name"
    else:
        frame, key = rows, "primary_artist"
    grouped = frame.groupby(key, sort=False)
    table = grouped.agg(songs=("track_id", "nunique"), chart_slots=("position", "size"), days_present=("date", "nunique"),
                        chart_points=("chart_points", "sum"), avg_rank=("position", "mean"), best_rank=("position", "min"),
                        slots_in_top10=("in_top10", "sum"), days_at_number_one=("is_number_one", "sum"),
                        first_date=("date", "min"), last_date=("date", "max"))
    table.index.name = "artist"
    table = table.reset_index()
    table["adi"] = table["chart_points"] / total_points * 100.0 if total_points > 0 else np.nan
    table["share_of_slots_pct"] = table["chart_slots"] / (CHART_SIZE * max(int(n_days), 1)) * 100.0
    table["points_per_day"] = table["chart_points"] / max(int(n_days), 1)
    for column in ("slots_in_top10", "days_at_number_one"):
        table[column] = table[column].astype(int)
    table = table.sort_values(["adi", "artist"], ascending=[False, True]).reset_index(drop=True)
    table["adi_rank"] = np.arange(1, len(table) + 1)
    return table[ARTIST_COLUMNS]


def artist_period_adi(rows: pd.DataFrame, window: pd.DataFrame, period: str = "month", basis: str = "primary",
                      bridge: "pd.DataFrame | None" = None, artists=None) -> pd.DataFrame:
    """
    The ADI of artists in every day / week / month: chart points of the artist in the period / all chart points of the period x 100.
    `window` holds every row of the chosen dates (the whole chart). With `artists` the table has one row for every artist and every
    period, with 0 where the artist had no points, so that a line chart does not skip empty months.
    Columns: period, artist, points, total_points, adi.
    """
    totals = add_period(window, period).groupby("period")["chart_points"].sum().rename("total_points")
    if rows.empty or totals.empty:
        return pd.DataFrame(columns=["period", "artist", "points", "total_points", "adi"])
    if basis == "credits":
        if bridge is None:
            raise ValueError("the 'credits' view needs the bridge table")
        frame, key = explode_credits(rows, bridge), "artist_name"
    else:
        frame, key = rows, "primary_artist"
    if artists is not None:
        frame = frame[frame[key].isin(list(artists))]
    points = add_period(frame, period).groupby(["period", key])["chart_points"].sum().rename("points")
    if artists is not None:
        full = pd.MultiIndex.from_product([totals.index, list(dict.fromkeys(artists))], names=["period", key])
        points = points.reindex(full, fill_value=0)
    table = points.reset_index().rename(columns={key: "artist"})
    table = table.merge(totals, left_on="period", right_index=True, how="left")
    table["adi"] = table["points"] / table["total_points"] * 100.0
    return table.sort_values(["period", "artist"]).reset_index(drop=True)


def concentration_by_period(rows: pd.DataFrame, period: str = "month", holder: str = "primary_artist") -> pd.DataFrame:
    """
    How concentrated the chart points are in every period: the number of artists, the HHI (0 to 10,000), the effective number of
    artists (10,000 / HHI), the share of the biggest 1, 5 and 10 artists (in %) and the Gini coefficient.
    """
    columns = ["period", "artists", "hhi", "effective_number", "top1_pct", "top5_pct", "top10_pct", "gini"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    points = add_period(rows, period).groupby(["period", holder])["chart_points"].sum()
    out = []
    for period_start, values in points.groupby(level=0):
        v = values.to_numpy(dtype=float)
        if v.sum() <= 0:
            continue
        out.append({"period": period_start, "artists": int(len(v)), "hhi": conc.hhi(v), "effective_number": conc.effective_number(v),
                    "top1_pct": conc.top_n_share(v, 1) * 100.0, "top5_pct": conc.top_n_share(v, 5) * 100.0,
                    "top10_pct": conc.top_n_share(v, 10) * 100.0, "gini": conc.gini(v)})
    return pd.DataFrame(out, columns=columns)


# ============================================================================================== popularity trend
def popularity_trend(rows: pd.DataFrame, days: int = 7) -> dict:
    """
    The Popularity Score Trend of the selection: the mean 7-day popularity of the songs on the chart, averaged over the first
    `days` chart days and over the last `days` chart days of the selection, and the change between the two.
    """
    empty = {"start": np.nan, "end": np.nan, "change": np.nan}
    if rows.empty or "popularity_trend_7d" not in rows.columns:
        return empty
    daily = rows.groupby("date")["popularity_trend_7d"].mean().dropna().sort_index()
    if daily.empty:
        return empty
    start, end = float(daily.iloc[:days].mean()), float(daily.iloc[-days:].mean())
    return {"start": start, "end": end, "change": end - start}


# ============================================================================================ the top strip
def headline(rows: pd.DataFrame, window: pd.DataFrame, summary: pd.DataFrame, songs: "pd.DataFrame | None" = None,
             artists: "pd.DataFrame | None" = None) -> dict:
    """
    The numbers of the top strip. `rows` = the selected rows, `window` = every row of the chosen dates, `summary` = daily_summary of
    the chosen dates. `songs` / `artists` can be passed when the page has already built them (it saves one calculation).
    """
    songs = song_table(rows) if songs is None else songs
    n_days = int(window["date"].nunique()) if not window.empty else 0
    out = {"days": n_days, "song_days": int(len(rows)), "songs": int(len(songs)),
           "artists": int(rows["primary_artist"].nunique()) if not rows.empty else 0,
           "avg_rank": float(rows["position"].mean()) if not rows.empty else np.nan,
           "best_rank": int(rows["position"].min()) if not rows.empty else None,
           "avg_popularity": float(rows["popularity_clean"].mean()) if not rows.empty else np.nan,
           "median_days_on_chart": median_or_nan(songs["days_on_chart"]) if not songs.empty else np.nan,
           "longest_days_on_chart": int(songs["days_on_chart"].max()) if not songs.empty else 0,
           "median_volatility": median_or_nan(songs["rank_volatility_index"]) if not songs.empty else np.nan,
           "explicit_share_rows": float(rows["is_explicit"].astype(bool).mean()) if not rows.empty else np.nan,
           "explicit_share_chart": float(summary["explicit_share"].mean()) if not summary.empty else np.nan,
           "turnover_pct": float(summary["turnover_rate_pct"].mean()) if not summary.empty else np.nan,
           "new_entries_per_day": float(summary["new_entries"].mean()) if not summary.empty else np.nan,
           "popularity_trend": popularity_trend(rows)}
    if artists is not None and not artists.empty:
        top = artists.iloc[0]
        out.update(adi_leader=str(top["artist"]), adi_leader_value=float(top["adi"]), adi_top10=float(artists["adi"].head(10).sum()))
    else:
        out.update(adi_leader=None, adi_leader_value=np.nan, adi_top10=np.nan)
    return out
