"""
src/dash/survival_live.py  -  how long new songs stay on the chart, for whatever the filters select (Phase 6B, live).

WHAT THIS FILE IS
    prepare_stays     the Phase 6B table of stays (one row per new song) with the release type added from dim_track
    filter_stays      the stays that match the sidebar (arrival date, debut place, release type, explicit, collaboration, artist, song)
    group_labels      a group name for every song under one of six groupings (all songs, debut place, release type, explicit content,
                      collaboration, songs of the artist arriving the same day)
    live_survival     the survival curve of every group, with a band from the artist bootstrap, plus a summary table and the log-rank test
    live_hazard       the chance that a given chart day is the song's last, for age windows (day 1, day 2, day 3, 4 to 7 ...)

WHY WE NEED IT
    Phase 6B measured the stay of 894 new songs once, for the whole file, and saved the curves. The dashboard lets a person ask
    "and for singles that arrived in 2025?" The same estimators (Kaplan-Meier survival and the artist-aware log-rank test of
    src/survival/estimators.py) are run again on the selected songs, so that the dashboard and the notebook can never disagree:
    with no filters on, the curve here equals the saved curve of Phase 6B.

    A "stay" is counted in chart days (1 on the arrival day). A song still on the chart on the last day of the file has not finished
    its stay; it is "censored" (exit_seen = False) and counts as at risk up to its last day only. The bands come from the ARTIST
    bootstrap: whole artists are drawn with replacement, because the songs of one artist arrive, stay and leave together.

WHAT HAPPENS WHEN IT RUNS
    The bootstrap draws the same artists for every group (fixed seed), so the bands are reproducible and two groups are compared on
    the same draws. A group with fewer than `min_songs` songs is listed in the summary with a note and gets no curve: a curve of
    five songs is not worth a picture. On this page the sidebar's place range applies to the DEBUT place of the song.

EXPECTED OUTCOME
    All new songs: S(3) = 63 %, S(7) = 43 %, S(30) = 19 %, median stay 6 chart days (Phase 6B).

MOTIVE
    Ask the question again for any group, with the same method and the same honesty about uncertainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dash.filters import FilterState, credited_track_ids
from src.survival.estimators import life_table, logrank_k, median_stay, restricted_mean, survival_grid

GROUPINGS = {
    "none": "All new songs",
    "debut": "Debut place",
    "release": "Release type",
    "explicit": "Explicit content",
    "collab": "Collaboration",
    "same_day": "Songs by the artist arriving the same day",
}
GROUP_ORDER = {
    "none": ["All new songs"],
    "debut": ["1 to 10", "11 to 20", "21 to 30", "31 to 40", "41 to 50"],
    "release": ["Single", "Album or compilation"],
    "explicit": ["Not explicit", "Explicit"],
    "collab": ["Solo", "Collaboration"],
    "same_day": ["1 song (alone)", "2 to 4 songs", "5 or more songs"],
}
AGES_OF_INTEREST = (1, 3, 7, 14, 30, 60)
HAZARD_EDGES = [0, 1, 2, 3, 7, 14, 30, 60, 120, 10_000]
SUMMARY_COLUMNS = ["group", "songs", "artists", "exits_seen", "still_on_chart", "median_stay", "median_low", "median_high"] + \
                  [f"s{a}" for a in AGES_OF_INTEREST] + ["restricted_mean_30", "note"]


def prepare_stays(stays: pd.DataFrame, dim_track: pd.DataFrame) -> pd.DataFrame:
    """The Phase 6B stays with the album type added and the arrival date as a real date."""
    table = stays.copy()
    table["entry_date"] = pd.to_datetime(table["entry_date"])
    table["exit_seen"] = table["exit_seen"].astype(bool)
    if "album_type" not in table.columns:
        table = table.merge(dim_track[["track_id", "album_type"]], on="track_id", how="left")
    return table


def filter_stays(stays: pd.DataFrame, state: FilterState, bridge: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """The stays of the songs that match the sidebar. The place range applies to the debut place of the song."""
    keep = (stays["entry_date"] >= pd.Timestamp(state.start)) & (stays["entry_date"] <= pd.Timestamp(state.end))
    keep &= (stays["entry_rank"] >= state.rank_low) & (stays["entry_rank"] <= state.rank_high)
    if state.album_types:
        keep &= stays["album_type"].isin(state.album_types)
    if state.explicit == "explicit":
        keep &= stays["is_explicit"].astype(bool)
    elif state.explicit == "clean":
        keep &= ~stays["is_explicit"].astype(bool)
    if state.collab == "collab":
        keep &= stays["is_collab"].astype(bool)
    elif state.collab == "solo":
        keep &= ~stays["is_collab"].astype(bool)
    if state.artists:
        keep &= stays["track_id"].isin(credited_track_ids(bridge, state.artists)) if bridge is not None \
            else stays["primary_artist"].isin(state.artists)
    if state.track_ids:
        keep &= stays["track_id"].isin(state.track_ids)
    return stays.loc[keep]


def group_labels(stays: pd.DataFrame, grouping: str) -> pd.Series:
    """The group of every song (a Series with the index of `stays`) under one of the six groupings."""
    if grouping not in GROUPINGS:
        raise ValueError(f"grouping must be one of {list(GROUPINGS)}")
    if grouping == "none":
        return pd.Series("All new songs", index=stays.index)
    if grouping == "debut":
        return stays["debut_band"].astype(str)
    if grouping == "release":
        return pd.Series(np.where(stays["is_single"].astype(bool), "Single", "Album or compilation"), index=stays.index)
    if grouping == "explicit":
        return pd.Series(np.where(stays["is_explicit"].astype(bool), "Explicit", "Not explicit"), index=stays.index)
    if grouping == "collab":
        return pd.Series(np.where(stays["is_collab"].astype(bool), "Collaboration", "Solo"), index=stays.index)
    size = stays["drop_size"].to_numpy()
    return pd.Series(np.where(size <= 1, "1 song (alone)", np.where(size <= 4, "2 to 4 songs", "5 or more songs")), index=stays.index)


def _artist_weights(artists: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """A (draws x songs) matrix: how many times the artist of each song was drawn in each bootstrap draw."""
    _, codes = np.unique(artists, return_inverse=True)
    n_artists = int(codes.max()) + 1
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, n_artists, n_artists), minlength=n_artists) for _ in range(n_boot)])
    return counts[:, codes].astype(float)


def live_survival(stays: pd.DataFrame, grouping: str = "none", max_age: int = 120, n_boot: int = 300, seed: int = 0,
                  min_songs: int = 10) -> dict:
    """
    The survival curve of every group of the selected stays.
    Returns {"curves", "summary", "logrank", "grouping"}:
      curves   long table: group, age (0 to max_age), survival, low, high (95 % artist-bootstrap band), songs
      summary  one row per group: songs, artists, exits seen, median stay (with band), S(1), S(3), S(7), S(14), S(30), S(60), a note
      logrank  {"p", "p_artist", "groups"} for two or more groups with enough songs, else None
    """
    curve_columns = ["group", "age", "survival", "low", "high", "songs"]
    empty = {"curves": pd.DataFrame(columns=curve_columns), "summary": pd.DataFrame(columns=SUMMARY_COLUMNS), "logrank": None,
             "grouping": grouping}
    if stays.empty:
        return empty
    labels = group_labels(stays, grouping).to_numpy()
    durations = stays["stay"].to_numpy(dtype=int)
    observed = stays["exit_seen"].to_numpy(dtype=bool)
    artists = stays["primary_artist"].to_numpy()
    weights = _artist_weights(artists, n_boot, seed)
    grid = np.arange(0, int(max_age) + 1)
    curves, summary, usable = [], [], []
    for group in GROUP_ORDER[grouping]:
        mask = labels == group
        n_songs = int(mask.sum())
        row = {"group": group, "songs": n_songs, "artists": int(pd.Series(artists[mask]).nunique()),
               "exits_seen": int(observed[mask].sum()), "still_on_chart": int((~observed[mask]).sum()), "note": ""}
        if n_songs < min_songs:
            row["note"] = f"fewer than {min_songs} songs, no curve"
            summary.append(row)
            continue
        usable.append(group)
        curve = survival_grid(durations[mask], observed[mask], grid)
        draw_weights = weights[:, mask]
        valid = draw_weights.sum(axis=1) > 0
        boot = survival_grid(durations[mask], observed[mask], grid, weights=draw_weights[valid])
        low, high = np.percentile(boot, [2.5, 97.5], axis=0)
        curves.append(pd.DataFrame({"group": group, "age": grid, "survival": curve, "low": low, "high": high, "songs": n_songs}))
        medians = median_stay(boot)
        row.update(median_stay=median_stay(curve), median_low=float(np.nanpercentile(medians, 2.5)) if np.isfinite(medians).any() else np.nan,
                   median_high=float(np.nanpercentile(medians, 97.5)) if np.isfinite(medians).any() else np.nan)
        for age in AGES_OF_INTEREST:
            row[f"s{age}"] = float(curve[min(age, len(curve) - 1)])
        row["restricted_mean_30"] = float(restricted_mean(curve, 30))
        summary.append(row)
    result = {"curves": pd.concat(curves, ignore_index=True) if curves else empty["curves"],
              "summary": pd.DataFrame(summary).reindex(columns=SUMMARY_COLUMNS), "logrank": None, "grouping": grouping}
    if len(usable) >= 2:
        keep = np.isin(labels, usable)
        test = logrank_k(durations[keep], observed[keep], labels[keep], clusters=artists[keep])
        result["logrank"] = {"p": test["p"], "p_artist": test.get("p_artist", np.nan), "groups": usable, "n_artists": test.get("n_clusters")}
    return result


def live_hazard(stays: pd.DataFrame, grouping: str = "none", edges=None, min_songs: int = 10) -> pd.DataFrame:
    """
    For every group and age window (chart day 1, 2, 3, 4 to 7 ...): songs reaching the window, exits and the chance that a chart day in
    the window is the song's last (the hazard). Columns: group, window, from_day, to_day, songs_reaching, exits, chart_days, hazard.
    """
    edges = HAZARD_EDGES if edges is None else edges
    if stays.empty:
        return pd.DataFrame(columns=["group", "window", "from_day", "to_day", "songs_reaching", "exits", "chart_days", "hazard"])
    labels = group_labels(stays, grouping).to_numpy()
    durations = stays["stay"].to_numpy(dtype=int)
    observed = stays["exit_seen"].to_numpy(dtype=bool)
    tables = []
    for group in GROUP_ORDER[grouping]:
        mask = labels == group
        if mask.sum() < min_songs:
            continue
        table = life_table(durations[mask], observed[mask], edges)
        tables.append(table.assign(group=group)[["group", "window", "from_day", "to_day", "songs_reaching", "exits", "chart_days", "hazard"]])
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=["group", "window", "from_day", "to_day", "songs_reaching", "exits", "chart_days", "hazard"])
