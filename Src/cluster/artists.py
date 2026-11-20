"""
src/cluster/artists.py  -  one row per artist, and a small principal-component analysis.

WHAT THIS FILE IS
    artist_table   for every artist: songs on the chart, chart days, days with at least one song, best and average place, share of chart days
                   in the top 10, chart days per song (staying power) and the most songs on the chart on one day (bursts)
    artist_matrix  the six columns the grouping uses, on a scale where a ratio counts as much as a difference (logs for counts), standardized
    pca            principal components by the singular value decomposition: scores, shares of the variance, loadings

WHY WE NEED IT
    The brief asks for the artist's footprint on the chart (number of songs, days on the chart, dominance over time). These are the
    columns that say it, and the PCA shows whether the artists form separate types or one continuum.

WHAT HAPPENS WHEN IT RUNS
    Counts are logged (an artist with 90 songs is not 90 times a different kind of artist than one with 1), the six columns are
    standardized to mean 0 and standard deviation 1, and the PCA is computed on them.

EXPECTED OUTCOME
    A table with one row per artist; a PCA whose first two components carry most of the variance if artists differ mainly by size and staying power.

MOTIVE
    Say what an artist's presence on the chart looks like in numbers, and let the data say whether artists come in types.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["log_songs", "log_days_per_song", "mean_place", "top10_share", "log_days_present", "log_peak_songs"]


def artist_table(rows: pd.DataFrame) -> pd.DataFrame:
    """rows: one row per song per day (date, track_id, position, primary_artist). Returns one row per artist."""
    grouped = rows.groupby("primary_artist")
    table = pd.DataFrame({"songs": grouped["track_id"].nunique(), "chart_days": grouped.size(), "days_present": grouped["date"].nunique(),
                          "best_place": grouped["position"].min(), "mean_place": grouped["position"].mean(),
                          "top10_share": grouped["position"].apply(lambda p: float((p <= 10).mean()))})
    table["days_per_song"] = table["chart_days"] / table["songs"]
    table["peak_songs"] = rows.groupby(["primary_artist", "date"]).size().groupby("primary_artist").max()
    return table.sort_index()


def artist_matrix(table: pd.DataFrame):
    """(standardized matrix of the six FEATURES, the unstandardized frame). Counts are logged first."""
    frame = pd.DataFrame({"log_songs": np.log(table["songs"]), "log_days_per_song": np.log(table["days_per_song"]), "mean_place": table["mean_place"],
                          "top10_share": table["top10_share"], "log_days_present": np.log(table["days_present"]),
                          "log_peak_songs": np.log(table["peak_songs"])})[FEATURES]
    values = frame.to_numpy(dtype=float)
    return (values - values.mean(axis=0)) / values.std(axis=0), frame


def pca(Z: np.ndarray):
    """Principal components of the (already centred) matrix Z: (scores, share of variance per component, loadings[component, column])."""
    u, s, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    for i in range(len(vt)):                                              # the sign of a component is arbitrary: make its largest loading positive
        if vt[i][np.abs(vt[i]).argmax()] < 0:
            vt[i], u[:, i] = -vt[i], -u[:, i]
    return u * s, s ** 2 / (s ** 2).sum(), vt
