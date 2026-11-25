"""
src/dash/compare.py  -  comparing groups of songs, and popularity against place.

WHAT THIS FILE IS
    group_profile                 for every group (explicit / not explicit, single / album, solo / collaboration ...): songs, typical days on
                                  the chart, average place, share of songs that reached the top 10, popularity, share of all chart points
    two_group_summary             the difference between two groups of songs: medians, Mann-Whitney p-value, Cliff's delta and its size word
    spearman                      rank correlation with its p-value and the number of pairs
    position_profile              popularity by chart place (the typical popularity of place 1, place 2 ... place 50)
    daily_rank_popularity_corr    the rank correlation between chart score and popularity of every single day
    share_by_period              the share of rows with a given property (for example explicit) in every month / week / day
    share_by_band                the same share by band of places (1 to 10, 11 to 20 ...)
    add_pop_change                adds the change of popularity over the last 7 snapshots to every row (to compare with the 7-day rank gain)
    duration_band                short / typical / long song-length bands

WHY WE NEED IT
    The pages "Explicit content", "Release and collaboration" and "Popularity and rank" all ask the same kind of question: do two
    groups of songs behave differently? One careful comparison function is better than four different ones. Two rules of the project
    are built in:
      * The unit is the SONG, not the song-day. A song that stays 300 days must not count 300 times in a comparison of songs.
      * A difference is described with an effect size (Cliff's delta: how often a random song of one group beats a random song of the
        other, from -1 to +1) and not only with a p-value. The p-value here treats songs as independent; songs of one artist are not,
        so the pages say plainly that the artist-aware tests are in the Evidence page (Phase 5 and 6B).

WHAT HAPPENS WHEN IT RUNS
    Groups with fewer than two songs give empty (NaN) statistics instead of errors. Spearman's coefficient is NaN when one variable
    does not vary.

EXPECTED OUTCOME
    On the whole file the singles-versus-albums comparison of days on the chart gives a small positive delta (singles stay longer),
    in line with hypothesis H2 of Phase 5.

MOTIVE
    Describe differences with sizes and with honesty about what the numbers can and cannot say.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.dash.metrics import add_period
from src.dash.movement import place_band
from src.inference.results import effect_label

DURATION_EDGES = [0.0, 2.5, 3.5, 4.5, 100.0]
DURATION_LABELS = ["under 2.5 min", "2.5 to 3.5 min", "3.5 to 4.5 min", "4.5 min or more"]


# ====================================================================================================== effect sizes
def spearman(x, y) -> dict:
    """Spearman's rho, its p-value and the number of complete pairs. rho and p are NaN with fewer than 3 pairs or no variation."""
    pairs = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    n = int(len(pairs))
    if n < 3 or pairs["x"].nunique() < 2 or pairs["y"].nunique() < 2:
        return {"rho": np.nan, "p": np.nan, "n": n, "size": ""}
    result = stats.spearmanr(pairs["x"], pairs["y"])
    rho = float(result.statistic)
    return {"rho": rho, "p": float(result.pvalue), "n": n, "size": effect_label("correlation", rho)}


def cliffs_delta(a, b) -> float:
    """How often a random member of `a` is larger than a random member of `b`, minus how often it is smaller (from -1 to +1)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    u = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").statistic
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def two_group_summary(a, b) -> dict:
    """
    The difference between two groups of numbers (one number per song). Returns the sizes, medians and means of both groups, the
    difference of the medians, the Mann-Whitney p-value, Cliff's delta (group a against group b) and its size word.
    """
    a = pd.Series(np.asarray(a, dtype=float)).dropna()
    b = pd.Series(np.asarray(b, dtype=float)).dropna()
    out = {"n_a": int(len(a)), "n_b": int(len(b)), "median_a": float(a.median()) if len(a) else np.nan,
           "median_b": float(b.median()) if len(b) else np.nan, "mean_a": float(a.mean()) if len(a) else np.nan,
           "mean_b": float(b.mean()) if len(b) else np.nan, "p": np.nan, "delta": np.nan, "size": ""}
    out["median_difference"] = out["median_a"] - out["median_b"]
    if len(a) >= 2 and len(b) >= 2 and (a.nunique() + b.nunique() > 2):
        out["p"] = float(stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").pvalue)
        out["delta"] = cliffs_delta(a, b)
        out["size"] = effect_label("cliffs_delta", out["delta"])
    return out


# ======================================================================================================== groups
def group_profile(songs: pd.DataFrame, by, labels: "dict | None" = None, order=None) -> pd.DataFrame:
    """
    One row per group of songs (the column or Series `by`; `labels` renames the group values, `order` fixes the row order).
    Columns: group, songs, share_of_songs, median_days, mean_days, mean_avg_rank, share_top10, share_number_one, mean_popularity,
    median_volatility, share_of_points.
    """
    columns = ["group", "songs", "share_of_songs", "median_days", "mean_days", "mean_avg_rank", "share_top10", "share_number_one",
               "mean_popularity", "median_volatility", "share_of_points"]
    if songs.empty:
        return pd.DataFrame(columns=columns)
    key = songs[by] if isinstance(by, str) else pd.Series(np.asarray(by), index=songs.index)
    if labels:
        key = key.map(labels).fillna(key)
    frame = songs.assign(_group=key.to_numpy())
    grouped = frame.groupby("_group", sort=False)
    table = pd.DataFrame({
        "songs": grouped.size(), "median_days": grouped["days_on_chart"].median(), "mean_days": grouped["days_on_chart"].mean(),
        "mean_avg_rank": grouped["avg_rank"].mean(), "share_top10": grouped["best_rank"].apply(lambda s: float((s <= 10).mean())),
        "share_number_one": grouped["days_at_number_one"].apply(lambda s: float((s > 0).mean())),
        "mean_popularity": grouped["avg_popularity"].mean(), "median_volatility": grouped["rank_volatility_index"].median(),
        "chart_points": grouped["chart_points"].sum()})
    table["share_of_songs"] = table["songs"] / table["songs"].sum()
    table["share_of_points"] = table["chart_points"] / table["chart_points"].sum()
    table = table.drop(columns="chart_points")
    if order is not None:
        table = table.reindex([g for g in order if g in table.index])
    table.index.name = "group"
    return table.reset_index()[columns]


def duration_band(minutes) -> pd.Series:
    """Song length in bands: under 2.5, 2.5 to 3.5, 3.5 to 4.5 and 4.5 minutes or more."""
    return pd.cut(pd.Series(minutes), bins=DURATION_EDGES, labels=DURATION_LABELS, right=False).astype("object")


# ============================================================================================ popularity and place
def position_profile(rows: pd.DataFrame) -> pd.DataFrame:
    """For every chart place: the number of song-days, the mean, median and quartiles of popularity, and the explicit share."""
    columns = ["position", "n", "mean", "median", "p25", "p75", "explicit_share"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    grouped = rows.groupby("position")
    table = pd.DataFrame({"n": grouped.size(), "mean": grouped["popularity_clean"].mean(), "median": grouped["popularity_clean"].median(),
                          "p25": grouped["popularity_clean"].quantile(0.25), "p75": grouped["popularity_clean"].quantile(0.75),
                          "explicit_share": grouped["is_explicit"].apply(lambda s: float(s.astype(bool).mean()))})
    return table.reset_index()[columns]


def daily_rank_popularity_corr(rows: pd.DataFrame, min_songs: int = 5) -> pd.DataFrame:
    """
    For every date: Spearman's rho between the chart score (51 - place) and the popularity of the songs on the chart that day, so
    that a POSITIVE rho means "the more popular songs sit higher". Columns: date, rho, n.
    """
    pairs = rows[["date", "position", "popularity_clean"]].dropna()
    if pairs.empty:
        return pd.DataFrame(columns=["date", "rho", "n"])
    by_date = pairs.groupby("date")
    score = by_date["position"].rank(method="average", ascending=False)          # place 1 gets the highest score
    popularity = by_date["popularity_clean"].rank(method="average")
    frame = pairs[["date"]].assign(_s=score, _p=popularity)
    grouped = frame.groupby("date")
    dx = frame["_s"] - grouped["_s"].transform("mean")
    dy = frame["_p"] - grouped["_p"].transform("mean")
    numerator = (dx * dy).groupby(frame["date"]).sum()
    denominator = np.sqrt((dx ** 2).groupby(frame["date"]).sum() * (dy ** 2).groupby(frame["date"]).sum())
    table = pd.DataFrame({"rho": numerator / denominator.where(denominator > 0), "n": grouped.size()})
    return table.loc[table["n"] >= min_songs].reset_index()


def add_pop_change(playlist: pd.DataFrame, lag: int = 7) -> pd.DataFrame:
    """
    The playlist table with one more column, `pop_change_lag`: the popularity of the song today minus its popularity `lag` snapshots
    ago (empty when the song was not on the chart exactly `lag` snapshots ago). Calculated on the full table so that filters
    cannot misalign the days; apply the filters afterwards.
    """
    before = playlist[["track_id", "snapshot_index", "popularity_clean"]].copy()
    before["snapshot_index"] = before["snapshot_index"] + int(lag)
    before = before.rename(columns={"popularity_clean": "_pop_before"})
    merged = playlist.merge(before, on=["track_id", "snapshot_index"], how="left")
    merged["pop_change_lag"] = merged["popularity_clean"] - merged["_pop_before"]
    merged.index = playlist.index
    return merged.drop(columns="_pop_before")


# ====================================================================================================== shares
def share_by_period(rows: pd.DataFrame, flag: str, period: str = "month") -> pd.DataFrame:
    """The share of rows with `flag` True in every day / week / month. Columns: period, share, rows."""
    if rows.empty:
        return pd.DataFrame(columns=["period", "share", "rows"])
    grouped = add_period(rows, period).groupby("period")[flag]
    return pd.DataFrame({"share": grouped.apply(lambda s: float(s.astype(bool).mean())), "rows": grouped.size()}).reset_index()


def share_by_band(rows: pd.DataFrame, flag: str) -> pd.DataFrame:
    """The share of rows with `flag` True in every band of places. Columns: band, share, rows."""
    if rows.empty:
        return pd.DataFrame(columns=["band", "share", "rows"])
    band = place_band(rows["position"]).to_numpy()
    grouped = rows.assign(_band=band).groupby("_band", observed=True)[flag]
    table = pd.DataFrame({"share": grouped.apply(lambda s: float(s.astype(bool).mean())), "rows": grouped.size()})
    table.index.name = "band"
    return table.reset_index()
