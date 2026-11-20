"""
app/views/release.py  -  page 14: release type, collaboration, release size and song length.

WHAT THIS FILE IS
    A comparison of groups of songs on the same measures: singles against album tracks, solo songs against collaborations, songs from
    small and large releases, short songs against long ones. The person picks the grouping and the measure; the page draws the groups,
    the spread inside each group, the share of the chart each group held over time, and a pairwise comparison with Cliff's delta.

WHY WE NEED IT
    The brief asks for an album type toggle and attribute panels. These attributes are known on the day a song is released, so they are
    the natural "what kind of song is it?" questions of a playlist manager.

WHAT HAPPENS WHEN IT RUNS
    Every group comparison uses one number per song. Groups with fewer than five songs are left out of the charts (the table keeps them)
    because a mean or a median of two songs says nothing. The Phase 5 tests that belong to the chosen grouping are shown below the live
    comparison, because those are the ones with artist-aware standard errors and a correction for ten tests.

EXPECTED OUTCOME
    A fair, labelled comparison of groups, with the formal result beside it.

MOTIVE
    Compare what can be compared, and say how strong the evidence is.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, download_csv, kpi_cards, page_header, plot, section, stop_if_empty, table
from app.ui.reference import hypothesis_cards
from app.ui.state import current
from src.dash import compare as C
from src.dash import metrics as M
from src.dash import texts as T

SIZE_ORDER = ["1 track", "2-3", "4-7", "8-15", "16-40", "41+"]
GROUPINGS = {
    "Release type": {"order": ["Single", "Album", "Compilation"], "tests": ["H2", "H10"]},
    "Collaboration": {"order": ["Solo", "Collaboration"], "tests": ["H9"]},
    "Release size": {"order": SIZE_ORDER, "tests": []},
    "Song length": {"order": list(C.DURATION_LABELS), "tests": ["H7"]},
}
# measure -> (column of the group profile, factor, number format, unit label, song-level column for the spread, lower is better)
MEASURES = {
    "Median days on the chart": ("median_days", 1, ".0f", "days", "days_on_chart", False),
    "Average place (1 = top)": ("mean_avg_rank", 1, ".1f", "place", "avg_rank", True),
    "Share that reached the top 10": ("share_top10", 100, ".0f", "% of songs", "days_on_chart", False),
    "Share that reached number one": ("share_number_one", 100, ".1f", "% of songs", "days_on_chart", False),
    "Average popularity": ("mean_popularity", 1, ".1f", "points", "avg_popularity", False),
    "Median rank volatility": ("median_volatility", 1, ".1f", "index", "rank_volatility_index", True),
    "Share of the songs": ("share_of_songs", 100, ".0f", "% of songs", "days_on_chart", False),
    "Share of the chart points": ("share_of_points", 100, ".0f", "% of points", "days_on_chart", False),
}
MIN_SONGS = 5


def _group_column(songs: pd.DataFrame, grouping: str) -> pd.Series:
    if grouping == "Release type":
        return songs["album_type"].str.title()
    if grouping == "Collaboration":
        return songs["is_collab"].map({True: "Collaboration", False: "Solo"})
    if grouping == "Release size":
        return songs["release_size_band"].astype(str)
    return pd.Series(C.duration_band(songs["duration_min"]).to_numpy(), index=songs.index)


def render() -> None:
    ctx = current()
    page_header("Content", "Release type, collaboration and length",
                "Compare groups of songs on the same measures: singles and album tracks, solo songs and collaborations, small and large releases, short and long songs.", ctx.chips)
    stop_if_empty(ctx.rows)
    rows, songs = ctx.rows, ctx.songs().copy()

    c1, c2 = st.columns([2.6, 2.4], vertical_alignment="bottom")
    grouping = c1.segmented_control("Compare songs by", list(GROUPINGS), default="Release type", key="rl_grouping") or "Release type"
    measure = c2.selectbox("Measure", list(MEASURES), key="rl_measure")
    order = GROUPINGS[grouping]["order"]
    songs["group"] = _group_column(songs, grouping)
    profile = C.group_profile(songs, "group", order=order)
    profile = profile[profile["group"].isin(order)].reset_index(drop=True)
    big = profile[profile["songs"] >= MIN_SONGS]
    if big.empty:
        callout("No group has enough songs in this selection. Widen the dates or the other filters.", title="Too few songs", tone="warn")
        st.stop()
    left_out = len(profile) - len(big)
    column, factor, fmt, unit, spread_col, low_is_better = MEASURES[measure]
    best = big.sort_values(column, ascending=low_is_better).iloc[0]

    kpi_cards(columns=4, items=[
        {"label": "Groups compared", "value": T.n(len(big)), "tone": "blue", "sub": f"{T.n(int(big['songs'].sum()))} songs" + (f"; {left_out} small group left out" if left_out else "")},
        {"label": "Largest group", "value": str(big.sort_values("songs", ascending=False).iloc[0]["group"]), "tone": "aqua", "sub": f"{T.pct(big['share_of_songs'].max(), 0)} of the songs"},
        {"label": f"Best on '{measure.lower()}'", "value": str(best["group"]), "tone": "green", "sub": f"{T.num(best[column] * factor, 1 if '.1' in fmt else 0)} {unit}"},
        {"label": "Songs with the longest stay", "value": str(big.sort_values("median_days", ascending=False).iloc[0]["group"]), "tone": "violet",
         "sub": f"median {T.num(big['median_days'].max(), 0)} days"},
    ])

    a, b = st.columns(2)
    with a:
        data = big.assign(value=big[column] * factor, label=big["group"] + " (" + big["songs"].astype(int).astype(str) + ")")
        fig = ch.vbar(data, "label", "value", f"{measure}, by group (songs in brackets)", colors={"value": theme.BLUE}, y_title=unit, fmt=fmt, show_text=True, height=400)
        plot(fig, key="rl_bars")
    with b:
        spread = songs[songs["group"].isin(big["group"])]
        palette = dict(zip(big["group"], theme.CATEGORICAL))
        fig = ch.box_chart(spread, spread_col, "group", [g for g in order if g in set(big["group"])], palette, "The spread of the songs inside each group", height=400,
                           y_title={"days_on_chart": "Days on the chart (log scale)", "avg_rank": "Average place", "avg_popularity": "Average popularity", "rank_volatility_index": "Rank volatility"}[spread_col])
        if spread_col == "days_on_chart":
            ch.log_axis(fig)
        plot(fig, key="rl_box")
    callout(f"On **{measure.lower()}**, **{best['group']}** comes out best (**{T.num(best[column] * factor, 1 if '.1' in fmt else 0)} {unit}**) and "
            f"**{big.sort_values(column, ascending=not low_is_better).iloc[0]['group']}** last (**{T.num(big.sort_values(column, ascending=not low_is_better).iloc[0][column] * factor, 1 if '.1' in fmt else 0)} {unit}**). "
            "Differences between groups are differences in what the songs did; the group itself is not shown to be the reason.")

    # ---- share of the chart over time ------------------------------------------------------------------------------
    section("How much of the chart each group held", "Share of the song-days of every month. The shares of a month add up to 100%.")
    mapping = songs.set_index("track_id")["group"]
    timed = M.add_period(rows.assign(group=rows["track_id"].map(mapping)).dropna(subset=["group"]), "month")
    shares = timed.groupby(["period", "group"]).size().unstack(fill_value=0)
    shares = shares.div(shares.sum(axis=1), axis=0).mul(100).reset_index()
    series = [g for g in order if g in shares.columns]
    fig = ch.vbar(shares, "period", series, "Share of the song-days, by month", barmode="stack", colors=dict(zip(series, theme.CATEGORICAL)), y_title="% of the song-days", fmt=".0f", height=380)
    fig.update_xaxes(tickformat="%b %Y")
    plot(fig, key="rl_time")

    # ---- two groups, head to head --------------------------------------------------------------------------------
    section("Two groups, head to head", "One number per song. Cliff's delta says how often a song of group A beats a song of group B on the measure (0 = no difference).")
    names = big.sort_values("songs", ascending=False)["group"].tolist()
    if len(names) >= 2:
        x1, x2, x3 = st.columns(3)
        group_a = x1.selectbox("Group A", names, index=0, key="rl_a")
        group_b = x2.selectbox("Group B", [n for n in names if n != group_a], index=0, key="rl_b")
        song_measure = x3.selectbox("Song-level measure", ["Days on the chart", "Average place", "Average popularity", "Rank volatility", "Chart points"], key="rl_song_measure")
        col = {"Days on the chart": "days_on_chart", "Average place": "avg_rank", "Average popularity": "avg_popularity", "Rank volatility": "rank_volatility_index", "Chart points": "chart_points"}[song_measure]
        summary = C.two_group_summary(songs.loc[songs["group"] == group_a, col], songs.loc[songs["group"] == group_b, col])
        kpi_cards(columns=4, items=[
            {"label": f"Median, {group_a}", "value": T.num(summary["median_a"], 1), "tone": "blue", "sub": f"{T.n(summary['n_a'])} songs"},
            {"label": f"Median, {group_b}", "value": T.num(summary["median_b"], 1), "tone": "orange", "sub": f"{T.n(summary['n_b'])} songs"},
            {"label": "Cliff's delta (A against B)", "value": T.num(summary["delta"], 2), "tone": "violet", "sub": summary["size"] or "no size word"},
            {"label": "Live p-value", "value": T.p_text(summary["p"]), "tone": "ink", "sub": "treats songs as independent"},
        ])
        st.caption("The live p-value is too eager when songs of one artist behave alike. The Phase 5 tests below allow for that.")
    hypothesis_cards(GROUPINGS[grouping]["tests"]) if GROUPINGS[grouping]["tests"] else st.caption(f"Phase 5 has no formal test for {grouping.lower()}; read the live comparison as description only.")

    section("The numbers behind the charts")
    shown = pd.DataFrame({"Group": profile["group"], "Songs": profile["songs"], "Share of songs (%)": (profile["share_of_songs"] * 100).round(1), "Median days": profile["median_days"],
                          "Mean days": profile["mean_days"].round(1), "Average place": profile["mean_avg_rank"].round(1), "Reached top 10 (%)": (profile["share_top10"] * 100).round(1),
                          "Reached #1 (%)": (profile["share_number_one"] * 100).round(1), "Average popularity": profile["mean_popularity"].round(1),
                          "Median volatility": profile["median_volatility"].round(1), "Share of points (%)": (profile["share_of_points"] * 100).round(1)})
    table(shown, height=38 * len(shown) + 40)
    download_csv(shown, f"songs_by_{grouping.lower().replace(' ', '_')}.csv", key="rl_csv")
