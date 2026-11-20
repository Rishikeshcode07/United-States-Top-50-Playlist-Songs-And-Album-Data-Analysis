"""
app/views/explicit.py  -  page 13: explicit and non-explicit songs.

WHAT THIS FILE IS
    Side-by-side panels for explicit and non-explicit songs: how big their share of the chart is and how it changed, in which places
    they sit, and how they compare on days on the chart, average place, popularity, volatility and chart points. A live comparison
    (medians, Cliff's delta, p-value) sits next to the Phase 5 results (H1, H10) so that a filtered comparison is never mistaken for a test.

WHY WE NEED IT
    "Explicit vs non-explicit panels" are a core module of the brief, and Explicit Content Share is one of its six KPIs.

WHAT HAPPENS WHEN IT RUNS
    The explicit share of the chart is a share of SONG-DAYS (a song on the chart for one day). Comparisons between the two groups use
    one number per SONG, because a song that stays 300 days must not count 300 times. Songs of one artist tend to share the flag, so
    the live p-value treats songs as independent and is too eager; the artist-aware result of Phase 5 is shown next to it.

EXPECTED OUTCOME
    The share over time and by place, a panel for each group, a chosen measure compared with its distribution, and the formal result.

MOTIVE
    Describe explicit content on the chart without claiming that the flag causes anything.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, kpi_cards, page_header, plot, section, stop_if_empty, table
from app.ui.reference import hypothesis_cards
from app.ui.state import current
from src.dash import compare as C
from src.dash import movement as V
from src.dash import texts as T

# measure -> (column of the song table, label, number format, True when a LOW value is better)
MEASURES = {
    "Days on the chart": ("days_on_chart", ".0f", False), "Average place (1 = top)": ("avg_rank", ".1f", True), "Best place reached": ("best_rank", ".0f", True),
    "Average popularity": ("avg_popularity", ".1f", False), "Rank volatility": ("rank_volatility_index", ".1f", True), "Chart points": ("chart_points", ",.0f", False),
}


def render() -> None:
    ctx = current()
    page_header("Content", "Explicit and non-explicit songs",
                "How much of the chart carries the explicit flag, where those songs sit and how they compare with songs that do not.", ctx.chips)
    stop_if_empty(ctx.rows)
    rows, songs = ctx.rows, ctx.songs().copy()
    songs["group"] = songs["is_explicit"].map({True: "Explicit", False: "Not explicit"})
    yes, no = songs[songs["is_explicit"]], songs[~songs["is_explicit"]]
    points = songs.groupby("group")["chart_points"].sum()
    share_points = float(points.get("Explicit", 0) / points.sum()) if points.sum() else float("nan")
    top10 = {g: float((part["best_rank"] <= 10).mean()) if len(part) else float("nan") for g, part in (("Explicit", yes), ("Not explicit", no))}
    kpi_cards(columns=5, items=[
        {"label": "Explicit song-days", "value": T.pct(rows["is_explicit"].astype(bool).mean(), 1), "tone": "orange", "sub": "of the places that match the filters"},
        {"label": "Explicit songs", "value": T.pct(len(yes) / len(songs), 1), "tone": "yellow", "sub": f"{T.n(len(yes))} of {T.n(len(songs))} songs"},
        {"label": "Share of chart points", "value": T.pct(share_points, 1), "tone": "pink", "sub": "earned by explicit songs"},
        {"label": "Median stay", "value": f"{T.num(yes['days_on_chart'].median(), 0)} / {T.num(no['days_on_chart'].median(), 0)}", "unit": "days", "tone": "aqua", "sub": "explicit / not explicit"},
        {"label": "Reached the top 10", "value": f"{T.pct(top10['Explicit'], 0)} / {T.pct(top10['Not explicit'], 0)}", "tone": "violet", "sub": "of songs, explicit / not explicit"},
    ])

    # ---- share over time and by place ---------------------------------------------------------------------------
    period = (st.segmented_control("Share over time, by", ["Week", "Month"], default="Month", key="ex_period") or "Month").lower()
    a, b = st.columns([3, 2])
    with a:
        chart = C.share_by_period(rows, "is_explicit", period).assign(kind="Whole chart (song-days)")
        arrivals = C.share_by_period(rows[rows["movement"] == "New entry"], "is_explicit", period).assign(kind="First-time arrivals")
        both = pd.concat([chart, arrivals], ignore_index=True).assign(share_pct=lambda d: d["share"] * 100)
        fig = ch.line_chart(both, "period", "share_pct", group="kind", title=f"Explicit share, by {period}", order=["Whole chart (song-days)", "First-time arrivals"],
                            colors={"Whole chart (song-days)": theme.ORANGE, "First-time arrivals": theme.VIOLET}, y_title="% explicit", fmt=".0f", suffix="%", markers=period == "month", height=380)
        fig.update_xaxes(tickformat="%b %Y")
        fig.update_yaxes(range=[0, 100])
        plot(fig, key="ex_time")
    with b:
        band = C.share_by_band(rows, "is_explicit").assign(share_pct=lambda d: d["share"] * 100)
        band["band"] = pd.Categorical(band["band"], categories=V.BAND_LABELS, ordered=True)
        fig = ch.vbar(band.sort_values("band"), "band", "share_pct", "Explicit share, by band of places", colors={"share_pct": theme.ORANGE}, y_title="% of song-days", x_title="Chart places",
                      fmt=".0f", suffix="%", show_text=True, height=380)
        fig.update_yaxes(range=[0, 100])
        plot(fig, key="ex_band")
    top_band = band.sort_values("share_pct", ascending=False).iloc[0]
    callout(f"**{T.pct(rows['is_explicit'].astype(bool).mean(), 0)}** of the song-days carry the explicit flag. The share is highest at places **{top_band['band']}** "
            f"(**{T.num(top_band['share_pct'], 0)}%**) and lowest at places **{band.sort_values('share_pct').iloc[0]['band']}** (**{T.num(band['share_pct'].min(), 0)}%**).")

    # ---- two panels ---------------------------------------------------------------------------------------------
    section("Two panels", "The same measures for each group. One number per song.")
    profile = C.group_profile(songs, "group", order=["Explicit", "Not explicit"]).set_index("group")
    left, right = st.columns(2)
    for column, group, color in ((left, "Explicit", "orange"), (right, "Not explicit", "blue")):
        with column:
            if group not in profile.index:
                st.info(f"No {group.lower()} song in this selection.")
                continue
            r = profile.loc[group]
            kpi_cards(columns=2, items=[
                {"label": f"{group} songs", "value": T.n(int(r["songs"])), "tone": color, "sub": f"{T.pct(r['share_of_songs'], 0)} of the songs; {T.pct(r['share_of_points'], 0)} of the points"},
                {"label": "Median days on the chart", "value": T.num(r["median_days"], 0), "tone": color, "sub": f"mean {T.num(r['mean_days'], 1)} days"},
                {"label": "Average place", "value": T.num(r["mean_avg_rank"], 1), "tone": color, "sub": f"{T.pct(r['share_top10'], 0)} reached the top 10"},
                {"label": "Average popularity", "value": T.num(r["mean_popularity"], 1), "tone": color, "sub": f"median volatility {T.num(r['median_volatility'], 1)}"},
            ])

    # ---- one measure, compared -------------------------------------------------------------------------------------
    section("Compare one measure", "Pick a measure. The boxes show the middle half of the songs of each group; the line inside the box is the median.")
    measure = st.columns([2, 3])[0].selectbox("Measure", list(MEASURES), key="ex_measure")
    column, fmt, low_is_better = MEASURES[measure]
    summary = C.two_group_summary(yes[column], no[column])
    a, b = st.columns([2, 3])
    with a:
        fig = ch.box_chart(songs, column, "group", ["Explicit", "Not explicit"], {"Explicit": theme.ORANGE, "Not explicit": theme.BLUE}, measure, height=380, y_title=measure)
        if column in ("days_on_chart", "chart_points"):
            ch.log_axis(fig, ticks=None if column == "days_on_chart" else [10, 30, 100, 300, 1000, 3000, 10000])
        plot(fig, key="ex_box")
    with b:
        delta, p = summary["delta"], summary["p"]
        direction = "higher" if (summary["median_difference"] or 0) > 0 else "lower"
        if low_is_better:
            direction = f"{direction} (lower is better)"
        kpi_cards(columns=3, items=[
            {"label": "Median, explicit", "value": T.num(summary["median_a"], 1), "tone": "orange", "sub": f"{T.n(summary['n_a'])} songs"},
            {"label": "Median, not explicit", "value": T.num(summary["median_b"], 1), "tone": "blue", "sub": f"{T.n(summary['n_b'])} songs"},
            {"label": "Cliff's delta", "value": T.num(delta, 2), "tone": "violet", "sub": f"{summary['size'] or 'no size word'}; 0 = no difference, +1 = explicit always larger"},
        ])
        callout(f"On **{measure.lower()}** the median of explicit songs is **{T.num(summary['median_a'], 1)}** and of other songs **{T.num(summary['median_b'], 1)}** "
                f"(difference {T.signed(summary['median_difference'], 1)}, {direction}). Cliff's delta is **{T.num(delta, 2)}**, a **{summary['size'] or 'n/a'}** difference. "
                f"The Mann-Whitney p-value is **{T.p_text(p)}**, but it treats songs as independent although songs of one artist usually share the flag, so it is too eager: read the Phase 5 result below.")

    hypothesis_cards(["H1", "H10"])

    section("Who makes the explicit songs")
    makers = yes.groupby("primary_artist").agg(songs=("track_id", "size"), median_days=("days_on_chart", "median"), points=("chart_points", "sum")).sort_values(["songs", "points"], ascending=False).head(12)
    all_songs = songs.groupby("primary_artist").size()
    makers["share_explicit"] = (makers["songs"] / all_songs.reindex(makers.index) * 100).round(0)
    shown = makers.reset_index().rename(columns={"primary_artist": "Artist", "songs": "Explicit songs", "median_days": "Median days", "points": "Chart points", "share_explicit": "% of the artist's songs"})
    table(shown, column_config={"% of the artist's songs": st.column_config.ProgressColumn("% of the artist's songs", min_value=0, max_value=100, format="%d%%")}, height=min(480, 38 * len(shown) + 40))
