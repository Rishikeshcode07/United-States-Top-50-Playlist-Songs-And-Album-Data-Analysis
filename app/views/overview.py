"""
app/views/overview.py  -  page 1: the executive summary.

WHAT THIS FILE IS
    Page 1 of the dashboard. It shows the six KPIs of the brief for the current filters (Days on Chart, Average Rank, Rank
    Volatility Index, Popularity Score Trend, Artist Dominance Index, Explicit Content Share), the artists that hold the chart,
    how much of the chart changes day by day, the chart of the last chosen day, six findings of the analysis, and a guide to the
    other pages.

WHY WE NEED IT
    The first screen has to answer "what is going on in this chart?" in ten seconds, and invite the reader to dig deeper.

WHAT HAPPENS WHEN IT RUNS
    `render()` reads the filtered rows from the shared state and takes every number from src/dash. The two charts about the chart
    as a whole (turnover, daily popularity) use the dates only, as explained in src/dash/filters.py. Nothing is written to disk.

EXPECTED OUTCOME
    A summary page whose numbers, charts and sentence under the charts all follow the sidebar filters.

MOTIVE
    A summary that is true for any filter, not a fixed picture.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, cards, kpi_cards, page_header, plot, section, stop_if_empty, table
from app.ui.state import current, report
from src.dash import compare as C
from src.dash import metrics as M
from src.dash import movement as V
from src.dash import texts as T

MOVE_TEXT = {"Up": "up", "Down": "down", "Same": "same", "New entry": "new", "Re-entry": "back", "Chart start": "start"}


def _move_label(row) -> str:
    kind = row["movement"]
    if kind in ("Up", "Down") and pd.notna(row["rank_change"]):
        return f"{MOVE_TEXT[kind]} {abs(int(row['rank_change']))}"
    return MOVE_TEXT.get(kind, kind)


def render() -> None:
    ctx = current()
    rows, window, summary = ctx.rows, ctx.window, ctx.summary
    page_header("Overview", "Executive summary",
                "The six KPIs of the brief, recalculated for the filters in the sidebar. Everything here describes what happened in the chart; "
                "nothing is a forecast.", ctx.chips)
    stop_if_empty(rows)
    songs, artists = ctx.songs(), ctx.artists()
    head = M.headline(rows, window, summary, songs, artists)
    trend = head["popularity_trend"]
    top10_share = float(rows["in_top10"].mean())

    kpi_cards(columns=6, items=[
        {"label": "Days on chart", "value": T.n(head["median_days_on_chart"]), "unit": "days (median)", "tone": "blue",
         "sub": f"Longest stay **{T.n(head['longest_days_on_chart'])}** days; average {T.num(songs['days_on_chart'].mean())} days per song",
         "help": "How many chart days a song has in the selection. The card shows the median over songs."},
        {"label": "Average rank", "value": T.num(head["avg_rank"]), "tone": "aqua",
         "sub": f"Best place reached **#{head['best_rank']}**; **{T.pct(top10_share, 0)}** of song-days are in the top 10",
         "help": "The mean chart place over all song-days in the selection (1 is the top, 50 the bottom)."},
        {"label": "Rank volatility index", "value": T.num(head["median_volatility"]), "tone": "yellow",
         "sub": "Typical swing of a song's daily place (standard deviation), median over songs",
         "help": "Standard deviation of a song's daily chart places. Songs with one day have no value."},
        {"label": "Popularity score trend", "value": T.num(trend["end"]), "unit": "points", "tone": "violet",
         "sub": (f"**{T.signed(trend['change'])}** points against the first week ({T.num(trend['start'])})" if pd.notna(trend["change"])
                 else "Not enough days to show a trend"),
         "help": "The 7-day average popularity of the songs on the chart in the last week of the selection, and its change from the first week."},
        {"label": "Artist dominance index", "value": T.pct(head["adi_leader_value"], 1, already_percent=True), "tone": "pink",
         "sub": f"**{head['adi_leader']}** leads; the top 10 artists hold **{T.pct(head['adi_top10'], 1, already_percent=True)}** of all chart points",
         "help": "An artist's chart points (51 minus the place, for every song-day) as a share of all chart points of the chosen dates."},
        {"label": "Explicit content share", "value": T.pct(head["explicit_share_rows"], 0), "tone": "orange",
         "sub": f"of the song-days in the selection; the whole chart averages **{T.pct(head['explicit_share_chart'], 0)}**",
         "help": "The share of song-days (a song on the chart for one day) that carry the explicit flag."},
    ])
    kpi_cards(columns=5, items=[
        {"label": "Songs", "value": T.n(head["songs"]), "tone": "ink", "sub": f"{T.n(head['song_days'])} song-days in the selection"},
        {"label": "Artists", "value": T.n(head["artists"]), "tone": "ink", "sub": "main artists with a song in the selection"},
        {"label": "Chart days", "value": T.n(head["days"]), "tone": "ink", "sub": T.span(window["date"].min(), window["date"].max())},
        {"label": "Daily turnover", "value": T.pct(head["turnover_pct"], 1, already_percent=True), "tone": "ink",
         "sub": "of the 50 places change hands on an average day (whole chart)"},
        {"label": "New songs a day", "value": T.num(head["new_entries_per_day"], 1), "tone": "ink", "sub": "first-time entries per chart day (whole chart)"},
    ])

    left, right = st.columns([3, 2])
    with left:
        top = artists.head(10).copy()
        top["label"] = top["artist"]
        fig = ch.hbar(top, "label", "adi", "Who holds the chart: the ten artists with the highest ADI", color=theme.BLUE, fmt=".1f", suffix="%",
                      x_title="Artist dominance index (% of all chart points)", customdata=["songs", "chart_slots"],
                      hover="%{y}<br>ADI %{x:.2f}%<br>%{customdata[0]} songs, %{customdata[1]:,} chart slots<extra></extra>", height=420)
        plot(fig, key="ov_artists")
    with right:
        flow = V.chart_flow(summary, 7)
        fig = ch.line_chart(flow, "date", "turnover_smooth", title="How much of the chart changes (7-day average)", y_title="% of places that changed hands",
                            fmt=".1f", suffix="%", fill=True, height=420)
        plot(fig, key="ov_turnover")
    busy = int(summary["is_high_turnover_day"].sum()) if not summary.empty else 0
    callout(f"**{head['adi_leader']}** holds **{T.pct(head['adi_leader_value'], 1, already_percent=True)}** of all chart points; the ten leading artists hold "
            f"**{T.pct(head['adi_top10'], 1, already_percent=True)}**, while {T.n(head['artists'])} artists appear in the selection. On an average day "
            f"**{T.pct(head['turnover_pct'], 1, already_percent=True)}** of the 50 places change hands, and {T.plural(busy, 'day')} were unusually busy.")

    left, right = st.columns(2)
    with left:
        daily = rows.groupby("date")["popularity_clean"].mean().rename("popularity").reset_index()
        daily["seven_day_average"] = daily["popularity"].rolling(7, min_periods=1).mean()
        fig = ch.line_chart(daily, "date", "seven_day_average", title="Average popularity score of the songs in the selection (7-day average)",
                            y_title="Popularity score (0 to 100)", fmt=".1f", height=340)
        plot(fig, key="ov_popularity")
    with right:
        monthly = C.share_by_period(rows, "is_explicit", "month")
        monthly["share_pct"] = monthly["share"] * 100
        fig = ch.vbar(monthly, "period", "share_pct", "Explicit share of the song-days, by month", colors={"share_pct": theme.ORANGE}, y_title="% of song-days",
                      fmt=".0f", suffix="%", height=340)
        fig.update_xaxes(tickformat="%b %Y")
        plot(fig, key="ov_explicit")

    last_day = window["date"].max()
    section(f"The chart on {T.day(last_day)}", "The Top 10 on the last chosen day. Songs outside the current filters are not listed.")
    latest = V.day_chart(rows, last_day).head(10)
    if latest.empty:
        callout("No song of the current selection is on the chart on the last chosen day.", title="Nothing to show", tone="warn")
    else:
        shown = pd.DataFrame({"Place": latest["position"], "Song": latest["song_clean"], "Artist": latest["artist_credit"],
                              "Move": latest.apply(_move_label, axis=1), "Days on chart": latest["chart_day_number"],
                              "Popularity": latest["popularity_clean"]})
        table(shown, column_config={"Popularity": st.column_config.ProgressColumn("Popularity", min_value=0, max_value=100, format="%d"),
                                    "Place": st.column_config.NumberColumn("Place", width=62), "Song": st.column_config.TextColumn("Song", width=360),
                                    "Artist": st.column_config.TextColumn("Artist", width=180), "Move": st.column_config.TextColumn("Move", width=90),
                                    "Days on chart": st.column_config.NumberColumn("Days on chart", width=115)}, height=388)

    findings = report("p4_eda_findings")
    if findings is not None and not findings.empty:
        section("What the analysis found", "Six findings of Phase 4, one from each block. The full list, with the caveat of each, is on the Evidence page.")
        picked = [findings[findings["block"] == block].iloc[0] for block in ("Overview", "Playlist ranking", "Artist", "Content attributes", "Popularity", "Song-level")
                  if (findings["block"] == block).any()]
        cards([{"tag": f"{row['finding_id']}  |  {row['block']}", "title": row["finding"], "text": row["evidence"]} for row in picked])

    pages = st.session_state.get("_pages")
    if pages:
        section("Where to go next")
        from app import nav
        columns = st.columns(len(nav.sections()))
        for column, name in zip(columns, nav.sections()):
            with column:
                st.markdown(f"**{name}**")
                for section_name, slug, title, *_ in nav.PAGES:
                    if section_name == name and slug in pages and slug != "overview":
                        st.page_link(pages[slug], label=title)
