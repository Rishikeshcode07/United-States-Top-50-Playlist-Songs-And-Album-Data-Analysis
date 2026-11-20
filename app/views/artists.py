"""
app/views/artists.py  -  page 9: the artist dominance leaderboard.

WHAT THIS FILE IS
    Who holds the chart. The Artist Dominance Index (ADI) of an artist is the artist's chart points divided by ALL chart points of the
    chosen dates, times 100, where a song-day earns 51 minus its place (50 points for place 1, 1 point for place 50, 1,275 points for
    a whole day). The page shows the ranking, the ADI over time for the leaders, and a table of every artist.

WHY WE NEED IT
    "Artist dominance leaderboard" is a core module of the brief, and ADI is one of its six KPIs.

WHAT HAPPENS WHEN IT RUNS
    Two ways to credit points. "Main artist": a song's points go to its first credited artist only, so the shares add up to 100 %.
    "Every credited artist": a featured artist earns the song's points as well, so the shares add up to more than 100 %. The
    denominator is always the whole chart of the chosen dates, so an artist filter shows how much of the chart that artist holds.
    A click on a bar opens a button that leads to the artist's profile page.

EXPECTED OUTCOME
    A leaderboard, a time chart for the leaders, and the exact numbers for every artist, with the formula next to them.

MOTIVE
    Make dominance measurable and comparable, and say plainly how it is measured.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, download_csv, kpi_cards, page_header, plot, section, stop_if_empty, table
from app.ui.state import current
from src.dash import metrics as M
from src.dash import texts as T
from src.eda import concentration as conc

BASIS = {"Main artist": "primary", "Every credited artist": "credits"}


def render() -> None:
    ctx = current()
    page_header("Artists", "Artist dominance",
                "Which artists hold the chart, by the Artist Dominance Index: the artist's chart points as a share of all chart points of the chosen dates.", ctx.chips)
    stop_if_empty(ctx.rows)

    c1, c2, c3, c4 = st.columns([2.4, 2, 1.6, 2], vertical_alignment="bottom")
    basis_name = c1.segmented_control("Credit the points to", list(BASIS), default="Main artist", key="ar_basis") or "Main artist"
    basis = BASIS[basis_name]
    top_n = c2.slider("Artists in the ranking", 5, 30, 15, key="ar_topn")
    period = (c3.segmented_control("ADI over time, by", ["Week", "Month"], default="Month", key="ar_period") or "Month").lower()
    follow = c4.slider("Artists to follow over time", 2, 8, 5, key="ar_follow")
    artists = ctx.artists(basis)
    stop_if_empty(artists)
    shares_total = float(artists["adi"].sum())
    leader = artists.iloc[0]
    values = artists["chart_points"].to_numpy(dtype=float)

    kpi_cards(columns=6, items=[
        {"label": "Leader", "value": str(leader["artist"]), "tone": "pink", "sub": f"ADI {T.pct(leader['adi'], 2, already_percent=True)}; {T.n(int(leader['songs']))} songs"},
        {"label": "Top 5 together", "value": T.pct(artists["adi"].head(5).sum(), 1, already_percent=True), "tone": "blue", "sub": "of all chart points"},
        {"label": "Top 10 together", "value": T.pct(artists["adi"].head(10).sum(), 1, already_percent=True), "tone": "aqua", "sub": "of all chart points"},
        {"label": "Artists on the chart", "value": T.n(len(artists)), "tone": "violet", "sub": f"{T.pct(artists['adi'].head(10).sum() / shares_total, 0) if shares_total else T.DASH} of the points go to the top 10"},
        {"label": "Effective number", "value": T.num(conc.effective_number(values), 0), "tone": "yellow", "sub": "equal artists that would give the same concentration (HHI)"},
        {"label": "Gini", "value": T.num(conc.gini(values), 2), "tone": "orange", "sub": "0 = equal shares, 1 = one artist holds everything"},
    ])
    if basis == "credits":
        st.caption(f"With every credited artist, the shares add up to {T.pct(shares_total, 1, already_percent=True)}, because a song with a featured artist earns its points for each of them.")

    a, b = st.columns([1, 1])
    with a:
        top = artists.head(top_n).copy()
        fig = ch.hbar(top, "artist", "adi", f"The {len(top)} artists with the highest ADI", color=theme.PINK, fmt=".2f", suffix="%", x_title="Artist dominance index (% of all chart points)",
                      customdata=["songs", "chart_slots", "best_rank"], hover="%{y}<br>ADI %{x:.2f}%<br>%{customdata[0]} songs, %{customdata[1]:,} chart slots<br>best place %{customdata[2]}<extra></extra>",
                      height=max(360, 27 * len(top) + 100))
        event = plot(fig, key="ar_bars", select=True)
        clicked = [p.get("y") for p in (event.selection.points if event is not None and getattr(event, "selection", None) else []) if p.get("y")]
        if clicked:
            if st.button(f"Open the profile of {ch.shorten(clicked[0], 30)}", icon=":material/open_in_new:", key="ar_open"):
                st.session_state["ap_artist"] = clicked[0]
                pages = st.session_state.get("_pages")
                if pages and "artist-profile" in pages:
                    st.switch_page(pages["artist-profile"])
        else:
            st.caption("Click a bar to open that artist's profile.")
    with b:
        names = artists["artist"].head(follow).tolist()
        over_time = M.artist_period_adi(ctx.rows, ctx.window, period, basis, ctx.data.bridge, artists=names)
        fig = ch.line_chart(over_time, "period", "adi", group="artist", title=f"ADI of the leading artists, by {period}", order=names, y_title="ADI in the period (%)", fmt=".1f", suffix="%",
                            markers=period == "month", height=max(360, 27 * top_n + 100))
        fig.update_xaxes(tickformat="%b %Y")
        plot(fig, key="ar_time")

    callout(f"**{leader['artist']}** leads with an ADI of **{T.pct(leader['adi'], 2, already_percent=True)}**: out of every 100 chart points of these dates, "
            f"{T.num(leader['adi'], 1)} went to this artist. The top 10 hold **{T.pct(artists['adi'].head(10).sum(), 1, already_percent=True)}** while {T.n(len(artists))} artists appear, "
            f"so the chart behaves as if about **{T.num(conc.effective_number(values), 0)}** equal artists shared it. This is a measure of the past chart, not a ranking of talent.")

    section("Many songs or a few hits?", "Every bubble is an artist. Bubbles that are far to the right and near the top have many songs that sit high on the chart.")
    pool = artists.head(max(top_n, 60))
    fig = ch.scatter_chart(pool, "songs", "avg_rank", size="chart_points", title="Songs and average place of the leading artists (bubble size = chart points)", x_title="Songs on the chart",
                           y_title="Average place (1 = top)", hover_cols=["artist", "adi", "best_rank"], hover_names={"artist": "Artist", "adi": "ADI %", "best_rank": "Best place"},
                           reverse_y=True, size_max=34, height=470, opacity=0.6)
    plot(fig, key="ar_bubble")

    section("Every artist")
    query = st.text_input("Find an artist", key="ar_find", placeholder="for example Drake").strip()
    view = artists if not query else artists[artists["artist"].str.casefold().str.contains(query.casefold(), regex=False)]
    shown = pd.DataFrame({"#": view["adi_rank"], "Artist": view["artist"], "ADI": view["adi"], "Songs": view["songs"], "Chart slots": view["chart_slots"], "Days present": view["days_present"],
                          "Share of slots": view["share_of_slots_pct"].round(2), "Average place": view["avg_rank"].round(1), "Best place": view["best_rank"], "Days at #1": view["days_at_number_one"],
                          "First day": view["first_date"].dt.date, "Last day": view["last_date"].dt.date})
    table(shown, column_config={"ADI": st.column_config.ProgressColumn("ADI (%)", min_value=0, max_value=float(max(artists["adi"].max(), 0.01)), format="%.2f"),
                                "#": st.column_config.NumberColumn("#", width="small"), "Share of slots": st.column_config.NumberColumn("Share of slots (%)", format="%.2f"),
                                "First day": st.column_config.DateColumn(format="D MMM YYYY"), "Last day": st.column_config.DateColumn(format="D MMM YYYY")},
          height=min(560, 36 * min(len(shown), 14) + 44))
    download_csv(shown, "artist_dominance.csv", key="ar_csv")
    with st.expander("How the ADI is calculated"):
        st.markdown(
            "- A song on the chart on one day earns **51 minus its place** in chart points: 50 points for place 1, 1 point for place 50. A whole day holds **1,275** points.\n"
            f"- The artist's chart points are added up over the chosen dates (**{T.n(M.total_chart_points(ctx.window))}** points on the whole chart of these dates).\n"
            "- **ADI = the artist's chart points / all chart points of the chosen dates x 100.** The denominator is always the whole chart, so filters change what is counted for the artist, not the size of the chart.\n"
            "- *Main artist* gives a song's points to its first credited artist. *Every credited artist* gives them to each credited artist, so the shares add up to more than 100.")
