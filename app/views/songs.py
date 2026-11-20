"""
app/views/songs.py  -  page 7: the song leaderboard.

WHAT THIS FILE IS
    A sortable, searchable table with one row per song (days on the chart, best and average place, rank volatility, chart points,
    popularity, release details), plus two charts that show how the songs are spread: a bubble chart of stay against average place
    and a comparison of how many songs and how many chart points each stay class holds.

WHY WE NEED IT
    The brief asks for song rankings and the KPIs Days on Chart, Average Rank and Rank Volatility Index for every song. This is the
    page where a person finds a song, sorts by any of them and opens its profile.

WHAT HAPPENS WHEN IT RUNS
    Every number is calculated from the rows that match the sidebar (so "days on chart" counts the days inside the chosen dates).
    The extra controls on this page (sort, stay class, minimum days, search) narrow the table further and do not change the sidebar.
    Clicking a row opens a button that jumps to the Song profile page with that song chosen.

EXPECTED OUTCOME
    One table that answers "which songs ...?" for any combination of filters, and a download of the table.

MOTIVE
    Make every song findable and comparable on the same measures.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, download_csv, kpi_cards, page_header, plot, section, stop_if_empty, table
from app.ui.state import current
from src.config import LONGEVITY_LABELS
from src.dash import compare as C
from src.dash import metrics as M
from src.dash import texts as T

SORTS = {
    "Chart points": ("chart_points", False), "Days on the chart": ("days_on_chart", False), "Best place": ("best_rank", True),
    "Average place": ("avg_rank", True), "Rank volatility (highest first)": ("rank_volatility_index", False),
    "Average popularity": ("avg_popularity", False), "Days at number one": ("days_at_number_one", False),
    "Days in the top 10": ("days_in_top10", False), "First day on the chart (newest first)": ("first_date", False),
}
COLOUR_BY = {"Release type": ("album_type", theme.ALBUM_COLORS), "Explicit content": ("explicit_label", theme.EXPLICIT_COLORS),
             "Collaboration": ("collab_label", theme.COLLAB_COLORS), "Stay class": ("longevity_class", theme.LONGEVITY_COLORS)}


def render() -> None:
    ctx = current()
    page_header("Songs", "Song leaderboard",
                "Every song of the selection on the same measures. Sort, search, and click a row to open the song's profile.", ctx.chips)
    stop_if_empty(ctx.rows)
    songs = ctx.songs().copy()
    songs["explicit_label"] = songs["is_explicit"].map({True: "Explicit", False: "Not explicit"})
    songs["collab_label"] = songs["is_collab"].map({True: "Collaboration", False: "Solo"})

    c1, c2, c3, c4 = st.columns([2, 2.2, 1.6, 2], vertical_alignment="bottom")
    sort_name = c1.selectbox("Sort by", list(SORTS), key="sg_sort")
    classes = c2.multiselect("Stay class", LONGEVITY_LABELS, default=[], key="sg_classes", placeholder="All classes",
                             help="Flash, Short, Medium, Long and Evergreen are the stay classes of the song table (by days on the chart).")
    longest = int(songs["days_on_chart"].max())
    min_days = c3.slider("At least this many days", 1, max(longest, 2), 1, key="sg_min")
    query = c4.text_input("Find a song or artist", key="sg_find", placeholder="for example Sabrina").strip()

    column, low_is_better = SORTS[sort_name]
    view = songs[songs["days_on_chart"] >= min_days]
    if classes:
        view = view[view["longevity_class"].isin(classes)]
    if query:
        needle = query.casefold()
        view = view[view["track_label"].str.casefold().str.contains(needle, regex=False) | view["artist_credit"].str.casefold().str.contains(needle, regex=False)]
    view = view.sort_values([column, "chart_points", "track_label"], ascending=[low_is_better, False, True], na_position="last").reset_index(drop=True)
    if view.empty:
        callout("No song matches these choices. Lower the minimum days, clear the stay class or the search.", title="No songs", tone="warn")
        st.stop()

    kpi_cards(columns=5, items=[
        {"label": "Songs listed", "value": T.n(len(view)), "tone": "blue", "sub": f"of {T.n(len(songs))} in the selection"},
        {"label": "Median stay", "value": T.num(M.median_or_nan(view["days_on_chart"]), 0), "unit": "days", "tone": "aqua", "sub": f"longest {T.n(view['days_on_chart'].max())} days"},
        {"label": "Reached the top 10", "value": T.pct((view["best_rank"] <= 10).mean(), 0), "tone": "violet", "sub": "of the listed songs"},
        {"label": "Held number one", "value": T.n(int((view["days_at_number_one"] > 0).sum())), "tone": "orange", "sub": "songs with at least one day at the top"},
        {"label": "Median volatility", "value": T.num(M.median_or_nan(view["rank_volatility_index"])), "tone": "yellow", "sub": "typical swing of the daily place"},
    ])

    top = view.copy()
    shown = pd.DataFrame({
        "#": range(1, len(top) + 1), "Song": top["song"], "Artist": top["artist_credit"], "Days": top["days_on_chart"], "Best place": top["best_rank"],
        "Average place": top["avg_rank"].round(1), "Volatility": top["rank_volatility_index"].round(2), "Chart points": top["chart_points"], "Days at #1": top["days_at_number_one"],
        "Popularity": top["avg_popularity"].round(1), "Stay class": top["longevity_class"], "Release": top["album_type"], "Explicit": top["is_explicit"].map({True: "yes", False: "no"}),
        "First day": top["first_date"].dt.date, "Last day": top["last_date"].dt.date})
    config = {"Song": st.column_config.TextColumn("Song", width=300), "Artist": st.column_config.TextColumn("Artist", width=170),
              "Days": st.column_config.ProgressColumn("Days on chart", min_value=0, max_value=int(longest), format="%d", width=140),
              "Popularity": st.column_config.ProgressColumn("Avg popularity", min_value=0, max_value=100, format="%.0f"),
              "Chart points": st.column_config.NumberColumn(format="%d"), "#": st.column_config.NumberColumn("#", width=48),
              "First day": st.column_config.DateColumn("First day", format="D MMM YYYY"), "Last day": st.column_config.DateColumn("Last day", format="D MMM YYYY")}
    event = table(shown, column_config=config, height=min(640, 36 * min(len(shown), 16) + 44), key="sg_table", select=True)
    picked = list(event.selection.rows) if event is not None and getattr(event, "selection", None) else []
    left, right = st.columns([3, 2], vertical_alignment="center")
    with left:
        download_csv(shown, "song_leaderboard.csv", key="sg_csv")
    with right:
        if picked:
            song = view.iloc[picked[0]]
            if st.button(f"Open the profile of {ch.shorten(song['song'], 30)}", icon=":material/open_in_new:", key="sg_open"):
                st.session_state["sp_song"] = int(song["track_id"])
                pages = st.session_state.get("_pages")
                if pages and "song-profile" in pages:
                    st.switch_page(pages["song-profile"])
        else:
            st.caption("Click a row to open that song's profile.")

    section("How the songs are spread")
    colour_name = st.segmented_control("Colour the bubbles by", list(COLOUR_BY), default="Release type", key="sg_colour") or "Release type"
    colour_col, palette = COLOUR_BY[colour_name]
    a, b = st.columns([3, 2])
    with a:
        fig = ch.scatter_chart(view, "days_on_chart", "avg_rank", color=colour_col, size="chart_points", title="Stay and average place (bubble size = chart points)", colors=palette,
                               x_title="Days on the chart (log scale)", y_title="Average place (1 = top)", hover_cols=["track_label", "best_rank"],
                               hover_names={"track_label": "Song", "best_rank": "Best place"}, reverse_y=True, size_max=30, height=470)
        ch.log_axis(fig, "x")
        plot(fig, key="sg_bubble")
    with b:
        profile = C.group_profile(view, "longevity_class", order=LONGEVITY_LABELS)
        long_form = pd.concat([profile.assign(measure="Share of songs", value=profile["share_of_songs"] * 100), profile.assign(measure="Share of chart points", value=profile["share_of_points"] * 100)])
        wide = long_form.pivot(index="group", columns="measure", values="value").reindex([g for g in LONGEVITY_LABELS if g in set(long_form["group"])]).reset_index()
        wide["group"] = wide["group"].str.split(" \\(").str[0]
        fig = ch.vbar(wide, "group", ["Share of songs", "Share of chart points"], "Songs and chart points by stay class", colors={"Share of songs": theme.AXIS, "Share of chart points": theme.BLUE},
                      y_title="% of the listed songs / points", fmt=".0f", suffix="%", show_text=True, height=470)
        plot(fig, key="sg_classes_chart")
    st.caption("Stay classes by days on the chart: Flash 1 to 3 days, Short 4 to 14, Medium 15 to 60, Long 61 to 180, Evergreen 181 or more.")
    if len(profile) >= 2:
        heavy = profile.sort_values("share_of_points", ascending=False).iloc[0]
        callout(f"**{heavy['group']}** songs are **{T.pct(heavy['share_of_songs'], 0)}** of the listed songs but earn **{T.pct(heavy['share_of_points'], 0)}** of their chart points. "
                "A song's stay class comes from the days it spent on the chart inside the chosen dates.")
