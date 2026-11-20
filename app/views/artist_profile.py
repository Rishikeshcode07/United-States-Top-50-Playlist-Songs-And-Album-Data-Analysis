"""
app/views/artist_profile.py  -  page 10: the profile of one artist.

WHAT THIS FILE IS
    Pick an artist and see how they appear on the chart: ADI (as main artist and counting every credit), songs, chart slots, best
    place, the ADI month by month next to other artists, when each of their songs was on the chart and which songs carry them.

WHY WE NEED IT
    The leaderboard ranks artists; a profile explains one artist's presence (one big hit, or many songs that came and went).

WHAT HAPPENS WHEN IT RUNS
    The songs of the artist are every song on which the artist is credited (a featured credit counts). ADI "as main artist" uses the
    songs that the artist leads; ADI "every credit" also counts features. Both divide by all chart points of the chosen dates. When
    the Phase 6C result file exists, the page also shows the artist's segment (it was found on the whole file, not on the filters).

EXPECTED OUTCOME
    A one-page picture of an artist's presence with the same numbers as the leaderboard.

MOTIVE
    Tell hit-makers from steady presences, and show it with data.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, kpi_cards, page_header, plot, section, stop_if_empty, table
from app.ui.state import current, report
from src.dash import filters as F
from src.dash import metrics as M
from src.dash import texts as T


def render() -> None:
    ctx = current()
    page_header("Artists", "Artist profile", "One artist's presence on the chart: how much of it they hold, which songs carry them and how that changed month by month.", ctx.chips)
    stop_if_empty(ctx.rows)
    credited = ctx.artists("credits")
    stop_if_empty(credited)
    names = credited["artist"].tolist()
    if st.session_state.get("ap_artist") not in names:
        st.session_state["ap_artist"] = names[0]
    artist = st.selectbox("Artist", names, key="ap_artist", help="Type to search. The list follows the sidebar filters and is ordered by ADI.")

    ids = F.credited_track_ids(ctx.data.bridge, [artist])
    mine = ctx.rows[ctx.rows["track_id"].isin(ids)]
    if mine.empty:
        callout("This artist has no song in the current selection.", title="Nothing to show", tone="warn")
        st.stop()
    lead_rows = mine[mine["primary_artist"] == artist]
    total = ctx.total_points
    songs = ctx.songs()
    songs = songs[songs["track_id"].isin(ids)].assign(role=lambda d: (d["primary_artist"] == artist).map({True: "main artist", False: "featured"}))
    adi_main = lead_rows["chart_points"].sum() / total * 100 if total else float("nan")
    adi_all = mine["chart_points"].sum() / total * 100 if total else float("nan")
    rank_all = int(credited.loc[credited["artist"] == artist, "adi_rank"].iloc[0])

    kpi_cards(columns=6, items=[
        {"label": "ADI as main artist", "value": T.pct(adi_main, 2, already_percent=True), "tone": "pink", "sub": f"{T.n(lead_rows['track_id'].nunique())} songs that the artist leads"},
        {"label": "ADI with every credit", "value": T.pct(adi_all, 2, already_percent=True), "tone": "blue", "sub": f"number {rank_all} of {T.n(len(credited))} credited artists"},
        {"label": "Songs", "value": T.n(songs["track_id"].nunique()), "tone": "aqua", "sub": f"{int((songs['role'] == 'featured').sum())} as a featured artist"},
        {"label": "Chart slots", "value": T.n(len(mine)), "tone": "violet", "sub": f"{T.pct(len(mine) / (50 * ctx.n_days), 1)} of all {T.n(50 * ctx.n_days)} places of these dates"},
        {"label": "Best place", "value": f"#{int(mine['position'].min())}", "tone": "green", "sub": f"{T.n(int(mine['is_number_one'].sum()))} song-days at number one"},
        {"label": "Days present", "value": T.n(mine["date"].nunique()), "unit": f"of {T.n(ctx.n_days)}", "tone": "yellow", "sub": f"average place {T.num(mine['position'].mean())}"},
    ])

    # ---- ADI over time -----------------------------------------------------------------------------------------
    default_others = [n for n in names[:4] if n != artist][:3]
    others = st.multiselect("Compare with up to four other artists", [n for n in names if n != artist], default=default_others, max_selections=4, key="ap_compare")
    a, b = st.columns(2)
    with a:
        main_line = M.artist_period_adi(ctx.rows, ctx.window, "month", "primary", ctx.data.bridge, artists=[artist]).assign(basis="Main artist")
        all_line = M.artist_period_adi(ctx.rows, ctx.window, "month", "credits", ctx.data.bridge, artists=[artist]).assign(basis="Every credit")
        both = pd.concat([main_line, all_line], ignore_index=True)
        fig = ch.line_chart(both, "period", "adi", group="basis", title=f"ADI of {artist}, by month", order=["Every credit", "Main artist"], colors={"Every credit": theme.BLUE, "Main artist": theme.PINK},
                            y_title="ADI in the month (%)", fmt=".2f", suffix="%", markers=True, height=380, dash={"Every credit": "dot"})
        fig.update_xaxes(tickformat="%b %Y")
        plot(fig, key="ap_adi")
    with b:
        chosen = [artist] + list(others)
        compare = M.artist_period_adi(ctx.rows, ctx.window, "month", "credits", ctx.data.bridge, artists=chosen)
        fig = ch.line_chart(compare, "period", "adi", group="artist", title="ADI by month, every credit counted", order=chosen, y_title="ADI in the month (%)", fmt=".2f", suffix="%", height=380)
        fig.update_xaxes(tickformat="%b %Y")
        plot(fig, key="ap_compare_chart")

    # ---- songs ------------------------------------------------------------------------------------------------
    section("The songs", "When each song was on the chart (first to last day inside the chosen dates) and how much of the artist's chart points it earned.")
    how_many = st.slider("Songs to draw", 5, 40, 15, key="ap_songs")
    top_songs = songs.sort_values(["chart_points", "track_label"], ascending=[False, True]).head(how_many)
    left, right = st.columns([3, 2])
    with left:
        fig = ch.span_chart(top_songs.assign(label=top_songs["track_label"].map(lambda t: ch.shorten(t, 44)), release=top_songs["album_type"].str.title()), "label", "first_date", "last_date",
                            color_col="release", colors=theme.ALBUM_COLORS, title="Stays of the artist's biggest songs", hover_cols=["days_on_chart", "best_rank"],
                            hover_names={"days_on_chart": "Days on the chart", "best_rank": "Best place"}, height=max(300, 28 * len(top_songs) + 120))
        plot(fig, key="ap_span")
    with right:
        share = top_songs.assign(label=top_songs["track_label"].map(lambda t: ch.shorten(t, 40)), share=top_songs["chart_points"] / mine["chart_points"].sum() * 100)
        fig = ch.hbar(share, "label", "share", "Share of the artist's chart points", color=theme.BLUE, fmt=".1f", suffix="%", height=max(300, 28 * len(top_songs) + 120))
        plot(fig, key="ap_share")
    top_share = float(songs.sort_values("chart_points", ascending=False)["chart_points"].head(1).sum() / mine["chart_points"].sum()) if len(songs) else float("nan")
    best_song = songs.sort_values("chart_points", ascending=False).iloc[0]
    callout(f"**{artist}** has {T.plural(len(songs), 'song')} in the selection. The biggest, **{best_song['song']}**, earns **{T.pct(top_share, 0)}** of the artist's chart points, "
            f"and the artist holds **{T.pct(adi_all, 2, already_percent=True)}** of all chart points when every credit counts (**{T.pct(adi_main, 2, already_percent=True)}** as main artist).")

    shown = pd.DataFrame({"Song": songs["song"], "Role": songs["role"], "Days": songs["days_on_chart"], "Best place": songs["best_rank"], "Average place": songs["avg_rank"].round(1),
                          "Chart points": songs["chart_points"], "Share of the artist's points": (songs["chart_points"] / mine["chart_points"].sum() * 100),
                          "Release": songs["album_type"], "Explicit": songs["is_explicit"].map({True: "yes", False: "no"}), "First day": songs["first_date"].dt.date,
                          "Last day": songs["last_date"].dt.date}).sort_values("Chart points", ascending=False)
    table(shown, column_config={"Song": st.column_config.TextColumn("Song", width="large"), "Role": st.column_config.TextColumn("Role", width="small"),
                                "Days": st.column_config.NumberColumn("Days", width="small"), "Best place": st.column_config.NumberColumn("Best place", width="small"),
                                "Average place": st.column_config.NumberColumn("Avg place", width="small"), "Chart points": st.column_config.NumberColumn("Points", width="small"),
                                "Share of the artist's points": st.column_config.ProgressColumn("Share of the artist's points (%)", min_value=0, max_value=100, format="%.1f", width="medium"),
                                "Release": st.column_config.TextColumn("Release", width="small"), "Explicit": st.column_config.TextColumn("Explicit", width="small"),
                                "First day": st.column_config.DateColumn(format="D MMM YYYY", width="small"), "Last day": st.column_config.DateColumn(format="D MMM YYYY", width="small")},
          height=min(460, 38 * min(len(shown), 11) + 44))

    # ---- profile of the catalogue, and the Phase 6C segment -----------------------------------------------------
    section("What the catalogue looks like", "The artist's songs against all songs of the selection.")
    everyone = ctx.songs()
    mix = pd.DataFrame({"Measure": ["Explicit songs", "Collaborations", "Singles", "Average song length (minutes)", "Median days on the chart"],
                        artist: [T.pct(songs["is_explicit"].mean(), 0), T.pct(songs["is_collab"].mean(), 0), T.pct((songs["album_type"] == "single").mean(), 0),
                                 T.num(songs["duration_min"].mean(), 1), T.num(songs["days_on_chart"].median(), 0)],
                        "All songs of the selection": [T.pct(everyone["is_explicit"].mean(), 0), T.pct(everyone["is_collab"].mean(), 0), T.pct((everyone["album_type"] == "single").mean(), 0),
                                                        T.num(everyone["duration_min"].mean(), 1), T.num(everyone["days_on_chart"].median(), 0)]})
    table(mix, height=38 * len(mix) + 40)
    segments = report("p6c_artists")
    if segments is not None and artist in set(segments["artist"]):
        row = segments[segments["artist"] == artist].iloc[0]
        callout(f"The segmentation of Phase 6C (made on the whole file, not on the current filters) puts **{artist}** in the segment **{row['segment']}** "
                f"({'a firm assignment' if bool(row['certain']) else 'a borderline assignment'}, agreement {T.num(row['agreement'], 2)} across repeated clusterings). "
                "Segments describe how artists behaved on the chart; they are not a rating.", title="Artist segment (Phase 6C)")
