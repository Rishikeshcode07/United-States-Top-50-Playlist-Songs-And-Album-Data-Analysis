"""
app/views/archetypes.py  -  page 16: song archetypes and artist segments (Phase 6C, with live filters).

WHAT THIS FILE IS
    The page about the SHAPE of a song's first month on the chart. Phase 6C gave every new song whose first 30 days are fully known a
    row of 30 chart scores (51 minus the place, 0 when off the chart) and let k-means find five archetypes: brief visitors, quick
    fades, month-long fades, slow climbers and lasting hits. The same idea applied to the artists gives four segments. This page
    shows the archetypes (cards, live trajectories for the songs you filter, one archetype opened up song by song, which kinds of
    song become which archetype), the artist segments (a map of the artists), and how firm the groups are.

WHY WE NEED IT
    "Days on chart" hides the way a song lives: two songs with 20 days can look nothing alike (one starts at 5 and fades, another
    climbs from 40 to 10). The archetypes give a playlist manager a small vocabulary for those shapes. A clustering method always
    returns groups, so the third tab is about how far to believe them.

WHAT HAPPENS WHEN IT RUNS
    The archetype of every song and the segment of every artist come from the notebook (reports/p6c_*.csv); nothing is re-clustered
    here, so the groups never change when a filter changes. The filters decide WHICH songs are averaged: the dates select songs by
    arrival date, the place range by debut place, and the other filters as everywhere else. The artist segments describe the whole
    file and ignore the filters (the artist filter only highlights).

EXPECTED OUTCOME
    About half of the songs are brief visitors; about one in six (slow climbers and lasting hits) is still on the chart on day 30;
    a few catalogue artists hold 59% of all chart days. Every group is firm under re-drawing the artists, the month-long fades and
    the slow climbers least so.

MOTIVE
    Describe how songs and artists live on the chart, and say how much each description can be trusted. These are descriptions of
    what happened, not forecasts and not boxes to put a new song into.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, cards, download_csv, kpi_cards, page_header, plot, section, table
from app.ui.components import group_table as side_table
from app.ui.state import current, require_reports
from app.ui.state import facts as saved_facts
from src.dash import survival_live as SL
from src.dash import texts as T

ARCHETYPES = ["Brief visitors", "Quick fades", "Month-long fades", "Slow climbers", "Lasting hits"]
SEGMENTS = ["Passing visitors", "Mid-chart stayers", "Top-chart hit-makers", "Catalogue artists"]
DAYS = list(range(1, 31))
DAY_COLUMNS = [f"day_{d:02d}" for d in DAYS]
MEASURES = {
    "Average chart score (51 minus the place, 0 = off the chart)": "score",
    "Share of the songs on the chart": "share",
    "Average place while on the chart": "place",
}
FIRMNESS_ORDER = ["firm", "likely", "border"]
FIRMNESS_TEXT = {"firm": "Firm (same archetype in at least 90% of the re-draws)", "likely": "Likely (60% to 90%)", "border": "On a border (under 60%)"}
FIRMNESS_COLORS = {FIRMNESS_TEXT["firm"]: theme.BLUE, FIRMNESS_TEXT["likely"]: "#86b6ef", FIRMNESS_TEXT["border"]: theme.ORANGE}
RISK_ORDER = ["lowest risk", "middle", "highest risk"]


# ============================================================================================================ helpers
def _trajectories(first: pd.DataFrame, ids, measure: str) -> pd.DataFrame:
    """One row per archetype and chart day (1 to 30): the average score, the share of songs on the chart, or the average place."""
    sub = first[first["track_id"].isin(ids)]
    if sub.empty:
        return pd.DataFrame(columns=["archetype", "day", "value"])
    long = sub.melt(id_vars=["track_id", "archetype"], value_vars=DAY_COLUMNS, var_name="day", value_name="score")
    long["day"] = long["day"].str[4:].astype(int)
    on = long["score"] > 0
    keys = [long["archetype"], long["day"]]
    if measure == "score":
        value = long["score"].groupby(keys).mean()
    elif measure == "share":
        value = on.groupby(keys).mean()
    else:
        value = (51 - long["score"]).where(on).groupby(keys).mean()
    return value.rename("value").reset_index()


def _archetype_card(row: pd.Series) -> dict:
    examples = str(row["examples"]).replace("; ", "  |  ")
    return {"tag": f"{T.pct(row['share_of_songs'], 0)} of the songs", "title": row["archetype"],
            "text": (f"{T.n(row['songs'])} songs by {T.n(row['artists'])} artists. They arrive around place {T.num(row['debut_place'], 0)}, reach place {T.num(row['best_place'], 0)} at best "
                     f"and spend {T.num(row['chart_days'], 1)} of their first 30 days on the chart; {T.pct(row['on_chart_day_30'], 0)} are still there on day 30. "
                     f"The most songs come from {row['biggest_artist']} ({T.n(row['biggest_artist_songs'])}). Closest to the centre: {examples}.")}


def _inside_figure(first: pd.DataFrame, chosen: pd.DataFrame, archetype: str, limit: int = 70):
    ids = chosen.loc[chosen["archetype"] == archetype, "track_id"]
    sub = first[first["track_id"].isin(ids)]
    if sub.empty:
        return None, 0
    labels = chosen.drop_duplicates("track_id").set_index("track_id")["track_label"]
    shown = sub.sample(min(limit, len(sub)), random_state=0)
    color = theme.ARCHETYPE_COLORS[archetype]
    fig = go.Figure()
    for _, row in shown.iterrows():
        title = ch.shorten(labels.get(row["track_id"], ""), 44)
        fig.add_scatter(x=DAYS, y=row[DAY_COLUMNS].to_numpy(dtype=float), mode="lines", line=dict(color=ch.with_alpha(color, 0.28), width=1.2), showlegend=False,
                        text=[title] * len(DAYS), hovertemplate="%{text}<br>chart day %{x}: score %{y:.0f}<extra></extra>")
    fig.add_scatter(x=DAYS, y=sub[DAY_COLUMNS].mean().to_numpy(), mode="lines", name=f"Average of all {len(sub)} songs", line=dict(color=color, width=3.6),
                    hovertemplate="average, chart day %{x}: score %{y:.1f}<extra></extra>")
    fig.update_yaxes(range=[-1, 51], title_text="Chart score (51 minus the place; 0 = off the chart)")
    fig.update_xaxes(range=[1, 30], title_text="Chart day since the song arrived")
    return ch.finish(fig, f"Inside the group: {archetype}", 430, "top"), len(shown)


# ================================================================================================= tab 1: the songs
def _songs_tab(ctx, tables: dict) -> None:
    group_table, per_song, first = tables["p6c_archetypes"], tables["p6c_song_archetypes"], tables["p6c_first_months"]
    stays = SL.prepare_stays(tables["p6b_stays"], ctx.data.dim_track)
    songs = per_song.assign(entry_date=pd.to_datetime(per_song["entry_date"])).merge(stays[["track_id", "album_type", "is_explicit", "is_collab"]], on="track_id", how="left")
    chosen = SL.filter_stays(songs, ctx.state, ctx.data.bridge)
    st.caption(f"{T.n(len(per_song))} new songs have a complete first month (the {T.n(len(stays) - len(per_song))} that arrived in the last 29 days of the file are left out, because their month is not finished). "
               "The groups come from the notebook and do not change; the sidebar chooses which songs are averaged: dates by arrival date, places by debut place.")
    if chosen.empty:
        callout("No song with a complete first month matches the filters. Widen the dates or the place range, or press Reset filters.", title="No songs in this selection", tone="warn")
        return

    picked = st.pills("Archetypes to show", ARCHETYPES, selection_mode="multi", default=ARCHETYPES, key="ar_pick", help="None chosen means all five.") or ARCHETYPES
    measure_name = st.columns([2, 3])[0].selectbox("The trajectories show", list(MEASURES), key="ar_measure")
    chosen = chosen[chosen["archetype"].isin(picked)]
    if chosen.empty:
        callout("None of the chosen archetypes has a song in this selection.", title="No songs", tone="warn")
        return
    sizes = chosen["archetype"].value_counts()
    kpi_cards(columns=5, items=[
        {"label": "Songs in the selection", "value": T.n(len(chosen)), "tone": "blue", "sub": f"of {T.n(len(per_song))} with a complete first month"},
        {"label": "Most common archetype", "value": str(sizes.index[0]), "tone": "aqua", "sub": f"{T.pct(sizes.iloc[0] / len(chosen), 0)} of these songs"},
        {"label": "On the chart on day 30", "value": T.pct((chosen["score_day_30"] > 0).mean(), 0), "tone": "green", "sub": "of these songs"},
        {"label": "Chart days in the first 30", "value": T.num(chosen["days_on_chart_first_month"].mean(), 1), "tone": "orange", "sub": "on average"},
        {"label": "Firmly placed", "value": T.pct((chosen["firmness"] == "firm").mean(), 0), "tone": "violet", "sub": "in the same archetype in at least 9 of 10 re-draws",
         "help": "Phase 6C re-drew whole artists 1,000 times and re-fitted the groups. A song is firmly placed when it lands in its own archetype in at least 90% of the draws."},
    ])

    section("The five archetypes", f"Found by k-means on the 30 daily chart scores of {T.n(len(per_song))} songs. The names were given by a written rule from the numbers, and describe the first month only.")
    ordered = group_table.set_index("archetype").reindex(ARCHETYPES).reset_index()
    cards([_archetype_card(row) for _, row in ordered.iterrows()])

    # ---- live trajectories ---------------------------------------------------------------------------------------------
    section("What the first 30 days look like", "Every line averages the songs of one archetype that match the filters. The groups themselves were fixed by the notebook.")
    measure = MEASURES[measure_name]
    lines = _trajectories(first, chosen["track_id"], measure)
    fmt = ".0%" if measure == "share" else ".1f"
    left, right = st.columns([3.1, 1.2])
    with left:
        fig = ch.line_chart(lines, "day", "value", group="archetype", order=[a for a in ARCHETYPES if a in picked], colors=theme.ARCHETYPE_COLORS, height=440, width=2.8,
                            x_title="Chart day since the song arrived", y_title=measure_name.split(" (")[0],
                            hover="%{fullData.name}<br>chart day %{x}: %{y:" + fmt + "}<extra></extra>")
        if measure == "place":
            ch.rank_axis(fig, 1, 50)
        elif measure == "share":
            fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
        fig.update_xaxes(range=[1, 30])
        plot(fig, key="ar_lines")
    with right:
        day30 = lines[lines["day"] == 30].set_index("archetype")["value"]
        if not day30.empty and day30.notna().any():
            best, worst = (day30.idxmax(), day30.idxmin()) if measure != "place" else (day30.idxmin(), day30.idxmax())
            word = {"score": "average chart score", "share": "share of songs on the chart", "place": "average place"}[measure]
            shown = (lambda v: T.pct(v, 0)) if measure == "share" else (lambda v: T.num(v, 1))
            callout(f"On day 30 the **{word}** is best for **{best}** ({shown(day30[best])}) and weakest for **{worst}** ({shown(day30[worst])}). "
                    "The archetypes separate mostly by what the song does after the first week: whether it fades, holds on or climbs.", title="Reading")
        else:
            callout("By day 30 none of the selected songs is still on the chart, so the last part of the lines is empty.", title="Reading", tone="warn")
        st.caption("A line is empty on days when none of its songs is on the chart (for the average place).")

    section("Inside one archetype", "The faint lines are single songs; the thick line is the average of the whole group. Hover over a faint line to see the song.")
    options = [a for a in ARCHETYPES if a in set(chosen["archetype"])]
    inside = st.columns([2, 3])[0].selectbox("Look inside", options, index=len(options) - 1 if "Slow climbers" not in options else options.index("Slow climbers"), key="ar_inside")
    figure, drawn = _inside_figure(first, chosen, inside)
    if figure is not None:
        plot(figure, key="ar_inside_chart")
        st.caption(f"Showing {T.n(drawn)} of {T.n(int((chosen['archetype'] == inside).sum()))} songs of this archetype in the selection (a fixed random sample when there are more than 70).")

    # ---- the mix and what goes with each archetype ------------------------------------------------------------------
    section("Which songs become which archetype", "Left: how the songs of your selection split, next to the whole file. Right: the share of each archetype's songs that have a given trait (whole file, with a 95% interval).")
    left, right = st.columns(2)
    with left:
        mix = pd.DataFrame({"archetype": ARCHETYPES})
        mix["In your selection"] = mix["archetype"].map(chosen["archetype"].value_counts(normalize=True)).fillna(0.0)
        mix["Whole file"] = mix["archetype"].map(per_song["archetype"].value_counts(normalize=True)).fillna(0.0)
        plot(ch.vbar(mix, "archetype", ["In your selection", "Whole file"], "Share of the songs, by archetype", colors={"In your selection": theme.BLUE, "Whole file": "#b7d3f6"},
                     y_title="Share of the songs", fmt=".0%", height=400, show_text=True), key="ar_mix")
    with right:
        attrs = tables["p6c_attributes"]
        trait = st.selectbox("Trait", list(dict.fromkeys(attrs["attribute"].tolist())), key="ar_trait")
        part = attrs[attrs["attribute"] == trait]
        reference = part[part["archetype"] == "All songs"]
        bars = part[part["archetype"] != "All songs"]
        fig = ch.interval_bars(bars, "archetype", "share", "low", "high", title=f"{trait}: share of each archetype's songs", fmt=".0%", y_title="Share of the songs", height=360)
        fig.update_traces(marker_color=[theme.ARCHETYPE_COLORS.get(a, theme.BLUE) for a in bars["archetype"]])
        if not reference.empty:
            fig.add_hline(y=float(reference["share"].iloc[0]), line=dict(color=theme.INK_2, dash="dot", width=1.4),
                          annotation_text=f"All songs: {T.pct(reference['share'].iloc[0], 0)}", annotation_position="top left", annotation_font=dict(size=12, color=theme.INK_2))
        plot(fig, key="ar_trait_chart")
    ref_share = attrs[attrs["archetype"] == "All songs"].set_index("attribute")["share"]
    every = attrs[attrs["archetype"] != "All songs"].assign(ref=lambda d: d["attribute"].map(ref_share))
    clear_count = ((every["low"] > every["ref"]) | (every["high"] < every["ref"])).groupby(every["attribute"]).sum()
    flat = [f"\"{name}\"" for name, n in clear_count.items() if n == 0]
    st.caption("A trait goes with an archetype only when its interval stays clear of the dotted line."
               + (f" These traits do not separate the archetypes at all: {', '.join(flat)}." if flat else "")
               + f" What you see are tendencies among {T.n(len(per_song))} songs, not rules for a new song.")

    # ---- the songs ------------------------------------------------------------------------------------------------------
    section("The songs", "Click a row to open that song's profile. Use the column headers to sort and the search icon of the table to find a song or an artist.")
    view = chosen.assign(order=chosen["archetype"].map({a: i for i, a in enumerate(ARCHETYPES)})).sort_values(["order", "entry_date"], ascending=[True, False]).reset_index(drop=True)
    shown = pd.DataFrame({"Song": view["track_label"], "Artist": view["primary_artist"], "Arrived": view["entry_date"].dt.date, "Debut place": view["entry_rank"],
                          "Archetype": view["archetype"], "Placement": view["firmness"], "Agreement (%)": (view["agreement"] * 100).round(0),
                          "Chart days in the first 30": view["days_on_chart_first_month"], "Score, day 1": view["score_day_1"], "Score, day 30": view["score_day_30"],
                          "Arrival-day risk": view["risk_third_arrival"]})
    event = table(shown, column_config={"Song": st.column_config.TextColumn(width="large"), "Arrived": st.column_config.DateColumn(format="D MMM YYYY"),
                                        "Agreement (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                                        "Chart days in the first 30": st.column_config.ProgressColumn(min_value=0, max_value=30, format="%d")},
                  height=min(560, 36 * min(len(shown), 13) + 44), key="ar_table", select=True)
    rows = list(event.selection.rows) if event is not None and getattr(event, "selection", None) else []
    a, b = st.columns([3, 2], vertical_alignment="center")
    with a:
        download_csv(shown, "song_archetypes.csv", key="ar_csv")
    with b:
        if rows:
            song = view.iloc[rows[0]]
            if st.button(f"Open the profile of {ch.shorten(song['track_label'], 30)}", icon=":material/open_in_new:", key="ar_open"):
                st.session_state["sp_song"] = int(song["track_id"])
                pages = st.session_state.get("_pages")
                if pages and "song-profile" in pages:
                    st.switch_page(pages["song-profile"])
        else:
            st.caption("Click a row to open that song's profile.")


# ============================================================================================== tab 2: the artists
def _artists_tab(ctx, tables: dict) -> None:
    segments, artists = tables["p6c_artist_segments"], tables["p6c_artists"]
    st.caption(f"{T.n(len(artists))} artists with at least one song on the chart, grouped by their footprint: how many songs, how long they last, how high they sit and how many are on at once. "
               "This is a picture of the whole file, so the sidebar filters do not apply here.")
    ordered = segments.set_index("segment").reindex(SEGMENTS).reset_index()
    cards([{"tag": f"{T.pct(row['share_of_artists'], 0)} of the artists", "title": row["segment"],
            "text": (f"{T.n(row['artists'])} artists with {T.n(row['songs_total'])} songs. The typical artist has {T.num(row['songs_median'], 0)} song(s), {T.num(row['chart_days_median'], 0)} chart days "
                     f"and an average place of {T.num(row['mean_place_median'], 0)}. This group holds {T.pct(row['share_of_chart_days'], 0)} of all chart days. For example: {row['examples']}.")}
           for _, row in ordered.iterrows()], min_width=245)

    section("A map of the artists", "Every dot is an artist. Artists close together have a similar footprint. The size of a dot is the artist's number of chart days.")
    chosen = st.pills("Segments to show", SEGMENTS, selection_mode="multi", default=SEGMENTS, key="ar_segments", help="None chosen means all four.") or SEGMENTS
    c3, c2 = st.columns([2.2, 1.6], vertical_alignment="bottom")
    names = sorted(artists["artist"].tolist(), key=str.casefold)
    highlighted = c3.selectbox("Highlight an artist", ["(none)"] + names, key="ar_highlight")
    only_certain = c2.toggle("Only certain artists", value=False, key="ar_certain", help="Keep only the artists that land in their own segment in at least 9 of 10 re-draws of the artists.")
    frame = artists[artists["segment"].isin(chosen)]
    if only_certain:
        frame = frame[frame["certain"]]
    left, right = st.columns([3.1, 1.3])
    with left:
        plotted = frame.assign(songs_text=frame["songs"].astype(int).astype(str), days_text=frame["chart_days"].astype(int).astype(str), best_text=frame["best_place"].astype(int).astype(str),
                               agreement_text=(frame["agreement"] * 100).round(0).astype(int).astype(str) + "%")
        fig = ch.scatter_chart(plotted, "pc1", "pc2", color="segment", size="chart_days", order=[s for s in SEGMENTS if s in chosen], colors=theme.SEGMENT_COLORS, height=520,
                               name_col="artist", hover_cols=["songs_text", "days_text", "best_text", "agreement_text"],
                               hover_names={"songs_text": "Songs", "days_text": "Chart days", "best_text": "Best place", "agreement_text": "Same segment in re-draws"},
                               x_title="Axis 1: more songs, more chart days and better places, to the right",
                               y_title="Axis 2: many short-lived songs (up) or few long-lived ones (down)", opacity=0.75, size_max=30)
        if highlighted != "(none)":
            row = artists[artists["artist"] == highlighted].iloc[0]
            fig.add_scatter(x=[row["pc1"]], y=[row["pc2"]], mode="markers+text", text=[highlighted], textposition="top center", textfont=dict(size=13, color=theme.INK),
                            marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color=theme.INK, width=2.6)), showlegend=False, hoverinfo="skip")
        plot(fig, key="ar_map")
    with right:
        pca = saved_facts("p6c_facts")
        shares = (f"Axis 1 carries {T.pct(pca['pc1_share'], 0)} of the differences between artists and Axis 2 another {T.pct(pca['pc2_share'], 0)}. "
                  if "pc1_share" in pca and "pc2_share" in pca else "")
        callout(shares + "The artists on the far right have many songs, many chart days, a good average place and a high share of top-10 days. "
                "High on Axis 2 means many songs that each stay a short time (album drops); low means few songs that each stay long.", title="How to read the map")
        if highlighted != "(none)":
            row = artists[artists["artist"] == highlighted].iloc[0]
            callout(f"**{highlighted}** belongs to **{row['segment']}**: {T.n(row['songs'])} song(s), {T.n(row['chart_days'])} chart days on {T.n(row['days_present'])} different days, "
                    f"best place {T.n(row['best_place'])}, average place {T.num(row['mean_place'], 1)}. In {T.pct(row['agreement'], 0)} of the re-draws the artist lands in this segment"
                    + (" (certain)." if row["certain"] else " (not certain)."), title="The highlighted artist")

    section("Who holds the chart", "Share of the artists next to the share of all chart days: a few artists hold most of the chart.")
    left, right = st.columns([3, 2.4])
    with left:
        share = ordered.assign(share_of_artists_pct=ordered["share_of_artists"], share_of_days=ordered["share_of_chart_days"])
        fig = ch.vbar(share, "segment", ["share_of_artists_pct", "share_of_days"], "Share of the artists and of the chart days", names={"share_of_artists_pct": "Share of the artists", "share_of_days": "Share of the chart days"},
                      colors={"share_of_artists_pct": "#b7d3f6", "share_of_days": theme.BLUE}, y_title="Share", fmt=".0%", height=380, show_text=True)
        plot(fig, key="ar_share")
    with right:
        top = ordered.set_index("segment")
        cat = top.loc["Catalogue artists"]
        callout(f"The **{T.n(cat['artists'])} catalogue artists** are {T.pct(cat['share_of_artists'], 0)} of the artists but hold **{T.pct(cat['share_of_chart_days'], 0)}** of all chart days: "
                f"the typical one has {T.num(cat['songs_median'], 0)} songs, up to {T.num(cat['peak_songs_median'], 0)} of them on the chart at once. The {T.n(top.loc['Passing visitors']['artists'])} passing visitors "
                f"are {T.pct(top.loc['Passing visitors']['share_of_artists'], 0)} of the artists and hold {T.pct(top.loc['Passing visitors']['share_of_chart_days'], 1)} of the chart days. "
                "This matches the concentration page: the chart is concentrated in a few catalogues.", title="What to notice")
        st.caption("The segments are descriptions of a footprint, not types of artist that a new act can be assigned to.")

    section("The artists")
    table_frame = pd.DataFrame({"Artist": artists["artist"], "Segment": artists["segment"], "Songs": artists["songs"], "Chart days": artists["chart_days"], "Days on the chart": artists["days_present"],
                                "Best place": artists["best_place"], "Average place": artists["mean_place"].round(1), "Days in the top 10 (%)": (artists["top10_share"] * 100).round(0),
                                "Chart days per song": artists["days_per_song"].round(1), "Most songs at once": artists["peak_songs"], "Same segment in re-draws (%)": (artists["agreement"] * 100).round(0),
                                "Certain": artists["certain"].map({True: "yes", False: "no"})}).sort_values("Chart days", ascending=False)
    table_frame = table_frame[table_frame["Segment"].isin(chosen)]
    table(table_frame, height=min(480, 36 * min(len(table_frame), 11) + 44))
    download_csv(table_frame, "artist_segments.csv", key="ar_csv2")


# ============================================================================================== tab 3: how firm
def _firm_tab(tables: dict) -> None:
    scan, stability, stress = tables["p6c_k_scan_songs"], tables["p6c_stability"], tables["p6c_stress_tests"]
    landmark, thirds, per_song = tables["p6c_landmark"], tables["p6c_risk_thirds"], tables["p6c_song_archetypes"]
    st.caption("A clustering method always returns groups, whether the data hold groups or not. Phase 6C therefore spent most of its effort on how far to believe them: "
               "whole artists were drawn again 1,000 times (the songs of one artist are alike), artists were held out, and the method, the month and the songs were changed.")

    section("Why five groups", "The rule: count up from 2 groups, stop at the first number where some group dissolves (it is re-found in under 60% of its songs), "
                               "and take the largest number before that point at which every group is firm (at least 75%).")
    chosen_k = int(scan.loc[scan["chosen"], "k"].iloc[0])
    long = scan.melt(id_vars="k", value_vars=["silhouette", "ari_mean", "weakest"], var_name="measure", value_name="value")
    long["measure"] = long["measure"].map({"silhouette": "Silhouette (how separate the groups look)", "ari_mean": "Agreement of the whole grouping across re-draws (ARI)",
                                           "weakest": "Re-found share of the weakest group (Jaccard)"})
    left, right = st.columns([3.1, 1.3])
    with left:
        fig = ch.line_chart(long, "k", "value", group="measure", markers=True, height=420, width=2.6, x_title="Number of groups", y_title="Score (1 = perfect)",
                            colors={long["measure"].unique()[0]: theme.VIOLET, long["measure"].unique()[1]: theme.BLUE, long["measure"].unique()[2]: theme.ORANGE},
                            hover="%{fullData.name}<br>%{x} groups: %{y:.2f}<extra></extra>")
        fig.add_vrect(x0=chosen_k - 0.4, x1=chosen_k + 0.4, fillcolor=theme.BLUE_SOFT, opacity=0.9, line_width=0, layer="below")
        fig.add_hline(y=0.75, line=dict(color=theme.AXIS, dash="dot", width=1.3), annotation_text="firm: 0.75", annotation_position="bottom right", annotation_font=dict(size=11, color=theme.MUTED))
        fig.add_hline(y=0.60, line=dict(color=theme.AXIS, dash="dash", width=1.3), annotation_text="dissolved: 0.60", annotation_position="bottom right", annotation_font=dict(size=11, color=theme.MUTED))
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(range=[0.3, 1.02])
        plot(fig, key="ar_kscan")
    with right:
        row = scan[scan["chosen"]].iloc[0]
        nxt = scan[scan["k"] == chosen_k + 1]
        extra = f" With {chosen_k + 1} groups the weakest falls to {T.num(nxt['weakest'].iloc[0], 2)} and {T.plural(int(nxt['dissolved_groups'].iloc[0]), 'group')} dissolve." if not nxt.empty else ""
        callout(f"With **{chosen_k} groups** all {int(row['firm_groups'])} are firm: the weakest is re-found with a score of **{T.num(row['weakest'], 2)}**, and the whole grouping agrees with its re-draws at an ARI of "
                f"**{T.num(row['ari_mean'], 2)}**.{extra} Two groups look more separate on the silhouette, but that split only divides the brief visitors from everything else.", title="What the scan says")

    section("How firm each archetype is", "Re-found share (Jaccard score) of every archetype over 1,000 draws of whole artists. The bar is the average; the thin line is the middle 95% of the draws. "
                                          "Above the dotted line (0.75) an archetype counts as firm; below the dashed line (0.60) it counts as dissolved.")
    left, right = st.columns(2)
    with left:
        ordered = stability.set_index("archetype").reindex(ARCHETYPES).reset_index()
        fig = ch.interval_bars(ordered, "archetype", "jaccard_mean", "jaccard_low", "jaccard_high", title="Re-found share of each archetype (Jaccard score)", fmt=".2f", y_title="Jaccard score", height=400)
        fig.update_traces(marker_color=[theme.ARCHETYPE_COLORS[a] for a in ordered["archetype"]])
        fig.add_hline(y=0.75, line=dict(color=theme.INK_2, dash="dot", width=1.4))
        fig.add_hline(y=0.60, line=dict(color=theme.INK_2, dash="dash", width=1.4))
        fig.update_yaxes(range=[0, 1.05])
        plot(fig, key="ar_jaccard")
    with right:
        firm = pd.crosstab(per_song["archetype"], per_song["firmness"], normalize="index").reindex(index=ARCHETYPES, columns=FIRMNESS_ORDER).fillna(0.0)
        firm.columns = [FIRMNESS_TEXT[c] for c in firm.columns]
        frame = firm.reset_index().rename(columns={"archetype": "archetype"})
        fig = ch.vbar(frame, "archetype", list(firm.columns), "How firmly each song sits in its archetype", barmode="stack", colors=FIRMNESS_COLORS, y_title="Share of the songs", fmt=".0%", height=400, legend="bottom")
        plot(fig, key="ar_firmness")
    weakest = ordered.sort_values("jaccard_mean").iloc[0]
    callout(f"Every archetype is firm on average (above 0.75). The weakest is **{weakest['archetype']}** (**{T.num(weakest['jaccard_mean'], 2)}**), which dissolves in **{T.pct(weakest['share_of_draws_dissolved'], 0)}** of the draws; "
            f"only **{T.pct(weakest['share_songs_agree_90'], 0)}** of its songs are in place in at least 90% of the draws, against **{T.pct(ordered['share_songs_agree_90'].max(), 0)}** for the firmest archetype. "
            f"Overall {T.pct((per_song['firmness'] == 'firm').mean(), 0)} of the songs are firmly placed and {T.n(int((per_song['firmness'] == 'border').sum()))} sit on a border. "
            "If one archetype has to be treated with care, it is a month-long fade or a slow climber.")

    section("Do the archetypes survive a change of method?", "Each run changes one thing and asks whether the same groups come back. ARI 1 = the same grouping; about 0 = no agreement.")
    runs = stress.sort_values("ari", ascending=False).reset_index(drop=True)
    tone = [theme.BLUE if v >= 0.85 else ("#86b6ef" if v >= 0.7 else theme.ORANGE) for v in runs["ari"]]
    left, right = st.columns([3, 2.4])
    with left:
        fig = ch.hbar(runs, "run", "ari", "Agreement with the original grouping (ARI)", colors=tone, fmt=".2f", height=max(300, 36 * len(runs) + 80))
        fig.update_xaxes(range=[0, 1.12])
        plot(fig, key="ar_stress")
    with right:
        callout("**Blue**: the same archetypes come back (ARI 0.85 or more). **Light blue**: most of them come back (0.70 to 0.85). **Orange**: the grouping moves a lot. "
                "Counting only presence (the place is ignored) is expected to move it most, because the slow climbers and the lasting hits are told apart mainly by their places.", title="How to read the bars")
    with st.expander("What each run changes (the numbers behind the chart)"):
        shown = pd.DataFrame({"Run": runs["run"], "What changes": runs["what_changes"], "Songs": runs["songs_used"], "ARI": runs["ari"].round(2),
                              "Songs in the matching archetype (%)": (runs["same_group_share"] * 100).round(0)})
        table(shown, column_config={"What changes": st.column_config.TextColumn(width="large")}, height=min(440, 36 * len(shown) + 44))
        download_csv(shown, "archetype_stress_tests.csv", key="ar_csv3")

    section("Could we have seen it coming on the arrival day?", "Phase 6B scored every new song's risk of leaving early from what was known on its arrival day and split the songs into thirds.")
    left, right = st.columns([1, 1])
    with left:
        spread = thirds.set_index("archetype").reindex(ARCHETYPES).reset_index()
        fig = ch.vbar(spread, "archetype", RISK_ORDER, "Share of each archetype's songs in each arrival-day risk third", barmode="stack",
                      names={"lowest risk": "Lowest risk third", "middle": "Middle third", "highest risk": "Highest risk third"},
                      colors={"lowest risk": theme.BLUE, "middle": "#b7d3f6", "highest risk": theme.ORANGE}, y_title="Share of the songs", fmt=".0%", height=400)
        plot(fig, key="ar_thirds")
    with right:
        lasting, brief = spread.set_index("archetype").loc["Lasting hits"], spread.set_index("archetype").loc["Brief visitors"]
        callout(f"The arrival-day score does lean the right way: **{T.pct(lasting['lowest risk'], 0)}** of the lasting hits came from the lowest-risk third and **{T.pct(lasting['highest risk'], 0)}** from the highest; "
                f"for the brief visitors it is **{T.pct(brief['lowest risk'], 0)}** and **{T.pct(brief['highest risk'], 0)}**. But every archetype has songs in every third, so the arrival day alone does not tell the shape of the month.",
                title="What to notice")
        stable = landmark[landmark["enough_to_estimate"]]
        if not stable.empty:
            st.markdown("**Songs that reached day 30**")
            side_table(["Archetype", "Still going 90 days on", "Median further stay"],
                       [(theme.ARCHETYPE_COLORS.get(r["archetype"]), r["archetype"],
                         [T.pct(r["still_going_90"], 0), f"{T.num(r['median_further_days'], 0)} days ({T.num(r['median_low'], 0)} to {T.num(r['median_high'], 0)})"])
                        for _, r in stable.iterrows()])
            who = " and ".join(f"the {a.lower()}" for a in stable["archetype"])
            counts = " and ".join(T.n(n) for n in stable["at_risk_day_30"])
            st.caption(f"Only {who} have enough songs still on the chart at day 30 for this estimate ({counts} songs). \"Still going 90 days on\" is the share of those songs that are still on the chart "
                       "90 days after day 30; the median further stay counts the extra chart days after day 30, with its 95% band in brackets.")


def render() -> None:
    ctx = current()
    page_header("Research", "Song archetypes and artist segments",
                "The shape of a song's first month (Phase 6C), the footprint of an artist, and how far to believe the groups.", ctx.chips)
    tables = require_reports("p6c_archetypes", "p6c_song_archetypes", "p6c_first_months", "p6c_attributes", "p6c_artist_segments", "p6c_artists", "p6c_k_scan_songs",
                             "p6c_stability", "p6c_stress_tests", "p6c_landmark", "p6c_risk_thirds", "p6b_stays")
    songs, artists, firm = st.tabs(["Song archetypes", "Artist segments", "How firm are the groups"])
    with songs:
        _songs_tab(ctx, tables)
    with artists:
        _artists_tab(ctx, tables)
    with firm:
        _firm_tab(tables)
