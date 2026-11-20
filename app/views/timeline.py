"""
app/views/timeline.py  -  page 2: the playlist timeline explorer.

WHAT THIS FILE IS
    The page where a reader travels through the chart one day at a time. Pick a day (slider, previous / next buttons, or the Play
    button) and the page shows the Top 50 of that day as bars coloured by how each song moved, the day's numbers, the biggest
    climbers and falls, the arrivals and the artists that hold the day. Below, two extra views: how today's top songs got there
    (a place map of the last days) and a side-by-side comparison of any two days.

WHY WE NEED IT
    The brief asks for a "playlist timeline explorer". A static chart cannot show how a chart behaves; moving through the days can.

WHAT HAPPENS WHEN IT RUNS
    The sidebar filters decide which songs count as "selected". Songs that the filters leave out stay on the chart in grey so the
    day still looks like a real Top 50. While Play is on, only the day view re-draws by itself (every second or so) and moves to the
    next chart day; the two extra views wait until the person pauses.

EXPECTED OUTCOME
    A day you can scrub, replay and compare, with every number recalculated from the chosen filters.

MOTIVE
    Make the history of the chart something you can watch, not only read.
"""
from __future__ import annotations

from bisect import bisect_right

import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, download_csv, kpi_cards, page_header, plot, section, stop_if_empty, table
from app.ui.state import current
from src.dash import movement as V
from src.dash import texts as T

DAY, DAY_B = "tl_day", "tl_day_b"
PLAY, SPEED, SEEN = "tl_play", "tl_speed", "tl_seen"
SPEEDS = {"Slow": 2.0, "Normal": 1.2, "Fast": 0.7}


# ---------------------------------------------------------------------------------------------------------- helpers
def _snapshot_dates(ctx) -> list:
    """Every chart day inside the chosen dates, as plain dates, oldest first."""
    return [d.date() for d in ctx.summary["date"].sort_values()]


def _step(delta: int, dates: list) -> None:
    """Callback of the previous / next buttons: move by `delta` chart days (it runs before the slider is drawn)."""
    now = st.session_state.get(DAY) or dates[-1]
    index = max(bisect_right(dates, now) - 1, 0)
    st.session_state[DAY] = dates[min(max(index + delta, 0), len(dates) - 1)]


def _day_table(ctx, day) -> pd.DataFrame:
    """The Top 50 of `day` (the whole chart); `selected` says whether the song matches the sidebar filters."""
    on_day = ctx.rows.loc[ctx.rows["date"] == pd.Timestamp(day), "track_id"]
    return V.day_chart(ctx.data.playlist, day, keep_ids=on_day)


def _move_word(row) -> str:
    kind = row["movement"]
    if kind in ("Up", "Down") and pd.notna(row["rank_change"]):
        return f"{kind.lower()} {abs(int(row['rank_change']))}"
    return {"New entry": "new", "Re-entry": "back", "Chart start": "start"}.get(kind, kind.lower())


def _small(frame: pd.DataFrame, extra: "dict | None" = None) -> pd.DataFrame:
    """The compact table used in the right-hand column."""
    out = pd.DataFrame({"Place": frame["position"], "Song": frame["song_clean"], "Artist": frame["primary_artist"]})
    for name, values in (extra or {}).items():
        out[name] = values
    return out


# ----------------------------------------------------------------------------------------------------- the day view
def _day_view(ctx, dates: list) -> None:
    """Slider, buttons, KPIs, the Top 50 bars and the side panels for the chosen day (this part is what re-draws while Play is on)."""
    playing = bool(st.session_state.get(PLAY))
    if playing and st.session_state.get(SEEN) == st.session_state.get(DAY):          # nothing was moved by hand: this run is a timer tick
        now = st.session_state.get(DAY)
        st.session_state[DAY] = dates[(max(bisect_right(dates, now) - 1, 0) + 1) % len(dates)]

    left, middle, right = st.columns([1, 12, 1], vertical_alignment="bottom")
    with left:
        st.button("", icon=":material/skip_previous:", key="tl_prev", on_click=_step, args=(-1, dates), help="One chart day earlier")
    with right:
        st.button("", icon=":material/skip_next:", key="tl_next", on_click=_step, args=(1, dates), help="One chart day later")
    with middle:
        if len(dates) > 1:
            st.slider("Chart day", min_value=dates[0], max_value=dates[-1], key=DAY, format="ddd D MMM YYYY",
                      help="Drag to travel through the chart. The sidebar dates set the first and the last day.")
        else:
            st.session_state[DAY] = dates[0]
            st.caption(f"Only one chart day in the chosen dates: {T.day(dates[0])}")
    st.session_state[SEEN] = st.session_state[DAY]

    day = V.snapshot_on_or_before(ctx.calendar, pd.Timestamp(st.session_state[DAY]))
    if day is None:
        day = pd.Timestamp(dates[0])
    chart = _day_table(ctx, day)
    mine = chart[chart["selected"]]
    last_snapshot = pd.Timestamp(dates[-1]) == day and day == pd.Timestamp(ctx.bounds.last_date)

    if mine.empty:
        callout(f"None of the songs on the chart on {T.day(day)} matches the current filters. The whole Top 50 is shown in grey. "
                "Move to another day or widen the filters.", title="Nothing selected on this day", tone="warn")
    number_one = chart.iloc[0] if not chart.empty else None
    arrivals = mine[mine["movement"].isin(["New entry", "Re-entry"])]
    kpi_cards(columns=6, items=[
        {"label": "Chart day", "value": T.day(day), "tone": "blue", "sub": f"day {T.n(int(chart['snapshot_index'].iloc[0]))} of the chart" if not chart.empty else ""},
        {"label": "Songs selected", "value": f"{len(mine)} of {len(chart)}", "tone": "aqua", "sub": "the rest are grey in the chart below"},
        {"label": "Arrivals", "value": T.n(len(arrivals)), "tone": "violet",
         "sub": f"{int((arrivals['movement'] == 'New entry').sum())} new and {int((arrivals['movement'] == 'Re-entry').sum())} returning songs"},
        {"label": "Leave next day", "value": T.n(int(mine["exits_next_snapshot"].sum())) if not last_snapshot else T.DASH, "tone": "red",
         "sub": "songs on today's chart that are gone the next day" if not last_snapshot else "this is the last day of the data"},
        {"label": "Average popularity", "value": T.num(mine["popularity_clean"].mean()), "unit": "points", "tone": "yellow",
         "sub": f"{T.pct(mine['is_explicit'].astype(bool).mean(), 0) if len(mine) else T.DASH} of these songs are explicit"},
        {"label": "Number one", "value": (str(number_one["song_clean"])[:24] + ("..." if len(str(number_one["song_clean"])) > 24 else "")) if number_one is not None else T.DASH,
         "tone": "orange", "sub": f"{number_one['primary_artist']}, day {int(number_one['days_in_stint'])} of this run" if number_one is not None else ""},
    ])

    left, right = st.columns([3, 2])
    with left:
        show_outside = st.toggle("Show the songs outside the filters in grey", value=True, key="tl_outside",
                                 help="Switch off to draw only the songs that match the filters.")
        shown = chart if show_outside else mine
        fig = ch.day_bars(shown.assign(selected=shown["selected"]), title=f"The Top 50 on {T.day(day)}", height=max(260, 17 * len(shown) + 120),
                          show_outside=show_outside)
        plot(fig, key="tl_bars")
    with right:
        tabs = st.tabs(["Biggest climbers", "Biggest falls", "Arrivals"])
        with tabs[0]:
            ups = mine[mine["rank_change"] > 0].sort_values("rank_change", ascending=False).head(6)
            table(_small(ups, {"Up by": ups["rank_change"].astype("Int64")}), height=248) if not ups.empty else st.caption("No song climbed on this day.")
        with tabs[1]:
            downs = mine[mine["rank_change"] < 0].sort_values("rank_change").head(6)
            table(_small(downs, {"Down by": (-downs["rank_change"]).astype("Int64")}), height=248) if not downs.empty else st.caption("No song fell on this day.")
        with tabs[2]:
            table(_small(arrivals, {"How": arrivals["movement"].map({"New entry": "new", "Re-entry": "returning"})}), height=248) \
                if not arrivals.empty else st.caption("No song arrived on this day.")
        points = chart.groupby("primary_artist")["chart_points"].sum().sort_values(ascending=False).head(8).rename("points").reset_index()
        points["share"] = points["points"] / 1275 * 100
        fig = ch.hbar(points, "primary_artist", "share", "Who holds this day (share of the day's chart points)", color=theme.PINK, fmt=".1f", suffix="%", height=330)
        plot(fig, key="tl_artists")
        if st.toggle("Show the album covers of the top 5", value=False, key="tl_covers", help="The pictures are loaded from the web, so they need an internet connection."):
            top5 = chart.head(5)
            for column, (_, song) in zip(st.columns(5), top5.iterrows()):
                with column:
                    if isinstance(song["album_cover_url"], str) and song["album_cover_url"].startswith("http"):
                        st.image(song["album_cover_url"], width=110)
                    st.caption(f"{int(song['position'])}. {song['song_clean']}")
    download_csv(chart[["position", "song_clean", "artist_credit", "movement", "rank_change", "days_in_stint", "popularity_clean", "is_explicit", "album_type"]]
                 .rename(columns={"position": "place", "song_clean": "song", "days_in_stint": "days_in_run", "popularity_clean": "popularity"}),
                 f"top50_{day:%Y-%m-%d}.csv", "Download this day's chart (CSV)", key="tl_csv")
    st.session_state["_tl_day"] = day


# ------------------------------------------------------------------------------------------------------ extra views
def _tapestry(ctx, day) -> None:
    """The place of today's top songs over the days before: rows = songs, columns = days."""
    c1, c2 = st.columns(2)
    n_songs = c1.slider("Songs to follow (today's top places)", 5, 25, 12, key="tl_n")
    back = c2.select_slider("Days to look back", options=[14, 30, 60, 90], value=30, key="tl_back")
    chart = _day_table(ctx, day)
    top = chart[chart["selected"]].head(n_songs)
    if top.empty:
        callout("No selected song on this day, so there is nothing to follow.", tone="warn")
        return
    snapshots = ctx.calendar.drop_duplicates("date").set_index("date")["snapshot_index"]
    end_index = int(snapshots.loc[day])
    window = ctx.data.playlist[(ctx.data.playlist["snapshot_index"] > end_index - back) & (ctx.data.playlist["snapshot_index"] <= end_index)
                               & ctx.data.playlist["track_id"].isin(top["track_id"])]
    matrix = window.pivot(index="track_id", columns="date", values="position").reindex(top["track_id"])
    labels = dict(zip(top["track_id"], [f"{int(p)}  {ch.shorten(s, 28)}" for p, s in zip(top["position"], top["song_clean"])]))
    matrix.index = [labels[i] for i in matrix.index]
    scale = [(0.0, "#104281"), (0.25, "#2a78d6"), (0.6, "#b7d3f6"), (1.0, "#f4f3ef")]
    fig = ch.heat(matrix, f"Where the top {len(top)} songs of {T.day(day)} were in the {back} days before", colorscale=scale, zmin=1, zmax=50, fmt=".0f",
                  show_text=back <= 30, height=max(300, 28 * len(top) + 130), colorbar_title="Place", hover="%{y}<br>%{x|%d %b %Y}: place %{z:.0f}<extra></extra>")
    plot(fig, key="tl_tapestry")
    callout("Dark cells are high places, light cells low places, blank cells days off the chart. A song that is dark only in the last "
            "columns is a recent climber; a long dark row is a song that held the top for weeks. Row labels start with today's place.")


def _compare(ctx, dates: list, day) -> None:
    """Two days side by side: who rose, who fell, who arrived, who left."""
    c1, c2 = st.columns([2, 1])
    default_b = dates[max(bisect_right(dates, day.date()) - 1 - 7, 0)]
    if DAY_B not in st.session_state or not (dates[0] <= st.session_state[DAY_B] <= dates[-1]):
        st.session_state[DAY_B] = default_b
    if len(dates) < 2:
        st.caption("Two different days are needed for a comparison.")
        return
    with c1:
        st.slider("Compare with this earlier or later day", min_value=dates[0], max_value=dates[-1], key=DAY_B, format="ddd D MMM YYYY")
    top_n = c2.slider("How many places to compare", 10, 30, 20, key="tl_topn")
    day_b = V.snapshot_on_or_before(ctx.calendar, pd.Timestamp(st.session_state[DAY_B])) or pd.Timestamp(dates[0])
    a, b = V.day_chart(ctx.rows, day_b), V.day_chart(ctx.rows, day)
    if a.empty and b.empty:
        callout("No selected song on either day.", tone="warn")
        return
    merged = pd.merge(a[["track_id", "track_label", "position"]].rename(columns={"position": "place_a"}),
                      b[["track_id", "track_label", "position"]].rename(columns={"position": "place_b"}), on=["track_id", "track_label"], how="outer")
    merged = merged[(merged["place_a"] <= top_n) | (merged["place_b"] <= top_n)].rename(columns={"track_label": "label"})
    fig = ch.slope_chart(merged, T.day(day_b), T.day(day), top_n, f"The top {top_n}: {T.day(day_b)} against {T.day(day)}", height=max(460, 26 * top_n + 120))
    plot(fig, key="tl_slope")
    both = merged.dropna(subset=["place_a", "place_b"])
    both = both[(both["place_a"] <= top_n) & (both["place_b"] <= top_n)]
    kpi_cards(columns=4, items=[
        {"label": "Climbed", "value": T.n(int((both["place_b"] < both["place_a"]).sum())), "tone": "green", "sub": "in the top N on both days"},
        {"label": "Fell", "value": T.n(int((both["place_b"] > both["place_a"]).sum())), "tone": "red", "sub": "in the top N on both days"},
        {"label": "Arrived", "value": T.n(int(((merged["place_b"] <= top_n) & ~(merged["place_a"] <= top_n)).sum())), "tone": "blue", "sub": "in the top N only on the later day"},
        {"label": "Left", "value": T.n(int(((merged["place_a"] <= top_n) & ~(merged["place_b"] <= top_n)).sum())), "tone": "violet", "sub": "in the top N only on the other day"},
    ])


# ------------------------------------------------------------------------------------------------------------- page
def render() -> None:
    ctx = current()
    page_header("Playlist", "Timeline explorer",
                "Travel through the chart one day at a time. The bars show the Top 50 of the day, coloured by how each song moved since the day before.",
                ctx.chips)
    stop_if_empty(ctx.rows)
    dates = _snapshot_dates(ctx)
    if not dates:
        stop_if_empty(ctx.summary)
    if DAY not in st.session_state or not (dates[0] <= st.session_state[DAY] <= dates[-1]):
        st.session_state[DAY] = dates[-1]

    control, speed, note = st.columns([1.1, 2, 4], vertical_alignment="center")
    control.toggle("Play", key=PLAY, help="Moves through the chart by itself, one chart day at a time. Pause to open the other two views.")
    speed.select_slider("Speed", options=list(SPEEDS), value="Normal", key=SPEED, label_visibility="collapsed")
    note.caption("Press Play to watch the chart move; the other views appear when it is paused.")
    playing = bool(st.session_state.get(PLAY))

    if playing and len(dates) > 1:
        st.fragment(run_every=SPEEDS[st.session_state.get(SPEED, "Normal")])(_day_view)(ctx, dates)
        callout("Play is on. Switch it off to open the trajectory and comparison views.", title="Paused views", tone="info")
        return
    _day_view(ctx, dates)
    day = st.session_state["_tl_day"]

    section("Look closer", "Two views of the day above. They follow the day you chose with the slider.")
    tab_a, tab_b = st.tabs(["How today's top songs got here", "Compare two days"])
    with tab_a:
        _tapestry(ctx, day)
    with tab_b:
        _compare(ctx, dates, day)
