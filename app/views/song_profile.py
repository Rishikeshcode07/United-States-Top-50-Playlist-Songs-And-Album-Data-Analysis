"""
app/views/song_profile.py  -  page 8: the profile of one song.

WHAT THIS FILE IS
    Pick one song and see its whole life on the chart: its KPIs, its place and popularity day by day, how many days it spent in each
    band of places, each of its runs, and how it compares with the other songs of the selection.

WHY WE NEED IT
    The leaderboard compares songs; a profile explains one. It is the page to open when someone asks "what happened to this song?".

WHAT HAPPENS WHEN IT RUNS
    Everything is calculated from the rows of this song that match the sidebar filters (so with narrow dates the profile shows only
    that part of its life). "Better than" figures compare the song with every other song of the same selection: 100 % would mean
    better than all of them. A day off the chart is a gap, not a zero.

EXPECTED OUTCOME
    A one-page story of a song, with numbers that match the leaderboard.

MOTIVE
    Let a reader go from "the top songs" to "this song" in one click.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, facts, kpi_cards, page_header, plot, section, stop_if_empty, table
from app.ui.state import current
from src.dash import movement as V
from src.dash import texts as T

# measure -> (column, True when a LOW value is better, label)
COMPARE = [("days_on_chart", False, "Days on the chart"), ("best_rank", True, "Best place"), ("avg_rank", True, "Average place"),
           ("chart_points", False, "Chart points"), ("avg_popularity", False, "Average popularity"), ("rank_volatility_index", True, "Steadiness (low volatility)")]


def _share_better(values: pd.Series, mine: float, low_is_better: bool) -> float:
    """The share of the other songs that this song beats on a measure (0 to 1); NaN when the song has no value."""
    values = values.dropna()
    if pd.isna(mine) or values.empty:
        return float("nan")
    return float((values > mine).mean() if low_is_better else (values < mine).mean())


def render() -> None:
    ctx = current()
    page_header("Songs", "Song profile", "One song from its first chart day to its last: place, popularity, runs and how it compares with the other songs.", ctx.chips)
    stop_if_empty(ctx.rows)
    songs = ctx.songs().sort_values(["chart_points", "track_label"], ascending=[False, True]).reset_index(drop=True)
    labels = dict(zip(songs["track_id"], songs["track_label"]))
    if st.session_state.get("sp_song") not in labels:
        st.session_state["sp_song"] = int(songs["track_id"].iloc[0])
    pick, cover_col = st.columns([4, 1], vertical_alignment="bottom")
    track_id = pick.selectbox("Song", list(labels), key="sp_song", format_func=lambda i: labels.get(i, str(i)), help="Type to search. The list follows the sidebar filters.")
    show_cover = cover_col.toggle("Show the album cover", key="sp_cover", help="The picture is loaded from the web.")
    song = songs[songs["track_id"] == track_id].iloc[0]
    mine = ctx.rows[ctx.rows["track_id"] == track_id].sort_values("date")

    kpi_cards(columns=6, items=[
        {"label": "Days on chart", "value": T.n(int(song["days_on_chart"])), "tone": "blue", "sub": f"{T.span(song['first_date'], song['last_date'])}"},
        {"label": "Best place", "value": f"#{int(song['best_rank'])}", "tone": "green", "sub": f"first reached {T.day(mine.loc[mine['position'] == song['best_rank'], 'date'].min())}; {int(song['days_at_number_one'])} days at number one"},
        {"label": "Average place", "value": T.num(song["avg_rank"]), "tone": "aqua", "sub": f"median {T.num(song['median_rank'], 0)}; {T.pct(song['share_in_top10'], 0)} of its days in the top 10"},
        {"label": "Rank volatility", "value": T.num(song["rank_volatility_index"]), "tone": "yellow", "sub": "standard deviation of its daily place" if pd.notna(song["rank_volatility_index"]) else "needs two or more days"},
        {"label": "Chart points", "value": T.n(int(song["chart_points"])), "tone": "pink", "sub": f"{T.num(song['chart_points'] / song['days_on_chart'], 1)} a day (51 minus the place)"},
        {"label": "Popularity", "value": T.num(song["avg_popularity"], 0), "unit": "average", "tone": "violet",
         "sub": f"peak {T.num(song['peak_popularity'], 0)}; {T.signed(song['popularity_change'], 0)} from first to last day"},
    ])

    left, right = st.columns([3, 1])
    with left:
        days = ctx.summary["date"].sort_values()
        span = days[(days >= song["first_date"]) & (days <= song["last_date"])]
        daily = mine.set_index("date")[["position", "popularity_clean", "popularity_trend_7d"]].reindex(span).rename_axis("date").reset_index()
        plot(ch.song_timeline(daily, f"{song['track_label']}"), key="sp_timeline")
    with right:
        st.markdown(f"**{song['song']}**")
        st.caption(song["artist_credit"])
        if show_cover and isinstance(song["album_cover_url"], str) and song["album_cover_url"].startswith("http"):
            st.image(song["album_cover_url"], width=180)
        facts([("Release type", str(song["album_type"]).title()), ("Release size", f"{int(song['total_tracks'])} track{'s' if song['total_tracks'] != 1 else ''}"),
               ("Explicit", "yes" if song["is_explicit"] else "no"), ("Collaboration", "yes" if song["is_collab"] else "no"), ("Length", f"{T.num(song['duration_min'], 1)} min"),
               ("Stay class", song["longevity_class"]), ("First day", T.day(song["first_date"])), ("Last day", T.day(song["last_date"]))])

    a, b = st.columns([2, 3])
    with a:
        band = V.place_band(mine["position"]).value_counts().reindex(V.BAND_LABELS, fill_value=0).rename("days").reset_index().rename(columns={"index": "band", "position": "band"})
        band.columns = ["band", "days"]
        fig = ch.vbar(band, "band", "days", "Days spent in each band of places", colors={"days": theme.BLUE}, x_title="Chart places", y_title="Days", show_text=True, height=340)
        plot(fig, key="sp_bands")
    with b:
        runs = V.runs_table(mine, ctx.bounds.first_snapshot, ctx.bounds.last_snapshot).sort_values("start_date").reset_index(drop=True)
        shown = pd.DataFrame({"Run": range(1, len(runs) + 1), "First day": runs["start_date"].dt.date, "Last day": runs["end_date"].dt.date, "Days": runs["days"],
                              "Arrived at": runs["entry_position"], "Best place": runs["best_position"], "Left from": runs["exit_position"],
                              "Began as": runs["entry_kind"].map({"New entry": "new", "Re-entry": "return", "Chart start": "start"}).fillna(""),
                              "Complete": runs["complete"].map({True: "yes", False: "no"})})
        st.markdown("##### The runs of this song")
        table(shown, column_config={"Run": st.column_config.NumberColumn("Run", width="small"), "First day": st.column_config.DateColumn(format="D MMM YYYY"),
                                    "Last day": st.column_config.DateColumn(format="D MMM YYYY")}, height=min(300, 38 * len(shown) + 40))
        st.caption("A run is an unbroken stay on the chart. Complete = no when the stay began before or ended after the days of the data or of the chosen dates.")

    section("How this song compares", f"Against the other {T.n(len(songs) - 1)} songs of the same selection. 100 % would mean better than all of them.")
    others = songs[songs["track_id"] != track_id]
    shown_value = {"days_on_chart": lambda v: T.n(int(v)), "best_rank": lambda v: f"#{int(v)}", "avg_rank": lambda v: T.num(v, 1), "chart_points": lambda v: T.n(int(v)),
                   "avg_popularity": lambda v: T.num(v, 1), "rank_volatility_index": lambda v: T.num(v, 2)}
    rows = [{"Measure": label, "This song": shown_value[col](song[col]) if pd.notna(song[col]) else T.DASH, "Better than": _share_better(others[col], song[col], low) * 100}
            for col, low, label in COMPARE]
    comp = pd.DataFrame(rows)
    table(comp, column_config={"Better than": st.column_config.ProgressColumn("Better than (% of other songs)", min_value=0, max_value=100, format="%.0f%%")}, height=38 * len(comp) + 40)
    strong = comp.dropna(subset=["Better than"]).sort_values("Better than", ascending=False)
    if len(strong) >= 2:
        callout(f"**{song['song']}** is strongest on **{strong.iloc[0]['Measure'].lower()}** (better than **{strong.iloc[0]['Better than']:.0f}%** of the other songs) and weakest on "
                f"**{strong.iloc[-1]['Measure'].lower()}** (**{strong.iloc[-1]['Better than']:.0f}%**). These are comparisons inside the current selection; they do not say why the song did well.")
