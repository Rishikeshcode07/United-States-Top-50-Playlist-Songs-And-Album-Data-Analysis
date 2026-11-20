"""
app/views/rank_trends.py  -  page 5: song ranking trends.

WHAT THIS FILE IS
    The page that draws how the chart place of several songs moved over time. The person either lets the page pick the top songs
    by a measure (most chart points, longest stay, best place ...) or chooses songs by name, then looks at them as lines, as one
    small panel per song, or as a heatmap. A switch lines the songs up by their debut day so that songs of different years can
    be compared at the same age.

WHY WE NEED IT
    "Song ranking trend charts" are a core module of the brief. A single number such as the average rank hides the shape of a
    song's life: a quick launch and a long fade look very different from a slow climb.

WHAT HAPPENS WHEN IT RUNS
    The songs come from the selection of the sidebar. A day on which the song was not on the chart is a gap in the line (it is not
    filled in). The optional smoothing is a 7-day moving average of the place and is only a visual aid; every table shows exact numbers.

EXPECTED OUTCOME
    Up to twelve songs drawn together, a short table of their numbers and a reading note that is written from those numbers.

MOTIVE
    Let the reader see the shape of each song's run, not only its average.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, download_csv, kpi_cards, page_header, plot, section, stop_if_empty, table
from app.ui.state import current
from src.dash import movement as V
from src.dash import texts as T

# what the page can rank songs by: label -> (column of the song table, True when a LOW value is better)
MEASURES = {
    "Chart points (51 minus the place, summed)": ("chart_points", False),
    "Days on the chart": ("days_on_chart", False),
    "Best place reached": ("best_rank", True),
    "Average place (best first)": ("avg_rank", True),
    "Average popularity": ("avg_popularity", False),
    "Days at number one": ("days_at_number_one", False),
    "Rank volatility (most volatile first)": ("rank_volatility_index", False),
}
MAX_SONGS = 12


def _pick_songs(songs: pd.DataFrame) -> list:
    """The track ids to draw, from the widgets of the page."""
    mode = st.segmented_control("Songs to show", ["Top songs by a measure", "Choose songs by name"], default="Top songs by a measure", key="rt_mode") or "Top songs by a measure"
    if mode.startswith("Top"):
        a, b = st.columns([3, 2])
        measure = a.selectbox("Rank the songs by", list(MEASURES), key="rt_measure")
        how_many = b.slider("How many songs", 1, MAX_SONGS, 4, key="rt_n")
        column, low_is_better = MEASURES[measure]
        ordered = songs.sort_values([column, "chart_points", "track_label"], ascending=[low_is_better, False, True], na_position="last")
        return ordered["track_id"].head(how_many).tolist()
    labels = dict(zip(songs["track_id"], songs["track_label"]))
    valid = set(labels)
    st.session_state["rt_pick"] = [i for i in st.session_state.get("rt_pick", []) if i in valid][:MAX_SONGS]
    if "rt_pick" not in st.session_state or not st.session_state["rt_pick"]:
        st.session_state["rt_pick"] = songs.sort_values("chart_points", ascending=False)["track_id"].head(5).tolist()
    return st.multiselect(f"Songs (up to {MAX_SONGS})", list(labels), key="rt_pick", format_func=lambda i: labels.get(i, str(i)), max_selections=MAX_SONGS,
                          placeholder="Type a song or an artist")


def render() -> None:
    ctx = current()
    page_header("Songs", "Song ranking trends",
                "The chart place of the songs you pick, day by day. Place 1 is at the top of every chart, and a gap in a line means the song was off the chart.", ctx.chips)
    stop_if_empty(ctx.rows)
    songs = ctx.songs()
    ids = _pick_songs(songs)
    v1, v2, v3 = st.columns([2.2, 1.3, 1.3], vertical_alignment="bottom")
    view = v1.segmented_control("View", ["Lines", "One panel per song", "Heatmap"], default="Lines", key="rt_view") or "Lines"
    align = v2.toggle("Line up at the debut day", key="rt_align", help="Puts every song's first chart day at 0, so that songs from different months can be compared at the same age.")
    smooth = v3.toggle("7-day average", key="rt_smooth", help="Smooths each line with a 7-day moving average of the place (a visual aid only).")
    if not ids:
        callout("Choose at least one song above.", title="No song chosen", tone="warn")
        st.stop()

    names = dict(zip(songs["track_id"], songs["track_label"]))
    chosen = ctx.rows[ctx.rows["track_id"].isin(ids)]
    if align:
        matrix = chosen.pivot(index="calendar_age_days", columns="track_id", values="position").reindex(columns=[i for i in ids if i in set(chosen["track_id"])])
        matrix = matrix.reindex(range(0, int(matrix.index.max()) + 1)) if not matrix.empty else matrix
        matrix.index.name = None
    else:
        matrix = V.rank_matrix(ctx.rows, ids).reindex(ctx.summary["date"].sort_values())
    if smooth and not matrix.empty:
        matrix = matrix.rolling(7, min_periods=1).mean().where(matrix.notna())          # days off the chart stay empty
    low, high = ctx.state.rank_low, ctx.state.rank_high

    best = songs[songs["track_id"].isin(ids)]
    kpi_cards(columns=4, items=[
        {"label": "Songs shown", "value": T.n(len(ids)), "tone": "blue", "sub": f"{T.n(int(best['days_on_chart'].sum()))} chart days together"},
        {"label": "Best place reached", "value": f"#{int(best['best_rank'].min())}" if len(best) else T.DASH, "tone": "green",
         "sub": f"by {best.sort_values('best_rank')['track_label'].iloc[0]}" if len(best) else ""},
        {"label": "Longest stay", "value": T.n(int(best["days_on_chart"].max())) if len(best) else T.DASH, "unit": "days", "tone": "violet",
         "sub": f"{best.sort_values('days_on_chart', ascending=False)['track_label'].iloc[0]}" if len(best) else ""},
        {"label": "Most volatile", "value": T.num(best["rank_volatility_index"].max()) if len(best) else T.DASH, "tone": "yellow",
         "sub": f"rank volatility index of {best.sort_values('rank_volatility_index', ascending=False)['track_label'].iloc[0]}" if best["rank_volatility_index"].notna().any() else "needs songs with two or more days"},
    ])

    x_title = "Days since the song's first chart day" if align else None
    if view == "Lines":
        fig = ch.rank_lines(matrix, names, f"Chart place of {len(ids)} song{'s' if len(ids) != 1 else ''}", height=520, x_title=x_title, low=low, high=high, date_axis=not align,
                            markers=len(matrix) <= 40)
    elif view == "One panel per song":
        fig = ch.rank_panels(matrix, names, columns=3 if len(ids) > 4 else 2, title="One panel per song (place 1 at the top)", low=low, high=high, date_axis=not align)
    else:
        grid = matrix.T.copy()
        grid.index = [ch.shorten(names.get(i, i), 40) for i in grid.index]
        scale = [(0.0, "#104281"), (0.25, "#2a78d6"), (0.6, "#b7d3f6"), (1.0, "#f4f3ef")]
        fig = ch.heat(grid, "Place of every song on every day (dark = high place, blank = off the chart)", colorscale=scale, zmin=1, zmax=50, show_text=False,
                      height=max(300, 34 * len(ids) + 140), colorbar_title="Place", hover="%{y}<br>%{x}: place %{z:.0f}<extra></extra>")
    plot(fig, key="rt_chart")

    if len(best):
        longest, volatile = best.sort_values("days_on_chart", ascending=False).iloc[0], best.sort_values("rank_volatility_index", ascending=False).iloc[0]
        steady = best.dropna(subset=["rank_volatility_index"]).sort_values("rank_volatility_index")
        text = (f"**{longest['track_label']}** stayed longest, **{T.n(int(longest['days_on_chart']))} days**, and reached place **#{int(longest['best_rank'])}**. ")
        if len(steady) >= 2:
            text += (f"The steadiest place belongs to **{steady.iloc[0]['track_label']}** (volatility {T.num(steady.iloc[0]['rank_volatility_index'])}), "
                     f"the biggest daily swings to **{volatile['track_label']}** ({T.num(volatile['rank_volatility_index'])}). ")
        text += "A low volatility means the song held about the same place from day to day; it says nothing about whether that place was high."
        callout(text)

    section("The numbers behind the chart")
    shown = best.set_index("track_id").loc[[i for i in ids if i in set(best["track_id"])]].reset_index()
    out = pd.DataFrame({"Song": shown["track_label"], "Days": shown["days_on_chart"], "Best place": shown["best_rank"], "Average place": shown["avg_rank"].round(1),
                        "Volatility": shown["rank_volatility_index"].round(2), "Days at #1": shown["days_at_number_one"], "Days in top 10": shown["days_in_top10"],
                        "First day": shown["first_date"].dt.date, "Last day": shown["last_date"].dt.date})
    table(out, column_config={"First day": st.column_config.DateColumn("First day", format="D MMM YYYY"), "Last day": st.column_config.DateColumn("Last day", format="D MMM YYYY")},
          height=min(480, 38 * len(out) + 40))
    download_csv(out, "ranking_trend_songs.csv", key="rt_csv")
