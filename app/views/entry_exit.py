"""
app/views/entry_exit.py  -  page 4: entries and exits.

WHAT THIS FILE IS
    The page about the two moments that decide a song's life on the chart: the day it arrives (and at which place) and the day it
    leaves (and from which place). It uses "runs": one run is one unbroken stay of a song on the chart (a song that leaves and
    comes back has two runs).

WHY WE NEED IT
    Arrival place and exit place tell a manager whether songs are launched high and fade away slowly, or arrive low and climb, or
    are removed from the top. That is the shape of a song's life.

WHAT HAPPENS WHEN IT RUNS
    Runs are built from the selected rows. A run counts as "complete" only when both its first and its last day were seen (a run
    that was already on the chart on the first day of the file, or is still there on the last day, is cut off and is left out of the
    questions that need both ends). The place-range filter therefore shortens runs; the page says how many runs are complete.

EXPECTED OUTCOME
    Where songs arrive, where they leave from, how long they stay, and the latest arrivals and exits as tables.

MOTIVE
    Describe how songs enter and exit, without guessing about why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, download_csv, kpi_cards, page_header, plot, section, stop_if_empty, table
from app.ui.state import current
from src.dash import movement as V
from src.dash import texts as T


def _place_counts(positions: pd.Series, kinds: "pd.Series | None" = None) -> pd.DataFrame:
    """Number of runs at every place from 1 to 50 (one column per kind when `kinds` is given)."""
    index = pd.RangeIndex(1, 51, name="place")
    if kinds is None:
        return positions.value_counts().reindex(index, fill_value=0).rename("runs").reset_index()
    table_ = pd.crosstab(positions, kinds).reindex(index, fill_value=0)
    return table_.reset_index()


def render() -> None:
    ctx = current()
    page_header("Playlist", "Entries and exits",
                "Where songs arrive on the chart, where they leave from and how long they stay in between.", ctx.chips)
    stop_if_empty(ctx.rows)
    runs = V.runs_table(ctx.rows, ctx.bounds.first_snapshot, ctx.bounds.last_snapshot)
    stop_if_empty(runs)
    done = runs[runs["complete"]]
    arrived = runs[runs["began_here"]]
    left = runs[runs["ended_here"]]

    kpi_cards(columns=6, items=[
        {"label": "Chart runs", "value": T.n(len(runs)), "tone": "blue", "sub": f"{T.n(len(done))} complete, {T.n(len(runs) - len(done))} cut off by the dates or the data"},
        {"label": "Typical stay", "value": T.num(done["days"].median(), 0) if len(done) else T.DASH, "unit": "days (median)", "tone": "aqua",
         "sub": f"of complete runs; the longest lasted {T.n(done['days'].max()) if len(done) else T.DASH} days"},
        {"label": "Typical arrival place", "value": T.num(arrived["entry_position"].median(), 0) if len(arrived) else T.DASH, "tone": "violet",
         "sub": f"{T.pct((arrived['entry_position'] <= 10).mean(), 0) if len(arrived) else T.DASH} of arrivals are in the top 10"},
        {"label": "Typical exit place", "value": T.num(left["exit_position"].median(), 0) if len(left) else T.DASH, "tone": "red",
         "sub": f"{T.pct((left['exit_position'] > 40).mean(), 0) if len(left) else T.DASH} of exits are from places 41 to 50"},
        {"label": "Returning runs", "value": T.pct((arrived["entry_kind"] == "Re-entry").mean(), 0) if len(arrived) else T.DASH, "tone": "yellow",
         "sub": "of the arrivals are songs that had left the chart before"},
        {"label": "Left from the top 20", "value": T.pct((left["exit_position"] <= 20).mean(), 0) if len(left) else T.DASH, "tone": "orange",
         "sub": "of the exits (a song removed while still high did not fade)"},
    ])

    a, b = st.columns(2)
    with a:
        counts = _place_counts(arrived["entry_position"], arrived["entry_kind"].replace({"New entry": "New songs", "Re-entry": "Returning songs"}))
        series = [c for c in ("New songs", "Returning songs") if c in counts.columns]
        fig = ch.vbar(counts, "place", series, "Where runs begin", barmode="stack", colors={"New songs": theme.BLUE, "Returning songs": theme.VIOLET},
                      x_title="Chart place of the first day of the run", y_title="Runs", height=360)
        plot(fig, key="ee_arrive")
    with b:
        counts = _place_counts(left["exit_position"])
        fig = ch.vbar(counts, "place", "runs", "Where runs end (the last day on the chart)", colors={"runs": theme.RED}, x_title="Chart place of the last day of the run",
                      y_title="Runs", height=360)
        plot(fig, key="ee_exit")
    if len(arrived):
        mode_in = V.place_band(arrived["entry_position"]).value_counts().idxmax()
        mode_out = V.place_band(left["exit_position"]).value_counts().idxmax() if len(left) else T.DASH
        callout(f"Most runs begin at places **{mode_in}** and most end at places **{mode_out}**. "
                f"**{T.pct((arrived['entry_position'] <= 10).mean(), 0)}** of the runs begin in the top 10, while **{T.pct((left['exit_position'] > 40).mean(), 0) if len(left) else T.DASH}** "
                "end at places 41 to 50. This describes where songs enter and leave the chart; it does not explain why.")

    section("From arrival to exit", "Complete runs only: both the first and the last day were seen.")
    value = st.radio("The grid shows", ["Number of runs", "Median days on the chart"], horizontal=True, key="ee_value")
    c1, c2 = st.columns([1, 1])
    with c1:
        matrix = V.entry_exit_matrix(runs, "count" if value == "Number of runs" else "median_days")
        flat = np.sort(matrix.to_numpy(dtype=float).ravel()[~np.isnan(matrix.to_numpy(dtype=float).ravel())])
        cap = float(flat[-2]) if value == "Number of runs" and len(flat) > 1 and flat[-2] > 0 else None       # one huge cell would wash out the others
        matrix.index = [f"arrived {i}" for i in matrix.index]
        matrix.columns = [f"left {c}" for c in matrix.columns]
        fig = ch.heat(matrix, "Runs by arrival place (rows) and exit place (columns)", fmt=",.0f", height=380, zmin=0, zmax=cap,
                      colorbar_title="Runs" if value == "Number of runs" else "Days", hover="%{y}, %{x}<br>%{z:,.0f}<extra></extra>")
        plot(fig, key="ee_matrix")
    with c2:
        done_view = done.assign(arrival=V.place_band(done["entry_position"]).to_numpy())
        fig = ch.box_chart(done_view, "days", "arrival", V.BAND_LABELS, theme.BAND_COLORS, "How long songs stay, by the place they arrived at", height=380,
                           y_title="Days on the chart (log scale)")
        ch.log_axis(fig)
        fig.update_xaxes(title_text="Arrival place")
        plot(fig, key="ee_box")
    if cap is not None:
        st.caption("The darkest colour is capped at the second-largest cell so that the smaller cells can still be told apart; the numbers in the cells are exact.")

    d1, d2 = st.columns(2)
    with d1:
        sc = done.assign(kind=done["entry_kind"].replace({"New entry": "New songs", "Re-entry": "Returning songs"}))
        fig = ch.scatter_chart(sc, "entry_position", "days", color="kind", title="Arrival place and length of the stay", colors={"New songs": theme.BLUE, "Returning songs": theme.VIOLET},
                               x_title="Chart place on the first day", y_title="Days on the chart (log scale)", hover_cols=["track_label", "exit_position"],
                               hover_names={"track_label": "Song", "exit_position": "Left from place"}, opacity=0.55, fixed_size=7, height=380)
        ch.log_axis(fig)
        plot(fig, key="ee_scatter")
    with d2:
        shape = done.assign(places=(done["entry_position"] - done["best_position"]).clip(lower=0))
        fig = ch.histogram_chart(shape, "places", None, "How far songs climb after arriving", bin_size=2, start=0,
                                 x_title="Places gained between the first day and the best day of the run", y_title="Runs", height=380)
        plot(fig, key="ee_climb")

    section("The latest arrivals and exits", "The most recent first. An exit is the last chart day of a run.")
    n_show = st.slider("How many to list", 5, 60, 15, key="ee_n")
    t1, t2 = st.tabs(["Latest arrivals", "Latest exits"])
    arrivals = V.entries_table(ctx.rows).sort_values(["date", "position"], ascending=[False, True]).head(n_show)
    exits = V.exits_table(ctx.rows)
    exits = exits[exits["date"] < pd.Timestamp(ctx.bounds.last_date)].sort_values(["date", "position"], ascending=[False, True]).head(n_show)
    with t1:
        shown = pd.DataFrame({"Date": arrivals["date"].dt.date, "Song": arrivals["track_label"], "Arrived at place": arrivals["position"],
                              "How": arrivals["movement"].map({"New entry": "new song", "Re-entry": "returning"})})
        table(shown, column_config={"Date": st.column_config.DateColumn("Date", format="ddd D MMM YYYY")}, height=min(430, 38 * len(shown) + 40))
        download_csv(shown, "latest_arrivals.csv", key="ee_csv_a")
    with t2:
        shown = pd.DataFrame({"Last day": exits["date"].dt.date, "Song": exits["track_label"], "Left from place": exits["position"]})
        table(shown, column_config={"Last day": st.column_config.DateColumn("Last day", format="ddd D MMM YYYY")}, height=min(430, 38 * len(shown) + 40))
        download_csv(shown, "latest_exits.csv", key="ee_csv_b")
