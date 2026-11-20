"""
app/views/movement.py  -  page 3: chart movement and turnover.

WHAT THIS FILE IS
    The page about the chart as a whole: how many songs arrive and leave, what share of the 50 places changes hands, which days
    were unusually busy and which weekdays are busy.

WHY WE NEED IT
    A chart can look stable for weeks and then be rebuilt in one day (the data holds a day on which 49 of the 50 songs left).
    Playlist managers need to know how often that happens and when.

WHAT HAPPENS WHEN IT RUNS
    These measures describe the whole Top 50 (not only the songs that match the filters), so only the dates of the sidebar apply
    here. Every number comes from the daily summary table: turnover = (new entries + re-entries) / 50 x 100; a "busy day" is a day
    with more arrivals than the far-out fence of the data (the third quartile plus three times the interquartile range).

EXPECTED OUTCOME
    The busiest days, the weekday pattern and a calendar of the whole period, with the numbers behind them.

MOTIVE
    Separate "the chart is stable" from "the chart is rebuilt now and then".
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, download_csv, kpi_cards, page_header, plot, section, stop_if_empty, table
from app.ui.state import current
from src.dash import metrics as M
from src.dash import movement as V
from src.dash import texts as T

CALENDAR_MEASURES = {
    "Turnover (% of the places that changed hands)": ("turnover_rate_pct", 1.0, "%", ".0f"),
    "New songs (first-time entries)": ("new_entries", 1.0, "", ".0f"),
    "Returning songs (re-entries)": ("re_entries", 1.0, "", ".0f"),
    "Songs that left": ("exits", 1.0, "", ".0f"),
    "Average popularity of the chart": ("avg_popularity", 1.0, "", ".1f"),
    "Explicit share of the chart (%)": ("explicit_share", 100.0, "%", ".0f"),
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def render() -> None:
    ctx = current()
    summary = ctx.summary
    page_header("Playlist", "Movement and turnover",
                "How much of the Top 50 changes hands. These measures describe the whole chart, so only the dates in the sidebar apply here.", ctx.chips[:1])
    stop_if_empty(summary)
    flow = V.chart_flow(summary, 7)

    c1, c2, _ = st.columns([1.2, 2, 2])
    period = (c1.segmented_control("Group the days by", ["Day", "Week", "Month"], default="Week", key="mv_period") or "Week").lower()
    measure = c2.selectbox("The calendar shows", list(CALENDAR_MEASURES), key="mv_measure")

    busiest = flow.loc[flow["turnover_rate_pct"].idxmax()]
    busy_days = int(flow["is_high_turnover_day"].sum())
    kpi_cards(columns=5, items=[
        {"label": "Average daily turnover", "value": T.pct(flow["turnover_rate_pct"].mean(), 1, already_percent=True), "tone": "blue",
         "sub": f"Median day {T.pct(flow['turnover_rate_pct'].median(), 0, already_percent=True)}: most days change little, a few change a lot",
         "help": "(new entries + re-entries) divided by the 50 places, averaged over the days."},
        {"label": "Busiest day", "value": T.day(busiest["date"]), "tone": "red",
         "sub": f"{T.pct(busiest['turnover_rate_pct'], 0, already_percent=True)} of the places changed: {int(busiest['new_entries'])} new and {int(busiest['re_entries'])} returning songs"},
        {"label": "Unusually busy days", "value": T.n(busy_days), "tone": "orange",
         "sub": f"of {T.n(len(flow))} chart days ({T.pct(busy_days / len(flow), 1)})", "help": "Days with more arrivals than the far-out fence of the data (Q3 + 3 x IQR)."},
        {"label": "New songs a day", "value": T.num(flow["new_entries"].mean(), 2), "tone": "aqua", "sub": f"plus {T.num(flow['re_entries'].mean(), 2)} returning songs a day"},
        {"label": "Songs leaving a day", "value": T.num(flow["exits"].mean(), 2), "tone": "violet", "sub": "songs of the day before that are no longer on the chart"},
    ])

    # ---- arrivals and departures, turnover ---------------------------------------------------------------------
    grouped = M.add_period(flow, period).groupby("period").agg(new=("new_entries", "sum"), back=("re_entries", "sum"), out=("exits", "sum"),
                                                              turnover=("turnover_rate_pct", "mean"), days=("date", "size")).reset_index()
    grouped["out_down"] = -grouped["out"]
    left, right = st.columns(2)
    with left:
        fig = ch.vbar(grouped, "period", ["new", "back", "out_down"], f"Songs coming and going, by {period}", barmode="relative",
                      names={"new": "New songs", "back": "Returning songs", "out_down": "Songs that left"},
                      colors={"new": theme.BLUE, "back": theme.VIOLET, "out_down": theme.RED}, y_title="Songs", downward=["out_down"], height=380)
        fig.update_xaxes(tickformat="%d %b %Y" if period != "month" else "%b %Y")
        plot(fig, key="mv_inout")
    with right:
        if period == "day":
            fig = ch.line_chart(flow, "date", "turnover_smooth", title="Turnover, 7-day average", y_title="% of places that changed hands", fmt=".1f", suffix="%",
                                fill=True, height=380)
        else:
            fig = ch.line_chart(grouped, "period", "turnover", title=f"Average daily turnover, by {period}", y_title="% of places that changed hands",
                                fmt=".1f", suffix="%", markers=True, fill=True, height=380)
        plot(fig, key="mv_turnover")

    # ---- calendar -----------------------------------------------------------------------------------------------
    column, factor, suffix, fmt = CALENDAR_MEASURES[measure]
    cells = V.calendar_cells(summary, column)
    cells["value"] = cells["value"] * factor
    section("The calendar", "One cell per chart day, weeks across and weekdays down. Pale cells are quiet days, dark cells busy ones. Days without a snapshot stay empty.")
    fig = ch.calendar_heat(cells, title=measure, fmt=fmt, suffix=suffix, label=measure.split(" (")[0], height=330, colorbar_title="")
    plot(fig, key="mv_calendar")

    # ---- weekday pattern ----------------------------------------------------------------------------------------
    weekday = flow.assign(weekday=flow["date"].dt.day_of_week).groupby("weekday").agg(turnover=("turnover_rate_pct", "mean"), arrivals=("arrivals", "mean"),
                                                                                    exits=("exits", "mean"), days=("date", "size")).reindex(range(7))
    weekday["day"] = WEEKDAYS
    weekday["short"] = [d[:3] for d in WEEKDAYS]
    weekday = weekday.reset_index(drop=True)
    left, right = st.columns([2, 3])
    with left:
        fig = ch.vbar(weekday.dropna(subset=["turnover"]), "short", "turnover", "Average turnover by weekday", colors={"turnover": theme.BLUE}, y_title="% of places that changed hands",
                      fmt=".1f", suffix="%", show_text=True, height=360)
        plot(fig, key="mv_weekday")
    with right:
        top = weekday.dropna(subset=["turnover"])
        hi, lo = top.loc[top["turnover"].idxmax()], top.loc[top["turnover"].idxmin()]
        ratio = hi["turnover"] / lo["turnover"] if lo["turnover"] else float("nan")
        callout(f"**{hi['day']}** is the busiest weekday: on average **{T.pct(hi['turnover'], 1, already_percent=True)}** of the places change hands, "
                f"about **{T.num(ratio, 1)} times** the rate of **{lo['day']}** ({T.pct(lo['turnover'], 1, already_percent=True)}). "
                f"Across the whole period {T.pct(flow['turnover_rate_pct'].mean(), 1, already_percent=True)} of the places change on an average day, "
                f"and {T.plural(busy_days, 'day')} changed far more than usual. This is how the chart behaved; it does not say why.")
        by_day = pd.DataFrame({"Weekday": top["day"], "Chart days": top["days"].astype(int), "Turnover": top["turnover"], "Arrivals a day": top["arrivals"],
                               "Songs leaving a day": top["exits"]})
        table(by_day, column_config={"Turnover": st.column_config.NumberColumn("Turnover (%)", format="%.1f"),
                                     "Arrivals a day": st.column_config.NumberColumn(format="%.2f"), "Songs leaving a day": st.column_config.NumberColumn(format="%.2f")},
              height=38 * len(by_day) + 40)
        st.caption("The weekday of a chart day is the calendar weekday of its snapshot date. A weekday with few days in the chosen dates is less reliable.")

    # ---- busiest days table -------------------------------------------------------------------------------------
    section("The busiest days", "The days with the highest turnover in the chosen dates.")
    count = st.slider("How many days to list", 5, 40, 15, key="mv_top")
    top_days = flow.sort_values(["turnover_rate_pct", "date"], ascending=[False, True]).head(count)
    shown = pd.DataFrame({"Date": top_days["date"].dt.date, "Turnover": top_days["turnover_rate_pct"], "New songs": top_days["new_entries"],
                          "Returning songs": top_days["re_entries"], "Songs that left": top_days["exits"],
                          "Unusually busy": top_days["is_high_turnover_day"].map({True: "yes", False: "no"})})
    table(shown, column_config={"Date": st.column_config.DateColumn("Date", format="ddd D MMM YYYY"),
                                "Turnover": st.column_config.ProgressColumn("Turnover (%)", min_value=0, max_value=100, format="%.0f")}, height=min(420, 38 * len(shown) + 40))
    download_csv(shown, "busiest_days.csv", key="mv_csv")
