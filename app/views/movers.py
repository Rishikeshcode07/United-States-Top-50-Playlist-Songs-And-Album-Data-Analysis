"""
app/views/movers.py  -  page 6: the movers (who climbed, who fell, who arrived, who left).

WHAT THIS FILE IS
    Two ways to look at movement. First, "between two days": pick an end day and a window (1, 3, 7, 14 or 30 chart days) and see which
    songs climbed or fell the most, which arrived and which left. Second, "whole runs": the fastest climbs to a song's peak and the
    slowest fades after it, with thresholds the person can change.

WHY WE NEED IT
    The chart place on one day says little; the change over a week says whether a song is rising. Playlist managers read movement first.

WHAT HAPPENS WHEN IT RUNS
    Two days are compared only for songs in the SAME run (a song that left and came back is counted as an arrival, not as a climb).
    The whole-run view uses complete runs for falls (both ends seen) and runs whose start was seen for climbs; a fall is only counted
    when the song left from place 21 or lower, because a song that vanishes from the top 20 was removed rather than faded.

EXPECTED OUTCOME
    Ranked lists and bars of places gained or lost, plus the distribution of 7-day gains.

MOTIVE
    Show movement with the right comparison (same run, same window) so that "climbed 12 places" always means the same thing.
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

WINDOWS = {"1 day": 1, "3 days": 3, "7 days": 7, "14 days": 14, "30 days": 30}


def _bars(frame: pd.DataFrame, value: str, title: str, color: str, sign: int) -> "ch.go.Figure":
    """Horizontal bars of places gained (sign +1) or lost (sign -1) for the songs in `frame`."""
    if frame.empty:
        return ch.empty_figure("No songs in this group", 260)
    data = frame.assign(label=frame["track_label"].map(lambda t: ch.shorten(t, 46)), shown=sign * frame[value])
    return ch.hbar(data, "label", "shown", title, color=color, fmt=".0f", customdata=["position_start", "position_end"],
                   hover="%{y}<br>place %{customdata[0]} to place %{customdata[1]}<extra></extra>", height=max(240, 30 * len(data) + 100), x_title="Places")


def render() -> None:
    ctx = current()
    page_header("Songs", "Movers",
                "Who climbed, who fell, who arrived and who left between two days, and which songs climbed fastest or faded slowest over their whole run.", ctx.chips)
    stop_if_empty(ctx.rows)
    dates = [d.date() for d in ctx.summary["date"].sort_values()]
    if not dates:
        stop_if_empty(ctx.summary)

    st.markdown("##### Between two days")
    c1, c2, c3 = st.columns([3, 3.3, 1.7], vertical_alignment="bottom")
    if "mv_end" not in st.session_state or not (dates[0] <= st.session_state["mv_end"] <= dates[-1]):
        st.session_state["mv_end"] = dates[-1]
    with c1:
        if len(dates) > 1:
            st.slider("End day", min_value=dates[0], max_value=dates[-1], key="mv_end", format="ddd D MMM YYYY")
        else:
            st.caption(f"Only one chart day in the chosen dates: {T.day(dates[0])}")
    window_name = c2.segmented_control("Compare with the chart", list(WINDOWS), default="7 days", key="mv_window") or "7 days"
    top_n = c3.slider("Songs to list", 5, 25, 10, key="mv_topn")
    result = V.moves_between(ctx.rows, ctx.calendar, pd.Timestamp(st.session_state["mv_end"]), WINDOWS[window_name])

    if result["start_date"] is None:
        callout(f"The chart {window_name} before {T.day(result['end_date']) if result['end_date'] is not None else 'this day'} is not in the data. "
                "Choose a later end day or a shorter window.", title="Nothing to compare", tone="warn")
    else:
        moved, entered, left = result["moved"], result["entered"], result["left"]
        climbed, fell = moved[moved["places_gained"] > 0], moved[moved["places_gained"] < 0]
        kpi_cards(columns=5, items=[
            {"label": "Climbed", "value": T.n(len(climbed)), "tone": "green", "sub": f"of {T.n(len(moved))} songs on the chart on both days"},
            {"label": "Fell", "value": T.n(len(fell)), "tone": "red", "sub": f"{T.n(int((moved['places_gained'] == 0).sum()))} kept the same place"},
            {"label": "Arrived", "value": T.n(len(entered)), "tone": "blue", "sub": "on the chart now, not in the same run before"},
            {"label": "Left", "value": T.n(len(left)), "tone": "violet", "sub": "on the chart before, not now"},
            {"label": "Biggest move", "value": (f"+{int(climbed['places_gained'].max())}" if len(climbed) else T.DASH), "unit": "places", "tone": "ink",
             "sub": climbed.iloc[0]["track_label"] if len(climbed) else "no song climbed"},
        ])
        a, b = st.columns(2)
        with a:
            plot(_bars(climbed.head(top_n), "places_gained", f"Biggest climbers, {T.day(result['start_date'])} to {T.day(result['end_date'])}", theme.GREEN, 1), key="mvr_up")
        with b:
            plot(_bars(fell.sort_values("places_gained").head(top_n), "places_gained", "Biggest falls", theme.RED, -1), key="mvr_down")
        if len(moved):
            callout(f"Between **{T.day(result['start_date'])}** and **{T.day(result['end_date'])}**, **{len(climbed)}** songs climbed and **{len(fell)}** fell. "
                    f"The largest climb was **{int(climbed['places_gained'].max()) if len(climbed) else 0} places** and the largest fall **{int(-fell['places_gained'].min()) if len(fell) else 0} places**. "
                    f"**{len(entered)}** songs arrived and **{len(left)}** left in the same time.")
        t1, t2, t3 = st.tabs(["Arrived", "Left", "All songs on both days"])
        with t1:
            table(pd.DataFrame({"Song": entered["track_label"], "Place now": entered["position"], "Days in this run": entered["days_in_run"]}), height=min(380, 38 * len(entered) + 40)) \
                if len(entered) else st.caption("No song arrived in this window.")
        with t2:
            table(pd.DataFrame({"Song": left["track_label"], "Place before": left["position"], "Days in this run before": left["days_in_run"]}), height=min(380, 38 * len(left) + 40)) \
                if len(left) else st.caption("No song left in this window.")
        with t3:
            full = pd.DataFrame({"Song": moved["track_label"], "Place before": moved["position_start"], "Place now": moved["position_end"], "Places gained": moved["places_gained"],
                                 "Popularity change": moved["popularity_change"].round(1)})
            table(full, height=min(420, 38 * len(full) + 40))
            download_csv(full, "movers_between_two_days.csv", key="mvr_csv")

    # ---- whole runs --------------------------------------------------------------------------------------------
    section("Over a whole run", "Fastest climbs to the peak and slowest fades after it. The thresholds below decide which runs count.")
    runs = V.runs_table(ctx.rows, ctx.bounds.first_snapshot, ctx.bounds.last_snapshot)
    t1, t2, t3 = st.columns(3)
    min_gain = t1.slider("Climb at least this many places to the peak", 5, 40, 10, key="mvr_gain")
    min_after = t2.slider("Fade for at least this many days after the peak", 7, 60, 14, key="mvr_after")
    min_exit = t3.slider("Leave from place this low or lower", 11, 50, 21, key="mvr_exit")
    tables = V.rise_decline_tables(runs, min_gain, min_after, min_exit)
    risers, decliners = tables["risers"], tables["decliners"]
    a, b = st.columns(2)
    with a:
        if risers.empty:
            st.info("No run climbed that far in the chosen dates. Lower the threshold or widen the dates.")
        else:
            data = risers.head(top_n).assign(label=lambda d: d["track_label"].map(lambda t: ch.shorten(t, 46)))
            fig = ch.hbar(data, "label", "rise_per_week", "Fastest climbs to the peak (places per week)", color=theme.GREEN, fmt=".1f", customdata=["entry_position", "best_position", "days_to_peak"],
                          hover="%{y}<br>%{x:.1f} places a week<br>from place %{customdata[0]} to place %{customdata[1]} in %{customdata[2]} days<extra></extra>",
                          height=max(240, 30 * len(data) + 100), x_title="Places gained per week")
            plot(fig, key="mvr_rise")
    with b:
        if decliners.empty:
            st.info("No complete run faded for that long and left from that low a place. Change the thresholds or widen the dates.")
        else:
            data = decliners.head(top_n).assign(label=lambda d: d["track_label"].map(lambda t: ch.shorten(t, 46)))
            fig = ch.hbar(data, "label", "fall_per_week", "Slowest fades after the peak (places lost per week)", color=theme.RED, fmt=".1f", customdata=["best_position", "exit_position", "days_after_peak"],
                          hover="%{y}<br>%{x:.1f} places a week<br>from place %{customdata[0]} to place %{customdata[1]} in %{customdata[2]} days<extra></extra>",
                          height=max(240, 30 * len(data) + 100), x_title="Places lost per week")
            plot(fig, key="mvr_fall")
    st.caption(f"{T.n(len(risers))} runs meet the climb threshold and {T.n(len(decliners))} meet the fade threshold. A small number of places lost per week means a slow, long fade.")

    section("How big are the weekly moves?", "Places gained over seven chart days, for every song-day that was on the chart seven days earlier. Positive = climbed.")
    gains = ctx.rows["rank_gain_7d"].dropna()
    if gains.empty:
        st.caption("No song-day in the selection has a seven-day history.")
    else:
        up, down = float((gains > 0).mean()), float((gains < 0).mean())
        fig = ch.histogram_chart(gains.to_frame("gain"), "gain", None, "Places gained in seven days", bin_size=2, start=-50, end=50, x_title="Places gained (negative = fell)", y_title="Song-days",
                                 height=330)
        fig.add_vline(x=0, line=dict(color=theme.AXIS, dash="dot", width=1.2))
        plot(fig, key="mvr_hist")
        callout(f"In **{T.pct(up, 0)}** of the song-days the song stood higher than seven days earlier and in **{T.pct(down, 0)}** lower. The median weekly change is "
                f"**{T.signed(gains.median(), 0)} places**; the typical song moves little, and the big moves are rare.")
