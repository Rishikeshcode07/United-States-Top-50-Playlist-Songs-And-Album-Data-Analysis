"""
app/views/longevity.py  -  page 15: longevity and survival (the Phase 6B method, run live).

WHAT THIS FILE IS
    The page that answers "how long does a new song stay on the chart?" for any group of songs. It draws a survival curve for each
    group (with a 95% band), the chance that a given chart day is the song's last, and a table of the numbers. Below them it shows
    the formal Phase 6B results for the whole file: where on the chart songs leave from, which facts go with leaving sooner, and
    how much longer a song that has already lasted is likely to stay.

WHY WE NEED IT
    Longevity is the business question behind the whole project: a playlist manager wants to know which new songs will still be
    there next week. A plain "average days on chart" is unfair to songs that have not left yet; the survival method counts them
    correctly (as "still going"), so the numbers on this page can be trusted for recent songs too.

WHAT HAPPENS WHEN IT RUNS
    The 894 new songs of reports/p6b_stays.csv (songs whose arrival was seen) are filtered by the sidebar: the dates select songs by
    their ARRIVAL date, the place range by their DEBUT place, and the other filters as everywhere else. The curves, bands and the
    log-rank test come from src/dash/survival_live.py, the same estimators as the notebook, so with nothing filtered the curves equal
    the saved Phase 6B curves (the page checks this and says so). The estimate is cached, so moving the horizon slider is instant.

EXPECTED OUTCOME
    All new songs: 63% still there after 3 chart days, 43% after 7, 19% after 30, median stay 6 chart days. Songs that debut in the
    top 10 last much longer than songs that debut lower; explicit content and collaboration make no clear difference.

MOTIVE
    Ask the longevity question again for any group, with the same method and the same honesty about uncertainty.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import badge, callout, download_csv, group_table, kpi_cards, page_header, plot, section, stop_if_empty, table
from app.ui.state import current, report, require_reports
from src.dash import survival_live as SL
from src.dash import texts as T

MAX_AGE = 120
GROUP_COLORS = {
    "none": {"All new songs": theme.BLUE},
    "debut": theme.BAND_COLORS,
    "release": theme.ALBUM_COLORS,
    "explicit": theme.EXPLICIT_COLORS,
    "collab": theme.COLLAB_COLORS,
    "same_day": {"1 song (alone)": theme.BLUE, "2 to 4 songs": theme.ORANGE, "5 or more songs": theme.VIOLET},
}
# the name of each grouping in reports/p6b_survival_curves.csv (column "split")
SAVED_SPLIT = {"none": "All songs", "debut": "Debut place", "release": "Release type", "explicit": "Explicit content",
               "collab": "Collaboration", "same_day": "Songs by the artist arriving the same day"}
READING_COLORS = {"Leaves sooner": theme.RED, "Stays longer": theme.BLUE, "No clear difference": "#8f8e84"}
HOVER_PLAIN = "%{fullData.name}<br>%{x}: %{y:.1%}<extra></extra>"


@st.cache_data(show_spinner="Estimating the survival curves", max_entries=24)
def _estimate(stays: pd.DataFrame, grouping: str, min_songs: int) -> dict:
    """Curves with artist-bootstrap bands, summary table, log-rank test and hazard table of the selected stays."""
    result = SL.live_survival(stays, grouping, max_age=MAX_AGE, n_boot=300, seed=0, min_songs=min_songs)
    result["hazard"] = SL.live_hazard(stays, grouping, min_songs=min_songs)
    return result


def _window_label(window) -> str:
    text = str(window)
    return f"Day {text}" if text.isdigit() or text.endswith("+") else f"Days {text}"


def _notebook_gap(curves: pd.DataFrame, grouping: str) -> "float | None":
    """The largest difference between the live curves and the curves saved by notebook 06b (None if they cannot be compared)."""
    saved = report("p6b_survival_curves")
    if saved is None or curves.empty:
        return None
    merged = curves.merge(saved[saved["split"] == SAVED_SPLIT[grouping]], on=["group", "age"], suffixes=("", "_saved"))
    return None if merged.empty else float((merged["survival"] - merged["survival_saved"]).abs().max())


def _reading(summary: pd.DataFrame, logrank, grouping: str, n_songs: int) -> str:
    """The note under the curves, written from the numbers on the screen."""
    usable = summary[(summary["note"] == "") & summary["median_stay"].notna()]
    if usable.empty:
        return "No group has enough songs for a curve."
    if grouping == "none" or len(usable) < 2:
        row = usable.iloc[0]
        return (f"Half of these new songs were gone within **{T.num(row['median_stay'], 0)} chart days**. "
                f"**{T.pct(row['s7'], 0)}** were still there after a week and **{T.pct(row['s30'], 0)}** after 30 days.")
    ranked = usable.sort_values("restricted_mean_30", ascending=False)
    best, worst = ranked.iloc[0], ranked.iloc[-1]
    text = (f"Measured as the average number of chart days in a song's first 30 days, **{best['group']}** songs stay longest "
            f"(**{T.num(best['restricted_mean_30'], 1)}** of 30 days; median stay {T.num(best['median_stay'], 0)} days) and **{worst['group']}** songs "
            f"stay the shortest (**{T.num(worst['restricted_mean_30'], 1)}** of 30 days; median {T.num(worst['median_stay'], 0)} days). ")
    if logrank:
        p = logrank["p_artist"]
        verdict = "The curves differ by more than chance would explain." if pd.notna(p) and p < 0.05 else "The curves do not differ clearly."
        text += (f"{verdict} The artist-aware log-rank test gives **{T.p_text(p)}**; the version that treats every song as independent gives "
                 f"{T.p_text(logrank['p'])}, which is too eager, because the songs of one artist arrive and leave together "
                 f"({T.n(logrank['n_artists'])} artists carry the evidence, not {T.n(n_songs)} songs). "
                 "This live test is not corrected for the number of groupings you try; the corrected tests are in the Phase 6B panel below.")
    return text


def _hazard_figure(hazard: pd.DataFrame, grouping: str, log_scale: bool):
    frame = hazard.assign(label=hazard["window"].map(_window_label))
    labels = list(dict.fromkeys(frame["label"].tolist()))
    order = [g for g in SL.GROUP_ORDER[grouping] if g in set(frame["group"])]
    fig = ch.line_chart(frame, "label", "hazard", group="group", order=order, colors=GROUP_COLORS[grouping], markers=True, height=410,
                        y_title="Chance that the day is the song's last", hover=HOVER_PLAIN)
    fig.update_xaxes(categoryorder="array", categoryarray=labels, title_text="How long the song has been on the chart")
    if log_scale:
        ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
        ch.log_axis(fig, ticks=ticks)
        fig.update_yaxes(ticktext=[f"{t * 100:g}%" for t in ticks])
    else:
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig


def _summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Group": summary["group"], "Songs": summary["songs"], "Artists": summary["artists"], "Left the chart": summary["exits_seen"],
        "Still on the chart": summary["still_on_chart"], "Median stay (days)": summary["median_stay"],
        "Median, lower end": summary["median_low"], "Median, upper end": summary["median_high"],
        "Still there after 1 day (%)": summary["s1"] * 100, "after 3 days (%)": summary["s3"] * 100, "after 7 days (%)": summary["s7"] * 100,
        "after 14 days (%)": summary["s14"] * 100, "after 30 days (%)": summary["s30"] * 100, "after 60 days (%)": summary["s60"] * 100,
        "Chart days in the first 30": summary["restricted_mean_30"], "Note": summary["note"]})


def render() -> None:
    ctx = current()
    page_header("Research", "Longevity and survival",
                "How long new songs stay on the chart. The Phase 6B method is run again, live, on the songs you select.", ctx.chips)
    saved = require_reports("p6b_stays")
    all_stays = SL.prepare_stays(saved["p6b_stays"], ctx.data.dim_track)
    chosen = SL.filter_stays(all_stays, ctx.state, ctx.data.bridge)
    st.caption(f"{T.n(len(all_stays))} new songs are followed here (the songs already on the chart on the first day are left out because their arrival was not seen). "
               "On this page the sidebar dates select songs by their arrival date and the place range selects them by the place they debuted at; "
               "every song is followed to the end of the file, whatever the end date.")
    stop_if_empty(chosen, "No new song matches the filters. The dates select songs by arrival date, the place range by debut place: widen them or press Reset filters.")

    names = SL.GROUPINGS
    c1, c2, c3, c4 = st.columns([2.4, 2.2, 1.3, 1.1], vertical_alignment="bottom")
    shown_name = c1.selectbox("Compare songs by", list(names.values()), index=1, key="lg_grouping")
    grouping = next(key for key, text in names.items() if text == shown_name)
    horizon = c2.slider("Chart days shown", 14, MAX_AGE, 60, key="lg_horizon")
    min_songs = int(c3.number_input("Smallest group", 5, 50, 10, step=5, key="lg_min", help="Groups with fewer songs get no curve: a curve of a handful of songs is not worth drawing."))
    bands = c4.toggle("95% bands", value=True, key="lg_bands")

    overall = _estimate(chosen, "none", min_songs)
    if overall["curves"].empty:
        callout(f"Only {T.n(len(chosen))} new songs match, which is fewer than the smallest group ({min_songs}). Widen the filters or lower the smallest group.",
                title="Too few songs for a curve", tone="warn")
        st.stop()
    result = _estimate(chosen, grouping, min_songs)
    top = overall["summary"].iloc[0]
    kpi_cards(columns=6, items=[
        {"label": "New songs", "value": T.n(top["songs"]), "tone": "blue", "sub": f"from {T.n(top['artists'])} artists"},
        {"label": "Median stay", "value": T.num(top["median_stay"], 0), "unit": "chart days", "tone": "aqua",
         "sub": f"95% band {T.num(top['median_low'], 0)} to {T.num(top['median_high'], 0)} days", "help": "Half of the songs left within this many chart days (counted from the arrival day as day 1)."},
        {"label": "Still there after 3 days", "value": T.pct(top["s3"], 0), "tone": "orange", "sub": "stayed on the chart longer than 3 chart days"},
        {"label": "Still there after 7 days", "value": T.pct(top["s7"], 0), "tone": "yellow", "sub": "stayed on the chart longer than a week"},
        {"label": "Still there after 30 days", "value": T.pct(top["s30"], 0), "tone": "green", "sub": "stayed on the chart longer than 30 chart days"},
        {"label": "Not finished yet", "value": T.n(top["still_on_chart"]), "tone": "violet", "sub": "stays whose end the file does not show",
         "help": "The song is still on the chart, or has been away for under a week when the file ends. The method counts it as still going, up to its last day, instead of treating it as gone."},
    ])

    # ---- survival curves -------------------------------------------------------------------------------------------
    section("Survival curves", "The share of new songs still on the chart, chart day by chart day. A curve that stays high belongs to a group whose songs stay. "
                               "Where a curve crosses the dotted 50% line is the median stay.")
    left, right = st.columns([3.1, 1.2])
    with left:
        plot(ch.survival_chart(result["curves"], order=SL.GROUP_ORDER[grouping], colors=GROUP_COLORS[grouping], max_age=int(horizon), bands=bands, height=470), key="lg_curves")
    with right:
        summary = result["summary"]
        palette = theme.colors_for(list(summary["group"]), GROUP_COLORS[grouping])
        group_table(["Group", "Median stay", "At day 30"],
                    [(palette[row["group"]], row["group"],
                      row["note"] if row["note"] else [f"{T.num(row['median_stay'], 0)} days", T.pct(row["s30"], 0)]) for _, row in summary.iterrows()])
        gap = _notebook_gap(result["curves"], grouping) if len(chosen) == len(all_stays) else None
        if gap is not None:
            st.markdown(badge("Matches Phase 6B", "pass") + f"<br><span style='font-size:0.82rem;color:#52514e'>With all new songs selected, these curves equal the curves saved by "
                        f"notebook 06b (largest difference {gap:.4f}).</span>", unsafe_allow_html=True)
    callout(_reading(summary, result["logrank"], grouping, len(chosen)))

    # ---- hazard ------------------------------------------------------------------------------------------------------
    section("When songs leave", "The chance that a chart day is the song's last day, by how long the song has already been on the chart. Lower means safer.")
    hazard = result["hazard"]
    left, right = st.columns([3.1, 1.2])
    with right:
        log_scale = st.toggle("Log scale", value=True, key="lg_log", help="A log scale makes the small chances of the later days readable next to the big chances of the first days.")
    with left:
        if hazard.empty:
            callout("No group has enough songs for this chart.", title="Too few songs", tone="warn")
        else:
            plot(_hazard_figure(hazard, grouping, log_scale), key="lg_hazard")
    with right:
        base = overall["hazard"]
        first = base[base["window"].astype(str) == "1"]
        late = base[base["from_day"] >= 31]
        if not first.empty and late["chart_days"].sum() > 0:
            callout(f"Of the new songs that reach their first chart day, **{T.pct(first['hazard'].iloc[0], 0)}** leave on it. "
                    f"From day 31 on, the chance that a given day is the last is only **{T.pct(late['exits'].sum() / late['chart_days'].sum(), 1)}**. "
                    "A song that has lasted a while keeps lasting.", title="Reading")
    with st.expander("The numbers behind the curves and the chances of leaving"):
        shown = _summary_table(result["summary"])
        table(shown, column_config={c: st.column_config.NumberColumn(format="%.1f") for c in shown.columns if c.endswith("(%)") or c in ("Chart days in the first 30", "Median stay (days)", "Median, lower end", "Median, upper end")},
              height=min(360, 38 * len(shown) + 40))
        download_csv(shown, f"survival_by_{grouping}.csv", key="lg_csv")
        if not hazard.empty:
            gone = hazard.assign(window=hazard["window"].map(_window_label))[["group", "window", "songs_reaching", "exits", "chart_days", "hazard"]]
            gone = gone.rename(columns={"group": "Group", "window": "Chart days", "songs_reaching": "Songs reaching the window", "exits": "Left the chart",
                                        "chart_days": "Chart days lived", "hazard": "Chance a day is the last"})
            table(gone, column_config={"Chance a day is the last": st.column_config.NumberColumn(format="%.3f")}, height=min(360, 38 * len(gone) + 40))
            download_csv(gone, f"chance_of_leaving_by_{grouping}.csv", label="Download the chances of leaving (CSV)", key="lg_csv2")

    # ---- the formal results of Phase 6B ----------------------------------------------------------------------------
    section("What Phase 6B found on the whole file", "These formal results do not change with the sidebar. They use artist-aware intervals and a correction for the number of comparisons.")
    where, ratios, further, compared = st.tabs(["Where songs leave from", "What goes with leaving sooner", "Songs that have already lasted", "Group comparison"])
    with where:
        danger = report("p6b_danger_zone")
        if danger is None:
            st.caption("Run notebook 06b_survival_analysis to add this panel.")
        else:
            frame = danger.assign(places=danger["places"].astype(str).str.replace("-", " to ", regex=False))
            a, b = st.columns(2)
            with a:
                fig = ch.interval_bars(frame, "places", "chance_last_day", "low", "high", title="Chance that a chart day is the song's last, by place", fmt=".1%",
                                       y_title="Chance of leaving that day", x_title="Places on the chart", height=380)
                fig.update_yaxes(tickformat=".0%")
                plot(fig, key="lg_danger")
            with b:
                fig = ch.vbar(frame, "places", ["share_of_chart_days", "share_of_exits"], "Share of all chart days and share of all exits, by place",
                              names={"share_of_chart_days": "Share of chart days", "share_of_exits": "Share of exits"},
                              colors={"share_of_chart_days": theme.BLUE, "share_of_exits": theme.RED}, y_title="Share of the total", fmt=".0%", height=380)
                fig.update_yaxes(tickformat=".0%")
                plot(fig, key="lg_danger2")
            low_places = danger.tail(2)
            callout(f"A song in places 1 to 5 has a **{T.pct(danger['chance_last_day'].iloc[0], 1)}** chance of leaving that day; in places 46 to 50 the chance is "
                    f"**{T.pct(danger['chance_last_day'].iloc[-1], 0)}**. Places 41 to 50 hold **{T.pct(low_places['share_of_chart_days'].sum(), 0)}** of the chart days of new songs "
                    f"but **{T.pct(low_places['share_of_exits'].sum(), 0)}** of their exits: songs leave from the bottom of the chart, not from the top.")
    with ratios:
        hr = report("p6b_hazard_ratios")
        if hr is None:
            st.caption("Run notebook 06b_survival_analysis to add this panel.")
        else:
            hr = hr.assign(reading=hr.apply(lambda r: "Leaves sooner" if r["low"] > 1 else ("Stays longer" if r["high"] < 1 else "No clear difference"), axis=1))
            plot(ch.forest_chart(hr, "input", "hazard_ratio", "low", "high", title="Hazard ratio of leaving on a given chart day (1 = no difference)", color_col="reading",
                                 colors=READING_COLORS, note="unit", x_title="Hazard ratio (log scale): to the right of 1, songs leave sooner; to the left, they stay longer"), key="lg_forest")
            clear = hr[hr["reading"] != "No clear difference"]
            strongest = hr.sort_values("hazard_ratio", ascending=False).iloc[0]
            callout(f"A hazard ratio of 2 means that on any chart day the song is twice as likely to be on its last day. The strongest signal is **{strongest['input'].lower()}** "
                    f"(**{T.num(strongest['hazard_ratio'], 1)}** {strongest['unit']}). "
                    f"{T.plural(len(clear), 'input')} of {len(hr)} have an interval that stays clear of 1; the rest (release type, explicit content, collaboration and song length among them) "
                    "show no clear difference once the debut place and the album drop are allowed for. These are associations in this file, not causes.")
            hr_table = hr[["input", "unit", "hazard_ratio", "low", "high", "block_p_holm", "reading"]].rename(columns={
                "input": "Input", "unit": "What one step means", "hazard_ratio": "Hazard ratio", "low": "Lower end", "high": "Upper end",
                "block_p_holm": "Adjusted p-value (whole block)", "reading": "Reading"})
            with st.expander("The numbers behind this chart"):
                table(hr_table, column_config={"Hazard ratio": st.column_config.NumberColumn(format="%.2f"), "Lower end": st.column_config.NumberColumn(format="%.2f"),
                                               "Upper end": st.column_config.NumberColumn(format="%.2f"), "Adjusted p-value (whole block)": st.column_config.NumberColumn(format="%.4f")},
                      height=min(420, 38 * len(hr_table) + 40))
                download_csv(hr_table, "hazard_ratios_phase_6b.csv", key="lg_csv3")
    with further:
        stay = report("p6b_further_stay")
        if stay is None:
            st.caption("Run notebook 06b_survival_analysis to add this panel.")
        else:
            frame = stay.assign(after=stay["age"].map(lambda a: "On arrival" if a == 0 else f"After {int(a)} days"))
            a, b = st.columns([3, 2])
            with a:
                plot(ch.interval_bars(frame, "after", "median_further_stay", "low", "high", title="How many more chart days half of the songs that got this far go on to stay", fmt=".0f",
                                      y_title="Median further stay (chart days)", x_title="Chart days already lived", color=theme.AQUA, height=380), key="lg_further")
            with b:
                band = [f"{T.num(m, 0)}  ({T.num(lo, 0)} to {T.num(hi, 0)})" for m, lo, hi in zip(frame["median_further_stay"], frame["low"], frame["high"])]
                shown = pd.DataFrame({"Lived so far": frame["after"], "Still there (%)": frame["still_on_chart"] * 100, "Median further stay": band})
                table(shown, column_config={"Still there (%)": st.column_config.NumberColumn(format="%.0f")}, height=38 * len(shown) + 40)
                st.caption("Further stay in chart days, with its 95% band in brackets. \"Still there\" is the share of all new songs that lasted this long.")
            row30 = stay[stay["age"] == 30]
            if not row30.empty:
                callout(f"A new song that is still on the chart after 30 days has a median further stay of **{T.num(row30['median_further_stay'].iloc[0], 0)} chart days**, "
                        f"against **{T.num(stay['median_further_stay'].iloc[0], 0)}** for a song on its arrival day: the longer a song has lasted, the longer it tends to keep lasting.")
    with compared:
        group = report("p6b_group_comparison")
        if group is None:
            st.caption("Run notebook 06b_survival_analysis to add this panel.")
        else:
            export = pd.DataFrame({
                "Grouping": group["split"], "Group": group["group"], "Songs": group["songs"], "Artists": group["artists"], "Median stay (days)": group["median_stay"],
                "Still there after 7 days (%)": group["share_left_after_7"] * 100, "Still there after 30 days (%)": group["share_left_after_30"] * 100,
                "Compared with": group["reference_group"].fillna("reference group"),
                "Difference in chart days (first 30)": group["chart_days_of_first_30_difference"],
                "Lower end": group["difference_low"], "Upper end": group["difference_high"],
                "Adjusted p-value (artist-aware)": group["log_rank_p_artist_aware_holm"]})
            # on the screen the interval sits in brackets next to the difference, so that the table fits the page without sideways scrolling
            difference = ["-" if pd.isna(d) else f"{T.signed(d, 1)}  ({T.num(lo, 1)} to {T.num(hi, 1)})".replace("-", "−")
                          for d, lo, hi in zip(export["Difference in chart days (first 30)"], export["Lower end"], export["Upper end"])]
            shown = pd.DataFrame({
                "Grouping": export["Grouping"].replace({"Songs by the artist arriving the same day": "Same-day songs"}), "Group": export["Group"], "Songs": export["Songs"],
                "Median (days)": export["Median stay (days)"], "After 7 days (%)": export["Still there after 7 days (%)"],
                "After 30 days (%)": export["Still there after 30 days (%)"], "Compared with": export["Compared with"], "Difference, first 30 days": difference,
                "Adjusted p-value": export["Adjusted p-value (artist-aware)"]})
            table(shown, column_config={"After 7 days (%)": st.column_config.NumberColumn(format="%.0f"), "After 30 days (%)": st.column_config.NumberColumn(format="%.0f"),
                                        "Adjusted p-value": st.column_config.NumberColumn(format="%.4f")}, height=min(560, 38 * len(shown) + 40))
            download_csv(export, "group_comparison_phase_6b.csv", key="lg_csv4")
            st.caption("\"After 7 days\" and \"After 30 days\" are the shares of the group's songs still on the chart then. The adjusted p-value belongs to the whole grouping (it asks whether any of its groups differ). "
                       "The difference is the extra (or missing) number of chart days in a song's first 30 days, compared with the reference group, with its 95% band in brackets.")

    with st.expander("How these curves are made"):
        st.markdown(
            "- A song's **stay** is counted in chart days: day 1 is its arrival day, and the stay ends at the last chart day before the song first stays away for 7 snapshot days or more. Shorter breaks do not end the stay, and the days of a break are not counted.\n"
            "- A stay is **finished** when the file shows it end (the song came back after a long absence, or has been away for a week up to the end of the file). Any other song may still be on the chart, so it is counted as lasting at least this long (this is called censoring). That is how recent songs are kept from being mistaken for short ones.\n"
            "- The curves are **Kaplan-Meier** estimates. The shaded bands are 95% intervals from drawing **whole artists** with replacement 300 times, because the songs of one artist arrive and leave together.\n"
            "- The **log-rank test** compares the curves of two or more groups; the artist-aware version lets every artist count once instead of every song.\n"
            "- Everything here describes how songs behaved in this file. It does not say that a debut place, a release type or anything else causes a longer or shorter stay.")
