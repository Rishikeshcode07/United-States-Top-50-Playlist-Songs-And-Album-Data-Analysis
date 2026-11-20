"""
app/views/methods.py  -  page 18: data and methods.

WHAT THIS FILE IS
    The page that says where the numbers come from and how far they can be trusted: the data-quality scorecard and the quality gate of
    the pipeline, what the cleaning changed, the check that shows the file's calendar runs about a year ahead of real dates, the
    songs that were merged because they had two credit strings, a glossary of every measure used on the dashboard, the data dictionary,
    which pages calculate live and which show results saved by a notebook, which result files are present, the run record of the
    pipeline, and the limits of the analysis.

WHY WE NEED IT
    A dashboard is only as honest as its definitions. A reader who sees "Artist Dominance Index 12.3" or "median stay 6 days" must be
    able to find, in one place, what the number counts, what was cleaned before it, and what it does not claim. The page also lets a
    reviewer check that the notebooks have been run and that the data are the data the notebooks described.

WHAT HAPPENS WHEN IT RUNS
    It reads the saved quality files (reports/data_quality_scorecard.csv, quality_gate.csv, cleaning_log.csv, kpi_dictionary.csv,
    data_dictionary.csv, calendar_reality_check.csv, track_identity_decisions.csv, last_run.json) and the list of files in reports/.
    It recalculates nothing, so the sidebar filters do not apply. A file that is missing switches off its own block and names the
    command that creates it (python -m src.etl.run_pipeline).

EXPECTED OUTCOME
    A quality gate of 25 passed tests out of 25, 31 scorecard checks (17 pass, 8 warn and handled, 6 info), a calendar gap of about 367
    days for 20 well-known songs, 35 title-and-credit decisions, a glossary, and all result files of Phases 4 to 6C marked as found.

MOTIVE
    Make every number traceable and every limit visible.
"""
from __future__ import annotations

import platform
from datetime import datetime
from importlib import metadata

import pandas as pd
import streamlit as st

from app import nav
from app.ui import charts as ch
from app.ui import theme
from app.ui.components import Raw, badge, callout, download_csv, esc, facts, html_table, kpi_cards, page_header, plot, section
from app.ui.state import NOTEBOOK_OF, current, facts as saved_facts, report
from src.config import default_paths
from src.dash import texts as T
from src.dash.load import reports_available

STATUS_ORDER = ["PASS", "WARN", "INFO", "FAIL"]
STATUS_COLORS = {"PASS": theme.GREEN, "WARN": theme.YELLOW, "INFO": "#86b6ef", "FAIL": theme.RED}
STATUS_TONES = {"PASS": "pass", "WARN": "warn", "INFO": "info", "FAIL": "fail"}
FIX_COMMAND = "python -m src.etl.run_pipeline"

# the measures of the dashboard: (term, what it means, where it is used)
BRIEF_KPIS = {"Days on Chart", "Average Rank", "Rank Volatility Index", "Popularity Score Trend", "Artist Dominance Index", "Explicit Content Share"}

GLOSSARY = [
    ("Chart day (snapshot)", "One daily snapshot of the Top 50. The file has 555 snapshots, from 18 May 2024 to 27 November 2025. Four calendar days of that span have no snapshot; they are kept as gaps and never filled, and time is counted in snapshots.", "Everywhere"),
    ("Place (position)", "1 is the top of the chart and 50 the bottom. A song that moves 'up' gets a smaller number.", "Everywhere"),
    ("Chart points (chart score)", "51 minus the place: place 1 earns 50 points, place 50 earns 1. A full chart holds 1,275 points a day. A day off the chart is 0 points.", "Artist dominance, archetypes"),
    ("Days on chart", "The number of snapshots on which a song appears. A song that leaves and comes back counts all its days.", "Song pages, release page"),
    ("Average place", "The mean of the song's places over its chart days. Lower is better.", "Song pages"),
    ("Rank Volatility Index", "The standard deviation of the song's daily places (sample standard deviation, so at least two days are needed). Higher means a less stable position.", "Song pages, release page"),
    ("Rank change and 7-day rank gain", "Rank change = yesterday's place minus today's place, so a positive number means the song moved up. The 7-day rank gain compares today with the place 7 snapshots ago (positive = climbed).", "Movers, timeline"),
    ("Popularity score and its trend", "The 0 to 100 popularity score of the source file. Zeros were treated as missing and repaired (cleaning step 2.5). The popularity trend is the 7-day rolling mean inside a chart run.", "Popularity page"),
    ("Artist Dominance Index (ADI)", "An artist's chart points divided by all chart points of the chosen dates, times 100. On the 'first credit' basis every song counts for its first-credited artist, so all shares add up to 100. On the 'every credit' basis a collaboration counts for each credited artist, so shares add up to more than 100 (about 114 on the whole file).", "Artist pages"),
    ("Turnover", "(new entries + re-entries) divided by 50, times 100, for one day: the share of the 50 places that changed hands.", "Movement page"),
    ("New entry, re-entry, exit", "A new entry is the first day a song is ever on the chart; a re-entry is a return after an absence; an exit is a song that was on the previous snapshot and is not on this one.", "Movement, entry and exit"),
    ("Busy day (high-turnover day)", "A day with more arrivals than the far-out fence of the file: the third quartile of daily arrivals plus three times the interquartile range.", "Movement page"),
    ("Explicit share", "The share of song-days (or of songs, where the page says so) flagged explicit. The daily share is the number of explicit songs among the 50 of that day.", "Explicit page"),
    ("HHI, effective number, Gini, top-N share", "HHI = the sum of the squared percentage shares of the artists (0 to 10,000). Effective number = 10,000 divided by HHI: the chart behaves as if that many equal artists shared it. Gini: 0 = everyone equal, 1 = one holder has everything. Top-N share = the share of the N largest artists.", "Concentration page"),
    ("Run (stint)", "An unbroken stretch of consecutive snapshots on the chart. Any absence starts a new run.", "Songs, movers"),
    ("Longevity class", "Flash (1 to 3 days), Short (4 to 14), Medium (15 to 60), Long (61 to 180) or Evergreen (181 or more) by days on the chart.", "Songs, artists"),
    ("Release size band", "The number of tracks on the release: 1 track, 2-3, 4-7, 8-15, 16-40, 41 or more.", "Release page"),
    ("Stay (Phase 6B)", "Counted in chart days: day 1 is the arrival day, and the stay ends at the last chart day before the song first stays away for 7 snapshot days or more. Songs already on the chart on the first day are left out because their arrival was not seen.", "Longevity page"),
    ("Survival curve, censoring, median stay", "The survival curve is the share of new songs still on the chart after a number of chart days. A stay that has not ended when the file ends is censored: it counts as still going, so recent songs are not mistaken for short ones. The median stay is where the curve crosses 50%.", "Longevity page"),
    ("Hazard and hazard ratio", "The hazard is the chance that a given chart day is the song's last, among the songs that reached it. A hazard ratio of 2 means a song is twice as likely to leave on any given chart day as the reference song (Cox proportional-hazards model).", "Longevity, evidence"),
    ("Cliff's delta", "How often a random song of group A beats a random song of group B on a measure, minus how often it loses: from -1 to +1, 0 = no difference. Under 0.147 is called negligible, up to 0.33 small, up to 0.474 medium, beyond that large.", "Explicit, release"),
    ("Artist-aware test", "The songs of one artist are alike (they arrive, stay and leave together), so counting every song as independent makes p-values too small. The artist-aware tests treat whole artists as the unit (cluster-robust standard errors or an artist bootstrap).", "Evidence"),
    ("Holm correction", "Ten hypotheses were tested at once. The Holm correction adjusts the p-values so that the chance of even one false 'supported' claim stays below 5%. 'Supported' needs the corrected p-value below 0.05.", "Evidence"),
    ("Archetype and segment", "A group of songs with a similar first month (archetype), or of artists with a similar footprint (segment), found by k-means in Phase 6C. They describe what happened; they are not boxes to sort a new song into.", "Archetypes page"),
    ("Jaccard score and ARI", "How firmly a group is found again when the artists are re-drawn. Jaccard 1 = exactly the same songs; 0.75 or more is called firm, under 0.60 dissolved. ARI (adjusted Rand index) is the agreement of the whole grouping: 1 = the same, about 0 = chance.", "Archetypes page"),
]

# which pages calculate live and what saved results they read
LINEAGE = {
    "overview": ("Live", "Chart tables of the pipeline; the 'what the analysis found' block reads the Phase 4 findings."),
    "timeline": ("Live", "Chart tables of the pipeline."),
    "movement": ("Live", "Daily summary table of the pipeline."),
    "entry-exit": ("Live", "Chart tables of the pipeline."),
    "rank-trends": ("Live", "Chart tables of the pipeline."),
    "movers": ("Live", "Chart tables of the pipeline."),
    "songs": ("Live", "Chart tables of the pipeline."),
    "song-profile": ("Live", "Chart tables of the pipeline."),
    "artists": ("Live", "Chart tables and the track-artist bridge."),
    "artist-profile": ("Live", "Chart tables and bridge; the artist's segment comes from Phase 6C."),
    "concentration": ("Live", "Chart tables and bridge; the formulas of src/eda/concentration.py."),
    "popularity": ("Live", "Chart tables."),
    "explicit": ("Live and saved", "Chart tables; the formal tests H1 and H10 come from Phase 5."),
    "release": ("Live and saved", "Chart tables; the formal tests H2, H7, H9 and H10 come from Phase 5."),
    "longevity": ("Live and saved", "Phase 6B method run live on reports/p6b_stays.csv; the formal Phase 6B panels are saved results."),
    "archetypes": ("Saved, averaged live", "The groups come from notebook 06c and never change; the filters choose which songs are averaged."),
    "evidence": ("Saved", "Notebooks 04, 05, 06a and 06b."),
    "methods": ("Saved", "The pipeline's quality files and run record."),
}
EXPECTED = ["p4_eda_findings", "p5_hypothesis_register", "p5_robustness", "p6a_model_register", "p6a_stress_tests", "p6b_stays", "p6b_survival_curves", "p6b_danger_zone",
            "p6b_hazard_ratios", "p6b_further_stay", "p6b_group_comparison", "p6b_ladder", "p6b_risk_thirds", "p6c_archetypes", "p6c_song_archetypes", "p6c_first_months",
            "p6c_attributes", "p6c_artist_segments", "p6c_artists", "p6c_k_scan_songs", "p6c_stability", "p6c_stress_tests", "p6c_landmark", "p6c_risk_thirds"]


def _missing(name: str) -> None:
    st.caption(f"reports/{name}.csv is not there. Create it with `{FIX_COMMAND}`.")


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


# ====================================================================================================== tab 1
def _quality_tab(ctx) -> None:
    run = saved_facts("last_run")
    scorecard, gate, cleaning = report("data_quality_scorecard"), report("quality_gate"), report("cleaning_log")
    calendar, identity = report("calendar_reality_check"), report("track_identity_decisions")
    data = ctx.data
    gate_text = f"{run['gate']['passed']} of {run['gate']['total']}" if run.get("gate") else (f"{int((gate['status'] == 'PASS').sum())} of {len(gate)}" if gate is not None else "n/a")
    validation = run.get("validation", {})
    kpi_cards(columns=6, items=[
        {"label": "Rows in the source file", "value": T.n(run.get("source", {}).get("rows", 0)) if run else "n/a", "tone": "blue", "sub": "before cleaning"},
        {"label": "Song-days analysed", "value": T.n(len(data.playlist)), "tone": "aqua", "sub": f"{T.n(ctx.bounds.snapshots)} snapshot days of 50 places"},
        {"label": "Songs", "value": T.n(len(data.dim_track)), "tone": "orange", "sub": "one track per song after the merge of credit strings"},
        {"label": "Artists", "value": T.n(len(data.dim_artist)), "tone": "violet", "sub": "every credited artist"},
        {"label": "Quality gate", "value": gate_text, "tone": "green", "sub": "tests passed before the data were saved"},
        {"label": "Scorecard", "value": f"{validation.get('WARN', 0)} warnings" if validation else "n/a", "tone": "yellow",
         "sub": f"{validation.get('PASS', 0)} pass, {validation.get('INFO', 0)} info, {validation.get('FAIL', 0)} fail"},
    ])

    section("Data-quality scorecard", "31 checks of the raw file, in six dimensions. A warning is a finding with a recorded action, not a failure.")
    if scorecard is None:
        _missing("data_quality_scorecard")
    else:
        counts = pd.crosstab(scorecard["dimension"], scorecard["status"]).reindex(columns=STATUS_ORDER, fill_value=0).reset_index()
        left, right = st.columns([2, 3])
        with left:
            fig = ch.vbar(counts, "dimension", [s for s in STATUS_ORDER if counts[s].sum()], "Checks by dimension and result", barmode="stack", colors=STATUS_COLORS,
                          y_title="Checks", fmt=".0f", height=400, legend="top")
            plot(fig, key="mt_scorecard")
        with right:
            warned = scorecard[scorecard["status"] == "WARN"]
            if len(warned):
                callout(f"**{len(warned)} checks gave a warning.** Each one has a recorded action: for example the day with two snapshots (2025-03-01) keeps only the latest block, the four calendar days without a snapshot "
                        "stay gaps (no rows are invented), and the 29 songs that appeared under two credit strings were merged. No warning needed the data to be changed beyond that.", title="What the warnings were")
            n_fail = int((scorecard["status"] == "FAIL").sum())
            facts([("PASS", "the check found nothing to act on"), ("WARN", "something was found; the action taken is recorded"),
                   ("INFO", "worth knowing; no change was needed"), ("FAIL", f"a rule was broken ({n_fail} in this file)")], title="What the results mean", wide=True)
        status = st.segmented_control("Show", ["All"] + STATUS_ORDER, default="All", key="mt_status") or "All"
        view = scorecard if status == "All" else scorecard[scorecard["status"] == status]
        export = pd.DataFrame({"Dimension": view["dimension"], "Check": view["check"], "Result": view["status"], "Affected": view["affected"], "Of": view["total"], "Unit": view["unit"],
                               "Affected (%)": view["pct_affected"], "Action": view["action"]})
        if view.empty:
            st.caption(f"No check in this file has the result {status}.")
        else:
            html_table(["Dimension", "Check", "Result", "Affected", "Action"],
                       [[row["dimension"], row["check"], Raw(badge(row["status"], STATUS_TONES.get(row["status"], "mute"))),
                         f"{T.n(row['affected'])} of {T.n(row['total'])} {row['unit']}", "No action needed" if row["status"] == "PASS" else row["action"]]
                        for _, row in view.iterrows()], max_height=470, nowrap_cols=(0,))
        download_csv(export, "data_quality_scorecard.csv", key="mt_csv1")

    section("The quality gate", "25 tests that the cleaned tables must pass before they are saved. If one fails, the pipeline stops and nothing is saved.")
    if gate is None:
        _missing("quality_gate")
    else:
        with st.expander(f"Show the {len(gate)} tests ({int((gate['status'] == 'PASS').sum())} passed)"):
            html_table(["ID", "Test", "Result", "Detail"],
                       [[str(row["id"]), row["test"], Raw(badge(str(row["status"]), STATUS_TONES.get(str(row["status"]), "mute"))), "" if pd.isna(row["detail"]) else str(row["detail"])]
                        for _, row in gate.iterrows()], max_height=480)

    section("What the cleaning changed", "Every change to the raw file, in order, with the number of rows it touched. Nothing was deleted without a line here.")
    if cleaning is None:
        _missing("cleaning_log")
    else:
        shown = pd.DataFrame({"Step": cleaning["step"].astype(str), "Rule": cleaning["rule"], "Rows before": cleaning["rows_before"], "Rows after": cleaning["rows_after"],
                              "Rows affected": cleaning["rows_affected"], "Note": cleaning["note"]})
        html_table(list(shown.columns), [[str(r["Step"]), r["Rule"], T.n(r["Rows before"]), T.n(r["Rows after"]), T.n(r["Rows affected"]), "" if pd.isna(r["Note"]) else r["Note"]]
                                         for _, r in shown.iterrows()], numeric_cols=(2, 3, 4))
        download_csv(shown, "cleaning_log.csv", key="mt_csv2")

    section("Is the calendar real?", "For 20 well-known songs the real release date is known. The gap to the first day on the chart tells whether the file's dates match real dates.")
    if calendar is None:
        _missing("calendar_reality_check")
    else:
        frame = calendar.sort_values("gap_days").reset_index(drop=True)
        left, right = st.columns([3, 2])
        with left:
            fig = ch.forest_chart(frame, "song", "gap_days", "gap_days", "gap_days", title="Days from the real release to the first chart day in the file", reference=365, log_x=False,
                                  x_title="Days (the dashed line is one year)", fmt=".0f", height=max(420, 26 * len(frame) + 120))
            fig.update_xaxes(range=[362, 373])
            plot(fig, key="mt_calendar")
        with right:
            median = float(frame["gap_days"].median())
            callout(f"A song that is new on the chart should appear within days or weeks of its release. Here the median gap is **{T.num(median, 0)} days**, almost exactly a year, and it varies by only a few days "
                    "from song to song. The file's calendar therefore seems to run **about one year ahead** of real dates.\n\n"
                    "The dashboard keeps the dates exactly as provided and does not shift them. Every measure is relative (days since arrival, places, shares), so nothing depends on the absolute year; "
                    "the weekday of a date is the weekday recorded in the file.", title="What this means")

    section("One song, two credit strings", "A song can appear under two credits (for example 'Lil Durk' and 'Lil Durk & J. Cole'). The rule: same title, a shared artist, and never on the chart on the same day, means the same song.")
    if identity is None:
        _missing("track_identity_decisions")
    else:
        pick = st.segmented_control("Decision", ["All", "same song", "different song"], default="All", key="mt_identity") or "All"
        view = identity if pick == "All" else identity[identity["verdict"] == pick]
        shown = pd.DataFrame({"Title": view["song"], "Main credit": view["main_credit"], "Other credit": view["other_credit"], "Shared artist": view["shares_an_artist"].map({True: "yes", False: "no"}),
                              "Days together": view["dates_on_chart_together"], "Decision": view["verdict"]})
        html_table(list(shown.columns), [[title, main, other, shared, str(int(days)), Raw(badge(verdict, "info" if verdict == "same song" else "mute"))]
                                         for title, main, other, shared, days, verdict in shown.itertuples(index=False, name=None)], numeric_cols=(4,), max_height=430)
        st.caption(f"{int((identity['verdict'] == 'same song').sum())} pairs were merged into one song; {int((identity['verdict'] == 'different song').sum())} looked alike but were kept apart "
                   "(no shared artist, for example 'Kali' and 'Kaliii'). \"Days together\" counts the days on which both credit strings were on the chart.")


# ====================================================================================================== tab 2
def _definitions_tab() -> None:
    st.caption("The measures below are the ones of the dashboard.")
    section("Glossary", "What every measure on the dashboard counts. Type a word to narrow the list.")
    word = st.columns([2, 3])[0].text_input("Find a term", placeholder="for example volatility, turnover, points", key="mt_find").strip().lower()
    glossary = pd.DataFrame(GLOSSARY, columns=["Measure", "What it means", "Used on"])
    shown = glossary
    if word:
        shown = glossary[glossary.apply(lambda column: column.str.lower().str.contains(word, regex=False)).any(axis=1)]
    st.caption(f"{len(shown)} of {len(glossary)} terms shown.")
    if shown.empty:
        callout("No term contains that word. Try a shorter word, or clear the box.", title="No match", tone="warn")
    else:
        html_table(["Measure", "What it means", "Used on"], [[Raw(f"<b>{esc(r['Measure'])}</b>"), r["What it means"], r["Used on"]] for _, r in shown.iterrows()])
    download_csv(shown, "glossary.csv", key="mt_csv3")

    section("The KPIs", "The six KPIs of the project brief and five measures added in the pipeline: where each is stored, and its formula.")
    kpis = report("kpi_dictionary")
    if kpis is None:
        _missing("kpi_dictionary")
    else:
        shown = kpis.rename(columns={"kpi": "KPI", "where_to_find_it": "Where it is stored", "formula": "Formula", "interpretation": "How to read it"})
        shown.insert(1, "Source", ["Brief" if name in BRIEF_KPIS else "Added" for name in shown["KPI"]])
        html_table(list(shown.columns), [[Raw(f"<b>{esc(row['KPI'])}</b>"), Raw(badge(row["Source"], "info" if row["Source"] == "Brief" else "mute")), row["Where it is stored"], row["Formula"], row["How to read it"]]
                                         for _, row in shown.iterrows()])
        download_csv(shown, "kpi_dictionary.csv", key="mt_csv4")

    section("Data dictionary", "Every column of every table that the pipeline saves: its type, its meaning and an example. Choose a table, or type a word to find a column.")
    dictionary = report("data_dictionary")
    if dictionary is None:
        _missing("data_dictionary")
    else:
        tables = ["All tables"] + list(dict.fromkeys(dictionary["table"].tolist()))
        left, right = st.columns([2, 3])
        pick = left.selectbox("Table", tables, key="mt_table")
        word = right.text_input("Find a column", placeholder="for example popularity, date, rank", key="mt_dict_find").strip().lower()
        view = dictionary if pick == "All tables" else dictionary[dictionary["table"] == pick]
        if word:
            view = view[view.apply(lambda column: column.astype(str).str.lower().str.contains(word, regex=False)).any(axis=1)]
        shown = view.rename(columns={"table": "Table", "column": "Column", "data_type": "Type", "definition": "Meaning", "example": "Example"})
        st.caption(f"{len(shown)} of {len(dictionary)} columns shown.")
        if shown.empty:
            callout("No column matches that word in this table. Try a shorter word, or choose All tables.", title="No match", tone="warn")
        else:
            html_table(list(shown.columns), [[row["Table"], Raw(f"<b>{esc(row['Column'])}</b>"), row["Type"], row["Meaning"], "" if pd.isna(row["Example"]) else str(row["Example"])]
                                             for _, row in shown.iterrows()], max_height=520, nowrap_cols=(0, 2))
        download_csv(shown, "data_dictionary.csv", key="mt_csv5")


# ====================================================================================================== tab 3
def _scope_tab(ctx) -> None:
    section("What this dashboard claims, and what it does not")
    left, right = st.columns(2)
    with left:
        callout(f"It **measures what happened** on one chart: the United States Top 50 playlist, {T.n(ctx.bounds.snapshots)} snapshot days, {T.n(len(ctx.data.dim_track))} songs. Rankings, shares, stays and trends are historical facts about this file. "
                "Where a formal test exists (Phase 5), the page shows it next to the live number.", title="What it claims")
    with right:
        callout("It does **not** predict, and it does **not** recommend. Differences between groups are associations, not causes. The models of Phase 6A are backtests on artists they had not seen. "
                "The archetypes and segments describe patterns; they are not boxes to sort a new song or a new artist into.", title="What it does not claim", tone="warn")
    section("Limits to keep in mind")
    facts(wide=True, pairs=[
        ("One chart, one market", "The file covers a single playlist and a single country. Nothing here says how another playlist behaves."),
        ("The calendar", "The dates run about a year ahead of real dates (see the calendar check). Weekday and month patterns are those recorded in the file."),
        ("Songs already on the chart on day 1", "Their arrival was not seen, so the 50 songs of the first snapshot are left out of every stay and archetype."),
        ("The end of the file", "Songs that arrived in the last 29 days have no complete first month (archetypes), and stays that are not finished are counted as still going."),
        ("Missing days", "Four calendar days have no snapshot. They are kept as gaps; no rows were invented."),
        ("Credits", "A collaboration counts for every credited artist on the 'every credit' basis of the ADI, which is why those shares add up to more than 100."),
        ("Popularity", "Zeros in the popularity score were treated as missing and repaired; the original column is kept and the repair method recorded."),
        ("Songs are not independent", "Songs of one artist are alike. Live p-values on the dashboard treat songs as independent and are therefore too eager; the formal tests allow for artists."),
    ])

    section("Where each page gets its numbers", "Live = recalculated from the cleaned tables for whatever the sidebar selects. Saved = read unchanged from a result file that a notebook wrote.")
    rows = []
    for page in nav.PAGES:
        section_name, url, title = page[0], page[1], page[2]
        kind, source = LINEAGE.get(url, ("", ""))
        rows.append([section_name, Raw(f"<b>{esc(title)}</b>"), kind, source])
    html_table(["Section", "Page", "Numbers are", "Source"], rows)

    section("Result files", "The saved results that the dashboard reads. A missing file switches off its own panel; the notebook named in the last column creates it.")
    present = reports_available(EXPECTED)
    folder = default_paths().reports_dir
    rows = []
    for name in EXPECTED:
        path = folder / f"{name}.csv"
        notebook = NOTEBOOK_OF.get(name.split("_")[0], "the notebooks")
        rows.append({"File": f"reports/{name}.csv", "Found": "yes" if present[name] else "no",
                     "Size (KB)": round(path.stat().st_size / 1024, 1) if path.exists() else None,
                     "Saved on": datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y %H:%M") if path.exists() else "",
                     "Created by": notebook})
    files = pd.DataFrame(rows)
    found = int((files["Found"] == "yes").sum())
    st.markdown(badge(f"{found} of {len(files)} found", "pass" if found == len(files) else "warn"), unsafe_allow_html=True)
    html_table(list(files.columns), [[name, Raw(badge(is_found, "pass" if is_found == "yes" else "fail")), "" if pd.isna(size) else f"{size:,.1f}", saved, made]
                                     for name, is_found, size, saved, made in files.itertuples(index=False, name=None)], numeric_cols=(2,), max_height=440)
    if found < len(files):
        absent = sorted({row["Created by"] for _, row in files.iterrows() if row["Found"] == "no"})
        callout(f"{len(files) - found} result files are missing. Open {' and '.join(absent)}, choose Run All, and refresh this page. The pages that do not need them keep working.", title="Some results are missing", tone="warn")

    section("This run", "The record of the last pipeline run and the software this dashboard is running on.")
    run = saved_facts("last_run")
    left, right = st.columns(2)
    with left:
        if run:
            source = run.get("source", {})
            facts([("Pipeline run", f"{run.get('status', '')}, version {run.get('pipeline_version', '')}"), ("Started (UTC)", str(run.get("started_at_utc", ""))),
                   ("Finished (UTC)", str(run.get("finished_at_utc", ""))), ("Source file", f"{source.get('name', '')}, {T.n(source.get('rows', 0))} rows, {T.n(source.get('size_bytes', 0) / 1024)} KB"),
                   ("Fingerprint (SHA-256)", str(source.get("sha256", ""))[:24] + "..."), ("Seconds in total", str(run.get("timings_seconds", {}).get("total", "")))])
        else:
            _missing("last_run")
            st.caption("reports/last_run.json is written by the pipeline. Run it with `python -m src.etl.run_pipeline`.")
    with right:
        facts([("Python", platform.python_version()), ("Streamlit", _version("streamlit")), ("pandas", _version("pandas")), ("numpy", _version("numpy")),
               ("Plotly", _version("plotly")), ("SciPy", _version("scipy")), ("Data folder", str(default_paths().processed_dir.name) + "/")])
    st.caption("The fingerprint identifies the exact source file the pipeline read. If the file is replaced, run the pipeline and the notebooks again, then restart the dashboard.")


def render() -> None:
    ctx = current()
    page_header("Research", "Data and methods",
                "Where the numbers come from, what was cleaned, what every measure means, and what the analysis does not claim. The sidebar filters do not apply here.", [])
    quality, definitions, scope = st.tabs(["Data and quality", "Definitions", "Scope and sources"])
    with quality:
        _quality_tab(ctx)
    with definitions:
        _definitions_tab()
    with scope:
        _scope_tab(ctx)
