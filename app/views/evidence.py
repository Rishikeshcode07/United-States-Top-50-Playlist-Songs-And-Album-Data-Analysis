"""
app/views/evidence.py  -  page 17: evidence and findings (Phases 4, 5, 6A and 6B in one place).

WHAT THIS FILE IS
    The page where every formal result of the project can be found and checked: the ten pre-planned hypothesis tests of Phase 5 (with
    the textbook p-value, the artist-aware p-value and the corrected p-value side by side), the 31 findings of the exploration in
    Phase 4 (searchable), and what the Phase 6A models and the Phase 6B survival model could and could not do in hindsight, with their
    stress tests.

WHY WE NEED IT
    A dashboard recalculates numbers for any filter, but a filtered number is not a test. The tests and the findings were fixed in
    notebooks, with a correction for ten tests and with artists (not songs) as the unit that is re-drawn. This page shows those
    results unchanged, so that every claim on the other pages can be traced to a number, a verdict and a caveat.

WHAT HAPPENS WHEN IT RUNS
    The page only reads the saved results (reports/p5_*.csv, p4_eda_findings.csv, p6a_*.csv, p6b_ladder.csv, p6b_risk_thirds.csv); the
    sidebar filters do not apply. A result file that is missing (the notebook was not run) switches off its own tab and says which
    notebook creates it. Nothing is recalculated, so the numbers are exactly the notebooks' numbers.

EXPECTED OUTCOME
    Six of the ten hypotheses supported after the correction (singles stay longer, entry place goes with run length, new songs arrive
    on Mondays and Tuesdays, ...), two suggestive (explicit songs), two not supported; models that are only modestly better than a
    coin on arrival-day facts and clearly better once the first three chart days are known.

MOTIVE
    Show the evidence next to its strength and its caveats, and keep every model result labelled as a backtest on past songs, not a
    forecast and not a recommendation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import Raw, badge, callout, cards, download_csv, esc, facts, html_table, kpi_cards, page_header, plot, section, table
from app.ui.reference import effect_text
from app.ui.state import NOTEBOOK_OF, report
from src.dash import texts as T

VERDICT_COLORS = {"Supported": theme.BLUE, "Suggestive": theme.YELLOW, "Not supported": "#8f8e84"}
P_FLOOR = 1e-4
STEPS = ["release", "timing", "artist", "debut", "early"]
STEP_TEXT = {"release": "release facts", "timing": "+ timing", "artist": "+ artist's record", "debut": "+ debut day", "early": "+ first 3 chart days"}
STEP_HELP = ("release facts = single or album, length, explicit, collaboration; timing = weekday and days of history; artist's record = the artist's earlier chart days and popularity; "
             "debut day = place, popularity and album drop on the arrival day; first 3 chart days = place, move and popularity on days 1 to 3. Every step adds to the one before.")
MODEL_NAMES = {"baseline": "Baseline (no inputs)", "logistic": "Logistic regression", "ridge": "Ridge regression", "boosting": "Gradient boosting", "forest": "Random forest"}
MODEL_COLORS = {"Baseline (no inputs)": "#8f8e84", "Logistic regression": theme.BLUE, "Ridge regression": theme.BLUE, "Gradient boosting": theme.ORANGE, "Random forest": theme.AQUA}
METRICS = {
    "binary": {"Area under the curve (0.5 = a coin, 1 = perfect)": ("auc", True), "Average precision (higher is better)": ("average_precision", True), "Brier score (lower is better)": ("brier", False)},
    "multiclass": {"Balanced accuracy (0.33 = a coin)": ("balanced_accuracy", True), "Macro F1 (higher is better)": ("macro_f1", True),
                   "Weighted kappa (0 = chance)": ("weighted_kappa", True), "Accuracy": ("accuracy", True)},
    "regression": {"R squared (0 = no better than the average)": ("r2", True), "Mean absolute error (lower is better)": ("mae", False), "Root mean squared error (lower is better)": ("rmse", False)},
}


def _missing(name: str, tab) -> None:
    notebook = NOTEBOOK_OF.get(name.split("_")[0], "the notebooks")
    tab.caption(f"This tab needs reports/{name}.csv, which {notebook} creates. Run the notebook (Run All) and refresh the page.")


# ====================================================================================================== tab 1
def _tests_tab(register: pd.DataFrame, robust: "pd.DataFrame | None") -> None:
    st.markdown("Ten questions were fixed **before** the data were examined. Each was tested twice: with the textbook test (which treats every song as independent) and with an **artist-aware** test "
                "(the songs of one artist are alike, so whole artists are the unit), and the ten p-values were corrected together (Holm). "
                "**Supported** = the corrected p-value is below 0.05. **Suggestive** = the artist-aware p-value is below 0.05 but the corrected one is not (a lead, not a result). "
                "**Not supported** = the artist-aware p-value is 0.05 or more (which is not the same as proof of no effect).")
    frame = register.copy()
    frame["label"] = frame["hypothesis"]
    order = frame["hypothesis"].tolist()
    fig = go.Figure()
    series = [("planned_p", "Textbook test (songs as independent)", "#b7d3f6", "circle-open"), ("primary_p", "Artist-aware test", theme.BLUE, "circle"),
              ("holm_p", "After the correction for ten tests (Holm)", theme.ORANGE, "diamond")]
    low, high = frame[[c for c, *_ in series]].min(axis=1).clip(lower=P_FLOOR), frame[[c for c, *_ in series]].max(axis=1).clip(lower=P_FLOOR)
    # one short line per hypothesis from its smallest to its largest p-value; the None between two lines keeps them apart
    line_x, line_y = [], []
    for lo, hi, label in zip(low, high, frame["label"]):
        line_x += [float(lo), float(hi), None]
        line_y += [label, label, None]
    fig.add_scatter(x=line_x, y=line_y, mode="lines", line=dict(color=theme.GRID, width=3), hoverinfo="skip", showlegend=False, connectgaps=False)
    for column, name, color, symbol in series:
        fig.add_scatter(x=frame[column].clip(lower=P_FLOOR), y=frame["label"], mode="markers", name=name, marker=dict(color=color, size=11, symbol=symbol, line=dict(width=2, color=color)),
                        customdata=np.array([T.p_text(v) for v in frame[column]]), hovertemplate="%{y}: %{customdata}<extra>" + name + "</extra>")
    fig.add_vline(x=0.05, line=dict(color=theme.INK_2, dash="dash", width=1.5))
    # on a log axis the position of a text label is given as log10 of the value (a line is given as the value itself)
    fig.add_annotation(x=float(np.log10(0.05)), y=1, yref="paper", text="0.05", showarrow=False, yanchor="bottom", font=dict(size=12, color=theme.INK_2))
    ticks = [P_FLOOR, 0.001, 0.01, 0.05, 0.1, 0.5, 1]
    fig.update_xaxes(type="log", tickmode="array", tickvals=ticks, ticktext=["< 0.0001", "0.001", "0.01", "0.05", "0.1", "0.5", "1"], range=[np.log10(P_FLOOR / 1.6), 0.1], title_text="p-value (log scale; smaller = stronger evidence)")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=order, showgrid=True)
    plot(ch.finish(fig, "The three p-values of each hypothesis", 430, "top"), key="ev_pvalues")
    callout("A hypothesis is **Supported** when the orange diamond is to the left of 0.05. Moving from the light circle to the blue dot shows the cost of treating songs as independent; moving on to the orange diamond shows the cost "
            "of testing ten questions at once. Explicit content (H1, H10) loses its support at the last step; singles staying longer (H2), entry place (H5) and the Monday and Tuesday arrivals (H6) keep it.")

    section("The register", "One row per hypothesis: the verdict, the size of the effect with its 95% interval, and the three p-values of the chart above.")
    tone = {"Supported": "pass", "Suggestive": "warn", "Not supported": "mute"}
    rows = [[r["hypothesis"], r["question"], Raw(badge(r["verdict"], tone.get(r["verdict"], "info"))),
             Raw(f"{esc(T.num(r['effect'], 2))} ({esc(T.num(r['effect_low'], 2))} to {esc(T.num(r['effect_high'], 2))})<span class=\"sub\">{esc(r['effect_unit'])}</span>"),
             _p_cell(r["planned_p"]), _p_cell(r["primary_p"]), _p_cell(r["holm_p"])]
            for _, r in register.iterrows()]
    html_table(["ID", "Question", "Verdict", "Effect (95% interval)", "Textbook p", "Artist-aware p", "Corrected p"], rows, numeric_cols=(4, 5, 6))
    shown = pd.DataFrame({"ID": register["hypothesis"], "Question": register["question"], "Verdict": register["verdict"], "Unit": register["unit"], "n": register["n"],
                          "Effect": [effect_text(row).replace("Effect: ", "") for _, row in register.iterrows()],
                          "Textbook p": register["planned_p"], "Artist-aware p": register["primary_p"], "Corrected p (Holm)": register["holm_p"]})
    download_csv(shown, "hypothesis_register_phase_5.csv", key="ev_csv1")

    section("Look at one test", "The question, how it was tested, how big the effect is, and what to be careful about.")
    choice = st.selectbox("Hypothesis", register["hypothesis"] + ":  " + register["question"], key="ev_hypothesis")
    row = register[register["hypothesis"] == choice.split(":")[0]].iloc[0]
    kpi_cards(columns=4, items=[
        {"label": f"{row['hypothesis']}  |  Verdict", "value": row["verdict"], "tone": {"Supported": "blue", "Suggestive": "yellow", "Not supported": "ink"}.get(row["verdict"], "blue"),
         "sub": f"{T.n(row['n'])} {row['unit']}s" if not str(row["unit"]).startswith(("song", "day", "month", "calendar", "complete")) else f"{T.n(row['n'])} x {row['unit']}"},
        {"label": "Effect", "value": T.num(row["effect"], 2), "tone": "aqua", "sub": f"{row['effect_unit']}; 95% interval {T.num(row['effect_low'], 2)} to {T.num(row['effect_high'], 2)}"},
        {"label": f"Standard size ({row['standard_metric']})", "value": T.num(row["standard_size"], 2), "tone": "orange", "sub": f"a {row['size_word']} effect"},
        {"label": "Corrected p-value (Holm)", "value": _p_cell(row["holm_p"]), "tone": "violet", "sub": f"artist-aware {T.p_text(row['primary_p'])}; textbook {T.p_text(row['planned_p'])}"},
    ])
    left, right = st.columns([3, 2])
    with left:
        facts([("Planned test", row["planned_test"]), ("Artist-aware test", row["primary_test"])])
        callout(row["caveat"], title="Caveat", tone="warn")
    with right:
        if robust is not None and (robust["hypothesis"] == row["hypothesis"]).any():
            test = robust[robust["hypothesis"] == row["hypothesis"]].iloc[0]
            first_half = "not repeated (the claim compares the halves)" if pd.isna(test["first_half_p"]) else f"{T.num(test['first_half_estimate'], 2)} ({T.p_text(test['first_half_p'])})"
            second_half = "not repeated" if pd.isna(test["second_half_p"]) else f"{T.num(test['second_half_estimate'], 2)} ({T.p_text(test['second_half_p'])})"
            direction = {True: "yes", False: "no"}.get(test["same_direction_in_both_halves"], "not applicable") if pd.notna(test["same_direction_in_both_halves"]) else "not applicable"
            facts([("All data", f"{T.num(test['estimate_all_data'], 2)} ({T.p_text(test['primary_p_all_data'])})"), ("First half of the file", first_half), ("Second half of the file", second_half),
                   ("Same direction in both halves", str(direction)),
                   ("Without the biggest artist", f"{T.num(test['estimate_without_biggest_artist'], 2)} ({T.p_text(test['primary_p_without_biggest_artist'])})"),
                   ("Corrected p without the biggest artist", _p_cell(test["holm_p_without_biggest_artist"])),
                   ("Verdict without the biggest artist", str(test["verdict_without_biggest_artist"]))], title="Does it hold up?")
        else:
            st.caption("The robustness table (reports/p5_robustness.csv) is not available.")

    if robust is not None:
        section("Robustness of the ten verdicts", "Each test was repeated on the two halves of the file and without the artist with the most songs.")
        grid = pd.DataFrame({"ID": robust["hypothesis"], "Verdict (all data)": robust["verdict"],
                             "Same direction in both halves": robust["same_direction_in_both_halves"].map({True: "yes", False: "no"}).fillna("not applicable"),
                             "Verdict without the biggest artist": robust["verdict_without_biggest_artist"], "Corrected p without the biggest artist": [_p_cell(v) for v in robust["holm_p_without_biggest_artist"]]})
        table(grid, height=38 * len(grid) + 40)
        moved = grid[grid["Verdict (all data)"] != grid["Verdict without the biggest artist"]]
        callout(f"The verdict stays the same without the biggest artist for **{len(grid) - len(moved)} of {len(grid)}** hypotheses. "
                + (f"It moves down one step for {', '.join(moved['ID'])} (the evidence there leans on a few big catalogues)." if len(moved) else "") +
                " A verdict that moves with one artist is a reason for care, not a reason to drop the result.", title="What to notice")


# ====================================================================================================== tab 2
def _findings_tab(findings: pd.DataFrame) -> None:
    st.markdown("The exploration of Phase 4 (playlist ranking, content attributes, popularity, artists, single songs) produced "
                f"**{len(findings)} findings**. Each has its evidence in numbers and the caveat that goes with it. Search for a word, or narrow by block.")
    blocks = list(dict.fromkeys(findings["block"].tolist()))
    picked = st.pills("Blocks", blocks, selection_mode="multi", default=blocks, key="ev_blocks", help="None chosen means all blocks.") or blocks
    word = st.columns([2, 3])[0].text_input("Search the findings", placeholder="for example explicit, Taylor Swift, weekend", key="ev_search").strip()
    view = findings[findings["block"].isin(picked)]
    if word:
        mask = view[["finding", "evidence", "caveat"]].apply(lambda col: col.str.contains(word, case=False, regex=False, na=False)).any(axis=1)
        view = view[mask]
    st.caption(f"{T.n(len(view))} of {T.n(len(findings))} findings shown.")
    if view.empty:
        callout("No finding contains that word. Try a shorter word or add more blocks.", title="Nothing found", tone="warn")
        return
    cards([{"tag": f"{row['finding_id']}  |  {row['block']}  |  {row['figure']}", "title": row["finding"], "text": f"{row['evidence']} **Caveat.** {row['caveat']}"} for _, row in view.iterrows()])
    download_csv(view.rename(columns={"finding_id": "ID", "block": "Block", "figure": "Figure in the notebook", "finding": "Finding", "evidence": "Evidence", "caveat": "Caveat"}),
                 "findings_phase_4.csv", key="ev_csv2")


# ====================================================================================================== tab 3
def _best_sentence(rows: pd.DataFrame, metric: str, higher_better: bool, label: str, step: str) -> str:
    models = rows[(rows["step"] == step) & (rows["model"] != "baseline")].dropna(subset=[metric])
    if models.empty:
        return ""
    best = models.sort_values(metric, ascending=not higher_better).iloc[0]
    base = rows[(rows["step"] == step) & (rows["model"] == "baseline")]
    text = f"With the information up to **{STEP_TEXT[step].replace('+ ', '')}**, the best model is **{MODEL_NAMES.get(best['model'], best['model'])}** at **{T.num(best[metric], 2)}**"
    if pd.notna(best.get(metric + "_low")) and pd.notna(best.get(metric + "_high")):
        text += f" (95% interval {T.num(best[metric + '_low'], 2)} to {T.num(best[metric + '_high'], 2)})"
    if not base.empty and pd.notna(base[metric].iloc[0]):
        text += f", against **{T.num(base[metric].iloc[0], 2)}** for the baseline that uses no inputs"
    return text + "."


def _models_tab(model_register: pd.DataFrame, stress: "pd.DataFrame | None", ladder: "pd.DataFrame | None", thirds: "pd.DataFrame | None") -> None:
    callout("Everything on this tab is a **backtest on past songs**. The models were trained on some artists and judged only on artists they had not seen. "
            "The numbers say how much the facts in this file carry about an outcome. They are not forecasts, and they are not recommendations to promote, drop or sign anything: "
            "a model that is only slightly better than a coin should not drive a decision.", title="How to read this tab", tone="warn")
    titles = model_register.drop_duplicates("question").set_index("question")[["title", "kind"]]
    options = [f"{q}:  {titles.loc[q, 'title']}" for q in titles.index]
    c1, c2 = st.columns([3, 2.6], vertical_alignment="bottom")
    choice = c1.selectbox("Question", options, key="ev_question")
    question = choice.split(":")[0]
    kind = titles.loc[question, "kind"]
    metric_name = c2.selectbox("Score", list(METRICS[kind]), key=f"ev_metric_{kind}")
    metric, higher_better = METRICS[kind][metric_name]
    rows = model_register[model_register["question"] == question].copy()
    songs = int(rows["songs"].max())
    steps = [s for s in STEPS if s in set(rows["step"])]
    rows["model_name"] = rows["model"].map(MODEL_NAMES).fillna(rows["model"])
    rows["step_name"] = rows["step"].map(STEP_TEXT)
    low_col, high_col = metric + "_low", metric + "_high"
    plotted = rows.dropna(subset=[metric])
    left, right = st.columns([3.2, 1.3])
    with left:
        order = [MODEL_NAMES[m] for m in MODEL_NAMES if MODEL_NAMES[m] in set(plotted["model_name"])]
        fig = ch.interval_bars(plotted.assign(**{low_col: plotted[low_col].fillna(plotted[metric]), high_col: plotted[high_col].fillna(plotted[metric])}), "step_name", metric, low_col, high_col,
                               title=f"{titles.loc[question, 'title']}: {metric_name.split(' (')[0].lower()} by information step", group="model_name", order=order, colors=MODEL_COLORS,
                               fmt=".2f", y_title=metric_name.split(" (")[0], height=440)
        fig.update_xaxes(categoryorder="array", categoryarray=[STEP_TEXT[s] for s in steps])
        if metric in ("auc",):
            fig.add_hline(y=0.5, line=dict(color=theme.AXIS, dash="dot", width=1.3))
        if metric in ("r2", "weighted_kappa"):
            fig.add_hline(y=0, line=dict(color=theme.AXIS, dash="dot", width=1.3))
            fig.update_yaxes(rangemode="normal")
        plot(fig, key="ev_models")
    with right:
        last = steps[-1]
        callout(_best_sentence(rows, metric, higher_better, metric_name, last) or "No model result for this score.", title="Reading")
        st.caption(f"{T.n(songs)} songs. {STEP_HELP}")
    shown = rows[["step_name", "model_name", "songs", metric, low_col, high_col]].rename(columns={"step_name": "Information", "model_name": "Model", "songs": "Songs", metric: metric_name.split(" (")[0],
                                                                                                low_col: "Lower end", high_col: "Upper end"})
    with st.expander("The numbers behind this chart"):
        table(shown, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in (metric_name.split(" (")[0], "Lower end", "Upper end")}, height=min(420, 38 * len(shown) + 40))
        download_csv(shown, f"models_{question}.csv", key="ev_csv3")

    if stress is not None and (stress["question"] == question).any():
        section("Stress tests", "The same model, judged again after changing one thing: another split of the artists, a stronger or weaker penalty, no Taylor Swift, or training on early songs and judging on later ones.")
        part = stress[stress["question"] == question].reset_index(drop=True)
        part = part.assign(check_name=part["check"].str.replace("as in the notebook", "As in the notebook"), reading="")
        fig = ch.forest_chart(part, "check_name", "score", "low", "high", title="Area under the curve in each stress test (the notebook's own run on top)", reference=0.5, log_x=False,
                              color_col="model", colors={"Logistic regression": theme.BLUE, "Gradient boosting": theme.ORANGE}, x_title="Area under the curve (0.5 = a coin)", height=max(360, 30 * len(part) + 110))
        fig.update_xaxes(range=[0.45, 0.95])
        plot(fig, key="ev_stress")
        low, high = float(part["score"].min()), float(part["score"].max())
        callout(f"Across {T.n(len(part))} stress runs the area under the curve stays between **{T.num(low, 2)}** and **{T.num(high, 2)}**: the result does not hinge on one split, one penalty or one artist. "
                "The later-songs runs are noisier because far fewer songs are left to judge.", title="What to notice")
    elif stress is not None:
        st.caption("Stress tests were run for the questions C1, C2 and E2 only.")

    if ladder is not None and thirds is not None:
        section("How long a song stays: what a survival model could rank (Phase 6B)", "The C-index is the chance that, of two songs, the model gives the higher risk to the one that left sooner (0.5 = a coin, 1 = perfect).")
        left, right = st.columns(2)
        with left:
            frame = ladder.assign(step_name=ladder["step"], view_name=ladder["view"].str.capitalize())
            fig = ch.interval_bars(frame, "step_name", "c_index", "low", "high", title="C-index of the survival model, by information step", group="view_name", colors={"Arrival day": theme.BLUE, "After three chart days": theme.AQUA},
                                   fmt=".2f", y_title="C-index (0.5 = a coin)", height=400)
            fig.add_hline(y=0.5, line=dict(color=theme.AXIS, dash="dot", width=1.3))
            fig.update_yaxes(range=[0.4, 0.85], rangemode="normal")
            plot(fig, key="ev_ladder")
        with right:
            risk = thirds.assign(view_name=thirds["view"].str.capitalize(), third_name=thirds["third"].str.capitalize())
            fig = ch.interval_bars(risk, "third_name", "median", "median_low", "median_high", title="Median stay of the songs in each risk third", group="view_name", colors={"Arrival day": theme.BLUE, "After three chart days": theme.AQUA},
                                   fmt=".0f", y_title="Median stay (chart days)", height=400)
            plot(fig, key="ev_thirds")
        arrival = ladder[(ladder["view"] == "arrival day")].iloc[-1]
        early = ladder[(ladder["view"] != "arrival day")].iloc[-1]
        callout(f"Knowing only the arrival day, the best survival model ranks two songs correctly **{T.pct(arrival['c_index'], 0)}** of the time; once the first three chart days are known it is **{T.pct(early['c_index'], 0)}**. "
                "Most of the gain comes from where the song debuts and how it moves in its first days, not from release type, explicit content or collaboration. "
                "The risk thirds are built from the model's score, and the median stay differs a lot between them; that is a description of the past, not a rule for the next song.", title="What to notice")


def _p_cell(value) -> str:
    """A p-value as it is written in a table: '0.021', or '< 0.001' when it is very small."""
    if pd.isna(value):
        return "n/a"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def render() -> None:
    page_header("Research", "Evidence and findings",
                "Every formal result of the project in one place, with its strength and its caveats. The sidebar filters do not apply here.", [])
    register, robust, findings = report("p5_hypothesis_register"), report("p5_robustness"), report("p4_eda_findings")
    models, stress, ladder, thirds = report("p6a_model_register"), report("p6a_stress_tests"), report("p6b_ladder"), report("p6b_risk_thirds")
    counts = register["verdict"].value_counts() if register is not None else pd.Series(dtype=int)
    kpi_cards(columns=5, items=[
        {"label": "Pre-planned tests", "value": T.n(len(register)) if register is not None else "n/a", "tone": "blue", "sub": "Phase 5, corrected for ten tests"},
        {"label": "Supported", "value": T.n(counts.get("Supported", 0)), "tone": "green", "sub": "the corrected p-value is below 0.05"},
        {"label": "Suggestive", "value": T.n(counts.get("Suggestive", 0)), "tone": "yellow", "sub": "a lead that does not survive the correction"},
        {"label": "Exploration findings", "value": T.n(len(findings)) if findings is not None else "n/a", "tone": "aqua", "sub": "Phase 4, each with its caveat"},
        {"label": "Model questions", "value": T.n(models["question"].nunique()) if models is not None else "n/a", "tone": "violet", "sub": "Phase 6A, judged on unseen artists"},
    ])
    tests, find, fit = st.tabs(["Hypothesis tests", "Exploration findings", "Models in hindsight"])
    with tests:
        if register is None:
            _missing("p5_hypothesis_register", st)
        else:
            _tests_tab(register, robust)
    with find:
        if findings is None:
            _missing("p4_eda_findings", st)
        else:
            _findings_tab(findings)
    with fit:
        if models is None:
            _missing("p6a_model_register", st)
        else:
            _models_tab(models, stress, ladder, thirds)
