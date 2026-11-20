"""
app/views/concentration.py  -  page 11: how concentrated is the chart?

WHAT THIS FILE IS
    The page about the chart as a whole in terms of artists: how many artists share the chart points, how unequal the shares are
    (Gini, HHI, effective number of artists, Lorenz curve, Pareto bars) and how that changed week by week or month by month.

WHY WE NEED IT
    "Taylor Swift holds 7.5 %" is a fact about one artist. "Ten artists hold 41 % and the Gini is 0.7" is a fact about the whole chart,
    and it can be compared month by month.

WHAT HAPPENS WHEN IT RUNS
    The shares are shares of the chart points of the SELECTED songs (the main artist gets a song's points), so with an artist or
    release-type filter the page describes concentration inside that group. The measures come from src/eda/concentration.py, the
    same functions as in the Phase 4 notebook.

EXPECTED OUTCOME
    Concentration numbers that match the notebook for the whole period, plus curves for any filter.

MOTIVE
    One definition of concentration for the notebook, the dashboard and the paper.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, download_csv, kpi_cards, page_header, plot, section, stop_if_empty, table
from app.ui.state import current
from src.dash import metrics as M
from src.dash import texts as T
from src.eda import concentration as conc


def render() -> None:
    ctx = current()
    page_header("Artists", "Concentration",
                "How many artists share the chart points, and how unequal the shares are. Shares are taken among the songs of the selection.", ctx.chips)
    stop_if_empty(ctx.rows)
    points = ctx.rows.groupby("primary_artist")["chart_points"].sum().sort_values(ascending=False)
    values = points.to_numpy(dtype=float)
    summary = conc.concentration_summary(values, top_ns=(1, 5, 10, 20))

    kpi_cards(columns=6, items=[
        {"label": "Artists", "value": T.n(summary["holders"]), "tone": "blue", "sub": "main artists with chart points"},
        {"label": "Top 1 share", "value": T.pct(summary["top_1_share"], 1), "tone": "pink", "sub": f"{points.index[0]}"},
        {"label": "Top 10 share", "value": T.pct(summary["top_10_share"], 1), "tone": "violet", "sub": f"top 20 hold {T.pct(summary['top_20_share'], 0)}"},
        {"label": "Effective number", "value": T.num(summary["effective_number"], 0), "tone": "aqua", "sub": "equal artists with the same HHI"},
        {"label": "HHI", "value": T.n(summary["hhi"]), "tone": "yellow", "sub": "0 to 10,000; sum of squared % shares"},
        {"label": "Gini", "value": T.num(summary["gini"], 2), "tone": "orange", "sub": "0 = equal, 1 = one holder"},
    ])

    period = (st.segmented_control("Look at the chart by", ["Week", "Month"], default="Month", key="co_period") or "Month").lower()
    by_period = M.concentration_by_period(ctx.rows, period)
    left, right = st.columns(2)
    with left:
        long = by_period.melt(id_vars="period", value_vars=["top1_pct", "top5_pct", "top10_pct"], var_name="group", value_name="share")
        long["group"] = long["group"].map({"top1_pct": "Biggest artist", "top5_pct": "Top 5 artists", "top10_pct": "Top 10 artists"})
        fig = ch.line_chart(long, "period", "share", group="group", title=f"Share of the chart points held by the biggest artists, by {period}", order=["Top 10 artists", "Top 5 artists", "Biggest artist"],
                            colors={"Biggest artist": theme.PINK, "Top 5 artists": theme.VIOLET, "Top 10 artists": theme.BLUE}, y_title="% of the chart points", fmt=".1f", suffix="%",
                            markers=period == "month", height=380)
        fig.update_xaxes(tickformat="%b %Y")
        plot(fig, key="co_top")
    with right:
        effective = by_period.melt(id_vars="period", value_vars=["artists", "effective_number"], var_name="group", value_name="count")
        effective["group"] = effective["group"].map({"artists": "Artists with points", "effective_number": "Effective number of artists"})
        fig = ch.line_chart(effective, "period", "count", group="group", title=f"How many artists share the chart, by {period}", order=["Artists with points", "Effective number of artists"],
                            colors={"Artists with points": theme.AXIS, "Effective number of artists": theme.BLUE}, y_title="Artists", fmt=".0f", markers=period == "month", height=380)
        fig.update_xaxes(tickformat="%b %Y")
        plot(fig, key="co_effective")
    if len(by_period) >= 2:
        first, last = by_period.iloc[0], by_period.iloc[-1]
        peak = by_period.loc[by_period["top10_pct"].idxmax()]
        callout(f"Over the whole selection the top 10 artists hold **{T.pct(summary['top_10_share'], 1)}** of the chart points and the chart behaves as if **{T.num(summary['effective_number'], 0)}** equal artists shared it "
                f"({T.n(summary['holders'])} artists appear). The top-10 share was highest in **{peak['period']:%b %Y}** (**{T.pct(peak['top10_pct'], 1, already_percent=True)}**) and lowest in "
                f"**{by_period.loc[by_period['top10_pct'].idxmin(), 'period']:%b %Y}** (**{T.pct(by_period['top10_pct'].min(), 1, already_percent=True)}**). Periods with few chart days can look more extreme.")

    section("How unequal are the shares?")
    a, b = st.columns(2)
    with a:
        x, y = conc.lorenz_curve(values)
        plot(ch.lorenz_chart(x, y, summary["gini"], "Lorenz curve of the chart points", height=420), key="co_lorenz")
    with b:
        n_bars = st.slider("Artists in the bar chart", 5, 40, 20, key="co_n", help="The bars show each artist's share; the line is the running total.")
        top = points.head(n_bars)
        share = top / points.sum() * 100
        cumulative = conc.cumulative_share(values)[: len(top)] * 100
        plot(ch.pareto_chart(top.index.tolist(), share.tolist(), cumulative.tolist(), f"The {len(top)} biggest artists and their running total", height=420), key="co_pareto")
    st.caption("A Lorenz curve that bends far below the dashed diagonal means a few artists hold most of the points. The Gini is twice the area between the curve and the diagonal.")

    section("The numbers by period")
    shown = pd.DataFrame({"Period": by_period["period"].dt.date, "Artists": by_period["artists"], "HHI": by_period["hhi"].round(0), "Effective number": by_period["effective_number"].round(1),
                          "Biggest artist (%)": by_period["top1_pct"].round(1), "Top 5 (%)": by_period["top5_pct"].round(1), "Top 10 (%)": by_period["top10_pct"].round(1), "Gini": by_period["gini"].round(3)})
    table(shown, column_config={"Period": st.column_config.DateColumn("Period starting", format="D MMM YYYY")}, height=min(430, 36 * min(len(shown), 10) + 44))
    download_csv(shown, f"concentration_by_{period}.csv", key="co_csv")
