"""
app/views/popularity.py  -  page 12: popularity against chart place.

WHAT THIS FILE IS
    How the popularity score (0 to 100, from the streaming platform) lines up with the chart place: where the songs of every place sit,
    whether the more popular songs stand higher on a given day, how that link changed over time, and whether a change in popularity
    comes together with, or before, a change in place.

WHY WE NEED IT
    "Popularity vs rank scatter plots" are a core module of the brief. The chart is ordered by plays, the popularity score by recent
    listening, so the two should be close but are not the same thing; the gap is what is interesting.

WHAT HAPPENS WHEN IT RUNS
    All views use the song-days that match the sidebar. The correlation is Spearman's rho (based on ranks, so one very popular song
    cannot dominate it). A positive rho means that the more popular songs stand higher on the chart. The "before or together"
    view compares the 7-day change of popularity with the 7-day change of place in the same week and in the following week; it
    shows association only, because both numbers come from the same platform and a third thing (a new release, a viral clip) can move both.

EXPECTED OUTCOME
    A density map and a scatter of popularity against place, the daily correlation as a line, and the lead-lag bars.

MOTIVE
    Show what the popularity score does and does not tell us about the chart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.ui import charts as ch
from app.ui import theme
from app.ui.components import callout, kpi_cards, page_header, plot, section, stop_if_empty
from app.ui.state import current
from src.dash import compare as C
from src.dash import metrics as M
from src.dash import texts as T


def _density(rows: pd.DataFrame) -> pd.DataFrame:
    """Number of song-days for every place (columns) and popularity band (rows, 100 at the top)."""
    pairs = rows[["position", "popularity_clean"]].dropna()
    counts, _, _ = np.histogram2d(pairs["popularity_clean"], pairs["position"], bins=[np.arange(0, 105, 5), np.arange(0.5, 51.5, 1.0)])
    labels = [f"{lo} to {lo + 5}" for lo in range(0, 100, 5)]
    grid = pd.DataFrame(counts, index=labels, columns=range(1, 51))
    return grid.iloc[::-1]


def _lead_lag(ctx) -> pd.DataFrame:
    """Every selected song-day with its 7-day popularity change, its 7-day place gain, and the place gain of the NEXT seven days."""
    playlist = ctx.data.playlist
    change = C.add_pop_change(playlist, 7)["pop_change_lag"]
    frame = ctx.rows.assign(pop_change_lag=change.reindex(ctx.rows.index))
    later = playlist[["track_id", "snapshot_index", "rank_gain_7d"]].rename(columns={"rank_gain_7d": "gain_next_week"}).assign(snapshot_index=lambda d: d["snapshot_index"] - 7)
    return frame.merge(later, on=["track_id", "snapshot_index"], how="left")



def _link_words(result: dict, label: str) -> str:
    """One sentence about a rank correlation that follows its size and its sign (a negligible rho is 'no clear link', never 'goes with')."""
    rho, size = result["rho"], result["size"]
    if pd.isna(rho):
        return f"{label} the link cannot be measured"
    if size in ("", "negligible"):
        return f"{label} there is no clear link between the change in popularity and the change in place (rho **{T.num(rho, 2)}**, {T.n(result['n'])} song-days)"
    way = "a rise in popularity goes with a climb in the chart" if rho > 0 else "a rise in popularity goes with a fall in the chart"
    return f"{label} {way} (rho **{T.num(rho, 2)}**, {size}, {T.n(result['n'])} song-days)"


def render() -> None:
    ctx = current()
    page_header("Content", "Popularity and chart place",
                "How the popularity score lines up with the chart place, day by day and song by song.", ctx.chips)
    stop_if_empty(ctx.rows)
    rows, songs = ctx.rows, ctx.songs()
    pooled = C.spearman(51 - rows["position"], rows["popularity_clean"])
    daily = C.daily_rank_popularity_corr(rows)
    top, low = rows.loc[rows["position"] <= 10, "popularity_clean"], rows.loc[rows["position"] > 20, "popularity_clean"]
    trend = M.popularity_trend(rows)
    kpi_cards(columns=5, items=[
        {"label": "Average popularity", "value": T.num(rows["popularity_clean"].mean()), "unit": "points", "tone": "violet", "sub": f"median {T.num(rows['popularity_clean'].median(), 0)}; spread (std) {T.num(rows['popularity_clean'].std())}"},
        {"label": "Popularity and place", "value": T.num(pooled["rho"], 2), "unit": "rho", "tone": "blue", "sub": f"Spearman over {T.n(pooled['n'])} song-days; positive = popular songs stand higher"},
        {"label": "Typical day", "value": T.num(daily["rho"].median(), 2) if len(daily) else T.DASH, "unit": "rho", "tone": "aqua", "sub": "median of the daily correlations inside one day's chart"},
        {"label": "Top 10 against places 21-50", "value": T.signed(top.median() - low.median(), 0) if len(top) and len(low) else T.DASH, "unit": "points", "tone": "orange", "sub": "difference of median popularity of song-days"},
        {"label": "Popularity trend", "value": T.signed(trend["change"], 1) if pd.notna(trend["change"]) else T.DASH, "unit": "points", "tone": "pink", "sub": "last week against the first week of the selection"},
    ])

    # ---- place against popularity -------------------------------------------------------------------------------
    section("Place against popularity", "Where the song-days of every place sit on the popularity scale.")
    view = st.segmented_control("Show", ["Density map", "Sample of song-days"], default="Density map", key="po_view") or "Density map"
    left, right = st.columns([3, 2])
    with left:
        if view == "Density map":
            fig = ch.heat(_density(rows), "Song-days by chart place (across) and popularity score (down)", fmt=",.0f", show_text=False, height=470, colorbar_title="Song-days",
                          hover="place %{x}, popularity %{y}<br>%{z:,.0f} song-days<extra></extra>")
            fig.update_xaxes(title_text="Chart place", dtick=5)
            fig.update_yaxes(title_text="Popularity score")
        else:
            sample = rows[["position", "popularity_clean", "track_label", "date"]].dropna().sample(min(3500, len(rows)), random_state=0)
            sample = sample.assign(day=sample["date"].dt.strftime("%d %b %Y"))
            fig = ch.scatter_chart(sample, "position", "popularity_clean", title="A random sample of song-days", x_title="Chart place", y_title="Popularity score", hover_cols=["track_label", "day"],
                                   hover_names={"track_label": "Song", "day": "Day"}, opacity=0.35, fixed_size=5, height=470)
        plot(fig, key="po_density")
    with right:
        profile = C.position_profile(rows)
        fig = ch.ribbon_chart(profile, "position", "median", "p25", "p75", "Median popularity at every place", color=theme.VIOLET, height=470, x_title="Chart place",
                              y_title="Popularity score", fmt=".0f")
        plot(fig, key="po_ribbon")
    callout(f"Across **{T.n(pooled['n'])}** song-days the rank correlation between popularity and chart place is **{T.num(pooled['rho'], 2)}** ({pooled['size'] or 'no size word'}): "
            f"songs in the top 10 have a median popularity of **{T.num(top.median(), 0)}**, songs in places 21 to 50 **{T.num(low.median(), 0)}**. "
            "The score is related to the place but does not decide it; many songs of every popularity sit at every place.")

    # ---- daily correlation, and the song level ---------------------------------------------------------------
    a, b = st.columns(2)
    with a:
        if daily.empty:
            st.info("Too few songs on each day to calculate a daily correlation.")
        else:
            daily = daily.assign(smooth=daily["rho"].rolling(28, min_periods=7).mean())
            fig = ch.line_chart(daily, "date", "rho", title="Does the more popular song stand higher? (daily correlation)", y_title="Spearman rho within one day's chart", fmt=".2f", height=380)
            fig.add_scatter(x=daily["date"], y=daily["smooth"], mode="lines", name="28-day average", line=dict(color=theme.BLUE, width=3), hovertemplate="28-day average %{y:.2f}<extra></extra>")
            fig.add_hline(y=0, line=dict(color=theme.AXIS, dash="dot", width=1.2))
            fig.update_traces(line=dict(color="#b7d3f6", width=1.2), selector=dict(mode="lines", name=""))
            fig.update_layout(showlegend=True)
            plot(fig, key="po_daily")
    with b:
        songs_ok = songs.dropna(subset=["avg_popularity", "avg_rank"])
        sc = C.spearman(songs_ok["avg_popularity"], -songs_ok["avg_rank"])
        fig = ch.scatter_chart(songs_ok, "avg_popularity", "avg_rank", color="longevity_class", size="days_on_chart", title="Song by song: average popularity and average place", colors=theme.LONGEVITY_COLORS,
                               x_title="Average popularity score", y_title="Average place (1 = top)", hover_cols=["track_label", "days_on_chart"], hover_names={"track_label": "Song", "days_on_chart": "Days"},
                               reverse_y=True, size_max=22, opacity=0.6, height=380)
        plot(fig, key="po_songs")
        st.caption(f"Across {T.n(sc['n'])} songs, rho = {T.num(sc['rho'], 2)} ({sc['size'] or 'n/a'}). Bubble size = days on the chart.")

    # ---- before or together ---------------------------------------------------------------------------------------
    section("Before, or together?", "Does a change in popularity come with the change in place, or before it? Seven-day changes, historical association only.")
    frame = _lead_lag(ctx).dropna(subset=["pop_change_lag"])
    if len(frame) < 30:
        st.info("Too few song-days with a seven-day history in this selection.")
        return
    edges = [-np.inf, -5, -2, -0.5, 0.5, 2, 5, np.inf]
    names = ["fell by more than 5", "fell by 2 to 5", "fell by 0.5 to 2", "about the same", "rose by 0.5 to 2", "rose by 2 to 5", "rose by more than 5"]
    frame = frame.assign(band=pd.cut(frame["pop_change_lag"], edges, labels=names))
    grouped = frame.groupby("band", observed=True).agg(songs_days=("pop_change_lag", "size"), same_week=("rank_gain_7d", "mean"), next_week=("gain_next_week", "mean")).reindex(names).reset_index()
    same, nxt = C.spearman(frame["pop_change_lag"], frame["rank_gain_7d"]), C.spearman(frame["pop_change_lag"], frame["gain_next_week"])
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = ch.vbar(grouped, "band", ["same_week", "next_week"], "Average place gain, by how popularity changed in the last seven days",
                      names={"same_week": "Same seven days", "next_week": "Following seven days"}, colors={"same_week": theme.BLUE, "next_week": theme.ORANGE},
                      y_title="Average places gained (positive = climbed)", fmt=".1f", height=400)
        fig.update_xaxes(title_text="Popularity of the song, compared with seven days earlier")
        fig.add_hline(y=0, line=dict(color=theme.AXIS, width=1.2))
        plot(fig, key="po_leadlag")
    with c2:
        hardest = grouped.dropna(subset=["same_week"]).sort_values("same_week").iloc[0] if grouped["same_week"].notna().any() else None
        if same["size"] in ("", "negligible") and nxt["size"] in ("", "negligible") and pd.notna(same["rho"]) and pd.notna(nxt["rho"]):
            text = (f"In the **same seven days** there is no clear link between the change in popularity and the change in place (rho **{T.num(same['rho'], 2)}**, {T.n(same['n'])} song-days); "
                    f"the same holds for the **following seven days** (rho **{T.num(nxt['rho'], 2)}**, {T.n(nxt['n'])} song-days). "
                    "A change in popularity over a week therefore says very little about a change in place, either at the same time or a week later. ")
        else:
            text = (f"{_link_words(same, 'In the **same seven days**')}. {_link_words(nxt, 'In the **following seven days**')}. "
                    "If popularity moved ahead of the chart, the link would be stronger for the following week than for the same week. ")
        if hardest is not None:
            text += (f"The largest average loss of places in the same seven days, **{T.num(abs(hardest['same_week']), 1)} places**, belongs to the songs whose popularity {str(hardest['band']).lower()}. ")
        text += "Both numbers come from the same platform, so this is an association, not proof that one drives the other."
        callout(text, title="Reading the bars")
