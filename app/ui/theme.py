"""
app/ui/theme.py  -  the look of the dashboard: colours, the chart template and the CSS.

WHAT THIS FILE IS
    The colour tokens of the project (the same palette as the notebook charts, see src/eda/style.py; a test checks that they
    still agree), the meaning of each colour (explicit = orange, not explicit = blue, single = green, album = violet ...), the
    Plotly template that every chart uses, and the CSS that gives the page its cards, header, chips and tabs.

WHY WE NEED IT
    A dashboard looks like one product when every chart and every card follows the same few rules. Putting the rules in one file
    means a page never chooses a colour by itself: it asks for "the explicit colour" or "the album colour".

    Design rules used here
      * warm off-white page, white cards with a hairline border, near-black text, one strong accent (blue);
      * colour carries meaning and keeps it on every page (orange is always explicit songs, green is always up and so on);
      * no emoji and no decorative icons; the only icons are the plain symbols of the menu;
      * numbers use tabular figures so that columns of numbers line up;
      * system fonts only (Segoe UI on Windows, San Francisco on a Mac), so nothing is downloaded and the page looks the same offline.

WHAT HAPPENS WHEN IT RUNS
    `register_template()` adds the template "atlantic" to Plotly and makes it the default. `inject_css()` writes the CSS into the page
    (it is the only function here that needs Streamlit, and it imports it only when called, so the rest can be tested without it).

EXPECTED OUTCOME
    Every chart of the dashboard has the same fonts, grid, legend and hover box without any page asking for it.

MOTIVE
    One look, defined once.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from src.config import LONGEVITY_LABELS

# ------------------------------------------------------------------------------------------------ colour tokens
BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN, VIOLET, RED]
SURFACE, PAGE, CARD = "#fcfcfb", "#f9f9f7", "#ffffff"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#6b6a65", "#e1e0d9", "#c3c2b7"
BLUE_DARK, BLUE_SOFT = "#1c5cab", "#eaf2fc"
SEQUENTIAL = ["#eaf2fc", "#b7d3f6", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]            # light -> dark blue (heatmaps)
DIVERGING = ["#104281", "#2a78d6", "#b7d3f6", "#f0efec", "#f4c2be", "#c74845", "#762221"]  # blue = good / more, red = bad / less
FONT = "'Source Sans', 'Source Sans Pro', 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Arial, 'Liberation Sans', sans-serif"

# ------------------------------------------------------------------------------------------ what each colour means
EXPLICIT_COLORS = {"Explicit": ORANGE, "Not explicit": BLUE}
ALBUM_COLORS = {"single": AQUA, "album": VIOLET, "compilation": YELLOW, "Single": AQUA, "Album or compilation": VIOLET,
                "Album": VIOLET, "Compilation": YELLOW}
COLLAB_COLORS = {"Solo": BLUE, "Collaboration": PINK}
MOVE_COLORS = {"Up": GREEN, "Down": RED, "Same": "#8f8e84", "New entry": BLUE, "Re-entry": VIOLET, "Chart start": "#8f8e84"}
LONGEVITY_COLORS = dict(zip(LONGEVITY_LABELS, ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]))      # Flash (lightest) ... Evergreen (darkest)
BAND_COLORS = {"1 to 10": "#104281", "11 to 20": "#1c5cab", "21 to 30": "#2a78d6", "31 to 40": "#5598e7", "41 to 50": "#86b6ef"}
ARCHETYPE_COLORS = {"Brief visitors": "#8f8e84", "Quick fades": ORANGE, "Month-long fades": YELLOW, "Slow climbers": AQUA, "Lasting hits": BLUE}      # Phase 6C song groups
SEGMENT_COLORS = {"Passing visitors": "#8f8e84", "Mid-chart stayers": AQUA, "Top-chart hit-makers": ORANGE, "Catalogue artists": BLUE}            # Phase 6C artist groups
GOOD, BAD, NEUTRAL = GREEN, RED, AXIS


def colors_for(names, mapping=None) -> dict:
    """A colour for every name: the meaningful colour when `mapping` knows it, otherwise the next colour of the palette."""
    mapping = mapping or {}
    out, spare = {}, iter(CATEGORICAL * 4)
    for name in names:
        out[name] = mapping.get(name) or next(spare)
    return out


# --------------------------------------------------------------------------------------------- the Plotly template
def build_template() -> go.layout.Template:
    """The Plotly template of the dashboard (fonts, grid, legend, hover box, colour sequence)."""
    axis = dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="outside", tickcolor=AXIS, ticklen=4, automargin=True,
                title=dict(font=dict(size=12, color=MUTED)), tickfont=dict(size=12, color=INK_2))
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT, size=13, color=INK_2), colorway=CATEGORICAL, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=15, color=INK, family=FONT), x=0.0, xanchor="left", y=0.965, yanchor="top", pad=dict(l=4)),
        margin=dict(l=14, r=16, t=56, b=14), xaxis=axis, yaxis=axis,
        legend=dict(font=dict(size=12, color=INK_2), bgcolor="rgba(0,0,0,0)", title=dict(font=dict(size=12, color=MUTED))),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
        colorscale=dict(sequential=[(i / (len(SEQUENTIAL) - 1), c) for i, c in enumerate(SEQUENTIAL)],
                        diverging=[(i / (len(DIVERGING) - 1), c) for i, c in enumerate(DIVERGING)]),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=MUTED, activecolor=BLUE), hovermode="closest",
    )
    return template


def register_template(name: str = "atlantic") -> str:
    """Add the template to Plotly under `name` and make it the default for every figure made afterwards."""
    pio.templates[name] = build_template()
    pio.templates.default = name
    return name


# ------------------------------------------------------------------------------------------------------- the CSS
CSS = """
:root{
  --blue:#2a78d6; --blue-dark:#1c5cab; --blue-soft:#eaf2fc; --orange:#eb6834; --aqua:#1baf7a; --yellow:#eda100; --pink:#e87ba4;
  --green:#008300; --violet:#6250d6; --red:#e34948;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#6b6a65; --grid:#e1e0d9; --axis:#c3c2b7; --page:#f9f9f7; --card:#ffffff;
  --font:'Source Sans','Source Sans Pro','Segoe UI',-apple-system,BlinkMacSystemFont,Roboto,Arial,'Liberation Sans',sans-serif;
  color-scheme: light;
}
html, body, .stApp, [data-testid="stAppViewContainer"]{ background:var(--page); color:var(--ink); }
.pg-kicker,.pg-title,.pg-sub,.chip,.kpi,.callout,.card,.sec,.badge,.songline{ font-family:var(--font); }
[data-testid="stHeader"]{ background:rgba(249,249,247,0); }
.block-container{ padding:4.6rem 2.2rem 3.5rem 2.2rem; max-width:1560px; }
h1,h2,h3,h4{ letter-spacing:-0.01em; color:var(--ink); }
[data-testid="stSidebar"]{ background:#f2f1ec; border-right:1px solid var(--grid); }
[data-testid="stSidebar"] .block-container{ padding-top:1rem; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3{ font-size:.78rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin:1.1rem 0 .2rem 0; }
footer{ visibility:hidden; }

/* page header */
.pg-kicker{ font-size:.74rem; font-weight:750; letter-spacing:.13em; text-transform:uppercase; color:var(--blue); margin:0 0 .25rem 0; }
.pg-title{ font-size:2.1rem; font-weight:760; line-height:1.15; margin:0; color:var(--ink); }
.pg-sub{ color:var(--ink2); font-size:1.02rem; line-height:1.5; margin:.5rem 0 .1rem 0; max-width:64rem; }
.chips{ display:flex; flex-wrap:wrap; gap:.4rem; margin:.7rem 0 1.0rem 0; }
.chip{ background:#e9f1fb; color:#1c5cab; border:1px solid #c9dcf5; border-radius:999px; padding:.16rem .65rem; font-size:.78rem; font-weight:650; }
.chip.muted{ background:#eeede8; color:var(--ink2); border-color:var(--grid); }

/* KPI cards */
.kpis{ display:grid; grid-template-columns:repeat(auto-fit,minmax(205px,1fr)); gap:14px; margin:.4rem 0 1.2rem 0; }
.kpis.cols-6{ grid-template-columns:repeat(6,minmax(0,1fr)); } .kpis.cols-5{ grid-template-columns:repeat(5,minmax(0,1fr)); }
.kpis.cols-4{ grid-template-columns:repeat(4,minmax(0,1fr)); } .kpis.cols-3{ grid-template-columns:repeat(3,minmax(0,1fr)); }
@media (max-width:1439px){ .kpis.cols-6,.kpis.cols-5{ grid-template-columns:repeat(3,minmax(0,1fr)); } .kpis.cols-4{ grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width:820px){ .kpis{ grid-template-columns:repeat(2,minmax(0,1fr)) !important; } }
.kpi{ background:var(--card); border:1px solid var(--grid); border-radius:14px; padding:14px 16px 13px 18px; position:relative; overflow:hidden; }
.kpi:before{ content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background:var(--accent,var(--blue)); }
.kpi-label{ font-size:.72rem; font-weight:750; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); }
.kpi-value{ font-size:1.95rem; font-weight:770; line-height:1.15; margin:.28rem 0 .12rem 0; color:var(--ink); font-variant-numeric:tabular-nums; }
.kpi-value small{ font-size:1rem; font-weight:650; color:var(--ink2); margin-left:.15rem; }
.kpi-value.long{ font-size:1.42rem; line-height:1.22; margin:.4rem 0 .22rem 0; }
.kpi-sub{ font-size:.82rem; color:var(--ink2); line-height:1.35; }
.up{ color:var(--green); font-weight:700; } .down{ color:var(--red); font-weight:700; }

/* sections, callouts, cards */
.sec{ margin:1.5rem 0 .5rem 0; }
.sec-title{ font-size:1.18rem; font-weight:720; color:var(--ink); margin:0; }
.sec-note{ color:var(--muted); font-size:.9rem; margin:.1rem 0 0 0; max-width:62rem; }
.facts{ display:grid; grid-template-columns:max-content 1fr; gap:.42rem 1.1rem; margin:.5rem 0 0 0; padding:.85rem 1.05rem; background:var(--card); border:1px solid var(--grid);
  border-radius:12px; font-size:.9rem; }
.facts dt{ color:var(--muted); margin:0; } .facts dd{ margin:0; color:var(--ink); font-weight:600; text-align:right; }
.facts-title{ grid-column:1 / -1; font-size:.68rem; font-weight:760; letter-spacing:.11em; text-transform:uppercase; color:var(--blue); margin-bottom:.1rem; }
.facts.wide{ grid-template-columns:minmax(9rem,14rem) 1fr; gap:.55rem 1.4rem; }
.facts.wide dt{ font-weight:600; color:var(--ink2); }
.facts.wide dd{ text-align:left; font-weight:400; color:var(--ink2); line-height:1.45; }
.minitable-wrap.scroll{ overflow:auto; }
.minitable-wrap.scroll thead th{ position:sticky; top:0; background:var(--card); z-index:1; box-shadow:0 1px 0 var(--grid); }
.minitable.wide th{ text-align:left; white-space:nowrap; }
.minitable.wide td{ text-align:left; white-space:normal; vertical-align:top; font-weight:500; line-height:1.38; }
.minitable.wide th.num, .minitable.wide td.num{ text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
.minitable.wide td.nw{ white-space:nowrap; }
.minitable .sub{ display:block; color:var(--muted); font-size:.8rem; font-weight:400; margin-top:.15rem; }
.minitable-wrap{ background:var(--card); border:1px solid var(--grid); border-radius:12px; overflow:hidden; margin:.5rem 0 .6rem 0; }
.minitable{ width:100%; border-collapse:collapse; font-size:.88rem; }
.minitable th, .minitable td{ border-left:0; border-right:0; border-top:0; }
.minitable .badge{ white-space:nowrap; }
.minitable th{ text-align:right; font-size:.66rem; letter-spacing:.09em; text-transform:uppercase; color:var(--muted); font-weight:700; padding:.6rem .75rem .5rem .75rem; border-bottom:1px solid var(--grid); }
.minitable td{ text-align:right; color:var(--ink); font-weight:600; padding:.5rem .75rem; border-bottom:1px solid var(--grid); white-space:nowrap; }
.minitable tr:last-child td{ border-bottom:0; }
.minitable th:first-child, .minitable td:first-child{ text-align:left; }
.minitable td.note{ color:var(--muted); font-weight:500; text-align:left; white-space:normal; }
.minitable .sw{ display:inline-block; width:.72rem; height:.72rem; border-radius:3px; margin-right:.55rem; vertical-align:-1px; }
.callout{ background:#fff; border:1px solid var(--grid); border-left:5px solid var(--blue); border-radius:10px; padding:.8rem 1.05rem; margin:.7rem 0 1.1rem 0;
  color:var(--ink2); font-size:.96rem; line-height:1.55; }
.callout b{ color:var(--ink); }
.callout-title{ font-size:.7rem; font-weight:760; letter-spacing:.11em; text-transform:uppercase; color:var(--blue); margin-bottom:.2rem; }
.callout.warn{ border-left-color:var(--yellow); } .callout.warn .callout-title{ color:#8a6100; }
.card{ background:var(--card); border:1px solid var(--grid); border-radius:14px; padding:14px 18px; height:100%; }
.card h4{ margin:0 0 .3rem 0; font-size:1.02rem; font-weight:720; }
.card p{ margin:0; color:var(--ink2); font-size:.92rem; line-height:1.5; }
.card .tag{ display:inline-block; font-size:.7rem; font-weight:760; letter-spacing:.08em; color:var(--blue-dark); background:var(--blue-soft); border-radius:6px; padding:.1rem .45rem; margin-bottom:.35rem; }
.cards{ display:flex; flex-wrap:wrap; gap:14px; margin:.4rem 0 1.2rem 0; }
.cards > .card{ flex:1 1 var(--card-min,290px); min-width:0; height:auto; }
.badge{ display:inline-block; padding:.08rem .5rem; border-radius:6px; font-size:.74rem; font-weight:720; }
.badge-pass{ background:#e3f5ec; color:#0b6b43; } .badge-warn{ background:#fdf1d3; color:#7a5200; } .badge-fail{ background:#fbe3e2; color:#9b1f1e; }
.badge-info{ background:var(--blue-soft); color:var(--blue-dark); } .badge-mute{ background:#eceae3; color:#52514e; }
.songline{ display:flex; gap:14px; align-items:center; }

/* charts, tabs, widgets */
[data-testid="stPlotlyChart"]{ background:#fff; border:1px solid var(--grid); border-radius:14px; padding:0; overflow:hidden; }
button[data-baseweb="tab"]{ font-weight:650; font-size:.95rem; }
[data-testid="stDataFrame"]{ border:1px solid var(--grid); border-radius:10px; }
[data-testid="stExpander"]{ background:#fff; border-radius:10px; }
div[data-testid="stMetric"]{ background:#fff; border:1px solid var(--grid); border-radius:12px; padding:10px 14px; }
"""


def inject_css() -> None:
    """Write the CSS into the page. Call once per run, near the top of the entry file."""
    import streamlit as st
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
