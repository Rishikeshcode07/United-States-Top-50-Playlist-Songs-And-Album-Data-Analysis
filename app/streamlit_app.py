"""
app/streamlit_app.py  -  the entry file of the dashboard.

HOW TO START IT (from the project folder, with the virtual environment active)
    streamlit run app/streamlit_app.py

WHAT THIS FILE IS
    The only file that Streamlit runs directly. On every click it:
      1. makes sure the project folder is on Python's path (so `src` and `app` can be imported from anywhere),
      2. sets the page (title, icon, wide layout) and loads the chart style,
      3. reads the cleaned tables (once per server; a friendly message if they are not there yet),
      4. draws the filter widgets in the sidebar and works out "the selection",
      5. draws the menu (a bar at the top, with the six sections as drop-downs) and runs the page that is open.

WHY WE NEED IT
    The filters live HERE, outside the pages, so that a choice such as "explicit songs only, January to March 2025" stays in place
    when the person moves from one page to another.

WHAT HAPPENS WHEN IT RUNS
    Streamlit runs this file from top to bottom on every click. Reading the data and the heavy tables is cached, so a click
    costs a few milliseconds plus whatever the open page draws.

EXPECTED OUTCOME
    A wide dashboard with a menu of 18 pages in 6 sections and one set of filters that every page obeys.

MOTIVE
    One place that decides what "the selection" is.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                      # lets `import src...` and `import app...` work wherever Streamlit was started
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="US Top 50 Playlist Analytics", page_icon=str(ROOT / "app" / "assets" / "favicon.png"),
                   layout="wide", initial_sidebar_state="expanded")

from app import nav  # noqa: E402
from app.ui import components as ui  # noqa: E402
from app.ui import state, theme  # noqa: E402
from src.dash.load import DataNotReady, PIPELINE_COMMAND  # noqa: E402

theme.register_template()
theme.inject_css()
st.logo(str(ROOT / "app" / "assets" / "logo.svg"), size="large")

try:
    state.get_data()
except DataNotReady as problem:
    ui.page_header("Dashboard", "The chart data is not built yet",
                   "The dashboard reads the cleaned tables that Phase 3 creates. They are not in the data folder yet.")
    ui.callout(f"{problem}", title="What to do", tone="warn")
    st.code(f"{PIPELINE_COMMAND}\nstreamlit run app/streamlit_app.py", language="bash")
    st.stop()

bounds = state.get_bounds()
filters = state.sidebar_filters(bounds)
context = state.build_context(bounds, filters)
st.session_state[state.CTX_KEY] = context
with st.sidebar:
    st.caption(state.selected_note(context))

menu = nav.build()
st.session_state["_pages"] = {page.url_path: page for pages in menu.values() for page in pages}
current_page = st.navigation(menu, position="top")
current_page.run()
