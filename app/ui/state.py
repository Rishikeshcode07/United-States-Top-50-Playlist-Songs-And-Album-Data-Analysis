"""
app/ui/state.py  -  the data, the sidebar filters and the numbers that are worth keeping between clicks.

WHAT THIS FILE IS
    get_data / get_bounds     read the six cleaned tables ONCE per server (not on every click) and share them between pages
    report / facts            read one result file of Phases 4 to 6C (checked again if the file changes), or None
    require_reports           stops a page politely, naming the notebook to run, when its result files are missing
    sidebar_filters           draws the filter widgets in the sidebar and returns the choices as a FilterState
    build_context / current   one `Context` object per click: the selected rows, the rows of the chosen dates, the daily summary
                              of those dates, and (on demand) the song table and the artist table
    songs_for / artists_for   the song table and the artist table for one combination of filters, remembered between clicks

WHY WE NEED IT
    Streamlit runs the whole script again on every click. Without care, every click would read the files again and recompute every
    table. Here the tables are read once, each filter combination is calculated once, and every page just asks `current()`
    for "the selection".

    How the sidebar works: the widgets keep their values in `st.session_state` under fixed keys, so the choices survive when the
    person moves from page to page. "Reset filters" and the quick date ranges change those values from a callback (a callback runs
    before the widgets are drawn, which is the moment when Streamlit allows it).

WHAT HAPPENS WHEN IT RUNS
    `get_data` raises `DataNotReady` (a friendly message) when data/processed/ is empty; the entry file shows it and stops.
    Everything that is cached is only READ afterwards: pages must not change a shared table in place.

EXPECTED OUTCOME
    A click on a filter recomputes the selected rows (a few milliseconds) and whatever the open page needs, nothing else.

MOTIVE
    Fast clicks, and one place that knows what "the selection" is.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from app.ui.components import stretch_kwargs
from src.config import CHART_SIZE, default_paths
from src.dash import filters as F
from src.dash import metrics as M
from src.dash.load import Bounds, Tables, chart_bounds, load_tables, read_facts, read_report

# ------------------------------------------------------------------------------------------------------ widget keys
K_PRESET, K_DATES, K_RANKS, K_TYPES = "f_preset", "f_dates", "f_ranks", "f_types"
K_EXPLICIT, K_COLLAB, K_ARTISTS, K_SONGS = "f_explicit", "f_collab", "f_artists", "f_songs"
FILTER_KEYS = (K_PRESET, K_DATES, K_RANKS, K_TYPES, K_EXPLICIT, K_COLLAB, K_ARTISTS, K_SONGS)
TYPE_CHOICES = ["Single", "Album", "Compilation"]
EXPLICIT_CHOICES = ["All", "Explicit", "Not explicit"]
COLLAB_CHOICES = ["All", "Collaborations", "Solo"]
CTX_KEY = "_ctx"

# which notebook creates which result files (used in the "not built yet" message)
NOTEBOOK_OF = {"p4": "notebooks/04_eda_playlist_song_artist.ipynb", "p5": "notebooks/05_hypothesis_testing.ipynb",
               "p6a": "notebooks/06a_supervised_models.ipynb", "p6b": "notebooks/06b_survival_analysis.ipynb",
               "p6c": "notebooks/06c_clustering.ipynb"}


# ----------------------------------------------------------------------------------------------------- shared data
@st.cache_resource(show_spinner="Reading the chart data")
def get_data() -> Tables:
    """The six cleaned tables. Read once per server and shared by every page; never change them in place."""
    return load_tables()


@st.cache_resource(show_spinner=False)
def get_bounds() -> Bounds:
    return chart_bounds(get_data().playlist)


def _mtime(name: str, suffix: str) -> float:
    path = default_paths().reports_dir / f"{name}{suffix}"
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False, max_entries=64)
def _report(name: str, mtime: float, dates: tuple):
    return read_report(name, dates=dates)


@st.cache_data(show_spinner=False, max_entries=16)
def _facts(name: str, mtime: float):
    return read_facts(name)


def report(name: str, dates: tuple = ()) -> "pd.DataFrame | None":
    """A result file of reports/ (for example "p6b_stays"), or None when it has not been created. Read again when the file changes."""
    return _report(name, _mtime(name, ".csv"), tuple(dates))


def facts(name: str) -> dict:
    """The facts file of a phase (for example "p6c_facts"), or an empty dictionary."""
    return _facts(name, _mtime(name, ".json"))


def require_reports(*names: str) -> dict:
    """
    Read the named result files. If any is missing, show a calm message that names the notebook that creates it and stop the page.
    Returns {name: table}.
    """
    tables = {name: report(name) for name in names}
    missing = [name for name, table in tables.items() if table is None]
    if missing:
        notebooks = sorted({NOTEBOOK_OF.get(name.split("_")[0], "the notebooks") for name in missing})
        st.warning(f"This page needs the result file{'s' if len(missing) > 1 else ''} {', '.join(f'reports/{m}.csv' for m in missing)}, "
                   f"which {'are' if len(missing) > 1 else 'is'} created by {' and '.join(notebooks)}. Run the notebook "
                   f"(Run All), then refresh this page. The other pages do not need it.")
        st.stop()
    return tables


# ------------------------------------------------------------------------------------------------ the sidebar
def _presets(bounds: Bounds) -> dict:
    """The quick date ranges: name -> (first day, last day)."""
    first, last = bounds.first_date.date(), bounds.last_date.date()
    presets = {"Whole period": (first, last), "Last 30 days": (max(first, last - timedelta(days=29)), last),
               "Last 90 days": (max(first, last - timedelta(days=89)), last), "Last 6 months": (max(first, last - timedelta(days=182)), last)}
    for year in range(first.year, last.year + 1):
        presets[f"Calendar year {year}"] = (max(first, date(year, 1, 1)), min(last, date(year, 12, 31)))
    return presets


def _apply_preset(bounds: Bounds) -> None:
    choice = st.session_state.get(K_PRESET)
    span = _presets(bounds).get(choice)
    if span:
        st.session_state[K_DATES] = span


def _mark_custom() -> None:
    st.session_state[K_PRESET] = "Custom"


def _reset(bounds: Bounds) -> None:
    for key, value in _defaults(bounds).items():
        st.session_state[key] = value


def _defaults(bounds: Bounds) -> dict:
    return {K_PRESET: "Whole period", K_DATES: (bounds.first_date.date(), bounds.last_date.date()), K_RANKS: (1, CHART_SIZE),
            K_TYPES: [], K_EXPLICIT: "All", K_COLLAB: "All", K_ARTISTS: [], K_SONGS: []}


def state_from_session(bounds: Bounds) -> F.FilterState:
    """The FilterState that the widget values in the session describe (safe when a value is missing or empty)."""
    ss, defaults = st.session_state, _defaults(bounds)
    raw = ss.get(K_DATES)
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        dates = tuple(raw)
    elif isinstance(raw, (tuple, list)) and len(raw) == 1:          # the slider handle was dragged onto one day
        dates = (raw[0], raw[0])
    elif raw is None or isinstance(raw, (tuple, list)):
        dates = defaults[K_DATES]
    else:
        dates = (raw, raw)
    low, high = ss.get(K_RANKS) or defaults[K_RANKS]
    return F.FilterState.create(
        dates[0], dates[1], low, high, [t.lower() for t in (ss.get(K_TYPES) or [])],
        {"Explicit": "explicit", "Not explicit": "clean"}.get(ss.get(K_EXPLICIT), "all"),
        {"Collaborations": "collab", "Solo": "solo"}.get(ss.get(K_COLLAB), "all"),
        ss.get(K_ARTISTS) or [], ss.get(K_SONGS) or [])


@st.cache_data(show_spinner=False)
def _artist_names() -> list:
    data = get_data()
    return F.artist_options(data.bridge, data.dim_artist)


@st.cache_data(show_spinner=False)
def _song_labels() -> dict:
    return F.song_options(get_data().dim_track)


def sidebar_filters(bounds: Bounds) -> F.FilterState:
    """Draw the filter widgets in the sidebar and return the choices as a FilterState."""
    for key, value in _defaults(bounds).items():
        st.session_state.setdefault(key, value)
    first, last = bounds.first_date.date(), bounds.last_date.date()
    labels = _song_labels()
    with st.sidebar:
        st.markdown("### Filters")
        st.selectbox("Quick date range", ["Custom"] + list(_presets(bounds)), key=K_PRESET, on_change=_apply_preset, args=(bounds,))
        st.slider("Dates", min_value=first, max_value=last, key=K_DATES, format="D MMM YYYY", on_change=_mark_custom,
                  help="The first and last day of the chart to look at.")
        st.slider("Chart places", 1, CHART_SIZE, key=K_RANKS,
                  help="Place 1 is the top of the chart, place 50 is the bottom. Narrow it to look at the top 10 only, or at the lower half.")
        st.pills("Release type", TYPE_CHOICES, selection_mode="multi", key=K_TYPES, help="None chosen means all release types.")
        st.segmented_control("Explicit content", EXPLICIT_CHOICES, key=K_EXPLICIT)
        st.segmented_control("Collaboration", COLLAB_CHOICES, key=K_COLLAB, help="A collaboration is a song with more than one credited artist.")
        st.multiselect("Artists", _artist_names(), key=K_ARTISTS, placeholder="All artists",
                       help="Keeps the songs on which any of these artists is credited (a feature credit counts).")
        st.multiselect("Songs", list(labels), key=K_SONGS, format_func=lambda i: labels.get(i, str(i)), placeholder="All songs")
        st.button("Reset filters", on_click=_reset, args=(bounds,), **stretch_kwargs(st.button))
    return state_from_session(bounds)


# ------------------------------------------------------------------------------------------------------ the context
@st.cache_data(show_spinner=False, max_entries=40)
def songs_for(key: tuple) -> pd.DataFrame:
    """The song table for one combination of filters (remembered between clicks)."""
    data = get_data()
    return M.song_table(F.apply_filters(data.playlist, F.FilterState.from_key(key), data.bridge))


@st.cache_data(show_spinner=False, max_entries=40)
def artists_for(key: tuple, basis: str = "primary") -> pd.DataFrame:
    """The artist table (ADI) for one combination of filters and one basis ("primary" or "credits")."""
    data = get_data()
    state = F.FilterState.from_key(key)
    window = F.keep_dates(data.playlist, state)
    rows = F.apply_filters(data.playlist, state, data.bridge)
    return M.artist_table(rows, M.total_chart_points(window), int(window["date"].nunique()), basis, data.bridge)


@dataclass
class Context:
    """Everything a page needs to know about the current selection."""
    data: Tables
    bounds: Bounds
    state: F.FilterState
    rows: pd.DataFrame          # the selected rows (every filter applied)
    window: pd.DataFrame        # every row of the chosen dates (the whole chart; the denominator of the ADI)
    summary: pd.DataFrame       # daily_summary of the chosen dates (chart-wide measures)

    def songs(self) -> pd.DataFrame:
        return songs_for(self.state.key())

    def artists(self, basis: str = "primary") -> pd.DataFrame:
        return artists_for(self.state.key(), basis)

    @property
    def chips(self) -> list:
        return F.describe(self.state, self.bounds)

    @property
    def n_days(self) -> int:
        return int(self.window["date"].nunique())

    @property
    def total_points(self) -> float:
        return M.total_chart_points(self.window)

    @property
    def calendar(self) -> pd.DataFrame:
        return self.data.daily_summary[["date", "snapshot_index"]]


def build_context(bounds: Bounds, state: F.FilterState) -> Context:
    data = get_data()
    return Context(data=data, bounds=bounds, state=state, rows=F.apply_filters(data.playlist, state, data.bridge),
                   window=F.keep_dates(data.playlist, state), summary=F.keep_dates(data.daily_summary, state))


def current() -> Context:
    """The context of the current click. When a page runs without the entry file (in tests), the whole chart is used."""
    ctx = st.session_state.get(CTX_KEY)
    if ctx is None:
        bounds = get_bounds()
        ctx = build_context(bounds, F.default_state(bounds))
        st.session_state[CTX_KEY] = ctx
    return ctx


def selected_note(ctx: Context) -> str:
    """'12,345 of 27,750 song-days selected' for the sidebar footer."""
    return f"{len(ctx.rows):,} of {len(ctx.data.playlist):,} song-days selected"
