"""
tests/test_app_pages.py  -  all 18 dashboard pages, run headlessly on the real data with many filter choices.

WHAT THIS FILE IS
    A browser-free run of every page with Streamlit's own test runner (AppTest). Each page is opened with the whole chart and then
    pushed through nine filter choices in a row: last 30 days, top 10 and explicit only, singles and solo songs in the lower half, one
    artist, three songs, one single day, a day with no chart at all, a selection that is completely empty, and back to the whole chart.
    The same page object is kept between the choices, so a page that keeps an old pick (a song that the new filters no longer contain)
    is caught too (Streamlit 1.50's test runner cannot run one page object twice, so there each choice gets a new page object and the
    tests that move a widget are skipped). More tests open the entry file, move the sidebar widgets, press "Reset filters", and run
    every page with the result files missing and with the cleaned tables missing.

WHY WE NEED IT
    The sidebar filters can produce any selection, including an empty one. A page that is fine on the whole chart can still divide by
    zero on a single day or raise an error on an empty table. Opening 18 pages by hand with nine filter choices each is more than
    anyone will do before a presentation; the test does it in a few minutes.

WHAT HAPPENS WHEN IT RUNS
    The real tables in data/processed and the result files in reports/ are read (never changed). The tests are skipped when the cleaned
    tables have not been built. No server is started and nothing is written. The result is a list of the pages and choices that raised an
    exception or an error box, or that lost their header.

EXPECTED OUTCOME
    All tests pass with Streamlit 1.50 and with the newest version.

MOTIVE
    Every page, with every filter, on the first click of the person who opens the dashboard.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import nav  # noqa: E402
from app.ui import state  # noqa: E402
from src.dash import filters as F  # noqa: E402
from src.dash.load import DataNotReady, chart_bounds  # noqa: E402

pytestmark = pytest.mark.usefixtures("dash_real")      # skipped (not failed) when data/processed has not been built

ROOT = Path(__file__).resolve().parents[1]
ENTRY = str(ROOT / "app" / "streamlit_app.py")           # an absolute path: newer Streamlit versions resolve a relative one differently
MODULES = [entry[3] for entry in nav.PAGES]

# One page inside the same shell as the entry file (theme, sidebar filters, context), without the menu.
PAGE_SCRIPT = """
import importlib, sys
sys.path.insert(0, __ROOT__)
import streamlit as st
from app.ui import state, theme
theme.register_template()
theme.inject_css()
bounds = state.get_bounds()
filters = state.sidebar_filters(bounds)
st.session_state[state.CTX_KEY] = state.build_context(bounds, filters)
importlib.import_module("app.views.__MODULE__").render()
"""


def run_page(module: str, **session_values) -> AppTest:
    script = PAGE_SCRIPT.replace("__ROOT__", repr(str(ROOT))).replace("__MODULE__", module)
    at = AppTest.from_string(script, default_timeout=300)
    for key, value in session_values.items():
        at.session_state[key] = value
    at.run()
    return at


@pytest.fixture(scope="module")
def can_rerun() -> bool:
    """
    Can this Streamlit version's test runner run a page with a single-choice segmented control (the Explicit and Collaboration
    filters) a second time? Streamlit 1.50 cannot (its test runner raises a ValueError while it collects the widget values; the
    page itself is fine in a browser); newer versions can. The probe uses the same kind of widget and the same session value.
    """
    # The choices have more than one letter on purpose: the old runner walks through the letters of the chosen text, so a one-letter
    # choice would hide the fault.
    probe = AppTest.from_string("import streamlit as st\nst.session_state.setdefault('k', 'alpha')\nst.segmented_control('x', ['alpha', 'beta'], key='k')\n")
    try:
        probe.run()
        probe.run()
    except Exception:  # noqa: BLE001 - any failure of the probe means "not possible with this version"
        return False
    return not probe.exception


@pytest.fixture
def rerunnable(can_rerun):
    """Skips a test that must change a widget and run the same page again, when the test runner of this Streamlit version cannot."""
    if not can_rerun:
        pytest.skip("the test runner of this Streamlit version cannot run a page with a single-choice segmented control twice")


def problems(at: AppTest, label: str) -> list:
    """What went wrong on this run: an exception, an error box, or a page without its header."""
    found = [f"[{label}] exception: {e.value[:400]}" for e in at.exception]
    found += [f"[{label}] error box: {e.value[:300]}" for e in at.error]
    if not any('class="pg-title"' in m.value for m in at.markdown):
        found.append(f"[{label}] the page header is missing")
    return found


# ================================================================================================================ the choices
def filter_state(values: dict) -> F.FilterState:
    """The FilterState that the widget values describe (the same mapping as the sidebar, for counting rows without a browser)."""
    start, end = values["f_dates"]
    low, high = values["f_ranks"]
    return F.FilterState.create(start, end, low, high, [t.lower() for t in values["f_types"]],
                                {"Explicit": "explicit", "Not explicit": "clean"}.get(values["f_explicit"], "all"),
                                {"Collaborations": "collab", "Solo": "solo"}.get(values["f_collab"], "all"),
                                values["f_artists"], values["f_songs"])


@pytest.fixture(scope="module")
def choices(dash_real) -> SimpleNamespace:
    data = dash_real
    bounds = chart_bounds(data.playlist)
    first, last = bounds.first_date.date(), bounds.last_date.date()
    days = sorted(pd.to_datetime(data.daily_summary["date"]).dt.date.unique())
    middle = days[len(days) // 2]
    defaults = {"f_preset": "Whole period", "f_dates": (first, last), "f_ranks": (1, 50), "f_types": [], "f_explicit": "All", "f_collab": "All",
                "f_artists": [], "f_songs": []}
    artists = F.artist_options(data.bridge, data.dim_artist)
    songs = list(F.song_options(data.dim_track))
    scenarios = {
        "last 30 days": {"f_preset": "Last 30 days", "f_dates": (max(first, last - timedelta(days=29)), last)},
        "top 10, explicit only": {"f_ranks": (1, 10), "f_explicit": "Explicit"},
        "singles, solo, lower half": {"f_types": ["Single"], "f_collab": "Solo", "f_ranks": (26, 50)},
        "one artist": {"f_artists": [artists[0]]},
        "three songs": {"f_songs": songs[:3]},
        "one single day": {"f_preset": "Custom", "f_dates": (middle, middle)},
    }
    gaps = sorted(set(pd.date_range(first, last).date) - set(days))          # a calendar day on which the playlist was not captured
    if gaps:
        scenarios["a day with no chart"] = {"f_preset": "Custom", "f_dates": (gaps[0], gaps[0])}
    credited = F.credited_track_ids(data.bridge, [artists[0]])
    foreign_song = next(track for track in songs if track not in credited)
    for candidate in ({"f_artists": [artists[0]], "f_songs": [foreign_song]},
                      {"f_ranks": (1, 1), "f_types": ["Compilation"], "f_explicit": "Explicit", "f_collab": "Collaborations"},
                      {"f_dates": (middle, middle), "f_ranks": (1, 1), "f_types": ["Compilation"]}):
        if len(F.apply_filters(data.playlist, filter_state({**defaults, **candidate}), data.bridge)) == 0:
            scenarios["an empty selection"] = candidate
            break
    scenarios["the whole chart again"] = {}
    return SimpleNamespace(bounds=bounds, defaults=defaults, scenarios=scenarios, days=days, gaps=gaps)


def test_the_choices_include_a_really_empty_selection(choices, dash_real):
    """If no combination of filters is empty, the empty-page path would silently go untested."""
    assert "an empty selection" in choices.scenarios
    values = {**choices.defaults, **choices.scenarios["an empty selection"]}
    assert len(F.apply_filters(dash_real.playlist, filter_state(values), dash_real.bridge)) == 0


def test_the_single_day_choice_is_one_day(choices, dash_real):
    values = {**choices.defaults, **choices.scenarios["one single day"]}
    window = F.keep_dates(dash_real.playlist, filter_state(values))
    assert window["date"].nunique() == 1


# ============================================================================================ every page, every filter choice
@pytest.mark.parametrize("module", MODULES)
def test_a_page_draws_through_every_filter_choice(module, choices, dash_real, can_rerun):
    at = run_page(module)
    found = problems(at, "the whole chart")
    for name, overrides in choices.scenarios.items():
        values = {**choices.defaults, **overrides}
        if can_rerun:                                   # the same page object: a pick that the new filters no longer contain is tested too
            for key, value in values.items():
                at.session_state[key] = value
            at.run()
        else:                                           # an older test runner cannot run one page object twice: a new page for each choice
            at = run_page(module, **values)
        found += problems(at, name)
        expected = len(F.apply_filters(dash_real.playlist, filter_state(values), dash_real.bridge))
        received = len(at.session_state[state.CTX_KEY].rows)                     # proof that the choice really reached the page
        if received != expected:
            found.append(f"[{name}] the page received {received} song-days, the filters select {expected}")
    assert not found, f"page '{module}':\n" + "\n".join(found)


def test_the_checker_notices_a_broken_page():
    """The tests above only mean something if an exception, an error box or a missing header is really reported."""
    broken = AppTest.from_string("import streamlit as st\nst.markdown('<div class=\"pg-title\">x</div>', unsafe_allow_html=True)\nraise RuntimeError('boom')").run()
    assert any("boom" in line for line in problems(broken, "a"))
    errored = AppTest.from_string("import streamlit as st\nst.error('bad')").run()
    found = problems(errored, "b")
    assert any("error box" in line for line in found) and any("header is missing" in line for line in found)


@pytest.mark.parametrize("module", MODULES)
def test_a_page_draws_at_least_one_chart_or_table_on_the_whole_chart(module):
    at = run_page(module)
    shown = len(at.get("plotly_chart")) + len(at.dataframe) + sum('class="minitable' in m.value or 'class="kpi' in m.value for m in at.markdown)
    assert shown >= 1, f"page '{module}' shows no chart, table or KPI card"


# ================================================================================================================ the sidebar
@pytest.mark.usefixtures("rerunnable")
def test_the_rank_slider_narrows_the_selection():
    at = run_page("overview")
    at.sidebar.slider(key="f_ranks").set_value((1, 10)).run()
    ctx = at.session_state[state.CTX_KEY]
    assert not at.exception
    assert ctx.rows["position"].min() >= 1 and ctx.rows["position"].max() <= 10
    assert "places 1 to 10" in ctx.chips


@pytest.mark.usefixtures("rerunnable")
def test_a_quick_date_range_sets_the_dates(choices):
    at = run_page("overview")
    at.sidebar.selectbox(key="f_preset").select("Last 30 days").run()
    start, end = at.session_state["f_dates"]
    assert not at.exception
    assert end == choices.bounds.last_date.date() and (end - start).days == 29
    assert at.session_state[state.CTX_KEY].n_days <= 30


@pytest.mark.usefixtures("rerunnable")
def test_moving_the_date_slider_switches_the_quick_range_to_custom(choices):
    at = run_page("overview")
    first = choices.bounds.first_date.date()
    at.sidebar.slider(key="f_dates").set_range(first, first + timedelta(days=40)).run()
    assert not at.exception
    assert at.session_state["f_preset"] == "Custom"
    assert at.session_state[state.CTX_KEY].window["date"].max().date() <= first + timedelta(days=40)


@pytest.mark.usefixtures("rerunnable")
def test_reset_filters_puts_every_filter_back(choices):
    chosen = {**choices.defaults, "f_preset": "Custom", "f_ranks": (1, 10), "f_explicit": "Explicit", "f_types": ["Album"], "f_collab": "Solo"}
    at = run_page("overview", **chosen)
    assert at.session_state[state.CTX_KEY].state.rank_high == 10
    next(button for button in at.sidebar.button if button.label == "Reset filters").click().run()
    assert not at.exception
    for key, value in choices.defaults.items():
        assert at.session_state[key] == value, key
    assert at.session_state[state.CTX_KEY].state.is_whole_chart(choices.bounds)


def test_every_filter_has_a_widget_in_the_sidebar():
    at = run_page("overview")
    for key in state.FILTER_KEYS:
        assert key in at.session_state, key
    assert len(at.sidebar.slider) == 2 and len(at.sidebar.multiselect) == 2 and len(at.sidebar.selectbox) == 1


# ================================================================================================================ the entry file
def test_the_entry_file_opens_the_overview_with_the_filters():
    at = AppTest.from_file(ENTRY, default_timeout=300).run()
    assert not at.exception
    assert any("pg-title" in m.value and "Executive summary" in m.value for m in at.markdown)
    for key in state.FILTER_KEYS:
        assert key in at.session_state, key
    assert any("song-days selected" in c.value for c in at.sidebar.caption)


@pytest.fixture
def no_data(monkeypatch):
    """The cleaned tables are 'not built': the first thing a person sees after copying the project without running the pipeline."""
    def not_built(folder=None):
        raise DataNotReady(["playlist_daily"], "data/processed")

    st.cache_resource.clear()
    monkeypatch.setattr(state, "load_tables", not_built)
    yield
    st.cache_resource.clear()


def test_the_entry_file_says_what_to_do_when_the_data_is_not_built(no_data):
    at = AppTest.from_file(ENTRY, default_timeout=120).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "not built yet" in text
    assert any("python -m src.etl.run_pipeline" in c.value for c in at.code)
    assert len(at.sidebar.slider) == 0                        # no filters are drawn without data


# ============================================================================================== missing result files (notebooks)
@pytest.fixture
def no_reports(monkeypatch):
    """None of the result files of the notebooks exists: the dashboard opened before the notebooks were run."""
    st.cache_data.clear()
    monkeypatch.setattr(state, "read_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(state, "read_facts", lambda *args, **kwargs: {})
    yield
    st.cache_data.clear()


@pytest.mark.parametrize("module", MODULES)
def test_a_page_without_the_result_files_still_opens(module, no_reports):
    at = run_page(module)
    assert not problems(at, module + " without result files")


@pytest.mark.parametrize("module, notebook", [("longevity", "06b_survival_analysis.ipynb"), ("archetypes", "06c_clustering.ipynb")])
def test_a_page_that_needs_a_result_file_names_the_notebook_that_creates_it(module, notebook, no_reports):
    at = run_page(module)
    warnings = [w.value for w in at.warning]
    assert any("needs the result file" in w and notebook in w for w in warnings), warnings
    assert not at.exception
