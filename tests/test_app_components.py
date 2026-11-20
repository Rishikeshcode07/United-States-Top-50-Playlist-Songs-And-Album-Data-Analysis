"""
tests/test_app_components.py  -  the building blocks of every dashboard page (app/ui/components.py).

WHAT THIS FILE IS
    Tests of the header, the KPI cards, the callout, the text cards, the fact list, both small tables, the badge, the chart and table
    wrappers, the CSV download and the "nothing selected" stop. Streamlit is replaced by a small recorder, so the tests see exactly
    the HTML or the keywords that would have been sent to the browser.

WHY WE NEED IT
    Song and artist names come from the data and can hold "&", "<" or quotes. The blocks must escape them, close every tag they open
    and keep the same class names that the style sheet uses. A page that breaks because of one odd title is a bug that no one finds by
    looking at the screen.

WHAT HAPPENS WHEN IT RUNS
    Each test calls one block with small hand-made input and reads what was written. Nothing is drawn, no server is started and no
    file is written. The tests are skipped when Streamlit is not installed.

EXPECTED OUTCOME
    All tests pass with Streamlit 1.50 and with the newest version.

MOTIVE
    The same safe header, cards and tables on all 18 pages.
"""
from __future__ import annotations

import functools
from html.parser import HTMLParser

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.ui import components as ui  # noqa: E402

NASTY = 'Tom & Jerry <script>alert("x")</script> \'quoted\''


# =============================================================================================================== recorders
class Page:
    """Replaces st.markdown: keeps every HTML text that a block writes."""

    def __init__(self):
        self.blocks: list = []

    def __call__(self, body, *args, **kwargs):
        assert kwargs.get("unsafe_allow_html") is True, "a block that writes HTML must say so"
        self.blocks.append(body)

    @property
    def html(self) -> str:
        assert len(self.blocks) == 1, f"expected one element, got {len(self.blocks)}"
        return self.blocks[0]


@pytest.fixture
def page(monkeypatch) -> Page:
    recorder = Page()
    monkeypatch.setattr(st, "markdown", recorder)
    return recorder


class Stopped(Exception):
    pass


class Balanced(HTMLParser):
    """Checks that every tag that is opened is closed in the right order (the page would look broken otherwise)."""
    VOID = {"br", "hr", "img", "input", "meta", "link"}

    def __init__(self):
        super().__init__()
        self.stack: list = []
        self.problems: list = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"</{tag}> closes {self.stack[-1] if self.stack else 'nothing'}")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()


def assert_well_formed(markup: str) -> None:
    parser = Balanced()
    parser.feed(markup)
    parser.close()
    assert not parser.problems and not parser.stack, (parser.problems, parser.stack)


def assert_no_raw_markup_from_data(markup: str) -> None:
    """The nasty title must never reach the page as live HTML."""
    assert "<script" not in markup and "</script" not in markup
    assert 'alert("x")' not in markup


# ================================================================================================================ escaping
def test_esc_escapes_every_character_that_can_break_html():
    assert ui.esc(NASTY) == "Tom &amp; Jerry &lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &#x27;quoted&#x27;"
    assert ui.esc(42) == "42" and ui.esc(None) == "None"


def test_rich_turns_double_stars_into_bold_after_escaping():
    assert ui.rich("**Drake** & <b>friends</b>") == "<b>Drake</b> &amp; &lt;b&gt;friends&lt;/b&gt;"
    assert ui.rich("a lone ** star") == "a lone ** star"
    assert ui.rich("**a** and **b**") == "<b>a</b> and <b>b</b>"


def test_a_name_with_double_stars_inside_cannot_open_a_tag():
    assert "<script" not in ui.rich("**<script>**")


# ==================================================================================================================== header
def test_page_header_has_kicker_title_subtitle_and_chips(page):
    ui.page_header("Playlist", "Timeline explorer", "Open the **Top 50** of any day.", chips=["Whole period", "Places 1 to 10", "Explicit only"])
    out = page.html
    assert 'class="pg-kicker">Playlist<' in out
    assert 'class="pg-title" role="heading" aria-level="1">Timeline explorer<' in out
    assert 'class="pg-sub">Open the <b>Top 50</b> of any day.<' in out
    assert out.count('<span class="chip') == 3
    assert 'class="chip muted">Whole period<' in out                      # the first chip (the period) is quieter than the filters
    assert 'class="chip">Places 1 to 10<' in out
    assert_well_formed(out)


def test_page_header_without_subtitle_or_chips_writes_neither(page):
    ui.page_header("Songs", "Song leaderboard")
    assert "pg-sub" not in page.html and "chips" not in page.html
    assert_well_formed(page.html)


def test_page_header_escapes_names_from_the_data(page):
    ui.page_header(NASTY, NASTY, NASTY, chips=[NASTY, NASTY])
    assert_no_raw_markup_from_data(page.html)
    assert_well_formed(page.html)


def test_section_heading_with_and_without_a_note(page):
    ui.section("Leaders", "Ranked by **ADI**.")
    assert 'class="sec-title">Leaders<' in page.html and 'class="sec-note">Ranked by <b>ADI</b>.<' in page.html
    assert_well_formed(page.html)
    page.blocks.clear()
    ui.section(NASTY)
    assert "sec-note" not in page.html
    assert_no_raw_markup_from_data(page.html)


# ================================================================================================================ KPI cards
def test_kpi_cards_show_label_value_unit_context_and_help(page):
    ui.kpi_cards([{"label": "Days on Chart", "value": "27", "unit": "days", "sub": "median **12**", "tone": "orange", "help": "Number of days with a place."},
                  {"label": "Average Rank", "value": "21.4"}])
    out = page.html
    assert out.count('class="kpi"') == 2 and 'class="kpis"' in out
    assert "--accent:#eb6834" in out and "--accent:#2a78d6" in out                  # orange, and blue as the default
    assert '<div class="kpi-label">Days on Chart</div>' in out
    assert '<div class="kpi-value">27<small>days</small></div>' in out
    assert '<div class="kpi-sub">median <b>12</b></div>' in out
    assert 'title="Number of days with a place."' in out
    assert_well_formed(out)


def test_kpi_cards_fall_back_to_blue_for_an_unknown_tone(page):
    ui.kpi_cards([{"label": "x", "value": "1", "tone": "purple-ish"}])
    assert "--accent:#2a78d6" in page.html


def test_a_long_kpi_value_gets_a_smaller_font_class(page):
    ui.kpi_cards([{"label": "Points", "value": "1,275,000.5"}, {"label": "Rank", "value": "12.5"}])
    assert page.html.count("kpi-value long") == 1


def test_kpi_cards_can_fix_the_number_of_columns(page):
    ui.kpi_cards([{"label": "a", "value": "1"}], columns=3)
    assert 'class="kpis cols-3"' in page.html


def test_kpi_cards_escape_the_text_of_every_field(page):
    ui.kpi_cards([{"label": NASTY, "value": NASTY, "sub": NASTY, "unit": NASTY, "help": NASTY}])
    assert_no_raw_markup_from_data(page.html)
    assert_well_formed(page.html)


# ===================================================================================================== callout and cards
def test_callout_has_the_default_title_and_bold_text(page):
    ui.callout("**Rock** is up 3 places.")
    assert page.html == '<div class="callout"><div class="callout-title">What to notice</div><b>Rock</b> is up 3 places.</div>'


def test_a_warning_callout_has_its_own_class_and_title(page):
    ui.callout("Few songs.", title="Careful", tone="warn")
    assert 'class="callout warn"' in page.html and ">Careful<" in page.html


def test_callout_escapes_a_song_name(page):
    ui.callout(f"The leader is **{NASTY}**.", title=NASTY)
    assert_no_raw_markup_from_data(page.html)
    assert_well_formed(page.html)


def test_cards_share_the_width_and_show_the_tag_only_when_there_is_one(page):
    ui.cards([{"title": "One", "text": "First **finding**.", "tag": "H1"}, {"title": "Two", "text": "Second."}], min_width=320)
    out = page.html
    assert 'style="--card-min:320px"' in out and out.count('class="card"') == 2
    assert out.count('class="tag"') == 1 and "<b>finding</b>" in out
    assert_well_formed(out)


def test_cards_escape_title_text_and_tag(page):
    ui.cards([{"title": NASTY, "text": NASTY, "tag": NASTY}])
    assert_no_raw_markup_from_data(page.html)
    assert_well_formed(page.html)


# ==================================================================================================================== facts
def test_facts_is_a_definition_list(page):
    ui.facts([("First day", "18 May 2024"), ("Songs", "832")], title="Profile")
    out = page.html
    assert out.startswith('<dl class="facts">') and 'class="facts-title">Profile<' in out
    assert "<dt>First day</dt><dd>18 May 2024</dd>" in out and "<dt>Songs</dt><dd>832</dd>" in out
    assert_well_formed(out)


def test_wide_facts_are_marked_for_sentences(page):
    ui.facts([("Rule", "A sentence that is longer than a number.")], wide=True)
    assert '<dl class="facts wide">' in page.html and "facts-title" not in page.html


def test_facts_escape_labels_and_values(page):
    ui.facts([(NASTY, NASTY)], title=NASTY)
    assert_no_raw_markup_from_data(page.html)
    assert_well_formed(page.html)


# ================================================================================================================== tables
def test_html_table_escapes_text_but_keeps_raw_html(page):
    ui.html_table(["Name", "Status"], [[NASTY, ui.Raw(ui.badge("PASS", "pass"))], ["plain", "text <b>"]])
    out = page.html
    assert_no_raw_markup_from_data(out)
    assert '<span class="badge badge-pass">PASS</span>' in out                       # Raw: written by the app, shown as HTML
    assert "text &lt;b&gt;" in out                                                  # everything else: shown as text
    assert out.count("<tr>") == 3 and out.count("<th ") == 2                         # one header row and two body rows; two header cells
    assert_well_formed(out)


def test_html_table_aligns_numbers_and_keeps_short_labels_on_one_line(page):
    ui.html_table(["Label", "Sentence", "Number"], [["a", "long text", "12"]], numeric_cols=(2,), nowrap_cols=(0,))
    out = page.html
    assert '<th class="num">Number</th>' in out and '<td class="num">12</td>' in out
    assert '<td class="nw">a</td>' in out and '<td class="">long text</td>' in out


def test_html_table_scrolls_inside_its_own_frame_only_when_asked(page):
    ui.html_table(["a"], [["1"]])
    assert 'class="minitable-wrap"' in page.html and "max-height" not in page.html
    page.blocks.clear()
    ui.html_table(["a"], [["1"]], max_height=240)
    assert 'class="minitable-wrap scroll" style="max-height:240px"' in page.html


def test_html_table_without_rows_still_has_a_header(page):
    ui.html_table(["a", "b"], [])
    assert "<th" in page.html and "<tr></tr>" not in page.html.split("<tbody>")[1]
    assert_well_formed(page.html)


def test_group_table_draws_a_swatch_per_group_and_a_note_for_a_small_group(page):
    ui.group_table(["Group", "Songs", "Median"], [("#2a78d6", "Explicit", ["120", "14"]), (None, "Not explicit", ["300", "9"]), ("#eb6834", "Tiny", "too few songs")])
    out = page.html
    assert out.count('class="sw"') == 2 and 'background:#2a78d6' in out
    assert '<td class="note" colspan="2">too few songs</td>' in out
    assert_well_formed(out)


def test_group_table_escapes_names_and_colours(page):
    ui.group_table(["Group", "n"], [('red" onload="x', NASTY, [NASTY])])
    assert_no_raw_markup_from_data(page.html)
    assert 'onload="x' not in page.html                                           # the quote of the colour cannot close the attribute
    assert_well_formed(page.html)


def test_badge_has_a_class_for_its_tone_and_escapes_the_text():
    assert ui.badge("PASS", "pass") == '<span class="badge badge-pass">PASS</span>'
    assert ui.badge("<x>", "warn") == '<span class="badge badge-warn">&lt;x&gt;</span>'
    assert ui.badge("n/a") == '<span class="badge badge-info">n/a</span>'


# ======================================================================================= keywords for Streamlit elements
def test_stretch_kwargs_uses_the_new_width_keyword_when_the_function_has_one():
    def newer(data, width: "Width" = "content"):                                    # noqa: F821  (only the annotation text matters)
        return None

    assert ui.stretch_kwargs(newer) == {"width": "stretch"}


def test_stretch_kwargs_uses_use_container_width_on_older_versions():
    def older(data, use_container_width: bool = False):
        return None

    assert ui.stretch_kwargs(older) == {"use_container_width": True}


def test_stretch_kwargs_for_a_function_without_either_keyword():
    def neither(data):
        return None

    assert ui.stretch_kwargs(neither) == {}


@pytest.mark.parametrize("element", ["plotly_chart", "dataframe", "button"])
def test_stretch_kwargs_has_a_working_keyword_for_the_installed_streamlit(element):
    function = getattr(st, element)
    keywords = ui.stretch_kwargs(function)
    assert len(keywords) == 1
    assert next(iter(keywords)) in {"width", "use_container_width"}
    import inspect
    assert next(iter(keywords)) in inspect.signature(function).parameters


def _recording(monkeypatch, name: str) -> list:
    """Replace st.<name> by a recorder that keeps the signature of the original (stretch_kwargs reads it)."""
    calls: list = []
    original = getattr(st, name)

    @functools.wraps(original)
    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return {"recorded": name}

    monkeypatch.setattr(st, name, recorder)
    return calls


def test_plot_sends_the_figure_with_the_dashboard_toolbar(monkeypatch):
    calls = _recording(monkeypatch, "plotly_chart")
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    result = ui.plot(fig, key="chart-1", height=480)
    (args, kwargs), = calls
    assert args == (fig,) and fig.layout.height == 480
    assert kwargs["key"] == "chart-1" and kwargs["config"] is ui.PLOT_CONFIG
    assert kwargs.get("width") == "stretch" or kwargs.get("use_container_width") is True
    assert "on_select" not in kwargs
    assert result == {"recorded": "plotly_chart"}


def test_plot_toolbar_has_no_logo_and_no_selection_tools():
    assert ui.PLOT_CONFIG["displaylogo"] is False
    assert {"lasso2d", "select2d"} <= set(ui.PLOT_CONFIG["modeBarButtonsToRemove"])
    assert ui.PLOT_CONFIG["toImageButtonOptions"]["format"] == "png"


def test_plot_with_select_returns_the_click_event(monkeypatch):
    calls = _recording(monkeypatch, "plotly_chart")
    ui.plot(go.Figure(), key="k", select=True)
    kwargs = calls[0][1]
    assert kwargs["on_select"] == "rerun" and tuple(kwargs["selection_mode"]) == ("points",)


def test_plot_leaves_the_height_of_the_figure_alone_when_none_is_given(monkeypatch):
    _recording(monkeypatch, "plotly_chart")
    fig = go.Figure()
    fig.update_layout(height=333)
    ui.plot(fig)
    assert fig.layout.height == 333


def test_table_hides_the_index_and_fills_the_width(monkeypatch):
    calls = _recording(monkeypatch, "dataframe")
    frame = pd.DataFrame({"a": [1, 2, 3]})
    ui.table(frame, key="t")
    (args, kwargs), = calls
    assert args[0] is frame and kwargs["hide_index"] is True and kwargs["key"] == "t"
    assert "height" not in kwargs and "on_select" not in kwargs
    assert kwargs.get("width") == "stretch" or kwargs.get("use_container_width") is True


def test_table_is_never_taller_than_its_rows(monkeypatch):
    calls = _recording(monkeypatch, "dataframe")
    ui.table(pd.DataFrame({"a": [1, 2, 3]}), height=1000)
    assert calls[0][1]["height"] == 35 * 4 + 3
    ui.table(pd.DataFrame({"a": range(100)}), height=400)
    assert calls[1][1]["height"] == 400


def test_table_can_show_the_index_and_let_the_person_click_a_row(monkeypatch):
    calls = _recording(monkeypatch, "dataframe")
    ui.table(pd.DataFrame({"a": [1]}), index=True, select=True)
    kwargs = calls[0][1]
    assert kwargs["hide_index"] is False and kwargs["on_select"] == "rerun" and kwargs["selection_mode"] == "single-row"


# ================================================================================================================ download
def test_download_csv_opens_correctly_in_excel(monkeypatch):
    calls: list = []
    monkeypatch.setattr(st, "download_button", lambda *args, **kwargs: calls.append((args, kwargs)))
    ui.download_csv(pd.DataFrame({"song": ["Déjà vu", "A, B"], "n": [1, 2]}), "songs.csv", key="dl")
    (args, kwargs), = calls
    label, data = args
    assert label == "Download this table (CSV)" and kwargs == {"file_name": "songs.csv", "mime": "text/csv", "key": "dl"}
    assert data.startswith(b"\xef\xbb\xbf")                                          # the byte-order mark that makes Excel read UTF-8
    text = data.decode("utf-8-sig")
    assert text.splitlines()[0] == "song,n" and "Déjà vu" in text and '"A, B"' in text


def test_download_csv_has_no_index_column(monkeypatch):
    calls: list = []
    monkeypatch.setattr(st, "download_button", lambda *args, **kwargs: calls.append(args))
    ui.download_csv(pd.DataFrame({"x": [5]}, index=[7]), "x.csv")
    assert calls[0][1].decode("utf-8-sig") == "x\n5\n" or calls[0][1].decode("utf-8-sig").replace("\r\n", "\n") == "x\n5\n"


# ================================================================================================================ stop page
@pytest.fixture
def stopper(monkeypatch):
    def stop():
        raise Stopped()

    monkeypatch.setattr(st, "stop", stop)


@pytest.mark.parametrize("empty", [pd.DataFrame({"a": []}), None])
def test_an_empty_selection_shows_a_warning_and_stops_the_page(page, stopper, empty):
    with pytest.raises(Stopped):
        ui.stop_if_empty(empty)
    assert 'class="callout warn"' in page.html and "No songs in this selection" in page.html and "Reset filters" in page.html
    assert_well_formed(page.html)


def test_a_selection_with_rows_does_not_stop_the_page(page, stopper):
    ui.stop_if_empty(pd.DataFrame({"a": [1]}))
    assert page.blocks == []


def test_the_message_of_the_stop_can_be_chosen(page, stopper):
    with pytest.raises(Stopped):
        ui.stop_if_empty(pd.DataFrame(), "Pick at least two songs.")
    assert "Pick at least two songs." in page.html
