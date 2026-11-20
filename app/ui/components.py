"""
app/ui/components.py  -  the small building blocks that every page is made of.

WHAT THIS FILE IS
    page_header     kicker (the section), title, one-sentence subtitle and the chips that list the active filters
    kpi_cards       a row of KPI cards (label, big number, one line of context, an accent colour)
    section         a heading for a block of the page, with an optional note under it
    callout         the "What to notice" box: a short sentence or two, written from the numbers on the screen
    cards           a grid of text cards (used for the findings and the page guide)
    facts           a compact two-column list (label, value) without a table header
    group_table     a small table for the side of a chart, one row per group, with a colour swatch
    html_table      a table whose cells wrap (for sentences), optionally scrolling inside its own frame; `Raw` marks cells that are HTML
    badge           a small coloured status word (PASS, WARN, FAIL ...)
    plot            draws a Plotly figure with the toolbar of the dashboard
    table           draws a table with sensible defaults; download_csv offers the same table as a file
    stop_if_empty   shows a calm message and stops the page when the filters leave nothing to show
    stretch_kwargs  the keyword that makes a Streamlit element fill the width (it changed its name between versions)

WHY WE NEED IT
    Eighteen pages that each build their own HTML would drift apart. With these blocks a page reads like an outline:
    header, KPI row, chart, callout, table.

    Text safety: song and artist names come from the data and may contain "&", "<" or quotes. Every text that goes into the HTML
    passes through `rich()` or `esc()`, which escape it first (`rich` then turns **double stars** into bold). A name can
    therefore never break the page. The only text that is not escaped is a `Raw` value, which the app itself writes.

WHAT HAPPENS WHEN IT RUNS
    Each function writes one element to the page. Nothing is cached here.

EXPECTED OUTCOME
    The same header, cards, callouts and tables on every page.

MOTIVE
    Pages that are short, consistent and safe.
"""
from __future__ import annotations

import html
import inspect
import re

import pandas as pd
import streamlit as st

PLOT_CONFIG = {"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
               "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "atlantic_top50_chart"}}
_BOLD = re.compile(r"\*\*(.+?)\*\*")
TONES = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "pink": "#e87ba4", "green": "#008300",
         "violet": "#6250d6", "red": "#e34948", "ink": "#0b0b0b"}


# ------------------------------------------------------------------------------------------------------ text safety
def esc(text) -> str:
    """HTML-escape any value."""
    return html.escape(str(text), quote=True)


def rich(text) -> str:
    """Escape a text for HTML, then turn **double stars** into bold."""
    return _BOLD.sub(r"<b>\1</b>", esc(text))


def stretch_kwargs(func) -> dict:
    """The keyword that makes a Streamlit element fill the width: width="stretch" on current versions, use_container_width=True on older ones."""
    parameters = inspect.signature(func).parameters
    if "width" in parameters and "Width" in str(parameters["width"].annotation):
        return {"width": "stretch"}
    if "use_container_width" in parameters:
        return {"use_container_width": True}
    return {}


# --------------------------------------------------------------------------------------------------------- header
def page_header(kicker: str, title: str, subtitle: str = "", chips=None) -> None:
    """The top of a page. `chips` is the list of phrases that describe the active filters (see src.dash.filters.describe)."""
    parts = [f'<div class="pg-kicker">{esc(kicker)}</div>', f'<div class="pg-title" role="heading" aria-level="1">{esc(title)}</div>']
    if subtitle:
        parts.append(f'<div class="pg-sub">{rich(subtitle)}</div>')
    if chips:
        parts.append('<div class="chips">' + "".join(
            f'<span class="chip{" muted" if i == 0 else ""}">{esc(c)}</span>' for i, c in enumerate(chips)) + "</div>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    """A heading for one block of the page."""
    note_html = f'<div class="sec-note">{rich(note)}</div>' if note else ""
    st.markdown(f'<div class="sec"><div class="sec-title">{esc(title)}</div>{note_html}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------------ KPI cards
def kpi_cards(items, columns: "int | None" = None) -> None:
    """
    A responsive row of KPI cards. `columns` fixes the number of cards per row on a wide screen (3 per row on a narrower one);
    without it the cards fill the width as they fit. Every item is a dict with
        label  the KPI name (small capitals)          value  the big number, already formatted
        sub    one line of context (optional)         tone   blue / orange / aqua / yellow / pink / green / violet / red
        unit   a small unit after the number (optional)   help   a longer definition, shown when the mouse rests on the card
    """
    cards_html = []
    for item in items:
        accent = TONES.get(item.get("tone", "blue"), TONES["blue"])
        unit = f"<small>{esc(item['unit'])}</small>" if item.get("unit") else ""
        sub = f'<div class="kpi-sub">{rich(item["sub"])}</div>' if item.get("sub") else ""
        tip = f' title="{esc(item["help"])}"' if item.get("help") else ""
        cards_html.append(f'<div class="kpi" style="--accent:{accent}"{tip}><div class="kpi-label">{esc(item["label"])}</div>'
                          f'<div class="kpi-value{" long" if len(str(item["value"])) > 9 else ""}">{esc(item["value"])}{unit}</div>{sub}</div>')
    css = f"kpis cols-{int(columns)}" if columns else "kpis"
    st.markdown(f'<div class="{css}">' + "".join(cards_html) + "</div>", unsafe_allow_html=True)


def callout(text: str, title: str = "What to notice", tone: str = "info") -> None:
    """The reading note under a chart. `tone` = "info" (blue) or "warn" (yellow, for a caution)."""
    css = "callout warn" if tone == "warn" else "callout"
    st.markdown(f'<div class="{css}"><div class="callout-title">{esc(title)}</div>{rich(text)}</div>', unsafe_allow_html=True)


def cards(items, min_width: int = 290) -> None:
    """
    A grid of text cards. Every item is a dict with title, text and (optional) tag. The cards of a row share the width, so a last
    row with fewer cards is stretched instead of leaving a hole; `min_width` (pixels) decides how many cards fit in a row.
    """
    html_cards = []
    for item in items:
        tag = f'<span class="tag">{esc(item["tag"])}</span><br>' if item.get("tag") else ""
        html_cards.append(f'<div class="card">{tag}<h4>{esc(item["title"])}</h4><p>{rich(item["text"])}</p></div>')
    st.markdown(f'<div class="cards" style="--card-min:{int(min_width)}px">' + "".join(html_cards) + "</div>", unsafe_allow_html=True)


def facts(pairs, title: "str | None" = None, wide: bool = False) -> None:
    """
    A compact list of facts: `pairs` is a list of (label, value). Used for the side panel of the profile pages. `title` is a small heading inside the card.
    `wide=True` is for values that are whole sentences: the labels get a fixed column and the values are left-aligned, in a normal weight.
    """
    head = f'<div class="facts-title">{esc(title)}</div>' if title else ""
    body = "".join(f"<dt>{esc(label)}</dt><dd>{esc(value)}</dd>" for label, value in pairs)
    st.markdown(f'<dl class="facts{" wide" if wide else ""}">{head}{body}</dl>', unsafe_allow_html=True)


class Raw(str):
    """Text that is already HTML, written by this app and never taken from a file. Everything that is not `Raw` is escaped before it is shown."""


def html_table(headers, rows, numeric_cols=(), max_height: "int | None" = None, nowrap_cols=()) -> None:
    """
    A table whose cells wrap, for text that does not fit a data grid (a question, a sentence). `rows` is a list of lists of cells; a cell is plain
    text (escaped) or `Raw` HTML. The columns whose position is in `numeric_cols` are right-aligned and kept on one line; the ones in `nowrap_cols`
    are kept on one line and stay left-aligned (short labels). With `max_height` (pixels) a long table scrolls inside its own frame and keeps its
    header in view.
    """
    head = "".join(f'<th class="{"num" if i in numeric_cols else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = ""
    for row in rows:
        cells = "".join(f'<td class="{"num" if i in numeric_cols else "nw" if i in nowrap_cols else ""}">{cell if isinstance(cell, Raw) else esc(cell)}</td>'
                        for i, cell in enumerate(row))
        body += f"<tr>{cells}</tr>"
    frame = f'class="minitable-wrap scroll" style="max-height:{int(max_height)}px"' if max_height else 'class="minitable-wrap"'
    st.markdown(f'<div {frame}><table class="minitable wide"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


def group_table(headers, rows) -> None:
    """
    A small table for the side of a chart, one row per group. `headers` are the column titles (the first one is for the group name);
    `rows` is a list of (colour or None, name, values): `values` is a list with one text per remaining column, or one text that is
    shown as a note across the whole row (for a group that is too small to be measured).
    """
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = ""
    for color, name, values in rows:
        swatch = f'<span class="sw" style="background:{esc(color)}"></span>' if color else ""
        if isinstance(values, str):
            cells = f'<td class="note" colspan="{max(len(headers) - 1, 1)}">{esc(values)}</td>'
        else:
            cells = "".join(f"<td>{esc(v)}</td>" for v in values)
        body += f"<tr><td>{swatch}{esc(name)}</td>{cells}</tr>"
    st.markdown(f'<div class="minitable-wrap"><table class="minitable"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


def badge(text: str, tone: str = "info") -> str:
    """The HTML of a small status badge (tone = pass / warn / fail / info). Use it inside another HTML text."""
    return f'<span class="badge badge-{esc(tone)}">{esc(text)}</span>'


# ------------------------------------------------------------------------------------------------ charts, tables
def plot(fig, key=None, height=None, select: bool = False):
    """
    Draw a Plotly figure. `select=True` lets the person click a mark on the chart; the click comes back as the return value
    (`result.selection.points`), and the page reruns.
    """
    if height:
        fig.update_layout(height=height)
    kwargs = {"config": PLOT_CONFIG, "key": key, **stretch_kwargs(st.plotly_chart)}
    if select:
        kwargs.update(on_select="rerun", selection_mode=("points",))
    return st.plotly_chart(fig, **kwargs)


def table(df: pd.DataFrame, column_config=None, height=None, key=None, index: bool = False, select: bool = False):
    """
    A table with the dashboard's defaults (full width, no index column). With `select=True` the person can click one row; the result
    is then the selection event (`event.selection.rows` = the positions of the clicked rows), and the page reruns on every click.
    """
    kwargs = {"hide_index": not index, "column_config": column_config, "key": key, **stretch_kwargs(st.dataframe)}
    if height:
        kwargs["height"] = min(int(height), 35 * (len(df) + 1) + 3)      # never taller than the rows need (no empty row at the end)
    if select:
        kwargs.update(on_select="rerun", selection_mode="single-row")
    return st.dataframe(df, **kwargs)


def download_csv(df: pd.DataFrame, file_name: str, label: str = "Download this table (CSV)", key=None) -> None:
    """A download button for a table. The file opens correctly in Excel (UTF-8 with a byte-order mark, so that names like Beyonce with an accent survive)."""
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=file_name, mime="text/csv", key=key)


def stop_if_empty(frame: pd.DataFrame, message: str = "Nothing matches the current filters. Widen the dates or the place range in the sidebar, "
                  "or press Reset filters.") -> None:
    """If `frame` has no rows, say so calmly and stop the page here (the part of the page above stays)."""
    if frame is None or len(frame) == 0:
        callout(message, title="No songs in this selection", tone="warn")
        st.stop()
