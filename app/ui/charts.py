"""
app/ui/charts.py  -  the Plotly chart builders of the dashboard.

WHAT THIS FILE IS
    finish           gives any figure the dashboard's title, height, legend position and axis titles
    hbar             a horizontal bar chart, biggest bar on top, with value labels
    vbar             vertical bars (grouped or stacked)
    line_chart       one line per group over time (or over any x), with optional markers, area fill and smoothing
    heat             a heatmap from a table (rows x columns), with optional value labels
    scatter_chart    a scatter or bubble chart
    histogram_chart  a histogram, one colour per group, bars side by side or overlaid
    rank_axis        turns a y axis into a "place" axis: place 1 at the top, every 10th place labelled
    rank_lines       the chart place of several songs over time (one line each, gaps where the song was off the chart)
    rank_panels      the same, one small panel per song
    day_bars         the Top 50 of one day as bars, coloured by how each song moved, songs outside the filters dimmed
    slope_chart      the Top N of two days joined by lines (who rose, who fell, who arrived, who left)
    calendar_heat    a calendar of days (weeks across, weekdays down) coloured by one measure
    survival_chart   survival curves with their uncertainty bands
    lorenz_chart     the Lorenz curve of chart points (how unequal the shares are)
    pareto_chart     bars of each holder's share with the running total as a line (who holds the first 10, 20, 30 % ...)
    ribbon_chart     a middle line with a band around it (median and middle half)
    box_chart        box plots of one number for several groups
    span_chart       horizontal bars from a first day to a last day (when did each song stay on the chart)
    song_timeline    one song's chart place and popularity over time, in two panels that share the date axis
    log_axis         a logarithmic axis with readable ticks (1, 2, 5, 10, 20, 50 ...) for numbers that range from 1 to hundreds

WHY WE NEED IT
    The pages decide WHAT to show; these functions decide HOW it is drawn, so that all 18 pages share one look: the same title
    position, legend position, grid, hover box and number formats. All of them are pure functions (a table in, a figure out), so
    they can be tested without a browser.

WHAT HAPPENS WHEN IT RUNS
    Each function builds a Plotly figure with the project template (theme.register_template must have been called, as the entry
    file does). An empty table gives an empty figure with a short message in the middle instead of an error.

EXPECTED OUTCOME
    Figures that look the same whichever page made them.

MOTIVE
    Draw once, reuse everywhere.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.ui import theme


# ----------------------------------------------------------------------------------------------------- common parts
def empty_figure(message: str = "No data for the current filters", height: int = 320) -> go.Figure:
    """An empty figure with a message in the middle (used when the selection has no rows)."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=15, color=theme.MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height)
    return fig


def finish(fig: go.Figure, title: "str | None" = None, height: int = 360, legend: "str | None" = "top", x_title: "str | None" = None,
           y_title: "str | None" = None) -> go.Figure:
    """Title, height, legend position (top / bottom / right / None) and axis titles in the dashboard's style."""
    top = 56 if title else 24
    layout = dict(height=height, title=dict(text=title) if title else None, showlegend=legend is not None)
    if legend == "top":
        layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0)
        top += 26
    elif legend == "bottom":
        layout["legend"] = dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0)
    elif legend == "right":
        layout["legend"] = dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01)
    layout["margin"] = dict(l=14, r=20 if legend != "right" else 10, t=top, b=14)
    fig.update_layout(**layout)
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    return fig


def rank_axis(fig: go.Figure, low: int = 1, high: int = 50, row=None, col=None) -> go.Figure:
    """Make the y axis a chart-place axis: place 1 at the top, labelled at place 1 and then every 10 places."""
    step = 10 if high - low > 20 else 5
    ticks = sorted({int(low)} | {t for t in range(step, int(high) + 1, step) if t > low})             # for example 1, 10, 20, 30, 40, 50
    kwargs = dict(autorange=False, range=[high + 0.5, low - 0.5], tickmode="array", tickvals=ticks, title_text="Chart place")
    if row is not None:
        fig.update_yaxes(row=row, col=col, **kwargs)
    else:
        fig.update_yaxes(**kwargs)
    return fig


# ------------------------------------------------------------------------------------------------------------ bars
def percent_ticks(fig: go.Figure, values, axis: str = "y") -> None:
    """Axis ticks written as percentages (10%, 20%) for a chart whose values are shares between 0 and 1 (one decimal when all shares are small)."""
    values = np.asarray(values, dtype=float)
    top = float(np.nanmax(values)) if values.size and np.isfinite(values).any() else 1.0
    tickformat = ".0%" if top >= 0.1 else ".1%"
    (fig.update_xaxes if axis == "x" else fig.update_yaxes)(tickformat=tickformat)


def hbar(df: pd.DataFrame, label: str, value: str, title: "str | None" = None, color: str = theme.BLUE, colors=None, fmt: str = ",.1f",
         suffix: str = "", height: "int | None" = None, x_title: "str | None" = None, hover: "str | None" = None, customdata=None,
         text: "list | None" = None) -> go.Figure:
    """
    Horizontal bars, the first row of `df` on top. `colors` = one colour per row (or None for `color`), `fmt` = number format of the
    labels, `hover` = a Plotly hovertemplate and `customdata` the columns it can use.
    """
    if df.empty:
        return empty_figure()
    labels = df[label].astype(str).tolist()
    values = df[value].astype(float).tolist()
    labels_text = text if text is not None else [f"{v:{fmt}}{suffix}" for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors if colors is not None else color), text=labels_text,
                           textposition="outside", cliponaxis=False, textfont=dict(size=12, color=theme.INK_2),
                           customdata=None if customdata is None else df[customdata].to_numpy(), hovertemplate=hover or "%{y}: %{x:" + fmt + "}" + suffix + "<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="", tickfont=dict(size=12, color=theme.INK))
    if values and min(values) >= 0 and max(values) > 0:
        fig.update_xaxes(range=[0, max(values) * 1.17], showgrid=True)                  # head room, so that the label of the longest bar is not cut off
    else:
        fig.update_xaxes(rangemode="tozero", showgrid=True)
    fig.update_layout(bargap=0.28)
    if fmt.endswith("%"):
        percent_ticks(fig, values, axis="x")
    return finish(fig, title, height or max(220, 26 * len(df) + 90), legend=None, x_title=x_title)


def vbar(df: pd.DataFrame, x: str, y, title: "str | None" = None, colors: "dict | None" = None, names: "dict | None" = None, barmode: str = "group",
         height: int = 360, y_title: "str | None" = None, x_title: "str | None" = None, fmt: str = ",.0f", suffix: str = "", legend: "str | None" = "top",
         show_text: bool = False, hover: "str | None" = None, downward: "list | None" = None) -> go.Figure:
    """
    Vertical bars. `y` is one column name or a list of column names (one series each); barmode = group / stack / relative.
    A column listed in `downward` holds negative numbers that are drawn below the axis but shown as positive numbers in the hover box.
    """
    if df.empty:
        return empty_figure()
    columns = [y] if isinstance(y, str) else list(y)
    palette = colors or {}
    fig = go.Figure()
    for i, column in enumerate(columns):
        name = (names or {}).get(column, column)
        color = palette.get(column) or palette.get(name) or theme.CATEGORICAL[i % len(theme.CATEGORICAL)]
        flip = column in (downward or [])
        shown = f"%{{customdata:{fmt}}}" if flip else f"%{{y:{fmt}}}"
        fig.add_bar(x=df[x], y=df[column], name=name, marker_color=color, text=df[column] if show_text else None, customdata=df[column].abs() if flip else None,
                    texttemplate=f"%{{text:{fmt}}}{suffix}" if show_text else None, textposition="outside" if show_text and barmode == "group" else "auto",
                    cliponaxis=False, hovertemplate=hover or f"{name}<br>%{{x}}: {shown}{suffix}<extra></extra>")
    fig.update_layout(barmode=barmode, bargap=0.25)
    if fmt.endswith("%"):
        percent_ticks(fig, np.concatenate([np.abs(df[c].to_numpy(dtype=float)) for c in columns]))
    return finish(fig, title, height, legend if len(columns) > 1 else None, x_title, y_title)


# ----------------------------------------------------------------------------------------------------------- lines
def line_chart(df: pd.DataFrame, x: str, y: str, group: "str | None" = None, title: "str | None" = None, colors: "dict | None" = None,
               order: "list | None" = None, height: int = 360, y_title: "str | None" = None, x_title: "str | None" = None, fmt: str = ",.1f",
               suffix: str = "", markers: bool = False, fill: bool = False, width: float = 2.2, legend: "str | None" = "top",
               hover: "str | None" = None, dash: "dict | None" = None, step: bool = False) -> go.Figure:
    """One line per value of `group` (or a single line). `colors` maps group -> colour, `order` fixes the legend order, `fill=True` fills under the line."""
    if df.empty:
        return empty_figure()
    groups = [None] if group is None else (order or list(dict.fromkeys(df[group].tolist())))
    palette = theme.colors_for([g for g in groups if g is not None], colors)
    fig = go.Figure()
    for g in groups:
        part = df if g is None else df[df[group] == g]
        if part.empty:
            continue
        color = theme.BLUE if g is None else palette[g]
        fig.add_scatter(x=part[x], y=part[y], mode="lines+markers" if markers else "lines", name=str(g) if g is not None else y,
                        line=dict(color=color, width=width, shape="hv" if step else "linear", dash=(dash or {}).get(g, "solid")),
                        marker=dict(size=6), fill="tozeroy" if fill else None, fillcolor=_rgba(color, 0.12) if fill else None,
                        hovertemplate=hover or f"{'' if g is None else str(g) + '<br>'}%{{x|%d %b %Y}}: %{{y:{fmt}}}{suffix}<extra></extra>")
    return finish(fig, title, height, legend if group is not None else None, x_title, y_title)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


with_alpha = _rgba                                                                  # the public name, for pages that draw their own traces


# ------------------------------------------------------------------------------------------------------- heatmaps
def heat(matrix: pd.DataFrame, title: "str | None" = None, colorscale=None, zmin=None, zmax=None, fmt: str = ",.0f", show_text: bool = True,
         height: int = 360, x_title: "str | None" = None, y_title: "str | None" = None, colorbar_title: "str | None" = None,
         hover: "str | None" = None, reverse_y: bool = True) -> go.Figure:
    """A heatmap of a table: rows of `matrix` down, columns across. NaN cells stay empty."""
    if matrix.empty:
        return empty_figure()
    z = matrix.to_numpy(dtype=float)
    scale = colorscale or [(i / (len(theme.SEQUENTIAL) - 1), c) for i, c in enumerate(theme.SEQUENTIAL)]
    fig = go.Figure(go.Heatmap(z=z, x=[str(c) for c in matrix.columns] if not isinstance(matrix.columns, pd.DatetimeIndex) else matrix.columns,
                               y=[str(r) for r in matrix.index] if not isinstance(matrix.index, pd.DatetimeIndex) else matrix.index,
                               colorscale=scale, zmin=zmin, zmax=zmax, xgap=2, ygap=2, hoverongaps=False,
                               text=np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:{fmt}}")(np.nan_to_num(z))) if show_text else None,
                               texttemplate="%{text}" if show_text else None, textfont=dict(size=12),
                               colorbar=dict(title=dict(text=colorbar_title or ""), thickness=12, outlinewidth=0),
                               hovertemplate=hover or "%{y} / %{x}: %{z:" + fmt + "}<extra></extra>"))
    if reverse_y:
        fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showgrid=False, side="bottom")
    fig.update_yaxes(showgrid=False)
    return finish(fig, title, height, legend=None, x_title=x_title, y_title=y_title)


# ------------------------------------------------------------------------------------------------------- scatter
def scatter_chart(df: pd.DataFrame, x: str, y: str, color: "str | None" = None, size: "str | None" = None, title: "str | None" = None,
                  colors: "dict | None" = None, order: "list | None" = None, height: int = 420, x_title: "str | None" = None,
                  y_title: "str | None" = None, hover_cols: "list | None" = None, hover_names: "dict | None" = None, opacity: float = 0.7,
                  size_max: float = 26, reverse_y: bool = False, legend: "str | None" = "top", name_col: "str | None" = None,
                  fixed_size: float = 8) -> go.Figure:
    """A scatter (or bubble) chart. `color` = a column of categories (one trace each), `size` = a numeric column for bubble area."""
    if df.empty:
        return empty_figure()
    groups = [None] if color is None else (order or list(dict.fromkeys(df[color].tolist())))
    palette = theme.colors_for([g for g in groups if g is not None], colors)
    fig = go.Figure()
    sizes_all = df[size].astype(float).clip(lower=0) if size else None
    scale = (2.0 * float(sizes_all.max()) / (size_max ** 2)) if size and float(sizes_all.max()) > 0 else None
    for g in groups:
        part = df if g is None else df[df[color] == g]
        if part.empty:
            continue
        custom = part[hover_cols].to_numpy() if hover_cols else None
        lines = [f"%{{customdata[{i}]}}" for i in range(len(hover_cols or []))]
        names_h = hover_names or {}
        extra = "<br>".join(f"{names_h.get(c, c)}: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols or []))
        head = "%{text}<br>" if name_col else ""
        marker = dict(color=theme.BLUE if g is None else palette[g], opacity=opacity, line=dict(width=0.6, color="#ffffff"))
        if size and scale:
            marker.update(size=part[size].astype(float).clip(lower=0), sizemode="area", sizeref=scale, sizemin=4)
        else:
            marker.update(size=fixed_size)
        fig.add_scatter(x=part[x], y=part[y], mode="markers", name=str(g) if g is not None else "", marker=marker, customdata=custom,
                        text=part[name_col] if name_col else None, hovertemplate=head + extra + "<extra></extra>" if hover_cols or name_col else None)
    if reverse_y:
        seen = df[y].astype(float).dropna()
        low, high = float(seen.min()), float(seen.max())
        pad = max((high - low) * 0.04, 0.5)
        fig.update_yaxes(autorange=False, range=[high + pad, low - pad])               # the best place (the smallest number) at the top
    return finish(fig, title, height, legend if color is not None else None, x_title, y_title)


def histogram_chart(df: pd.DataFrame, value: str, group: "str | None" = None, title: "str | None" = None, colors: "dict | None" = None,
                    order: "list | None" = None, bins: "int | None" = None, bin_size: "float | None" = None, height: int = 340,
                    x_title: "str | None" = None, y_title: str = "Songs", histnorm: "str | None" = None, barmode: str = "overlay",
                    legend: "str | None" = "top", start: "float | None" = None, end: "float | None" = None) -> go.Figure:
    """A histogram of one column, one colour per group. `histnorm="percent"` compares groups of different sizes fairly."""
    if df.empty:
        return empty_figure()
    groups = [None] if group is None else (order or list(dict.fromkeys(df[group].tolist())))
    palette = theme.colors_for([g for g in groups if g is not None], colors)
    fig = go.Figure()
    for g in groups:
        part = df if g is None else df[df[group] == g]
        if part.empty:
            continue
        xbins = dict(size=bin_size, start=start, end=end) if bin_size else None
        fig.add_histogram(x=part[value], name=str(g) if g is not None else value, marker_color=theme.BLUE if g is None else palette[g],
                          opacity=0.72 if barmode == "overlay" and len(groups) > 1 else 1.0, nbinsx=bins, xbins=xbins, histnorm=histnorm)
    fig.update_layout(barmode=barmode, bargap=0.06)
    return finish(fig, title, height, legend if group is not None else None, x_title, "Share of songs (%)" if histnorm == "percent" else y_title)


# ------------------------------------------------------------------------------------------------------ rank charts
LOG_TICKS = [1, 2, 3, 5, 7, 10, 14, 20, 30, 50, 100, 200, 400]


def log_axis(fig: go.Figure, axis: str = "y", ticks: "list | None" = None) -> go.Figure:
    """Turn one axis of `fig` into a logarithmic axis whose ticks are readable numbers (Plotly's own ticks on a log axis are 2, 5, 1 ...)."""
    ticks = ticks or LOG_TICKS
    kwargs = dict(type="log", tickmode="array", tickvals=ticks, ticktext=[f"{t:g}" for t in ticks])
    (fig.update_yaxes if axis == "y" else fig.update_xaxes)(**kwargs)
    return fig


def shorten(text, limit: int = 44) -> str:
    """Cut a long label to `limit` characters (with an ellipsis) so that it fits on an axis."""
    text = str(text)
    return text if len(text) <= limit else text[: max(limit - 1, 1)].rstrip() + "\u2026"


def rank_lines(matrix: pd.DataFrame, names: "dict | None" = None, title: "str | None" = None, colors: "dict | None" = None, height: int = 420,
               x_title: "str | None" = None, markers: bool = False, low: int = 1, high: int = 50, date_axis: bool = True,
               legend: "str | None" = "top") -> go.Figure:
    """
    The chart place of several songs over time. `matrix` has one row per date (or per day since the debut) and one column per song;
    an empty cell is a day off the chart and is drawn as a gap. `names` maps the column to the label of the song.
    """
    if matrix is None or matrix.empty or matrix.shape[1] == 0:
        return empty_figure()
    names = names or {}
    ids = list(matrix.columns)
    palette = theme.colors_for(ids, colors)
    fig = go.Figure()
    for song in ids:
        label = shorten(names.get(song, song), 48)
        where = "%{x|%d %b %Y}" if date_axis else "day %{x}"
        fig.add_scatter(x=matrix.index, y=matrix[song], mode="lines+markers" if markers else "lines", name=label, connectgaps=False,
                        line=dict(color=palette[song], width=2.2), marker=dict(size=5),
                        hovertemplate=f"{label}<br>{where}: place %{{y:.0f}}<extra></extra>")
    rank_axis(fig, low, high)
    return finish(fig, title, height, legend, x_title)


def rank_panels(matrix: pd.DataFrame, names: "dict | None" = None, columns: int = 3, row_height: int = 190, title: "str | None" = None,
                colors: "dict | None" = None, low: int = 1, high: int = 50, date_axis: bool = True) -> go.Figure:
    """The same as rank_lines, but one small panel per song (easier to read when the songs cross each other a lot)."""
    if matrix is None or matrix.empty or matrix.shape[1] == 0:
        return empty_figure()
    names = names or {}
    ids = list(matrix.columns)
    columns = max(1, min(int(columns), len(ids)))
    n_rows = math.ceil(len(ids) / columns)
    palette = theme.colors_for(ids, colors)
    fig = make_subplots(rows=n_rows, cols=columns, subplot_titles=[shorten(names.get(i, i), 34) for i in ids],
                        vertical_spacing=min(0.16, 0.5 / max(n_rows, 1)), horizontal_spacing=0.06)
    for k, song in enumerate(ids):
        where = "%{x|%d %b %Y}" if date_axis else "day %{x}"
        fig.add_scatter(x=matrix.index, y=matrix[song], mode="lines", connectgaps=False, showlegend=False, line=dict(color=palette[song], width=2.2),
                        hovertemplate=f"{shorten(names.get(song, song), 48)}<br>{where}: place %{{y:.0f}}<extra></extra>", row=k // columns + 1, col=k % columns + 1)
    fig.update_yaxes(autorange=False, range=[high + 0.5, low - 0.5], dtick=10 if high - low > 20 else 5)
    fig.update_annotations(font=dict(size=12, color=theme.INK))
    fig.update_layout(height=max(260, row_height * n_rows + 60), title=dict(text=title) if title else None, showlegend=False,
                      margin=dict(l=14, r=16, t=70 if title else 40, b=14))
    return fig


def day_bars(day: pd.DataFrame, title: "str | None" = None, height: int = 900, value: str = "popularity_clean",
             x_title: str = "Popularity score (0 to 100)", show_outside: bool = True) -> go.Figure:
    """
    The chart of one day as horizontal bars, place 1 on top. `day` comes from movement.day_chart (columns position, song_clean,
    artist_credit, movement, rank_change, popularity_clean, selected). Bars are coloured by how the song moved; songs that the
    sidebar filters leave out are drawn in grey (or not at all with `show_outside=False`).
    """
    if day.empty:
        return empty_figure("No chart on this day")
    day = day.sort_values("position")
    labels = [f"{int(p)}   {shorten(s, 30)} - {shorten(a, 22)}" for p, s, a in zip(day["position"], day["song_clean"], day["artist_credit"])]
    day = day.assign(_label=labels)
    fig = go.Figure()
    outside = day[~day["selected"]]
    kinds = ["Up", "Down", "Same", "New entry", "Re-entry", "Chart start"]
    for kind in kinds:
        part = day[day["selected"] & (day["movement"] == kind)]
        if part.empty:
            continue
        moved = [f"{kind.lower()} {abs(int(c))}" if kind in ("Up", "Down") and pd.notna(c) else kind.lower() for c in part["rank_change"]]
        fig.add_bar(x=part[value], y=part["_label"], orientation="h", name=kind, marker_color=theme.MOVE_COLORS[kind], customdata=moved,
                    hovertemplate="%{y}<br>Popularity %{x:.0f}<br>%{customdata}<extra></extra>")
    if show_outside and not outside.empty:
        fig.add_bar(x=outside[value], y=outside["_label"], orientation="h", name="Outside the filters", marker_color="#e2e1da",
                    hovertemplate="%{y}<br>Popularity %{x:.0f}<br>outside the current filters<extra></extra>")
    order = day["_label"].tolist() if show_outside else day.loc[day["selected"], "_label"].tolist()
    fig.update_yaxes(categoryorder="array", categoryarray=order, autorange="reversed", showgrid=False, ticks="", tickfont=dict(size=11, color=theme.INK))
    fig.update_xaxes(range=[0, 104], tickmode="array", tickvals=[0, 20, 40, 60, 80, 100], showgrid=True)
    fig.update_layout(barmode="overlay", bargap=0.22)
    return finish(fig, title, height, "top", x_title)


def slope_chart(table: pd.DataFrame, left_label: str, right_label: str, top_n: int = 20, title: "str | None" = None, height: int = 640) -> go.Figure:
    """
    The Top `top_n` of two days. `table` has the columns label, place_a, place_b (empty = not on the chart that day). A song that is in
    the Top N on both days is a line (green = climbed, red = fell, grey = same place). A song in the Top N on one day only is a single
    dot: blue on the right (it arrived) or violet on the left (it left the Top N).
    """
    if table.empty:
        return empty_figure("No songs to compare")
    fig = go.Figure()
    colors = {"Up": theme.GREEN, "Down": theme.RED, "Same": "#8f8e84", "Arrived": theme.BLUE, "Left": theme.VIOLET}
    for row in table.itertuples():
        a_in = pd.notna(row.place_a) and row.place_a <= top_n
        b_in = pd.notna(row.place_b) and row.place_b <= top_n
        name = shorten(row.label, 34)
        if a_in and b_in:
            kind = "Up" if row.place_b < row.place_a else ("Down" if row.place_b > row.place_a else "Same")
            fig.add_scatter(x=[0, 1], y=[row.place_a, row.place_b], mode="lines+markers+text", showlegend=False,
                            line=dict(color=colors[kind], width=2.2), marker=dict(size=7, color=colors[kind]),
                            text=[f"{name}  {int(row.place_a)}", f"{int(row.place_b)}  {name}"], textposition=["middle left", "middle right"],
                            textfont=dict(size=11, color=theme.INK),
                            hovertemplate=f"{row.label}<br>{left_label}: place {int(row.place_a)}<br>{right_label}: place {int(row.place_b)}<extra></extra>")
        elif b_in:
            fig.add_scatter(x=[1], y=[row.place_b], mode="markers+text", showlegend=False, marker=dict(size=8, color=colors["Arrived"]),
                            text=[f"{int(row.place_b)}  {name}"], textposition="middle right", textfont=dict(size=11, color=theme.INK),
                            hovertemplate=f"{row.label}<br>{right_label}: place {int(row.place_b)}<br>{left_label}: outside the top {top_n}<extra></extra>")
        elif a_in:
            fig.add_scatter(x=[0], y=[row.place_a], mode="markers+text", showlegend=False, marker=dict(size=8, color=colors["Left"]),
                            text=[f"{name}  {int(row.place_a)}"], textposition="middle left", textfont=dict(size=11, color=theme.INK),
                            hovertemplate=f"{row.label}<br>{left_label}: place {int(row.place_a)}<br>{right_label}: outside the top {top_n}<extra></extra>")
    names = {"Up": "Climbed", "Down": "Fell", "Same": "Same place", "Arrived": "Arrived in the top " + str(top_n), "Left": "Left the top " + str(top_n)}
    for kind, color in colors.items():
        fig.add_scatter(x=[None], y=[None], mode="lines+markers", line=dict(color=color, width=3), marker=dict(color=color, size=8), name=names[kind])
    fig.update_xaxes(range=[-0.62, 1.62], tickmode="array", tickvals=[0, 1], ticktext=[left_label, right_label], showgrid=False, side="top",
                     tickfont=dict(size=13, color=theme.INK), zeroline=False)
    fig.update_yaxes(autorange=False, range=[top_n + 0.6, 0.4], tickmode="array", tickvals=list(range(1, top_n + 1, 1 if top_n <= 25 else 5)),
                     title_text="Chart place", showgrid=False)
    finish(fig, title, height, "bottom")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0), margin=dict(l=14, r=20, t=90 if title else 60, b=40))
    return fig


# ---------------------------------------------------------------------------------------------- calendar and curves
def calendar_heat(cells: pd.DataFrame, title: "str | None" = None, colorscale=None, height: int = 300, fmt: str = ",.1f", suffix: str = "",
                  colorbar_title: "str | None" = None, label: str = "Value", zmin=None, zmax=None) -> go.Figure:
    """
    One cell for every day: weeks across (the first day of the week is Monday), weekdays down. `cells` comes from
    movement.calendar_cells (columns date, weekday, week_start, value).
    """
    if cells.empty:
        return empty_figure()
    z = cells.pivot(index="weekday", columns="week_start", values="value").reindex(range(7))
    when = cells.assign(_text=cells["date"].dt.strftime("%a %d %b %Y")).pivot(index="weekday", columns="week_start", values="_text").reindex(range(7)).fillna("")
    scale = colorscale or [(i / (len(theme.SEQUENTIAL) - 1), c) for i, c in enumerate(theme.SEQUENTIAL)]
    fig = go.Figure(go.Heatmap(z=z.to_numpy(dtype=float), x=z.columns, y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], customdata=when.to_numpy(),
                               colorscale=scale, zmin=zmin, zmax=zmax, xgap=2, ygap=2, hoverongaps=False,
                               colorbar=dict(title=dict(text=colorbar_title or ""), thickness=12, outlinewidth=0),
                               hovertemplate="%{customdata}<br>" + label + ": %{z:" + fmt + "}" + suffix + "<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickformat="%b %Y")
    return finish(fig, title, height, None)


def survival_chart(curves: pd.DataFrame, order: "list | None" = None, colors: "dict | None" = None, title: "str | None" = None, height: int = 430,
                   max_age: "int | None" = None, bands: bool = True, y_title: str = "Share of songs still on the chart",
                   x_title: str = "Chart days since the song entered") -> go.Figure:
    """
    Survival curves (step lines) with their 95 % bands. `curves` has the columns group, age, survival, low, high, songs
    (see survival_live.live_survival). A dotted line marks the median stay (where a curve crosses 50 %).
    """
    if curves.empty:
        return empty_figure("Too few songs for a curve")
    groups = order or list(dict.fromkeys(curves["group"].tolist()))
    palette = theme.colors_for(groups, colors)
    band_alpha = 0.15 if len(groups) <= 3 else 0.07                            # with many groups the bands overlap, so they are drawn fainter
    fig = go.Figure()
    for g in groups:
        part = curves[curves["group"] == g]
        if max_age is not None:
            part = part[part["age"] <= max_age]
        if part.empty:
            continue
        color = palette[g]
        if bands:
            fig.add_scatter(x=part["age"], y=part["high"], mode="lines", line=dict(width=0, shape="hv"), hoverinfo="skip", showlegend=False, legendgroup=str(g))
            fig.add_scatter(x=part["age"], y=part["low"], mode="lines", line=dict(width=0, shape="hv"), fill="tonexty", fillcolor=_rgba(color, band_alpha),
                            hoverinfo="skip", showlegend=False, legendgroup=str(g))
        fig.add_scatter(x=part["age"], y=part["survival"], mode="lines", name=f"{g} ({int(part['songs'].iloc[0])} songs)", legendgroup=str(g),
                        line=dict(color=color, width=2.4, shape="hv"),
                        hovertemplate=f"{g}<br>day %{{x}}: %{{y:.0%}} still on the chart<extra></extra>")
    fig.add_hline(y=0.5, line=dict(color=theme.AXIS, dash="dot", width=1.2))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig.update_xaxes(range=[0, max_age if max_age is not None else float(curves["age"].max())])
    return finish(fig, title, height, "top", x_title, y_title)


def lorenz_chart(x_share, y_share, gini: float, title: "str | None" = None, height: int = 380, name: str = "Chart points",
                 x_title: str = "Share of artists (the smallest first)", y_title: str = "Share of chart points") -> go.Figure:
    """The Lorenz curve: the further it bends away from the diagonal, the more the points sit with a few artists."""
    x_share, y_share = np.asarray(x_share, dtype=float), np.asarray(y_share, dtype=float)
    if len(x_share) == 0:
        return empty_figure()
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=theme.AXIS, dash="dash", width=1.6), name="Equal shares", hoverinfo="skip")
    fig.add_scatter(x=x_share, y=y_share, mode="lines", fill="tonexty", fillcolor=_rgba(theme.BLUE, 0.13), line=dict(color=theme.BLUE, width=2.4),
                    name=f"{name} (Gini {gini:.2f})", hovertemplate="The smallest %{x:.0%} of artists hold %{y:.1%}<extra></extra>")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(tickformat=".0%", range=[0, 1.01])
    return finish(fig, title, height, "top", x_title, y_title)


def ribbon_chart(df: pd.DataFrame, x: str, mid: str, low: str, high: str, title: "str | None" = None, color: str = theme.BLUE, height: int = 360,
                 x_title: "str | None" = None, y_title: "str | None" = None, fmt: str = ".1f", mid_name: str = "Median",
                 band_name: str = "Middle half of the songs") -> go.Figure:
    """A line with a band around it (for example the median popularity at every place with the 25th to 75th percentile band)."""
    if df.empty:
        return empty_figure()
    fig = go.Figure()
    fig.add_scatter(x=df[x], y=df[high], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=df[x], y=df[low], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=_rgba(color, 0.18), name=band_name, hoverinfo="skip")
    fig.add_scatter(x=df[x], y=df[mid], mode="lines", line=dict(color=color, width=2.4), name=mid_name,
                    hovertemplate=f"{mid_name}: %{{y:{fmt}}}<extra></extra>")
    return finish(fig, title, height, "top", x_title, y_title)


def box_chart(df: pd.DataFrame, value: str, group: str, order: "list | None" = None, colors: "dict | None" = None, title: "str | None" = None,
              height: int = 340, y_title: "str | None" = None, horizontal: bool = False) -> go.Figure:
    """Box plots of one number for several groups (the box is the middle half, the line inside is the median)."""
    if df.empty:
        return empty_figure()
    groups = order or list(dict.fromkeys(df[group].tolist()))
    palette = theme.colors_for(groups, colors)
    fig = go.Figure()
    for g in groups:
        part = df.loc[df[group] == g, value].dropna()
        if part.empty:
            continue
        kwargs = dict(x=part) if horizontal else dict(y=part)
        fig.add_box(name=str(g), marker_color=palette[g], line=dict(width=1.6), boxpoints=False, **kwargs)
    return finish(fig, title, height, None, y_title if not horizontal else None, None if not horizontal else y_title)


def span_chart(df: pd.DataFrame, label: str, start: str, end: str, color_col: "str | None" = None, colors: "dict | None" = None,
               title: "str | None" = None, height: "int | None" = None, hover_cols: "list | None" = None, hover_names: "dict | None" = None,
               legend: "str | None" = "top") -> go.Figure:
    """Horizontal bars from the first to the last day of every row (a song's stay, an artist's songs one under the other)."""
    if df.empty:
        return empty_figure()
    frame = df.sort_values(start).reset_index(drop=True)
    groups = [None] if color_col is None else list(dict.fromkeys(frame[color_col].tolist()))
    palette = theme.colors_for([g for g in groups if g is not None], colors)
    fig = go.Figure()
    names_h = hover_names or {}
    for g in groups:
        part = frame if g is None else frame[frame[color_col] == g]
        width = (part[end] - part[start]).dt.total_seconds() * 1000.0 + 86_400_000.0         # a one-day stay still gets a visible bar
        custom = part[hover_cols].to_numpy() if hover_cols else None
        extra = "".join(f"<br>{names_h.get(c, c)}: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols or []))
        fig.add_bar(base=part[start], x=width, y=part[label], orientation="h", name=str(g) if g is not None else "",
                    marker_color=theme.BLUE if g is None else palette[g], customdata=custom, hovertemplate="%{y}" + extra + "<extra></extra>")
    fig.update_xaxes(type="date")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=frame[label].tolist(), showgrid=False, ticks="")
    fig.update_layout(barmode="overlay", bargap=0.3)
    return finish(fig, title, height or max(240, 30 * len(frame) + 100), legend if color_col else None)


def song_timeline(daily: pd.DataFrame, title: "str | None" = None, height: int = 540) -> go.Figure:
    """
    One song over time: the chart place (top panel, place 1 at the top) and the popularity score with its 7-day average (bottom panel).
    `daily` has one row per snapshot date between the song's first and last day (columns date, position, popularity_clean,
    popularity_trend_7d); a day off the chart has an empty place and is drawn as a gap.
    """
    if daily.empty:
        return empty_figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.58, 0.42],
                        subplot_titles=("Chart place (1 = top)", "Popularity score (0 to 100)"))
    fig.add_scatter(x=daily["date"], y=daily["position"], mode="lines+markers", name="Place", connectgaps=False, line=dict(color=theme.BLUE, width=2.2),
                    marker=dict(size=4), hovertemplate="%{x|%d %b %Y}: place %{y:.0f}<extra></extra>", row=1, col=1)
    fig.add_scatter(x=daily["date"], y=daily["popularity_clean"], mode="lines", name="Popularity", connectgaps=False, line=dict(color=theme.ORANGE, width=1.6),
                    hovertemplate="%{x|%d %b %Y}: popularity %{y:.0f}<extra></extra>", row=2, col=1)
    fig.add_scatter(x=daily["date"], y=daily["popularity_trend_7d"], mode="lines", name="7-day average", connectgaps=False,
                    line=dict(color=theme.VIOLET, width=2.4, dash="dot"), hovertemplate="%{x|%d %b %Y}: 7-day average %{y:.1f}<extra></extra>", row=2, col=1)
    seen = daily["position"].dropna()
    low, high = (max(0.5, float(seen.min()) - 1), min(50.5, float(seen.max()) + 1.5)) if len(seen) else (0.5, 50.5)
    fig.update_yaxes(autorange=False, range=[high, low], row=1, col=1)
    fig.update_annotations(font=dict(size=12, color=theme.MUTED), x=0, xanchor="left")
    fig.update_layout(height=height, title=dict(text=title) if title else None, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
                      margin=dict(l=14, r=20, t=90 if title else 60, b=14))
    return fig


def pareto_chart(labels, share_pct, cumulative_pct, title: "str | None" = None, height: int = 400, bar_color: str = theme.BLUE,
                 y_title: str = "Share of chart points (%)", x_title: "str | None" = None) -> go.Figure:
    """Bars of each holder's share (biggest first) and the running total as a line on a second axis (0 to 100 %)."""
    if len(labels) == 0:
        return empty_figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=list(labels), y=list(share_pct), name="Share of chart points", marker_color=bar_color,
                hovertemplate="%{x}<br>%{y:.2f}% of the chart points<extra></extra>")
    fig.add_scatter(x=list(labels), y=list(cumulative_pct), name="Running total", mode="lines+markers", line=dict(color=theme.ORANGE, width=2.4), marker=dict(size=5),
                    customdata=list(range(1, len(labels) + 1)), hovertemplate="The top %{customdata} hold %{y:.1f}%<extra></extra>", secondary_y=True)
    fig.update_yaxes(title_text=y_title, secondary_y=False, rangemode="tozero")
    # tickmode "linear" is set on purpose: newer Plotly.js versions line the ticks of a second axis up with the first axis ("sync"), which gives 25.2, 50.5, 75.7
    fig.update_yaxes(title_text="Running total (%)", secondary_y=True, range=[0, 100], showgrid=False, tickmode="linear", tick0=0, dtick=20)
    fig.update_xaxes(tickangle=-40, title_text=x_title)
    fig.update_layout(bargap=0.25)
    return finish(fig, title, height, "top")


# ------------------------------------------------------------------------------------------------ intervals (research pages)
def interval_bars(df: pd.DataFrame, x: str, y: str, low: str, high: str, title: "str | None" = None, color: str = theme.BLUE, colors: "dict | None" = None,
                  group: "str | None" = None, order: "list | None" = None, height: int = 360, y_title: "str | None" = None, x_title: "str | None" = None,
                  fmt: str = ".0%", names: bool = False) -> go.Figure:
    """
    Bars with a 95 % interval drawn as a thin line (for example the chance of leaving on a chart day, with the bootstrap interval).
    `group` makes one bar per group at every x value; `fmt` is the Plotly number format of the value in the hover box.
    """
    if df.empty:
        return empty_figure()
    groups = [None] if group is None else (order or list(dict.fromkeys(df[group].tolist())))
    palette = theme.colors_for([g for g in groups if g is not None], colors)
    fig = go.Figure()
    for g in groups:
        part = df if g is None else df[df[group] == g]
        if part.empty:
            continue
        paint = color if g is None else palette[g]
        up = (part[high] - part[y]).clip(lower=0).to_numpy(dtype=float)
        down = (part[y] - part[low]).clip(lower=0).to_numpy(dtype=float)
        fig.add_bar(x=part[x], y=part[y], name=str(g) if g is not None else y, marker_color=paint,
                    error_y=dict(type="data", symmetric=False, array=up, arrayminus=down, color=theme.INK_2, thickness=1.3, width=3),
                    customdata=np.column_stack([part[low].to_numpy(dtype=float), part[high].to_numpy(dtype=float)]),
                    hovertemplate=f"{'' if g is None else str(g) + '<br>'}%{{x}}: %{{y:{fmt}}}<br>95% interval %{{customdata[0]:{fmt}}} to %{{customdata[1]:{fmt}}}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(rangemode="tozero")
    if fmt.endswith("%"):
        percent_ticks(fig, df[high].to_numpy(dtype=float))
    return finish(fig, title, height, "top" if group is not None else None, x_title, y_title)


FOREST_TICKS = [0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24]


def forest_chart(df: pd.DataFrame, label: str, estimate: str, low: str, high: str, title: "str | None" = None, reference: float = 1.0, log_x: bool = True,
                 color_col: "str | None" = None, colors: "dict | None" = None, height: "int | None" = None, x_title: "str | None" = None, fmt: str = ".2f",
                 note: "str | None" = None) -> go.Figure:
    """
    One row per input: a dot for the estimate and a line for its 95 % interval, with a dashed line at the value that means "no difference"
    (1 for a hazard ratio or an odds ratio, 0 for a difference). `color_col` colours the rows by a reading word, `note` names a column whose
    text is added to the hover box (for example what one step of the input means). The rows keep the order of `df`.
    """
    if df.empty:
        return empty_figure()
    frame = df.reset_index(drop=True)
    labels = frame[label].astype(str).tolist()
    groups = [None] if color_col is None else list(dict.fromkeys(frame[color_col].tolist()))
    palette = theme.colors_for([g for g in groups if g is not None], colors)
    fig = go.Figure()
    for g in groups:
        part = frame if g is None else frame[frame[color_col] == g]
        paint = theme.BLUE if g is None else palette[g]
        fig.add_scatter(x=part[estimate], y=part[label].astype(str), mode="markers", name=str(g) if g is not None else "Estimate", showlegend=g is not None,
                        marker=dict(color=paint, size=10, symbol="diamond"),
                        error_x=dict(type="data", symmetric=False, array=(part[high] - part[estimate]).clip(lower=0).to_numpy(dtype=float),
                                     arrayminus=(part[estimate] - part[low]).clip(lower=0).to_numpy(dtype=float), color=paint, thickness=2, width=5),
                        customdata=np.column_stack([part[low].to_numpy(dtype=float), part[high].to_numpy(dtype=float),
                                                    (part[note].astype(str) if note else pd.Series("", index=part.index)).to_numpy()]),
                        hovertemplate=f"%{{y}}<br>%{{x:{fmt}}} (95% interval %{{customdata[0]:{fmt}}} to %{{customdata[1]:{fmt}}})"
                                      + ("<br>%{customdata[2]}" if note else "") + "<extra></extra>")
    fig.add_vline(x=reference, line=dict(color=theme.AXIS, dash="dash", width=1.5))
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=labels, showgrid=True, automargin=True)
    if log_x:
        lowest, highest = float(frame[low].min()), float(frame[high].max())
        ticks = [t for t in FOREST_TICKS if lowest / 1.35 <= t <= highest * 1.35] or [1]
        fig.update_xaxes(type="log", tickmode="array", tickvals=ticks, ticktext=[f"{t:g}" for t in ticks], range=[math.log10(max(lowest, 1e-6) / 1.2), math.log10(highest * 1.2)])
    return finish(fig, title, height or max(260, 96 + 36 * len(frame)), "top" if color_col else None, x_title, None)
