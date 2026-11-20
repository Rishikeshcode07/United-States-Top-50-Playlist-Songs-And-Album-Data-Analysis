"""
app/nav.py  -  the list of dashboard pages, in the order of the menu.

WHAT THIS FILE IS
    PAGES    every page: (section, url name, menu title, module in app/views, material icon, one-line description)
    build    creates the Streamlit page objects for the menu (st.navigation) from that list

WHY WE NEED IT
    The menu, the "where to go next" cards of the overview and the tests all need the same list. Keeping it here means a page is
    added in one place.

WHAT HAPPENS WHEN IT RUNS
    `build()` imports each page module from app/views and wraps its `render` function as a page. The first page is the default.

EXPECTED OUTCOME
    A menu bar across the top with six sections (Overview, Playlist, Songs, Artists, Content, Research) and 18 pages.

MOTIVE
    One list of pages.
"""
from __future__ import annotations

import importlib

PAGES = [
    ("Overview", "overview", "Executive summary", "overview", "space_dashboard",
     "The six KPIs of the brief, the leading artists, how much of the chart changes, and what the analysis found."),
    ("Playlist", "timeline", "Timeline explorer", "timeline", "timeline",
     "Open the Top 50 of any day, replay the chart day by day, and compare two days."),
    ("Playlist", "movement", "Chart movement", "movement", "swap_vert",
     "Arrivals, exits and turnover over time, a calendar of busy days and the weekday pattern."),
    ("Playlist", "entry-exit", "Entry and exit", "entry_exit", "login",
     "Where songs enter the chart, where they leave from, and how long they stay in between."),
    ("Songs", "rank-trends", "Ranking trends", "rank_trends", "show_chart",
     "Follow chosen songs through the chart: lines, small multiples, heatmap, aligned at the debut day."),
    ("Songs", "movers", "Risers and decliners", "movers", "trending_up",
     "Who climbed or fell most over a window of days, and the fastest risers and slowest decliners of whole stays."),
    ("Songs", "songs", "Song leaderboard", "songs", "music_note",
     "Every song with its KPIs: days on chart, average rank, volatility, popularity. Sort, search and download."),
    ("Songs", "song-profile", "Song profile", "song_profile", "album",
     "One song in detail: its path through the chart, popularity, runs, and how it compares with the others."),
    ("Artists", "artists", "Artist dominance", "artists", "groups",
     "The Artist Dominance Index (ADI) leaderboard, its change over time and the shape of an artist's presence."),
    ("Artists", "artist-profile", "Artist profile", "artist_profile", "person",
     "One artist in detail: ADI over time, every song as a timeline bar, release mix, comparison with others."),
    ("Artists", "concentration", "Concentration", "concentration", "donut_small",
     "How concentrated the chart is: HHI, effective number of artists, top-N share, Lorenz curve."),
    ("Content", "popularity", "Popularity and rank", "popularity", "scatter_plot",
     "How the popularity score relates to the chart place, for song-days and for songs."),
    ("Content", "explicit", "Explicit content", "explicit", "explicit",
     "Explicit against not explicit: share over time, by place, and song by song with effect sizes."),
    ("Content", "release", "Release and collaboration", "release", "library_music",
     "Singles, albums, compilations, collaborations, release size and song length compared."),
    ("Research", "longevity", "Longevity and survival", "longevity", "hourglass_bottom",
     "How long new songs stay (Phase 6B), for any group, with the uncertainty band."),
    ("Research", "archetypes", "Archetypes and segments", "archetypes", "category",
     "The five first-month patterns of songs and the four kinds of artists (Phase 6C)."),
    ("Research", "evidence", "Evidence", "evidence", "fact_check",
     "The hypothesis tests of Phase 5, the findings of Phase 4 and the out-of-sample checks of Phase 6A."),
    ("Research", "methods", "Data and methods", "methods", "menu_book",
     "Data quality, definitions of every KPI, the data dictionary, and what is and is not claimed."),
]


def sections() -> list:
    """The section names in menu order."""
    seen = []
    for section, *_ in PAGES:
        if section not in seen:
            seen.append(section)
    return seen


def build() -> dict:
    """{section: [st.Page, ...]} for st.navigation. The first page of the list is the default one."""
    import streamlit as st
    menu: dict = {}
    for index, (section, slug, title, module, icon, _description) in enumerate(PAGES):
        view = importlib.import_module(f"app.views.{module}")
        page = st.Page(view.render, title=title, icon=f":material/{icon}:", url_path=slug, default=(index == 0))
        menu.setdefault(section, []).append(page)
    return menu
