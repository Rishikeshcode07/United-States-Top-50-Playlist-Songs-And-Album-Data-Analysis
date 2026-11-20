"""
src/dash  -  the numbers behind the Streamlit dashboard (Phase 7), written without any Streamlit code.

WHAT THIS FILE IS
    The marker that makes the folder `src/dash/` a Python package. The folder is the calculation layer of the dashboard.
    Every number, table and curve that a dashboard page shows is produced by a plain function in one of these files, and the
    pages (app/views/) only choose which function to call and how to draw the result.

        load.py           reads the seven cleaned tables and the result files of Phases 4 to 6C; explains what to do when they are missing
        filters.py        the filter state (dates, places, release type, explicit, collaboration, artists, songs) and how it cuts the table
        metrics.py        the KPIs of the brief and every table of the pages: songs, artists, entries and exits, concentration ...
        movement.py       the chart as a whole: turnover, arrivals and exits, the busiest days, the calendar and weekday pattern
        compare.py        two groups side by side (explicit and not explicit, single and album ...): shares, medians, effect sizes
        survival_live.py  the survival curves of Phase 6B, recomputed for whatever the filters select
        texts.py          small functions that turn numbers into the plain sentences shown under the charts

WHY WE NEED IT
    A dashboard that mixes calculation and drawing cannot be tested without a browser. Here the calculations can be tested with
    ordinary pytest on the real tables: the song table must reproduce `dim_track`, the artist table must reproduce `dim_artist`,
    the live survival curve must reproduce the Phase 6B curve. The pages then have very little left that can go wrong.

WHAT HAPPENS WHEN IT RUNS
    Nothing happens when this file is imported. app/streamlit_app.py and the tests import the modules of this folder.

EXPECTED OUTCOME
    Pure pandas, numpy and scipy (the packages that are already in requirements.txt). No Streamlit, no Plotly.

MOTIVE
    Calculate every number in one place and test it once.
"""
