"""
app  -  the Streamlit dashboard (Phase 7).

WHAT THIS FILE IS
    The marker that makes the folder `app/` a Python package. The folder itself is the dashboard:

        streamlit_app.py   the entry file: page settings, the sidebar filters, the menu and the page that is open
        nav.py             the list of the 18 pages (section, address, title, module, icon, one-line description) and the menu built from it
        ui/                the shared design layer: colours and chart style (theme.py), small building blocks such as KPI cards
                           (components.py), the chart builders (charts.py), the filter state and caching (state.py)
        views/             one file per dashboard page (every file has a `render()` function)
        assets/            the small icon of the browser tab and the logo

WHY WE NEED IT
    The calculations live in src/dash (no Streamlit there). This folder only decides what to show and how it looks, so that
    a page stays short and the numbers can be tested without a browser.

WHAT HAPPENS WHEN IT RUNS
    Nothing happens when this file is imported. The dashboard is started from the project folder with
    `streamlit run app/streamlit_app.py`, and it opens in the browser.

EXPECTED OUTCOME
    A dashboard of 18 pages in six sections (Overview, Playlist, Songs, Artists, Content, Research), all driven by the same
    sidebar filters.

MOTIVE
    Pages that stay short because the design and the numbers are written once, elsewhere.
"""
