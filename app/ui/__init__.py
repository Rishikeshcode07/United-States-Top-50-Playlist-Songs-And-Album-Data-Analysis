"""
app/ui  -  the shared design layer of the dashboard.

WHAT THIS FILE IS
    The marker that makes the folder `app/ui/` a Python package. The folder holds everything that is shared by the pages:

        theme.py        colours, fonts, the Plotly chart template and the CSS of the page
        components.py   page header, KPI cards, callouts, text cards, small and wrapped tables, tables and downloads
        charts.py       the Plotly chart builders (pure functions: a table in, a figure out)
        state.py        loading the data once, the sidebar filters, and the numbers cached for each filter combination
        reference.py    the "what the formal analysis said" cards that sit next to a page's own live numbers

WHY WE NEED IT
    Eighteen pages that each chose their own colours, fonts and HTML would drift apart. With one design layer a page reads
    like an outline: header, KPI row, chart, callout, table.

WHAT HAPPENS WHEN IT RUNS
    Nothing happens when this file is imported. The pages import the modules they need from here.

EXPECTED OUTCOME
    The same look, the same chart style and the same behaviour of the filters on every page.

MOTIVE
    Design once, use on every page.
"""
