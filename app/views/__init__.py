"""
app/views  -  one file per dashboard page.

WHAT THIS FILE IS
    The marker that makes the folder `app/views/` a Python package. Every file has a `render()` function that the menu
    (app/nav.py, started by streamlit_app.py) calls when the page is opened:

        Overview    overview.py            Executive summary
        Playlist    timeline.py            Timeline explorer (with a day-by-day replay)
                    movement.py            Chart movement (turnover, arrivals and exits, calendar)
                    entry_exit.py          Entry and exit behaviour
        Songs       rank_trends.py         Ranking trends
                    movers.py              Risers and decliners
                    songs.py               Song leaderboard
                    song_profile.py        Song profile
        Artists     artists.py             Artist dominance
                    artist_profile.py      Artist profile
                    concentration.py       Market concentration
        Content     popularity.py          Popularity against rank
                    explicit.py            Explicit content
                    release.py             Release type and collaboration
        Research    longevity.py           Longevity and survival
                    archetypes.py          Archetypes and segments
                    evidence.py            Evidence (hypotheses, findings, checks)
                    methods.py             Data and methods

WHY WE NEED IT
    One page per question keeps each file short and lets a person go straight to the question they have, while the sidebar
    filters stay the same everywhere.

WHAT HAPPENS WHEN IT RUNS
    Nothing happens when this file is imported. A page runs only when it is opened; it reads the filtered tables from the
    shared state, calls the calculation layer (src/dash) and draws the result with the building blocks of app/ui.

EXPECTED OUTCOME
    Eighteen pages that look and behave alike, each with KPI cards, interactive charts, a short "what to notice" text and a
    table that can be downloaded.

MOTIVE
    A page is an outline of calls, not a pile of calculations.
"""
