"""
src/dash/filters.py  -  what the person chose in the sidebar, and how it cuts the table.

WHAT THIS FILE IS
    FilterState      one small, frozen object that holds every choice of the sidebar:
                         dates              first and last day
                         places             lowest and highest chart place (1 = the top of the chart)
                         album types        single / album / compilation (empty = all of them)
                         explicit           all / only explicit / only not explicit
                         collaboration      all / only collaborations / only solo songs
                         artists            songs credited to any of these artists (empty = all)
                         songs              these songs only (empty = all)
    apply_filters    the rows of playlist_daily that match the state
    keep_dates       only the date part of the state (for the chart-wide measures, see below)
    describe         the state as a short list of plain phrases ("5 Jan 2025 to 28 Feb 2025", "places 1 to 10", ...)
    artist_options / song_options   the lists that fill the artist and song pickers

WHY WE NEED IT
    The brief asks for a date range selector, artist and song filters, a rank range slider and an album type toggle. Every page
    must react to the same choices in the same way, so the rule lives here once. A frozen state with a tuple key is also what
    lets Streamlit cache an expensive table for each combination of filters.

    Two kinds of numbers exist and they must not be mixed up:
        row measures         depend on WHICH rows are kept (a song's average place, an artist's chart points, the explicit share
                             of the rows). They use `apply_filters`.
        chart-wide measures  describe the whole Top 50 on a day (how many songs entered, the turnover). A filter on artists or
                             places would give nonsense ("turnover of explicit songs"), so they use `keep_dates` only.

WHAT HAPPENS WHEN IT RUNS
    `FilterState.create` tidies the input (swaps start and end if they are the wrong way round, keeps places inside 1 to 50,
    removes repeats, ignores unknown words), so a half-finished widget can never produce a broken state.
    `apply_filters` combines the conditions with AND. An artist filter keeps a song when ANY of its credited artists is chosen
    (a feature credit counts), using the song-artist bridge table.

EXPECTED OUTCOME
    The default state (the whole date range, places 1 to 50, nothing else chosen) returns every one of the 27,750 rows.

MOTIVE
    One definition of "the selected rows" for sixteen pages.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from src.config import CHART_SIZE

ALBUM_TYPES = ("single", "album", "compilation")
EXPLICIT_CHOICES = ("all", "explicit", "clean")
COLLAB_CHOICES = ("all", "collab", "solo")


def to_date(value) -> date:
    """A date, a datetime, a pandas Timestamp or an ISO text (2025-01-31) as a plain `date`."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


@dataclass(frozen=True)
class FilterState:
    """Every choice of the sidebar. Build it with `FilterState.create(...)`; it is immutable and hashable."""
    start: date
    end: date
    rank_low: int = 1
    rank_high: int = CHART_SIZE
    album_types: tuple = ()
    explicit: str = "all"
    collab: str = "all"
    artists: tuple = ()
    track_ids: tuple = ()

    @classmethod
    def create(cls, start, end, rank_low=1, rank_high=CHART_SIZE, album_types=(), explicit="all", collab="all",
               artists=(), track_ids=()):
        """Tidy the input and return a state that is always valid."""
        first, last = to_date(start), to_date(end)
        if first > last:
            first, last = last, first
        low = max(1, min(int(rank_low), CHART_SIZE))
        high = max(1, min(int(rank_high), CHART_SIZE))
        if low > high:
            low, high = high, low
        types = tuple(t for t in ALBUM_TYPES if t in set(album_types))
        if len(types) == len(ALBUM_TYPES):                       # every type chosen = no restriction
            types = ()
        return cls(start=first, end=last, rank_low=low, rank_high=high, album_types=types,
                   explicit=explicit if explicit in EXPLICIT_CHOICES else "all",
                   collab=collab if collab in COLLAB_CHOICES else "all",
                   artists=tuple(sorted({str(a) for a in artists})),
                   track_ids=tuple(sorted({int(t) for t in track_ids})))

    def key(self) -> tuple:
        """A tuple of plain values: the cache key for this combination of filters."""
        return (self.start.isoformat(), self.end.isoformat(), self.rank_low, self.rank_high, self.album_types,
                self.explicit, self.collab, self.artists, self.track_ids)

    @classmethod
    def from_key(cls, key):
        """The state that `key()` was made from."""
        start, end, low, high, types, explicit, collab, artists, ids = key
        return cls.create(start, end, low, high, types, explicit, collab, artists, ids)

    def is_whole_chart(self, bounds) -> bool:
        """True when nothing is restricted: all dates, places 1 to 50, no other choice."""
        return (self.start <= to_date(bounds.first_date) and self.end >= to_date(bounds.last_date)
                and self.rank_low == 1 and self.rank_high == CHART_SIZE and not self.album_types
                and self.explicit == "all" and self.collab == "all" and not self.artists and not self.track_ids)

    def with_dates(self, start, end):
        """The same state with other dates (used by quick ranges and by page-level date pickers)."""
        return FilterState.create(start, end, self.rank_low, self.rank_high, self.album_types, self.explicit, self.collab,
                                  self.artists, self.track_ids)


def default_state(bounds) -> FilterState:
    """The state that restricts nothing: the whole date range of the file."""
    return FilterState.create(bounds.first_date, bounds.last_date)


def credited_track_ids(bridge: pd.DataFrame, artists) -> set:
    """The ids of the songs on which any of the artists is credited (main or featured)."""
    names = set(artists)
    if not names:
        return set()
    return set(bridge.loc[bridge["artist_name"].isin(names), "track_id"].astype(int))


def keep_dates(frame: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    """The rows of any table with a `date` column that fall in the chosen dates (and nothing else)."""
    dates = frame["date"]
    return frame.loc[(dates >= pd.Timestamp(state.start)) & (dates <= pd.Timestamp(state.end))]


def apply_filters(playlist: pd.DataFrame, state: FilterState, bridge: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """
    The rows of playlist_daily that match every part of the state. Without a bridge table the artist filter falls back to the
    primary artist only.
    """
    keep = (playlist["date"] >= pd.Timestamp(state.start)) & (playlist["date"] <= pd.Timestamp(state.end))
    keep &= (playlist["position"] >= state.rank_low) & (playlist["position"] <= state.rank_high)
    if state.album_types:
        keep &= playlist["album_type"].isin(state.album_types)
    if state.explicit == "explicit":
        keep &= playlist["is_explicit"].astype(bool)
    elif state.explicit == "clean":
        keep &= ~playlist["is_explicit"].astype(bool)
    if state.collab == "collab":
        keep &= playlist["is_collab"].astype(bool)
    elif state.collab == "solo":
        keep &= ~playlist["is_collab"].astype(bool)
    if state.artists:
        if bridge is not None:
            keep &= playlist["track_id"].isin(credited_track_ids(bridge, state.artists))
        else:
            keep &= playlist["primary_artist"].isin(state.artists)
    if state.track_ids:
        keep &= playlist["track_id"].isin(state.track_ids)
    return playlist.loc[keep]


def _day(value: date) -> str:
    return f"{value.day} {value.strftime('%b %Y')}"


def describe(state: FilterState, bounds) -> list:
    """The state as a short list of plain phrases, for the chips under the page title. Dates are always listed."""
    first, last = to_date(bounds.first_date), to_date(bounds.last_date)
    phrases = [f"{_day(state.start)} to {_day(state.end)}"]
    if state.rank_low != 1 or state.rank_high != CHART_SIZE:
        phrases.append(f"places {state.rank_low} to {state.rank_high}")
    if state.album_types:
        phrases.append("release type: " + ", ".join(state.album_types))
    if state.explicit != "all":
        phrases.append("explicit songs only" if state.explicit == "explicit" else "songs that are not explicit")
    if state.collab != "all":
        phrases.append("collaborations only" if state.collab == "collab" else "solo songs only")
    if state.artists:
        phrases.append(f"{len(state.artists)} artist{'s' if len(state.artists) != 1 else ''}")
    if state.track_ids:
        phrases.append(f"{len(state.track_ids)} song{'s' if len(state.track_ids) != 1 else ''}")
    if state.start == first and state.end == last:
        phrases[0] = "whole period"
    return phrases


def artist_options(bridge: pd.DataFrame, dim_artist: "pd.DataFrame | None" = None) -> list:
    """The artist names for the picker: the artists with the most chart days first (so the well-known names are easy to find)."""
    if dim_artist is not None and "chart_slot_days" in dim_artist.columns:
        ordered = dim_artist.sort_values(["chart_slot_days", "artist_name"], ascending=[False, True])
        return ordered["artist_name"].tolist()
    return sorted(bridge["artist_name"].unique().tolist())


def song_options(dim_track: pd.DataFrame) -> dict:
    """{track_id: 'Song by Artist'} for the song picker, the songs with the most chart days first."""
    ordered = dim_track.sort_values(["days_on_chart", "track_label"], ascending=[False, True])
    return dict(zip(ordered["track_id"].astype(int), ordered["track_label"]))
