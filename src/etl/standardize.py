"""
src/etl/standardize.py  -  step 3 of the pipeline: give every row a clean, trustworthy IDENTITY.

WHAT THIS FILE IS
    The logic of Phase 1, Steps 5-7 turned into functions. It takes the typed table and adds the columns that say
    WHO and WHAT a row is: tidy artist names, tidy titles, matching keys, a stable `track_id`, and three flag columns.
    It does not delete a single row.

WHY WE NEED IT
    The raw file has no song ID. A title alone is not an identity: different songs share a title, and the same song
    is sometimes credited a little differently over time ("A & B" one month, "A" the next). Counting unique songs or
    unique artists on such text would give wrong numbers. So we build the identity in four moves:
        1. split each artist string into individual artists and give each artist ONE official spelling
        2. tidy the title and build comparison keys (lower-case, straight quotes; accents are kept)
        3. decide whether two credits of one title are the same song (shares an artist AND never on the chart together)
        4. number the songs: track_id = 1, 2, 3 ... in order of first appearance in the file

WHAT HAPPENS WHEN IT RUNS
    `standardize(df, qlog)` returns the validated table: 10 original columns + 11 derived columns + 3 flag columns,
    plus a small table that lists every credit decision (so the merge is never a black box).
    Seven consistency checks are added to the scorecard.

EXPECTED OUTCOME
    For the project file: 27,800 rows x 24 columns, 973 title+credit pairs resolved into 944 songs,
    and flags: 50 superseded rows, 189 zero-popularity rows, 42 'compilation' rows.

MOTIVE
    "How many unique songs did an artist chart?" is only a fair question once a song has exactly one identity.
"""
from __future__ import annotations

import pandas as pd

from src.config import ALLOWED_ALBUM_TYPES, EXPECTED_COLUMNS, SEPARATOR
from src.etl.errors import DataContractError
from src.etl.text_utils import make_key, normalize_text, split_artists
from src.logger import get_logger

log = get_logger(__name__)

DERIVED_COLUMNS = ["song_clean", "artist_clean", "primary_artist", "artist_count", "artists_all",
                   "song_key", "artist_key", "track_key", "track_id", "track_credit", "snapshot_seq"]
FLAG_COLUMNS = ["flag_superseded_snapshot", "flag_popularity_zero", "flag_unexpected_album_type"]
TRACK_ATTRIBUTES = ["duration_ms", "album_type", "total_tracks", "is_explicit", "album_cover_url"]


def standardize(df_typed, qlog):
    """Add identity columns and flags. Returns (validated_table, credit_decisions_table)."""
    df = df_typed.copy()

    # Step 5.2 - number the blocks of a date: 1 = first time a (date, rank) pair appears in the file, 2 = second time ...
    # The raw file is never re-ordered, so "first" and "second" refer to the position in the file.
    df["snapshot_seq"] = df.groupby(["date", "position"]).cumcount() + 1

    df = _standardize_artists(df)
    df = _standardize_titles(df)
    df, decisions = _resolve_track_identity(df, qlog)
    _check_attribute_stability(df, qlog)
    df = _add_flags(df)

    original_cols = list(EXPECTED_COLUMNS)
    validated = df[original_cols + DERIVED_COLUMNS + FLAG_COLUMNS].copy()          # a logical column order; helpers dropped
    log.info("Standardized: %s rows, %s songs, %s individual artists", f"{len(validated):,}",
             f"{validated['track_id'].nunique():,}", f"{validated['artists_all'].str.split(' | ', regex=False).explode().nunique():,}")
    return validated, decisions


# ====================================================================================== 1. artists
def _standardize_artists(df):
    """Split every credit into individual artists and keep ONE official spelling per artist."""
    # Speed trick: only a few hundred distinct strings exist, so split those once and map the result to every row.
    split_map = {a: split_artists(a) for a in df["artist"].unique()}
    df["artist_list"] = df["artist"].map(split_map)

    # Count how often each spelling appears, then keep the most frequent spelling of each matching key.
    # (Ties are broken alphabetically, so the result never depends on the pandas version.)
    spelling_counts = df["artist_list"].explode().value_counts()
    spelling_table = (spelling_counts.rename_axis("spelling").reset_index(name="rows")
                      .assign(key=lambda t: t["spelling"].map(make_key)))
    official_spelling = (spelling_table.sort_values(["key", "rows", "spelling"], ascending=[True, False, True])
                         .drop_duplicates("key")
                         .set_index("key")["spelling"])
    n_variant_names = int((spelling_table.groupby("key").size() > 1).sum())
    log.debug("Artist names written in more than one way: %s", n_variant_names)

    df["artist_list"] = df["artist_list"].map(lambda names: [official_spelling[make_key(n)] for n in names])
    if any("|" in name for names in df["artist_list"] for name in names):
        raise DataContractError("An artist name contains the character '|', which this pipeline uses as a joiner. "
                                "Choose another joiner in standardize.py.")

    df["artist_clean"] = df["artist_list"].map(lambda names: SEPARATOR.join(names))      # 'Lil Durk & J. Cole'
    df["artists_all"] = df["artist_list"].map(lambda names: " | ".join(names))           # 'Lil Durk | J. Cole' (easy to split later)
    df["primary_artist"] = df["artist_list"].map(lambda names: names[0])                 # the first (lead) artist
    df["artist_count"] = df["artist_list"].map(len)                                      # how many artists are on the track
    return df.drop(columns=["artist_list"])                                              # lists do not save cleanly to CSV


# ====================================================================================== 2. titles and keys
def _standardize_titles(df):
    """Tidy titles for display and build the keys used to COMPARE and group (never to display)."""
    df["song_clean"] = df["song"].map(normalize_text)
    df["song_key"] = df["song"].map(make_key)
    df["artist_key"] = df["artist_clean"].map(make_key)
    df["track_key"] = df["song_key"] + " || " + df["artist_key"]          # strict identity: title + credit exactly as written
    # One official display spelling per track: the most common spelling of its title (ties: alphabetical)
    df["song_clean"] = df.groupby("track_key")["song_clean"].transform(lambda s: s.mode().iloc[0])
    return df


# ====================================================================================== 3. track identity
def _resolve_track_identity(df, qlog):
    """Decide which credits of one title are the SAME song, then number the songs with track_id."""
    # Use only ONE block per date, so the double snapshot of a date can not create fake "same day" overlaps.
    latest_block = df["snapshot_seq"] == df.groupby("date")["snapshot_seq"].transform("max")
    kept = df[latest_block]

    # One row per (title, artist credit): how many days it was on the chart, which dates, which individual artists.
    credits = (kept.groupby(["song_key", "artist_key"])
                   .agg(artist_clean=("artist_clean", "first"), artists=("artists_all", "first"),
                        days=("date", "nunique"), dates=("date", lambda s: frozenset(s)))
                   .reset_index())
    title_display = kept.drop_duplicates("song_key").set_index("song_key")["song_clean"]    # nice-looking title for reports

    credits_per_title = credits.groupby("song_key")["artist_key"].nunique()
    qlog.add("Consistency", "Titles with several artist credits", "A title is expected to carry one artist credit",
             int((credits_per_title > 1).sum()), credits["song_key"].nunique(), unit="titles", severity="INFO",
             action="Decide per title: same song or different songs (below)")

    # Compare every extra credit with the title's main credit (the one with most days on the chart).
    # Same song  <=>  the two credits share an artist AND were never on the chart on the same day.
    main_credit_of = {}          # (title key, artist key) -> the artist key of the main credit it belongs to
    verdict_rows = []
    for title, group in credits.groupby("song_key"):
        group = group.sort_values(["days", "artist_key"], ascending=[False, True])          # most days first
        main = group.iloc[0]
        main_credit_of[(title, main["artist_key"])] = main["artist_key"]
        for _, other in group.iloc[1:].iterrows():
            shares_artist = bool(set(main["artists"].split(" | ")) & set(other["artists"].split(" | ")))
            days_together = len(main["dates"] & other["dates"])                              # dates where both were on the chart
            same_song = shares_artist and days_together == 0
            main_credit_of[(title, other["artist_key"])] = main["artist_key"] if same_song else other["artist_key"]
            verdict_rows.append({"song": title_display[title], "main_credit": main["artist_clean"],
                                 "other_credit": other["artist_clean"], "shares_an_artist": shares_artist,
                                 "dates_on_chart_together": days_together,
                                 "verdict": "same song" if same_song else "different song"})
    decisions = pd.DataFrame(verdict_rows, columns=["song", "main_credit", "other_credit", "shares_an_artist",
                                                    "dates_on_chart_together", "verdict"])

    # Give every row its resolved identity. factorize numbers the songs 0, 1, 2 ... in order of first appearance.
    df["track_key_resolved"] = [f"{s} || {main_credit_of.get((s, a), a)}" for s, a in zip(df["song_key"], df["artist_key"])]
    df["track_id"] = pd.factorize(df["track_key_resolved"])[0] + 1

    display_credit = {(t, ak): ac for t, ak, ac in zip(credits["song_key"], credits["artist_key"], credits["artist_clean"])}
    df["track_credit"] = [display_credit.get((s, main_credit_of.get((s, a), a)), ac)       # the main credit, as readable text
                          for s, a, ac in zip(df["song_key"], df["artist_key"], df["artist_clean"])]

    n_merged = int((decisions["verdict"] == "same song").sum())
    qlog.add("Uniqueness", "A song keeps one artist credit over time",
             "Same title + shared artist + never on the chart together = the same song",
             n_merged, len(credits), unit="title+credit pairs", severity="WARN",
             action="Merged into one track_id (Step 6.5)")
    log.info("Tracks by strict key (title + credit): %s | after resolution (track_id): %s",
             f"{df['track_key'].nunique():,}", f"{df['track_id'].nunique():,}")
    return df.drop(columns=["track_key_resolved"]), decisions


# ====================================================================================== 4. consistency checks and flags
def _check_attribute_stability(df, qlog):
    """Does a song keep the same duration, album type, size, explicit flag and artwork on every day it appears?"""
    distinct_per_track = df.groupby("track_id")[TRACK_ATTRIBUTES].nunique()       # distinct values per track, per attribute
    n_tracks = len(distinct_per_track)
    for col in TRACK_ATTRIBUTES:
        qlog.add("Consistency", f"{col} is stable per track", "A track keeps one value across all the days it appears",
                 int((distinct_per_track[col] > 1).sum()), n_tracks, unit="tracks", severity="INFO",
                 action="Phase 2: pick a rule (most recent value) when building one row per track")


def _add_flags(df):
    """Mark problem rows instead of deleting them. A later step decides what to do with each flag."""
    blocks_that_day = df.groupby("date")["snapshot_seq"].transform("max")                  # blocks of this row's date
    df["flag_superseded_snapshot"] = (df["snapshot_seq"] < blocks_that_day).astype(bool)   # True for the earlier block(s)
    df["flag_popularity_zero"] = (df["popularity"] == 0).astype(bool)
    album_clean = df["album_type"].astype("string").str.strip().str.casefold()
    df["flag_unexpected_album_type"] = (~album_clean.isin(ALLOWED_ALBUM_TYPES).fillna(False).astype(bool)).astype(bool)
    return df
