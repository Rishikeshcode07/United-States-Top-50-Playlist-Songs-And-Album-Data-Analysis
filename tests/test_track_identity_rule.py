"""
tests/test_track_identity_rule.py  -  the rule that decides whether two artist credits of one title are ONE song or TWO.

WHAT THIS FILE IS
    The rule of the project log (Decision 5): two credits of the same title are the same song only if they share an artist
    AND were never on the chart on the same day. These tests feed tiny hand-made tables to `_resolve_track_identity`, so
    each one shows one half of the rule with a few rows:

        shares an artist, never on the chart together   -> same song   ("Ann & Bob" for a while, later just "Ann")
        shares an artist, but on the chart together      -> two songs   (a remix and the original at the same time)
        shares no artist                                 -> two songs   (two unrelated songs with one title)
        a date captured twice                            -> only the latest block counts, so the stale rows can not
                                                            create a fake "together"

WHY WE NEED IT
    On the real file the "on the chart together" half of the rule is never needed (all 29 merges are clean, see
    reports/track_identity_decisions.csv), and the hand-made chart does not contain it either. A rule that no data triggers
    is a rule that nothing protects: somebody could delete it and every other test would stay green. It matters the day a
    new file contains a remix next to its original.

WHAT HAPPENS WHEN IT RUNS
    Each test builds a table with one row per song, credit and date, runs the identity step and reads the verdicts.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Every rule of the project log should fail a test when it is removed.
"""
import pandas as pd

from src.etl.standardize import _resolve_track_identity
from src.etl.text_utils import make_key
from src.etl.validate import QualityLog

DAYS = pd.date_range("2024-01-01", periods=10)


def rows(title, credit, days, block=1):
    """One row per day on which this credit of this title was on the chart (block 2 = the second capture of a date)."""
    return [{"date": day, "snapshot_seq": block, "song_clean": title, "song_key": make_key(title),
             "artist_clean": credit, "artist_key": make_key(credit), "artists_all": " | ".join(credit.split(" & ")),
             "track_key": make_key(title) + " || " + make_key(credit)}
            for day in days]


def resolve(*groups):
    """Run the identity step on the given groups of rows. Returns (rows with track_id, the decisions table)."""
    frame = pd.DataFrame([row for group in groups for row in group])
    return _resolve_track_identity(frame, QualityLog())


def test_a_shared_artist_and_no_common_day_means_one_song():
    resolved, decisions = resolve(rows("Echo", "Ann & Bob", DAYS[0:4]), rows("Echo", "Ann", DAYS[5:8]))
    assert decisions["verdict"].tolist() == ["same song"]
    assert decisions.loc[0, "shares_an_artist"] and decisions.loc[0, "dates_on_chart_together"] == 0
    assert resolved["track_id"].nunique() == 1
    assert set(resolved["track_credit"]) == {"Ann & Bob"}                    # the credit with most days names the song


def test_a_shared_artist_but_a_common_day_means_two_songs():
    resolved, decisions = resolve(rows("Echo", "Ann & Bob", DAYS[0:5]), rows("Echo", "Ann", DAYS[3:6]))
    assert decisions["verdict"].tolist() == ["different song"]
    assert decisions.loc[0, "shares_an_artist"] and decisions.loc[0, "dates_on_chart_together"] == 2
    assert resolved["track_id"].nunique() == 2
    assert resolved.groupby("track_credit")["track_id"].nunique().eq(1).all()      # each credit keeps its own id


def test_one_common_day_is_enough_to_keep_the_songs_apart():
    resolved, decisions = resolve(rows("Echo", "Ann & Bob", DAYS[0:5]), rows("Echo", "Ann", DAYS[4:7]))
    assert decisions["dates_on_chart_together"].tolist() == [1]
    assert decisions["verdict"].tolist() == ["different song"] and resolved["track_id"].nunique() == 2


def test_no_shared_artist_means_two_songs_even_when_they_never_met_on_the_chart():
    resolved, decisions = resolve(rows("Daylight", "David", DAYS[0:3]), rows("Daylight", "Drake", DAYS[5:8]))
    assert decisions["verdict"].tolist() == ["different song"]
    assert not decisions.loc[0, "shares_an_artist"] and decisions.loc[0, "dates_on_chart_together"] == 0
    assert resolved["track_id"].nunique() == 2


def test_the_credit_with_more_days_wins_and_a_tie_goes_to_the_alphabet():
    resolved, _ = resolve(rows("Echo", "Ann & Bob", DAYS[0:3]), rows("Echo", "Ann", DAYS[5:8]))
    assert set(resolved["track_credit"]) == {"Ann"}                          # 3 days each: "ann" sorts before "ann & bob"
    resolved, _ = resolve(rows("Echo", "Ann & Bob", DAYS[0:4]), rows("Echo", "Ann", DAYS[5:8]))
    assert set(resolved["track_credit"]) == {"Ann & Bob"}                    # 4 days against 3


def test_a_stale_second_capture_cannot_create_a_fake_common_day():
    # On DAYS[3] the date was captured twice. The first (stale) block still showed "Ann"; the second block, which is the
    # one that counts, shows "Ann & Bob". Counting the stale row would put "Ann" and "Ann & Bob" on the chart together.
    stale = rows("Echo", "Ann", DAYS[3:4], block=1)
    latest = rows("Echo", "Ann & Bob", DAYS[3:4], block=2)
    resolved, decisions = resolve(rows("Echo", "Ann & Bob", DAYS[0:3]), stale, latest, rows("Echo", "Ann", DAYS[5:7]))
    assert decisions["verdict"].tolist() == ["same song"]
    assert decisions.loc[0, "dates_on_chart_together"] == 0


def test_titles_with_one_credit_need_no_decision_and_titles_are_judged_separately():
    resolved, decisions = resolve(rows("Solo", "Zed", DAYS[0:3]),
                                  rows("Echo", "Ann & Bob", DAYS[0:4]), rows("Echo", "Ann", DAYS[5:8]),
                                  rows("Daylight", "David", DAYS[0:3]), rows("Daylight", "Drake", DAYS[5:8]))
    assert sorted(decisions["song"]) == ["Daylight", "Echo"]
    assert resolved["track_id"].nunique() == 1 + 1 + 2                       # Solo, Echo (merged), Daylight twice
