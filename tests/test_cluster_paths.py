"""
tests/test_cluster_paths.py  -  checks of the first-month table of the songs (src/cluster/paths.py).

WHAT THIS FILE IS
    A chart of four songs worked out by hand: a song that is off the chart on some days, a song that comes back after a gap, a song that arrives so
    late that its month is unfinished (dropped), a song whose month ends exactly on the last snapshot day (kept), and chart rows of a song that is
    not in the list (ignored). The result must not depend on the order of the rows or of the songs.

WHY WE NEED IT
    Every song group of notebook 06c is a group of rows of this table.

EXPECTED OUTCOME
    The matrix equals the one written out by hand: score 51 - place on a chart day, 0 on an absent day.

MOTIVE
    One wrong column shift would move every song one day and the groups would still look plausible.
"""
import numpy as np
import pandas as pd

from src.cluster.paths import chart_score, first_month_matrix

SONGS = pd.DataFrame({"track_id": ["a", "b", "c", "d"], "entry_snapshot": [0, 4, 8, 7]})
CHART = pd.DataFrame({
    "track_id": ["a", "a", "a", "b", "b", "c", "d", "d", "d", "zz", "a"],
    "s":        [0,   1,   3,   4,   6,   8,   7,   8,   9,   5,    9],
    "position": [1,   50,  5,   10,  20,  2,   3,   3,   3,   1,    7],
})
N_SNAPSHOTS, HORIZON = 10, 3


def test_chart_score_is_51_minus_place():
    assert list(chart_score(np.array([1, 2, 25, 50]))) == [50, 49, 26, 1]
    assert list(chart_score(np.array([1, 10]), top=10)) == [10, 1]


def test_matrix_by_hand():
    matrix, kept = first_month_matrix(CHART, SONGS, N_SNAPSHOTS, horizon=HORIZON)
    assert list(kept) == [True, True, False, True]                                     # song c arrives on day 8: 8 + 3 > 10
    expected = np.array([[50, 1, 0],                                                   # a: place 1, place 50, absent (s = 2); s = 3 and 9 are outside the month
                         [41, 0, 31],                                                  # b: place 10, absent, place 20
                         [48, 48, 48]])                                                # d: arrives on 7, 7 + 3 = 10 = N_SNAPSHOTS: still kept
    assert np.array_equal(matrix, expected)


def test_rows_of_songs_that_are_not_listed_are_ignored_and_the_order_of_rows_does_not_matter():
    shuffled = CHART.sample(frac=1.0, random_state=2).reset_index(drop=True)
    a, _ = first_month_matrix(CHART, SONGS, N_SNAPSHOTS, horizon=HORIZON)
    b, _ = first_month_matrix(shuffled, SONGS, N_SNAPSHOTS, horizon=HORIZON)
    assert np.array_equal(a, b)
    assert a.shape == (3, 3)


def test_the_rows_follow_the_order_of_the_song_list():
    songs = SONGS.iloc[[3, 0, 1]].reset_index(drop=True)
    matrix, kept = first_month_matrix(CHART, songs, N_SNAPSHOTS, horizon=HORIZON)
    assert list(kept) == [True, True, True]
    assert np.array_equal(matrix, np.array([[48, 48, 48], [50, 1, 0], [41, 0, 31]]))


def test_a_longer_month_and_the_first_column_is_the_arrival_day():
    matrix, kept = first_month_matrix(CHART, SONGS, N_SNAPSHOTS, horizon=6)
    assert list(kept) == [True, True, False, False]                                    # d: 7 + 6 > 10
    assert matrix.shape == (2, 6) and (matrix[:, 0] > 0).all()
    assert list(matrix[0]) == [50, 1, 0, 46, 0, 0]                                     # a returns on s = 3 at place 5
