"""
tests/test_survival_estimators.py  -  checks of the survival curve, hazard and log-rank test (src/survival/estimators.py).

WHAT THIS FILE IS
    A stay of five songs worked out by hand, the Kaplan-Meier curve compared with the one of src/inference/survival.py on 300 songs with many ties,
    weights compared with repeating rows, the window hazard, the median and restricted mean, and the log-rank test compared with lifelines
    (the numbers were calculated once with lifelines 0.30 and are written into the test). For the artist-aware test: duplicating every song inside
    its own artist inflates the usual test by the number of copies and leaves the artist-aware one unchanged.

WHY WE NEED IT
    The curves, bands, medians and p-values of the notebook come from these functions.

EXPECTED OUTCOME
    Agreement to at least nine digits with the references.

MOTIVE
    A curve that is a little off looks exactly like a curve that is right, so the numbers are checked.
"""
import numpy as np
import pytest

from src.inference.survival import kaplan_meier
from src.survival.estimators import (life_table, logrank_k, median_further_stay, median_stay, restricted_mean, survival_grid,
                                     window_hazard, window_labels)


def many(n=300):
    i = np.arange(n)
    t = 1 + (i * 7 + (i * i) % 11) % 40 + (i % 3 == 0) * 2
    e = (i % 7) != 0
    return t, e, i % 25


def test_five_songs_by_hand():
    t, e = [1, 2, 2, 3, 5], [1, 1, 0, 1, 0]
    s = survival_grid(t, e, [0, 1, 2, 3, 4, 5, 9])
    assert np.allclose(s, [1, 0.8, 0.6, 0.3, 0.3, 0.3, 0.3])                      # after the last stay the curve stays where it is


def test_agrees_with_the_kaplan_meier_of_the_inference_module():
    t, e, _ = many()
    km = kaplan_meier(t, e)
    for age in (1, 3, 8, 15, 30, 41):
        expected = km.loc[km["time"] <= age, "survival"].iloc[-1]
        assert survival_grid(t, e, [age])[0] == pytest.approx(expected, abs=1e-12)


def test_integer_weights_are_the_same_as_repeating_rows():
    t, e, _ = many(60)
    w = (np.arange(60) % 3).astype(float)                                          # 0, 1 or 2 copies
    rep = np.repeat(np.arange(60), w.astype(int))
    grid = np.arange(0, 45)
    assert np.allclose(survival_grid(t, e, grid, weights=w), survival_grid(t[rep], e[rep], grid))
    for edges in ([0, 2, 5, 12, 10000],):
        got = window_hazard(t, e, edges, weights=w)
        want = window_hazard(t[rep], e[rep], edges)
        assert all(np.allclose(a, b) for a, b in zip(got, want))


def test_a_matrix_of_weights_gives_one_row_per_draw():
    t, e, _ = many(80)
    W = np.vstack([np.ones(80), (np.arange(80) % 2).astype(float), np.arange(80) % 3])
    grid = [0, 4, 9, 20]
    both = survival_grid(t, e, grid, weights=W)
    assert both.shape == (3, 4)
    for r in range(3):
        assert np.allclose(both[r], survival_grid(t, e, grid, weights=W[r]))
    exits, days, hazard = window_hazard(t, e, [0, 3, 10, 10000], weights=W)
    assert exits.shape == days.shape == hazard.shape == (3, 3)


def test_window_hazard_by_hand():
    t, e = [1, 2, 2, 3, 5], [1, 1, 0, 1, 0]
    exits, days, hazard = window_hazard(t, e, [0, 1, 3, 10000])
    # day 1: all five songs are there: 5 chart days, 1 exit. Window (1, 3]: day 2 holds 4 songs, day 3 holds 2: 6 chart days, 2 exits. Window (3, ...): the song with 5 days has days 4 and 5: 2 chart days, no exit
    assert list(exits) == [1, 2, 0] and list(days) == [5, 6, 2]
    assert hazard[0] == pytest.approx(0.2) and hazard[1] == pytest.approx(1 / 3) and hazard[2] == 0


def test_window_exits_add_up_to_all_exits_and_days_to_all_chart_days():
    t, e, _ = many()
    exits, days, _ = window_hazard(t, e, [0, 1, 2, 3, 7, 14, 30, 60, 120, 10000])
    assert exits.sum() == e.sum() and days.sum() == t.sum()


def test_life_table_and_labels():
    t, e, _ = many()
    table = life_table(t, e, [0, 1, 2, 3, 7, 14, 30, 60, 120, 10000])
    assert list(table["window"][:5]) == ["1", "2", "3", "4-7", "8-14"] and table["window"].iloc[-1] == "121+"
    assert table["exits"].sum() == e.sum() and table["songs_reaching"].iloc[0] == len(t)
    assert window_labels([0, 1, 4, 10000]) == ["1", "2-4", "5+"]


def test_median_and_restricted_mean():
    s = np.array([1, 0.8, 0.6, 0.3, 0.3])
    assert median_stay(s) == 3
    assert np.isnan(median_stay(np.array([1, 0.9, 0.8])))
    assert list(median_stay(np.vstack([s, [1, 0.4, 0.3, 0.2, 0.1]]))) == [3, 1]
    assert median_further_stay(s, 1) == 2                                           # from 0.8 down to 0.4: S(3) = 0.3
    assert median_further_stay(s, 0) == 3
    assert np.isnan(median_further_stay(np.array([1, 0.9, 0.8]), 1))
    assert restricted_mean(s, 3) == pytest.approx(1 + 0.8 + 0.6)
    assert list(restricted_mean(np.vstack([s, s]), 2)) == [1.8, 1.8]
    with pytest.raises(ValueError):
        restricted_mean(s, 9)


# lifelines 0.30 on the toy data below
LIFELINES = {3: (0.2210976356502556, 0.8953426204431131), 2: (0.06928084431717106, 0.792386761429135)}


def toy(n=90):
    i = np.arange(n)
    return 1 + (i * 7 + (i * i) % 11) % 13, (i % 5) != 0


@pytest.mark.parametrize("k", [2, 3])
def test_logrank_agrees_with_lifelines(k):
    t, e = toy()
    result = logrank_k(t, e, np.arange(len(t)) % k)
    assert result["chi2"] == pytest.approx(LIFELINES[k][0], rel=1e-9)
    assert result["p"] == pytest.approx(LIFELINES[k][1], rel=1e-9)
    assert result["df"] == k - 1 and result["observed"].sum() == pytest.approx(e.sum()) and result["expected"].sum() == pytest.approx(e.sum())


def test_logrank_two_groups_agrees_with_the_inference_module():
    from src.inference.survival import logrank_test
    t, e, artist = many()
    g = artist % 2
    mine = logrank_k(t, e, g)
    theirs = logrank_test(t[g == 0], e[g == 0], t[g == 1], e[g == 1])
    assert mine["chi2"] == pytest.approx(theirs.statistic if hasattr(theirs, "statistic") else theirs["statistic"], rel=1e-9)


def test_artist_aware_logrank_does_not_grow_when_songs_are_copied_inside_their_artist():
    t, e, artist = many(200)
    g = (np.arange(200) % 2)
    base = logrank_k(t, e, g, clusters=artist)
    copies = 4
    t4, e4, g4, a4 = (np.tile(x, copies) for x in (t, e, g, artist))
    big = logrank_k(t4, e4, g4, clusters=a4)
    assert big["chi2"] == pytest.approx(copies * base["chi2"], rel=0.05)                # the usual test counts every copy as new evidence
    assert big["chi2_artist"] == pytest.approx(base["chi2_artist"], rel=1e-9)           # the artist-aware one does not


def test_artist_aware_logrank_is_close_to_the_usual_one_when_every_song_is_its_own_cluster():
    t, e, _ = many(300)
    g = np.arange(300) % 3
    r = logrank_k(t, e, g, clusters=np.arange(300))
    assert r["chi2_artist"] == pytest.approx(r["chi2"], rel=0.25)


def test_bad_input_is_refused():
    with pytest.raises(ValueError):
        survival_grid([0, 1], [1, 1], [1])                                              # a stay of 0 days
    with pytest.raises(ValueError):
        survival_grid([1.5, 2], [1, 1], [1])
    with pytest.raises(ValueError):
        survival_grid([1, 2], [1], [1])
    with pytest.raises(ValueError):
        window_hazard([1, 2], [1, 1], [1, 2])                                           # must start at 0
    with pytest.raises(ValueError):
        logrank_k([1, 2, 3], [1, 1, 1], [0, 0, 0])
    with pytest.raises(ValueError):
        survival_grid([1, 2], [1, 1], [-1])
