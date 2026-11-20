"""
tests/test_survival_cox.py  -  checks of the Cox model and the C-index (src/survival/cox.py).

WHAT THIS FILE IS
    The coefficients of the Cox model are compared with statsmodels (PHReg, Efron ties) and lifelines 0.30 (with a ridge penalty) on a
    fixed set of 90 songs with 13 different stay lengths, so with many ties (the numbers were calculated once and are written into the
    test). The C-index is compared with lifelines. Then: the bootstrap version equals the plain one for weights of 1 and equals repeating
    rows for whole-number weights, the cross-validated scores never saw the artist of the song they score (checked by changing that
    artist's outcomes and seeing that its scores do not move), and the scorer fills gaps and flags them.

WHY WE NEED IT
    "What can be told from the arrival day" is answered by these numbers; they must be right and they must not leak.

EXPECTED OUTCOME
    Coefficients to six digits, a C-index identical to lifelines, and scores that do not depend on the outcomes of the artist being scored.

MOTIVE
    A model that is tested on songs it has already seen looks wonderful and means nothing.
"""
import numpy as np
import pandas as pd
import pytest

from src.survival.cox import CoxScorer, concordance, concordance_pairs, cross_validated_risk, fit_cox, weighted_concordance


def toy(n=90):
    i = np.arange(n)
    time = 1 + (i * 7 + (i * i) % 11) % 13
    event = (i % 5) != 0
    x = np.column_stack([((i * 3) % 10) / 10.0, (((i * 5) % 7) - 3) / 3.0, (i % 2).astype(float)])
    return time, event, x, i % 18


def test_unpenalised_fit_agrees_with_statsmodels_phreg():
    time, event, X, _ = toy()
    fit = fit_cox(time, event, X, ridge=0.0)
    assert fit.converged and fit.n_events == 72
    assert fit.coef == pytest.approx([0.61499314, 0.06747636, -0.15095229], abs=1e-6)
    assert fit.log_likelihood == pytest.approx(-256.3455349501285, abs=1e-6)


@pytest.mark.parametrize("ridge, expected", [(1.0, [0.60414647, 0.06589998, -0.14705285]), (5.0, [0.56461112, 0.06027743, -0.13310325])])
def test_ridge_fit_agrees_with_lifelines_on_standardised_inputs(ridge, expected):
    # lifelines standardises the inputs (standard deviation with ddof = 1) and penalises the size of the standardised coefficients
    # by `penalizer` on the log-likelihood divided by n: the same as a ridge of n * penalizer on the standardised inputs here
    time, event, X, _ = toy()
    sd = X.std(axis=0, ddof=1)
    fit = fit_cox(time, event, (X - X.mean(axis=0)) / sd, ridge=ridge)
    assert fit.coef / sd == pytest.approx(expected, abs=2e-6)


def test_ridge_shrinks_towards_zero():
    time, event, X, _ = toy()
    sizes = [np.abs(fit_cox(time, event, X, ridge=r).coef).sum() for r in (0.0, 1.0, 10.0, 1000.0)]
    assert sizes == sorted(sizes, reverse=True) and sizes[-1] < 0.01 * sizes[0] + 0.01


def test_risk_is_the_linear_predictor():
    time, event, X, _ = toy()
    fit = fit_cox(time, event, X)
    assert fit.risk(X) == pytest.approx(X @ fit.coef)


def test_concordance_agrees_with_lifelines():
    time, event, X, _ = toy()
    risk = X[:, 0] * 2 - X[:, 1] + (np.arange(90) % 4) * 0.5                    # with ties in the risk score
    assert concordance(time, event, risk) == pytest.approx(0.4623937213865271, abs=1e-12)
    assert concordance(time, event, -risk) == pytest.approx(1 - 0.4623937213865271, abs=1e-12)


def test_concordance_by_hand():
    time, event = np.array([1, 2, 3]), np.array([True, True, False])
    assert concordance(time, event, np.array([3.0, 2.0, 1.0])) == 1.0           # the song that leaves first has the highest risk
    assert concordance(time, event, np.array([1.0, 1.0, 1.0])) == 0.5           # all tied: a coin flip
    pairs = concordance_pairs(time, event)
    assert sorted(zip(*pairs)) == [(0, 1), (0, 2), (1, 2)]
    # two songs that leave on the same day are not compared; a song that was still there on that day is
    i, j = concordance_pairs(np.array([2, 2, 2]), np.array([True, True, False]))
    assert sorted(zip(i, j)) == [(0, 2), (1, 2)]
    with pytest.raises(ValueError):
        concordance(np.array([1, 1]), np.array([True, True]), np.array([1.0, 2.0]))


def test_weights_of_one_give_the_plain_c_index_and_whole_number_weights_repeat_rows():
    time, event, X, _ = toy()
    risk = X[:, 0] - X[:, 1] + 0.1 * np.arange(90) % 3
    pairs = concordance_pairs(time, event)
    assert weighted_concordance(pairs, risk, np.ones(90)) == pytest.approx(concordance(time, event, risk))
    w = (np.arange(90) % 3).astype(float)
    rep = np.repeat(np.arange(90), w.astype(int))
    assert weighted_concordance(pairs, risk, w) == pytest.approx(concordance(time[rep], event[rep], risk[rep]))
    many = weighted_concordance(pairs, risk, np.vstack([np.ones(90), w]))
    assert many.shape == (2,) and many[1] == pytest.approx(weighted_concordance(pairs, risk, w))
    assert np.isnan(weighted_concordance(pairs, risk, np.zeros(90)))


def frame_of(n=200):
    i = np.arange(n)
    return pd.DataFrame({"a": ((i * 7) % 13) / 13.0, "b": ((i * 11) % 17).astype(float), "gap": np.where(i % 9 == 0, np.nan, ((i * 5) % 8).astype(float))}), i % 20


def test_cross_validated_scores_never_saw_the_artist_they_score():
    frame, artist = frame_of()
    n = len(frame)
    time = 1 + (np.arange(n) * 7 + (np.arange(n) ** 2) % 11) % 13
    event = (np.arange(n) % 5) != 0
    kept = cross_validated_risk(frame, time, event, artist, ["a", "b", "gap"], n_splits=5, n_repeats=2, seed=1)
    assert kept.shape == (2, n) and np.isfinite(kept).all()
    changed_event, changed_time = event.copy(), time.copy()
    mine = artist == 3
    changed_event[mine] = ~changed_event[mine]
    changed_time[mine] = 14 - changed_time[mine]
    again = cross_validated_risk(frame, changed_time, changed_event, artist, ["a", "b", "gap"], n_splits=5, n_repeats=2, seed=1)
    # the models that score the songs of artist 3 are trained on the other artists only: changing artist 3's outcomes cannot move those scores ...
    assert np.allclose(again[:, mine], kept[:, mine], atol=1e-12)
    # ... while the scores of everybody else do move, because their models were trained with artist 3's (changed) songs
    assert not np.allclose(again[:, ~mine], kept[:, ~mine], atol=1e-6)


def test_cross_validated_scores_are_repeatable():
    frame, artist = frame_of()
    n = len(frame)
    time, event = 1 + (np.arange(n) * 7) % 13, (np.arange(n) % 4) != 0
    one = cross_validated_risk(frame, time, event, artist, ["a", "b"], n_repeats=2, seed=5)
    two = cross_validated_risk(frame, time, event, artist, ["a", "b"], n_repeats=2, seed=5)
    assert np.array_equal(one, two) and not np.array_equal(one[0], one[1])


def test_scorer_fills_gaps_flags_them_and_scales_with_the_training_songs():
    frame, _ = frame_of()
    n = len(frame)
    time, event = 1 + (np.arange(n) * 7) % 13, (np.arange(n) % 4) != 0
    scorer = CoxScorer(["a", "b", "gap"], log_columns=["b"]).fit(frame, time, event)
    assert scorer.flagged_ == ["gap"] and scorer.fit_.coef.shape == (4,)
    new = frame.iloc[:5].copy()
    new.loc[new.index[0], "gap"] = np.nan
    assert np.isfinite(scorer.score(new)).all()
    with pytest.raises(RuntimeError):
        CoxScorer(["a"]).score(frame)


def test_bad_input_is_refused():
    time, event, X, _ = toy()
    with pytest.raises(ValueError):
        fit_cox(time[:-1], event, X)
    with pytest.raises(ValueError):
        fit_cox(time, np.zeros(90, dtype=bool), X)
    with pytest.raises(ValueError):
        fit_cox(time, event, X, ridge=-1)
    with pytest.raises(ValueError):
        fit_cox(time, event, np.where(X > 0.5, np.nan, X))
