"""
src/inference/survival.py  -  how long does a song stay on the chart, when some songs are still there at the end?

WHAT THIS FILE IS
    Two small tools for "time until an event" data:

        kaplan_meier    the survival curve: after t days, what share of songs is still on the chart?
        logrank_test    do two groups of songs leave the chart at the same rate?

WHY WE NEED IT
    A song that is still on the chart on the last day of the file has not finished its stay: we only know that it lasted AT LEAST
    that long. Averages and medians of "days on chart" that treat such a song as finished are too short; dropping such songs
    throws away exactly the long stays. Survival analysis keeps them: a "censored" song counts as being at risk until the day
    it was last seen, and does not count as an exit.

    Phase 6 uses the same idea for the models of chart lifetime; here it gives a check of the "singles stay longer" question that
    handles the end of the file correctly.

WHAT HAPPENS WHEN IT RUNS
    `kaplan_meier` returns a table with one row per day on which something happened. `logrank_test` returns the chi-square test
    of equal exit rates with the observed and expected exits of each group.

EXPECTED OUTCOME
    The curve starts at 1 and only goes down; with no censoring it is the share of songs that stayed longer than t. The log-rank
    test agrees with scipy.stats.logrank (the tests check this).

MOTIVE
    Use every song, including the ones that are still running.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.inference.results import InferenceResult


def _clean(durations, observed):
    t = np.asarray(durations, dtype=float).ravel()
    e = np.asarray(observed, dtype=bool).ravel()
    if len(t) == 0 or len(t) != len(e):
        raise ValueError("durations and observed must be lists of the same length (at least one entry)")
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError("durations must be finite and not negative")
    return t, e


def kaplan_meier(durations, observed, confidence: float = 0.95) -> pd.DataFrame:
    """
    The Kaplan-Meier estimate. `durations` = days until the exit or until the song was last seen; `observed` = True when the exit
    was seen, False when the song was still there (censored). One row per distinct time at which an exit happened:

        at_risk    songs still being followed just before that time
        events     exits at that time
        survival   share still on the chart after that time = product over earlier times of (1 - events / at_risk)
        ci_low, ci_high   Greenwood interval on the log(-log) scale, which stays between 0 and 1

    A first row at time 0 with survival 1 is added so that the curve can be drawn from the start.
    """
    t, e = _clean(durations, observed)
    times = np.unique(t[e])
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    rows = [{"time": 0.0, "at_risk": len(t), "events": 0, "survival": 1.0, "ci_low": 1.0, "ci_high": 1.0}]
    survival, greenwood = 1.0, 0.0
    for time in times:
        at_risk = int((t >= time).sum())
        events = int(((t == time) & e).sum())
        survival *= 1.0 - events / at_risk
        if at_risk > events:
            greenwood += events / (at_risk * (at_risk - events))
        if 0.0 < survival < 1.0:
            se_log_log = np.sqrt(greenwood) / abs(np.log(survival))
            low, high = survival ** np.exp(z * se_log_log), survival ** np.exp(-z * se_log_log)
        else:
            low = high = survival
        rows.append({"time": float(time), "at_risk": at_risk, "events": events, "survival": float(survival), "ci_low": float(low), "ci_high": float(high)})
    return pd.DataFrame(rows)


def median_survival(curve: pd.DataFrame) -> float:
    """The first time at which the survival curve is at or below one half (NaN when it never gets there: more than half are still running)."""
    below = curve.loc[curve["survival"] <= 0.5, "time"]
    return float(below.iloc[0]) if len(below) else float("nan")


def logrank_test(durations_a, observed_a, durations_b, observed_b, name: str = "log-rank test") -> InferenceResult:
    """
    The log-rank test of "both groups leave the chart at the same rate at every moment". At each time on which an exit happens,
    the exits are shared out between the groups in proportion to who is at risk; the test adds up observed minus expected exits of
    group A over all those times and divides by the variance:
        chi2 = (O_A - E_A)^2 / V     (1 degree of freedom)
    The estimate is the hazard ratio (O_A / E_A) / (O_B / E_B): above 1 means that group A leaves faster.
    """
    ta, ea = _clean(durations_a, observed_a)
    tb, eb = _clean(durations_b, observed_b)
    times = np.unique(np.concatenate([ta[ea], tb[eb]]))
    observed_total_a = expected_a = variance = 0.0
    observed_total_b = expected_b = 0.0
    for time in times:
        n_a, n_b = float((ta >= time).sum()), float((tb >= time).sum())
        d_a, d_b = float(((ta == time) & ea).sum()), float(((tb == time) & eb).sum())
        n, d = n_a + n_b, d_a + d_b
        observed_total_a += d_a
        observed_total_b += d_b
        expected_a += d * n_a / n
        expected_b += d * n_b / n
        if n > 1:
            variance += d * (n_a / n) * (n_b / n) * (n - d) / (n - 1)
    chi2 = float((observed_total_a - expected_a) ** 2 / variance)
    ratio = (observed_total_a / expected_a) / (observed_total_b / expected_b)
    return InferenceResult(name, chi2, float(stats.chi2.sf(chi2, 1)), df=1, estimate=float(ratio), n=len(ta) + len(tb),
                           detail={"observed_a": observed_total_a, "expected_a": expected_a, "observed_b": observed_total_b, "expected_b": expected_b})
