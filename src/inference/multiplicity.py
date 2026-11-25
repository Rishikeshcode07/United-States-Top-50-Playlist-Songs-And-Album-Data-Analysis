"""
src/inference/multiplicity.py  -  what to do when you run many tests at once (Bonferroni, Holm, Benjamini-Hochberg).

WHAT THIS FILE IS
    Three ways to turn the raw p-values of a family of tests into adjusted p-values:

        bonferroni            multiply every p-value by the number of tests (simple, and the most cautious)
        holm                  the step-down version of Bonferroni: never less powerful, controls the same error
        benjamini_hochberg    controls the false discovery rate instead (the expected share of false alarms among the
                              results we call significant), which allows more discoveries than Holm

    `adjust(p_values, method)` calls one of them by name. An adjusted p-value can be compared with the usual 0.05 directly.

WHY WE NEED IT
    A test with a 5% false-alarm rate is fine once. Run ten of them on data where nothing is going on and the chance that at
    least one comes out "significant" is 1 - 0.95^10 = 40%. The ten hypotheses of Phase 5 were found by looking at the same
    data, so the correction is not a formality here.

    * Holm controls the family-wise error rate: the chance of even one false alarm in the whole family stays at or below 5%.
      This is the rule of the project, because a false claim in the research paper costs more than a missed one.
    * Benjamini-Hochberg is shown next to it as the more generous view.

WHAT HAPPENS WHEN IT RUNS
    Each function takes a list of p-values and returns a numpy array of adjusted p-values in the SAME ORDER (capped at 1).

EXPECTED OUTCOME
    holm(p) >= p and bonferroni(p) >= holm(p) for every test; benjamini_hochberg(p) <= holm(p).

MOTIVE
    Say how many chances we gave ourselves to be fooled, and charge for them.
"""
from __future__ import annotations

import numpy as np


def _clean(p_values) -> np.ndarray:
    array = np.asarray(p_values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError("there are no p-values to adjust")
    if np.isnan(array).any() or (array < 0).any() or (array > 1).any():
        raise ValueError("p-values must be numbers between 0 and 1")
    return array


def bonferroni(p_values) -> np.ndarray:
    """Multiply each p-value by the number of tests (capped at 1)."""
    array = _clean(p_values)
    return np.minimum(array * array.size, 1.0)


def holm(p_values) -> np.ndarray:
    """
    Holm's step-down adjustment. Sort the p-values from the smallest; the k-th smallest is multiplied by (m - k + 1); the
    results are made non-decreasing (a bigger raw p-value can never get a smaller adjusted one).
    """
    array = _clean(p_values)
    m = array.size
    order = np.argsort(array, kind="stable")
    stepped = (m - np.arange(m)) * array[order]
    adjusted_sorted = np.minimum(np.maximum.accumulate(stepped), 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


def benjamini_hochberg(p_values) -> np.ndarray:
    """
    Benjamini-Hochberg step-up adjustment (the "q-values"). Sort from the LARGEST; the k-th smallest is multiplied by m / k;
    the results are made non-decreasing in the sorted order by taking running minimums from the top.
    """
    array = _clean(p_values)
    m = array.size
    order = np.argsort(array, kind="stable")
    stepped = array[order] * m / (np.arange(m) + 1.0)
    adjusted_sorted = np.minimum(np.minimum.accumulate(stepped[::-1])[::-1], 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


METHODS = {"bonferroni": bonferroni, "holm": holm, "bh": benjamini_hochberg, "benjamini-hochberg": benjamini_hochberg}


def adjust(p_values, method: str = "holm") -> np.ndarray:
    """Adjust a family of p-values by name: 'bonferroni', 'holm' or 'bh'."""
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}'. Choose one of {sorted(METHODS)}")
    return METHODS[method](p_values)


def family_wise_false_alarm_chance(number_of_tests: int, alpha: float = 0.05) -> float:
    """The chance of at least one false alarm among independent tests of true null hypotheses: 1 - (1 - alpha)^m."""
    if number_of_tests < 1:
        raise ValueError("there must be at least one test")
    return float(1.0 - (1.0 - alpha) ** number_of_tests)
