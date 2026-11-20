"""
src/inference/power.py  -  how big an effect could we have seen, and how to show that two things are "the same".

WHAT THIS FILE IS
    Power and sample-size tools for the two kinds of comparison used in the project:

        two proportions      power_two_proportions, minimum_detectable_difference, required_n_per_group
        two means            power_two_sample_t, minimum_detectable_effect_t
        equivalence          tost_equivalence ("two one-sided tests": is the difference small enough to call it no difference?)

WHY WE NEED IT
    "Not significant" does not mean "no effect". It can also mean "we looked with too small a magnifying glass". The power of a
    test is the chance that it finds an effect of a given size when that effect is real. A study with 122 collaborations and 822
    solo songs can only see a difference of more than about 13 percentage points; a smaller one stays hidden whether it exists or not.
    So for every result that is NOT significant, the notebook reports the smallest effect that the data could have detected
    (the "minimum detectable effect") and, where it makes sense, an equivalence test.

    Rows that belong to the same artist are not independent, so they carry less information than their number suggests. The
    `design_effect` argument shrinks the sample sizes to the "effective" size (n / design effect) before power is calculated.

WHAT HAPPENS WHEN IT RUNS
    Pure calculation with normal and non-central t distributions; no random numbers.

EXPECTED OUTCOME
    Power between the significance level (no effect) and 1 (a huge effect); a smaller minimum detectable effect for a bigger
    sample; the textbook numbers (d = 0.5 with 64 per group gives 80% power).

MOTIVE
    Make "we found nothing" a statement with a size on it.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats

from src.inference.results import InferenceResult


def _check_alpha_power(alpha: float, power: float | None = None) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and not 0 < power < 1:
        raise ValueError("power must be between 0 and 1")


# ================================================================================================ two proportions
def power_two_proportions(p1: float, p2: float, n1: float, n2: float, alpha: float = 0.05, design_effect: float = 1.0) -> float:
    """
    The chance that a two-sided two-proportion z test (the pooled-variance one) rejects "equal" when the true proportions are p1
    and p2. `n1` and `n2` are shrunk by the design effect first. Normal approximation:
        power = Phi((d - z * se0) / se1) + Phi((-d - z * se0) / se1)
    with d = |p1 - p2|, se0 the standard error under "equal" (pooled) and se1 the standard error under the stated difference.
    """
    _check_alpha_power(alpha)
    if design_effect < 1.0 - 1e-12:
        raise ValueError("the design effect cannot be below 1")
    n1, n2 = n1 / design_effect, n2 / design_effect
    pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    se1 = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = float(stats.norm.ppf(1 - alpha / 2))
    d = abs(p1 - p2)
    return float(stats.norm.cdf((d - z * se0) / se1) + stats.norm.cdf((-d - z * se0) / se1))


def minimum_detectable_difference(p_base: float, n1: float, n2: float, alpha: float = 0.05, power: float = 0.8, design_effect: float = 1.0) -> float:
    """
    The smallest INCREASE over `p_base` that the test would find with the stated power: the difference d for which
    power_two_proportions(p_base + d, p_base, n1, n2) equals `power`. (A decrease of the same size is slightly different because
    the variance of a proportion depends on where it is.)
    """
    _check_alpha_power(alpha, power)
    upper = 1.0 - p_base - 1e-9
    if power_two_proportions(p_base + upper, p_base, n1, n2, alpha, design_effect) < power:
        return float("nan")                                           # not even the largest possible difference is enough
    return float(optimize.brentq(lambda d: power_two_proportions(p_base + d, p_base, n1, n2, alpha, design_effect) - power, 1e-9, upper))


def required_n_per_group(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0) -> tuple[int, int]:
    """
    How many rows each group needs to detect the difference between p1 and p2 with the stated power (two-sided z test). `ratio` is
    the size of group 2 divided by the size of group 1. Returns (n1, n2).
    """
    _check_alpha_power(alpha, power)
    if p1 == p2:
        raise ValueError("the two proportions must differ")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    pooled = (p1 + ratio * p2) / (1 + ratio)
    numerator = (z_alpha * math.sqrt((1 + 1 / ratio) * pooled * (1 - pooled)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)) ** 2
    n1 = math.ceil(numerator / (p1 - p2) ** 2)
    return n1, math.ceil(n1 * ratio)


# ================================================================================================ two means
def power_two_sample_t(effect_d: float, n1: float, n2: float, alpha: float = 0.05, design_effect: float = 1.0) -> float:
    """
    The power of a two-sided two-sample t test (equal variances) for a true standardised difference `effect_d` (Cohen's d), from
    the non-central t distribution: the t statistic is centred on d * sqrt(n1 n2 / (n1 + n2)) instead of on zero.
    """
    _check_alpha_power(alpha)
    n1, n2 = n1 / design_effect, n2 / design_effect
    df = n1 + n2 - 2
    centre = abs(effect_d) * math.sqrt(n1 * n2 / (n1 + n2))
    critical = float(stats.t.ppf(1 - alpha / 2, df))
    return float(stats.nct.sf(critical, df, centre) + stats.nct.cdf(-critical, df, centre))


def minimum_detectable_effect_t(n1: float, n2: float, alpha: float = 0.05, power: float = 0.8, design_effect: float = 1.0) -> float:
    """
    The smallest standardised difference (Cohen's d) that a two-sample t test finds with the stated power.

    The search starts with a small d and widens until the power is reached. A fixed upper limit would not do: for very small samples
    (nine months against ten months, say) the non-central t distribution cannot be evaluated at a large d and returns "not a number",
    and power at such a d is 1 in any case, so a failed evaluation counts as full power.
    """
    _check_alpha_power(alpha, power)

    def reached(d: float) -> float:
        value = power_two_sample_t(d, n1, n2, alpha, design_effect)
        return 1.0 if not math.isfinite(value) else value

    upper = 0.25
    while reached(upper) < power:
        upper *= 1.5
        if upper > 1_000:
            return float("nan")
    return float(optimize.brentq(lambda d: reached(d) - power, 1e-6, upper))


# ================================================================================================ equivalence
def tost_equivalence(estimate: float, se: float, margin: float, df: float | None = None, alpha: float = 0.05,
                     name: str = "TOST equivalence test") -> InferenceResult:
    """
    "Two one-sided tests". To claim that a difference is negligible, a non-significant result is not enough; we test two things:
        1. the difference is above  -margin   (otherwise the groups could differ by more than the margin downwards)
        2. the difference is below  +margin   (otherwise they could differ by more than the margin upwards)
    The p-value of the equivalence test is the LARGER of the two one-sided p-values. Equivalence is shown when it is below alpha,
    which is the same as the (1 - 2 alpha) confidence interval lying completely inside (-margin, +margin).
    `df` = None uses the normal distribution, a number uses the t distribution (use G - 1 for cluster-robust standard errors).
    """
    _check_alpha_power(alpha)
    if se <= 0 or margin <= 0:
        raise ValueError("the standard error and the margin must be above zero")
    distribution = stats.norm if df is None else stats.t(df)
    z_lower = (estimate + margin) / se                       # against "difference <= -margin"
    z_upper = (estimate - margin) / se                       # against "difference >= +margin"
    p_lower = float(distribution.sf(z_lower))
    p_upper = float(distribution.cdf(z_upper))
    critical = float(distribution.ppf(1 - alpha))
    low, high = estimate - critical * se, estimate + critical * se
    p_value = max(p_lower, p_upper)
    return InferenceResult(name, float(z_lower if p_lower >= p_upper else z_upper), p_value, df=df, estimate=float(estimate),
                           ci_low=float(low), ci_high=float(high),
                           detail={"margin": margin, "p_lower": p_lower, "p_upper": p_upper, "equivalent": bool(p_value < alpha),
                                   "confidence_level": 1 - 2 * alpha})
