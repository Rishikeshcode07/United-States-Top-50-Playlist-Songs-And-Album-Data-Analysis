"""
src/inference/results.py  -  one common shape for the result of a test, and plain words for the size of an effect.

WHAT THIS FILE IS
    InferenceResult   a small record that every test function of this package returns: the name of the test, the test
                      statistic, the degrees of freedom, the p-value, the estimate (the effect in its own units, for example
                      "+7 percentage points") with its confidence interval, and the number of rows used. `detail` holds anything
                      extra (expected counts, group means, the standard error ...).
    effect_label      turns an effect size into one of the words negligible / small / medium / large, using the usual benchmark
                      of each measure (Cohen's h, Cohen's d, Cliff's delta, eta squared, Cramer's V ...).

WHY WE NEED IT
    Ten different tests would otherwise return ten different shapes (a tuple here, a named tuple there, a bare p-value
    somewhere else), and every cell of the notebook would have to remember which is which. With one record, the notebook can
    treat every test the same way: read `.p_value`, read `.estimate`, put the row into the register.

    A p-value says whether an effect is detectable; it does not say whether it matters. A big data set can make a tiny
    difference "significant", and a small data set can hide a large one. So every test in this project is reported with an
    effect size and a confidence interval, and the effect size gets a plain word through `effect_label`.

WHAT HAPPENS WHEN IT RUNS
    Nothing by itself. The other modules of `src/inference` create `InferenceResult` objects; the notebook reads them.

EXPECTED OUTCOME
    `InferenceResult(...)` objects that print readably and can be turned into a dictionary with `.as_dict()`.

MOTIVE
    Report every test the same way: how big, how sure, how many rows, and which test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The usual lower bounds for "small", "medium" and "large" (Cohen, 1988; Romano et al., 2006 for Cliff's delta).
# They are rules of thumb for orientation, not laws: what counts as a big effect depends on what the effect is worth.
EFFECT_BENCHMARKS = {
    "cohens_h": (0.2, 0.5, 0.8),               # difference between two proportions
    "cohens_d": (0.2, 0.5, 0.8),               # difference between two means in standard deviations
    "hedges_g": (0.2, 0.5, 0.8),               # Cohen's d with a small-sample correction
    "cliffs_delta": (0.147, 0.33, 0.474),      # how often one group beats the other, from -1 to +1
    "eta_squared": (0.01, 0.06, 0.14),         # share of the variance that the groups explain
    "partial_eta_squared": (0.01, 0.06, 0.14),
    "epsilon_squared": (0.01, 0.06, 0.14),     # the rank-based cousin of eta squared (Kruskal-Wallis)
    "omega_squared": (0.01, 0.06, 0.14),
    "cramers_v": (0.1, 0.3, 0.5),              # strength of a link between two categories
    "cohens_w": (0.1, 0.3, 0.5),               # distance between observed and expected shares
    "correlation": (0.1, 0.3, 0.5),            # Pearson, Spearman, Kendall
    "kendalls_w": (0.1, 0.3, 0.5),             # agreement between rankings
}

LABELS = ("negligible", "small", "medium", "large")


def effect_label(metric: str, value: float) -> str:
    """The word for an effect size: 'negligible', 'small', 'medium' or 'large' (the sign is ignored)."""
    if metric not in EFFECT_BENCHMARKS:
        raise ValueError(f"unknown effect-size measure '{metric}'. Known: {sorted(EFFECT_BENCHMARKS)}")
    magnitude = abs(float(value))
    small, medium, large = EFFECT_BENCHMARKS[metric]
    if magnitude < small:
        return LABELS[0]
    if magnitude < medium:
        return LABELS[1]
    if magnitude < large:
        return LABELS[2]
    return LABELS[3]


@dataclass(frozen=True)
class InferenceResult:
    """The result of one test. Only `name`, `statistic` and `p_value` are always there."""

    name: str
    statistic: float
    p_value: float
    df: object = None                       # a number, or a pair of numbers for an F test
    estimate: float | None = None           # the effect in its own units
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        row = {"name": self.name, "statistic": self.statistic, "df": self.df, "p_value": self.p_value,
               "estimate": self.estimate, "ci_low": self.ci_low, "ci_high": self.ci_high, "n": self.n}
        row.update(self.detail)
        return row

    def __str__(self) -> str:
        pieces = [f"{self.name}: statistic = {self.statistic:.4g}"]
        if self.df is not None:
            pieces.append(f"df = {self.df}")
        pieces.append(f"p = {self.p_value:.3g}")
        if self.estimate is not None:
            interval = "" if self.ci_low is None else f" [{self.ci_low:.4g}, {self.ci_high:.4g}]"
            pieces.append(f"estimate = {self.estimate:.4g}{interval}")
        return ", ".join(pieces)
