"""
src/inference/register.py  -  the register of hypotheses: one row per question, with its test, its effect and its verdict.

WHAT THIS FILE IS
    HypothesisRegister   a list of the ten hypotheses of Phase 5. For each one it keeps the question, the unit of analysis, the
                         planned (classical) test with its p-value, the primary test that allows for related rows with its p-value,
                         the effect in its own units with an interval, a standardised effect size with a plain word, and a caveat.
                         When the table is built it adds the Holm-adjusted p-value, the Benjamini-Hochberg q-value and the verdict.
    format_p             writes a p-value the way a paper does ("p < 0.001", "p = 0.034").
    save_table           writes any small table as a CSV file the same careful way (used for the robustness table).

WHY WE NEED IT
    Ten separate test results are easy to lose and easy to misquote. One register says in one table what was tested, how, what
    came out, how big it is and what to be careful about. It is saved as reports/p5_hypothesis_register.csv and is what the
    dashboards and the research paper of the later phases read.

    The verdict follows a rule that is written down BEFORE the results are seen:
        Supported       the Holm-adjusted p-value is below 0.05: the claim survives the correction for ten tests
        Suggestive      the raw p-value is below 0.05 but the adjusted one is not: a lead worth following up on new data
        Not supported   the raw p-value is 0.05 or more: no evidence of an effect here (which is not the same as evidence of
                        no effect; see the power column of the notebook)

WHAT HAPPENS WHEN IT RUNS
    `add` stores a row (it refuses an empty field). `to_frame` returns the table with the adjusted p-values and verdicts.
    `save` writes it as a CSV file (UTF-8 with a byte-order mark so that Excel on Windows shows the text correctly).

EXPECTED OUTCOME
    reports/p5_hypothesis_register.csv with ten rows.

MOTIVE
    Every claim in the project can be traced to a test, a number and a verdict.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.inference.multiplicity import benjamini_hochberg, holm

COLUMNS = ["hypothesis", "question", "unit", "n", "planned_test", "planned_p", "primary_test", "primary_p", "holm_p", "bh_q",
           "effect", "effect_low", "effect_high", "effect_unit", "standard_metric", "standard_size", "size_word", "verdict", "caveat"]
REQUIRED = ["hypothesis", "question", "unit", "n", "planned_test", "planned_p", "primary_test", "primary_p", "effect", "effect_unit",
            "standard_metric", "standard_size", "size_word", "caveat"]


def format_p(p_value: float) -> str:
    """0.00004 -> 'p < 0.001', 0.034 -> 'p = 0.034', 0.42 -> 'p = 0.42'."""
    if p_value < 0.001:
        return "p < 0.001"
    if p_value < 0.1:
        return f"p = {p_value:.3f}"
    return f"p = {p_value:.2f}"


def verdict(raw_p: float, adjusted_p: float, alpha: float = 0.05) -> str:
    """The verdict rule of the project (see the module text)."""
    if adjusted_p < alpha:
        return "Supported"
    if raw_p < alpha:
        return "Suggestive"
    return "Not supported"


def save_table(frame: pd.DataFrame, path) -> None:
    """Write a table as a CSV file (UTF-8 with a byte-order mark, so that Excel on Windows shows the text correctly). It is written to a
    temporary file first and moved into place in one step, so that a crash cannot leave half a file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class HypothesisRegister:
    """The ten rows of Phase 5. `add` stores a row; `to_frame` adds the adjusted p-values and the verdicts."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self._rows: list[dict] = []

    def add(self, **fields) -> None:
        unknown = set(fields) - set(COLUMNS)
        if unknown:
            raise ValueError(f"unknown field(s): {sorted(unknown)}")
        for name in REQUIRED:
            value = fields.get(name)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"a hypothesis needs a non-empty '{name}'")
        if any(row["hypothesis"] == fields["hypothesis"] for row in self._rows):
            raise ValueError(f"the hypothesis {fields['hypothesis']} is already in the register")
        for name in ("planned_p", "primary_p"):
            if not 0.0 <= float(fields[name]) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        self._rows.append(dict(fields))

    def __len__(self) -> int:
        return len(self._rows)

    def to_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=COLUMNS)
        frame = pd.DataFrame(self._rows)
        for column in COLUMNS:
            if column not in frame:
                frame[column] = None
        frame["holm_p"] = holm(frame["primary_p"].to_numpy(dtype=float))
        frame["bh_q"] = benjamini_hochberg(frame["primary_p"].to_numpy(dtype=float))
        frame["verdict"] = [verdict(raw, adjusted, self.alpha) for raw, adjusted in zip(frame["primary_p"], frame["holm_p"])]
        return frame[COLUMNS]

    def save(self, path) -> None:
        """Write the register as a CSV file, through a temporary file so that a crash cannot leave half a file behind."""
        save_table(self.to_frame(), path)
