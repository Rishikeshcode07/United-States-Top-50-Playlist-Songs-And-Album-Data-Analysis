"""
src/eda/findings.py  -  a tiny notebook for the notebook: the numbers we quote and the findings we draw from them.

WHAT THIS FILE IS
    Two small classes.

        Facts      a dictionary of named numbers ("single_median_days" -> 13.0) that refuses to overwrite a name and refuses
                   numbers that are not finite. Every number that appears in a sentence of the EDA notebook is stored here
                   by the code that computed it.
        Findings   a list of findings. Each one has an id (F01, F02 ...), the analysis block it belongs to, the chart that
                   shows it, a plain-English statement, the evidence (the numbers behind the statement) and a caveat.

WHY WE NEED IT
    A written insight is only as good as the number inside it, and numbers typed by hand are the ones that go wrong. With
    `Facts` a sentence is written from a computed value, so the text and the table can not disagree. `Findings` turns the
    notebook into a register that can be read without opening the notebook (reports/p4_eda_findings.csv): the research paper
    and the executive summary of Phase 9 will be written from it.

WHAT HAPPENS WHEN IT RUNS
    `facts.save(path)` writes a JSON file, `findings.save(path)` writes a CSV file. Both are written to a temporary file
    first and then moved into place, so a crash can not leave a half-written file behind.

EXPECTED OUTCOME
    reports/p4_facts.json and reports/p4_eda_findings.csv after the notebook has run.

MOTIVE
    Every claim in the project can be traced to a chart and to a number.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd

FINDING_COLUMNS = ["finding_id", "block", "figure", "finding", "evidence", "caveat"]


def _write_atomic(path, writer):
    """Let `writer(temporary_path)` write the file, then move it into place in one step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class Facts(dict):
    """
    A dictionary of named numbers. A name can be set once; numbers must be finite (no NaN, no infinity). The same rules hold
    when the facts are given to the constructor or to `update` (plain `dict` would skip the checks there).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(f"the fact '{key}' is already set (a name can only be used once)")
        if hasattr(value, "item"):                       # numpy numbers -> plain Python numbers
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"the fact '{key}' is not a finite number ({value})")
        if not isinstance(value, (bool, int, float, str)):
            raise TypeError(f"the fact '{key}' must be a number, a true/false value or a text, not {type(value).__name__}")
        super().__setitem__(key, value)

    def save(self, path):
        """Write the facts as a JSON file, sorted by name."""
        text = json.dumps(dict(sorted(self.items())), indent=1, ensure_ascii=False) + "\n"
        _write_atomic(path, lambda target: Path(target).write_text(text, encoding="utf-8"))


class Findings:
    """The register of findings. `add` returns the id of the new finding (F01, F02 ...)."""

    def __init__(self):
        self._rows = []

    def add(self, block, figure, finding, evidence, caveat=""):
        for name, text in (("block", block), ("figure", figure), ("finding", finding), ("evidence", evidence)):
            if text is None or not str(text).strip():
                raise ValueError(f"a finding needs a non-empty '{name}'")
        finding_id = f"F{len(self._rows) + 1:02d}"
        self._rows.append({"finding_id": finding_id, "block": block, "figure": figure, "finding": finding,
                           "evidence": evidence, "caveat": caveat})
        return finding_id

    def __len__(self):
        return len(self._rows)

    def to_frame(self):
        return pd.DataFrame(self._rows, columns=FINDING_COLUMNS)

    def save(self, path):
        """Write the register as a CSV file. UTF-8 with a byte-order mark, so that Excel on Windows shows the text correctly."""
        frame = self.to_frame()
        _write_atomic(path, lambda target: frame.to_csv(target, index=False, encoding="utf-8-sig"))
