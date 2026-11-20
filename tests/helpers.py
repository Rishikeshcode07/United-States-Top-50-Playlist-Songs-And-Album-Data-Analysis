"""
tests/helpers.py  -  three tiny helper functions that several test files share.

WHAT THIS FILE IS
    `copy_tables` (an independent copy of the dictionary of tables), `same_values` (compare two columns and treat
    "empty equals empty") and `scalar` (run a SQL query and return the single number it gives back).

WHY WE NEED IT
    In pandas, NaN == NaN is False, so a plain `==` says that two empty cells are different. And a test that damages a
    table on purpose must never damage the copy that the other tests are using.

WHAT HAPPENS WHEN IT RUNS
    Nothing by itself; the test files import these functions.

EXPECTED OUTCOME
    Shorter, easier-to-read tests.

MOTIVE
    Keep the tests about the data, not about pandas details.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sqlalchemy as sa


def copy_tables(tables):
    """A deep, independent copy of {table name: DataFrame}. Tests that break a table on purpose work on a copy."""
    return {name: table.copy() for name, table in tables.items()}


def same_values(actual, expected, tolerance=0.0):
    """
    True when two columns hold the same values, row by row. Two empty cells count as equal.
    `tolerance` is the largest allowed difference for numbers (0.005 for a value rounded to 2 decimals).
    """
    a = pd.Series(list(actual))
    b = pd.Series(list(expected))
    if len(a) != len(b):
        return False
    both_empty = a.isna() & b.isna()
    if a.dtype.kind in "biufc" and b.dtype.kind in "biufc":
        close = (a.astype(float) - b.astype(float)).abs() <= tolerance + 1e-12
        return bool((close | both_empty).all())
    return bool(((a == b) | both_empty).all())


def scalar(connection, query, **parameters):
    """Run one SQL query and return the first value of the first row."""
    return connection.execute(sa.text(query), parameters).scalar_one()


def mismatches(actual, expected, tolerance=0.0):
    """The row numbers where two columns differ (empty list = identical). Used in assert messages."""
    a = pd.Series(list(actual)).reset_index(drop=True)
    b = pd.Series(list(expected)).reset_index(drop=True)
    bad = []
    for i in range(min(len(a), len(b))):
        x, y = a[i], b[i]
        if pd.isna(x) and pd.isna(y):
            continue
        if pd.isna(x) != pd.isna(y):
            bad.append(i)
        elif isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number)):
            if abs(float(x) - float(y)) > tolerance + 1e-12:
                bad.append(i)
        elif x != y:
            bad.append(i)
    return bad
