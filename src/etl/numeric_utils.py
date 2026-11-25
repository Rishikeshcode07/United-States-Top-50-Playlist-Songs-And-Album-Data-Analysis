"""
src/etl/numeric_utils.py  -  one small helper: rounding the way Excel, SQL and Power BI round.

WHAT THIS FILE IS
    `round_half_up(values, decimals)`. A value that sits exactly in the middle, such as 4.625, is rounded UP (to 4.63).

WHY WE NEED IT
    pandas and numpy round "half to even": 4.625 becomes 4.62 and 4.375 becomes 4.38. Excel, SQL's ROUND() and Power BI
    round half up: 4.625 becomes 4.63. A KPI table written by Python (4.62) would then disagree by 0.01 with the same KPI
    recomputed in SQL or DAX (4.63). The automatic tests found exactly this case (a song with 8 chart days), so every
    rounding of a KPI in this project now uses half-up.

WHAT HAPPENS WHEN IT RUNS
    The value is shifted by the number of decimals, 0.5 is added, the result is cut down to a whole number and shifted
    back. A tiny extra 1e-9 removes binary floating-point noise (2.675 is stored as 2.67499999999999982...). Our values are
    averages of whole numbers, so a real value is never closer than 1e-6 to a rounding border: the extra 1e-9 can not
    change a value that is not an exact tie.

EXPECTED OUTCOME
    round_half_up(4.625, 2) -> 4.63 | round_half_up(2.675, 2) -> 2.68 | round_half_up(-0.125, 2) -> -0.13 | NaN stays NaN.

MOTIVE
    One number, one value, in Python, SQL, Excel and Power BI.
"""
from __future__ import annotations

import numpy as np

_EPSILON = 1e-9


def round_half_up(values, decimals=0):
    """Round a number, a numpy array or a pandas Series half away from zero (the Excel / SQL rule)."""
    factor = 10.0 ** decimals
    shifted = values * factor
    return np.sign(shifted) * np.floor(np.abs(shifted) + 0.5 + _EPSILON) / factor
