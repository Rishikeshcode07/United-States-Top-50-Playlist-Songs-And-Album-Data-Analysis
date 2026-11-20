"""
tests/test_numeric_utils.py  -  tests for round_half_up, the rounding rule shared by Python, SQL, Excel and Power BI.

WHAT THIS FILE IS
    Examples of exact "half way" values (4.625, 2.675, -0.125 ...) and the answer every spreadsheet and database gives.

WHY WE NEED IT
    numpy rounds 4.625 to 4.62, SQL and Excel round it to 4.63. A 0.01 difference between a table and a dashboard is the
    kind of thing a reviewer notices. The SQL-versus-pandas tests found it once; this file makes sure it stays fixed.

WHAT HAPPENS WHEN IT RUNS
    Small numbers go in, rounded numbers come out. One test also asks a real SQL engine (SQLite) for its own ROUND().

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    One number, one value, in every tool.
"""
import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

from src.etl.numeric_utils import round_half_up


@pytest.mark.parametrize("value, decimals, expected", [
    (4.625, 2, 4.63), (4.375, 2, 4.38), (2.675, 2, 2.68), (1.005, 2, 1.01), (0.5, 0, 1.0), (1.5, 0, 2.0), (2.5, 0, 3.0),
    (-0.125, 2, -0.13), (-2.5, 0, -3.0), (0.0625, 3, 0.063), (4.624, 2, 4.62), (4.626, 2, 4.63), (7.0, 2, 7.0), (0.0, 2, 0.0),
])
def test_exact_halves_go_up_and_everything_else_rounds_normally(value, decimals, expected):
    assert round_half_up(value, decimals) == pytest.approx(expected, abs=1e-12)


def test_it_works_on_a_series_and_keeps_missing_values_missing():
    result = round_half_up(pd.Series([4.625, np.nan, 3.0, 2.675]), 2)
    assert result.iloc[0] == 4.63 and np.isnan(result.iloc[1]) and result.iloc[2] == 3.0 and result.iloc[3] == 2.68


def test_the_result_has_no_binary_noise():
    assert str(round_half_up(4.625, 2)) == "4.63"
    assert round_half_up(pd.Series([0.1 + 0.2]), 2).iloc[0] == 0.3


def test_it_agrees_with_the_round_function_of_a_real_sql_engine():
    # Values that a computer can store exactly (x.125, x.375, x.5 ...): every SQL engine rounds these half up.
    # (SQLite rounds the binary double 2.67499999... of "2.675" down, while PostgreSQL's exact DECIMAL, Excel and this
    # function round it up. For averages of whole numbers over n days the exact cases are by far the most common.)
    values = [4.625, 4.375, 10.125, 7.5, 3.14159, 0.0625]
    with sa.create_engine("sqlite://").connect() as connection:
        sql = [connection.execute(sa.text("SELECT ROUND(CAST(:v AS DECIMAL(12,4)), 2)"), {"v": v}).scalar_one() for v in values]
    assert [float(x) for x in round_half_up(pd.Series(values), 2)] == pytest.approx([float(x) for x in sql])


def test_the_whole_number_case_and_the_default():
    assert round_half_up(2.5) == 3.0 and round_half_up(2.4) == 2.0
