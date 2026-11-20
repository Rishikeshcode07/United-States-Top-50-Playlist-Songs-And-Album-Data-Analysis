"""
tests/test_dash_texts.py  -  the small functions that turn numbers into the words and figures shown on the dashboard.

WHAT THIS FILE IS
    Tests of src/dash/texts.py: counts with thousands separators, percentages, signed changes (with a real minus sign), dates,
    plurals, p-values, lists of names and the effect-size phrase.

WHY WE NEED IT
    Every "What to notice" sentence of the dashboard is built from these pieces. A missing number must never print as "nan",
    "1 songs" must never appear, and a negative number must read the same on every page.

WHAT HAPPENS WHEN IT RUNS
    Pure functions, no files and no data: the tests take a fraction of a second and need only pandas.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    A sentence that is wrong in one small detail makes a reader doubt the numbers around it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dash import texts as T

MINUS = "−"


# ------------------------------------------------------------------------------------------------------------ numbers
def test_n_writes_thousands_separators():
    assert T.n(27750) == "27,750"
    assert T.n(0) == "0"
    assert T.n(1234567.891, 1) == "1,234,567.9"
    assert T.n(np.int64(944)) == "944"


@pytest.mark.parametrize("missing", [None, float("nan"), np.nan, pd.NaT])
def test_a_missing_number_becomes_a_dash_never_nan(missing):
    for function in (T.n, T.num, T.signed, T.pct, T.day):
        assert function(missing) == T.DASH
    assert "nan" not in T.n(float("nan")).lower()


def test_negative_numbers_use_the_real_minus_sign():
    assert T.n(-0.04, 2) == f"{MINUS}0.04"
    assert T.num(-1234.5, 1) == f"{MINUS}1,234.5"
    assert "-" not in T.num(-6.04)


def test_a_number_that_rounds_to_zero_has_no_sign():
    assert T.n(-0.001, 2) == "0.00"
    assert T.n(-0.4, 0) == "0"
    assert T.signed(-0.0001) == "0.0"
    assert T.signed(0.0001) == "0.0"


def test_num_defaults_to_one_decimal():
    assert T.num(12.345) == "12.3"
    assert T.num(2.0) == "2.0"
    assert T.num(12.3456, 3) == "12.346"


def test_pct_takes_a_share_or_a_percentage():
    assert T.pct(0.4812) == "48.1%"
    assert T.pct(48.12, already_percent=True) == "48.1%"
    assert T.pct(0.5, digits=0) == "50%"
    assert T.pct(0) == "0.0%"


def test_signed_shows_a_plus_for_a_rise_and_a_minus_for_a_fall():
    assert T.signed(2.5) == "+2.5"
    assert T.signed(-1.5) == f"{MINUS}1.5"
    assert T.signed(2, 0, "%") == "+2%"
    assert T.signed(-2, 0, " places") == f"{MINUS}2 places"
    assert T.signed(0) == "0.0"


# -------------------------------------------------------------------------------------------------------------- dates
def test_day_has_no_leading_zero():
    assert T.day(pd.Timestamp("2024-05-18")) == "18 May 2024"
    assert T.day("2024-07-02") == "2 Jul 2024"
    assert T.day(pd.Timestamp("2025-11-27 13:45")) == "27 Nov 2025"


def test_span_joins_two_dates_and_collapses_one_day():
    assert T.span("2024-05-18", "2025-11-27") == "18 May 2024 to 27 Nov 2025"
    assert T.span("2024-05-18", "2024-05-18") == "18 May 2024"


# ----------------------------------------------------------------------------------------------------------- words
@pytest.mark.parametrize("count, word, many, expected", [
    (1, "song", None, "1 song"), (2, "song", None, "2 songs"), (0, "song", None, "0 songs"),
    (3, "party", "parties", "3 parties"), (1, "party", "parties", "1 party"), (1500, "song", None, "1,500 songs"),
    (2.0, "day", None, "2 days"), (np.int64(1), "run", None, "1 run"),
])
def test_plural(count, word, many, expected):
    assert T.plural(count, word, many) == expected


def test_plural_of_a_missing_count_keeps_the_plural_word():
    assert T.plural(None, "song") == f"{T.DASH} songs"
    assert T.plural(float("nan"), "party", "parties") == f"{T.DASH} parties"


@pytest.mark.parametrize("p, expected", [
    (0.0000004, "p < 0.001"), (0.0009999, "p < 0.001"), (0.001, "p = 0.001"), (0.0432, "p = 0.043"), (0.5, "p = 0.500"), (1.0, "p = 1.000"),
])
def test_p_text(p, expected):
    assert T.p_text(p) == expected


def test_p_text_of_a_missing_value_says_so():
    assert T.p_text(float("nan")) == "p not available"
    assert T.p_text(None) == "p not available"


def test_names_reads_like_a_sentence():
    assert T.names([]) == "no one"
    assert T.names(["A"]) == "A"
    assert T.names(["A", "B"]) == "A and B"
    assert T.names(["A", "B", "C"]) == "A, B and C"
    assert T.names(list("ABCD")) == "A, B, C and 1 more"
    assert T.names(list("ABCDEFG"), limit=5) == "A, B, C, D, E and 2 more"
    assert T.names([1, 2]) == "1 and 2"


def test_size_phrase_names_the_size_and_the_direction():
    assert T.size_phrase(0.2328, "small") == "a small effect (+0.23)"
    assert T.size_phrase(-0.5, "large") == f"a large effect ({MINUS}0.50)"
    assert T.size_phrase(0.01, "unusual") == "an unusual effect (+0.01)"


def test_size_phrase_without_a_number_or_a_size_word_says_it_cannot_be_measured():
    expected = "an effect that cannot be measured with so few songs"
    assert T.size_phrase(None, "small") == expected
    assert T.size_phrase(float("nan"), "small") == expected
    assert T.size_phrase(0.2, "") == expected


def test_no_text_helper_uses_a_causal_phrase():
    """The project describes history and associations. The fixed words of these helpers must not claim a cause."""
    import inspect
    source = inspect.getsource(T).lower()
    for phrase in ("because of", "caused by", "leads to", "results in", "due to", "thanks to"):
        assert phrase not in source
