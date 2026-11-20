"""
src/dash/texts.py  -  numbers written as plain text, and the small sentences under the charts.

WHAT THIS FILE IS
    n, num, pct, signed   a count with thousands separators (27,750), a decimal number, a percentage, a signed change (+2.5)
    day, span             a date as "18 May 2024", two dates as "18 May 2024 to 27 Nov 2025"
    plural                "1 song", "2 songs"
    p_text                a p-value as people write it: "p < 0.001" or "p = 0.043"
    names                 "A", "A and B", "A, B and C"
    size_phrase           an effect size and its size word in one phrase

WHY WE NEED IT
    The dashboard writes a short "What to notice" sentence under many charts, from the numbers on the screen. Writing those
    sentences by hand for every filter would be impossible; so they are built from these small, tested pieces. They never
    use causal words: the whole project describes history and associations.

WHAT HAPPENS WHEN IT RUNS
    Pure functions. A missing number (NaN or None) becomes an en dash, so a sentence never shows "nan".

EXPECTED OUTCOME
    n(27750) = "27,750"; pct(0.4812) = "48.1%"; p_text(0.0000004) = "p < 0.001".

MOTIVE
    Sentences that read like a person wrote them, for any filter.
"""
from __future__ import annotations

import math

import pandas as pd

DASH = "–"          # the en dash shown for a missing number
MINUS = "−"


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or (x is pd.NaT) or (isinstance(x, pd.Timestamp) and pd.isna(x))


def n(x, digits: int = 0) -> str:
    """27750 -> '27,750'. A negative number gets a real minus sign (-0.04 -> '−0.04'); a number that rounds to zero gets no sign at all."""
    if _missing(x):
        return DASH
    text = f"{x:,.{digits}f}"
    if text.startswith("-"):
        text = text[1:] if float(text[1:].replace(",", "")) == 0 else MINUS + text[1:]
    return text


def num(x, digits: int = 1) -> str:
    """12.345 -> '12.3'."""
    return n(x, digits)


def pct(share, digits: int = 1, already_percent: bool = False) -> str:
    """A share (0.4812) as '48.1%'. With already_percent=True the number is taken as a percentage (48.12 -> '48.1%')."""
    if _missing(share):
        return DASH
    return f"{(share if already_percent else share * 100):.{digits}f}%"


def signed(x, digits: int = 1, suffix: str = "") -> str:
    """A change with its sign: 2.5 -> '+2.5', -1.5 -> '-1.5' (with a real minus sign), 0.0001 -> '0.0' (no sign for zero)."""
    if _missing(x):
        return DASH
    text = f"{abs(x):,.{digits}f}"
    if float(text.replace(",", "")) == 0:
        return f"{text}{suffix}"
    return f"{'+' if x > 0 else MINUS}{text}{suffix}"


def day(value) -> str:
    """A date as '18 May 2024'."""
    if _missing(value):
        return DASH
    value = pd.Timestamp(value)
    return f"{value.day} {value.strftime('%b %Y')}"


def span(first, last) -> str:
    """Two dates as '18 May 2024 to 27 Nov 2025' (a single day if both are the same)."""
    return day(first) if pd.Timestamp(first) == pd.Timestamp(last) else f"{day(first)} to {day(last)}"


def plural(count, word: str, many: "str | None" = None) -> str:
    """plural(1, 'song') -> '1 song'; plural(2, 'song') -> '2 songs'; plural(3, 'party', 'parties') -> '3 parties'."""
    if _missing(count):
        return f"{DASH} {many or word + 's'}"
    count = int(count)
    return f"{count:,} {word if count == 1 else (many or word + 's')}"


def p_text(p) -> str:
    """0.0000004 -> 'p < 0.001'; 0.0432 -> 'p = 0.043'."""
    if _missing(p):
        return "p not available"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def names(items, limit: int = 3) -> str:
    """['A'] -> 'A'; ['A', 'B'] -> 'A and B'; ['A', 'B', 'C', 'D'] -> 'A, B, C and 1 more' (with limit=3)."""
    items = [str(i) for i in items]
    if not items:
        return "no one"
    if len(items) > limit:
        return ", ".join(items[:limit]) + f" and {len(items) - limit} more"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def size_phrase(value, size_word: str, digits: int = 2) -> str:
    """0.2328, 'small' -> 'a small effect (+0.23)'."""
    if _missing(value) or not size_word:
        return "an effect that cannot be measured with so few songs"
    article = "an" if size_word[0] in "aeiou" else "a"
    return f"{article} {size_word} effect ({signed(value, digits)})"
