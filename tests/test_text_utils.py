"""
tests/test_text_utils.py  -  tests for the three text helpers (normalize_text, make_key, split_artists).

WHAT THIS FILE IS
    Small examples with a known right answer, for the functions that clean artist names and song titles.

WHY WE NEED IT
    "Unique songs per artist" and "Artist Dominance Index" are only right when every artist is written the same way
    and every collaboration is split in the right place. These three functions decide that, so they get the most
    careful tests: each trap we found in the data (comma, "&" inside a band name, accents, curly quotes) has one test.

WHAT HAPPENS WHEN IT RUNS
    pytest calls every function whose name starts with test_ and reports the ones whose assert is false.

EXPECTED OUTCOME
    All tests pass in well under a second.

MOTIVE
    A rule that is written down but not tested is only a wish.
"""
import pytest

from src.config import PROTECTED_NAMES
from src.etl.text_utils import make_key, normalize_text, split_artists


# ---------------------------------------------------------------------------------------------- normalize_text
def test_normalize_text_trims_and_collapses_spaces():
    assert normalize_text("  Lil   Durk \t ") == "Lil Durk"


def test_normalize_text_keeps_accents():
    assert normalize_text("Beyoncé") == "Beyoncé"


def test_normalize_text_writes_an_accent_in_one_standard_form():
    # "e" followed by a separate accent mark looks identical to the single character "é"; after normalizing they ARE identical
    assert normalize_text("Beyoncé") == normalize_text("Beyoncé")


def test_normalize_text_turns_look_alike_characters_into_plain_ones():
    assert normalize_text("ﬁrst") == "first"          # the "fi" ligature becomes the two letters f and i
    assert normalize_text("１２３") == "123"            # full-width digits become normal digits


def test_normalize_text_accepts_non_text_input():
    assert normalize_text(123) == "123"


# ---------------------------------------------------------------------------------------------- make_key
def test_make_key_ignores_upper_and_lower_case():
    assert make_key("Not Like Us") == make_key("NOT LIKE US")


def test_make_key_treats_curly_and_straight_quotes_as_equal():
    assert make_key("Thinkin’ Bout You") == make_key("Thinkin' Bout You")
    assert make_key("“Hello”") == make_key('"Hello"')


def test_make_key_keeps_accent_differences():
    # An accent is a real difference between two spellings, so the keys must differ.
    assert make_key("Beyoncé") != make_key("Beyonce")


def test_make_key_is_idempotent():
    once = make_key("  Thinkin’  Bout YOU ")
    assert make_key(once) == once


# ---------------------------------------------------------------------------------------------- split_artists
@pytest.mark.parametrize("text, expected", [
    ("Solo Artist", ["Solo Artist"]),
    ("Lil Durk & J. Cole", ["Lil Durk", "J. Cole"]),
    ("Kendrick Lamar & SZA & Drake", ["Kendrick Lamar", "SZA", "Drake"]),
    ("A &  B", ["A", "B"]),                                      # extra spaces do not matter
    ("  Solo  ", ["Solo"]),
])
def test_split_artists_ordinary_credits(text, expected):
    assert split_artists(text) == expected


def test_a_comma_is_not_a_separator():
    assert split_artists("Tyler, The Creator") == ["Tyler, The Creator"]
    assert split_artists("Post Malone, Morgan Wallen") == ["Post Malone, Morgan Wallen"]


@pytest.mark.parametrize("name", PROTECTED_NAMES)
def test_protected_band_names_are_never_split(name):
    assert split_artists(name) == [name]


def test_a_protected_name_next_to_a_real_collaboration():
    assert split_artists("Earth, Wind & Fire & Someone") == ["Earth, Wind & Fire", "Someone"]
    assert split_artists("Someone & Earth, Wind & Fire") == ["Someone", "Earth, Wind & Fire"]


def test_split_keeps_the_order_of_the_credit():
    assert split_artists("C & B & A") == ["C", "B", "A"]


def test_split_never_returns_an_empty_name():
    for text in ("A & B", "Solo", "Earth, Wind & Fire", "Tyler, The Creator"):
        assert all(name.strip() for name in split_artists(text))
