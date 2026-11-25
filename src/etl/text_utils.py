"""
src/etl/text_utils.py  -  small, tested helpers that clean artist names and song titles.

WHAT THIS FILE IS
    Three functions: `normalize_text` (tidy one piece of text), `make_key` (turn a name into a matching key)
    and `split_artists` (split "Lil Durk & J. Cole" into ["Lil Durk", "J. Cole"]).

WHY WE NEED IT
    The brief asks us to standardize artist names. Our Phase 1 audit found three traps:
      1. collaborations are joined with " & ";
      2. a COMMA is not a separator ("Tyler, The Creator" is ONE artist);
      3. some real acts contain "&" themselves ("Earth, Wind & Fire"), so we protect them before splitting.

WHAT HAPPENS WHEN IT RUNS
    Nothing by itself. clean.py calls these functions for every artist string and song title.

EXPECTED OUTCOME
    Same results as the notebook helpers of Phase 1 (they are the same code). tests/test_text_utils.py proves it.

MOTIVE
    "Unique songs per artist" and "artist dominance" are only correct when each artist is written
    the same way every time.
"""
from __future__ import annotations

import re
import unicodedata

from src.config import PROTECTED_NAMES, SEPARATOR


def normalize_text(text):
    """Tidy one piece of text: one standard unicode form, single spaces, no spaces at the ends. Accents are KEPT."""
    text = unicodedata.normalize("NFKC", str(text))    # one standard form for every character
    text = re.sub(r"\s+", " ", text)                   # many spaces/tabs -> one space
    return text.strip()


def make_key(text):
    """Matching key: normalized, lower-case, curly quotes made straight. Used to COMPARE names, never to display them."""
    text = normalize_text(text).casefold()
    return (text.replace("’", "'").replace("‘", "'")      # curly single quotes -> straight
                .replace("“", '"').replace("”", '"'))     # curly double quotes -> straight


def split_artists(artist_string):
    """Split 'A & B & C' into ['A', 'B', 'C'] while protecting names that really contain '&'."""
    text = normalize_text(artist_string)
    placeholders = {}
    for i, name in enumerate(PROTECTED_NAMES):          # hide each protected name behind a safe token
        if name in text:
            token = f"@@PROTECTED{i}@@"
            text = text.replace(name, token)
            placeholders[token] = name
    parts = [p.strip() for p in text.split(SEPARATOR)]
    return [placeholders.get(p, p) for p in parts if p]  # put the real names back
