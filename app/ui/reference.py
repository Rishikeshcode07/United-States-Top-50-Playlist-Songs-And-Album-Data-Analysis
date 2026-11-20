"""
app/ui/reference.py  -  "what the formal analysis said" cards that pages can place next to their own live numbers.

WHAT THIS FILE IS
    A small helper that reads the Phase 5 hypothesis register (reports/p5_hypothesis_register.csv) and draws one card for each
    hypothesis the page asks for: the question, the verdict, the size of the effect with its interval, and the caveat.

WHY WE NEED IT
    A dashboard recalculates numbers for any filter, but a filtered comparison is not a test. The formal tests of Phase 5 (with
    artist-aware standard errors and a correction for ten tests) are the honest reference, so the explicit, release and longevity
    pages show them next to the live comparison.

WHAT HAPPENS WHEN IT RUNS
    If the file is missing (notebook 05 was not run) the helper says so in one line and draws nothing else. The numbers are shown
    exactly as the notebook saved them; nothing is recalculated here.

EXPECTED OUTCOME
    The same verdict and caveat in the notebook, in the paper and on the dashboard.

MOTIVE
    Keep the live comparison and the formal result side by side, so that the live one is never mistaken for a test.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui.components import cards, section
from app.ui.state import report
from src.dash import texts as T


def _effect_text(row) -> str:
    """One sentence about the size of the effect, from the columns of the register."""
    if pd.isna(row["effect"]):
        return ""
    text = f"Effect: {T.num(row['effect'], 2)}, in {row['effect_unit']}."
    if pd.notna(row.get("effect_low")) and pd.notna(row.get("effect_high")):
        text += f" The 95% interval runs from {T.num(row['effect_low'], 2)} to {T.num(row['effect_high'], 2)}"
        text += f"; by the usual yardsticks this is a {row['size_word']} effect." if isinstance(row.get("size_word"), str) and row["size_word"] else "."
    elif isinstance(row.get("size_word"), str) and row["size_word"]:
        text += f" By the usual yardsticks this is a {row['size_word']} effect."
    if pd.notna(row.get("holm_p")):
        text += f" Adjusted p-value (Holm, ten tests): {T.p_text(row['holm_p'])}."
    return text


effect_text = _effect_text                                           # the public name, for the evidence page


def hypothesis_cards(ids, title: str = "What the formal test said (Phase 5)", note: str = "") -> None:
    """Draw one card per hypothesis id (for example ["H1", "H10"]) from the Phase 5 register."""
    register = report("p5_hypothesis_register")
    if register is None or register.empty:
        st.caption("The formal tests of Phase 5 are not available yet (run notebook 05_hypothesis_testing to add them here).")
        return
    chosen = register[register["hypothesis"].isin(list(ids))]
    if chosen.empty:
        return
    section(title, note or "These are the results of the pre-planned tests on the whole file. They do not change when the sidebar filters change.")
    cards([{"tag": f"{row['hypothesis']}  |  {row['verdict']}", "title": row["question"], "text": f"{_effect_text(row)} {row['caveat']}".strip()} for _, row in chosen.iterrows()])
