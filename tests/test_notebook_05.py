"""
tests/test_notebook_05.py  -  structure checks of the Phase 5 notebook (notebooks/05_hypothesis_testing.ipynb) and of the three files it writes.

WHAT THIS FILE IS
    Checks of what the notebook IS, not of what it computes (the statistical methods are tested in test_inference_*.py).
    The notebook must be valid, delivered without outputs, every code cell must compile, every code cell must have its five-part
    plain-English explanation directly above it, every chart must have a "What to notice" note directly below it, no sentence may
    contain an unfilled gap or a double negative such as "fell by about -0.5", the 17 charts must be numbered 1 to 17 without a
    hole, and every number that the code looks up in FACTS must exist in the facts file the notebook saved. The last tests read the
    three files the notebook writes (the facts, the hypothesis register and the robustness table) and check that they are complete,
    that the verdicts follow the one rule of the project, and that the closing text of the notebook agrees with the register.

WHY WE NEED IT
    The explanations, the notes and the verdicts are the point of this notebook. A cell that lost its explanation during an edit,
    a sentence that still holds a "{{gap}}", a verdict that no longer follows the rule, or a closing text that disagrees with the
    saved register would otherwise be found by the reader and not by us.

WHAT HAPPENS WHEN IT RUNS
    The notebook is read as plain JSON/nbformat; nothing is executed. The tests that read the three files are skipped when those
    files are not there yet (run the notebook once to create them).

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    A teaching notebook, and the verdicts that later phases read, whose structure cannot silently rot.
"""
import ast
import json
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from src.inference.multiplicity import adjust
from src.inference.register import COLUMNS, verdict

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "05_hypothesis_testing.ipynb"
FACTS_FILE = ROOT / "reports" / "p5_facts.json"
REGISTER_FILE = ROOT / "reports" / "p5_hypothesis_register.csv"
ROBUSTNESS_FILE = ROOT / "reports" / "p5_robustness.csv"
FIGURE_DIR = ROOT / "reports" / "figures"

FIVE_PARTS = ["**What this code is.**", "**Why we need it.**", "**What will happen.**", "**Expected outcome.**", "**Motive.**"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U00002B00-\U00002BFF\U0000FE0F]")
CLAIMS = [f"H{number}" for number in range(1, 11)]
VERDICTS = {"Supported", "Suggestive", "Not supported"}
# the order in which the sections test the claims (the same two-group battery serves H1 and H9, and so on)
SECTION_OF_CLAIM = {"H1": 4, "H9": 5, "H2": 6, "H10": 7, "H3": 8, "H4": 9, "H5": 10, "H6": 11, "H7": 12, "H8": 13}

pytestmark = pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebooks/05_hypothesis_testing.ipynb is not there")


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


@pytest.fixture(scope="module")
def register():
    return pd.read_csv(REGISTER_FILE, encoding="utf-8-sig", keep_default_na=False)


@pytest.fixture(scope="module")
def robustness():
    return pd.read_csv(ROBUSTNESS_FILE, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def facts():
    return json.loads(FACTS_FILE.read_text(encoding="utf-8"))


def _code_cells(notebook):
    return [(index, cell) for index, cell in enumerate(notebook.cells) if cell.cell_type == "code"]


def _markdown_text(notebook):
    return [(index, cell.source) for index, cell in enumerate(notebook.cells) if cell.cell_type == "markdown"]


CHART_NAME = re.compile(r"""["'](p5_\d\d_[a-z0-9_]+)["']""")             # a chart is named by a string such as "p5_07_h3_popularity_tiers", in save_fig(...) or in fig_name=...


def _chart_cells(notebook):
    return [(index, cell) for index, cell in _code_cells(notebook) if CHART_NAME.search(cell.source)]


def _chart_names(notebook):
    names = []
    for _, cell in _chart_cells(notebook):
        names += CHART_NAME.findall(cell.source)
    return names


# ================================================================================ the file itself
def test_the_notebook_is_valid_and_has_the_cells_we_expect(notebook):
    nbformat.validate(notebook)
    assert len(_code_cells(notebook)) == 37
    assert notebook.cells[0].cell_type == "markdown" and notebook.cells[0].source.startswith("# Phase 5")


def test_it_is_delivered_without_outputs(notebook):
    for index, cell in _code_cells(notebook):
        assert cell.outputs == [] and cell.execution_count is None, f"cell {index} still carries an output"


def test_every_code_cell_compiles(notebook):
    for index, cell in _code_cells(notebook):
        ast.parse(cell.source, filename=f"cell {index}")


def test_no_cell_holds_a_path_that_belongs_to_one_machine(notebook):
    for index, cell in enumerate(notebook.cells):
        for needle in ("/home/", "/tmp/", "scratchpad", "C:\\", "D:\\", "/Users/"):
            assert needle not in cell.source, f"cell {index} contains {needle!r}"


# ================================================================================ the teaching structure
def test_every_code_cell_has_the_five_part_explanation_directly_above_it(notebook):
    for index, cell in _code_cells(notebook):
        above = notebook.cells[index - 1]
        assert above.cell_type == "markdown", f"cell {index} has no explanation above it"
        positions = [above.source.find(label) for label in FIVE_PARTS]
        assert all(position >= 0 for position in positions), f"explanation of cell {index} misses one of the five parts"
        assert positions == sorted(positions), f"explanation of cell {index} has its five parts in the wrong order"


def test_every_chart_has_a_what_to_notice_note_directly_below_it(notebook):
    charts = _chart_cells(notebook)
    assert len(charts) == 17
    for index, _ in charts:
        below = notebook.cells[index + 1]
        assert below.cell_type == "markdown" and below.source.startswith("**What to notice**"), f"chart in cell {index} has no note"


def test_the_17_charts_are_numbered_one_to_seventeen_exactly_once(notebook):
    names = _chart_names(notebook)
    assert [int(name[3:5]) for name in names] == list(range(1, 18))
    assert len(set(names)) == 17


def test_the_sixteen_sections_follow_each_other(notebook):
    numbers = [int(match.group(1)) for _, text in _markdown_text(notebook) for match in [re.match(r"## Section (\d+) - ", text)] if match]
    assert numbers == list(range(1, 17))


def test_every_claim_has_its_own_section_in_the_planned_order(notebook):
    headings = {}
    for _, text in _markdown_text(notebook):
        match = re.match(r"## Section (\d+) - (H\d+) - ", text)
        if match:
            headings[match.group(2)] = int(match.group(1))
    assert headings == SECTION_OF_CLAIM


def test_the_notebook_ends_with_the_closing_summary_and_the_hand_over_to_phase_6(notebook):
    last = notebook.cells[-1]
    assert last.cell_type == "markdown" and last.source.startswith("## What Phase 5 adds up to")
    assert "Phase 6" in last.source and "p5_hypothesis_register.csv" in last.source


# ================================================================================ the sentences
def test_no_sentence_contains_an_unfilled_gap(notebook):
    for index, text in _markdown_text(notebook):
        for marker in ("{{", "}}", "??", "\u2026", "TODO", "TBD", "XXX"):
            assert marker not in text, f"markdown cell {index} contains {marker!r}"
        assert not re.search(r"\bnan\b|\binf\b", text), f"markdown cell {index} prints a nan or an inf"


def test_no_sentence_says_that_something_fell_or_was_less_by_a_negative_number(notebook):
    """'fell by about -0.47 points' and '-3.4 points less popular' are double negatives: the sign is already in the verb."""
    patterns = [r"\b(fell|falls|fall|dropped|drops|drop|decreased|decreases)\b (by|of) (about )?-\d",
                r"(?<![\w.])-\d[\d.,]* points \*?(less|lower|fewer|smaller)\*?\b"]
    for index, text in _markdown_text(notebook):
        for pattern in patterns:
            assert not re.search(pattern, text), f"markdown cell {index} has a double negative ({pattern})"


def test_no_emoji_anywhere_and_no_cloud_storage_in_the_code_or_the_text(notebook):
    for index, cell in enumerate(notebook.cells):
        assert not EMOJI.search(cell.source), f"cell {index} contains an emoji"
        assert not re.search(r"\bS3\b|boto3|s3://", cell.source, flags=re.IGNORECASE), f"cell {index} mentions S3"


def test_the_notebook_never_calls_the_workflow_traditional_etl(notebook):
    assert not any("traditional ETL" in cell.source for cell in notebook.cells)


# ================================================================================ the facts file
@pytest.mark.skipif(not FACTS_FILE.exists(), reason="run the notebook once to create reports/p5_facts.json")
def test_every_fact_the_code_looks_up_exists_in_the_facts_file(notebook, facts):
    wanted = set()
    for _, cell in _code_cells(notebook):
        wanted |= set(re.findall(r"\b(?:FACTS|F)\[\s*['\"]([a-z0-9_]+)['\"]\s*\]", cell.source))
    assert len(wanted) > 250
    assert wanted - set(facts) == set(), "the code uses facts that the saved file does not hold"


@pytest.mark.skipif(not FACTS_FILE.exists(), reason="run the notebook once to create reports/p5_facts.json")
def test_the_facts_file_is_sorted_and_holds_only_plain_finite_values():
    text = FACTS_FILE.read_text(encoding="utf-8")
    facts = json.loads(text)
    assert list(facts) == sorted(facts)
    assert all(isinstance(value, (bool, int, float, str)) for value in facts.values())
    assert all(np.isfinite(value) for value in facts.values() if isinstance(value, float) and not isinstance(value, bool))
    assert text.endswith("}\n")


# ================================================================================ the hypothesis register
@pytest.mark.skipif(not REGISTER_FILE.exists(), reason="run the notebook once to create reports/p5_hypothesis_register.csv")
def test_the_register_holds_the_ten_claims_with_every_field_filled(register):
    assert list(register.columns) == COLUMNS
    assert register["hypothesis"].tolist() == CLAIMS
    for column in ("question", "unit", "planned_test", "primary_test", "effect_unit", "standard_metric", "size_word", "caveat"):
        assert (register[column].astype(str).str.strip() != "").all(), f"an empty {column}"
    assert not register["caveat"].str.contains(r"[{}]|\bnan\b", regex=True).any()
    assert (register["n"] > 0).all()


@pytest.mark.skipif(not REGISTER_FILE.exists(), reason="run the notebook once to create reports/p5_hypothesis_register.csv")
def test_the_register_p_values_are_probabilities_and_the_corrections_only_make_them_larger(register):
    for column in ("planned_p", "primary_p", "holm_p", "bh_q"):
        assert register[column].between(0, 1).all(), f"{column} is not a probability"
    assert (register["holm_p"] >= register["primary_p"] - 1e-15).all()
    assert (register["bh_q"] >= register["primary_p"] - 1e-15).all()
    assert (register["bh_q"] <= register["holm_p"] + 1e-12).all(), "Benjamini-Hochberg is the gentler correction"
    assert np.allclose(register["holm_p"], adjust(register["primary_p"].to_numpy(), "holm"), rtol=1e-9, atol=1e-15)
    assert np.allclose(register["bh_q"], adjust(register["primary_p"].to_numpy(), "bh"), rtol=1e-9, atol=1e-15)


@pytest.mark.skipif(not REGISTER_FILE.exists(), reason="run the notebook once to create reports/p5_hypothesis_register.csv")
def test_every_verdict_follows_the_one_rule_of_the_project(register):
    expected = [verdict(raw, adjusted) for raw, adjusted in zip(register["primary_p"], register["holm_p"])]
    assert register["verdict"].tolist() == expected
    assert set(register["verdict"]) <= VERDICTS


@pytest.mark.skipif(not REGISTER_FILE.exists(), reason="run the notebook once to create reports/p5_hypothesis_register.csv")
def test_every_effect_lies_inside_its_own_interval(register):
    assert (register["effect_low"] <= register["effect"]).all() and (register["effect"] <= register["effect_high"]).all()


@pytest.mark.skipif(not (REGISTER_FILE.exists() and FACTS_FILE.exists()), reason="run the notebook once to create the saved files")
def test_the_counts_stored_as_facts_are_the_counts_of_the_register(register, facts):
    for name, key in (("Supported", "supported"), ("Suggestive", "suggestive"), ("Not supported", "not_supported")):
        chosen = register.loc[register["verdict"] == name, "hypothesis"].tolist()
        assert facts[f"fam_n_{key}"] == len(chosen)
        assert facts[f"fam_list_{key}"] == ", ".join(sorted(chosen, key=lambda h: int(h[1:])))
    assert facts["fam_primary_significant"] == int((register["primary_p"] < 0.05).sum())
    assert facts["fam_planned_significant"] == int((register["planned_p"] < 0.05).sum())


@pytest.mark.skipif(not REGISTER_FILE.exists(), reason="run the notebook once to create reports/p5_hypothesis_register.csv")
def test_the_closing_text_names_the_same_verdicts_as_the_register(notebook, register):
    closing = notebook.cells[-1].source
    for name, word in (("Supported", "supported"), ("Suggestive", "suggestive"), ("Not supported", "not supported")):
        chosen = register.loc[register["verdict"] == name, "hypothesis"].tolist()
        assert f"{len(chosen)} are {word} ({', '.join(chosen)})" in closing, f"the closing text does not agree with the register on '{word}'"


# ================================================================================ the robustness table
@pytest.mark.skipif(not (ROBUSTNESS_FILE.exists() and REGISTER_FILE.exists()), reason="run the notebook once to create the saved files")
def test_the_robustness_table_has_one_row_per_claim_and_repeats_the_register_verdicts(robustness, register):
    assert robustness["hypothesis"].tolist() == CLAIMS
    assert robustness["verdict"].tolist() == register["verdict"].tolist()
    assert np.allclose(robustness["primary_p_all_data"], register["primary_p"], rtol=1e-9)
    assert np.allclose(robustness["estimate_all_data"], register["effect"], rtol=1e-9)


@pytest.mark.skipif(not ROBUSTNESS_FILE.exists(), reason="run the notebook once to create reports/p5_robustness.csv")
def test_the_halves_are_filled_for_nine_claims_and_left_empty_for_h8(robustness):
    halves = ["first_half_estimate", "first_half_p", "second_half_estimate", "second_half_p", "same_direction_in_both_halves"]
    h8 = robustness["hypothesis"] == "H8"
    assert robustness.loc[h8, halves].isna().all(axis=None), "H8 compares the halves itself and has none of its own"
    assert robustness.loc[~h8, halves].notna().all(axis=None)
    assert robustness.loc[~h8, "first_half_p"].between(0, 1).all() and robustness.loc[~h8, "second_half_p"].between(0, 1).all()


@pytest.mark.skipif(not ROBUSTNESS_FILE.exists(), reason="run the notebook once to create reports/p5_robustness.csv")
def test_the_verdicts_without_the_biggest_artist_follow_the_rule_and_the_holm_correction(robustness):
    raw, holm = robustness["primary_p_without_biggest_artist"].to_numpy(), robustness["holm_p_without_biggest_artist"].to_numpy()
    assert np.allclose(holm, adjust(raw, "holm"), rtol=1e-9, atol=1e-15), "Holm must be run again on the ten new p-values"
    assert robustness["verdict_without_biggest_artist"].tolist() == [verdict(r, h) for r, h in zip(raw, holm)]


@pytest.mark.skipif(not (ROBUSTNESS_FILE.exists() and FACTS_FILE.exists()), reason="run the notebook once to create the saved files")
def test_the_stress_test_counts_stored_as_facts_are_the_counts_of_the_table(robustness, facts):
    checked = robustness["same_direction_in_both_halves"].dropna().astype(str)
    assert facts["rob_same_direction"] == int((checked == "True").sum())
    assert facts["rob_claims_checked"] == len(checked) == 9
    changed = robustness["verdict"] != robustness["verdict_without_biggest_artist"]
    assert facts["rob_wo_changed"] == int(changed.sum())
    assert facts["rob_wo_changed_list"] == (", ".join(robustness.loc[changed, "hypothesis"]) or "none")
    strength = {"Supported": 2, "Suggestive": 1, "Not supported": 0}
    weaker = sum(strength[after] < strength[before] for before, after in zip(robustness["verdict"], robustness["verdict_without_biggest_artist"]))
    assert facts["rob_wo_weaker"] == weaker and facts["rob_wo_stronger"] == int(changed.sum()) - weaker


# ================================================================================ the charts
@pytest.mark.skipif(not FIGURE_DIR.exists(), reason="no figure folder yet")
def test_the_chart_files_named_in_the_code_exist_and_are_real_pictures(notebook):
    names = _chart_names(notebook)
    missing = [name for name in names if not (FIGURE_DIR / f"{name}.png").exists()]
    if missing:
        pytest.skip(f"run the notebook once to draw the charts (missing: {missing[:3]} ...)")
    for name in names:
        header = (FIGURE_DIR / f"{name}.png").read_bytes()[:8]
        assert header == b"\x89PNG\r\n\x1a\n", f"{name}.png is not a PNG file"
