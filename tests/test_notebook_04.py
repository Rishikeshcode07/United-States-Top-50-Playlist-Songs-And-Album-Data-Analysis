"""
tests/test_notebook_04.py  -  structure checks of the Phase 4 notebook (notebooks/04_eda_playlist_song_artist.ipynb).

WHAT THIS FILE IS
    Checks of what the notebook IS, not of what it computes (the computed numbers are tested in test_concentration.py,
    test_eda_ranking.py and the other test_eda_*.py files). The notebook must be valid, delivered without outputs, every code cell
    must compile, every code cell must have its five-part plain-English explanation directly above it, every chart must have a
    "What to notice" note directly below it, no sentence may contain an unfilled gap, the 32 charts must be numbered 1 to 32
    without a hole, and every number that the code looks up in FACTS must exist in the facts file the notebook saved.

WHY WE NEED IT
    The explanations and the notes are the point of this notebook for a reader who is learning. A cell that lost its
    explanation during an edit, a sentence that still holds a "{{gap}}", or a findings file that no longer matches the code
    would otherwise be found by the reader and not by us.

WHAT HAPPENS WHEN IT RUNS
    The notebook is read as plain JSON/nbformat; nothing is executed. The last tests read the two files that the notebook
    writes (reports/p4_facts.json and reports/p4_eda_findings.csv) and are skipped when those files are not there yet
    (run the notebook once to create them).

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    A teaching notebook whose structure cannot silently rot.
"""
import ast
import json
import re
from pathlib import Path

import nbformat
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "04_eda_playlist_song_artist.ipynb"
FACTS_FILE = ROOT / "reports" / "p4_facts.json"
FINDINGS_FILE = ROOT / "reports" / "p4_eda_findings.csv"
FIGURE_DIR = ROOT / "reports" / "figures"

FIVE_PARTS = ["**What this code is.**", "**Why we need it.**", "**What will happen.**", "**Expected outcome.**", "**Motive.**"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U00002B00-\U00002BFF\U0000FE0F]")

pytestmark = pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebooks/04_eda_playlist_song_artist.ipynb is not there")


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def _code_cells(notebook):
    return [(index, cell) for index, cell in enumerate(notebook.cells) if cell.cell_type == "code"]


def _markdown_text(notebook):
    return [(index, cell.source) for index, cell in enumerate(notebook.cells) if cell.cell_type == "markdown"]


# ================================================================================ the file itself
def test_the_notebook_is_valid_and_has_the_cells_we_expect(notebook):
    nbformat.validate(notebook)
    assert len(_code_cells(notebook)) == 38
    assert notebook.cells[0].cell_type == "markdown" and notebook.cells[0].source.startswith("# Phase 4")


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
    charts = [(index, cell) for index, cell in _code_cells(notebook) if "save_fig(" in cell.source]
    assert len(charts) == 32
    for index, _ in charts:
        below = notebook.cells[index + 1]
        assert below.cell_type == "markdown" and below.source.startswith("**What to notice**"), f"chart in cell {index} has no note"


def test_the_32_charts_are_numbered_one_to_thirty_two_exactly_once(notebook):
    names = []
    for _, cell in _code_cells(notebook):
        names += re.findall(r'save_fig\(\s*fig\s*,\s*"(p4_\d\d_[a-z0-9_]+)"', cell.source)
    assert [int(name[3:5]) for name in names] == list(range(1, 33))
    assert len(set(names)) == 32


def test_the_nine_sections_follow_each_other(notebook):
    numbers = [int(match.group(1)) for _, text in _markdown_text(notebook) for match in [re.match(r"## Section (\d+) - ", text)] if match]
    assert numbers == list(range(1, 10))


# ================================================================================ the sentences
def test_no_sentence_contains_an_unfilled_gap(notebook):
    for index, text in _markdown_text(notebook):
        for marker in ("{{", "}}", "??", "\u2026", "TODO", "TBD", "XXX"):
            assert marker not in text, f"markdown cell {index} contains {marker!r}"
        assert not re.search(r"\bnan\b|\binf\b", text), f"markdown cell {index} prints a nan or an inf"


def test_no_emoji_anywhere_and_no_cloud_storage_in_the_code_or_the_text(notebook):
    for index, cell in enumerate(notebook.cells):
        assert not EMOJI.search(cell.source), f"cell {index} contains an emoji"
        assert not re.search(r"\bS3\b|boto3|s3://", cell.source, flags=re.IGNORECASE), f"cell {index} mentions S3"


def test_the_notebook_never_calls_the_workflow_traditional_etl(notebook):
    assert not any("traditional ETL" in cell.source for cell in notebook.cells)


# ================================================================================ the two files the notebook writes
@pytest.mark.skipif(not FACTS_FILE.exists(), reason="run the notebook once to create reports/p4_facts.json")
def test_every_fact_the_code_looks_up_exists_in_the_facts_file(notebook):
    facts = json.loads(FACTS_FILE.read_text(encoding="utf-8"))
    wanted = set()
    for _, cell in _code_cells(notebook):
        wanted |= set(re.findall(r"\b(?:FACTS|F)\[\s*['\"]([a-z0-9_]+)['\"]\s*\]", cell.source))
    assert len(wanted) > 300
    assert wanted - set(facts) == set(), "the code uses facts that the saved file does not hold"


@pytest.mark.skipif(not FACTS_FILE.exists(), reason="run the notebook once to create reports/p4_facts.json")
def test_the_facts_file_is_sorted_and_holds_only_plain_values():
    text = FACTS_FILE.read_text(encoding="utf-8")
    facts = json.loads(text)
    assert list(facts) == sorted(facts)
    assert all(isinstance(value, (bool, int, float, str)) for value in facts.values())
    assert text.endswith("}\n")


@pytest.mark.skipif(not FINDINGS_FILE.exists(), reason="run the notebook once to create reports/p4_eda_findings.csv")
def test_the_findings_file_is_complete():
    findings = pd.read_csv(FINDINGS_FILE, encoding="utf-8-sig", keep_default_na=False)
    assert list(findings.columns) == ["finding_id", "block", "figure", "finding", "evidence", "caveat"]
    assert findings["finding_id"].tolist() == [f"F{number:02d}" for number in range(1, len(findings) + 1)]
    assert len(findings) >= 25
    for column in ("block", "figure", "finding", "evidence", "caveat"):
        assert (findings[column].str.strip() != "").all(), f"an empty {column}"
    assert set(findings["block"]) == {"Playlist ranking", "Song-level", "Artist", "Popularity", "Content attributes", "Overview"}
    referenced = {int(number) for text in findings["figure"] for number in re.findall(r"\d+", text)}
    assert referenced <= set(range(1, 33))
    assert not findings["evidence"].str.contains(r"[{}]|\bnan\b", regex=True).any()


@pytest.mark.skipif(not FIGURE_DIR.exists(), reason="no figure folder yet")
def test_the_chart_files_named_in_the_code_exist_and_are_real_pictures(notebook):
    names = []
    for _, cell in _code_cells(notebook):
        names += re.findall(r'save_fig\(\s*fig\s*,\s*"(p4_\d\d_[a-z0-9_]+)"', cell.source)
    missing = [name for name in names if not (FIGURE_DIR / f"{name}.png").exists()]
    if missing:
        pytest.skip(f"run the notebook once to draw the charts (missing: {missing[:3]} ...)")
    for name in names:
        header = (FIGURE_DIR / f"{name}.png").read_bytes()[:8]
        assert header == b"\x89PNG\r\n\x1a\n", f"{name}.png is not a PNG file"
