"""
tests/test_app_nav.py  -  the list of dashboard pages (app/nav.py) and the house rules for the dashboard source files.

WHAT THIS FILE IS
    Checks that the menu list is complete and tidy (18 pages in six sections, every page file exists and has a `render` function, no
    page file is left out of the menu, names and web addresses are unique), that the dashboard assets exist, that every dashboard file
    starts with the five-heading explanation, and that no dashboard source file contains an emoji.

WHY WE NEED IT
    The menu, the "where to go next" cards on the overview and the page files must agree. A page file that nobody lists is never shown,
    and a listed page without a `render` function breaks the whole menu with one click. The file checks keep the project in the form
    that was agreed (plain-English headers, no emoji).

WHAT HAPPENS WHEN IT RUNS
    The files are read as text and parsed (never imported or run), so the tests need neither Streamlit nor the data.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Adding a page later means adding it in one place, and the tests say if something was forgotten.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from app import nav

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ROOT / "app" / "views"
HEADINGS = ["WHAT THIS FILE IS", "WHY WE NEED IT", "WHAT HAPPENS WHEN IT RUNS", "EXPECTED OUTCOME", "MOTIVE"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-⛿✅❌✨⭐⬆⬇]")
SECTIONS = ["Overview", "Playlist", "Songs", "Artists", "Content", "Research"]


# ================================================================================================================ the list
def test_the_menu_has_eighteen_pages_in_six_sections():
    assert len(nav.PAGES) == 18                                    # the brief asked for more than nine or ten
    assert nav.sections() == SECTIONS


def test_every_entry_has_six_filled_fields():
    for entry in nav.PAGES:
        assert len(entry) == 6 and all(isinstance(field, str) and field.strip() for field in entry), entry


def test_names_web_addresses_and_modules_are_unique():
    titles = [entry[2] for entry in nav.PAGES]
    slugs = [entry[1] for entry in nav.PAGES]
    modules = [entry[3] for entry in nav.PAGES]
    for values in (titles, slugs, modules):
        assert len(set(values)) == len(values)


def test_web_addresses_are_short_words_joined_by_hyphens():
    for _section, slug, *_rest in nav.PAGES:
        assert re.fullmatch(r"[a-z]+(-[a-z]+)*", slug), slug


def test_the_first_page_is_the_overview():
    assert nav.PAGES[0][1] == "overview" and nav.PAGES[0][3] == "overview"


def test_the_pages_of_a_section_are_next_to_each_other():
    order = [entry[0] for entry in nav.PAGES]
    collapsed = [section for i, section in enumerate(order) if i == 0 or order[i - 1] != section]
    assert collapsed == SECTIONS


def test_icons_are_material_icon_names_not_emoji():
    for entry in nav.PAGES:
        assert re.fullmatch(r"[a-z]+(_[a-z]+)*", entry[4]), entry[4]


def test_descriptions_are_one_sentence_that_ends_with_a_full_stop():
    for entry in nav.PAGES:
        description = entry[5]
        assert description.endswith(".") and 30 <= len(description) <= 160, description
        assert not EMOJI.search(description)


def test_titles_are_short_enough_for_the_top_menu():
    for entry in nav.PAGES:
        assert len(entry[2]) <= 28, entry[2]


# ================================================================================================================= the files
def test_every_listed_page_has_a_file_with_a_render_function():
    for entry in nav.PAGES:
        path = VIEWS / f"{entry[3]}.py"
        assert path.exists(), path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert "render" in functions, f"{path.name} has no render()"


def test_no_page_file_is_left_out_of_the_menu():
    files = {path.stem for path in VIEWS.glob("*.py") if path.stem != "__init__"}
    assert files == {entry[3] for entry in nav.PAGES}


def test_the_entry_file_and_the_assets_exist():
    assert (ROOT / "app" / "streamlit_app.py").exists()
    for asset in ("favicon.png", "logo.svg"):
        assert (ROOT / "app" / "assets" / asset).stat().st_size > 0


def test_the_streamlit_theme_file_exists_and_uses_the_palette():
    config = ROOT / ".streamlit" / "config.toml"
    assert config.exists()
    text = config.read_text(encoding="utf-8").lower()
    assert "[theme]" in text and "#2a78d6" in text


# ========================================================================================================= the house rules
def dashboard_files() -> list:
    files = sorted((ROOT / "app").rglob("*.py")) + sorted((ROOT / "src" / "dash").glob("*.py"))
    files += sorted((ROOT / "tests").glob("test_app_*.py")) + sorted((ROOT / "tests").glob("test_dash_*.py"))
    return [path for path in files if "__pycache__" not in path.parts]


@pytest.mark.parametrize("path", dashboard_files(), ids=lambda p: str(p.relative_to(ROOT)).replace("\\", "/"))
def test_every_file_starts_with_the_five_heading_explanation(path):
    docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")), clean=False) or ""
    assert docstring.strip(), f"{path.name} has no header docstring"
    positions = [docstring.find(heading) for heading in HEADINGS]
    assert all(position >= 0 for position in positions), (path.name, [h for h, p in zip(HEADINGS, positions) if p < 0])
    assert positions == sorted(positions), f"{path.name}: the five headings are not in the agreed order"


@pytest.mark.parametrize("path", sorted((ROOT / "app").rglob("*.py")), ids=lambda p: str(p.relative_to(ROOT)).replace("\\", "/"))
def test_no_dashboard_source_file_contains_an_emoji(path):
    assert not EMOJI.search(path.read_text(encoding="utf-8")), f"{path.name} contains an emoji; use a Material icon or plain words"


def test_the_pages_never_use_the_word_traditional_etl():
    """A wording rule of the project: the workflow is described by what it does, never as 'traditional ETL'."""
    for path in sorted((ROOT / "app").rglob("*.py")) + sorted((ROOT / "src" / "dash").glob("*.py")):
        assert "traditional etl" not in path.read_text(encoding="utf-8").lower(), path.name
