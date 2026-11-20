"""
tests/test_notebook_06c.py  -  checks of the Phase 6C notebook (notebooks/06c_clustering.ipynb) and of the fourteen files it writes.

WHAT THIS FILE IS
    Checks of what the notebook IS and of what its saved files SAY, not of what the grouping method computes (k-means, the silhouette, the
    adjusted Rand index, the artist bootstrap and the artist features are tested in test_cluster_*.py). The notebook must be valid, delivered
    without outputs, every code cell must compile and have its five-part plain-English explanation directly above it, every chart must have a
    "What to notice" note directly below it, no sentence may contain an unfilled gap or a double negative such as "fell by about -0.5", the
    11 charts must be numbered 1 to 11 without a hole, and every number that the code looks up in FACTS must exist in the facts file that the
    notebook saved.
    The later tests read the files the notebook writes. They check that the tables add up to the facts and to each other (832 songs in five
    archetypes, 243 artists in four segments, 944 songs and 27,750 chart rows behind the artist table), that the rule for the number of
    groups, applied again to the saved scans, gives the groups the notebook uses, and then that the claims the text makes hold in the tables:
    which archetype is the firmest and which the softest, which stress runs move the grouping and which do not, that the staying of the
    lasting hits and the slow climbers is estimated and that of the other three is not (too few songs are left on day 30), that the
    arrival-day risk thirds are ordered by archetype, which arrival-day facts go with an archetype and which do not, what the four artist
    segments are, how the two groupings relate, and that the closing text quotes the saved numbers. The claims about artists and archetypes
    are re-derived from the saved song table by a second, independent piece of code (a pairs count, an adjusted Rand index, a top ten).

WHY WE NEED IT
    The explanations, the notes and the closing statements are the point of this notebook, and the saved tables are what the dashboard, the
    Power BI report and the paper will read. A cell that lost its explanation during an edit, a sentence that still holds a "{{gap}}", a
    table that no longer adds up to the facts, or a closing text that disagrees with the saved numbers would otherwise be found by the
    reader and not by us. A clustering notebook has one more risk: a grouping that is not firm is easy to present as if it were, so the
    checks that carry the honesty of the text (the firmness of every archetype, the soft ones named, the stress runs, the shuffled
    comparison) are written as tests too.

WHAT HAPPENS WHEN IT RUNS
    The notebook is read as plain JSON/nbformat; nothing is executed. The tests that read the saved files are skipped when those files are
    not there yet (run the notebook once to create them).

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    A teaching notebook, and the tables that later phases read, whose structure and claims cannot silently rot.
"""
import ast
import json
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "06c_clustering.ipynb"
REPORTS = ROOT / "reports"
FIGURE_DIR = REPORTS / "figures"
FACTS_FILE = REPORTS / "p6c_facts.json"

# the thirteen tables of the notebook and the shape each has on the real data (rows, columns)
TABLES = {"p6c_k_scan_songs": (7, 10), "p6c_archetypes": (5, 15), "p6c_song_archetypes": (832, 14), "p6c_first_months": (832, 32),
          "p6c_stability": (5, 12), "p6c_stress_tests": (10, 7), "p6c_landmark": (5, 15), "p6c_risk_thirds": (5, 7),
          "p6c_attributes": (48, 5), "p6c_k_scan_artists": (6, 9), "p6c_artist_segments": (4, 17), "p6c_artists": (243, 14),
          "p6c_archetype_by_segment": (20, 7)}
TABLE_FILES = {name: REPORTS / f"{name}.csv" for name in TABLES}
ALL_FILES = [FACTS_FILE, *TABLE_FILES.values()]

# the five song archetypes and the four artist segments, in the order of the notebook, with the short word that the facts use for each
NAMES = ["Brief visitors", "Quick fades", "Month-long fades", "Slow climbers", "Lasting hits"]
KEYS = ["brief", "quick", "month_long", "slow", "lasting"]
SEGMENTS = ["Passing visitors", "Mid-chart stayers", "Top-chart hit-makers", "Catalogue artists"]
SEGMENT_KEYS = ["visitors", "stayers", "hitmakers", "catalogue"]

FIVE_PARTS = ["**What this code is.**", "**Why we need it.**", "**What will happen.**", "**Expected outcome.**", "**Motive.**"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U00002B00-\U00002BFF\U0000FE0F]")

pytestmark = pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebooks/06c_clustering.ipynb is not there")


def _needs(*files):
    missing = [file.name for file in files if not file.exists()]
    return pytest.mark.skipif(bool(missing), reason=f"run the notebook once to create {', '.join(missing)}")


ALL_TABLES = _needs(*TABLE_FILES.values())
EVERYTHING = _needs(*ALL_FILES)


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


@pytest.fixture(scope="module")
def facts():
    return json.loads(FACTS_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def table():
    """table('p6c_archetypes') reads one of the thirteen saved tables."""
    cache = {}

    def read(name):
        if name not in cache:
            cache[name] = pd.read_csv(TABLE_FILES[name], encoding="utf-8-sig")
        return cache[name]
    return read


def _code_cells(notebook):
    return [(index, cell) for index, cell in enumerate(notebook.cells) if cell.cell_type == "code"]


def _markdown_text(notebook):
    return [(index, cell.source) for index, cell in enumerate(notebook.cells) if cell.cell_type == "markdown"]


def _all_text(notebook):
    return "\n".join(text for _, text in _markdown_text(notebook))


# a chart is a cell that saves a figure: save_fig(fig, "p6c_10_artist_segments", FIG_DIR). The saving cell of Section 10 names all the figures
# in a list, which is why the call itself is looked for and not any string that looks like a figure name.
CHART_NAME = re.compile(r"""save_fig\(\s*fig\s*,\s*["'](p6c_\d\d_[a-z0-9_]+)["']""")


def _chart_cells(notebook):
    return [(index, cell) for index, cell in _code_cells(notebook) if CHART_NAME.search(cell.source)]


def _chart_names(notebook):
    names = []
    for _, cell in _chart_cells(notebook):
        names += CHART_NAME.findall(cell.source)
    return names


def choose_k(scan, floor, dissolved):
    """
    The rule of Section 3, written here a second time so that the saved scans are judged by the rule and not by the code that made them.
    Count up from the smallest number of groups and stop at the first one in which some group dissolves (mean Jaccard below `dissolved`);
    among the numbers before that point take the largest in which every group is firm (mean Jaccard at least `floor`).
    """
    before = []
    for row in scan.itertuples():
        if row.weakest < dissolved:
            break
        before.append(row)
    firm = [row.k for row in before if row.weakest >= floor]
    return max(firm), before[-1].k


def adjusted_rand(a, b):
    """The adjusted Rand index of two labelings, from the table of counts (a second implementation, independent of src.cluster)."""
    counts = pd.crosstab(np.asarray(a), np.asarray(b)).to_numpy().astype(float)
    pairs = lambda x: x * (x - 1) / 2
    inside, rows, cols, total = pairs(counts).sum(), pairs(counts.sum(axis=1)).sum(), pairs(counts.sum(axis=0)).sum(), pairs(counts.sum())
    expected = rows * cols / total
    return float((inside - expected) / ((rows + cols) / 2 - expected))


# ================================================================================ the file itself
def test_the_notebook_is_valid_and_has_the_cells_we_expect(notebook):
    nbformat.validate(notebook)
    assert len(notebook.cells) == 77 and len(_code_cells(notebook)) == 26
    assert notebook.cells[0].cell_type == "markdown" and notebook.cells[0].source.startswith("# Phase 6C")


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


def test_the_grouping_methods_are_the_project_own_and_no_machine_learning_library_is_imported(notebook):
    """k-means, the silhouette and the adjusted Rand index live in src/cluster; scipy is used only for Ward's method and the matching of labels."""
    for index, cell in _code_cells(notebook):
        assert not re.search(r"^\s*(import|from)\s+(sklearn|lifelines|sksurv|statsmodels)\b", cell.source, flags=re.M), f"cell {index} imports a library that does the grouping"
    sources = "\n".join(cell.source for _, cell in _code_cells(notebook))
    assert "src.cluster" in sources and "from src.cluster.kmeans import" in sources and "from src.cluster.stability import" in sources


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
    assert len(charts) == 11
    for index, _ in charts:
        below = notebook.cells[index + 1]
        assert below.cell_type == "markdown" and below.source.startswith("**What to notice**"), f"chart in cell {index} has no note"


def test_the_11_charts_are_numbered_one_to_eleven_exactly_once(notebook):
    names = _chart_names(notebook)
    assert [int(name[4:6]) for name in names] == list(range(1, 12))
    assert len(set(names)) == 11


def test_every_chart_cell_and_its_explanation_carry_the_number_of_their_chart(notebook):
    for index, cell in _chart_cells(notebook):
        number = int(CHART_NAME.search(cell.source).group(1)[4:6])
        assert cell.source.startswith(f"# Figure {number:02d} "), f"cell {index} does not start with its figure number"
        what_it_is = notebook.cells[index - 1].source.split("**Why we need it.**")[0]
        assert re.search(rf"\bFigure {number}\b", what_it_is), f"the first part of the explanation above cell {index} does not name Figure {number}"


def test_the_saving_cell_lists_the_same_charts_as_the_chart_cells(notebook):
    """Section 10 checks that every chart was written during the run, so the names it holds must be the names of the charts the notebook draws."""
    listed = None
    for _, cell in _code_cells(notebook):
        found = re.search(r"FIGURE_NAMES\s*=\s*\[(.*?)\]", cell.source, flags=re.S)
        if found:
            listed = re.findall(r"p6c_\d\d_[a-z0-9_]+", found.group(1))
    assert listed == _chart_names(notebook)


def test_the_ten_sections_follow_each_other(notebook):
    numbers = [int(match.group(1)) for _, text in _markdown_text(notebook) for match in [re.match(r"## Section (\d+) - ", text)] if match]
    assert numbers == list(range(1, 11))


def test_every_figure_and_section_the_text_refers_to_exists(notebook):
    text = _all_text(notebook)
    figures = {int(number) for number in re.findall(r"\bFigures? (\d+)", text)}
    sections = {int(number) for number in re.findall(r"\bSections? (\d+)", text)}
    assert figures and figures <= set(range(1, 12)), f"a reference to a figure that does not exist: {sorted(figures - set(range(1, 12)))}"
    assert sections and sections <= set(range(1, 11)), f"a reference to a section that does not exist: {sorted(sections - set(range(1, 11)))}"


def test_the_introduction_names_every_file_and_the_number_of_charts(notebook):
    intro = notebook.cells[0].source
    for name in TABLES:
        assert f"`{name}.csv`" in intro, f"the introduction does not name {name}.csv"
    assert "`p6c_facts.json`" in intro and "11 charts" in intro
    assert len(TABLES) + 1 == 14
    rows = [line for line in intro.splitlines() if re.match(r"\| \d+ \|", line)]
    assert len(rows) == 10, "the table of sections in the introduction must list the ten sections"


def test_the_notebook_ends_with_the_closing_summary_and_the_hand_over_to_the_next_phases(notebook):
    last = notebook.cells[-1]
    assert last.cell_type == "markdown" and last.source.startswith("### What Phase 6C adds up to")
    for phase in ("Phase 7", "Phase 8"):
        assert phase in last.source, f"the closing text does not mention {phase}"
    for name in ("p6c_song_archetypes.csv", "p6c_first_months.csv", "p6c_artists.csv", "p6c_artist_segments.csv", "p6c_facts.json"):
        assert name in last.source, f"the closing text does not name {name}, which the next phases read"


def test_the_closing_keeps_the_caveats_that_the_brief_requires(notebook):
    closing = notebook.cells[-1].source.lower()
    assert "association" in closing, "the closing must say that every result is an association"
    assert "not boxes into which a new song can be put" in closing, "the closing must say that the archetypes are not a way to sort new songs"
    assert "continuum" in closing, "the closing must say that the groups are regions of a continuum"
    assert "fixed after a first look" in closing, "the closing must disclose that the rule for the number of groups was fixed after looking at the data"
    assert "have no whole first month" in closing, "the closing must disclose the 62 songs left out"
    assert "partly built in" in closing, "the closing must say that the overlap of segments and archetypes is partly built in"
    assert "soft ones" in closing and "least certain" in closing, "the closing must name the soft archetypes and the least certain segment"
    assert "only a little more alike" in closing, "the closing must say that the songs of one artist are only a little more alike than chance"


def test_the_introduction_keeps_the_caveats_and_the_ground_rules(notebook):
    intro = notebook.cells[0].source
    assert "always returns groups" in intro, "the introduction must warn that a clustering method returns groups whether or not the data hold groups"
    assert "not prediction and not recommendation" in intro
    assert "every difference is an association" in intro
    assert "Groups are descriptions, not species" in intro and "The artist is the unit of doubt" in intro


# ================================================================================ the sentences
def test_no_sentence_contains_an_unfilled_gap(notebook):
    for index, text in _markdown_text(notebook):
        for marker in ("{{", "}}", "[[", "]]", "??", "\u2026", "TODO", "TBD", "XXX"):
            assert marker not in text, f"markdown cell {index} contains {marker!r}"
        assert not re.search(r"\bnan\b|\binf\b", text), f"markdown cell {index} prints a nan or an inf"


def test_no_sentence_says_that_something_fell_or_was_less_by_a_negative_number(notebook):
    """'fell by about -0.47 points' and '-3.4 points less' are double negatives: the sign is already in the verb."""
    patterns = [r"\b(fell|falls|fall|dropped|drops|drop|decreased|decreases|lower|smaller|less|fewer)\b (by|of) (about )?[-\u2212]\d",
                r"(?<![\w.])[-\u2212]\d[\d.,]* (points|chart days) \*?(less|lower|fewer|smaller)\*?\b",
                r"\buses [-\u2212]\d", r"[+]\d[\d.]* more chart days"]
    for index, text in _markdown_text(notebook):
        for pattern in patterns:
            assert not re.search(pattern, text), f"markdown cell {index} has a double negative or a doubled sign ({pattern})"


def test_no_percent_points_and_the_hand_typed_counts_are_the_right_ones(notebook):
    text = _all_text(notebook)
    assert not re.search(r"\d%\s+points", text), "write 'percentage points', not '% points'"
    assert "11 charts" in notebook.cells[0].source and "all 11 charts of this notebook" in text


def test_no_emoji_anywhere_and_no_cloud_storage_in_the_code_or_the_text(notebook):
    for index, cell in enumerate(notebook.cells):
        assert not EMOJI.search(cell.source), f"cell {index} contains an emoji"
        assert not re.search(r"(?i:boto3|s3://|amazon s3|aws s3)|\bS3\b", cell.source), f"cell {index} mentions S3"


def test_the_notebook_never_calls_the_workflow_traditional_etl(notebook):
    assert not any("traditional ETL" in cell.source for cell in notebook.cells)


# ================================================================================ the facts file
@_needs(FACTS_FILE)
def test_every_fact_the_code_looks_up_exists_in_the_facts_file(notebook, facts):
    wanted = set()
    for _, cell in _code_cells(notebook):
        wanted |= set(re.findall(r"\bFACTS\[\s*['\"]([a-z0-9_]+)['\"]\s*\]", cell.source))
    assert len(wanted) > 100
    assert wanted - set(facts) == set(), "the code uses facts that the saved file does not hold"


@_needs(FACTS_FILE)
def test_the_facts_file_is_sorted_and_holds_only_plain_finite_values():
    text = FACTS_FILE.read_text(encoding="utf-8")
    facts = json.loads(text)
    assert list(facts) == sorted(facts)
    assert len(facts) > 600
    assert all(isinstance(value, (bool, int, float, str)) for value in facts.values())
    assert all(np.isfinite(value) for value in facts.values() if isinstance(value, float) and not isinstance(value, bool))
    assert text.endswith("}\n")


@_needs(FACTS_FILE)
def test_the_counts_in_the_facts_add_up(facts):
    assert facts["n_kept"] + facts["n_dropped"] == facts["n_entrants"], "the songs with a whole first month and the 62 without are all the entrants"
    assert facts["n_entrants"] < facts["songs"], "the songs that were already on the chart on the first snapshot day have no arrival day"
    assert sum(facts[f"n_{key}"] for key in KEYS) == facts["n_kept"] == facts["n_song_rows"]
    assert sum(facts[f"artists_{key}"] for key in KEYS) >= facts["n_kept_artists"], "an artist can have songs in more than one archetype"
    assert facts["n_kept_artists"] <= facts["n_entrant_artists"] <= facts["n_artists_all"]
    assert facts["n_alive"] == sum(facts[f"alive_{key}"] for key in KEYS)
    assert facts["n_third_lowest"] + facts["n_third_middle"] + facts["n_third_highest"] == facts["n_kept"]
    assert facts["horizon"] == 30 and facts["horizon_less_one"] == 29 and facts["n_figures"] == 11
    assert facts["n_boot"] == 1000 and facts["n_perm"] == 1000 and facts["n_boot_scan"] < facts["n_boot"] and facts["n_init"] == 30
    assert facts["k"] == 5 and facts["k_artists"] == 4 and facts["k_less_one"] == 4 and facts["k_max"] == 8


# ================================================================================ the saved files
@ALL_TABLES
def test_the_thirteen_tables_have_the_shapes_the_notebook_reports(table):
    for name, shape in TABLES.items():
        assert table(name).shape == shape, f"{name} has the shape {table(name).shape}, expected {shape}"


@_needs(*TABLE_FILES.values(), FACTS_FILE)
def test_the_eleven_charts_the_code_names_are_in_the_figures_folder(notebook):
    names = _chart_names(notebook)
    assert len(names) == 11
    for name in names:
        path = FIGURE_DIR / f"{name}.png"
        assert path.exists() and path.stat().st_size > 20_000, f"{path.name} is missing or empty"


# ================================================================================ the claims of Section 3
@ALL_TABLES
def test_the_rule_for_the_number_of_groups_gives_the_same_answer_on_the_saved_scans(table, facts):
    songs, artists = table("p6c_k_scan_songs"), table("p6c_k_scan_artists")
    assert facts["floor"] == 0.75 and facts["dissolved"] == 0.60
    assert choose_k(songs, facts["floor"], facts["dissolved"]) == (facts["k"], facts["k_next"] - 1) == (5, 5)
    assert choose_k(artists, facts["floor"], facts["dissolved"]) == (facts["k_artists"], facts["k_artists_stop"]) == (4, 5)
    for scan, chosen, first_k, last_k in ((songs, 5, 2, 8), (artists, 4, 2, 7)):
        assert list(scan["k"]) == list(range(first_k, last_k + 1))
        assert list(scan.loc[scan["chosen"], "k"]) == [chosen], "exactly one number of groups is marked as chosen"
        assert ((scan["dissolved_groups"] > 0) == (scan["weakest"] < facts["dissolved"])).all(), "a group has dissolved exactly when the weakest mean Jaccard is under 0.60"
        assert ((scan["firm_groups"] == scan["k"]) == (scan["weakest"] >= facts["floor"])).all(), "every group is firm exactly when the weakest mean Jaccard is at least 0.75"
        assert (scan["smallest_group"] >= 1).all() and scan["silhouette"].between(-1, 1).all() and scan["ari_mean"].between(-1, 1).all()
        assert (scan["ari_p10"] <= scan["ari_mean"] + 1e-12).all()
    assert songs.loc[songs["chosen"], "weakest"].iloc[0] >= facts["floor"] and songs.loc[songs["k"] == facts["k_next"], "weakest"].iloc[0] < facts["dissolved"]


@ALL_TABLES
def test_the_scan_facts_are_the_scan_tables(table, facts):
    songs, artists = table("p6c_k_scan_songs"), table("p6c_k_scan_artists")
    for row in songs.itertuples():
        assert facts[f"sil_k{row.k}"] == pytest.approx(row.silhouette, rel=1e-9)
        assert facts[f"weakest_k{row.k}"] == pytest.approx(row.weakest, rel=1e-9)
        assert facts[f"ari_k{row.k}"] == pytest.approx(row.ari_mean, rel=1e-9)
        assert facts[f"smallest_k{row.k}"] == row.smallest_group
    for row in artists.itertuples():
        assert facts[f"art_sil_k{row.k}"] == pytest.approx(row.silhouette, rel=1e-9)
        assert facts[f"art_weakest_k{row.k}"] == pytest.approx(row.weakest, rel=1e-9)
        assert facts[f"art_ari_k{row.k}"] == pytest.approx(row.ari_mean, rel=1e-9)
    assert facts["sil_chosen"] == facts["sil_k5"] and facts["weakest_chosen"] == facts["weakest_k5"] and facts["ari_chosen"] == facts["ari_k5"]
    assert facts["weakest_next"] == facts["weakest_k6"] and facts["art_sil_chosen"] == facts["art_sil_k4"]
    assert facts["sil_best_k"] == int(songs.loc[songs["silhouette"].idxmax(), "k"]) == 2, "the silhouette on its own would pick two groups, which is why a firmness rule is used"
    assert abs(facts["ari_chosen"] - facts["boot_ari_mean"]) < 0.01, "the 300-draw scan and the 1000-draw run give the same picture of the chosen grouping"


# ================================================================================ the claims of Section 4
@ALL_TABLES
def test_the_archetype_table_adds_up_and_agrees_with_the_facts(table, facts):
    arch = table("p6c_archetypes")
    assert list(arch["archetype"]) == NAMES
    assert list(arch["songs"]) == [facts[f"n_{key}"] for key in KEYS] == [421, 192, 73, 69, 77]
    assert int(arch["songs"].sum()) == facts["n_kept"] == 832
    assert np.allclose(arch["share_of_songs"], arch["songs"] / 832) and np.allclose(arch["share_of_songs"], [facts[f"share_{key}"] for key in KEYS])
    assert facts["n_smallest"] == int(arch["songs"].min()) == facts["n_slow"], "the smallest archetype is the slow climbers"
    for column, stem in (("artists", "artists"), ("debut_place", "debut"), ("best_place", "best"), ("chart_days", "days"), ("last_chart_day_median", "lastday"),
                         ("on_chart_day_30", "day30"), ("place_day_30_median", "place30"), ("score_day_1", "score1"), ("score_day_30", "score30")):
        assert np.allclose(arch[column], [facts[f"{stem}_{key}"] for key in KEYS], rtol=1e-9, atol=1e-12), f"{column} of the table is not the fact {stem}_*"
    assert list(arch["biggest_artist"]) == [facts[f"top_artist_name_{key}"] for key in KEYS]
    assert np.allclose(arch["biggest_artist_songs"] / arch["songs"], [facts[f"top_artist_share_{key}"] for key in KEYS])


@ALL_TABLES
def test_the_archetypes_have_the_shapes_the_names_say(table, facts):
    arch = table("p6c_archetypes").set_index("archetype")
    assert arch["chart_days"].is_monotonic_increasing, "brief visitors < quick fades < month-long fades < slow climbers < lasting hits in chart days"
    assert arch["chart_days"].iloc[0] < 5 and arch["chart_days"].iloc[-1] > 29
    assert list(arch.index[arch["score_day_30"] > arch["score_day_1"]]) == ["Slow climbers"], "the slow climbers are the only archetype that gains during the month"
    assert arch["on_chart_day_30"].iloc[3:].min() > 0.85 > 0.15 > arch["on_chart_day_30"].iloc[:3].max(), "only the slow climbers and the lasting hits are still there on day 30"
    assert arch["debut_place"].idxmax() == "Slow climbers" and arch["debut_place"].idxmin() == "Lasting hits"
    assert arch.loc[["Quick fades", "Month-long fades", "Lasting hits"], "debut_place"].max() < 15, "fades and hits arrive high"
    assert arch.loc["Brief visitors", "debut_place"] > 30 and arch.loc["Slow climbers", "debut_place"] > 35, "brief visitors and slow climbers arrive low"
    assert arch["last_chart_day_median"].iloc[0] < arch["last_chart_day_median"].iloc[1] < arch["last_chart_day_median"].iloc[2] < arch["last_chart_day_median"].iloc[3]
    assert arch.loc["Lasting hits", "best_place"] < 5 < 10 < arch.loc["Slow climbers", "best_place"] < 20


@ALL_TABLES
def test_the_song_table_has_one_row_per_song_and_agrees_with_the_facts_and_the_other_tables(table, facts):
    songs, arch = table("p6c_song_archetypes"), table("p6c_archetypes")
    assert not songs["track_id"].duplicated().any() and len(songs) == facts["n_kept"]
    assert songs["primary_artist"].nunique() == facts["n_kept_artists"] == 223
    assert songs["archetype"].value_counts().reindex(NAMES).tolist() == arch["songs"].tolist()
    assert songs.groupby("archetype")["primary_artist"].nunique().reindex(NAMES).tolist() == arch["artists"].tolist()
    assert songs["agreement"].between(0, 1).all() and set(songs["firmness"]) == {"firm", "likely", "border"}
    firm, likely, border = (songs["firmness"] == name for name in ("firm", "likely", "border"))
    assert (songs.loc[firm, "agreement"] >= 0.9).all() and (songs.loc[border, "agreement"] < 0.6).all()
    assert (songs.loc[likely, "agreement"] >= 0.6).all() and (songs.loc[likely, "agreement"] < 0.9).all()
    assert int(border.sum()) == facts["n_agree_below_60"] == 12 and facts["share_agree_below_60"] == pytest.approx(float(border.mean()), abs=1e-12)
    assert facts["share_agree_90"] == pytest.approx(float(firm.mean()), abs=1e-12)
    assert songs["artist_segment"].notna().all() and set(songs["artist_segment"]) == set(SEGMENTS)
    assert songs["risk_third_arrival"].value_counts().to_dict() == {"lowest risk": facts["n_third_lowest"], "middle": facts["n_third_middle"], "highest risk": facts["n_third_highest"]}
    entry = pd.to_datetime(songs["entry_date"])
    assert entry.min() == pd.Timestamp("2024-05-19") and entry.max() <= pd.Timestamp("2025-10-29"), "the songs arrive after the first snapshot day and early enough for a whole month"
    assert facts["first_date"] == "18 May 2024" and facts["last_date"] == "27 November 2025" and facts["dropped_first"] == "30 October 2025"
    assert songs["entry_snapshot"].between(1, facts["snapshots"] - facts["horizon"]).all()
    assert songs["entry_rank"].between(1, 50).all()


@ALL_TABLES
def test_the_song_table_and_the_first_months_file_describe_the_same_songs(table, facts):
    songs, months = table("p6c_song_archetypes"), table("p6c_first_months")
    days = [f"day_{day:02d}" for day in range(1, 31)]
    assert list(months.columns) == ["track_id", "archetype", *days]
    assert (months["track_id"].to_numpy() == songs["track_id"].to_numpy()).all() and (months["archetype"] == songs["archetype"]).all()
    scores = months[days].to_numpy()
    assert scores.min() == 0 and scores.max() == 50 and np.allclose(scores, scores.round()), "a chart score is 51 minus a whole place, or 0 off the chart"
    assert (scores[:, 0] >= 1).all(), "a song is on the chart on the day it arrives"
    assert (songs["score_day_1"].to_numpy() == scores[:, 0]).all() and (songs["score_day_30"].to_numpy() == scores[:, 29]).all()
    assert (songs["entry_rank"].to_numpy() == 51 - scores[:, 0]).all(), "the arrival place is 51 minus the score of day 1"
    assert (songs["days_on_chart_first_month"].to_numpy() == (scores > 0).sum(axis=1)).all()
    assert facts["days_on_mean"] == pytest.approx(float((scores > 0).sum(axis=1).mean()), abs=1e-12) and facts["days_on_median"] == float(np.median((scores > 0).sum(axis=1)))
    assert facts["cells_on_chart"] == pytest.approx(float((scores > 0).mean()), abs=1e-12)
    assert facts["share_all_days"] == pytest.approx(float(((scores > 0).sum(axis=1) == 30).mean()), abs=1e-12)
    assert facts["share_3_days_or_fewer"] == pytest.approx(float(((scores > 0).sum(axis=1) <= 3).mean()), abs=1e-12)
    assert facts["share_on_last_day"] == pytest.approx(float((scores[:, 29] > 0).mean()), abs=1e-12)
    assert facts["debut_place_median"] == float(np.median(51 - scores[:, 0])) and facts["share_debut_top10"] == pytest.approx(float(((51 - scores[:, 0]) <= 10).mean()), abs=1e-12)
    means = months.groupby("archetype")[["day_01", "day_30"]].mean().reindex(NAMES)
    arch = table("p6c_archetypes")
    assert np.allclose(means["day_01"], arch["score_day_1"]) and np.allclose(means["day_30"], arch["score_day_30"])


# ================================================================================ the claims of Section 5
@ALL_TABLES
def test_the_stability_table_says_which_archetype_is_firmest_and_which_is_softest(table, facts):
    stab, arch, songs = table("p6c_stability"), table("p6c_archetypes"), table("p6c_song_archetypes")
    assert list(stab["archetype"]) == NAMES and list(stab["songs"]) == list(arch["songs"])
    assert ((stab["jaccard_low"] <= stab["jaccard_mean"]) & (stab["jaccard_mean"] <= stab["jaccard_high"])).all()
    assert (stab["jaccard_mean"] >= facts["floor"]).all(), "every archetype passes the rule that chose the number of groups"
    assert stab.loc[stab["jaccard_mean"].idxmax(), "archetype"] == facts["firmest_name"] == "Brief visitors"
    assert stab.loc[stab["jaccard_mean"].idxmin(), "archetype"] == facts["weakest_name"] == "Month-long fades"
    assert facts["firmest_jaccard"] == pytest.approx(float(stab["jaccard_mean"].max()), abs=1e-12) and facts["weakest_jaccard"] == pytest.approx(float(stab["jaccard_mean"].min()), abs=1e-12)
    order = stab.sort_values("jaccard_mean")["archetype"].tolist()
    assert order[:2] == ["Month-long fades", "Slow climbers"], "the two soft archetypes are the month-long fades and the slow climbers"
    assert stab["share_of_draws_dissolved"].iloc[[2, 3]].min() > 0.10 > stab["share_of_draws_dissolved"].iloc[[0, 1, 4]].max(), "only the two soft archetypes dissolve in more than 10% of the draws"
    for key, row in zip(KEYS, stab.itertuples()):
        for fact, value in ((f"jaccard_mean_{key}", row.jaccard_mean), (f"jaccard_low_{key}", row.jaccard_low), (f"jaccard_high_{key}", row.jaccard_high),
                            (f"share_of_draws_dissolved_{key}", row.share_of_draws_dissolved), (f"mean_agreement_{key}", row.mean_agreement),
                            (f"share_songs_agree_90_{key}", row.share_songs_agree_90), (f"share_songs_agree_below_60_{key}", row.share_songs_agree_below_60),
                            (f"heldout_recall_mean_{key}", row.heldout_recall_mean), (f"heldout_recall_min_{key}", row.heldout_recall_min)):
            assert facts[fact] == pytest.approx(float(value), abs=1e-12), fact
    # the agreement of every song, which the song table carries, is the number behind the averages of this table
    by = songs.groupby("archetype")["agreement"]
    assert np.allclose(by.mean().reindex(NAMES), stab["mean_agreement"], atol=1e-9)
    assert np.allclose(by.apply(lambda a: (a >= 0.9).mean()).reindex(NAMES), stab["share_songs_agree_90"], atol=1e-9)
    assert np.allclose(by.apply(lambda a: (a < 0.6).mean()).reindex(NAMES), stab["share_songs_agree_below_60"], atol=1e-9)
    assert int(round((stab["songs"] * stab["share_songs_agree_below_60"]).sum())) == facts["n_agree_below_60"]
    assert int(round((stab["songs"] * stab["share_songs_agree_90"]).sum())) == int(round(facts["share_agree_90"] * facts["n_kept"]))


@ALL_TABLES
def test_artists_held_out_go_back_to_their_own_archetype_and_the_bootstrap_agrees_with_the_scan(table, facts):
    stab = table("p6c_stability")
    assert ((stab["heldout_recall_min"] <= stab["heldout_recall_mean"]) & (stab["heldout_recall_mean"] <= stab["heldout_recall_max"])).all()
    assert (stab["heldout_recall_mean"] > 0.93).all() and facts["held_same_mean"] > 0.98 and facts["held_ari_mean"] > 0.95
    assert facts["held_same_min"] <= facts["held_same_mean"] and facts["held_ari_min"] <= facts["held_ari_mean"]
    assert 0.9 < facts["boot_ari_mean"] < 1 and facts["boot_ari_min"] < facts["boot_ari_p10"] < facts["boot_ari_mean"]
    assert 0.80 > facts["boot_ari_p10"] > 0.75, "the 10th percentile of the bootstrap ARI is about 0.80"
    assert facts["share_agree_90"] > 0.8 > 0.05 > facts["share_agree_below_60"]
    assert facts["logrank_p_naive"] < facts["logrank_p_artist"], "counting songs as independent can only make the evidence look stronger"


@ALL_TABLES
def test_the_stress_table_says_what_the_closing_says(table, facts):
    stress = table("p6c_stress_tests")
    assert list(stress["run"]) == ["Another random start", "Without the 10 biggest artists", f"Every artist capped at {facts['cap']} songs", "Ward's method",
                                   "Presence only", "Logarithmic score", "First month of 21 days", "First month of 45 days", "4 groups", "6 groups"]
    assert list(stress.columns) == ["run", "what_changes", "songs_used", "ari", "same_group_share", "ari_low", "ari_high"]
    ari = stress["ari"].to_numpy()
    for index, value in enumerate(ari, start=1):
        assert facts[f"stress_ari_{index}"] == pytest.approx(float(value), abs=1e-12), f"run {index}"
    for index in range(1, 9):
        assert facts[f"stress_same_{index}"] == pytest.approx(float(stress["same_group_share"].iloc[index - 1]), abs=1e-12), f"run {index}"
    assert stress["same_group_share"].iloc[8:].isna().all() and stress["same_group_share"].iloc[:8].between(0, 1).all(), "the share of songs in the same group is defined only when the number of groups is the same"
    assert stress["ari"].between(-1, 1).all()
    assert ari[0] > 0.99, "another random start finds the same groups"
    same_k = stress.iloc[:8]
    assert same_k["ari"].idxmin() == 4 and facts["stress_ari_min_same_k_name"] == "Presence only" == stress.loc[4, "run"]
    assert facts["stress_ari_min_same_k"] == pytest.approx(float(ari[4]), abs=1e-12) and ari[4] < 0.5
    lowest = set(stress["run"].iloc[np.argsort(ari)[:3]])
    assert lowest == {"Presence only", "Logarithmic score", "6 groups"}, "the grouping moves most when the picture of a song changes, or the number of groups"
    assert ari[3] > 0.85 and ari[6] > 0.9 and ari[1] > 0.75 and ari[7] > 0.75, "a change of method or of the length of the month leaves it nearly intact"
    assert list(stress["songs_used"]) == [832] * 7 + [facts["n_songs_45"], 832, 832] and facts["n_songs_45"] == 821 and facts["n_songs_21"] == 863
    cap = stress.iloc[2]
    assert cap["ari_low"] <= cap["ari"] <= cap["ari_high"] and stress[["ari_low", "ari_high"]].drop(index=2).isna().all().all()
    assert facts["cap_ari_low"] == pytest.approx(float(cap["ari_low"]), abs=1e-12) and facts["cap_ari_high"] == pytest.approx(float(cap["ari_high"]), abs=1e-12)
    assert facts["cap"] == 8 and facts["cap_repeats"] == 10 and 0 < facts["n_capped_fit"] < facts["n_kept"], "a cap at 8 songs per artist uses fewer songs than there are"


# ================================================================================ the claims of Section 6
@ALL_TABLES
def test_the_stay_after_day_30_is_estimated_only_where_enough_songs_are_left(table, facts):
    lm, arch = table("p6c_landmark"), table("p6c_archetypes")
    assert list(lm["archetype"]) == NAMES and list(lm["songs"]) == list(arch["songs"])
    assert int(lm["at_risk_day_30"].sum()) == facts["n_alive"] == 175 and np.allclose(lm["share_at_risk"], lm["at_risk_day_30"] / lm["songs"])
    assert (lm["at_risk_day_30"] <= lm["songs"]).all() and (lm["exits_seen_after"] <= lm["at_risk_day_30"]).all()
    assert facts["min_at_risk"] == 30 and list(lm["enough_to_estimate"]) == [False, False, False, True, True]
    assert ((lm["at_risk_day_30"] >= facts["min_at_risk"]) == lm["enough_to_estimate"]).all(), "an archetype is estimated when at least 30 of its songs are still there on day 30"
    values = [column for column in lm.columns if column.startswith(("still_going", "median"))]
    assert lm.loc[~lm["enough_to_estimate"], values].isna().all().all() and lm.loc[lm["enough_to_estimate"], values].notna().all().all()
    assert facts["share_alive"] == pytest.approx(facts["n_alive"] / facts["n_kept"], abs=1e-12)
    for key, row in zip(KEYS, lm.itertuples()):
        assert facts[f"alive_{key}"] == row.at_risk_day_30 and facts[f"exits_after_{key}"] == row.exits_seen_after
        assert facts[f"alive_share_{key}"] == pytest.approx(float(row.share_at_risk), abs=1e-12)
    assert facts["alive_lasting"] == facts["n_lasting"], "every lasting hit is still in its first stay on day 30"


@ALL_TABLES
def test_the_lasting_hits_keep_a_longer_survival_than_the_slow_climbers_and_the_band_says_so(table, facts):
    lm = table("p6c_landmark").set_index("archetype")
    for key, name in (("lasting", "Lasting hits"), ("slow", "Slow climbers")):
        row = lm.loc[name]
        for fact, column in ((f"s30_{key}", "still_going_30"), (f"s30_{key}_low", "still_going_30_low"), (f"s30_{key}_high", "still_going_30_high"),
                             (f"s90_{key}", "still_going_90"), (f"s90_{key}_low", "still_going_90_low"), (f"s90_{key}_high", "still_going_90_high"),
                             (f"median_{key}", "median_further_days"), (f"median_{key}_low", "median_low"), (f"median_{key}_high", "median_high")):
            assert facts[fact] == pytest.approx(float(row[column]), abs=1e-12), fact
        assert row["still_going_30_low"] <= row["still_going_30"] <= row["still_going_30_high"] and row["still_going_90_low"] <= row["still_going_90"] <= row["still_going_90_high"]
        assert row["still_going_90"] < row["still_going_30"] < 1, "survival only falls"
        assert row["median_low"] <= row["median_further_days"] <= row["median_high"]
    lasting, slow = lm.loc["Lasting hits"], lm.loc["Slow climbers"]
    assert facts["diff30"] == pytest.approx(float(lasting["still_going_30"] - slow["still_going_30"]), abs=1e-12)
    assert facts["diff90"] == pytest.approx(float(lasting["still_going_90"] - slow["still_going_90"]), abs=1e-12)
    assert facts["diff30_low"] > 0.1 and 0 < facts["diff90_low"] < 0.02, "the 30-day gap is clear; at 90 days the lower end of the band is only just above zero"
    assert facts["diff30_low"] <= facts["diff30"] <= facts["diff30_high"] and facts["diff90_low"] <= facts["diff90"] <= facts["diff90_high"]
    assert lasting["median_further_days"] > 3 * slow["median_further_days"] and lasting["median_low"] > slow["median_high"] + 20
    assert facts["logrank_p_naive"] < facts["logrank_p_artist"] < 0.05, "the curves differ with artists drawn again, and less sharply than if songs were independent"


@ALL_TABLES
def test_the_risk_thirds_table_adds_up_and_the_archetypes_are_ordered_by_it(table, facts):
    thirds, arch, songs = table("p6c_risk_thirds"), table("p6c_archetypes"), table("p6c_song_archetypes")
    assert list(thirds["archetype"]) == NAMES and list(thirds["songs"]) == list(arch["songs"])
    assert np.allclose(thirds[["lowest risk", "middle", "highest risk"]].sum(axis=1), 1.0)
    assert ((thirds["highest_low"] <= thirds["highest risk"]) & (thirds["highest risk"] <= thirds["highest_high"])).all()
    assert thirds["highest risk"].is_monotonic_decreasing and thirds["lowest risk"].is_monotonic_increasing, "from brief visitors to lasting hits the share in the highest-risk third falls at every step"
    assert thirds["highest risk"].iloc[0] > 0.4 > 0.15 > thirds["highest risk"].iloc[-1]
    shares = pd.crosstab(songs["archetype"], songs["risk_third_arrival"], normalize="index").reindex(NAMES)
    for third in ("lowest risk", "middle", "highest risk"):
        assert np.allclose(shares[third], thirds[third], atol=1e-9), f"the {third} column is not what the song table gives"
    by_name = thirds.set_index("archetype")
    for key, name in zip(KEYS, NAMES):
        assert facts[f"third_highest_{key}"] == pytest.approx(float(by_name.loc[name, "highest risk"]), abs=1e-12)
        assert facts[f"third_lowest_{key}"] == pytest.approx(float(by_name.loc[name, "lowest risk"]), abs=1e-12)
        assert facts[f"third_middle_{key}"] == pytest.approx(float(by_name.loc[name, "middle"]), abs=1e-12)
        assert facts[f"third_highest_{key}_low"] == pytest.approx(float(by_name.loc[name, "highest_low"]), abs=1e-12)
        assert facts[f"third_highest_{key}_high"] == pytest.approx(float(by_name.loc[name, "highest_high"]), abs=1e-12)
    persistent = songs["archetype"].isin(["Slow climbers", "Lasting hits"])
    by_third = persistent.groupby(songs["risk_third_arrival"]).mean()
    assert facts["persistent_in_lowest"] == pytest.approx(float(by_third["lowest risk"]), abs=1e-12) and facts["persistent_in_middle"] == pytest.approx(float(by_third["middle"]), abs=1e-12)
    assert facts["persistent_in_highest"] == pytest.approx(float(by_third["highest risk"]), abs=1e-12)
    assert facts["persistent_gap"] == pytest.approx(float(by_third["lowest risk"] - by_third["highest risk"]), abs=1e-12)
    assert 0 < facts["persistent_gap_low"] <= facts["persistent_gap"] <= facts["persistent_gap_high"], "the band of the gap does not include zero"
    assert facts["share_persistent"] == pytest.approx(float(persistent.mean()), abs=1e-12) and facts["not_persistent_in_lowest"] == pytest.approx(1 - float(by_third["lowest risk"]), abs=1e-12)
    assert by_third["lowest risk"] > by_third["middle"] > by_third["highest risk"]


# ================================================================================ the claims of Section 7
ATTRIBUTES = {"Arrives in the top 10": ("arrives_top10", 5), "Arrives in places 41 to 50": ("arrives_41_50", 5), "Arrives alone (no album drop)": ("arrives_alone", 3),
              "Arrives with an album drop of 5 or more songs": ("album_drop_5", 3), "Is a single": ("single", 3), "Is explicit": ("explicit", 0),
              "Is a collaboration": ("collab", 0), "Artist already on the chart the day before": ("artist_before", 1)}


def _attribute(table, attribute):
    """The rows of one attribute, indexed by 'All songs' and the five archetypes."""
    part = table("p6c_attributes")
    return part[part["attribute"] == attribute].set_index("archetype")


def _clear_above(part, name):
    return bool(part.loc[name, "low"] > part.loc["All songs", "share"])


def _clear_below(part, name):
    return bool(part.loc[name, "high"] < part.loc["All songs", "share"])


@ALL_TABLES
def test_the_attribute_table_counts_the_clear_ties_the_way_the_facts_do(table, facts):
    attr = table("p6c_attributes")
    assert list(attr["attribute"].unique()) == list(ATTRIBUTES)
    for attribute, (key, expected_clear) in ATTRIBUTES.items():
        part = _attribute(table, attribute)
        assert list(part.index) == ["All songs", *NAMES], attribute
        assert ((part["low"] <= part["share"]) & (part["share"] <= part["high"])).all() and part[["share", "low", "high"]].stack().between(0, 1).all()
        overall = float(part.loc["All songs", "share"])
        clear = [name for name in NAMES if not (part.loc[name, "low"] <= overall <= part.loc[name, "high"])]
        assert len(clear) == facts[f"attr_{key}_n_clear"] == expected_clear, f"{attribute}: the archetypes whose band stays clear of the share among all songs are {clear}"
        assert facts[f"attr_{key}_all"] == pytest.approx(overall, abs=1e-12)
        for name, short in zip(NAMES, KEYS):
            assert facts[f"attr_{key}_{short}"] == pytest.approx(float(part.loc[name, "share"]), abs=1e-12), (attribute, name)
            assert facts[f"attr_{key}_{short}_low"] == pytest.approx(float(part.loc[name, "low"]), abs=1e-12)
            assert facts[f"attr_{key}_{short}_high"] == pytest.approx(float(part.loc[name, "high"]), abs=1e-12)


@ALL_TABLES
def test_the_ties_the_closing_names_hold_in_the_attribute_table(table):
    top10 = _attribute(table, "Arrives in the top 10")
    for name in ("Lasting hits", "Month-long fades", "Quick fades"):
        assert _clear_above(top10, name), f"{name} arrive high"
    for name in ("Brief visitors", "Slow climbers"):
        assert _clear_below(top10, name), f"{name} do not arrive in the top 10"
    low = _attribute(table, "Arrives in places 41 to 50")
    for name in ("Slow climbers", "Brief visitors"):
        assert _clear_above(low, name), f"{name} arrive low"
    for name in ("Lasting hits", "Month-long fades", "Quick fades"):
        assert _clear_below(low, name)
    drop = _attribute(table, "Arrives with an album drop of 5 or more songs")
    assert _clear_above(drop, "Quick fades") and _clear_above(drop, "Month-long fades") and _clear_below(drop, "Slow climbers"), "the fades come in album drops and the slow climbers do not"
    assert drop.loc["Quick fades", "share"] > 0.7 and drop.loc["Month-long fades", "share"] > 0.65 and drop.loc["Slow climbers", "share"] < 0.05
    alone = _attribute(table, "Arrives alone (no album drop)")
    assert _clear_above(alone, "Slow climbers") and alone.loc["Slow climbers", "share"] > 0.85
    assert _clear_below(alone, "Quick fades") and _clear_below(alone, "Month-long fades")
    before = _attribute(table, "Artist already on the chart the day before")
    assert _clear_below(before, "Slow climbers") and not any(_clear_above(before, name) or _clear_below(before, name) for name in NAMES if name != "Slow climbers"), \
        "only the slow climbers differ from the other songs in whether the artist was already on the chart"
    assert before.loc["Slow climbers", "share"] < 0.25 and before.loc["All songs", "share"] > 0.5


@ALL_TABLES
def test_being_explicit_or_a_collaboration_tells_nothing_and_the_single_is_a_hint_at_most(table):
    for attribute in ("Is explicit", "Is a collaboration"):
        part = _attribute(table, attribute)
        assert not any(_clear_above(part, name) or _clear_below(part, name) for name in NAMES), f"{attribute} now differs for an archetype, the text says it does not"
    single = _attribute(table, "Is a single")
    assert (single.loc[NAMES, "share"] - single.loc["All songs", "share"]).abs().max() < 0.2, "the single changes the share by less than 20 points in every archetype"
    assert _clear_above(single, "Slow climbers") and _clear_above(single, "Lasting hits")


# ================================================================================ the claims of Section 8
@ALL_TABLES
def test_the_artist_tables_add_up_and_the_segments_are_what_the_text_says(table, facts):
    artists, seg = table("p6c_artists"), table("p6c_artist_segments")
    assert not artists["artist"].duplicated().any() and len(artists) == facts["n_artists_all"] == 243
    assert list(seg["segment"]) == SEGMENTS
    counts = artists["segment"].value_counts().reindex(SEGMENTS)
    assert list(counts) == list(seg["artists"]) == [facts[f"seg_artists_{key}"] for key in SEGMENT_KEYS] == [93, 86, 32, 32]
    assert int(artists["songs"].sum()) == facts["songs"] == int(seg["songs_total"].sum()) == 944
    assert int(artists["chart_days"].sum()) == facts["rows"] == int(seg["chart_days_total"].sum()) == 27750
    assert np.allclose(seg["share_of_artists"].sum(), 1) and np.allclose(seg["share_of_chart_days"].sum(), 1)
    assert np.allclose(seg["share_of_artists"], seg["artists"] / 243) and np.allclose(seg["share_of_chart_days"], seg["chart_days_total"] / 27750)
    for column, source in (("songs_median", "songs"), ("chart_days_median", "chart_days"), ("days_per_song_median", "days_per_song"), ("mean_place_median", "mean_place"),
                           ("top10_share_median", "top10_share"), ("days_present_median", "days_present"), ("peak_songs_median", "peak_songs")):
        assert np.allclose(seg[column], artists.groupby("segment")[source].median().reindex(SEGMENTS), atol=1e-9), f"{column} is not the median of the artist table"
    assert artists.groupby("segment")["certain"].mean().reindex(SEGMENTS).tolist() == pytest.approx(list(seg["share_certain"]), abs=1e-12)
    for key, row in zip(SEGMENT_KEYS, seg.itertuples()):
        assert facts[f"seg_songs_total_{key}"] == row.songs_total and facts[f"seg_share_of_chart_days_{key}"] == pytest.approx(float(row.share_of_chart_days), abs=1e-12)
        assert facts[f"seg_share_of_artists_{key}"] == pytest.approx(float(row.share_of_artists), abs=1e-12)
        assert facts[f"seg_chart_days_median_{key}"] == pytest.approx(float(row.chart_days_median), abs=1e-12)
        assert facts[f"seg_jaccard_mean_{key}"] == pytest.approx(float(row.jaccard_mean), abs=1e-12) and facts[f"seg_heldout_recall_{key}"] == pytest.approx(float(row.heldout_recall), abs=1e-12)
        assert facts[f"seg_share_certain_{key}"] == pytest.approx(float(row.share_certain), abs=1e-12) and facts[f"seg_examples_{key}"] == row.examples
    assert facts["max_artist_chart_days"] == int(artists["chart_days"].max()) == 2015 and facts["median_artist_chart_days"] == float(artists["chart_days"].median())
    assert facts["median_artist_songs"] == float(artists["songs"].median()) and facts["share_one_song"] == pytest.approx(float((artists["songs"] == 1).mean()), abs=1e-12)
    assert facts["share_chart_days_top10_artists"] == pytest.approx(float(artists["chart_days"].nlargest(10).sum() / 27750), abs=1e-12)


@ALL_TABLES
def test_the_four_segments_pass_the_rule_and_the_catalogue_artists_are_the_ones_with_many_songs(table, facts):
    artists, seg = table("p6c_artists"), table("p6c_artist_segments").set_index("segment")
    assert (seg["jaccard_mean"] >= facts["floor"]).all(), "every artist segment passes the rule that chose the number of groups"
    assert float(seg["jaccard_mean"].min()) == pytest.approx(facts["art_weakest_k4"], abs=0.01)
    assert (seg["heldout_recall"] > 0.94).all() and facts["art_held_same"] > 0.97
    assert seg["share_certain"].idxmin() == "Mid-chart stayers", "the mid-chart stayers are the least certain artist segment"
    catalogue = artists[artists["segment"] == "Catalogue artists"]
    others = artists[artists["segment"] != "Catalogue artists"]
    assert int(catalogue["songs"].min()) == facts["catalogue_min_songs"] == 6 and int(others["songs"].max()) == facts["others_max_songs"] == 5, "the catalogue artists are exactly the artists with six songs or more"
    assert seg.loc["Catalogue artists", "share_of_artists"] < 0.14 and seg.loc["Catalogue artists", "share_of_chart_days"] > 0.59
    assert seg["chart_days_median"].is_monotonic_increasing and seg["top10_share_median"].idxmax() == "Top-chart hit-makers" and seg["mean_place_median"].idxmin() == "Top-chart hit-makers"
    assert seg["songs_median"].drop("Catalogue artists").max() == 1 and seg.loc["Catalogue artists", "peak_songs_median"] > 5
    assert seg.loc["Passing visitors", "share_of_chart_days"] < 0.02 and seg.loc["Passing visitors", "share_of_artists"] > 0.35
    assert artists["agreement"].between(0, 1).all() and (artists["certain"] == (artists["agreement"] >= 0.9)).all()
    assert facts["art_share_certain"] == pytest.approx(float(artists["certain"].mean()), abs=1e-12) and facts["art_n_border"] == int((artists["agreement"] < 0.6).sum()) == 8
    assert facts["art_boot_ari_mean"] == pytest.approx(0.865, abs=0.005) and facts["art_boot_ari_p10"] < facts["art_boot_ari_mean"]


@_needs(FACTS_FILE)
def test_the_artist_map_is_one_band_and_the_first_two_components_hold_what_the_text_says(facts):
    assert facts["pc1_share"] > facts["pc2_share"] > facts["pc3_share"] > 0
    assert facts["pc12_share"] == pytest.approx(facts["pc1_share"] + facts["pc2_share"], abs=1e-12) and 0.75 < facts["pc12_share"] < 0.85
    assert facts["pc1_share"] + facts["pc2_share"] + facts["pc3_share"] < 1
    loadings = [facts[f"load1_{name}"] for name in ("log_songs", "log_days_present", "log_days_per_song", "log_peak_songs", "top10_share")]
    assert all(value > 0 for value in loadings) and facts["load1_mean_place"] < 0, "the first component runs from small and low on the chart to big and high on it"


# ================================================================================ the claims of Section 9
@ALL_TABLES
def test_the_archetype_mix_of_every_segment_adds_up_and_says_what_the_text_says(table, facts):
    mix, songs, arch = table("p6c_archetype_by_segment"), table("p6c_song_archetypes"), table("p6c_archetypes")
    assert list(mix["segment"].unique()) == SEGMENTS and mix.groupby("segment", sort=False)["archetype"].apply(list).tolist() == [NAMES] * 4
    assert np.allclose(mix.groupby("segment")["share"].sum(), 1.0) and np.allclose(mix["share"], mix["songs"] / mix["songs_in_segment"])
    assert ((mix["low"] <= mix["share"] + 1e-12) & (mix["share"] <= mix["high"] + 1e-12)).all()
    counts = pd.crosstab(songs["artist_segment"], songs["archetype"]).reindex(index=SEGMENTS, columns=NAMES)
    assert mix["songs"].tolist() == counts.to_numpy().ravel().tolist(), "the table is the cross-table of the two groupings in the song file"
    assert mix.groupby("segment")["songs_in_segment"].first().reindex(SEGMENTS).tolist() == counts.sum(axis=1).tolist()
    assert int(mix.groupby("segment")["songs_in_segment"].first().sum()) == facts["n_kept"]
    assert mix.groupby("archetype")["songs"].sum().reindex(NAMES).tolist() == arch["songs"].tolist()
    for seg_key, seg_name in zip(SEGMENT_KEYS, SEGMENTS):
        in_segment = int(counts.loc[seg_name].sum())
        assert facts[f"tog_n_{seg_key}"] == in_segment and facts[f"tog_share_songs_{seg_key}"] == pytest.approx(in_segment / 832, abs=1e-12)
        assert facts[f"tog_artists_{seg_key}"] == songs.loc[songs["artist_segment"] == seg_name, "primary_artist"].nunique()
        for key, name in zip(KEYS, NAMES):
            row = mix[(mix["segment"] == seg_name) & (mix["archetype"] == name)].iloc[0]
            assert facts[f"tog_{seg_key}_{key}"] == pytest.approx(float(row["share"]), abs=1e-12), (seg_key, key)
            assert facts[f"tog_{seg_key}_{key}_low"] == pytest.approx(float(row["low"]), abs=1e-12) and facts[f"tog_{seg_key}_{key}_high"] == pytest.approx(float(row["high"]), abs=1e-12)
            assert facts[f"tog_from_{seg_key}_{key}"] == pytest.approx(float(counts.loc[seg_name, name] / counts[name].sum()), abs=1e-12), (seg_key, key)


@ALL_TABLES
def test_the_mix_by_segment_has_the_pattern_the_text_describes(table, facts):
    mix = table("p6c_archetype_by_segment").set_index(["segment", "archetype"])["share"].unstack().reindex(index=SEGMENTS, columns=NAMES)
    assert mix.loc["Passing visitors", "Brief visitors"] > 0.9 and mix.loc["Passing visitors", ["Month-long fades", "Slow climbers", "Lasting hits"]].sum() == 0
    assert mix["Lasting hits"].idxmax() == "Top-chart hit-makers" and mix.loc["Top-chart hit-makers", "Lasting hits"] > 0.3
    assert mix["Slow climbers"].idxmax() == "Mid-chart stayers" and facts["tog_from_stayers_slow"] > 0.5, "more than half of all slow climbers are songs of mid-chart stayers"
    assert mix["Quick fades"].idxmax() == "Catalogue artists" and mix["Month-long fades"].idxmax() == "Catalogue artists"
    for key in ("quick", "month_long"):
        assert facts[f"tog_from_catalogue_{key}"] > facts["tog_share_songs_catalogue"] + 0.15, f"the catalogue artists have clearly more than their share of the {key} fades"
    assert 0.6 < facts["tog_share_songs_catalogue"] < 0.7, "the catalogue artists own about two thirds of the songs, so most fades coming from them is partly just their size"
    assert facts["tog_top_artist_quick"] == int(table("p6c_song_archetypes").query("primary_artist == 'Taylor Swift' and archetype == 'Quick fades'").shape[0]) == 43
    assert facts["top_artist_name_quick"] == "Taylor Swift"


@ALL_TABLES
def test_the_two_groupings_are_different_cuts_and_the_songs_of_one_artist_are_only_a_little_alike(table, facts):
    songs = table("p6c_song_archetypes")
    ari = adjusted_rand(songs["artist_segment"], songs["archetype"])
    assert facts["tog_ari"] == pytest.approx(ari, abs=1e-12) and abs(ari) < 0.05, "the segment of an artist says almost nothing about the archetype a song falls in"
    for minimum in (2, 5):
        values = []
        for _, group in songs.groupby("primary_artist"):
            n = len(group)
            if n >= minimum:
                counts = group["archetype"].value_counts().to_numpy()
                values.append((counts * (counts - 1)).sum() / (n * (n - 1)))      # pairs of songs in the same archetype / all pairs of songs of the artist
        assert facts[f"same_n_artists_{minimum}"] == len(values) and facts[f"same_obs_{minimum}"] == pytest.approx(float(np.mean(values)), abs=1e-12)
        low, mean, high, top = (facts[f"same_null_{part}_{minimum}"] for part in ("low", "mean", "high", "max"))
        observed = facts[f"same_obs_{minimum}"]
        assert low < mean < high < top and mean < observed < top, "the real share is above the shuffled average and below the highest of the 1,000 shuffles"
        assert observed - mean < 0.1, "only a little more alike than chance gives"
    assert facts["same_n_artists_2"] == 90 and facts["same_n_artists_5"] == 32 and facts["same_obs_2"] <= facts["same_null_high_2"], "with two or more songs the real share is inside the shuffled band"


@ALL_TABLES
def test_the_ten_biggest_artists_are_not_one_type_of_song(table, facts):
    songs = table("p6c_song_archetypes")
    counts = songs["primary_artist"].value_counts()
    assert counts.iloc[9] > counts.iloc[10], "the ten biggest artists are not a coin toss between two artists with the same number of songs"
    top = counts.head(10).index
    mix = pd.crosstab(songs["primary_artist"], songs["archetype"]).reindex(columns=NAMES).loc[top]
    assert facts["n_top_ten_artists"] == 10 and facts["n_top_ten_songs"] == int(counts.head(10).sum()) and facts["share_top_ten_songs"] == pytest.approx(float(counts.head(10).sum() / 832), abs=1e-12)
    assert facts["top_ten_min_archetypes"] == int((mix > 0).sum(axis=1).min()) == 3, "every one of the ten has songs in at least three archetypes"
    assert facts["top_ten_max_one_archetype"] == pytest.approx(float((mix.max(axis=1) / mix.sum(axis=1)).max()), abs=1e-12) and facts["top_ten_max_one_archetype"] < 0.85
    assert facts["top_ten_with_lasting_hit"] == int((mix["Lasting hits"] > 0).sum()) == 9
    assert facts["top_artist_all_name"] == facts["top_artist_name"] == counts.index[0] == "Taylor Swift" and facts["top_artist_songs"] == int(counts.iloc[0]) == 92
    assert facts["top_artist_all_songs"] == 96 and facts["top_artist_all_songs"] >= facts["top_artist_songs"]
    assert facts["top10_highest_name"] == "Lasting hits" and facts["drop5_highest_name"] == "Quick fades"


# ================================================================================ the closing text and the numbers
def _pct(value):
    return f"{100 * value:.0f}%"


@EVERYTHING
def test_the_closing_quotes_the_saved_numbers(notebook, facts):
    closing = notebook.cells[-1].source
    f = facts
    wanted = [f"{f['n_kept']} of the {f['n_entrants']} songs", f"{f['k']} archetypes", f"{f['n_brief']} *brief visitors* ({_pct(f['share_brief'])} of the songs, on the chart on {f['days_brief']:.1f} of the {f['horizon']} days",
              f"{f['n_quick']} *quick fades* ({f['days_quick']:.1f} days)", f"{f['n_month_long']} *month-long fades* ({f['days_month_long']:.1f} days, and {_pct(f['day30_month_long'])} of them still on the chart on day 30)",
              f"{f['n_slow']} *slow climbers* (arriving low, at place {f['debut_slow']:.0f} on average, and on the chart on {f['days_slow']:.1f} of the {f['horizon']} days)",
              f"{f['n_lasting']} *lasting hits* (arriving high, at place {f['debut_lasting']:.0f} on average, and on the chart on {f['days_lasting']:.1f} of the {f['horizon']} days)",
              f"Whole artists were drawn again {f['n_boot']:,} times", f"{f['boot_ari_mean']:.2f} (10th percentile {f['boot_ari_p10']:.2f})",
              f"{_pct(f['share_agree_90'])} of the songs stayed in their own archetype in at least 90% of the draws and {f['n_agree_below_60']} songs ({100 * f['share_agree_below_60']:.1f}%) in fewer than 60%",
              f"{_pct(f['held_same_mean'])} of the time", f"Jaccard {f['firmest_jaccard']:.2f}", f"{f['weakest_jaccard']:.2f}; they dissolved in {_pct(f['share_of_draws_dissolved_month_long'])} of the draws",
              f"{f['jaccard_mean_slow']:.2f}; {_pct(f['share_of_draws_dissolved_slow'])}", f"{f['k']} groups are firm and {f['k_next']} are not",
              f"leaving out the ten biggest gives an adjusted Rand index of {f['stress_ari_2']:.2f}",
              f"capping every artist at {f['cap']} songs gives {f['stress_ari_3']:.2f} on average, {f['cap_ari_low']:.2f} to {f['cap_ari_high']:.2f} over {f['cap_repeats']} caps",
              f"to a 45-day month ({f['stress_ari_8']:.2f})", f"({f['stress_ari_9']:.2f} with four, {f['stress_ari_10']:.2f} with six)",
              f"keeping only whether the song was on the chart gives {f['stress_ari_5']:.2f} and the logarithm of the score {f['stress_ari_6']:.2f}",
              f"(Ward, {f['stress_ari_4']:.2f})", f"of the number of random starts ({f['stress_ari_1']:.2f})", f"({f['stress_ari_7']:.2f} with 21 days)",
              f"Only {f['n_alive']} of the {f['n_kept']} songs ({_pct(f['share_alive'])})", f"all {f['alive_lasting']} lasting hits and {f['alive_slow']} of the {f['n_slow']} slow climbers",
              f"a survival of {_pct(f['s30_lasting'])} after another 30 days and {_pct(f['s90_lasting'])} after 90", f"a median further stay of {f['median_lasting']:.0f} days ({f['median_lasting_low']:.0f} to {f['median_lasting_high']:.0f})",
              f"the slow climbers {_pct(f['s30_slow'])} and {_pct(f['s90_slow'])}, with a median of {f['median_slow']:.0f} days ({f['median_slow_low']:.0f} to {f['median_slow_high']:.0f})",
              f"log-rank p = {f['logrank_p_artist']:.3f}", f"{f['logrank_p_naive']:.3f} if songs",
              f"{_pct(f['third_highest_brief'])} of the brief visitors but {_pct(f['third_highest_lasting'])} of the lasting hits", f"from {_pct(f['persistent_in_lowest'])} in the lowest-risk third to {_pct(f['persistent_in_highest'])} in the highest",
              f"band {100 * f['persistent_gap_low']:.0f} to {100 * f['persistent_gap_high']:.0f}", f"({_pct(f['attr_arrives_top10_lasting'])} and {_pct(f['attr_arrives_top10_month_long'])} in the top 10, against {_pct(f['attr_arrives_top10_all'])} of all songs)",
              f"({_pct(f['attr_album_drop_5_quick'])} and {_pct(f['attr_album_drop_5_month_long'])})", f"alone ({_pct(f['attr_arrives_alone_slow'])})",
              f"The {f['n_artists_all']} artists", f"carry {_pct(f['pc12_share'])} of the variation", f"{f['seg_artists_visitors']} passing visitors, {f['seg_artists_stayers']} mid-chart stayers, {f['seg_artists_hitmakers']} top-chart hit-makers and {f['seg_artists_catalogue']} catalogue artists",
              f"adjusted Rand index {f['art_boot_ari_mean']:.2f} over the artist draws", f"{_pct(f['art_held_same'])} of held-out artists",
              f"{_pct(f['seg_share_of_artists_catalogue'])} of the artists and have {_pct(f['seg_share_of_chart_days_catalogue'])} of all chart days",
              f"{_pct(f['tog_from_catalogue_quick'])} of the quick fades", f"{_pct(f['tog_hitmakers_lasting'])} of their songs", f"adjusted Rand index {f['tog_ari']:.3f}",
              f"{_pct(f['same_obs_2'])} of pairs for artists with two or more songs, against {_pct(f['same_null_mean_2'])} when the archetypes are shuffled",
              f"between {f['first_date']} and {f['last_date']}", f"{f['n_dropped']} songs that arrived in the last {f['horizon_less_one']} snapshot days", f"{f['songs']} songs in the file, {f['n_entrants']} of them arrived later"]
    for item in wanted:
        assert item in closing, f"the closing text does not hold {item!r}"


@_needs(FACTS_FILE)
def test_the_text_of_the_sections_quotes_the_decisive_numbers_of_the_facts(notebook, facts):
    text = _all_text(notebook)
    for item in (f"{facts['n_kept']}", f"{facts['n_artists_all']}", f"{facts['n_dropped']}", f"{facts['boot_ari_mean']:.2f}", f"{facts['stress_ari_5']:.2f}", f"{facts['weakest_next']:.2f}",
                 f"{facts['n_kept_artists']}"):
        assert item in text, f"the text never mentions {item}"
