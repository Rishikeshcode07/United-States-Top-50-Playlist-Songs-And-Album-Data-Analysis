"""
tests/test_notebook_06b.py  -  checks of the Phase 6B notebook (notebooks/06b_survival_analysis.ipynb) and of the thirteen files it writes.

WHAT THIS FILE IS
    Checks of what the notebook IS and of what its saved files SAY, not of what the estimators compute (the stays, the survival and hazard
    estimators and the Cox model are tested in test_survival_*.py). The notebook must be valid, delivered without outputs, every code cell
    must compile and have its five-part plain-English explanation directly above it, every chart must have a "What to notice" note directly
    below it, no sentence may contain an unfilled gap or a double negative such as "fell by about -0.5", the 12 charts must be numbered 1 to
    12 without a hole, and every number that the code looks up in FACTS must exist in the facts file the notebook saved.
    The later tests read the files the notebook writes. They check that the tables add up to the facts (the song table to the number of
    songs, exits and chart days; the hazard table to the same totals), that the curves start at 100% and never rise, and then that the
    claims the text makes about them hold in the tables: debut place, songs arriving together and release type differ after the adjustment
    and explicit content and collaboration do not; exactly two of the ten inputs are clear in the hazard model and five act differently by
    age; the chance that a chart day is the last one steps up with the place on the chart; the debut day (and not timing or the artist's
    record) is the step that improves the ranking of stays, and the first three chart days improve it a lot more; the three risk groups are
    ordered and the highest-risk group is clearly apart from the other two; and the nine stress runs say what the closing says.

WHY WE NEED IT
    The explanations, the notes and the closing statements are the point of this notebook, and the saved tables are what the dashboard, the
    Power BI report and the paper will read. A cell that lost its explanation during an edit, a sentence that still holds a "{{gap}}", a
    table that no longer adds up to the facts, or a closing text that disagrees with the saved numbers would otherwise be found by the
    reader and not by us.

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
NOTEBOOK = ROOT / "notebooks" / "06b_survival_analysis.ipynb"
REPORTS = ROOT / "reports"
FIGURE_DIR = REPORTS / "figures"
FACTS_FILE = REPORTS / "p6b_facts.json"

# the twelve tables of the notebook and the shape each has on the real data (rows, columns)
TABLES = {"p6b_stays": (894, 25), "p6b_survival_curves": (1815, 6), "p6b_hazard_by_age": (9, 9), "p6b_further_stay": (6, 5),
          "p6b_group_comparison": (14, 20), "p6b_debut_alone": (5, 17), "p6b_hazard_ratios": (14, 13), "p6b_effects_by_age": (42, 12),
          "p6b_danger_zone": (10, 21), "p6b_ladder": (7, 13), "p6b_risk_thirds": (6, 14), "p6b_stress_tests": (54, 14)}
TABLE_FILES = {name: REPORTS / f"{name}.csv" for name in TABLES}
ALL_FILES = [FACTS_FILE, *TABLE_FILES.values()]

FIVE_PARTS = ["**What this code is.**", "**Why we need it.**", "**What will happen.**", "**Expected outcome.**", "**Motive.**"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U00002B00-\U00002BFF\U0000FE0F]")

pytestmark = pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebooks/06b_survival_analysis.ipynb is not there")


def _needs(*files):
    missing = [file.name for file in files if not file.exists()]
    return pytest.mark.skipif(bool(missing), reason=f"run the notebook once to create {', '.join(missing)}")


ALL_TABLES = _needs(*TABLE_FILES.values())


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


@pytest.fixture(scope="module")
def facts():
    return json.loads(FACTS_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def table():
    """table('p6b_stays') reads one of the twelve saved tables."""
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


CHART_NAME = re.compile(r"""["'](p6b_\d\d_[a-z0-9_]+)["']""")             # a chart is named by a string such as "p6b_10_ladder", in save_fig(...)


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
    assert len(notebook.cells) == 99 and len(_code_cells(notebook)) == 31
    assert notebook.cells[0].cell_type == "markdown" and notebook.cells[0].source.startswith("# Phase 6B")


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


def test_the_survival_methods_are_the_project_own_and_no_survival_library_is_imported(notebook):
    for index, cell in _code_cells(notebook):
        assert not re.search(r"\b(lifelines|sksurv|statsmodels)\b", cell.source), f"cell {index} imports a survival or statistics library"
    assert any("src.survival" in cell.source for _, cell in _code_cells(notebook))


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
    assert len(charts) == 12
    for index, _ in charts:
        below = notebook.cells[index + 1]
        assert below.cell_type == "markdown" and below.source.startswith("**What to notice**"), f"chart in cell {index} has no note"


def test_the_12_charts_are_numbered_one_to_twelve_exactly_once(notebook):
    names = _chart_names(notebook)
    assert [int(name[4:6]) for name in names] == list(range(1, 13))
    assert len(set(names)) == 12


def test_every_chart_cell_and_its_explanation_carry_the_number_of_their_chart(notebook):
    for index, cell in _chart_cells(notebook):
        number = int(CHART_NAME.search(cell.source).group(1)[4:6])
        assert cell.source.startswith(f"# Figure {number:02d} "), f"cell {index} does not start with its figure number"
        what_it_is = notebook.cells[index - 1].source.split("**Why we need it.**")[0]
        assert re.search(rf"\bFigure {number}\b", what_it_is), f"the first part of the explanation above cell {index} does not name Figure {number}"


def test_the_ten_sections_follow_each_other(notebook):
    numbers = [int(match.group(1)) for _, text in _markdown_text(notebook) for match in [re.match(r"## Section (\d+) - ", text)] if match]
    assert numbers == list(range(1, 11))


def test_every_figure_and_section_the_text_refers_to_exists(notebook):
    text = _all_text(notebook)
    figures = {int(number) for number in re.findall(r"\bFigures? (\d+)", text)}
    sections = {int(number) for number in re.findall(r"\bSections? (\d+)", text)}
    assert figures and figures <= set(range(1, 13)), f"a reference to a figure that does not exist: {sorted(figures - set(range(1, 13)))}"
    assert sections and sections <= set(range(1, 11)), f"a reference to a section that does not exist: {sorted(sections - set(range(1, 11)))}"


def test_the_introduction_names_every_file_and_the_number_of_charts_and_files(notebook):
    intro = notebook.cells[0].source
    for name in TABLES:
        assert f"`{name}.csv`" in intro, f"the introduction does not name {name}.csv"
    assert "`p6b_facts.json`" in intro and "12 charts" in intro and "thirteen files" in intro
    assert len(TABLES) + 1 == 13
    rows = [line for line in intro.splitlines() if re.match(r"\| \d+ \|", line)]
    assert len(rows) == 10, "the table of sections in the introduction must list the ten sections"


def test_the_notebook_ends_with_the_closing_summary_and_the_hand_over_to_the_next_phases(notebook):
    last = notebook.cells[-1]
    assert last.cell_type == "markdown" and last.source.startswith("### What Phase 6B adds up to")
    for phase in ("Phase 6C", "Phase 7", "Phase 8"):
        assert phase in last.source, f"the closing text does not mention {phase}"


def test_the_closing_keeps_the_caveats_that_the_brief_requires(notebook):
    closing = notebook.cells[-1].source.lower()
    assert "association" in closing, "the closing must say that every result is an association"
    assert "not a promise" in closing, "the closing must say that the C-index is not a promise about future songs"
    assert "were already on the chart on the first day" in closing, "the closing must disclose the 50 songs left out"
    assert "treat artists as independent" in closing, "the closing must disclose that songs of different artists can leave on the same day"
    assert "was not tried" in closing, "the closing must disclose that a model with age-varying ratios was not tried"


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


def test_no_percent_points_and_no_hand_typed_count_of_files_is_out_of_date(notebook):
    text = _all_text(notebook)
    assert not re.search(r"\d%\s+points", text), "write 'percentage points', not '% points'"
    assert "thirteen files" in notebook.cells[0].source and "twelve tables" in notebook.cells[96].source


def test_no_emoji_anywhere_and_no_cloud_storage_in_the_code_or_the_text(notebook):
    for index, cell in enumerate(notebook.cells):
        assert not EMOJI.search(cell.source), f"cell {index} contains an emoji"
        assert not re.search(r"(?i:boto3|s3://|amazon s3|aws s3)|\bS3\b", cell.source), f"cell {index} mentions S3"        # a variable called s3 (the share on day 3) is not the storage service


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
    assert len(facts) > 1300
    assert all(isinstance(value, (bool, int, float, str)) for value in facts.values())
    assert all(np.isfinite(value) for value in facts.values() if isinstance(value, float) and not isinstance(value, bool))
    assert text.endswith("}\n")


@_needs(FACTS_FILE)
def test_the_counts_in_the_facts_add_up(facts):
    assert facts["n_songs"] == facts["n_exits"] + facts["n_censored"]
    assert facts["day_one_songs"] + facts["n_songs"] == facts["songs"], "the 50 songs of the first day and the 894 of the stays are all the songs"
    assert facts["open_late_arrivals"] + facts["open_long_stays"] - facts["open_both"] == facts["n_censored"], "every open stay is explained by one of the two reasons"
    assert facts["n_windows"] == 9 and facts["ph_n_windows"] == 3 and facts["n_boot"] == 1000


# ================================================================================ the saved files
@ALL_TABLES
def test_the_twelve_tables_have_the_shapes_the_notebook_reports(table):
    for name, shape in TABLES.items():
        assert table(name).shape == shape, f"{name} has the shape {table(name).shape}, expected {shape}"


@_needs(*TABLE_FILES.values(), FACTS_FILE)
def test_the_twelve_charts_the_code_names_are_in_the_figures_folder(notebook):
    names = _chart_names(notebook)
    assert len(names) == 12
    for name in names:
        path = FIGURE_DIR / f"{name}.png"
        assert path.exists() and path.stat().st_size > 20_000, f"{path.name} is missing or empty"


@ALL_TABLES
def test_the_song_table_has_one_row_per_song_and_agrees_with_the_facts(table, facts):
    stays = table("p6b_stays")
    assert not stays["track_id"].duplicated().any() and len(stays) == facts["n_songs"]
    assert stays["primary_artist"].nunique() == facts["n_artists"]
    assert int(stays["exit_seen"].sum()) == facts["n_exits"] and int((~stays["exit_seen"]).sum()) == facts["n_censored"]
    assert int(stays["stay"].sum()) == facts["n_days"], "the chart days of all stays are the chart days of the notebook"
    assert (stays["stay"] >= 1).all() and (stays["first_run_days"] >= 1).all() and (stays["first_run_days"] <= stays["stay"]).all()
    assert int((stays["stay"] > 30).sum()) == facts["n_gt30"] and int((stays["stay"] == 1).sum()) == facts["n_stay_one"]
    assert int(stays["stay"].max()) == facts["stay_max"]
    assert stays["risk_third_arrival"].notna().all() and stays["risk_third_arrival"].nunique() == 3
    short_run = stays["first_run_days"] < 3
    assert int(short_run.sum()) == facts["lm_excluded"]
    assert stays.loc[short_run, "risk_third_after_3_days"].isna().all() and stays.loc[~short_run, "risk_third_after_3_days"].notna().all()
    assert int((stays["short_breaks"] > 0).sum()) > 0 and set(stays["debut_band"]) == {"1 to 10", "11 to 20", "21 to 30", "31 to 40", "41 to 50"}


@ALL_TABLES
def test_the_curves_start_at_one_never_rise_and_stay_inside_their_bands(table):
    curves = table("p6b_survival_curves")
    assert set(curves["split"]) == {"All songs", "Debut place", "Songs by the artist arriving the same day", "Release type", "Explicit content", "Collaboration"}
    for (split, group), part in curves.groupby(["split", "group"]):
        part = part.sort_values("age")
        assert part["age"].iloc[0] == 0 and part["survival"].iloc[0] == pytest.approx(1.0), (split, group)
        assert (np.diff(part["survival"].to_numpy()) <= 1e-12).all(), f"the curve of {group} rises somewhere"
        assert ((part["low"] - 1e-9 <= part["survival"]) & (part["survival"] <= part["high"] + 1e-9)).all(), f"the curve of {group} leaves its band"
        assert part["survival"].between(0, 1).all()


@ALL_TABLES
def test_the_overall_curve_gives_the_shares_the_text_quotes(table, facts):
    curve = table("p6b_survival_curves")
    curve = curve[curve["split"] == "All songs"].set_index("age")
    for age in (1, 3, 7, 14, 30, 60, 90):
        assert facts[f"s{age}"] == pytest.approx(float(curve.loc[age, "survival"]), abs=1e-12), f"S({age})"
    assert 0.80 < facts["s1"] < 0.86 and 0.40 < facts["s7"] < 0.46 and 0.16 < facts["s30"] < 0.23


@ALL_TABLES
def test_the_hazard_by_age_table_adds_up_to_the_facts(table, facts):
    hazard = table("p6b_hazard_by_age")
    assert int(hazard["exits"].sum()) == facts["n_exits"] and int(hazard["chart_days"].sum()) == facts["n_days"]
    assert np.allclose(hazard["hazard"], hazard["exits"] / hazard["chart_days"])
    assert ((hazard["low"] <= hazard["hazard"] + 1e-12) & (hazard["hazard"] <= hazard["high"] + 1e-12)).all()
    assert hazard["hazard"].is_monotonic_decreasing, "the chance that a chart day is the last one falls with the age of the stay"
    assert hazard["hazard"].iloc[0] > 10 * hazard["hazard"].iloc[-1], "the first chart day is many times as dangerous as the days after the 120th"


# ================================================================================ the claims of Sections 4 to 6
SPLITS = {"Debut place": 5, "Songs by the artist arriving the same day": 3, "Release type": 2, "Explicit content": 2, "Collaboration": 2}


@ALL_TABLES
def test_every_split_divides_all_the_songs_and_all_the_exits(table, facts):
    groups = table("p6b_group_comparison")
    assert set(groups["split"]) == set(SPLITS)
    for split, count in SPLITS.items():
        part = groups[groups["split"] == split]
        assert len(part) == count, split
        assert int(part["songs"].sum()) == facts["n_songs"] and int(part["exits"].sum()) == facts["n_exits"], split


@ALL_TABLES
def test_three_splits_differ_after_the_artist_aware_test_and_the_adjustment_and_two_do_not(table, facts):
    groups = table("p6b_group_comparison")
    holm = groups.groupby("split")["log_rank_p_artist_aware_holm"].first()
    raw = groups.groupby("split")["log_rank_p_artist_aware"].first()
    plain = groups.groupby("split")["log_rank_p_songs_as_independent"].first()
    assert (holm >= raw - 1e-15).all() and (holm <= 1).all(), "the Holm adjustment can only raise a p-value"
    assert (plain <= raw + 1e-12).all(), "treating songs as independent can only make the evidence look stronger"
    for split in ("Debut place", "Songs by the artist arriving the same day", "Release type"):
        assert holm[split] < 0.05, f"{split} no longer differs after the adjustment"
    for split in ("Explicit content", "Collaboration"):
        assert holm[split] > 0.05, f"{split} now differs, the text says it does not"
    assert facts["n_splits_clear"] == 3 and facts["n_splits_tested"] == 5


@ALL_TABLES
def test_the_differences_in_chart_days_have_the_signs_and_bands_the_text_describes(table):
    groups = table("p6b_group_comparison")
    rows = groups[groups["reference_group"].notna()]
    assert len(rows) == 9
    debut = rows[rows["split"] == "Debut place"]
    assert (debut["difference_high"] < 0).all(), "every debut band uses clearly fewer chart days than a top-10 debut"
    single = rows[rows["split"] == "Release type"].iloc[0]
    assert single["difference_low"] > 0, "a single uses clearly more chart days than an album track, before the other inputs are held fixed"
    for split in ("Explicit content", "Collaboration"):
        row = rows[rows["split"] == split].iloc[0]
        assert row["difference_low"] < 0 < row["difference_high"], f"{split}: the band of the difference must include zero"


@ALL_TABLES
def test_the_debut_place_table_says_who_arrives_alone_and_how_much_of_the_recovery_is_the_mix(table, facts):
    debut = table("p6b_debut_alone")
    assert int(debut["songs"].sum()) == facts["n_songs"] and (debut["alone_songs"] + debut["others_songs"] == debut["songs"]).all()
    assert np.allclose(debut["share_alone"], debut["alone_songs"] / debut["songs"])
    assert debut["share_alone"].iloc[-1] > 0.8 > 0.3 > debut["share_alone"].iloc[0], "the bottom of the chart is where lone songs arrive"
    assert (debut["alone"] > debut["others"]).all(), "in every band the songs that arrive alone stay longer"
    crude = float(debut["crude"].iloc[4] - debut["crude"].iloc[3])
    mixed = float(debut["adjusted"].iloc[4] - debut["adjusted"].iloc[3])
    assert facts["recovery_crude"] == pytest.approx(crude, abs=1e-12) and facts["recovery_adjusted"] == pytest.approx(mixed, abs=1e-12)
    assert facts["recovery_share_explained"] == pytest.approx(1 - mixed / crude, abs=1e-12) and 0.5 < facts["recovery_share_explained"] < 0.9


@ALL_TABLES
def test_exactly_two_inputs_are_clear_in_the_hazard_model_and_the_single_is_not_one_of_them(table, facts):
    ratios = table("p6b_hazard_ratios")
    assert ((ratios["low"] <= ratios["hazard_ratio"]) & (ratios["hazard_ratio"] <= ratios["high"])).all()
    blocks = ratios.groupby("block")[["block_p", "block_p_holm"]].first()
    assert len(blocks) == facts["n_inputs"] == 10 and (blocks["block_p_holm"] >= blocks["block_p"] - 1e-15).all()
    clear = sorted(blocks.index[blocks["block_p_holm"] < 0.05])
    assert clear == ["Album drop", "Debut place"] and facts["n_inputs_clear"] == 2 and facts["inputs_clear"] == "debut place, album drop"
    single = ratios[ratios["input"] == "Single"].iloc[0]
    assert single["low"] < 1 < single["high"], "after the other inputs are held fixed the single is not clear"
    debut = ratios[ratios["input"].str.startswith("Debut place")]
    assert (debut["low"] > 1).all() and len(debut) == 4, "all four debut bands are clearly above a top-10 debut"
    assert ratios.loc[ratios["input"] == "Debut place 31 to 40", "hazard_ratio"].iloc[0] == debut["hazard_ratio"].max() or facts["hr_b4"] > facts["hr_b5"]
    drop = ratios[ratios["input"] == "Album drop"].iloc[0]
    assert drop["low"] > 1 and facts["hr_drop"] == pytest.approx(float(drop["hazard_ratio"]), abs=1e-12)


@_needs(FACTS_FILE)
def test_the_single_loses_its_advantage_once_the_album_drop_is_in_and_the_track_count_is_a_trap(facts):
    assert facts["ladder1_single_high"] < 1 and facts["ladder2_single_high"] < 1, "on its own, and with the debut place, the single looks protective"
    assert facts["ladder3_single_low"] < 1 < facts["ladder3_single_high"], "with the album drop in, the single is no longer clear"
    assert facts["ladder5_single_hr"] > 2 and facts["tracks_hr_low"] > 1, "with the track count in the model the single's ratio jumps"
    assert facts["corr_single_tracks"] < -0.9
    assert facts["ladder4_single_hr"] == pytest.approx(facts["hr_single"], abs=1e-12), "version 4 is the main model"


@ALL_TABLES
def test_five_of_the_ten_inputs_act_differently_by_age_and_the_debut_place_effect_sits_in_the_first_three_days(table, facts):
    effects = table("p6b_effects_by_age")
    assert len(effects) == 14 * 3 and set(effects["window_number"]) == {1, 2, 3}
    blocks = effects.groupby("block")[["equal_effects_p", "equal_effects_p_holm"]].first()
    assert len(blocks) == 10 and (blocks["equal_effects_p_holm"] >= blocks["equal_effects_p"] - 1e-15).all()
    flagged = sorted(blocks.index[blocks["equal_effects_p_holm"] < 0.05])
    assert len(flagged) == facts["ph_n_flagged"] == 5
    assert "Debut place" in flagged and "Album drop" in flagged
    assert ((effects["low"] <= effects["hazard_ratio"]) & (effects["hazard_ratio"] <= effects["high"])).all()
    for band in ("Debut place 31 to 40", "Debut place 41 to 50"):
        ratios = effects[effects["input"] == band].sort_values("window_number")["hazard_ratio"].to_numpy()
        assert ratios[0] > 5 * ratios[1] and ratios[0] > 3 * ratios[2], f"{band}: the effect must be mostly a first-three-days effect"


# ================================================================================ the claims of Section 7
@ALL_TABLES
def test_the_danger_zone_table_adds_up_and_the_chance_of_the_last_day_steps_up_with_the_place(table, facts):
    zone = table("p6b_danger_zone")
    assert int(zone["chart_days"].sum()) == facts["n_days"] and int(zone["exits"].sum()) == facts["n_exits"]
    assert np.allclose(zone["chance_last_day"], zone["exits"] / zone["chart_days"])
    assert np.allclose(zone["share_of_exits"].sum(), 1.0) and np.allclose(zone["share_of_chart_days"].sum(), 1.0)
    assert ((zone["low"] <= zone["chance_last_day"] + 1e-12) & (zone["chance_last_day"] <= zone["high"] + 1e-12)).all()
    steps = np.diff(zone["chance_last_day"].to_numpy())
    assert (steps > 0).all() and len(steps) == 9, "the chance rises at every one of the nine steps from places 1-5 to 46-50"
    assert zone["chance_last_day"].iloc[-1] > 50 * zone["chance_last_day"].iloc[0]
    assert facts["dz_bottom_exits"] == int(zone["exits"].iloc[-2:].sum()) and facts["dz_top_exits"] == int(zone["exits"].iloc[:2].sum())
    assert facts["dz_bottom_exit_share"] == pytest.approx(float(zone["share_of_exits"].iloc[-2:].sum()), abs=1e-12)
    assert facts["dz_bottom_days_share"] == pytest.approx(float(zone["share_of_chart_days"].iloc[-2:].sum()), abs=1e-12)
    assert facts["dz_bottom_exit_share"] > 0.5 > 0.25 > facts["dz_bottom_days_share"] > 0.1


@ALL_TABLES
def test_the_pattern_of_the_place_does_not_hang_on_the_busiest_exit_day(table, facts):
    zone = table("p6b_danger_zone")
    own = zone["chance_last_day_without_busiest_day"].to_numpy()
    assert (np.diff(own) > 0).all(), "without the busiest exit day the chance still rises with the place at every step"
    assert ((zone["without_busiest_day_low"] <= zone["chance_last_day_without_busiest_day"] + 1e-12)
            & (zone["chance_last_day_without_busiest_day"] <= zone["without_busiest_day_high"] + 1e-12)).all()
    assert facts["busy1_text"] == "27 December 2024" and facts["busy1_exits"] == 49
    later = [facts[f"busy{k}_exits"] for k in (2, 3, 4)]
    assert later == sorted(later, reverse=True) and facts["busy1_exits"] > 2.5 * later[0], "one day stands out from all the others"
    assert facts["busy_exits"] == facts["busy1_exits"] and facts["busy_text"] == facts["busy1_text"]
    assert facts["busy_share"] == pytest.approx(facts["busy1_exits"] / facts["n_exits"], abs=1e-12) and facts["busy_share"] < 0.06
    assert facts["busy_top10"] + facts["busy_bottom10"] <= facts["busy_exits"] and facts["busy_top10"] > 0.5 * facts["dz_top_exits"], "most of the exits from the top 10 were on that day"
    assert facts["exit_days"] == 321 and facts["busy5_share"] < 0.15


# ================================================================================ the claims of Section 8
@ALL_TABLES
def test_the_debut_day_is_the_step_that_improves_the_ranking_and_timing_and_artist_history_are_not(table, facts):
    ladder = table("p6b_ladder")
    arrival = ladder[ladder["view"] == "arrival day"].reset_index(drop=True)
    assert list(arrival["step"]) == ["Release", "+ Timing", "+ Artist history", "+ Debut day"] and list(arrival["inputs"]) == [5, 13, 20, 23]
    assert ((arrival["low"] <= arrival["c_index"]) & (arrival["c_index"] <= arrival["high"])).all()
    assert (arrival["c_index_in_sample"] > arrival["c_index"]).all(), "scored on the songs it was fitted on, every model looks better"
    assert np.allclose(arrival["gain"].iloc[1:], arrival["c_index"].diff().iloc[1:])
    for row in (1, 2):
        assert arrival["gain_low"].iloc[row] < 0 < arrival["gain_high"].iloc[row], "timing and artist history add nothing the data can show"
    assert arrival["gain_low"].iloc[3] > 0, "the debut day clearly improves the ranking"
    assert 0.50 < arrival["c_index"].iloc[0] < 0.60 < 0.62 < arrival["c_index"].iloc[3] < 0.68
    for key, value in zip(("c1", "c2", "c3", "c4"), arrival["c_index"]):
        assert facts[key] == pytest.approx(float(value), abs=1e-12)
    assert facts["c_range_max"] <= 0.02, "the three splits of every rung agree to two digits"


@ALL_TABLES
def test_after_three_chart_days_the_ranking_is_much_sharper_and_the_arrival_facts_add_nothing_clear(table, facts):
    ladder = table("p6b_ladder")
    landmark = ladder[ladder["view"] == "after three chart days"].reset_index(drop=True)
    assert list(landmark["step"]) == ["Arrival-day information", "+ first three chart days", "First three chart days alone"]
    assert landmark["gain_low"].iloc[1] > 0.05, "the first three chart days improve the ranking a lot"
    assert landmark["c_index"].iloc[1] > 0.70 > landmark["c_index"].iloc[0] > 0.55
    assert landmark["c_index"].iloc[2] > 0.70, "the first three chart days alone are nearly as good as everything together"
    assert facts["lm_extra_low"] < 0 < facts["lm_extra_high"], "what the arrival facts add on top of the first three chart days includes zero"
    assert facts["lm_songs"] + facts["lm_excluded"] == facts["n_songs"] and facts["lm_songs"] == 599
    assert 0.8 < facts["lm_excluded_quick"] < 0.95


@ALL_TABLES
def test_the_text_quotes_the_gains_and_their_bands_of_the_saved_ladder(notebook, table):
    text = _all_text(notebook)
    ladder = table("p6b_ladder")
    for step in ("+ Timing", "+ Artist history", "+ Debut day", "+ first three chart days"):
        row = ladder[ladder["step"] == step].iloc[0]
        for value in (row["gain"], row["gain_low"], row["gain_high"]):
            assert f"{value:+.3f}" in text, f"the text does not quote {value:+.3f} (the gain or band of {step})"


@ALL_TABLES
def test_the_risk_groups_are_ordered_and_the_highest_risk_group_is_clearly_apart(table, facts):
    thirds = table("p6b_risk_thirds")
    assert set(thirds["view"]) == {"arrival day", "after three chart days"}
    sizes = {"arrival day": facts["n_songs"], "after three chart days": facts["lm_songs"]}
    for view, part in thirds.groupby("view", sort=False):
        part = part.reset_index(drop=True)
        assert list(part["third"]) == ["lowest risk", "middle", "highest risk"]
        assert int(part["songs"].sum()) == sizes[view] and part["songs"].max() - part["songs"].min() <= 1
        assert part["median"].is_monotonic_decreasing and part["s7"].is_monotonic_decreasing and part["s30"].is_monotonic_decreasing, view
        assert part["s30_high"].iloc[2] < part["s30_low"].iloc[1], f"{view}: the band of the highest-risk third at day 30 must not come near the middle third"
        assert part["s30"].iloc[0] > 4 * part["s30"].iloc[2]
        assert part["artists"].iloc[2] < part["artists"].iloc[0], "the highest-risk third is made of fewer artists"
    arrival = thirds[thirds["view"] == "arrival day"].reset_index(drop=True)
    assert abs(arrival["s30_low"].iloc[0] - arrival["s30_high"].iloc[1]) < 0.02, "the lowest and the middle third are the closer pair, their day-30 bands meet"
    assert facts["n_pairs"] == 359919 and facts["n_pairs_lm"] > 100_000


# ================================================================================ the claims of Section 9
@ALL_TABLES
def test_the_stress_runs_include_the_main_model_and_say_what_the_closing_says(table, facts):
    stress = table("p6b_stress_tests")
    assert list(dict.fromkeys(stress["run"])) == ["gap1", "gap3", "gap7", "gap14", "gap30", "first", "second", "block", "top10"]
    assert facts["st_runs"] == 9 and (stress.groupby("run").size() == 6).all()
    ratios = table("p6b_hazard_ratios").set_index("input")
    main = stress[stress["run"] == "gap7"].set_index("input")
    for name in main.index:
        assert main.loc[name, "hazard_ratio"] == pytest.approx(float(ratios.loc[name, "hazard_ratio"]), rel=1e-8), f"the run with the usual choices must be the model of Section 6 ({name})"
    debut = stress[stress["input"].str.startswith("Debut place")]
    drop = stress[stress["input"] == "Album drop"]
    single = stress[stress["input"] == "Single"]
    assert len(debut) == 36 and (debut["low"] > 1).all() and facts["st_debut_clear"] == facts["st_debut_total"] == 36
    assert len(drop) == 9 and (drop["low"] > 1).all() and facts["st_drop_clear"] == facts["st_runs"] == 9
    assert len(single) == 9 and ((single["low"] < 1) & (single["high"] > 1)).all() and facts["st_single_clear"] == 0
    assert float(debut["hazard_ratio"].min()) == pytest.approx(facts["st_debut_min"], abs=1e-12) and float(debut["hazard_ratio"].max()) == pytest.approx(facts["st_debut_max"], abs=1e-12)
    one_per_run = stress.groupby("run")[["songs", "exits", "censored", "share_after_30_days", "median_stay"]].first()
    assert (one_per_run["exits"] + one_per_run["censored"] == one_per_run["songs"]).all()
    assert one_per_run.loc["gap7", "songs"] == facts["n_songs"] and one_per_run.loc["gap7", "exits"] == facts["n_exits"]
    shares = one_per_run.loc[["gap1", "gap3", "gap7", "gap14", "gap30"], "share_after_30_days"]
    assert shares.is_monotonic_increasing, "a longer gap rule joins more returning songs into one stay, so more songs reach day 30"
    assert facts["st_s30_gap_min"] == pytest.approx(float(shares.min()), abs=1e-12) and facts["st_s30_gap_max"] == pytest.approx(float(shares.max()), abs=1e-12)
    assert one_per_run.loc["top10", "songs"] == facts["n_songs"] - facts["top_artists_songs"]
    assert one_per_run.loc["block", "exits"] == facts["n_exits"] - facts["busy1_exits"], "the block of the busiest day counts as unfinished, not as removed"
    assert one_per_run.loc["block", "songs"] == facts["n_songs"]


# ================================================================================ the closing text and the numbers
@_needs(FACTS_FILE)
def test_the_closing_quotes_the_saved_numbers(notebook, facts):
    closing = notebook.cells[-1].source
    wanted = [f"{facts['n_songs']:,}", f"{100 * facts['s1']:.0f}%", f"{100 * facts['s7']:.0f}%", f"{100 * facts['s30']:.0f}%",
              f"{100 * facts['s30_low']:.0f}% to {100 * facts['s30_high']:.0f}%", f"{facts['stay_mean_raw']:.1f}", f"{facts['stay_max']}",
              f"{facts['c4']:.2f}", f"{facts['lm2']:.2f}", f"{facts['busy1_exits']} stays ended on {facts['busy1_text']}",
              f"{facts['st_debut_clear']} of {facts['st_debut_total']}", f"{facts['st_drop_clear']} of {facts['st_runs']}",
              f"{facts['hr_b4']:.1f}", f"{facts['hr_drop']:.1f}", f"{facts['hr_single']:.2f}", f"{facts['ph_n_flagged']} of the {facts['n_inputs']} inputs"]
    for item in wanted:
        assert item in closing, f"the closing text does not hold {item!r}"


@_needs(FACTS_FILE)
def test_the_busy_day_and_the_open_stays_are_described_with_the_facts_of_the_file(notebook, facts):
    text = _all_text(notebook)
    assert f"{facts['busy1_exits']} of {facts['n_exits']}" in text or f"{facts['busy1_exits']}" in text
    assert str(facts["open_late_arrivals"]) in text and str(facts["open_long_stays"]) in text
    assert f"{facts['n_censored']} stays" in text
