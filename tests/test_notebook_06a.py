"""
tests/test_notebook_06a.py  -  structure checks of the Phase 6A notebook (notebooks/06a_supervised_models.ipynb) and of the six files it writes.

WHAT THIS FILE IS
    Checks of what the notebook IS, not of what the models compute (the validation, the metrics, the models and the explanation tools are
    tested in test_ml_*.py). The notebook must be valid, delivered without outputs, every code cell must compile, every code cell must have
    its five-part plain-English explanation directly above it, every chart must have a "What to notice" note directly below it, no sentence
    may contain an unfilled gap or a double negative such as "fell by about -0.5", the 17 charts must be numbered 1 to 17 without a hole,
    and every number that the code looks up in FACTS must exist in the facts file the notebook saved.
    The later tests read the six files the notebook writes. They check that the tables are complete and follow one rule each: the model
    register has one row per question, step and model; the saved forecasts reproduce the scores of the register to the last digit; the
    break-even table follows the formula in its own header; the stress table agrees with the facts that summarise it. The last tests
    check that the closing text of the notebook says only what the saved numbers say (for example "no flexible model is clearly better"
    for the three Top-10 questions).

WHY WE NEED IT
    The explanations, the notes and the closing statements are the point of this notebook, and the saved tables are what the dashboard, the
    Power BI report and the paper will read. A cell that lost its explanation during an edit, a sentence that still holds a "{{gap}}",
    a table whose forecasts no longer give the register's scores, or a closing text that disagrees with the saved numbers would otherwise be
    found by the reader and not by us.

WHAT HAPPENS WHEN IT RUNS
    The notebook is read as plain JSON/nbformat; nothing is executed. The tests that read the saved files are skipped when those files are
    not there yet (run the notebook once to create them).

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    A teaching notebook, and the forecasts and tables that later phases read, whose structure cannot silently rot.
"""
import ast
import json
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from src.ml import metrics as M

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "06a_supervised_models.ipynb"
FACTS_FILE = ROOT / "reports" / "p6a_facts.json"
REGISTER_FILE = ROOT / "reports" / "p6a_model_register.csv"
PREDICTIONS_FILE = ROOT / "reports" / "p6a_oof_predictions.csv"
EXPLANATIONS_FILE = ROOT / "reports" / "p6a_explanations.csv"
DECISIONS_FILE = ROOT / "reports" / "p6a_decisions.csv"
STRESS_FILE = ROOT / "reports" / "p6a_stress_tests.csv"
FIGURE_DIR = ROOT / "reports" / "figures"
ALL_FILES = [FACTS_FILE, REGISTER_FILE, PREDICTIONS_FILE, EXPLANATIONS_FILE, DECISIONS_FILE, STRESS_FILE]

FIVE_PARTS = ["**What this code is.**", "**Why we need it.**", "**What will happen.**", "**Expected outcome.**", "**Motive.**"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U00002B00-\U00002BFF\U0000FE0F]")

# the seven questions: kind, and the steps of the information ladder at which each is asked
QUESTIONS = {"C1": ("binary", ["release", "timing", "artist"]),
             "C2": ("binary", ["release", "timing", "artist", "debut"]),
             "C3": ("multiclass", ["release", "timing", "artist", "debut"]),
             "R1": ("regression", ["release", "timing", "artist", "debut"]),
             "R2": ("regression", ["release", "timing", "artist", "debut"]),
             "E1": ("binary", ["release", "timing", "artist", "debut", "early"]),
             "E2": ("binary", ["release", "timing", "artist", "debut", "early"])}
STAY_CLASSES = ["Flash (1-3 days)", "Short (4-14)", "Sustained (15+)"]

pytestmark = pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebooks/06a_supervised_models.ipynb is not there")


def _needs(*files):
    missing = [file.name for file in files if not file.exists()]
    return pytest.mark.skipif(bool(missing), reason=f"run the notebook once to create {', '.join(missing)}")


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


@pytest.fixture(scope="module")
def facts():
    return json.loads(FACTS_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def register():
    return pd.read_csv(REGISTER_FILE, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def forecasts():
    return pd.read_csv(PREDICTIONS_FILE, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def explanations():
    return pd.read_csv(EXPLANATIONS_FILE, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def decisions():
    return pd.read_csv(DECISIONS_FILE, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def stress():
    return pd.read_csv(STRESS_FILE, encoding="utf-8-sig")


def _code_cells(notebook):
    return [(index, cell) for index, cell in enumerate(notebook.cells) if cell.cell_type == "code"]


def _markdown_text(notebook):
    return [(index, cell.source) for index, cell in enumerate(notebook.cells) if cell.cell_type == "markdown"]


CHART_NAME = re.compile(r"""["'](p6a_\d\d_[a-z0-9_]+)["']""")             # a chart is named by a string such as "p6a_15_gains", in save_fig(...) or in a call such as detail_figure(...)


def _chart_cells(notebook):
    return [(index, cell) for index, cell in _code_cells(notebook) if CHART_NAME.search(cell.source)]


def _chart_names(notebook):
    names = []
    for _, cell in _chart_cells(notebook):
        names += CHART_NAME.findall(cell.source)
    return names


def _all_text(notebook):
    return "\n".join(text for _, text in _markdown_text(notebook))


def _quoted(text, value, digits):
    """True when `text` holds `value` written with `digits` decimals, or the neighbour one unit of the last digit away (a different
    library version may move a result in its last printed digit; the text should then still count as written for these numbers)."""
    unit = 10.0 ** -digits
    candidates = {f"{value + shift * unit:.{digits}f}" for shift in (-1, 0, 1)}
    candidates |= {c.replace("-", "\u2212") for c in candidates}
    return any(candidate in text for candidate in candidates)


def _quoted_after(text, anchor, value, digits, slack=1):
    """Like `_quoted`, but the number has to stand right after the words that introduce it (`anchor`, a regular expression), and it may be
    up to `slack` units of the last printed digit away from `value`. A number that is looked for anywhere in a long paragraph can be found
    by accident, because the paragraph holds dozens of numbers and every number has neighbours; next to its own words it can not."""
    unit = 10.0 ** -digits
    candidates = {f"{value + shift * unit:.{digits}f}" for shift in range(-slack, slack + 1)}
    candidates |= {c.replace("-", "\u2212") for c in candidates}
    return any(re.search(anchor + r"[+]?" + re.escape(candidate) + r"(?![0-9])", text) for candidate in candidates)


# ================================================================================ the file itself
def test_the_notebook_is_valid_and_has_the_cells_we_expect(notebook):
    nbformat.validate(notebook)
    assert len(_code_cells(notebook)) == 35
    assert notebook.cells[0].cell_type == "markdown" and notebook.cells[0].source.startswith("# Phase 6A")


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
    assert [int(name[4:6]) for name in names] == list(range(1, 18))
    assert len(set(names)) == 17


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
    assert figures and figures <= set(range(1, 18)), f"a reference to a figure that does not exist: {sorted(figures - set(range(1, 18)))}"
    assert sections and sections <= set(range(1, 11)), f"a reference to a section that does not exist: {sorted(sections - set(range(1, 11)))}"


def test_the_notebook_ends_with_the_closing_summary_and_the_hand_over_to_phase_6b(notebook):
    last = notebook.cells[-1]
    assert last.cell_type == "markdown" and last.source.startswith("## What Phase 6A adds up to")
    assert "Phase 6B" in last.source
    for file in ALL_FILES:
        assert file.name in last.source, f"the closing text does not name {file.name}"


def test_the_closing_keeps_the_caveats_that_the_brief_requires(notebook):
    closing = notebook.cells[-1].source.lower()
    assert "causal" in closing, "the closing must say that nothing here is causal"
    assert "not advice" in closing and "historical analytics" in closing, "the closing must say that the forecasts are measurements, not advice"
    assert "chosen on the songs they are scored on" in closing, "the closing must disclose that the models of the detail charts were chosen on the same songs"


# ================================================================================ the sentences
def test_no_sentence_contains_an_unfilled_gap(notebook):
    for index, text in _markdown_text(notebook):
        for marker in ("{{", "}}", "??", "\u2026", "TODO", "TBD", "XXX"):
            assert marker not in text, f"markdown cell {index} contains {marker!r}"
        assert not re.search(r"\bnan\b|\binf\b", text), f"markdown cell {index} prints a nan or an inf"


def test_no_sentence_says_that_something_fell_or_was_less_by_a_negative_number(notebook):
    """'fell by about -0.47 points' and '-3.4 points less popular' are double negatives: the sign is already in the verb."""
    patterns = [r"\b(fell|falls|fall|dropped|drops|drop|decreased|decreases|lower|smaller|less|fewer)\b (by|of) (about )?[-\u2212]\d",
                r"(?<![\w.])[-\u2212]\d[\d.,]* points \*?(less|lower|fewer|smaller)\*?\b"]
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
@_needs(FACTS_FILE)
def test_every_fact_the_code_looks_up_exists_in_the_facts_file(notebook, facts):
    wanted = set()
    for _, cell in _code_cells(notebook):
        wanted |= set(re.findall(r"\b(?:FACTS|F)\[\s*['\"]([a-z0-9_]+)['\"]\s*\]", cell.source))
    assert len(wanted) > 100
    assert wanted - set(facts) == set(), "the code uses facts that the saved file does not hold"


@_needs(FACTS_FILE)
def test_the_facts_file_is_sorted_and_holds_only_plain_finite_values():
    text = FACTS_FILE.read_text(encoding="utf-8")
    facts = json.loads(text)
    assert list(facts) == sorted(facts)
    assert len(facts) > 1000
    assert all(isinstance(value, (bool, int, float, str)) for value in facts.values())
    assert all(np.isfinite(value) for value in facts.values() if isinstance(value, float) and not isinstance(value, bool))
    assert text.endswith("}\n")


# ================================================================================ the model register
@_needs(REGISTER_FILE)
def test_the_register_has_one_row_per_question_step_and_model(register):
    assert not register.duplicated(["question", "step", "model"]).any()
    assert set(register["question"]) == set(QUESTIONS)
    expected = 0
    for question, (kind, steps) in QUESTIONS.items():
        part = register[register["question"] == question]
        assert set(part["kind"]) == {kind}, question
        assert [s for s in ["release", "timing", "artist", "debut", "early"] if s in set(part["step"])] == steps, question
        linear = "ridge" if kind == "regression" else "logistic"
        for step in steps:
            models = set(part.loc[part["step"] == step, "model"])
            wanted = {"baseline", linear, "boosting"} | ({"forest"} if step == steps[-1] else set())
            assert models == wanted, f"{question} at {step}: {sorted(models)}"
            expected += len(wanted)
    assert len(register) == expected


@_needs(REGISTER_FILE)
def test_the_register_scores_are_filled_for_the_right_metrics_and_lie_inside_their_intervals(register):
    metrics = {"binary": ["auc", "average_precision", "brier", "log_loss"],
               "multiclass": ["balanced_accuracy", "macro_f1", "weighted_kappa", "accuracy", "log_loss"],
               "regression": ["r2", "mae", "rmse"]}
    every = {m for names in metrics.values() for m in names}
    for kind, names in metrics.items():
        part = register[register["kind"] == kind]
        for name in every:
            filled = part[[name, f"{name}_low", f"{name}_high"]].notna().all(axis=None)
            empty = part[[name, f"{name}_low", f"{name}_high"]].isna().all(axis=None)
            assert filled if name in names else empty, f"{kind}: {name} should be {'filled' if name in names else 'empty'}"
        for name in names:
            assert (part[f"{name}_low"] <= part[f"{name}_high"]).all(), f"{kind} {name}: an interval runs backwards"
    inside = (register["auc_low"] - 1e-9 <= register["auc"]) & (register["auc"] <= register["auc_high"] + 1e-9)
    assert inside[register["auc"].notna()].all(), "an AUC lies outside its own interval"


@_needs(REGISTER_FILE)
def test_a_baseline_that_knows_nothing_scores_as_a_coin(register):
    base = register[register["model"] == "baseline"]
    binary = base[base["kind"] == "binary"]
    assert np.allclose(binary["auc"], 0.5) and np.allclose(binary["auc_low"], 0.5) and np.allclose(binary["auc_high"], 0.5)
    multi = base[base["kind"] == "multiclass"]
    assert np.allclose(multi["balanced_accuracy"], 1 / 3) and np.allclose(multi["weighted_kappa"], 0.0)
    regression = base[base["kind"] == "regression"]
    assert regression["r2"].between(-0.1, 0.0).all(), "always forecasting the average of the training songs is a little worse than 0, never better"


@_needs(REGISTER_FILE, FACTS_FILE)
def test_the_scores_quoted_as_facts_are_the_scores_of_the_register(register, facts):
    pattern = re.compile(r"^(c1|c2|c3|e1|e2|r1|r2)_(release|timing|artist|debut|early)_(logistic|ridge|boosting|forest)_(auc|r2|balanced_accuracy)$")
    keys = [key for key in facts if pattern.match(key)]
    assert len(keys) > 40
    for key in keys:
        question, step, model, metric = pattern.match(key).groups()
        row = register[(register["question"] == question.upper()) & (register["step"] == step) & (register["model"] == model)]
        assert len(row) == 1, key
        assert facts[key] == pytest.approx(float(row.iloc[0][metric]), rel=1e-9, abs=1e-12), key


# ================================================================================ the saved forecasts
@_needs(REGISTER_FILE, PREDICTIONS_FILE)
def test_the_saved_forecasts_hold_one_row_per_song_for_the_last_step_of_every_question(register, forecasts):
    assert not forecasts.duplicated(["question", "step", "model", "track_id"]).any()
    for question, (kind, steps) in QUESTIONS.items():
        part = forecasts[forecasts["question"] == question]
        assert set(part["step"]) == {steps[-1]}
        linear = "ridge" if kind == "regression" else "logistic"
        assert set(part["model"]) == {"baseline", linear, "boosting", "forest"}
        songs = register.loc[(register["question"] == question) & (register["step"] == steps[-1]), "songs"].unique()
        assert len(songs) == 1
        assert part.groupby("model").size().eq(int(songs[0])).all(), question
        ids = part.groupby("model")["track_id"].apply(lambda s: tuple(sorted(s)))
        assert ids.nunique() == 1, f"{question}: the models forecast different songs"


@_needs(PREDICTIONS_FILE)
def test_forecast_chances_are_chances_and_the_outcomes_are_the_ones_of_the_question(forecasts):
    for question, (kind, _) in QUESTIONS.items():
        part = forecasts[forecasts["question"] == question]
        if kind == "binary":
            assert set(part["actual"].astype(float)) == {0.0, 1.0}
            assert part["forecast"].between(0, 1).all() and part["forecast"].notna().all()
        elif kind == "multiclass":
            assert set(part["actual"]) == set(STAY_CLASSES) and set(part["forecast_class"]) <= set(STAY_CLASSES)
            chances = part[["p_flash", "p_short", "p_sustained"]]
            assert chances.notna().all(axis=None) and chances.ge(0).all(axis=None)
            assert np.allclose(chances.sum(axis=1), 1.0)
            most_likely = chances.to_numpy().argmax(axis=1)
            assert (part["forecast_class"].map({c: i for i, c in enumerate(STAY_CLASSES)}).to_numpy() == most_likely).all()
        else:
            assert part["actual"].astype(float).notna().all() and part["forecast"].notna().all()


@_needs(REGISTER_FILE, PREDICTIONS_FILE)
def test_the_saved_forecasts_give_back_the_scores_of_the_register(register, forecasts):
    """The register was made from the same forecasts: scoring the saved file again must give its numbers, to the last digit."""
    checked = 0
    for (question, step, model), part in forecasts.groupby(["question", "step", "model"]):
        if model == "baseline":
            continue                                           # the baseline is scored as 'knows nothing' in the register
        row = register[(register["question"] == question) & (register["step"] == step) & (register["model"] == model)].iloc[0]
        kind = QUESTIONS[question][0]
        if kind == "binary":
            y, s = part["actual"].astype(float).to_numpy(), part["forecast"].to_numpy()
            again = {"auc": M.roc_auc(y, s), "brier": M.brier_score(y, s), "average_precision": M.average_precision(y, s)}
        elif kind == "regression":
            y, s = part["actual"].astype(float).to_numpy(), part["forecast"].to_numpy()
            again = {"r2": M.r_squared(y, s), "mae": M.mean_absolute_error(y, s), "rmse": M.root_mean_squared_error(y, s)}
        else:
            index = {c: i for i, c in enumerate(STAY_CLASSES)}
            y, guess = part["actual"].map(index).to_numpy(), part["forecast_class"].map(index).to_numpy()
            scores = M.matrix_scores(M.confusion_matrix(y, guess, 3))
            again = {"balanced_accuracy": scores["balanced_accuracy"], "accuracy": scores["accuracy"],
                     "log_loss": M.multiclass_log_loss(y, part[["p_flash", "p_short", "p_sustained"]].to_numpy())}
        for name, value in again.items():
            assert value == pytest.approx(float(row[name]), abs=1e-9), f"{question} {model} {name}"
            checked += 1
    assert checked >= 60


# ================================================================================ the explanations
@_needs(EXPLANATIONS_FILE)
def test_the_explanations_table_is_complete_and_every_estimate_lies_inside_its_interval(explanations):
    assert explanations[["kind", "question", "model", "input", "estimate"]].notna().all(axis=None)
    assert not explanations.duplicated(["kind", "question", "model", "input"]).any()
    odds = explanations[explanations["kind"] == "odds ratio per standard deviation"]
    assert set(odds["question"]) == {"C1", "C2"} and (odds["estimate"] > 0).all()
    assert odds["p_value"].between(0, 1).all()
    for name, part in explanations.groupby("kind"):
        if name.startswith("fall in"):
            assert part["spread"].ge(0).all() and part["spread"].notna().all(), name
        else:
            assert part[["low", "high"]].notna().all(axis=None), name
            assert ((part["low"] <= part["estimate"] + 1e-9) & (part["estimate"] <= part["high"] + 1e-9)).all(), f"{name}: an estimate outside its own interval"
    assert set(explanations.loc[explanations["kind"].str.startswith("fall in"), "question"]) == set(QUESTIONS)


@_needs(EXPLANATIONS_FILE, FACTS_FILE)
def test_the_importances_quoted_as_facts_are_the_ones_of_the_table(explanations, facts):
    falls = explanations[explanations["kind"].str.startswith("fall in")]
    blocks = {"Release details": "release", "Timing": "timing", "Artist's record": "artist", "Debut-day chart": "debut", "First three days": "early"}
    checked = 0
    for _, row in falls.iterrows():
        key = f"perm_{row['question'].lower()}_{blocks[row['input']]}"
        assert facts[key] == pytest.approx(float(row["estimate"]), rel=1e-9, abs=1e-12), key
        checked += 1
    assert checked == 29, "3 + 4 + 4 + 4 + 4 + 5 + 5 blocks"


# ================================================================================ the break-even table
@_needs(DECISIONS_FILE)
def test_the_break_even_table_follows_its_own_formula(decisions):
    assert set(decisions["question"]) == {"C2", "E1"}
    for question, part in decisions.groupby("question"):
        assert len(part) == 7 * 8 and not part.duplicated(["share_promoted", "value_ratio"]).any()
        assert (part["hits"] <= part["promoted"]).all()
        assert np.allclose(part["precision"], part["hits"] / part["promoted"])
        assert np.allclose(part["net"], part["hits"] * part["value_ratio"] - part["promoted"]), "net = hits x value - promoted"
        assert np.allclose(part["roi"], part["value_ratio"] * part["precision"] - 1), "return per unit = value x hit rate - 1"
        assert np.allclose(part["break_even"], 1 / part["value_ratio"]), "the hit rate needed to break even is 1 / value"
        assert ((part["roi"] > 1e-12) == (part["precision"] > part["break_even"] + 1e-12)).all(), "pays back exactly when the hit rate beats the break-even hit rate"
        for _, group in part.groupby("value_ratio"):
            ordered = group.sort_values("share_promoted")
            assert ordered["hits"].is_monotonic_increasing and ordered["promoted"].is_monotonic_increasing, "backing more songs can only find more of them"


@_needs(DECISIONS_FILE, FACTS_FILE)
def test_the_break_even_facts_are_the_numbers_of_the_table(decisions, facts):
    for question in ("C2", "E1"):
        code = question.lower()
        part = decisions[decisions["question"] == question]
        top = part[(np.isclose(part["share_promoted"], 0.10)) & (np.isclose(part["value_ratio"], 2.0))].iloc[0]
        assert facts[f"{code}_p10_promoted"] == int(top["promoted"]) and facts[f"{code}_p10_hits"] == int(top["hits"])
        assert facts[f"{code}_p10_break_even"] == pytest.approx(1 / float(top["precision"]), rel=1e-9)
        assert facts[f"{code}_pays_cells"] == int((part["roi"] > 0).sum()) and facts[f"{code}_grid_cells"] == len(part)
        for ratio in (4, 8, 12):
            row = part[(np.isclose(part["share_promoted"], 0.10)) & (np.isclose(part["value_ratio"], ratio))].iloc[0]
            assert facts[f"{code}_roi_p10_r{ratio}"] == pytest.approx(float(row["roi"]), rel=1e-9, abs=1e-12)


# ================================================================================ the stress tests
@_needs(STRESS_FILE, FACTS_FILE)
def test_the_stress_table_has_the_same_checks_for_every_question_and_model(stress, facts):
    assert set(stress["question"]) == {"C1", "C2", "E2"} and set(stress["model"]) == {"Logistic regression", "Gradient boosting"}
    biggest = facts["stress_biggest_artist"]
    for (question, model), part in stress.groupby(["question", "model"]):
        checks = list(part["check"])
        base = ["as in the notebook", "other artist split 1", "other artist split 2", "other artist split 3", "other artist split 4", f"without {biggest}",
                f"forward in time (songs from day {facts['stress_cutoff']} on)", "the same later songs, notebook forecasts"]
        settings = 2 if model == "Logistic regression" else 3
        assert len(checks) == len(base) + settings and set(base) <= set(checks), (question, model)
        assert sum(c.startswith("setting: ") for c in checks) == settings
        assert part["score"].between(0, 1).all()
        bars = part[part["check"].isin(["as in the notebook", f"without {biggest}", base[6], base[7]])]
        assert bars[["low", "high"]].notna().all(axis=None) and ((bars["low"] <= bars["score"]) & (bars["score"] <= bars["high"])).all()


@_needs(STRESS_FILE, FACTS_FILE, REGISTER_FILE)
def test_the_stress_facts_are_the_numbers_of_the_stress_table(stress, facts, register):
    biggest = facts["stress_biggest_artist"]
    shifts = {"seeds": [], "settings": [], "without": []}
    for (question, model), part in stress.groupby(["question", "model"]):
        own = float(part.loc[part["check"] == "as in the notebook", "score"].iloc[0])
        last = QUESTIONS[question][1][-1]
        key = "logistic" if model == "Logistic regression" else "boosting"
        assert own == pytest.approx(float(register[(register["question"] == question) & (register["step"] == last) & (register["model"] == key)]["auc"].iloc[0]), abs=1e-12), "the first row must be the register's own score"
        assert facts[f"stress_{question.lower()}_{key}_own"] == pytest.approx(own, abs=1e-12)
        shifts["seeds"] += [abs(s - own) for s in part.loc[part["check"].str.startswith("other artist split"), "score"]]
        shifts["settings"] += [abs(s - own) for s in part.loc[part["check"].str.startswith("setting: "), "score"]]
        shifts["without"].append(float(part.loc[part["check"] == f"without {biggest}", "score"].iloc[0]) - own)
        forward = float(part.loc[part["check"].str.startswith("forward in time"), "score"].iloc[0])
        assert facts[f"stress_{question.lower()}_{key}_forward"] == pytest.approx(forward, abs=1e-12)
    assert facts["stress_seeds_max_shift"] == pytest.approx(max(shifts["seeds"]), abs=1e-12)
    assert facts["stress_settings_max_shift"] == pytest.approx(max(shifts["settings"]), abs=1e-12)
    assert facts["stress_without_max_shift"] == pytest.approx(max(abs(s) for s in shifts["without"]), abs=1e-12)
    assert facts["stress_without_higher"] == sum(s > 0 for s in shifts["without"])
    assert facts["stress_without_smallest_rise"] == pytest.approx(min(shifts["without"]), abs=1e-12)
    assert facts["stress_max_shift"] == pytest.approx(max(facts["stress_seeds_max_shift"], facts["stress_settings_max_shift"], facts["stress_without_max_shift"]), abs=1e-12)
    assert facts["stress_train_before"] == facts["stress_cutoff"] - facts["stress_gap"]


# ================================================================================ the closing text and the numbers
@_needs(FACTS_FILE)
def test_the_closing_says_only_what_the_comparisons_of_the_models_show(notebook, facts):
    """The closing says that no flexible model is clearly better for C1, C2 and E1 (every interval includes zero) and that the flexible
    models are clearly better for C3, R2 and E2 (every interval is above zero). The saved comparisons must agree."""
    closing = notebook.cells[-1].source
    assert "no flexible model is clearly better than the straight-line model" in closing
    for question in ("c1", "c2", "e1"):
        for model in ("boosting", "forest"):
            low, high = facts[f"{question}_{model}_vs_logistic_low"], facts[f"{question}_{model}_vs_logistic_high"]
            assert low <= 0 <= high, f"{question} {model}: the interval of the difference no longer includes zero"
    assert "the flexible models are clearly better" in closing
    for question, linear in (("c3", "logistic"), ("e2", "logistic"), ("r2", "ridge")):
        for model in ("boosting", "forest"):
            assert facts[f"{question}_{model}_vs_{linear}_low"] > 0, f"{question} {model}: the difference is no longer clear of zero"


@_needs(FACTS_FILE)
def test_the_stress_claims_in_the_text_agree_with_the_facts(notebook, facts):
    text = _all_text(notebook)
    if facts["stress_without_higher"] == 6:
        assert "raised the AUC in 6 of the six cases" in text
    assert facts["stress_without_higher"] == 6 or "raised the AUC in 6 of the six cases" not in text, "the text claims a rise in all six cases"
    assert "In every row the bar of the forward score overlaps the bar of the notebook's score" in text
    for question in ("c1", "c2", "e2"):
        for model in ("logistic", "boosting"):
            forward = (facts[f"stress_{question}_{model}_forward_low"], facts[f"stress_{question}_{model}_forward_high"])
            same = (facts[f"stress_{question}_{model}_same_later_low"], facts[f"stress_{question}_{model}_same_later_high"])
            assert max(forward[0], same[0]) <= min(forward[1], same[1]), f"{question} {model}: the bars no longer overlap"


@_needs(FACTS_FILE)
def test_the_closing_quotes_the_saved_numbers(notebook, facts):
    """The text was written for the numbers of one run (scikit-learn 1.9.1). The straight-line models give the same numbers on every setup,
    but boosting and forest can move a little with the version of scikit-learn. Measured against a run on scikit-learn 1.4.0, the oldest
    version the project allows: the quoted numbers below moved by at most 2.1 units of their last printed digit (the importance of the
    debut-day chart for C3, three decimals). The test allows a few units (SLACK, by number of decimals); a bigger move means that the text
    and the saved numbers no longer tell the same story. Each number is looked for right after the words that introduce it."""
    closing = notebook.cells[-1].source
    quotes = [("c1_artist_logistic_auc", 2, r"Top-10 debut can be ranked with an AUC of "),
              ("c1_artist_logistic_auc_low", 2, r"\(interval "),
              ("c1_artist_logistic_auc_high", 2, r"\(interval \d\.\d\d to "),
              ("c2_debut_boosting_auc", 2, r"can be ranked with an AUC of up to "),
              ("c3_debut_forest_balanced_accuracy", 2, r"the stay class with a balanced accuracy of "),
              ("r2_best_model_r2", 2, r"the number of chart days with an R-squared of "),
              ("r1_best_model_r2", 2, r"\(best R-squared "),
              ("e1_early_best_auc", 2, r"After three days on the chart the AUC is "),
              ("e2_early_best_auc", 2, r"which rests on \d+ climbers, and "),
              ("perm_c1_artist", 3, r"Shuffling the artist's record costs the boosting model "),
              ("perm_c3_debut", 3, r"shuffling the debut-day chart costs "),
              ("perm_r2_debut", 3, r"of balanced accuracy and "),
              ("perm_e1_early", 3, r"shuffling them costs "),
              ("perm_e2_early", 3, r"of AUC for E1 and "),
              ("c2_p10_break_even", 1, r"if a climb were worth more than "),
              ("e1_p10_break_even", 1, r"on the debut day and more than "),
              ("c2_random_break_even", 0, r"after three days, against "),
              ("e1_random_break_even", 0, r"after three days, against \d+ and "),
              ("leak_gap_mean", 3, r"Random folds would have raised the AUC by "),
              ("leak_gap_max", 3, r"and by up to "),
              ("placebo_mean", 2, r"the \d+ scores averaged "),
              ("stress_max_shift", 3, r"moved the AUC by at most "),
              ("stress_c1_forward_drop", 3, r"C1 and E2 came out "),
              ("stress_e2_forward_drop", 3, r"C1 and E2 came out \d\.\d\d\d and "),
              ("c3_forest_vs_logistic", 3, r"the random forest minus the straight line is "),
              ("e2_forest_vs_logistic", 3, r"balanced accuracy for C3 and "),
              ("r2_boosting_vs_ridge", 3, r"boosting minus ridge regression is ")]
    slack = {0: 1, 1: 5, 2: 2, 3: 5}                           # units of the last digit: +-1 whole promotion, +-0.5, +-0.02, +-0.005
    for key, digits, anchor in quotes:
        assert _quoted_after(closing, anchor, facts[key], digits, slack[digits]), \
            f"the closing text does not quote {key} = {facts[key]:.{digits + 1}f} (within {slack[digits]} unit of the last digit) after the words '{anchor}'"
    assert facts["c2_p10_break_even"] < facts["c2_random_break_even"] and facts["e1_p10_break_even"] < facts["e1_random_break_even"], "the forecasts must lower the break-even value"


@_needs(FACTS_FILE)
def test_the_notes_under_the_gains_and_the_break_even_quote_the_saved_numbers(notebook, facts):
    text = _all_text(notebook)
    for key, digits in [("gain_c2_debut_p10", 2), ("gain_e1_early_p10", 2), ("gain_c1_artist_p10", 2), ("gain_c1_artist_p20", 2)]:
        assert _quoted(text, 100 * facts[key], 0), f"the notes do not quote {key}"
    for key in ("c2_roi_p10_r8", "e1_roi_p10_r8"):
        assert _quoted(text, facts[key], 2), f"the notes do not quote {key}"
    assert facts["c2_p10_hits"] == facts["gain_c2_debut_p10"] * facts["c2_positives_total"] or round(facts["gain_c2_debut_p10"] * facts["c2_positives_total"]) == facts["c2_p10_hits"]
    assert round(facts["gain_e1_early_p10"] * facts["e1_positives_total"]) == facts["e1_p10_hits"]


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
