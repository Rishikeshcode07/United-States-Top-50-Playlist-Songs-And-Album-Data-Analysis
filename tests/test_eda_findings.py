"""
tests/test_eda_findings.py  -  tests for the facts store and the register of findings (src/eda/findings.py).

WHAT THIS FILE IS
    Checks that `Facts` refuses a second value for the same name, refuses numbers that are not finite and refuses things that are
    not numbers, texts or true/false values, and that `Findings` numbers its entries F01, F02 ... and refuses empty text. Both
    save their file in one step, and the tests prove that a failed save leaves the old file untouched.

WHY WE NEED IT
    Every sentence of the EDA notebook is written from a stored fact, and the register is the list that the research paper
    is written from. If a name could be overwritten silently, two charts could quote two different numbers under one name, and
    the text would follow whichever was set last.

WHAT HAPPENS WHEN IT RUNS
    Small dictionaries and registers are built, saved into pytest's temporary folder and read back.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Every claim in the project can be traced to a chart and to a number, and the numbers cannot change behind our back.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.eda.findings import FINDING_COLUMNS, Facts, Findings, _write_atomic


# ================================================================================ Facts
def test_a_name_can_be_set_once():
    facts = Facts()
    facts["median_days"] = 7.0
    assert facts["median_days"] == 7.0
    with pytest.raises(KeyError, match="median_days"):
        facts["median_days"] = 8.0
    assert facts["median_days"] == 7.0


def test_numpy_numbers_become_plain_python_numbers():
    facts = Facts()
    facts["a"], facts["b"], facts["c"] = np.float64(1.5), np.int64(3), np.bool_(True)
    assert (type(facts["a"]), type(facts["b"]), type(facts["c"])) == (float, int, bool)
    assert json.loads(json.dumps(dict(facts))) == {"a": 1.5, "b": 3, "c": True}


def test_texts_and_true_false_values_are_allowed():
    facts = Facts()
    facts["leader"], facts["flag"], facts["count"] = "Taylor Swift", False, 0
    assert dict(facts) == {"leader": "Taylor Swift", "flag": False, "count": 0}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), np.float64("nan"), np.float32("inf")])
def test_numbers_that_are_not_finite_are_refused(bad):
    facts = Facts()
    with pytest.raises(ValueError, match="finite"):
        facts["x"] = bad
    assert "x" not in facts


@pytest.mark.parametrize("bad", [[1, 2], None, {"a": 1}, (1, 2), pd.Timestamp("2025-01-01")])
def test_things_that_are_not_a_number_a_text_or_a_flag_are_refused(bad):
    facts = Facts()
    with pytest.raises(TypeError):
        facts["x"] = bad


def test_the_rules_also_hold_for_the_constructor_update_and_setdefault():
    assert Facts({"a": np.int64(2)}, b=3.5) == {"a": 2, "b": 3.5}
    facts = Facts({"a": 1})
    with pytest.raises(KeyError):
        facts.update({"a": 2})
    with pytest.raises(ValueError):
        facts.update({"b": float("nan")})
    with pytest.raises(ValueError):
        Facts({"c": float("inf")})
    assert facts.setdefault("a", 99) == 1                 # already there: untouched
    assert facts.setdefault("z", 5) == 5 and facts["z"] == 5
    with pytest.raises(TypeError):
        facts.setdefault("y", [1])


def test_save_writes_sorted_json_with_a_final_newline_and_keeps_accents(tmp_path):
    facts = Facts()
    facts["zebra"], facts["alpha"], facts["name"] = 1, 2.5, "Beyoncé"
    path = tmp_path / "new folder" / "facts.json"                 # the folder does not exist yet and has a space in its name
    facts.save(path)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert list(json.loads(text)) == ["alpha", "name", "zebra"]
    assert "Beyoncé" in text                                       # not written as é
    assert json.loads(text) == {"alpha": 2.5, "name": "Beyoncé", "zebra": 1}


def test_a_save_leaves_no_temporary_file_behind(tmp_path):
    facts = Facts({"a": 1})
    facts.save(tmp_path / "facts.json")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["facts.json"]


def test_a_save_that_fails_halfway_leaves_the_old_file_and_no_debris(tmp_path):
    target = tmp_path / "register.csv"
    target.write_text("old content", encoding="utf-8")

    def broken_writer(temporary):
        temporary.write_text("half a file", encoding="utf-8")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        _write_atomic(target, broken_writer)
    assert target.read_text(encoding="utf-8") == "old content"
    assert [path.name for path in tmp_path.iterdir()] == ["register.csv"]


# ================================================================================ Findings
def _findings():
    register = Findings()
    first = register.add("Ranking", "Figure 4", "Number one is held by a handful of songs", "5 songs hold 50 % of the days", "Counts are of snapshots")
    second = register.add("Artist", "Figure 16", "Ten artists hold 41 % of the points", "Gini 0.78")
    return register, first, second


def test_ids_run_from_f01_and_add_returns_the_new_id():
    register, first, second = _findings()
    assert (first, second) == ("F01", "F02") and len(register) == 2
    assert register.to_frame()["finding_id"].tolist() == ["F01", "F02"]


def test_the_frame_has_the_documented_columns_and_an_empty_caveat_is_allowed():
    register, _, _ = _findings()
    frame = register.to_frame()
    assert list(frame.columns) == FINDING_COLUMNS == ["finding_id", "block", "figure", "finding", "evidence", "caveat"]
    assert frame.loc[1, "caveat"] == ""


def test_an_empty_register_still_has_its_columns():
    frame = Findings().to_frame()
    assert list(frame.columns) == FINDING_COLUMNS and len(frame) == 0


@pytest.mark.parametrize("field", ["block", "figure", "finding", "evidence"])
@pytest.mark.parametrize("empty", ["", "   ", "\n", None])
def test_an_empty_field_is_refused_and_named(field, empty):
    arguments = {"block": "B", "figure": "F", "finding": "text", "evidence": "number"}
    arguments[field] = empty
    register = Findings()
    with pytest.raises(ValueError, match=field):
        register.add(**arguments)
    assert len(register) == 0                                       # nothing was half-added


def test_the_csv_starts_with_a_byte_order_mark_so_that_excel_on_windows_reads_the_accents(tmp_path):
    register = Findings()
    register.add("Artist", "Figure 15", "Beyoncé and Rosalía", "2 artists")
    path = tmp_path / "reports" / "findings.csv"
    register.save(path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert pd.read_csv(path, encoding="utf-8-sig").loc[0, "finding"] == "Beyoncé and Rosalía"


def test_commas_quotes_and_line_breaks_survive_a_round_trip(tmp_path):
    register = Findings()
    register.add("Ranking", "Figure 1", 'He said "stay", then left', "line one\nline two, with a comma", "caveat")
    path = tmp_path / "findings.csv"
    register.save(path)
    loaded = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    pd.testing.assert_frame_equal(loaded, register.to_frame())


def test_the_csv_save_leaves_no_temporary_file_behind(tmp_path):
    register, _, _ = _findings()
    register.save(tmp_path / "findings.csv")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["findings.csv"]
