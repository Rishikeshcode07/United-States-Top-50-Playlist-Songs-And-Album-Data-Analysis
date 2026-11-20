"""
tests/test_inference_register.py  -  checks of the hypothesis register, the effect-size words and the result record.

WHAT THIS FILE IS
    Automatic tests of src/inference/register.py and src/inference/results.py: the Holm and Benjamini-Hochberg columns and the
    verdict rule, the refusal of incomplete rows, the CSV file, the way p-values are written, and the effect-size benchmarks.

WHY WE NEED IT
    The register is the file that the dashboards and the research paper read. A wrong verdict there would travel into every
    later phase.

WHAT HAPPENS WHEN IT RUNS
    pytest writes a CSV file into a temporary folder and reads it back.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    One table that can be trusted.
"""
import pandas as pd
import pytest

from src.inference.register import COLUMNS, HypothesisRegister, format_p, save_table, verdict
from src.inference.results import EFFECT_BENCHMARKS, InferenceResult, effect_label


def _row(name, primary_p, planned_p=0.01):
    return dict(hypothesis=name, question="A question?", unit="song", n=100, planned_test="chi-square", planned_p=planned_p,
                primary_test="cluster-robust z", primary_p=primary_p, effect=0.07, effect_low=0.01, effect_high=0.13,
                effect_unit="percentage points", standard_metric="cohens_h", standard_size=0.15, size_word="small", caveat="none")


def test_the_table_holds_holm_and_bh_columns_and_the_verdicts_of_the_rule():
    register = HypothesisRegister()
    for name, p in [("H1", 1e-5), ("H2", 0.02), ("H3", 0.04), ("H4", 0.5)]:
        register.add(**_row(name, p))
    table = register.to_frame()
    assert list(table.columns) == COLUMNS
    assert table["holm_p"].tolist() == pytest.approx([4e-5, 0.06, 0.08, 0.5])
    assert table["bh_q"].tolist() == pytest.approx([4e-5, 0.04, 0.0533333, 0.5], abs=1e-6)
    assert table["verdict"].tolist() == ["Supported", "Suggestive", "Suggestive", "Not supported"]


def test_the_verdict_rule_directly():
    assert verdict(0.001, 0.01) == "Supported"
    assert verdict(0.04, 0.2) == "Suggestive"
    assert verdict(0.2, 0.9) == "Not supported"
    assert verdict(0.05, 0.05) == "Not supported"


def test_incomplete_duplicate_and_odd_rows_are_refused():
    register = HypothesisRegister()
    register.add(**_row("H1", 0.1))
    with pytest.raises(ValueError):
        register.add(**_row("H1", 0.2))                                    # the same hypothesis twice
    missing = _row("H2", 0.1)
    missing["caveat"] = "  "
    with pytest.raises(ValueError):
        register.add(**missing)
    unknown = _row("H3", 0.1)
    unknown["colour"] = "blue"
    with pytest.raises(ValueError):
        register.add(**unknown)
    with pytest.raises(ValueError):
        register.add(**_row("H4", 1.5))
    assert len(register) == 1


def test_an_empty_register_gives_an_empty_table_with_the_columns():
    assert list(HypothesisRegister().to_frame().columns) == COLUMNS


def test_save_writes_a_csv_with_a_byte_order_mark_and_leaves_no_temporary_file(tmp_path):
    register = HypothesisRegister()
    register.add(**_row("H1", 0.003))
    path = tmp_path / "reports" / "register.csv"
    register.save(path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    back = pd.read_csv(path, encoding="utf-8-sig")
    assert list(back.columns) == COLUMNS and back["verdict"].iloc[0] == "Supported"
    assert [p.name for p in path.parent.iterdir()] == ["register.csv"]


def test_save_table_writes_any_table_atomically_and_creates_the_folder(tmp_path):
    frame = pd.DataFrame({"claim": ["H1", "H2"], "text": ["caf\u00e9", "plain"], "value": [1.5, float("nan")]})
    path = tmp_path / "new folder" / "robustness.csv"
    save_table(frame, path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    back = pd.read_csv(path, encoding="utf-8-sig")
    assert list(back["text"]) == ["caf\u00e9", "plain"] and back["value"].isna().tolist() == [False, True]
    assert [p.name for p in path.parent.iterdir()] == ["robustness.csv"]
    save_table(frame.iloc[:1], path)                                                              # writing again replaces the file in one step
    assert len(pd.read_csv(path, encoding="utf-8-sig")) == 1


def test_save_table_keeps_the_old_file_and_leaves_no_temporary_file_when_the_move_fails(tmp_path, monkeypatch):
    path = tmp_path / "out.csv"
    save_table(pd.DataFrame({"a": [1]}), path)
    before = path.read_bytes()

    def broken_replace(source, target):
        raise OSError("disk full")
    monkeypatch.setattr("src.inference.register.os.replace", broken_replace)
    with pytest.raises(OSError):
        save_table(pd.DataFrame({"a": [2, 3]}), path)                                             # the temporary file is written, then the move fails
    assert path.read_bytes() == before and [p.name for p in tmp_path.iterdir()] == ["out.csv"]


def test_p_values_are_written_like_a_paper():
    assert format_p(0.00004) == "p < 0.001"
    assert format_p(0.034) == "p = 0.034"
    assert format_p(0.42) == "p = 0.42"
    assert format_p(0.0999) == "p = 0.100"


def test_effect_size_words_follow_the_benchmarks():
    assert effect_label("cohens_h", 0.1) == "negligible"
    assert effect_label("cohens_h", -0.3) == "small"
    assert effect_label("cohens_h", 0.5) == "medium"
    assert effect_label("cohens_h", 0.9) == "large"
    assert effect_label("cliffs_delta", 0.2) == "small" and effect_label("cliffs_delta", 0.5) == "large"
    assert effect_label("eta_squared", 0.07) == "medium"
    assert effect_label("cramers_v", 0.05) == "negligible"
    for metric, thresholds in EFFECT_BENCHMARKS.items():
        assert list(thresholds) == sorted(thresholds)
    with pytest.raises(ValueError):
        effect_label("vibes", 0.5)


def test_the_result_record_prints_and_converts():
    result = InferenceResult("demo test", 2.5, 0.0123, df=10, estimate=0.5, ci_low=0.1, ci_high=0.9, n=50, detail={"extra": 1})
    assert "demo test" in str(result) and "p = 0.0123" in str(result) and "[0.1, 0.9]" in str(result)
    row = result.as_dict()
    assert row["p_value"] == 0.0123 and row["extra"] == 1
    with pytest.raises(Exception):
        result.p_value = 0.5                                              # the record is frozen
