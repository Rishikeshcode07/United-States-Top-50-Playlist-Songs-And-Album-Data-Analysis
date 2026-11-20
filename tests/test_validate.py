"""
tests/test_validate.py  -  tests for the data-quality scorecard (step 2 of the pipeline).

WHAT THIS FILE IS
    Two groups of tests. (1) On the hand-made chart: the scorecard must show exactly the problems we planted, and nothing
    else. (2) On damaged copies of that chart: every rule that is supposed to STOP the pipeline must really stop it.

WHY WE NEED IT
    A scorecard that never fails is useless, and a scorecard that fails for no reason is ignored. Both directions are tested:
    the planted problems must appear as WARN, and each "this can not continue" rule must raise a DataContractError.

WHAT HAPPENS WHEN IT RUNS
    The raw table of the synthetic chart is damaged in ONE way per test (a rank of 51, a text in a number column ...),
    validated, and the scorecard is inspected.

EXPECTED OUTCOME
    All tests pass. The synthetic scorecard has 23 PASS, 7 WARN, 1 INFO and 0 FAIL.

MOTIVE
    Prove that the safety net really catches what it promises to catch.
"""
import pandas as pd
import pytest

from src.etl.errors import DataContractError
from src.etl.validate import SCORECARD_COLUMNS, QualityLog, validate_raw
from tests import synthetic as syn

# what the planted problems of the hand-made chart look like in the scorecard: {check: (affected, total)}
PLANTED_WARNINGS = {
    "No missing calendar days": (1, 12),                 # 2024-01-06 has no snapshot
    "Exactly 50 rows per date": (1, 11),                 # 2024-01-09 has 100 rows (the stale block + the real one)
    "No rank repeated within a date": (50, 600),         # the stale block repeats ranks 1-50
    "No song repeated within a date": (47, 600),         # 48 songs are in both blocks; one is spelt differently (curly quote)
    "album_type is Single or Album": (12, 600),          # song 33 is a "compilation" (11 days + 1 stale row)
    "No disguised-missing popularity (0)": (8, 600),     # 3 + 5 zeros
    "A song keeps one artist credit over time": (1, 72), # song 12 is credited "A & B" for 3 days and "A" afterwards
}


@pytest.fixture()
def raw(plan):
    """The raw table of the synthetic chart, in the shape read_raw would give (text dates, plain columns). A fresh copy per test."""
    return syn.raw_frame(plan)


def validate(frame):
    qlog = QualityLog()
    typed = validate_raw(frame, qlog)
    return typed, qlog


# ================================================================================ the scorecard of the hand-made chart
def test_planted_problems_appear_as_warnings(synthetic_result):
    card = synthetic_result.scorecard.set_index("check")
    for check, (affected, total) in PLANTED_WARNINGS.items():
        assert card.loc[check, "status"] == "WARN", check
        assert (int(card.loc[check, "affected"]), int(card.loc[check, "total"])) == (affected, total), check


def test_nothing_else_is_a_warning_or_a_failure(synthetic_result):
    card = synthetic_result.scorecard
    others = card[~card["check"].isin(PLANTED_WARNINGS)]
    assert set(others["status"]) <= {"PASS", "INFO"}
    assert list(others.loc[others["status"] == "INFO", "check"]) == ["Titles with several artist credits"]


def test_the_two_titles_with_several_credits_are_song_12_and_angel(synthetic_result):
    row = synthetic_result.scorecard.set_index("check").loc["Titles with several artist credits"]
    assert (int(row["affected"]), int(row["total"])) == (2, 70)
    verdicts = synthetic_result.identity_decisions.set_index("song")["verdict"].to_dict()
    assert verdicts == {"Angel": "different song", "Song 12": "same song"}


def test_summary_counts_of_the_synthetic_chart(synthetic_result):
    assert synthetic_result.summary["validation"] == {"PASS": 23, "WARN": 7, "INFO": 1, "FAIL": 0}
    counts = synthetic_result.scorecard["status"].value_counts().to_dict()
    assert counts == {"PASS": 23, "WARN": 7, "INFO": 1}


def test_scorecard_has_the_documented_columns_and_is_sorted_worst_first(synthetic_result):
    card = synthetic_result.scorecard
    assert list(card.columns) == SCORECARD_COLUMNS
    order = {"FAIL": 0, "WARN": 1, "INFO": 2, "PASS": 3}
    ranks = card["status"].map(order).tolist()
    assert ranks == sorted(ranks)


# ================================================================================ what must STOP the pipeline
def test_a_missing_column_stops_at_once_and_names_the_column(raw):
    with pytest.raises(DataContractError, match="position"):
        validate(raw.drop(columns=["position"]))


def test_dates_in_another_format_stop_the_run(raw):
    raw["date"] = pd.to_datetime(raw["date"], format="%d-%m-%Y").dt.strftime("%Y-%m-%d")      # year-month-day instead
    with pytest.raises(DataContractError, match="Dates parse with DD-MM-YYYY"):
        validate(raw)


def test_text_in_a_number_column_stops_the_run(raw):
    raw["position"] = raw["position"].astype(object)
    raw.loc[5, "position"] = "abc"
    with pytest.raises(DataContractError, match="position is numeric"):
        validate(raw)


def test_dates_are_read_day_first(raw):
    one_row = raw.head(1).copy()
    one_row["date"] = "03-04-2025"                                       # 3 April 2025, NOT 4 March
    typed, _ = validate(one_row)
    assert typed["date"].iloc[0] == pd.Timestamp("2025-04-03")


# (column, row, bad value, the check that must FAIL)
DAMAGE = [
    ("position", 0, 51, "Rank inside 1-50"),
    ("position", 0, 0, "Rank inside 1-50"),
    ("position", 0, 2.5, "Rank is a whole number"),
    ("popularity", 0, 101, "Popularity inside 0-100"),
    ("popularity", 0, -1, "Popularity inside 0-100"),
    ("duration_ms", 0, 5_000, "Duration between 30 s and 20 min"),
    ("duration_ms", 0, 99_999_999, "Duration between 30 s and 20 min"),
    ("total_tracks", 0, 0, "total_tracks is a whole number >= 1"),
    ("song", 0, "   ", "No blank or placeholder text"),
    ("artist", 0, "N/A", "No blank or placeholder text"),
    ("artist", 0, None, "No missing values (NaN) in any column"),
    ("is_explicit", 0, "maybe", "is_explicit is True/False"),
]


@pytest.mark.parametrize("column, row, value, failing_check", DAMAGE)
def test_each_impossible_value_is_a_failure(raw, column, row, value, failing_check):
    raw[column] = raw[column].astype(object)
    raw.loc[row, column] = value
    _, qlog = validate(raw)
    assert failing_check in {r["check"] for r in qlog.failures()}
    with pytest.raises(DataContractError, match="FAILED"):
        qlog.raise_if_failed()


def test_a_date_with_a_missing_rank_is_a_failure(raw):
    raw = raw.drop(index=0).reset_index(drop=True)                       # one row less on the first date
    _, qlog = validate(raw)
    assert "Every date has all ranks 1-50" in {r["check"] for r in qlog.failures()}


def test_an_extra_column_is_only_a_warning(raw):
    raw["extra_note"] = "x"
    _, qlog = validate(raw)
    row = qlog.to_frame().set_index("check").loc["No unexpected extra column"]
    assert row["status"] == "WARN" and row["affected"] == 1
    assert qlog.failures() == []


def test_a_https_link_is_required_for_the_cover_url(raw):
    raw.loc[0, "album_cover_url"] = "not a link"
    _, qlog = validate(raw)
    assert qlog.to_frame().set_index("check").loc["album_cover_url looks like a link", "status"] == "WARN"


def test_validation_never_changes_the_table_it_receives(raw):
    before = raw.copy(deep=True)
    validate(raw)
    pd.testing.assert_frame_equal(raw, before)


# ================================================================================ the QualityLog itself
def test_a_check_with_nothing_affected_passes_whatever_its_severity():
    qlog = QualityLog()
    assert qlog.add("Validity", "c1", "rule", 0, 10, severity="FAIL") == "PASS"
    assert qlog.add("Validity", "c2", "rule", 3, 10, severity="WARN") == "WARN"
    assert qlog.add("Validity", "c3", "rule", 3, 10, severity="INFO") == "INFO"
    assert qlog.add("Validity", "c4", "rule", 3, 10, severity="FAIL") == "FAIL"


def test_adding_a_check_twice_keeps_only_the_latest_result():
    qlog = QualityLog()
    qlog.add("Validity", "same check", "rule", 5, 10, severity="WARN")
    qlog.add("Validity", "same check", "rule", 0, 10, severity="WARN")
    card = qlog.to_frame()
    assert len(card) == 1 and card.loc[0, "status"] == "PASS"


def test_percentages_and_an_empty_total():
    qlog = QualityLog()
    qlog.add("Validity", "a third", "rule", 1, 3, severity="WARN")
    qlog.add("Validity", "nothing to check", "rule", 0, 0, severity="WARN")
    card = qlog.to_frame().set_index("check")
    assert card.loc["a third", "pct_affected"] == 33.33
    assert card.loc["nothing to check", "pct_affected"] == 0.0


def test_summary_always_lists_the_four_statuses():
    assert QualityLog().summary() == {"PASS": 0, "WARN": 0, "INFO": 0, "FAIL": 0}


def test_the_failure_message_names_every_failed_check():
    qlog = QualityLog()
    qlog.add("Validity", "first rule", "rule", 2, 10, severity="FAIL", action="fix the first thing")
    qlog.add("Validity", "second rule", "rule", 1, 10, severity="FAIL", action="fix the second thing")
    qlog.add("Validity", "third rule", "rule", 1, 10, severity="WARN")
    with pytest.raises(DataContractError) as error:
        qlog.raise_if_failed("(stage x)")
    message = str(error.value)
    assert "2 data-quality check(s) FAILED (stage x)" in message
    assert "first rule" in message and "second rule" in message and "third rule" not in message
    assert "fix the first thing" in message


def test_no_failure_means_no_exception():
    qlog = QualityLog()
    qlog.add("Validity", "fine", "rule", 0, 10, severity="FAIL")
    qlog.raise_if_failed()                                              # must not raise
