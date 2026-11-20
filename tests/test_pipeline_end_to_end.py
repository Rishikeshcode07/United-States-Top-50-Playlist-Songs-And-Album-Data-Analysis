"""
tests/test_pipeline_end_to_end.py  -  the whole pipeline, from the CSV to the saved files and the command line.

WHAT THIS FILE IS
    (1) "Golden numbers": the real Atlantic file must always produce exactly the figures we verified by hand and in SQL
        (27,750 rows, 944 songs, 707,625 chart points, HHI 254.4 ...). If any of them moves, a rule changed.
    (2) Files: every promised output exists, CSV and Parquet agree, nothing is left half-written, the raw file is untouched.
    (3) Safety: a bad input or a failed quality gate publishes nothing and leaves the previous good files exactly as they were.
    (4) The command line: exit code 0 on success, 1 for a data problem with a readable message, 2 for a surprise.

WHY WE NEED IT
    Unit tests prove the parts. These tests prove the machine: that the parts are connected in the right order, that the
    stop-on-failure promise holds, and that running the pipeline twice gives the same answer.

WHAT HAPPENS WHEN IT RUNS
    The real file is processed once (session fixture). Failure scenarios use the small synthetic chart in temporary folders.

EXPECTED OUTCOME
    All tests pass. Tests on the real file are skipped when data/raw/Atlantic_United_States.csv is missing.

MOTIVE
    A pipeline you can trust is one that has been seen to succeed, to fail safely, and to repeat itself.
"""
import hashlib
import json
import os
import shutil

import pandas as pd
import pytest

from src.config import ProjectPaths
from src.etl import run_pipeline
from src.etl.errors import DataContractError, DataIntegrityError, PipelineError
from src.etl.run_pipeline import main, run
from tests import synthetic as syn

GOLDEN_SHA256 = "4b7b5871c98cde27a6a33d87be50b2473d305f87134ee4745d16cb66925a08d8"
GOLDEN_SHAPES = {"playlist_daily": (27_750, 54), "fact_chart_daily": (27_750, 21), "dim_track": (944, 47), "dim_artist": (283, 17),
                 "bridge_track_artist": (1_105, 4), "dim_date": (559, 15), "daily_summary": (555, 15)}
PROCESSED = ["playlist_daily", "fact_chart_daily", "dim_track", "dim_artist", "bridge_track_artist", "dim_date", "daily_summary"]
REPORTS = ["data_quality_scorecard.csv", "cleaning_log.csv", "quality_gate.csv", "data_dictionary.csv", "kpi_dictionary.csv",
           "track_identity_decisions.csv", "calendar_reality_check.csv", "last_run.json"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(folder):
    """{relative file name: sha256} of every file below a folder."""
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def make_project(tmp_path, raw_frame, name="project"):
    paths = ProjectPaths.under(tmp_path / name)
    paths.raw_file.parent.mkdir(parents=True)
    raw_frame.to_csv(paths.raw_file, index=False)
    return paths


# ================================================================================ golden numbers of the real file
def test_the_real_file_is_the_file_we_verified(real_result):
    assert real_result.source["sha256"] == GOLDEN_SHA256
    assert (real_result.source["size_bytes"], real_result.source["rows"]) == (3_802_835, 27_800)


def test_golden_table_shapes_and_checks(real_result):
    assert {name: t.shape for name, t in real_result.tables.items()} == GOLDEN_SHAPES
    assert real_result.summary["validation"] == {"PASS": 17, "WARN": 8, "INFO": 6, "FAIL": 0}
    assert real_result.summary["gate"] == {"passed": 25, "total": 25}
    assert real_result.status == "SUCCESS"


def test_golden_chart_numbers(real_result):
    t = real_result.tables
    fact, daily, dt = t["fact_chart_daily"], t["playlist_daily"], t["dim_track"]
    assert fact["chart_points"].sum() == 707_625 == 555 * 50 * 51 // 2
    assert daily["popularity_fill_method"].value_counts().to_dict() == {
        "original": 27_561, "interpolated_in_run": 168, "unfilled": 18, "song_median": 3}
    assert daily["movement"].value_counts().to_dict() == {
        "Down": 9_291, "Up": 8_293, "Same": 8_185, "Re-entry": 1_037, "New entry": 894, "Chart start": 50}
    assert dt["run_status"].value_counts().to_dict() == {
        "Complete history observed": 844, "Already on chart at start": 50, "Still on chart at end": 50}
    ds = t["daily_summary"]
    assert (int(ds["new_entries"].sum()), int(ds["re_entries"].sum()), int(ds["exits"].sum())) == (894, 1_037, 1_931)
    assert int(ds["is_high_turnover_day"].sum()) == 22
    assert round(float(fact["is_explicit"].mean()), 4) == 0.4795


def test_golden_song_and_artist_leaders(real_result):
    dt, da = real_result.tables["dim_track"], real_result.tables["dim_artist"]
    top = dt.sort_values(["days_on_chart", "track_id"], ascending=[False, True]).iloc[0]
    assert (top["track_label"], top["days_on_chart"], top["best_rank"]) == ("Something in the Orange by Zach Bryan", 536, 9)
    assert (da.iloc[0]["artist_name"], da.iloc[0]["dominance_index_primary"]) == ("Taylor Swift", 7.5312)
    shares = da["chart_points_primary"] / 707_625 * 100
    assert float((shares ** 2).sum()) == pytest.approx(254.427, abs=0.001)           # HHI: far below 1,500 = not concentrated


def test_golden_identity_decisions_and_calendar(real_result):
    decisions = real_result.identity_decisions["verdict"].value_counts().to_dict()
    assert decisions == {"same song": 29, "different song": 6}
    reality = real_result.calendar_reality
    assert len(reality) == 20 and abs(reality["gap_days"].median() - 367) <= 1       # the file's dates run about a year after real releases
    log = real_result.cleaning_log.set_index("step")
    assert (log.loc["2.1", "rows_affected"], log.loc["2.4", "rows_affected"], log.loc["2.5", "rows_affected"]) == (50, 1_137, 189)
    assert "one year ahead" in log.loc["1.6", "rule"] and "about 367 days" in log.loc["1.6", "note"]    # the offset is reported, not hidden


def test_the_pipeline_is_repeatable(real_result, real_paths):
    again = run(real_paths, load_db=False, write_files=False)
    for name, table in real_result.tables.items():
        pd.testing.assert_frame_equal(again.tables[name], table, obj=name)
    pd.testing.assert_frame_equal(again.scorecard, real_result.scorecard)
    pd.testing.assert_frame_equal(again.gate, real_result.gate)


# ================================================================================ files
def test_every_promised_file_exists_and_is_not_empty(real_result, real_paths):
    expected = ([real_paths.interim_dir / n for n in ["playlist_validated.parquet", "playlist_validated.csv", "playlist_clean.parquet"]]
                + [real_paths.processed_dir / f"{n}.{ext}" for n in PROCESSED for ext in ("parquet", "csv")]
                + [real_paths.reports_dir / n for n in REPORTS])
    assert len(expected) == 25
    for path in expected:
        assert path.is_file() and path.stat().st_size > 0, path.name
    assert len(real_result.written_files) == 24                                      # last_run.json is written at the very end
    assert not list(real_paths.root.rglob("*.tmp"))                                  # no half-written temporary files left behind


def test_csv_and_parquet_hold_the_same_rows(real_paths):
    for name in PROCESSED:
        parquet = pd.read_parquet(real_paths.processed_dir / f"{name}.parquet")
        csv = pd.read_csv(real_paths.processed_dir / f"{name}.csv", encoding="utf-8-sig")
        assert csv.shape == parquet.shape == GOLDEN_SHAPES[name], name
        assert list(csv.columns) == list(parquet.columns), name


def test_the_raw_file_is_never_modified(real_result, real_csv):
    assert sha(real_csv) == GOLDEN_SHA256


def test_last_run_json_and_the_log_file_tell_what_happened(real_result, real_paths):
    last = json.loads((real_paths.reports_dir / "last_run.json").read_text(encoding="utf-8"))
    assert last["status"] == "SUCCESS" and last["gate"] == {"passed": 25, "total": 25}
    assert last["source"]["sha256"] == GOLDEN_SHA256 and last["tables"]["dim_track"] == {"rows": 944, "columns": 47}
    log_text = (real_paths.logs_dir / "pipeline.log").read_text(encoding="utf-8")
    assert "PIPELINE FINISHED: SUCCESS" in log_text and "[9/9]" in log_text
    assert not (real_paths.warehouse_dir).exists()                                   # no database was asked for


def test_the_scorecard_and_gate_files_are_readable_reports(real_paths):
    card = pd.read_csv(real_paths.reports_dir / "data_quality_scorecard.csv")
    gate = pd.read_csv(real_paths.reports_dir / "quality_gate.csv", encoding="utf-8-sig")
    assert len(card) == 31 and set(card["status"]) == {"PASS", "WARN", "INFO"}
    assert len(gate) == 25 and (gate["status"] == "PASS").all()


# ================================================================================ stopping safely
@pytest.fixture()
def good_project(tmp_path, plan):
    """A synthetic project that has been processed successfully once (all output files exist)."""
    paths = make_project(tmp_path, syn.raw_frame(plan))
    run(paths, load_db=False)
    return paths


def damage_the_raw_file(paths, plan):
    damaged = syn.raw_frame(plan)
    damaged.loc[0, "position"] = 51
    damaged.to_csv(paths.raw_file, index=False)


def test_bad_input_publishes_nothing_and_keeps_the_last_good_files(good_project, plan):
    paths = good_project
    before = {"processed": snapshot(paths.processed_dir), "interim": snapshot(paths.interim_dir)}
    damage_the_raw_file(paths, plan)
    with pytest.raises(DataContractError, match="Rank inside 1-50"):
        run(paths, load_db=False)
    assert snapshot(paths.processed_dir) == before["processed"]                      # byte for byte the same
    assert snapshot(paths.interim_dir) == before["interim"]
    last = json.loads((paths.reports_dir / "last_run.json").read_text(encoding="utf-8"))
    assert last["status"] == "FAILED" and "Rank inside 1-50" in last["message"]
    card = pd.read_csv(paths.reports_dir / "data_quality_scorecard.csv")             # the evidence is saved for diagnosis
    assert (card["status"] == "FAIL").any()
    assert not list(paths.root.rglob("*.tmp"))


def test_a_failed_quality_gate_publishes_nothing(good_project, monkeypatch):
    paths = good_project
    before = snapshot(paths.processed_dir)
    real_build_all = run_pipeline.build_all

    def damaged_build_all(clean, clog):
        tables = real_build_all(clean, clog)
        tables["fact_chart_daily"]["chart_points"] = tables["fact_chart_daily"]["chart_points"] + 1     # silently wrong KPI table
        return tables

    monkeypatch.setattr(run_pipeline, "build_all", damaged_build_all)
    with pytest.raises(DataIntegrityError, match="quality gate failed"):
        run(paths, load_db=False)
    assert snapshot(paths.processed_dir) == before
    gate = pd.read_csv(paths.reports_dir / "quality_gate.csv", encoding="utf-8-sig")
    assert "FAIL" in set(gate["status"]) and "G25" in set(gate.loc[gate["status"] == "FAIL", "id"])


def test_a_locked_output_file_gives_a_friendly_message(tmp_path, plan, monkeypatch):
    paths = make_project(tmp_path, syn.raw_frame(plan))

    def locked(source, target):
        raise PermissionError("file is open in Excel")

    monkeypatch.setattr(os, "replace", locked)
    with pytest.raises(PipelineError, match="open in Excel or Power BI"):
        run(paths, load_db=False)
    assert not list(paths.root.rglob("*.tmp"))


def test_an_extra_column_in_the_raw_file_is_reported_and_ignored(tmp_path, plan):
    frame = syn.raw_frame(plan)
    frame["internal_note"] = "x"
    result = run(make_project(tmp_path, frame), load_db=False, write_files=False)
    assert result.status == "SUCCESS"
    assert result.scorecard.set_index("check").loc["No unexpected extra column", "status"] == "WARN"
    assert all("internal_note" not in t.columns for t in result.tables.values())


def test_a_missing_raw_file_is_explained(tmp_path):
    paths = ProjectPaths.under(tmp_path / "empty project")
    with pytest.raises(DataContractError, match="Copy Atlantic_United_States.csv"):
        run(paths, load_db=False, write_files=False)


# ================================================================================ the command line
def test_cli_success_with_the_default_sqlite_database(tmp_path, plan, capsys):
    paths = make_project(tmp_path, syn.raw_frame(plan), name="cli project")
    assert main(["--project-root", str(paths.root)]) == 0
    assert paths.sqlite_file.is_file() and paths.sqlite_file.stat().st_size > 0
    assert (paths.reports_dir / "last_run.json").is_file()


def test_cli_no_db_writes_files_but_no_database(tmp_path, plan):
    paths = make_project(tmp_path, syn.raw_frame(plan))
    assert main(["--project-root", str(paths.root), "--no-db"]) == 0
    assert (paths.processed_dir / "dim_track.parquet").is_file() and not paths.warehouse_dir.exists()


def test_cli_exit_code_1_and_a_readable_message_for_bad_data(tmp_path, plan, capsys):
    paths = make_project(tmp_path, syn.raw_frame(plan))
    damage_the_raw_file(paths, plan)
    assert main(["--project-root", str(paths.root), "--no-db"]) == 1
    err = capsys.readouterr().err
    assert "The pipeline stopped" in err and "Rank inside 1-50" in err and "logs/pipeline.log" in err


def test_cli_exit_code_1_for_a_missing_file(tmp_path, capsys):
    assert main(["--project-root", str(tmp_path / "nothing here"), "--no-db"]) == 1
    assert "Raw file not found" in capsys.readouterr().err


def test_cli_input_option_reads_another_file(tmp_path, plan):
    paths = ProjectPaths.under(tmp_path / "project")
    elsewhere = tmp_path / "downloads" / "my copy.csv"
    elsewhere.parent.mkdir()
    syn.raw_frame(plan).to_csv(elsewhere, index=False)
    assert main(["--project-root", str(paths.root), "--input", str(elsewhere), "--no-db"]) == 0
    assert (paths.processed_dir / "fact_chart_daily.csv").is_file()


def test_cli_exit_code_2_for_an_unexpected_error(tmp_path, plan, monkeypatch, capsys):
    paths = make_project(tmp_path, syn.raw_frame(plan))

    def boom(*args, **kwargs):
        raise RuntimeError("something nobody expected")

    monkeypatch.setattr(run_pipeline, "run", boom)
    assert main(["--project-root", str(paths.root), "--no-db"]) == 2
    assert "Unexpected error: RuntimeError" in capsys.readouterr().err


def test_cli_rejects_unknown_options():
    with pytest.raises(SystemExit) as stop:
        main(["--s3"])
    assert stop.value.code == 2
