"""
tests/test_dash_load.py  -  the dashboard's loader: reading the tables, and failing politely when they are missing.

WHAT THIS FILE IS
    Tests of src/dash/load.py. They save the synthetic tables into a temporary folder (whose name contains a space, like a Windows
    "My Projects" folder) and point the loader at it.

WHY WE NEED IT
    A first run without data/processed/ is the most likely way for the dashboard to fail. The promise is a calm message that names
    the command to run, never a stack trace, and a missing result file must never stop the pages that do not need it.

WHAT HAPPENS WHEN IT RUNS
    Everything is written to pytest's temporary folders; the project's own data/ and reports/ folders are not touched.

EXPECTED OUTCOME
    All tests pass without the real data.

MOTIVE
    Make "the files are not there yet" an ordinary, tested situation.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.dash import load


@pytest.fixture()
def processed_folder(tmp_path, synthetic_tables):
    folder = tmp_path / "my project" / "data processed"
    folder.mkdir(parents=True)
    for stem in load.TABLE_FILES.values():
        synthetic_tables[stem].to_parquet(folder / f"{stem}.parquet")
    return folder


def test_load_tables_reads_the_six_tables(processed_folder, synthetic_tables):
    tables = load.load_tables(processed_folder)
    assert len(tables.playlist) == len(synthetic_tables["playlist_daily"])
    assert len(tables.dim_track) == len(synthetic_tables["dim_track"])
    assert len(tables.bridge) == len(synthetic_tables["bridge_track_artist"])
    assert set(tables.playlist.columns) >= {"date", "position", "track_id", "chart_points", "popularity_clean"}


def test_playlist_comes_back_sorted_by_date_and_place(processed_folder):
    playlist = load.load_tables(processed_folder).playlist
    ordered = playlist.sort_values(["date", "position"]).reset_index(drop=True)
    assert playlist[["date", "position"]].equals(ordered[["date", "position"]])


def test_missing_tables_lists_exactly_the_missing_files(processed_folder):
    assert load.missing_tables(processed_folder) == []
    (processed_folder / "dim_artist.parquet").unlink()
    (processed_folder / "daily_summary.parquet").unlink()
    assert sorted(load.missing_tables(processed_folder)) == ["daily_summary", "dim_artist"]


def test_data_not_ready_names_the_files_and_the_command(processed_folder):
    (processed_folder / "dim_track.parquet").unlink()
    with pytest.raises(load.DataNotReady) as problem:
        load.load_tables(processed_folder)
    message = str(problem.value)
    assert "dim_track.parquet" in message
    assert "python -m src.etl.run_pipeline" in message
    assert problem.value.missing == ["dim_track"]


def test_an_empty_folder_reports_all_six_tables(tmp_path):
    with pytest.raises(load.DataNotReady) as problem:
        load.load_tables(tmp_path)
    assert len(problem.value.missing) == 6


def test_chart_bounds(processed_folder):
    tables = load.load_tables(processed_folder)
    bounds = load.chart_bounds(tables.playlist)
    assert bounds.first_date == tables.playlist["date"].min()
    assert bounds.last_date == tables.playlist["date"].max()
    assert bounds.first_snapshot == 1
    assert bounds.last_snapshot == bounds.snapshots == tables.playlist["date"].nunique()


def test_read_report_returns_none_when_the_file_is_missing(tmp_path):
    assert load.read_report("p6b_stays", tmp_path) is None


def test_read_report_parses_dates_and_ignores_unknown_date_columns(tmp_path):
    pd.DataFrame({"entry_date": ["2024-05-18", "2024-05-19"], "x": [1, 2]}).to_csv(tmp_path / "demo.csv", index=False)
    table = load.read_report("demo", tmp_path, dates=("entry_date", "not_a_column"))
    assert str(table["entry_date"].dtype).startswith("datetime64")
    assert table["x"].tolist() == [1, 2]


def test_read_report_survives_a_broken_file(tmp_path):
    (tmp_path / "broken.csv").write_bytes(b"\x00\x01\x02")
    (tmp_path / "ragged.csv").write_text('a,b\n1,2,3,4\n"unclosed,5\n', encoding="utf-8")
    assert load.read_report("broken", tmp_path) is None or isinstance(load.read_report("broken", tmp_path), pd.DataFrame)
    assert load.read_report("ragged", tmp_path) is None or isinstance(load.read_report("ragged", tmp_path), pd.DataFrame)


def test_read_facts(tmp_path):
    assert load.read_facts("p6c_facts", tmp_path) == {}
    (tmp_path / "p6c_facts.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert load.read_facts("p6c_facts", tmp_path) == {"a": 1}
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert load.read_facts("bad", tmp_path) == {}


def test_reports_available(tmp_path):
    (tmp_path / "one.csv").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "two.json").write_text("{}", encoding="utf-8")
    assert load.reports_available(["one", "two", "three"], tmp_path) == {"one": True, "two": True, "three": False}


def test_the_real_tables_have_the_documented_shapes(dash_real):
    assert dash_real.playlist.shape == (27_750, 54)
    assert dash_real.dim_track.shape == (944, 47)
    assert dash_real.daily_summary.shape == (555, 15)
    assert dash_real.bridge.shape == (1_105, 4)
