"""
tests/conftest.py  -  the shared building blocks ("fixtures") of the whole test suite.

WHAT THIS FILE IS
    pytest reads this file automatically. It defines the objects that many tests need, so that each test file can simply
    ask for them by name:

        plan, synthetic_paths, synthetic_result   the small hand-made chart (tests/synthetic.py) and the pipeline run on it
        synthetic_tables / tables                 its seven output tables (`tables` is a private copy a test may damage)
        real_csv, real_paths, real_result         the real Atlantic file and one full pipeline run on it
        sqlite_engine, real_sqlite                an empty SQLite database; one holding the real data
        postgres_engine                           an empty PostgreSQL schema (only when TEST_POSTGRES_URL is set)

WHY WE NEED IT
    Running the pipeline takes a few seconds. With "session" fixtures it runs ONCE and every test reuses the result.
    Everything is written into temporary folders, so testing never touches your real data/ folders or your real database.

WHAT HAPPENS WHEN IT RUNS
    Fixtures are created the first time a test asks for them and removed at the end. Tests that need the real CSV are
    skipped (not failed) when data/raw/Atlantic_United_States.csv is missing; tests that need PostgreSQL are skipped when
    the environment variable TEST_POSTGRES_URL is not set. Both kinds get a label (marker) automatically, so you can run
    only the fast tests with:   pytest -m "not realdata and not postgres"

EXPECTED OUTCOME
    Nothing to see here. A broken fixture shows up as an ERROR (not a failure) in the pytest report.

MOTIVE
    Tests should be quick to run, safe to run and easy to read.
"""
from __future__ import annotations

import os
import uuid

import pytest
import sqlalchemy as sa

from src.config import RAW_FILE, SQL_DIR, ProjectPaths
from src.etl.load import DbSettings, build_engine, load_warehouse
from src.etl.run_pipeline import run
from tests import synthetic as syn
from tests.helpers import copy_tables


@pytest.fixture(autouse=True)
def isolate_database_settings(monkeypatch):
    """Tests never read your real database settings: without this, a DB_TYPE=postgresql in your .env file would be used."""
    monkeypatch.setenv("DB_TYPE", "sqlite")


def pytest_collection_modifyitems(items):
    """Label tests automatically: 'realdata' needs the real CSV, 'postgres' needs a PostgreSQL server."""
    for item in items:
        names = set(getattr(item, "fixturenames", []))
        if "real_csv" in names:
            item.add_marker(pytest.mark.realdata)
        if "postgres_engine" in names:
            item.add_marker(pytest.mark.postgres)


# ====================================================================================== the hand-made chart
@pytest.fixture(scope="session")
def plan():
    """The hand-designed chart (11 snapshot dates + one stale block = 600 raw rows) and its true answers."""
    return syn.build_plan()


@pytest.fixture(scope="session")
def synthetic_paths(tmp_path_factory, plan):
    """A throw-away project folder that holds the synthetic CSV. The folder name has a space on purpose (like many Windows paths)."""
    root = tmp_path_factory.mktemp("synthetic project")
    paths = ProjectPaths.under(root)
    paths.raw_file.parent.mkdir(parents=True)
    syn.write_csv(plan, paths.raw_file)
    return paths


@pytest.fixture(scope="session")
def synthetic_result(synthetic_paths):
    """The pipeline run on the synthetic chart, in memory only (no files, no database)."""
    return run(synthetic_paths, load_db=False, write_files=False)


@pytest.fixture(scope="session")
def synthetic_tables(synthetic_result):
    """The seven output tables of that run. Shared by many tests: READ them, never change them."""
    return synthetic_result.tables


@pytest.fixture()
def tables(synthetic_tables):
    """A private copy of the seven tables, for a test that damages them on purpose."""
    return copy_tables(synthetic_tables)


# ====================================================================================== the real file
@pytest.fixture(scope="session")
def real_csv():
    """The real Atlantic CSV, or a skip (not a failure) when it is not in data/raw/."""
    if not RAW_FILE.exists():
        pytest.skip(f"{RAW_FILE} is not there, so the tests on the real data are skipped")
    return RAW_FILE


@pytest.fixture(scope="session")
def real_paths(tmp_path_factory, real_csv):
    """Throw-away output folders for the real file. The CSV itself is read from its normal place and never modified."""
    root = tmp_path_factory.mktemp("real project")
    return ProjectPaths.under(root, raw_file=real_csv)


@pytest.fixture(scope="session")
def real_result(real_paths):
    """One full pipeline run on the real file: files are written into the temporary folders, no database."""
    return run(real_paths, load_db=False, write_files=True)


# ====================================================================================== databases
@pytest.fixture()
def sqlite_engine(tmp_path):
    """An empty SQLite database in a temporary folder whose name contains a space."""
    engine = build_engine(DbSettings(db_type="sqlite"), tmp_path / "ware house" / "test.db")
    yield engine
    engine.dispose()                                   # releases the file (Windows can not delete an open file)


@pytest.fixture(scope="session")
def real_sqlite(tmp_path_factory, real_result):
    """A SQLite database that already holds the real data (loaded once, then only read by the tests)."""
    folder = tmp_path_factory.mktemp("sqlite real")
    engine = build_engine(DbSettings(db_type="sqlite"), folder / "real.db")
    load_warehouse(engine, real_result.tables, SQL_DIR)
    yield engine
    engine.dispose()


@pytest.fixture()
def postgres_engine():
    """
    An engine on a brand-new, empty PostgreSQL schema. Needs TEST_POSTGRES_URL, for example
        postgresql+psycopg2://user:password@localhost:5432/testdb
    The schema is created for the test and dropped afterwards, so nothing else in that database is touched.
    """
    url = os.environ.get("TEST_POSTGRES_URL", "").strip()
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not set, so the PostgreSQL tests are skipped")
    schema = "t_" + uuid.uuid4().hex[:12]
    admin = sa.create_engine(url)
    with admin.begin() as connection:
        connection.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
    engine = sa.create_engine(url, connect_args={"options": f"-csearch_path={schema}"}, pool_pre_ping=True)
    yield engine
    engine.dispose()
    with admin.begin() as connection:
        connection.execute(sa.text(f'DROP SCHEMA "{schema}" CASCADE'))
    admin.dispose()


# ====================================================================================== the dashboard (Phase 7)
@pytest.fixture(scope="session")
def dash_synthetic(synthetic_tables):
    """The synthetic chart as the `Tables` object that the dashboard reads (11 snapshots, a missing day, a double date, re-entries)."""
    from src.dash.load import Tables
    playlist = synthetic_tables["playlist_daily"].sort_values(["date", "position"]).reset_index(drop=True)
    return Tables(playlist=playlist, dim_track=synthetic_tables["dim_track"], dim_artist=synthetic_tables["dim_artist"],
                  daily_summary=synthetic_tables["daily_summary"], dim_date=synthetic_tables["dim_date"],
                  bridge=synthetic_tables["bridge_track_artist"])


@pytest.fixture(scope="session")
def dash_real():
    """The real processed tables of the project, or a skip (not a failure) when data/processed/ has not been built."""
    from src.config import default_paths
    from src.dash.load import load_tables, missing_tables
    if missing_tables(default_paths().processed_dir):
        pytest.skip("data/processed is not built (run python -m src.etl.run_pipeline)")
    return load_tables()
