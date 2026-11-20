"""
tests/test_load_database.py  -  tests for step 9 of the pipeline: loading the star schema into a database.

WHAT THIS FILE IS
    The same tests run on SQLite (always) and on PostgreSQL (only when the environment variable TEST_POSTGRES_URL points
    to a PostgreSQL server; otherwise those copies are skipped). Each test is named after a promise of the loader:

        every row arrives and the numbers add up       | the database itself refuses bad rows (keys, value rules)
        a reload gives the same database, no copies    | somebody else's tables are never dropped
        a failed load changes nothing (all or nothing) | every run leaves one line in the audit table etl_run_log

WHY WE NEED IT
    A loader is the part of a pipeline that can do real damage: it drops and recreates tables. These tests prove that it
    is repeatable, that it protects other people's data, and that a crash in the middle never leaves a half-built warehouse.

WHAT HAPPENS WHEN IT RUNS
    Every test gets a brand-new, empty database (a SQLite file in a temporary folder whose name has a space, or a new
    PostgreSQL schema that is dropped afterwards). The synthetic chart (550 rows) keeps every test fast.

EXPECTED OUTCOME
    All SQLite tests pass; PostgreSQL tests pass or are skipped.

MOTIVE
    Load like an accountant: every row that goes in is counted again when it comes out, and a mistake can be undone.
"""
import pytest
import sqlalchemy as sa

from src.config import DB_TABLE_ORDER, SQL_DIR, ProjectPaths
from src.etl import load as load_module
from src.etl.errors import DataContractError, LoadError
from src.etl.extract import file_fingerprint
from src.etl.load import load_warehouse, write_run_log
from src.etl.run_pipeline import run
from tests import synthetic as syn
from tests.helpers import scalar

EXPECTED_ROWS = {"dim_date": 12, "dim_track": 71, "dim_artist": 81, "bridge_track_artist": 82, "fact_chart_daily": 550,
                 "daily_summary": 11}
VIEWS = {"vw_chart_daily", "vw_song_scorecard", "vw_artist_leaderboard", "vw_artist_songs", "vw_monthly_trend",
         "vw_weekday_pattern", "vw_movement_by_month", "vw_longevity_summary"}


@pytest.fixture(params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgres)])
def engine(request):
    """An empty database: SQLite always, PostgreSQL when TEST_POSTGRES_URL is set (otherwise that copy is skipped)."""
    return request.getfixturevalue("sqlite_engine" if request.param == "sqlite" else "postgres_engine")


def table_names(engine):
    return set(sa.inspect(engine).get_table_names())


def count(engine, table):
    with engine.connect() as connection:
        return scalar(connection, f"SELECT COUNT(*) FROM {table}")


def value(engine, query, **parameters):
    with engine.connect() as connection:
        return scalar(connection, query, **parameters)


def project_with(tmp_path, raw_frame, name="damaged project"):
    """A throw-away project folder whose raw CSV is `raw_frame` (used to feed the pipeline a damaged file)."""
    paths = ProjectPaths.under(tmp_path / name)
    paths.raw_file.parent.mkdir(parents=True)
    raw_frame.to_csv(paths.raw_file, index=False)
    return paths


# ================================================================================ the load itself
def test_every_row_arrives_and_the_views_exist(engine, synthetic_tables):
    report = load_warehouse(engine, synthetic_tables, SQL_DIR)
    assert report.rows == EXPECTED_ROWS and report.total_rows == 807
    for table, expected in EXPECTED_ROWS.items():
        assert count(engine, table) == expected, table
    assert set(report.views) == VIEWS
    assert VIEWS <= set(sa.inspect(engine).get_view_names())


def test_the_numbers_survive_the_round_trip(engine, synthetic_tables):
    load_warehouse(engine, synthetic_tables, SQL_DIR)
    fact = synthetic_tables["fact_chart_daily"]
    assert value(engine, "SELECT SUM(chart_points) FROM fact_chart_daily") == fact["chart_points"].sum() == 14_025
    assert value(engine, "SELECT COUNT(*) FROM fact_chart_daily WHERE rank_change IS NULL") == fact["rank_change"].isna().sum()
    assert value(engine, "SELECT COUNT(*) FROM fact_chart_daily WHERE popularity_clean IS NULL") == 1       # the one unfillable zero
    assert value(engine, "SELECT COUNT(*) FROM fact_chart_daily WHERE is_explicit = TRUE") == fact["is_explicit"].sum()
    assert value(engine, "SELECT SUM(popularity_clean) FROM fact_chart_daily") == pytest.approx(fact["popularity_clean"].sum())
    assert str(value(engine, "SELECT MIN(date) FROM dim_date"))[:10] == "2024-01-01"
    assert str(value(engine, "SELECT MAX(date) FROM dim_date"))[:10] == "2024-01-12"
    assert value(engine, "SELECT COUNT(*) FROM dim_date WHERE has_snapshot = FALSE") == 1                  # the missing day


def test_accents_and_commas_in_names_survive(engine, synthetic_tables):
    load_warehouse(engine, synthetic_tables, SQL_DIR)
    for name in ("Beyoncé", "Beyonce", "Earth, Wind & Fire", "Tyler, The Creator"):
        assert value(engine, "SELECT COUNT(*) FROM dim_artist WHERE artist_name = :n", n=name) == 1, name


def test_reloading_gives_the_same_database_not_copies(engine, synthetic_tables):
    load_warehouse(engine, synthetic_tables, SQL_DIR)
    load_warehouse(engine, synthetic_tables, SQL_DIR)
    load_warehouse(engine, synthetic_tables, SQL_DIR)
    for table, expected in EXPECTED_ROWS.items():
        assert count(engine, table) == expected, table


def test_missing_tables_are_reported_before_anything_is_touched(engine, synthetic_tables):
    incomplete = {k: v for k, v in synthetic_tables.items() if k != "dim_artist"}
    with pytest.raises(LoadError, match="dim_artist"):
        load_warehouse(engine, incomplete, SQL_DIR)
    assert not (set(DB_TABLE_ORDER) & table_names(engine))


# ================================================================================ the database says no to bad rows
@pytest.mark.parametrize("statement", [
    "UPDATE fact_chart_daily SET track_id = 99999 WHERE date = '2024-01-01' AND position = 1",         # no such song
    "UPDATE fact_chart_daily SET position = 51, chart_points = 0 WHERE date = '2024-01-01' AND position = 1",   # position 1-50 only
    "UPDATE fact_chart_daily SET chart_points = 7 WHERE date = '2024-01-01' AND position = 1",         # must equal 51 - position
    "UPDATE dim_date SET month = 13 WHERE date = '2024-01-01'",                                          # month 1-12 only
    "UPDATE dim_track SET track_label = NULL WHERE track_id = 1",                                        # NOT NULL
    "INSERT INTO dim_date SELECT * FROM dim_date WHERE date = '2024-01-01'",                             # primary key twice
    "UPDATE bridge_track_artist SET artist_name = 'Nobody' WHERE track_id = 1",                         # no such artist
])
def test_the_database_refuses_rows_that_break_a_rule(engine, synthetic_tables, statement):
    load_warehouse(engine, synthetic_tables, SQL_DIR)
    with pytest.raises(sa.exc.IntegrityError):
        with engine.begin() as connection:
            connection.execute(sa.text(statement))


# ================================================================================ protecting other people's tables
def test_a_database_with_foreign_tables_is_not_touched(engine, synthetic_tables):
    with engine.begin() as connection:                                       # somebody else's table that happens to be called dim_date
        connection.execute(sa.text("CREATE TABLE dim_date (precious_column INTEGER)"))
        connection.execute(sa.text("INSERT INTO dim_date VALUES (42)"))
    with pytest.raises(LoadError) as error:
        load_warehouse(engine, synthetic_tables, SQL_DIR)
    assert "etl_run_log" in str(error.value) and "--force-replace" in str(error.value)
    assert value(engine, "SELECT precious_column FROM dim_date") == 42       # still there, still theirs


def test_force_replace_overrides_the_protection(engine, synthetic_tables):
    with engine.begin() as connection:
        connection.execute(sa.text("CREATE TABLE dim_date (precious_column INTEGER)"))
    load_warehouse(engine, synthetic_tables, SQL_DIR, force_replace=True)
    assert count(engine, "dim_date") == 12


def test_the_audit_log_is_not_written_into_a_foreign_database(engine):
    with engine.begin() as connection:
        connection.execute(sa.text("CREATE TABLE dim_date (precious_column INTEGER)"))
    assert write_run_log(engine, {"run_id": "r1", "started_at_utc": "2024-01-01T00:00:00Z", "status": "SUCCESS"}, SQL_DIR) is False
    assert "etl_run_log" not in table_names(engine)


# ================================================================================ all or nothing
def break_the_foreign_key(tables):
    fact = tables["fact_chart_daily"]
    fact["track_id"] = fact["track_id"] + (fact.index == 0) * 1000
    return tables


def add_an_unknown_column(tables):
    tables["dim_date"]["surprise"] = 1
    return tables


@pytest.mark.parametrize("damage, message", [(break_the_foreign_key, "failed"), (add_an_unknown_column, "differ")])
def test_a_failed_load_changes_nothing(engine, synthetic_tables, damage, message):
    load_warehouse(engine, synthetic_tables, SQL_DIR)
    with engine.begin() as connection:                                       # a marker that only survives if NOTHING was dropped
        connection.execute(sa.text("UPDATE dim_artist SET unique_tracks = 99 WHERE artist_name = 'Artist 21'"))
    broken = damage({name: table.copy() for name, table in synthetic_tables.items()})
    with pytest.raises(LoadError, match=message):
        load_warehouse(engine, broken, SQL_DIR)
    for table, expected in EXPECTED_ROWS.items():
        assert count(engine, table) == expected, table
    assert value(engine, "SELECT unique_tracks FROM dim_artist WHERE artist_name = 'Artist 21'") == 99
    assert VIEWS <= set(sa.inspect(engine).get_view_names())


def test_a_failed_load_on_an_empty_database_leaves_no_half_built_tables(engine, synthetic_tables):
    with pytest.raises(LoadError):
        load_warehouse(engine, break_the_foreign_key({n: t.copy() for n, t in synthetic_tables.items()}), SQL_DIR)
    assert not (set(DB_TABLE_ORDER) & table_names(engine))                    # even the CREATE TABLE statements were rolled back


def test_the_after_load_check_catches_lost_rows(engine, synthetic_tables, monkeypatch):
    real_insert = load_module._insert_table

    def lossy_insert(connection, name, table):
        sent = real_insert(connection, name, table)
        if name == "daily_summary":                                          # pretend the database swallowed one row
            connection.execute(sa.text("DELETE FROM daily_summary WHERE snapshot_index = 1"))
        return sent

    monkeypatch.setattr(load_module, "_insert_table", lossy_insert)
    with pytest.raises(LoadError, match="sent 11 rows but the database holds 10"):
        load_warehouse(engine, synthetic_tables, SQL_DIR)
    assert not (set(DB_TABLE_ORDER) & table_names(engine))


def verify(engine, tables):
    """Run only the after-load check: count what the database holds and compare it with the tables we pass in."""
    rows_sent = {name: len(tables[name]) for name in DB_TABLE_ORDER}
    with engine.connect() as connection:
        load_module._verify_load(connection, tables, rows_sent)


def test_the_after_load_check_accepts_a_correct_load(engine, synthetic_tables):
    load_warehouse(engine, synthetic_tables, SQL_DIR)
    verify(engine, synthetic_tables)                                         # no exception: counts and checksums agree


def test_the_after_load_check_catches_a_changed_total(engine, synthetic_tables):
    load_warehouse(engine, synthetic_tables, SQL_DIR)
    doctored = {name: table.copy() for name, table in synthetic_tables.items()}
    doctored["fact_chart_daily"].loc[0, "chart_points"] += 1                 # same number of rows, one value different
    with pytest.raises(LoadError, match="Checksum mismatch"):
        verify(engine, doctored)


def test_the_after_load_check_catches_a_wrong_number_of_songs(engine, synthetic_tables):
    load_warehouse(engine, synthetic_tables, SQL_DIR)
    doctored = {name: table.copy() for name, table in synthetic_tables.items()}
    fact = doctored["fact_chart_daily"]
    fact.loc[fact["track_id"] == fact["track_id"].max(), "track_id"] = fact["track_id"].min()    # two songs become one
    with pytest.raises(LoadError, match="distinct songs"):
        verify(engine, doctored)


# ================================================================================ the audit table
def test_a_successful_run_is_recorded_once_per_run(engine, synthetic_paths):
    first = run(synthetic_paths, engine=engine, write_files=False)
    second = run(synthetic_paths, engine=engine, write_files=False)
    with engine.connect() as connection:
        rows = connection.execute(sa.text("SELECT run_id, status, source_file, source_sha256, source_rows, fact_rows, "
                                          "track_count, artist_count, quality_checks_failed, gate_tests_passed, "
                                          "gate_tests_total, message FROM etl_run_log ORDER BY started_at_utc, run_id")).fetchall()
    assert len(rows) == 2 and {r[0] for r in rows} == {first.run_id, second.run_id}
    sha = file_fingerprint(synthetic_paths.raw_file)["sha256"]
    assert tuple(rows[0][1:]) == ("SUCCESS", "Atlantic_United_States.csv", sha, 600, 550, 71, 81, 0, 25, 25, "OK")
    for table, expected in EXPECTED_ROWS.items():                            # the data tables were refreshed, not doubled
        assert count(engine, table) == expected, table


def test_a_failed_run_is_recorded_and_the_good_data_stays(engine, synthetic_paths, plan, tmp_path):
    run(synthetic_paths, engine=engine, write_files=False)                   # a good load first
    damaged = syn.raw_frame(plan)
    damaged.loc[0, "position"] = 51                                          # impossible rank: the scorecard must stop the run
    paths = project_with(tmp_path, damaged)
    with pytest.raises(DataContractError, match="Rank inside 1-50"):
        run(paths, engine=engine, write_files=False)
    with engine.connect() as connection:
        statuses = [r[0] for r in connection.execute(sa.text("SELECT status FROM etl_run_log ORDER BY started_at_utc, run_id"))]
        message = scalar(connection, "SELECT message FROM etl_run_log WHERE status = 'FAILED'")
    assert sorted(statuses) == ["FAILED", "SUCCESS"]
    assert "Rank inside 1-50" in message
    for table, expected in EXPECTED_ROWS.items():                            # the warehouse still holds the last good data
        assert count(engine, table) == expected, table


def test_a_run_without_a_database_load_never_touches_the_database(engine, synthetic_paths):
    run(synthetic_paths, load_db=False, write_files=False)
    assert table_names(engine) == set()


# ================================================================================ the views on loaded data
def test_the_views_return_rows(engine, synthetic_tables):
    load_warehouse(engine, synthetic_tables, SQL_DIR)
    expected = {"vw_chart_daily": 550, "vw_song_scorecard": 71, "vw_artist_leaderboard": 81, "vw_artist_songs": 82,
                "vw_monthly_trend": 1}
    for view, rows in expected.items():
        assert count(engine, view) == rows, view
    for view in VIEWS:
        assert count(engine, view) > 0, view
