"""
tests/test_sql_files.py  -  tests for the SQL scripts of the sql/ folder and for the small reader that runs them.

WHAT THIS FILE IS
    (1) Unit tests of `read_sql_statements`, the function that cuts a .sql file into single statements while ignoring
        comments and text in quotes.
    (2) "Contract" tests: the SQL scripts must agree with the Python side. The columns of every CREATE TABLE must equal
        config.TABLE_COLUMNS (names AND order), keys and foreign keys must exist, every view must be created and dropped,
        and every KPI query must run.
    (3) A portability lint: the scripts must not contain words that only one database understands.

WHY WE NEED IT
    The loader inserts DataFrame columns into SQL columns by name. If someone adds a column to the DataFrame and forgets the
    SQL (or the other way round) the load would fail in a confusing way, or worse, silently drop the column. These tests
    make that mistake impossible to miss.

WHAT HAPPENS WHEN IT RUNS
    The scripts are executed on an empty SQLite database in a temporary folder and the database is inspected with SQLAlchemy.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Keep the Python code and the SQL code telling the same story.
"""
import re

import pytest
import sqlalchemy as sa

from src.config import DB_TABLE_ORDER, SQL_DIR, TABLE_COLUMNS
from src.etl.load import read_sql_statements, run_sql_file

VIEWS = ["vw_chart_daily", "vw_song_scorecard", "vw_artist_leaderboard", "vw_artist_songs", "vw_monthly_trend",
         "vw_weekday_pattern", "vw_movement_by_month", "vw_longevity_summary"]


def write(tmp_path, text):
    path = tmp_path / "script.sql"
    path.write_text(text, encoding="utf-8")
    return path


# ================================================================================ the statement reader
def test_it_splits_on_semicolons_and_drops_empty_pieces(tmp_path):
    path = write(tmp_path, "SELECT 1;\n\n SELECT 2 ;;\n")
    assert read_sql_statements(path) == ["SELECT 1", "SELECT 2"]


def test_it_removes_line_comments_and_block_comments(tmp_path):
    path = write(tmp_path, "-- a comment; with a semicolon\nSELECT 1; /* block; comment\n over lines */ SELECT 2; -- end")
    assert read_sql_statements(path) == ["SELECT 1", "SELECT 2"]


def test_a_semicolon_or_dashes_inside_quotes_are_part_of_the_text(tmp_path):
    path = write(tmp_path, "SELECT 'a;b -- not a comment'; SELECT \"odd;name\" FROM t;")
    assert read_sql_statements(path) == ["SELECT 'a;b -- not a comment'", 'SELECT "odd;name" FROM t']


def test_a_doubled_quote_inside_text_does_not_end_the_text(tmp_path):
    path = write(tmp_path, "SELECT 'it''s; fine'; SELECT 2;")
    assert read_sql_statements(path) == ["SELECT 'it''s; fine'", "SELECT 2"]


def test_the_last_statement_may_miss_its_semicolon(tmp_path):
    assert read_sql_statements(write(tmp_path, "SELECT 1;\nSELECT 2")) == ["SELECT 1", "SELECT 2"]


def test_an_empty_or_comment_only_file_has_no_statements(tmp_path):
    assert read_sql_statements(write(tmp_path, "")) == []
    assert read_sql_statements(write(tmp_path, "-- nothing here\n/* nor here */\n")) == []


def test_a_script_with_non_english_text_is_read_as_utf8(tmp_path):
    assert read_sql_statements(write(tmp_path, "SELECT 'Beyoncé';")) == ["SELECT 'Beyoncé'"]


# ================================================================================ the shipped scripts
@pytest.fixture()
def empty_schema(sqlite_engine):
    """An empty SQLite database with the audit table and the six tables of the star schema."""
    with sqlite_engine.begin() as connection:
        run_sql_file(connection, SQL_DIR / "00_etl_run_log.sql")
        run_sql_file(connection, SQL_DIR / "01_schema.sql")
    return sqlite_engine


def test_all_four_scripts_exist_and_have_statements():
    for name in ["00_etl_run_log.sql", "01_schema.sql", "02_views.sql", "03_kpi_queries.sql"]:
        assert read_sql_statements(SQL_DIR / name), name


def test_the_schema_creates_the_six_tables_of_the_loader(empty_schema):
    inspector = sa.inspect(empty_schema)
    assert set(DB_TABLE_ORDER) <= set(inspector.get_table_names())
    assert "etl_run_log" in inspector.get_table_names()


@pytest.mark.parametrize("table", DB_TABLE_ORDER)
def test_the_sql_columns_equal_the_python_columns_in_name_and_order(empty_schema, table):
    sql_columns = [c["name"] for c in sa.inspect(empty_schema).get_columns(table)]
    assert sql_columns == TABLE_COLUMNS[table]


EXPECTED_KEYS = {
    "dim_date": ["date"], "dim_track": ["track_id"], "dim_artist": ["artist_name"],
    "bridge_track_artist": ["track_id", "artist_order"], "fact_chart_daily": ["date", "position"], "daily_summary": ["date"],
}
EXPECTED_FOREIGN_KEYS = {
    "bridge_track_artist": {("track_id", "dim_track"), ("artist_name", "dim_artist")},
    "fact_chart_daily": {("date", "dim_date"), ("track_id", "dim_track")},
    "daily_summary": {("date", "dim_date")},
}


@pytest.mark.parametrize("table", DB_TABLE_ORDER)
def test_every_table_has_its_primary_key(empty_schema, table):
    key = sa.inspect(empty_schema).get_pk_constraint(table)["constrained_columns"]
    assert key == EXPECTED_KEYS[table]


@pytest.mark.parametrize("table", sorted(EXPECTED_FOREIGN_KEYS))
def test_the_foreign_keys_of_the_star(empty_schema, table):
    found = {(fk["constrained_columns"][0], fk["referred_table"]) for fk in sa.inspect(empty_schema).get_foreign_keys(table)}
    assert found == EXPECTED_FOREIGN_KEYS[table]


def test_the_schema_script_can_run_again_on_a_filled_database(empty_schema):
    with empty_schema.begin() as connection:
        run_sql_file(connection, SQL_DIR / "01_schema.sql")
        run_sql_file(connection, SQL_DIR / "01_schema.sql")                  # drop + create: running it twice is harmless
    assert set(DB_TABLE_ORDER) <= set(sa.inspect(empty_schema).get_table_names())


def test_the_audit_table_script_is_idempotent(empty_schema):
    with empty_schema.begin() as connection:
        run_sql_file(connection, SQL_DIR / "00_etl_run_log.sql")
        run_sql_file(connection, SQL_DIR / "00_etl_run_log.sql")


def test_the_schema_script_does_not_touch_the_audit_table(empty_schema):
    with empty_schema.begin() as connection:
        connection.execute(sa.text("INSERT INTO etl_run_log (run_id, started_at_utc, status) VALUES ('r1', '2024-01-01T00:00:00Z', 'SUCCESS')"))
        run_sql_file(connection, SQL_DIR / "01_schema.sql")
        assert connection.execute(sa.text("SELECT COUNT(*) FROM etl_run_log")).scalar_one() == 1   # the history of runs survives a reload


def test_every_view_is_created_and_dropped(empty_schema):
    schema_text = "\n".join(read_sql_statements(SQL_DIR / "01_schema.sql"))
    views_text = "\n".join(read_sql_statements(SQL_DIR / "02_views.sql"))
    created = re.findall(r"CREATE VIEW (\w+)", views_text)
    assert sorted(created) == sorted(VIEWS)
    for view in created:
        assert f"DROP VIEW IF EXISTS {view}" in schema_text, f"{view} is not dropped before the tables are dropped"
        assert f"DROP VIEW IF EXISTS {view}" in views_text, f"{view} is not dropped before it is created"


def test_tables_are_dropped_children_before_parents():
    statements = read_sql_statements(SQL_DIR / "01_schema.sql")
    dropped = [s.split()[-1] for s in statements if s.upper().startswith("DROP TABLE")]
    assert dropped == list(reversed(DB_TABLE_ORDER))                          # the exact reverse of the loading order


def test_the_views_run_on_the_empty_schema(empty_schema):
    with empty_schema.begin() as connection:
        run_sql_file(connection, SQL_DIR / "02_views.sql")
        for view in VIEWS:
            assert connection.execute(sa.text(f"SELECT COUNT(*) FROM {view}")).scalar_one() == 0, view
    assert set(VIEWS) <= set(sa.inspect(empty_schema).get_view_names())


def test_the_twelve_kpi_queries_run_on_the_empty_schema(empty_schema):
    with empty_schema.begin() as connection:
        run_sql_file(connection, SQL_DIR / "02_views.sql")                   # Q11 reads a view
        statements = read_sql_statements(SQL_DIR / "03_kpi_queries.sql")
        assert len(statements) == 12
        for number, statement in enumerate(statements, start=1):
            connection.exec_driver_sql(statement).fetchall()                 # raises if the SQL has an error


# ================================================================================ portability lint
NOT_PORTABLE = [r"::", r"`", r"\bILIKE\b", r"\bAUTOINCREMENT\b", r"\bAUTO_INCREMENT\b", r"\bSERIAL\b", r"\bSTRFTIME\b",
                r"\bDATE_TRUNC\b", r"\bGENERATE_SERIES\b", r"\bGROUP_CONCAT\b", r"\bSTRING_AGG\b", r"\bIFNULL\b", r"\bNVL\b",
                r"\bISNULL\s*\(", r"\bSTDDEV", r"\bTOP\s+\d", r"\bPRAGMA\b", r"\bUNSIGNED\b", r"\bENGINE\s*="]


@pytest.mark.parametrize("name", ["00_etl_run_log.sql", "01_schema.sql", "02_views.sql", "03_kpi_queries.sql"])
def test_the_scripts_use_only_sql_that_all_three_databases_understand(name):
    text = "\n".join(read_sql_statements(SQL_DIR / name))                      # comments are removed first: they may mention such words
    for pattern in NOT_PORTABLE:
        assert not re.search(pattern, text, flags=re.IGNORECASE), f"{name} uses {pattern}, which is not portable"
