"""
tests/test_build_engine.py  -  tests for the database settings and the engine builder (no database server needed).

WHAT THIS FILE IS
    Checks how `.env` settings become a database connection: defaults, the command-line override, missing settings,
    unsupported database names, passwords with special characters, and Windows folders that contain spaces.

WHY WE NEED IT
    A wrong connection setting should be explained in one clear sentence in the first second of a run, not as a long
    driver error after the analysis has finished. A password must also never appear in a log line.

WHAT HAPPENS WHEN IT RUNS
    Engines are only created, never connected, so these tests run without PostgreSQL or MySQL installed.

EXPECTED OUTCOME
    All tests pass.

MOTIVE
    Make the first thing a new user meets (the .env file) as safe and as clear as possible.
"""
import pytest

from src.etl.errors import LoadError
from src.etl.load import DbSettings, build_engine, describe_target, settings_from_env

KEYS = ["DB_TYPE", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]


@pytest.fixture()
def clean_env(monkeypatch):
    """No DB_* variable set, and anything the test (or python-dotenv) adds is removed again afterwards."""
    for key in KEYS:
        monkeypatch.setenv(key, "x")
        monkeypatch.delenv(key)
    return monkeypatch


def test_sqlite_is_the_default_and_needs_no_settings(clean_env, tmp_path):
    settings = settings_from_env(env_file=tmp_path / "missing.env")
    assert settings.db_type == "sqlite"


def test_the_command_line_choice_wins_over_the_env_file(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DB_TYPE=mysql\nDB_HOST=h\n", encoding="utf-8")
    assert settings_from_env("postgresql", env_file).db_type == "postgresql"
    assert settings_from_env(None, env_file).db_type == "mysql"


def test_values_come_from_the_env_file_and_real_variables_win(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# a comment\nDB_TYPE=postgresql\nDB_HOST=from_file\nDB_PORT=5433\nDB_NAME=atl\nDB_USER=me\nDB_PASSWORD=secret\n",
                        encoding="utf-8")
    clean_env.setenv("DB_HOST", "from_environment")
    settings = settings_from_env(env_file=env_file)
    assert (settings.host, settings.port, settings.name, settings.user, settings.password) == (
        "from_environment", "5433", "atl", "me", "secret")


def test_sqlite_in_a_folder_with_spaces(tmp_path):
    engine = build_engine(DbSettings("sqlite"), tmp_path / "my folder" / "sub dir" / "x.db")
    try:
        assert (tmp_path / "my folder" / "sub dir").is_dir()                 # the folder is created for you
        assert engine.dialect.name == "sqlite" and "my folder" in engine.url.database
    finally:
        engine.dispose()


def test_postgresql_engine_and_a_password_with_special_characters():
    engine = build_engine(DbSettings("postgresql", "db.example.com", "5432", "atl", "me", "p@ss:w/rd#1"), None)
    assert engine.dialect.name == "postgresql" and engine.url.drivername == "postgresql+psycopg2"
    assert engine.url.password == "p@ss:w/rd#1"                              # survives intact, nothing is cut at the @ or #
    assert (engine.url.host, engine.url.port, engine.url.database) == ("db.example.com", 5432, "atl")


def test_postgres_is_accepted_as_a_short_name():
    assert build_engine(DbSettings("postgres", "h", "", "n", "u", ""), None).dialect.name == "postgresql"


def test_mysql_engine_uses_pymysql_and_utf8mb4():
    engine = build_engine(DbSettings("mysql", "localhost", "3306", "atl", "me", "x"), None)
    assert engine.url.drivername == "mysql+pymysql" and engine.url.query["charset"] == "utf8mb4"


def test_the_password_is_hidden_when_the_target_is_printed():
    engine = build_engine(DbSettings("postgresql", "h", "5432", "atl", "me", "topsecret"), None)
    assert "topsecret" not in describe_target(engine) and "***" in describe_target(engine)


def test_missing_settings_are_listed_by_name():
    with pytest.raises(LoadError) as error:
        build_engine(DbSettings("postgresql", "", "", "", "", ""), None)
    for name in ("DB_HOST", "DB_NAME", "DB_USER"):
        assert name in str(error.value)
    assert ".env" in str(error.value)


def test_an_unsupported_database_is_refused_with_the_choices():
    with pytest.raises(LoadError) as error:
        build_engine(DbSettings("oracle", "h", "", "n", "u", ""), None)
    assert "sqlite, postgresql, mysql" in str(error.value)
