"""
src/etl/load.py  -  step 9 of the pipeline: load the tables into a database (SQLite, PostgreSQL or MySQL).

WHAT THIS FILE IS
    Everything that talks to the database: reading the settings from the .env file, opening a connection, running the
    SQL scripts of the sql/ folder, inserting the six star-schema tables, checking that what arrived equals what we sent,
    and writing one audit row per run into etl_run_log.

WHY WE NEED IT
    A CSV is a snapshot; a database is a shared, queryable, protected store. Power BI, SQL clients and the Streamlit app
    can all read the same tables. The load is a FULL REFRESH: drop, create, insert, verify. A full refresh is simple,
    and it is IDEMPOTENT: running it twice gives exactly the same database, never duplicates.

WHAT HAPPENS WHEN IT RUNS
    1. Safety guard: if the database already holds tables with our names but no etl_run_log, we refuse to touch it
       (it probably belongs to somebody else). Use --force-replace only if you are sure.
    2. One transaction does the work: 01_schema.sql -> insert the 6 tables (parents first) -> verify row counts and a
       checksum -> 02_views.sql. If anything fails, the transaction is rolled back and the old data stays.
       (PostgreSQL and SQLite roll back everything, including the DDL. MySQL commits DDL immediately, so after a
       failed run on MySQL simply run the pipeline again.)
    3. The caller writes the audit row with `write_run_log`.

EXPECTED OUTCOME
    A LoadReport such as: 6 tables, 31,196 rows, 8 views, 1 second. SQLite file: data/warehouse/atlantic_top50.db.

MOTIVE
    Load like an accountant: every row that goes in is counted again when it comes out.
"""
from __future__ import annotations

import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.engine import URL

from src.config import DB_TABLE_ORDER, PROJECT_ROOT
from src.etl.errors import LoadError
from src.logger import get_logger

log = get_logger(__name__)

INSERT_CHUNK = 5_000                                           # rows sent to the database in one go
INTEGER_LIKE_FLOAT_COLUMNS = {"fact_chart_daily": ["rank_change", "rank_gain_7d"]}   # whole numbers stored as float because of gaps
SUPPORTED = ("sqlite", "postgresql", "mysql")


# ====================================================================================== settings and engine
@dataclass(frozen=True)
class DbSettings:
    """Where to load. The values come from environment variables, normally from the .env file (never committed to Git)."""
    db_type: str = "sqlite"
    host: str = ""
    port: str = ""
    name: str = ""
    user: str = ""
    password: str = ""


def load_env(env_file=None):
    """Read the .env file (if there is one) into the environment. Real environment variables are never overwritten."""
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    except ImportError:
        log.debug("python-dotenv is not installed; using the real environment variables only")


def settings_from_env(db_type=None, env_file=None):
    """Read DB_TYPE, DB_HOST, DB_PORT, DB_NAME, DB_USER and DB_PASSWORD. `db_type` (from the command line) wins over the .env file."""
    load_env(env_file)
    get = lambda key: os.environ.get(key, "").strip()
    return DbSettings(db_type=(db_type or get("DB_TYPE") or "sqlite").lower(), host=get("DB_HOST"), port=get("DB_PORT"),
                      name=get("DB_NAME"), user=get("DB_USER"), password=os.environ.get("DB_PASSWORD", ""))


def build_engine(settings, sqlite_file):
    """Create the SQLAlchemy engine (the connection factory) for the chosen database."""
    kind = "postgresql" if settings.db_type == "postgres" else settings.db_type
    if kind == "sqlite":
        Path(sqlite_file).parent.mkdir(parents=True, exist_ok=True)
        engine = sa.create_engine(URL.create("sqlite", database=str(sqlite_file)))   # URL.create copes with spaces in Windows paths
        _prepare_sqlite(engine)
        return engine

    if kind not in SUPPORTED:
        raise LoadError(f"DB_TYPE '{settings.db_type}' is not supported. Choose one of: {', '.join(SUPPORTED)}. "
                        "(Snowflake is not part of this version of the pipeline.)")
    missing = [env for env, value in (("DB_HOST", settings.host), ("DB_NAME", settings.name), ("DB_USER", settings.user)) if not value]
    if missing:
        raise LoadError(f"DB_TYPE={kind} needs these settings in your .env file: {', '.join(missing)} "
                        "(see .env.example). The password is read from DB_PASSWORD.")
    driver = "postgresql+psycopg2" if kind == "postgresql" else "mysql+pymysql"
    url = URL.create(driver, username=settings.user, password=settings.password or None, host=settings.host,
                     port=int(settings.port) if settings.port else None, database=settings.name,
                     query={"charset": "utf8mb4"} if kind == "mysql" else {})
    return sa.create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})   # never wait forever for a dead server


def _prepare_sqlite(engine):
    """Three small SQLite settings: enforce foreign keys, make DDL part of the transaction, provide SQRT if it is missing."""
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection, connection_record):
        dbapi_connection.isolation_level = None                  # we start transactions ourselves (see _on_begin)
        dbapi_connection.execute("PRAGMA foreign_keys = ON")     # SQLite ignores foreign keys unless asked, every connection
        try:
            dbapi_connection.execute("SELECT SQRT(4)")
        except sqlite3.OperationalError:                         # an older SQLite build without math functions
            dbapi_connection.create_function("SQRT", 1, math.sqrt, deterministic=True)

    @event.listens_for(engine, "begin")
    def _on_begin(connection):
        connection.exec_driver_sql("BEGIN")                       # explicit BEGIN, so that CREATE/DROP TABLE can be rolled back too


def describe_target(engine):
    """A printable description of the database, with the password hidden."""
    return engine.url.render_as_string(hide_password=True)


# ====================================================================================== SQL scripts
def read_sql_statements(path):
    """
    Split a .sql file into single statements. Comments (-- ... and /* ... */) are removed first, quotes are respected,
    so a ';' inside a comment or a text value does not cut a statement in half.
    """
    text = Path(path).read_text(encoding="utf-8")
    statements, current = [], []
    quote = None
    i = 0
    while i < len(text):
        ch, nxt = text[i], text[i + 1] if i + 1 < len(text) else ""
        if quote:                                                  # inside '...' or "..."
            current.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            current.append(ch)
        elif ch == "-" and nxt == "-":                             # line comment: skip to the end of the line
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        elif ch == "/" and nxt == "*":                             # block comment: skip to the closing */
            end = text.find("*/", i + 2)
            i = len(text) if end == -1 else end + 2
            continue
        elif ch == ";":                                            # end of one statement
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
        else:
            current.append(ch)
        i += 1
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def run_sql_file(connection, path):
    """Run every statement of a .sql file on an open connection. Returns the number of statements."""
    statements = read_sql_statements(path)
    for statement in statements:
        connection.exec_driver_sql(statement)
    return len(statements)


# ====================================================================================== the load
@dataclass
class LoadReport:
    db_target: str
    rows: dict = field(default_factory=dict)         # rows loaded per table
    views: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def total_rows(self):
        return sum(self.rows.values())


def load_warehouse(engine, tables, sql_dir, force_replace=False):
    """Full refresh of the six star-schema tables and the views. Raises LoadError if anything goes wrong."""
    started = time.time()
    sql_dir = Path(sql_dir)
    missing_tables = [name for name in DB_TABLE_ORDER if name not in tables]
    if missing_tables:
        raise LoadError(f"These tables were not produced and can not be loaded: {missing_tables}")
    try:
        _guard_against_foreign_tables(engine, force_replace)
        with engine.begin() as connection:                               # the audit table first, in its own small transaction
            run_sql_file(connection, sql_dir / "00_etl_run_log.sql")

        report = LoadReport(db_target=describe_target(engine))
        with engine.begin() as connection:                               # ONE transaction: all or nothing
            run_sql_file(connection, sql_dir / "01_schema.sql")
            for name in DB_TABLE_ORDER:                                  # parents first, so that foreign keys are satisfied
                report.rows[name] = _insert_table(connection, name, tables[name])
            _verify_load(connection, tables, report.rows)
            run_sql_file(connection, sql_dir / "02_views.sql")
            report.views = sorted(sa.inspect(connection).get_view_names())
    except LoadError:
        raise
    except sa.exc.SQLAlchemyError as error:
        reason = str(getattr(error, "orig", None) or error).strip().splitlines()[0][:400]
        raise LoadError(f"Loading into {describe_target(engine)} failed ({type(error).__name__}): {reason}") from error

    report.seconds = round(time.time() - started, 2)
    log.info("Loaded %s rows into %s tables and %s views in %.1f s -> %s", f"{report.total_rows:,}", len(report.rows),
             len(report.views), report.seconds, report.db_target)
    return report


def _guard_against_foreign_tables(engine, force_replace):
    """Do not drop tables that this pipeline did not create. Our databases always contain etl_run_log."""
    inspector = sa.inspect(engine)
    existing = [name for name in DB_TABLE_ORDER if inspector.has_table(name)]
    if existing and not inspector.has_table("etl_run_log") and not force_replace:
        raise LoadError(f"The database already has table(s) {existing} but no etl_run_log table, so this pipeline did not "
                        "create them. Refusing to replace them. Use an empty database (or schema), or run again with "
                        "--force-replace if you are sure.")


def _prepare_for_database(name, table):
    """Make a copy of a table in the shapes databases like: plain dates, whole numbers as integers."""
    frame = table.copy()
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = frame[col].dt.date                              # 2024-05-18, not 2024-05-18 00:00:00
    for col in INTEGER_LIKE_FLOAT_COLUMNS.get(name, []):
        frame[col] = frame[col].astype("Int64")                          # 3.0 -> 3, gaps stay empty
    return frame


def _to_records(frame):
    """A list of {column: value} dictionaries with plain Python values; NaN and NA become None (SQL NULL)."""
    as_objects = frame.astype(object)
    return as_objects.where(frame.notna(), None).to_dict(orient="records")


def _insert_table(connection, name, table):
    """Insert one table in chunks and return the number of rows sent."""
    prepared = _prepare_for_database(name, table)
    target = sa.Table(name, sa.MetaData(), autoload_with=connection)     # read the column types back from the database
    db_columns = [c.name for c in target.columns]
    if set(db_columns) != set(prepared.columns):
        raise LoadError(f"{name}: the columns of the DataFrame and of the SQL table differ. "
                        f"Only in DataFrame: {sorted(set(prepared.columns) - set(db_columns))}; "
                        f"only in SQL: {sorted(set(db_columns) - set(prepared.columns))}")
    records = _to_records(prepared)
    for start in range(0, len(records), INSERT_CHUNK):
        connection.execute(sa.insert(target), records[start:start + INSERT_CHUNK])
    log.debug("Inserted %s rows into %s", f"{len(records):,}", name)
    return len(records)


def _verify_load(connection, tables, rows_sent):
    """Count what the database now holds and compare with what we sent: row counts per table plus one checksum."""
    for name, sent in rows_sent.items():
        stored = connection.execute(sa.text(f"SELECT COUNT(*) FROM {name}")).scalar_one()
        if stored != sent:
            raise LoadError(f"{name}: sent {sent:,} rows but the database holds {stored:,}.")
    fact = tables["fact_chart_daily"]
    stored_points = connection.execute(sa.text("SELECT SUM(chart_points) FROM fact_chart_daily")).scalar_one()
    if int(stored_points) != int(fact["chart_points"].sum()):
        raise LoadError(f"Checksum mismatch: SUM(chart_points) is {stored_points} in the database "
                        f"but {int(fact['chart_points'].sum())} in the DataFrame.")
    stored_tracks = connection.execute(sa.text("SELECT COUNT(DISTINCT track_id) FROM fact_chart_daily")).scalar_one()
    if int(stored_tracks) != int(fact["track_id"].nunique()):
        raise LoadError("The number of distinct songs in fact_chart_daily does not match the DataFrame.")


# ====================================================================================== audit log
def _target_is_ours(engine):
    inspector = sa.inspect(engine)
    return inspector.has_table("etl_run_log") or not any(inspector.has_table(name) for name in DB_TABLE_ORDER)


def write_run_log(engine, record, sql_dir):
    """
    Add one row to etl_run_log. Returns True if written. It never raises: a failing audit write must not hide the real
    result of the run. It also refuses to create the table inside a database that belongs to somebody else.
    """
    try:
        if not _target_is_ours(engine):
            log.warning("etl_run_log was not written: the database holds foreign tables with our names.")
            return False
        with engine.begin() as connection:
            run_sql_file(connection, Path(sql_dir) / "00_etl_run_log.sql")
            target = sa.Table("etl_run_log", sa.MetaData(), autoload_with=connection)
            allowed = {c.name for c in target.columns}
            row = {key: value for key, value in record.items() if key in allowed}
            if row.get("message"):
                row["message"] = str(row["message"])[:500]
            connection.execute(sa.insert(target), [row])
        return True
    except Exception as error:                                          # noqa: BLE001 - audit must never break the run
        log.warning("etl_run_log was not written: %s", str(error).splitlines()[0][:200] if str(error) else type(error).__name__)
        return False
