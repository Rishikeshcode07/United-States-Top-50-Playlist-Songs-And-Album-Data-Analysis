-- =====================================================================================================
-- sql/00_etl_run_log.sql  -  the audit table of the pipeline
--
-- WHAT THIS FILE IS
--     One table, etl_run_log. Every pipeline run adds ONE row: when it started and finished, whether it succeeded,
--     which source file it used (name, row count, SHA-256 fingerprint) and how many rows it loaded.
--
-- WHY WE NEED IT
--     When a dashboard number looks wrong, the first question is "which data is in the warehouse right now, and when
--     did it arrive?". This table answers it. It is also the pipeline's proof that THIS database belongs to this project:
--     the loader refuses to replace tables in a database that has our table names but no etl_run_log.
--
-- WHAT HAPPENS WHEN IT RUNS
--     CREATE TABLE IF NOT EXISTS: created on the first run, left untouched afterwards. It is never dropped.
--
-- EXPECTED OUTCOME
--     SELECT * FROM etl_run_log ORDER BY started_at_utc DESC;   -> the history of all runs, newest first.
--
-- MOTIVE
--     A warehouse without a load history is a warehouse nobody can audit.
--
-- Portable SQL: runs unchanged on SQLite, PostgreSQL and MySQL 8. Timestamps are stored as ISO-8601 text
-- (for example 2026-10-14T09:12:33Z) because that sorts correctly and means the same in every database.
-- =====================================================================================================

CREATE TABLE IF NOT EXISTS etl_run_log (
    run_id                 VARCHAR(36)   NOT NULL,   -- a random UUID, one per run
    started_at_utc         VARCHAR(30)   NOT NULL,
    finished_at_utc        VARCHAR(30),
    status                 VARCHAR(10)   NOT NULL,   -- SUCCESS or FAILED
    pipeline_version       VARCHAR(20),
    db_target              VARCHAR(300),             -- where we loaded (the password is never written here)
    source_file            VARCHAR(260),
    source_sha256          VARCHAR(64),              -- changes if even one character of the CSV changes
    source_rows            INTEGER,
    fact_rows              INTEGER,
    track_count            INTEGER,
    artist_count           INTEGER,
    quality_checks_failed  INTEGER,                  -- FAIL rows of the data-quality scorecard (a successful run has 0)
    gate_tests_passed      INTEGER,
    gate_tests_total       INTEGER,
    message                VARCHAR(500),
    CONSTRAINT pk_etl_run_log PRIMARY KEY (run_id)
);
