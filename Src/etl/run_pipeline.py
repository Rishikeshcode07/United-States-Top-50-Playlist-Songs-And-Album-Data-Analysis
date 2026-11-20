"""
src/etl/run_pipeline.py  -  THE pipeline: one command that runs every step, in order, and stops safely when something is wrong.

WHAT THIS FILE IS
    The conductor. It calls the modules of the project one after the other:

        [1] extract      read the raw CSV, fingerprint it                         src/etl/extract.py
        [2] validate     31-check data-quality scorecard (stops on FAIL)          src/etl/validate.py + standardize.py
        [3] standardize  artists, titles, song identity (track_id), flags         src/etl/standardize.py
        [4] clean        repairs + chart runs, every decision logged              src/etl/clean.py
        [5] features     KPIs and the seven output tables                         src/features/build_features.py
        [6] gate         25 automatic tests (stops on FAIL)                       src/etl/quality.py
        [7] document     data dictionary and KPI dictionary                       src/etl/documentation.py
        [8] save         Parquet + CSV files and reports (only after the gate)   this file
        [9] load         database: SQLite / PostgreSQL / MySQL (optional)         src/etl/load.py

WHY WE NEED IT
    A notebook is perfect for exploring; a pipeline is what you run every day. The pipeline gives the same result every
    time, runs without a person clicking cells, leaves a log file, and refuses to publish bad data.

WHAT HAPPENS WHEN IT RUNS
    From the project folder:
        python -m src.etl.run_pipeline                 # files + SQLite database (the default)
        python -m src.etl.run_pipeline --no-db         # files only
        python -m src.etl.run_pipeline --db postgresql # use the settings of your .env file
    If any step fails, nothing from that run is published: the previous good files stay as they were, the scorecard and
    gate results are saved for diagnosis, the log file says where it stopped, and the exit code is not 0.

EXPECTED OUTCOME
    About 15 seconds for the project file, a summary block on the screen and these files:
        data/interim/      playlist_validated.parquet|csv, playlist_clean.parquet
        data/processed/    the 7 tables as .parquet and .csv
        data/warehouse/    atlantic_top50.db (SQLite)
        reports/           data_quality_scorecard, cleaning_log, quality_gate, data_dictionary, kpi_dictionary,
                           track_identity_decisions, calendar_reality_check (.csv) and last_run.json
        logs/pipeline.log  the full log

MOTIVE
    Automation turns analysis into a product: repeatable, auditable and safe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import PIPELINE_VERSION, ProjectPaths, default_paths
from src.etl.clean import CleaningLog, apply_cleaning
from src.etl.documentation import build_data_dictionary, build_kpi_dictionary
from src.etl.errors import PipelineError
from src.etl.extract import file_fingerprint, read_raw
from src.etl.load import build_engine, load_env, load_warehouse, settings_from_env, write_run_log
from src.etl.quality import raise_if_gate_failed, run_quality_gate
from src.etl.standardize import standardize
from src.etl.validate import QualityLog, validate_raw
from src.features.build_features import build_all
from src.logger import close_logging, configure_logging, get_logger

log = get_logger("src.etl.run_pipeline")        # a fixed name, so the log lines look the same when started with python -m
TOTAL_STEPS = 9


@dataclass
class PipelineResult:
    """Everything a run produced. A notebook can use these objects directly, without reading the files."""
    run_id: str
    status: str = "RUNNING"
    started_at_utc: str = ""
    finished_at_utc: str = ""
    source: dict = field(default_factory=dict)
    raw_rows: int = 0
    validated: pd.DataFrame | None = None
    clean: pd.DataFrame | None = None
    tables: dict = field(default_factory=dict)
    scorecard: pd.DataFrame | None = None
    cleaning_log: pd.DataFrame | None = None
    gate: pd.DataFrame | None = None
    data_dictionary: pd.DataFrame | None = None
    kpi_dictionary: pd.DataFrame | None = None
    identity_decisions: pd.DataFrame | None = None
    calendar_reality: pd.DataFrame | None = None
    written_files: list = field(default_factory=list)
    load_report: object | None = None
    timings: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


# ====================================================================================== the run
def run(paths=None, *, load_db=True, db_type=None, force_replace=False, write_files=True, engine=None, settings=None):
    """
    Run the whole pipeline and return a PipelineResult. Raises a PipelineError (DataContractError, DataIntegrityError or
    LoadError) when a step fails; nothing from a failed run is published.

    paths         : a ProjectPaths object (the tests pass a temporary copy); default = the real project folders
    load_db       : load the database after the files are saved
    db_type       : "sqlite" | "postgresql" | "mysql" (default: DB_TYPE of the .env file, else sqlite)
    write_files   : False = keep everything in memory (handy in a notebook)
    engine        : an existing SQLAlchemy engine (the tests pass one)
    """
    paths = paths or default_paths()
    configure_logging(paths.logs_dir, to_file=write_files)
    result = PipelineResult(run_id=str(uuid.uuid4()), started_at_utc=_utc_now())
    clock = time.time()
    own_engine = False
    qlog = None
    try:
        log.info("=" * 100)
        log.info("Atlantic Top 50 pipeline %s | run %s", PIPELINE_VERSION, result.run_id)

        # ---- settings first: a wrong .env should fail in the first second, not after the analysis -----------------------
        if load_db and engine is None:
            settings = settings or settings_from_env(db_type)
            engine = build_engine(settings, paths.sqlite_file)
            own_engine = True

        # ---- 1. extract ------------------------------------------------------------------------------------------------
        _step(1, "Extract: read the raw CSV")
        t = time.time()
        raw = read_raw(paths.raw_file)
        result.raw_rows = len(raw)
        result.source = {**file_fingerprint(paths.raw_file), "rows": len(raw)}
        result.timings["extract"] = _lap(t)

        # ---- 2 + 3. validate and standardize ---------------------------------------------------------------------------
        _step(2, "Validate: data-quality scorecard")
        t = time.time()
        qlog = QualityLog()
        typed = validate_raw(raw, qlog)
        qlog.raise_if_failed()
        result.timings["validate"] = _lap(t)

        _step(3, "Standardize: artists, titles and song identity")
        t = time.time()
        result.validated, result.identity_decisions = standardize(typed, qlog)
        qlog.raise_if_failed()
        result.scorecard = qlog.to_frame()
        result.timings["standardize"] = _lap(t)
        log.info("Scorecard: %s", qlog.summary())

        # ---- 4 + 5. clean and build features ---------------------------------------------------------------------------
        _step(4, "Clean: repairs and chart runs")
        t = time.time()
        clog = CleaningLog()
        result.clean, result.calendar_reality = apply_cleaning(result.validated, clog)
        result.timings["clean"] = _lap(t)

        _step(5, "Features: KPIs and the seven tables")
        t = time.time()
        result.tables = build_all(result.clean, clog)
        result.cleaning_log = clog.to_frame()
        result.timings["features"] = _lap(t)

        # ---- 6. the quality gate: nothing is saved before it passes ----------------------------------------------------
        _step(6, "Quality gate: 25 automatic tests")
        t = time.time()
        result.gate = run_quality_gate(result.tables)
        raise_if_gate_failed(result.gate)
        result.timings["gate"] = _lap(t)

        # ---- 7. documentation ------------------------------------------------------------------------------------------
        _step(7, "Document: data dictionary and KPI dictionary")
        result.data_dictionary = build_data_dictionary(result.tables)
        result.kpi_dictionary = build_kpi_dictionary()

        # ---- 8. save ---------------------------------------------------------------------------------------------------
        _step(8, "Save: files and reports" if write_files else "Save: skipped (write_files=False)")
        t = time.time()
        if write_files:
            result.written_files = _save_outputs(paths, result)
        result.timings["save"] = _lap(t)

        # ---- 9. database -----------------------------------------------------------------------------------------------
        if load_db:
            _step(9, f"Load: {engine.dialect.name} database")
            t = time.time()
            result.load_report = load_warehouse(engine, result.tables, paths.sql_dir, force_replace=force_replace)
            result.timings["load"] = _lap(t)
        else:
            _step(9, "Load: skipped (--no-db)")

        # ---- finished --------------------------------------------------------------------------------------------------
        result.status = "SUCCESS"
        result.finished_at_utc = _utc_now()
        result.timings["total"] = _lap(clock)
        result.summary = _build_summary(paths, result, qlog, engine if load_db else None)
        if load_db:
            write_run_log(engine, _run_log_record(result, qlog, ""), paths.sql_dir)
        if write_files:
            _write_json_atomic(paths.reports_dir / "last_run.json", result.summary)
        _log_summary(result)
        return result

    except Exception as error:                                                   # noqa: BLE001 - we log, record and re-raise
        result.status = "FAILED"
        result.finished_at_utc = _utc_now()
        result.timings["total"] = _lap(clock)
        message = f"{type(error).__name__}: {str(error).strip()}"
        log.error("PIPELINE FAILED at step: %s", message.splitlines()[0])
        log.debug("Full error:\n%s", message)
        if write_files:                                    # diagnostics only; good files from earlier runs are never touched
            _save_diagnostics(paths, qlog, result)
            _write_json_atomic(paths.reports_dir / "last_run.json",
                               {"run_id": result.run_id, "status": "FAILED", "started_at_utc": result.started_at_utc,
                                "finished_at_utc": result.finished_at_utc, "message": message[:2000],
                                "pipeline_version": PIPELINE_VERSION})
        if load_db and engine is not None:
            write_run_log(engine, _run_log_record(result, qlog, message), paths.sql_dir)
        raise
    finally:
        if own_engine and engine is not None:
            engine.dispose()
        close_logging(keep_screen=True)                    # release the log file, keep printing to the screen


# ====================================================================================== helpers
def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lap(since):
    return round(time.time() - since, 2)


def _step(number, text):
    log.info("[%s/%s] %s", number, TOTAL_STEPS, text)


def _write_atomic(path, write_function):
    """Write to a temporary file first, then rename. A crash can never leave a half-written file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        write_function(temporary)
        os.replace(temporary, path)
    except PermissionError as error:
        temporary.unlink(missing_ok=True)
        raise PipelineError(f"Can not write {path}. Is the file open in Excel or Power BI? Close it and run again.") from error


def _write_json_atomic(path, payload):
    _write_atomic(path, lambda target: Path(target).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8"))


def _write_csv(path, frame, encoding="utf-8-sig"):
    _write_atomic(path, lambda target: frame.to_csv(target, index=False, encoding=encoding))


def _write_parquet(path, frame):
    _write_atomic(path, lambda target: frame.to_parquet(target, index=False))


def _save_outputs(paths, result):
    """Write every output file. Same file names and encodings as notebooks 01 and 02, so the pipeline replaces them."""
    written = []

    def parquet(path, frame):
        _write_parquet(path, frame)
        written.append(Path(path))

    def csv(path, frame, encoding="utf-8-sig"):
        _write_csv(path, frame, encoding)
        written.append(Path(path))

    # data/interim: checkpoints between the stages
    parquet(paths.interim_dir / "playlist_validated.parquet", result.validated)
    csv(paths.interim_dir / "playlist_validated.csv", result.validated, encoding="utf-8")                 # as in notebook 01
    parquet(paths.interim_dir / "playlist_clean.parquet", result.clean)

    # data/processed: the seven analysis tables, as Parquet (fast, keeps data types) and CSV (opens everywhere)
    for name, table in result.tables.items():
        parquet(paths.processed_dir / f"{name}.parquet", table)
        csv(paths.processed_dir / f"{name}.csv", table)

    # reports: scorecard, decision logs, dictionaries
    csv(paths.reports_dir / "data_quality_scorecard.csv", result.scorecard, encoding="utf-8")             # as in notebook 01
    csv(paths.reports_dir / "cleaning_log.csv", result.cleaning_log)
    csv(paths.reports_dir / "quality_gate.csv", result.gate)
    csv(paths.reports_dir / "data_dictionary.csv", result.data_dictionary)
    csv(paths.reports_dir / "kpi_dictionary.csv", result.kpi_dictionary)
    csv(paths.reports_dir / "track_identity_decisions.csv", result.identity_decisions)
    csv(paths.reports_dir / "calendar_reality_check.csv", result.calendar_reality)
    log.info("Saved %s files in data/interim, data/processed and reports", len(written))
    return written


def _save_diagnostics(paths, qlog, result):
    """After a failure keep the evidence: the (partial) scorecard and the gate results, so that the cause can be read."""
    try:
        if qlog is not None:
            _write_csv(paths.reports_dir / "data_quality_scorecard.csv", qlog.to_frame(), encoding="utf-8")
        if result.gate is not None:
            _write_csv(paths.reports_dir / "quality_gate.csv", result.gate)
    except Exception as error:                                                   # noqa: BLE001
        log.warning("Could not save the diagnostic reports: %s", error)


def _build_summary(paths, result, qlog, engine):
    gate = result.gate
    summary = {
        "run_id": result.run_id, "status": result.status, "pipeline_version": PIPELINE_VERSION,
        "started_at_utc": result.started_at_utc, "finished_at_utc": result.finished_at_utc,
        "source": result.source,
        "validation": qlog.summary(),
        "gate": {"passed": int((gate["status"] == "PASS").sum()), "total": int(len(gate))},
        "tables": {name: {"rows": int(len(t)), "columns": int(t.shape[1])} for name, t in result.tables.items()},
        "timings_seconds": result.timings,
    }
    if result.load_report is not None:
        summary["database"] = {"target": result.load_report.db_target, "rows": result.load_report.rows,
                               "views": result.load_report.views}
    return summary


def _run_log_record(result, qlog, message):
    """One row for the etl_run_log table (see sql/00_etl_run_log.sql)."""
    gate = result.gate
    facts = result.tables.get("fact_chart_daily")
    return {
        "run_id": result.run_id, "started_at_utc": result.started_at_utc, "finished_at_utc": result.finished_at_utc,
        "status": "SUCCESS" if result.status == "SUCCESS" else "FAILED", "pipeline_version": PIPELINE_VERSION,
        "db_target": result.load_report.db_target if result.load_report else None,
        "source_file": result.source.get("name"), "source_sha256": result.source.get("sha256"),
        "source_rows": result.raw_rows or None,
        "fact_rows": int(len(facts)) if facts is not None else None,
        "track_count": int(len(result.tables["dim_track"])) if "dim_track" in result.tables else None,
        "artist_count": int(len(result.tables["dim_artist"])) if "dim_artist" in result.tables else None,
        "quality_checks_failed": len(qlog.failures()) if qlog is not None else None,
        "gate_tests_passed": int((gate["status"] == "PASS").sum()) if gate is not None else None,
        "gate_tests_total": int(len(gate)) if gate is not None else None,
        "message": message or "OK",
    }


def _log_summary(result):
    s = result.summary
    log.info("=" * 100)
    log.info("PIPELINE FINISHED: %s in %.1f s", result.status, result.timings.get("total", 0))
    log.info("Source     : %s (%s rows, sha256 %s...)", s["source"]["name"], f"{s['source']['rows']:,}", s["source"]["sha256"][:12])
    log.info("Validation : %s", s["validation"])
    log.info("Gate       : %s of %s tests passed", s["gate"]["passed"], s["gate"]["total"])
    log.info("Tables     : %s", " | ".join(f"{n} {v['rows']:,}" for n, v in s["tables"].items()))
    if "database" in s:
        log.info("Database   : %s (%s rows, %s views)", s["database"]["target"], f"{sum(s['database']['rows'].values()):,}",
                 len(s["database"]["views"]))
    log.info("=" * 100)


# ====================================================================================== command line
def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.etl.run_pipeline",
                                     description="Run the Atlantic Top 50 pipeline: validate, clean, build tables, save, load.")
    parser.add_argument("--project-root", help="use another project folder (default: the folder of this project)")
    parser.add_argument("--input", help="path of the raw CSV (default: data/raw/Atlantic_United_States.csv)")
    parser.add_argument("--no-db", action="store_true", help="save the files but do not load a database")
    parser.add_argument("--db", choices=["sqlite", "postgresql", "mysql"],
                        help="database to load (default: DB_TYPE of the .env file, otherwise sqlite)")
    parser.add_argument("--force-replace", action="store_true",
                        help="replace tables with our names even if this pipeline did not create them (use with care)")
    args = parser.parse_args(argv)

    paths = ProjectPaths.under(args.project_root) if args.project_root else default_paths()
    if args.input:
        paths = replace(paths, raw_file=Path(args.input))
    try:
        run(paths, load_db=not args.no_db, db_type=args.db, force_replace=args.force_replace)
    except PipelineError as error:
        print(f"\nThe pipeline stopped:\n{error}\n\nDetails: logs/pipeline.log and reports/last_run.json", file=sys.stderr)
        return 1
    except Exception as error:                                                   # noqa: BLE001
        print(f"\nUnexpected error: {type(error).__name__}: {error}\nDetails: logs/pipeline.log", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
