"""
src/logger.py  -  one place that decides how the project writes its progress messages.

WHAT THIS FILE IS
    A small helper, `get_logger(name)`, that gives every module a logger. All loggers write to two places:
    the screen (short, only the important lines) and the file logs/pipeline.log (everything, with the time).

WHY WE NEED IT
    `print()` disappears when the window closes. A log file stays. When a pipeline fails at night,
    the log file is the first place to look: it says what ran, how long it took and where it stopped.

WHAT HAPPENS WHEN IT RUNS
    The first call to `get_logger` sets up the screen. A pipeline run (run_pipeline.py) also switches the log file on.
    Later calls reuse the same destinations, so you never get every message printed twice.

EXPECTED OUTCOME
    Lines such as:   2026-10-14 15:02:11 | INFO    | atlantic.etl.run_pipeline | [4/9] Clean: repairs and chart runs

MOTIVE
    A pipeline you can not see inside is a pipeline you can not trust.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config import LOGS_DIR

ROOT_NAME = "atlantic"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_dir=None, console_level=logging.INFO, to_file=True):
    """(Re)build the destinations: the screen and (if to_file) `<log_dir>/pipeline.log`. Calling it again replaces the old ones."""
    root = logging.getLogger(ROOT_NAME)
    root.setLevel(logging.DEBUG)          # the root lets everything through; each destination filters for itself
    root.propagate = False                # do not hand the messages to Python's root logger (that would print them twice)
    close_logging()                       # remove old destinations first

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    # Destination 1: the screen. On Windows the console may not know characters such as "e with accent",
    # so we ask it to show a ? instead of crashing.
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:                      # not every stream supports reconfigure (for example inside some notebooks)
        pass
    screen = logging.StreamHandler(sys.stdout)
    screen.setLevel(console_level)
    screen.setFormatter(formatter)
    root.addHandler(screen)

    # Destination 2: the log file (utf-8 so that artist names with accents are stored correctly).
    if not to_file:
        return root
    folder = Path(log_dir) if log_dir else LOGS_DIR
    try:
        folder.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(folder / "pipeline.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as error:               # a read-only folder must never stop the analysis
        root.warning("Could not open the log file in %s (%s). Logging to the screen only.", folder, error)
    return root


def close_logging(keep_screen=False):
    """Close and remove the destinations (the log file is released; needed on Windows, where an open file can not be deleted)."""
    root = logging.getLogger(ROOT_NAME)
    for handler in list(root.handlers):
        if keep_screen and not isinstance(handler, logging.FileHandler):
            continue                                                  # keep printing to the screen
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def get_logger(name):
    """Return the logger of one module, e.g. get_logger(__name__). Sets everything up on first use."""
    root = logging.getLogger(ROOT_NAME)
    if not root.handlers:
        configure_logging(to_file=False)          # simple use (a notebook, a test): the screen only. A pipeline run adds the file.
    short = name[4:] if name.startswith("src.") else name       # "src.etl.clean" -> "etl.clean"
    return logging.getLogger(f"{ROOT_NAME}.{short}")
