"""
src/etl/errors.py  -  the few named errors the pipeline can raise.

WHAT THIS FILE IS
    Four tiny error classes. They carry no logic; their NAME tells you what went wrong.

WHY WE NEED IT
    A pipeline that only says "AssertionError" is hard to debug at 2 a.m. With named errors, the log says
    "DataContractError: column 'position' is missing" and everyone knows where to look.

WHAT HAPPENS WHEN IT RUNS
    Nothing. Other files raise these errors when a rule is broken.

EXPECTED OUTCOME
    You will only meet these classes when something is wrong with the data or the database.

MOTIVE
    Fail fast and fail clearly: stop the run before bad data reaches a dashboard.
"""
from __future__ import annotations


class PipelineError(Exception):
    """Base class: every error raised on purpose by this project is a PipelineError."""


class DataContractError(PipelineError):
    """The raw file breaks the data contract (missing column, unreadable dates, rank outside 1-50 ...)."""


class DataIntegrityError(PipelineError):
    """A calculated table breaks a rule that must always hold (entries != exits, shares != 100% ...)."""


class LoadError(PipelineError):
    """Loading into the database failed, or the loaded data does not match what we sent."""
