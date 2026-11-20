"""
src/etl/extract.py  -  step 1 of the pipeline: read the raw CSV file and fingerprint it.

WHAT THIS FILE IS
    Two small functions. `read_raw` opens the CSV and returns it as a table. `file_fingerprint` returns the
    file name, its size and a SHA-256 "fingerprint" (a long code that changes if even one character of the file changes).

WHY WE NEED IT
    Extract is the "E" of ETL. Keeping it separate means the rest of the pipeline never needs to know
    where the data came from (a CSV today, an API or a database tomorrow). The fingerprint lets us prove,
    months later, WHICH version of the file produced a given dashboard.

WHAT HAPPENS WHEN IT RUNS
    The CSV is read exactly as it is: no cleaning, no type changes. If the file is missing or empty,
    we stop with a clear message instead of a confusing pandas error.

EXPECTED OUTCOME
    A table of 27,800 rows x 10 columns for the project file, and a dictionary such as
    {"name": "Atlantic_United_States.csv", "size_bytes": 3_5xx_xxx, "sha256": "ab12..."}.

MOTIVE
    Raw data is read-only evidence. Everything that happens to it later is recorded, never hidden.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.etl.errors import DataContractError
from src.logger import get_logger

log = get_logger(__name__)


def read_raw(path):
    """Read the raw CSV into a DataFrame, exactly as stored. Raises DataContractError if it can not be read."""
    path = Path(path)
    if not path.exists():
        raise DataContractError(
            f"Raw file not found: {path}\n"
            f"Copy Atlantic_United_States.csv into the folder '{path.parent}' and run the pipeline again.")
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DataContractError(f"The file {path.name} could not be read as a CSV: {error}") from error
    if df.empty:
        raise DataContractError(f"The file {path.name} was read but it contains no rows.")
    log.info("Read %s rows x %s columns from %s", f"{len(df):,}", df.shape[1], path.name)
    return df


def file_fingerprint(path, chunk_size=1024 * 1024):
    """Name, size in bytes and SHA-256 of a file. The file is read in 1 MB pieces so that big files do not fill the memory."""
    path = Path(path)
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            sha.update(block)
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha.hexdigest()}
