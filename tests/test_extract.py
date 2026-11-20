"""
tests/test_extract.py  -  tests for step 1 of the pipeline: reading the raw file and fingerprinting it.

WHAT THIS FILE IS
    Checks for `read_raw` (open the CSV, or stop with a clear message) and `file_fingerprint` (name, size, SHA-256).

WHY WE NEED IT
    The pipeline must never start from a half-read or wrongly decoded file, and the fingerprint is our proof of WHICH
    version of the data produced a dashboard. A wrong fingerprint would make that proof worthless.

WHAT HAPPENS WHEN IT RUNS
    Tiny files are written into a temporary folder (its name contains a space, like many Windows folders) and read back.

EXPECTED OUTCOME
    All tests pass. Every bad file ends in a DataContractError with a message a person can act on.

MOTIVE
    Fail at the very first step, in plain words, instead of three steps later with a confusing pandas error.
"""
import hashlib

import pandas as pd
import pytest

from src.etl.errors import DataContractError
from src.etl.extract import file_fingerprint, read_raw


@pytest.fixture()
def folder(tmp_path):
    spaced = tmp_path / "my data folder"
    spaced.mkdir()
    return spaced


def test_read_raw_returns_the_file_exactly_as_stored(folder):
    path = folder / "ok.csv"
    path.write_text("date,song\n18-05-2024,Beyoncé\n", encoding="utf-8")
    frame = read_raw(path)
    assert list(frame.columns) == ["date", "song"]
    assert frame.loc[0, "date"] == "18-05-2024"          # still text: dates are converted later, on purpose
    assert frame.loc[0, "song"] == "Beyoncé"


def test_read_raw_accepts_a_file_that_starts_with_a_byte_order_mark(folder):
    # Excel's "CSV UTF-8" adds an invisible marker at the start; the first column name must not be damaged by it
    path = folder / "bom.csv"
    path.write_bytes("﻿date,song\n18-05-2024,x\n".encode("utf-8"))
    assert list(read_raw(path).columns) == ["date", "song"]


def test_a_missing_file_gives_a_message_that_names_the_folder(folder):
    with pytest.raises(DataContractError) as error:
        read_raw(folder / "nothing.csv")
    assert "not found" in str(error.value)
    assert "my data folder" in str(error.value)


def test_a_zero_byte_file_is_refused(folder):
    path = folder / "empty.csv"
    path.write_bytes(b"")
    with pytest.raises(DataContractError, match="could not be read"):
        read_raw(path)


def test_a_file_with_a_header_but_no_rows_is_refused(folder):
    path = folder / "header_only.csv"
    path.write_text("date,song\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="no rows"):
        read_raw(path)


def test_a_file_in_another_encoding_is_refused_with_a_clear_message(folder):
    path = folder / "latin1.csv"
    path.write_bytes(b"date,song\n01-01-2024,Beyonc\xe9\n")        # "é" written the old Windows-1252 / Latin-1 way
    with pytest.raises(DataContractError, match="could not be read"):
        read_raw(path)


def test_the_fingerprint_of_a_known_text(folder):
    path = folder / "abc.txt"
    path.write_bytes(b"abc")
    result = file_fingerprint(path)
    assert result["name"] == "abc.txt"
    assert result["size_bytes"] == 3
    assert result["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"   # the textbook SHA-256 of "abc"


def test_the_fingerprint_does_not_depend_on_the_piece_size(folder):
    path = folder / "long.bin"
    path.write_bytes(bytes(range(256)) * 50)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert file_fingerprint(path, chunk_size=7)["sha256"] == expected
    assert file_fingerprint(path)["sha256"] == expected


def test_one_changed_character_changes_the_fingerprint(folder):
    first, second = folder / "a.csv", folder / "b.csv"
    first.write_text("date,song\n18-05-2024,Song A\n", encoding="utf-8")
    second.write_text("date,song\n18-05-2024,Song B\n", encoding="utf-8")
    assert file_fingerprint(first)["sha256"] != file_fingerprint(second)["sha256"]


def test_reading_does_not_modify_the_file(folder):
    path = folder / "ok.csv"
    path.write_text("date,song\n18-05-2024,x\n", encoding="utf-8")
    before = file_fingerprint(path)["sha256"]
    read_raw(path)
    assert file_fingerprint(path)["sha256"] == before


def test_the_real_file_has_the_promised_shape(real_csv):
    frame = read_raw(real_csv)
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (27_800, 10)
