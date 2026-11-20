"""
tests/test_documentation.py  -  tests for the data dictionary and the KPI dictionary (step 7 of the pipeline).

WHAT THIS FILE IS
    Checks that every column of every output table is described, that the numbers inside the descriptions follow the data,
    and that the pipeline stops when somebody adds a column and forgets to describe it.

WHY WE NEED IT
    The dictionaries are the user manual of the data (they also feed the Power BI and Streamlit "About the data" pages).
    A manual that silently misses a column, or that still says "555 dates" for a different file, is worse than no manual.

WHAT HAPPENS WHEN IT RUNS
    The dictionaries produced for the synthetic chart (11 snapshots, 1 missing day) and for the real file are inspected.

EXPECTED OUTCOME
    All tests pass: 173 rows for 173 columns, no empty definition, no leftover {placeholder}.

MOTIVE
    Documentation that is checked by a test stays true.
"""
import pandas as pd
import pytest

from src.config import TABLE_COLUMNS
from src.etl.documentation import build_data_dictionary, build_kpi_dictionary
from src.etl.errors import DataIntegrityError

BRIEF_KPIS = ["Days on Chart", "Average Rank", "Best Rank Achieved", "Rank Volatility Index", "Popularity Score Trend",
              "Artist Dominance Index", "Explicit Content Share"]


def test_one_row_per_column_of_every_table(synthetic_result):
    dictionary = synthetic_result.data_dictionary
    assert len(dictionary) == sum(len(columns) for columns in TABLE_COLUMNS.values()) == 173
    for table, columns in TABLE_COLUMNS.items():
        assert dictionary.loc[dictionary["table"] == table, "column"].tolist() == columns


def test_the_dictionary_columns(synthetic_result):
    assert list(synthetic_result.data_dictionary.columns) == ["table", "column", "data_type", "definition", "example"]


def test_every_definition_is_filled_and_has_no_leftover_placeholder(synthetic_result):
    definitions = synthetic_result.data_dictionary["definition"]
    assert definitions.notna().all() and (definitions.str.strip() != "").all()
    assert not definitions.str.contains(r"[{}]", regex=True).any()


def test_types_are_plain_words_that_do_not_depend_on_the_pandas_version(synthetic_result, real_result):
    allowed = {"whole number", "decimal number", "true/false", "date", "text"}
    assert set(synthetic_result.data_dictionary["data_type"]) <= allowed
    assert set(real_result.data_dictionary["data_type"]) <= allowed
    kinds = synthetic_result.data_dictionary.set_index(["table", "column"])["data_type"]
    assert kinds[("fact_chart_daily", "position")] == "whole number"
    assert kinds[("fact_chart_daily", "date")] == "date"
    assert kinds[("fact_chart_daily", "is_explicit")] == "true/false"
    assert kinds[("dim_track", "avg_rank")] == "decimal number"
    assert kinds[("dim_track", "track_label")] == "text"
    assert kinds[("dim_date", "snapshot_index")] == "whole number"          # a nullable integer column is still a whole number


def test_numbers_in_the_definitions_follow_the_data(synthetic_result, real_result):
    synthetic = synthetic_result.data_dictionary.set_index(["table", "column"])["definition"]
    assert "11 for the last" in synthetic[("dim_date", "snapshot_index")]
    assert "1 missing days" in synthetic[("dim_date", "has_snapshot")]
    assert "all 550 chart slots" in synthetic[("dim_artist", "share_of_slots_primary_pct")]

    real = real_result.data_dictionary.set_index(["table", "column"])["definition"]
    assert "555 for the last" in real[("dim_date", "snapshot_index")]
    assert "4 missing days" in real[("dim_date", "has_snapshot")]
    assert "all 27,750 chart slots" in real[("dim_artist", "share_of_slots_primary_pct")]


def test_examples_are_short(synthetic_result):
    assert synthetic_result.data_dictionary["example"].str.len().max() <= 45


def test_a_column_without_a_definition_stops_the_pipeline(tables):
    tables["dim_date"]["mystery_column"] = 1
    with pytest.raises(DataIntegrityError) as error:
        build_data_dictionary(tables)
    assert "mystery_column" in str(error.value) and "src/etl/documentation.py" in str(error.value)


def test_the_dictionary_never_changes_the_tables(tables, synthetic_tables):
    build_data_dictionary(tables)
    for name, table in synthetic_tables.items():
        pd.testing.assert_frame_equal(tables[name], table)


# ================================================================================ the KPI dictionary
def test_kpi_dictionary_has_the_kpis_of_the_brief():
    kpis = build_kpi_dictionary()
    assert list(kpis.columns) == ["kpi", "where_to_find_it", "formula", "interpretation"]
    assert len(kpis) == 11
    assert set(BRIEF_KPIS) <= set(kpis["kpi"])
    assert kpis["kpi"].is_unique
    assert (kpis.apply(lambda column: column.str.strip().ne("")).all()).all()


def test_every_kpi_points_to_a_real_table_and_column(synthetic_tables):
    for _, row in build_kpi_dictionary().iterrows():
        table, column = row["where_to_find_it"].split(".")
        assert table in synthetic_tables, row["kpi"]
        assert column in synthetic_tables[table].columns, f"{row['kpi']}: {row['where_to_find_it']} does not exist"
