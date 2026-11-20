"""
tests/test_sql_kpis.py  -  the 12 SQL KPI queries and the 8 views, checked against pandas on the REAL data.

WHAT THIS FILE IS
    sql/03_kpi_queries.sql recomputes the project KPIs "the long way" (GROUP BY, window functions, gaps-and-islands, HHI).
    Here every query is executed on a SQLite copy of the real warehouse and its answer is compared with the same number
    calculated from the pandas tables. Two completely different tools (SQL and pandas) must give the same answer.

WHY WE NEED IT
    The brief's KPIs will be shown in four places (notebooks, SQL, Streamlit, Power BI). If SQL and pandas disagree, one of
    them is wrong and the dashboards would contradict each other.

WHAT HAPPENS WHEN IT RUNS
    The session fixture `real_sqlite` loads the real tables once; each test runs one query and compares the result.

EXPECTED OUTCOME
    All tests pass. (Skipped automatically if data/raw/Atlantic_United_States.csv is not in the project.)

MOTIVE
    One number, one meaning, whichever tool you ask.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import SQL_DIR
from src.etl.load import read_sql_statements

STATEMENTS = read_sql_statements(SQL_DIR / "03_kpi_queries.sql")


def q(engine, number):
    """Run query Q<number> of the KPI file and return the answer as a DataFrame."""
    with engine.connect() as connection:
        result = connection.exec_driver_sql(STATEMENTS[number - 1])
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


def view(engine, name):
    with engine.connect() as connection:
        result = connection.exec_driver_sql(f"SELECT * FROM {name}")
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


@pytest.fixture(scope="module")
def t(real_result):
    return real_result.tables


@pytest.fixture(scope="module")
def fact_with_label(t):
    return t["fact_chart_daily"].merge(t["dim_track"][["track_id", "track_label"]], on="track_id")


def test_q01_days_on_chart(real_sqlite, t):
    sql = q(real_sqlite, 1)
    expected = t["dim_track"].sort_values(["days_on_chart", "track_id"], ascending=[False, True]).head(10)
    assert sql["track_label"].tolist() == expected["track_label"].tolist()
    assert sql["days_on_chart"].tolist() == expected["days_on_chart"].tolist()
    assert (sql.loc[0, "track_label"], sql.loc[0, "days_on_chart"]) == ("Something in the Orange by Zach Bryan", 536)


def test_q02_average_rank_and_best_rank(real_sqlite, t):
    sql = q(real_sqlite, 2)
    dt = t["dim_track"]
    expected = dt[dt["days_on_chart"] >= 30].sort_values(["avg_rank", "track_id"]).head(10)
    assert sql["track_label"].tolist() == expected["track_label"].tolist()
    np.testing.assert_allclose(sql["avg_rank"].astype(float), expected["avg_rank"], atol=0.0051)
    assert sql["best_rank"].tolist() == expected["best_rank"].tolist()


def test_q03_rank_volatility_matches_the_sample_standard_deviation(real_sqlite, t):
    sql = q(real_sqlite, 3)
    positions = t["fact_chart_daily"].groupby("track_id")["position"].agg(["std", "size"])
    ranked = positions[positions["size"] >= 30].sort_values(["std"], ascending=False, kind="stable")
    expected = ranked.reset_index().sort_values(["std", "track_id"], ascending=[False, True]).head(10)
    labels = t["dim_track"].set_index("track_id")["track_label"]
    assert sql["track_label"].tolist() == [labels[i] for i in expected["track_id"]]
    np.testing.assert_allclose(sql["rank_volatility_index"].astype(float), expected["std"], rtol=1e-9)


def test_q04_rolling_trend_recomputed_with_a_window_function_matches_the_stored_one(real_sqlite, t):
    sql = q(real_sqlite, 4)
    assert int(sql.loc[0, "rows_that_differ"]) == 0
    assert int(sql.loc[0, "rows_compared"]) == int(t["fact_chart_daily"]["popularity_trend_7d"].notna().sum())


def test_q05_artist_dominance_index(real_sqlite, t):
    sql = q(real_sqlite, 5)
    expected = t["dim_artist"].sort_values(["chart_points_primary", "artist_name"], ascending=[False, True]).head(10)
    assert sql["artist_name"].tolist() == expected["artist_name"].tolist()
    assert sql["chart_points"].astype(int).tolist() == expected["chart_points_primary"].tolist()
    np.testing.assert_allclose(sql["artist_dominance_index_pct"].astype(float), expected["dominance_index_primary"], atol=0.00006)


def test_q06_explicit_share_per_month(real_sqlite, t):
    sql = q(real_sqlite, 6)
    fact = t["fact_chart_daily"].merge(t["dim_date"][["date", "year_month"]], on="date")
    expected = fact.groupby("year_month")["is_explicit"].mean().mul(100).reset_index()
    assert sql["year_month"].tolist() == expected["year_month"].tolist()
    np.testing.assert_allclose(sql["explicit_share_pct"].astype(float), expected["is_explicit"], rtol=1e-9)


def test_q07_market_concentration_hhi(real_sqlite, t):
    sql = q(real_sqlite, 7)
    shares = t["dim_artist"]["chart_points_primary"] / t["fact_chart_daily"]["chart_points"].sum() * 100
    shares = shares[shares > 0]
    assert int(sql.loc[0, "artists"]) == len(shares) == 243
    assert float(sql.loc[0, "hhi"]) == pytest.approx(float((shares ** 2).sum()), rel=1e-9)
    assert 254 < float(sql.loc[0, "hhi"]) < 255                                # far below 1,500: an unconcentrated market


def test_q08_top_three_songs_of_every_month(real_sqlite, t, fact_with_label):
    sql = q(real_sqlite, 8)
    months = fact_with_label.merge(t["dim_date"][["date", "year_month"]], on="date")
    monthly = months.groupby(["year_month", "track_id", "track_label"]).size().reset_index(name="days_in_month")
    monthly = monthly.sort_values(["year_month", "days_in_month", "track_label"], ascending=[True, False, True])
    expected = monthly.groupby("year_month").head(3).reset_index(drop=True)
    assert len(sql) == len(expected) == 19 * 3
    assert sql["track_label"].tolist() == expected["track_label"].tolist()
    assert sql["days_in_month"].tolist() == expected["days_in_month"].tolist()


def test_q09_biggest_one_day_climbs(real_sqlite, fact_with_label):
    sql = q(real_sqlite, 9)
    moves = fact_with_label[fact_with_label["rank_change"].notna()]
    expected = moves.sort_values(["rank_change", "date", "track_label"], ascending=[False, True, True]).head(10)
    assert sql["track_label"].tolist() == expected["track_label"].tolist()
    assert sql["places_climbed"].astype(int).tolist() == expected["rank_change"].astype(int).tolist()


def test_q10_gaps_and_islands_count_the_runs(real_sqlite, t):
    sql = q(real_sqlite, 10)
    dt = t["dim_track"]
    expected = dt.sort_values(["n_runs", "days_on_chart", "track_id"], ascending=[False, False, True]).head(10)
    assert sql["track_label"].tolist() == expected["track_label"].tolist()
    assert sql["n_runs"].astype(int).tolist() == expected["n_runs"].tolist()
    assert sql["longest_run_days"].astype(int).tolist() == expected["longest_run_days"].tolist()
    assert sql["days_on_chart"].astype(int).tolist() == expected["days_on_chart"].tolist()


def test_q11_weekday_pattern(real_sqlite, t):
    sql = q(real_sqlite, 11)
    summary = t["daily_summary"].merge(t["dim_date"][["date", "day_of_week"]], on="date")
    summary = summary[summary["snapshot_index"] > 1]
    expected = summary.groupby("day_of_week").agg(snapshots=("date", "size"), turnover=("turnover_rate_pct", "mean"))
    assert sql["snapshots"].astype(int).tolist() == expected["snapshots"].tolist()
    np.testing.assert_allclose(sql["avg_turnover_rate_pct"].astype(float), expected["turnover"], rtol=1e-9)


def test_q12_row_counts_reconcile(real_sqlite):
    sql = q(real_sqlite, 12).set_index("table_name")["rows_count"]
    assert sql.to_dict() == {"dim_date": 559, "dim_track": 944, "dim_artist": 283, "bridge_track_artist": 1_105,
                             "fact_chart_daily": 27_750, "daily_summary": 555}
    assert int(sql.sum()) == 31_196


# ================================================================================ views
def test_view_row_counts_and_totals(real_sqlite, t):
    assert len(view(real_sqlite, "vw_chart_daily")) == 27_750
    assert len(view(real_sqlite, "vw_song_scorecard")) == 944
    assert len(view(real_sqlite, "vw_artist_songs")) == 1_105
    assert len(view(real_sqlite, "vw_monthly_trend")) == 19                    # May 2024 ... November 2025
    assert view(real_sqlite, "vw_movement_by_month")["rows_count"].sum() == 27_750
    assert view(real_sqlite, "vw_longevity_summary")["songs"].sum() == 944
    leaderboard = view(real_sqlite, "vw_artist_leaderboard")
    assert len(leaderboard) == 283
    assert leaderboard["cumulative_dominance_pct"].iloc[-1] == pytest.approx(100.0, abs=0.01)
    assert leaderboard["cumulative_dominance_pct"].is_monotonic_increasing


def test_the_song_scorecard_ranks_follow_the_kpis(real_sqlite):
    card = view(real_sqlite, "vw_song_scorecard")
    top = card.sort_values("longevity_rank").iloc[0]
    assert top["track_label"] == "Something in the Orange by Zach Bryan" and top["longevity_rank"] == 1
    assert card["average_rank_position"].min() == 1 and card["chart_points_rank"].min() == 1
