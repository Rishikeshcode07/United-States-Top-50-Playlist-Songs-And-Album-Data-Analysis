-- =====================================================================================================
-- sql/03_kpi_queries.sql  -  the KPIs of the project brief, written in SQL (12 ready-to-run queries)
--
-- WHAT THIS FILE IS
--     Twelve queries that recompute the KPIs "the long way" from the fact table, plus a few SQL techniques that every
--     analyst should own: window functions (ROW_NUMBER, LAG, AVG ... OVER), CTEs (WITH ...), gaps-and-islands and a
--     concentration index (HHI). The pipeline does NOT run this file; you do, in any SQL tool.
--
-- WHY WE NEED IT
--     1. Proof: the SQL answers match the Python tables (the automatic tests compare them number by number).
--     2. Skill: interviews and real projects ask for exactly these patterns.
--     3. Reuse: paste a query into Power BI (Get data > SQL) or into a Streamlit page.
--
-- WHAT HAPPENS WHEN IT RUNS
--     Open the database (SQLite: data/warehouse/atlantic_top50.db, or your PostgreSQL / MySQL database) in your SQL tool,
--     and run one query at a time. Each query starts with a comment that says what it answers.
--
-- EXPECTED OUTCOME
--     Q01 first row: "Something in the Orange by Zach Bryan" with 536 days (the longest-charting song in the file).
--     Q04: 0 rows differ from the stored trend.   Q07: HHI of about 254 (below 1,500 = an unconcentrated market).
--     Q12: six row counts that add up to 31,196.
--
-- MOTIVE
--     Be able to answer the same question in Python, SQL and Power BI, and get the same number.
--
-- PORTABILITY NOTES (SQLite / PostgreSQL / MySQL 8)
--   * ROUND(x, 2) needs a DECIMAL in PostgreSQL, so we write ROUND(CAST(x AS DECIMAL(12,4)), 2).
--     The Python tables round half UP (4.625 -> 4.63) like SQL, Excel and Power BI. PostgreSQL and MySQL use exact
--     decimals; SQLite uses binary numbers, so in rare cases (for example 2.675) its ROUND can differ by 0.01.
--   * Sample standard deviation is written with SUM / COUNT so that it runs everywhere. PostgreSQL and MySQL also have
--     STDDEV_SAMP(position); SQLite has no standard-deviation function at all.
--   * 100.0 * a / b avoids integer division. CASE WHEN flag THEN 1.0 ELSE 0.0 END turns a True/False column into 1/0.
-- =====================================================================================================


-- Q01 | KPI "Days on Chart": how many snapshots did each song appear on? (top 10; equals dim_track.days_on_chart)
SELECT t.track_label,
       COUNT(*) AS days_on_chart
FROM fact_chart_daily f
JOIN dim_track t ON t.track_id = f.track_id
GROUP BY t.track_id, t.track_label
ORDER BY days_on_chart DESC, t.track_id
LIMIT 10;


-- Q02 | KPI "Average Rank" and "Best Rank Achieved" for songs with at least 30 chart days (lower rank number = better)
SELECT t.track_label,
       ROUND(CAST(AVG(f.position) AS DECIMAL(12,4)), 2) AS avg_rank,
       MIN(f.position)                                  AS best_rank,
       COUNT(*)                                         AS days_on_chart
FROM fact_chart_daily f
JOIN dim_track t ON t.track_id = f.track_id
GROUP BY t.track_id, t.track_label
HAVING COUNT(*) >= 30
ORDER BY avg_rank ASC, t.track_id
LIMIT 10;


-- Q03 | KPI "Rank Volatility Index": sample standard deviation of position (n - 1 in the denominator, like pandas .std())
--      Variance = (sum of squares - (sum)^2 / n) / (n - 1). In PostgreSQL / MySQL you can write STDDEV_SAMP(f.position).
SELECT t.track_label,
       COUNT(*) AS days_on_chart,
       SQRT((SUM(1.0 * f.position * f.position) - SUM(1.0 * f.position) * SUM(1.0 * f.position) / COUNT(*))
            / (COUNT(*) - 1)) AS rank_volatility_index
FROM fact_chart_daily f
JOIN dim_track t ON t.track_id = f.track_id
GROUP BY t.track_id, t.track_label
HAVING COUNT(*) >= 30
ORDER BY rank_volatility_index DESC, t.track_id
LIMIT 10;


-- Q04 | KPI "Popularity Score Trend": 7-day rolling average inside one chart run, recomputed with a WINDOW FUNCTION.
--      Trick: chart_day_number - days_in_stint is constant inside a run and different for every run of the same song,
--      so it works as a run id. The query counts how many rows differ from the stored popularity_trend_7d (0 expected;
--      the stored value is rounded to 2 decimals, so differences up to 0.005 are rounding, not errors).
WITH rolled AS (
    SELECT f.popularity_trend_7d AS stored_trend,
           AVG(f.popularity_clean) OVER (
               PARTITION BY f.track_id, (f.chart_day_number - f.days_in_stint)
               ORDER BY f.date
               ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS recomputed_trend
    FROM fact_chart_daily f
)
SELECT COUNT(*) AS rows_compared,
       SUM(CASE WHEN ABS(recomputed_trend - stored_trend) <= 0.0051 THEN 0 ELSE 1 END) AS rows_that_differ
FROM rolled
WHERE stored_trend IS NOT NULL AND recomputed_trend IS NOT NULL;


-- Q05 | KPI "Artist Dominance Index" (lead-artist basis): an artist's chart points as a share of ALL chart points
SELECT b.artist_name,
       SUM(f.chart_points) AS chart_points,
       100.0 * SUM(f.chart_points) / (SELECT SUM(chart_points) FROM fact_chart_daily) AS artist_dominance_index_pct
FROM fact_chart_daily f
JOIN bridge_track_artist b ON b.track_id = f.track_id AND b.is_primary = TRUE
GROUP BY b.artist_name
ORDER BY chart_points DESC, b.artist_name
LIMIT 10;


-- Q06 | KPI "Explicit Content Share": percentage of chart rows flagged explicit, per month
SELECT d.year_month,
       100.0 * AVG(CASE WHEN f.is_explicit THEN 1.0 ELSE 0.0 END) AS explicit_share_pct
FROM fact_chart_daily f
JOIN dim_date d ON d.date = f.date
GROUP BY d.year_month
ORDER BY d.year_month;


-- Q07 | Market concentration: Herfindahl-Hirschman Index (HHI) of the chart = sum of squared artist shares (in %).
--      Below 1,500 = unconcentrated, 1,500 - 2,500 = moderately, above 2,500 = highly concentrated.
WITH shares AS (
    SELECT b.artist_name,
           100.0 * SUM(f.chart_points) / (SELECT SUM(chart_points) FROM fact_chart_daily) AS share_pct
    FROM fact_chart_daily f
    JOIN bridge_track_artist b ON b.track_id = f.track_id AND b.is_primary = TRUE
    GROUP BY b.artist_name
)
SELECT COUNT(*)                  AS artists,
       SUM(share_pct * share_pct) AS hhi
FROM shares;


-- Q08 | Top 3 songs of every month by number of chart days (ROW_NUMBER inside each month = "top N per group")
WITH monthly AS (
    SELECT d.year_month, t.track_label, COUNT(*) AS days_in_month
    FROM fact_chart_daily f
    JOIN dim_date  d ON d.date = f.date
    JOIN dim_track t ON t.track_id = f.track_id
    GROUP BY d.year_month, t.track_id, t.track_label
),
ranked AS (
    SELECT year_month, track_label, days_in_month,
           ROW_NUMBER() OVER (PARTITION BY year_month ORDER BY days_in_month DESC, track_label) AS place
    FROM monthly
)
SELECT year_month, place, track_label, days_in_month
FROM ranked
WHERE place <= 3
ORDER BY year_month, place;


-- Q09 | The 10 biggest one-day climbs, recomputed with LAG (the previous row of the same song).
--      rank_change is only filled when the song was on the chart the day before, so we keep exactly those rows.
WITH moves AS (
    SELECT f.date, t.track_label, f.position, f.rank_change,
           LAG(f.position) OVER (PARTITION BY f.track_id ORDER BY f.date) AS prev_position
    FROM fact_chart_daily f
    JOIN dim_track t ON t.track_id = f.track_id
)
SELECT date, track_label, prev_position, position, prev_position - position AS places_climbed
FROM moves
WHERE prev_position IS NOT NULL AND rank_change IS NOT NULL
ORDER BY places_climbed DESC, date, track_label
LIMIT 10;


-- Q10 | Gaps and islands: how many separate chart runs did each song have, and how long was the longest? (top 10 by runs)
WITH runs AS (
    SELECT track_id, (chart_day_number - days_in_stint) AS run_marker, COUNT(*) AS run_days
    FROM fact_chart_daily
    GROUP BY track_id, (chart_day_number - days_in_stint)
)
SELECT t.track_label,
       COUNT(*)        AS n_runs,
       MAX(r.run_days) AS longest_run_days,
       SUM(r.run_days) AS days_on_chart
FROM runs r
JOIN dim_track t ON t.track_id = r.track_id
GROUP BY t.track_id, t.track_label
ORDER BY n_runs DESC, days_on_chart DESC, t.track_id
LIMIT 10;


-- Q11 | Which weekday has the most churn? (uses the view vw_weekday_pattern; Monday = 1 ... Sunday = 7)
SELECT day_name, snapshots, avg_entries, avg_turnover_rate_pct, explicit_share
FROM vw_weekday_pattern
ORDER BY day_of_week;


-- Q12 | Reconciliation: how many rows does each table hold? (a quick health check after every load)
SELECT 'dim_date'            AS table_name, COUNT(*) AS rows_count FROM dim_date
UNION ALL SELECT 'dim_track',            COUNT(*) FROM dim_track
UNION ALL SELECT 'dim_artist',           COUNT(*) FROM dim_artist
UNION ALL SELECT 'bridge_track_artist',  COUNT(*) FROM bridge_track_artist
UNION ALL SELECT 'fact_chart_daily',     COUNT(*) FROM fact_chart_daily
UNION ALL SELECT 'daily_summary',        COUNT(*) FROM daily_summary;
