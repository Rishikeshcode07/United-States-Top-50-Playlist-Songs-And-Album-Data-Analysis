-- =====================================================================================================
-- sql/02_views.sql  -  analysis-ready VIEWS on top of the star schema
--
-- WHAT THIS FILE IS
--     Eight saved queries ("views"). A view stores no data; it is a named SELECT that always shows the current tables.
--
--         vw_chart_daily        the star "flattened": every daily row with its song and calendar attributes
--         vw_song_scorecard     one row per song with the six KPIs and league-table ranks
--         vw_artist_leaderboard one row per artist with dominance rank and cumulative share
--         vw_artist_songs       one row per (artist, song): the many-to-many relation, ready for drill-through
--         vw_monthly_trend      one row per month: average popularity, explicit/collab/single share, turnover
--         vw_weekday_pattern    one row per weekday: turnover and content mix
--         vw_movement_by_month  chart movements (Up/Down/Same/New entry/Re-entry) counted per month
--         vw_longevity_summary  how long songs stay, by longevity class and release format
--
-- WHY WE NEED IT
--     Power BI, Excel and analysts should not repeat the same joins and definitions again and again. A view puts
--     one agreed definition in the database. If the definition changes, it changes in ONE place.
--
-- WHAT HAPPENS WHEN IT RUNS
--     The loader runs this file after the tables are filled. Each view is dropped (if it exists) and created again.
--
-- EXPECTED OUTCOME
--     SELECT * FROM vw_monthly_trend ORDER BY year_month;     -> 19 rows (May 2024 ... Nov 2025)
--     SELECT * FROM vw_song_scorecard LIMIT 10;               -> the first ten songs with their KPIs
--
-- MOTIVE
--     "One source of truth" is only real if the truth lives in one place.
--
-- Portable SQL (SQLite, PostgreSQL, MySQL 8): no database-specific functions. Two habits make it portable:
--   * 100.0 * a / b instead of a / b           (integer division would round the result down in SQLite and PostgreSQL)
--   * CASE WHEN flag THEN 1.0 ELSE 0.0 END     (turns a True/False column into 1/0 so that AVG() gives a share)
-- Averages of the daily averages are exact here because every snapshot has exactly 50 rows.
-- =====================================================================================================

DROP VIEW IF EXISTS vw_chart_daily;
DROP VIEW IF EXISTS vw_song_scorecard;
DROP VIEW IF EXISTS vw_artist_leaderboard;
DROP VIEW IF EXISTS vw_artist_songs;
DROP VIEW IF EXISTS vw_monthly_trend;
DROP VIEW IF EXISTS vw_weekday_pattern;
DROP VIEW IF EXISTS vw_movement_by_month;
DROP VIEW IF EXISTS vw_longevity_summary;


-- ---------- 1. the star, flattened: one row per date x position, with song and calendar attributes ----------
CREATE VIEW vw_chart_daily AS
SELECT f.date,
       d.year, d.month, d.month_name, d.year_month, d.quarter, d.day_of_week, d.day_name, d.is_weekend,
       f.track_id, t.track_label, t.song_clean, t.artist_credit, t.primary_artist, t.is_collab,
       t.release_format, t.longevity_class, t.run_status,
       f.position, f.rank_tier, f.chart_points, f.popularity_clean, f.popularity_trend_7d,
       f.rank_change, f.movement, f.in_top10, f.is_number_one, f.chart_day_number, f.days_in_stint, f.is_explicit
FROM fact_chart_daily f
JOIN dim_track t ON t.track_id = f.track_id
JOIN dim_date  d ON d.date = f.date;


-- ---------- 2. one row per song with league-table ranks (RANK() leaves gaps after ties: 1, 2, 2, 4) ----------
CREATE VIEW vw_song_scorecard AS
SELECT t.track_id, t.track_label, t.song_clean, t.artist_credit, t.primary_artist, t.is_collab,
       t.release_format, t.is_explicit, t.longevity_class, t.run_status,
       t.days_on_chart, t.avg_rank, t.best_rank, t.rank_volatility_index, t.avg_daily_rank_move,
       t.avg_popularity, t.days_in_top10, t.days_at_number_one, t.n_runs, t.presence_ratio,
       RANK() OVER (ORDER BY t.days_on_chart DESC)                         AS longevity_rank,
       RANK() OVER (ORDER BY t.avg_rank ASC)                               AS average_rank_position,
       RANK() OVER (ORDER BY t.chart_points DESC)                          AS chart_points_rank
FROM dim_track t;


-- ---------- 3. artists ranked by dominance, with a running total (the "Pareto" curve of the chart) ----------
CREATE VIEW vw_artist_leaderboard AS
SELECT a.artist_name, a.unique_tracks, a.unique_tracks_as_primary, a.days_present, a.avg_rank, a.best_rank,
       a.days_in_top10, a.days_at_number_one, a.chart_points_primary, a.dominance_index_primary,
       RANK() OVER (ORDER BY a.dominance_index_primary DESC)               AS dominance_rank,
       SUM(a.dominance_index_primary) OVER (
           ORDER BY a.dominance_index_primary DESC, a.artist_name
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)               AS cumulative_dominance_pct
FROM dim_artist a;


-- ---------- 4. the many-to-many relation resolved: one row per (artist, song) ----------
CREATE VIEW vw_artist_songs AS
SELECT b.artist_name, b.artist_order, b.is_primary,
       t.track_id, t.track_label, t.days_on_chart, t.avg_rank, t.best_rank, t.avg_popularity, t.release_format,
       t.is_explicit, t.longevity_class
FROM bridge_track_artist b
JOIN dim_track t ON t.track_id = b.track_id;


-- ---------- 5. one row per month ----------
CREATE VIEW vw_monthly_trend AS
SELECT d.year_month, d.year, d.month,
       COUNT(*)                         AS snapshots,
       AVG(s.avg_popularity)            AS avg_popularity,
       AVG(s.explicit_share)            AS explicit_share,
       AVG(s.collab_share)              AS collab_share,
       AVG(s.single_share)              AS single_share,
       AVG(s.avg_duration_min)          AS avg_duration_min,
       AVG(s.turnover_rate_pct)         AS avg_turnover_rate_pct,
       SUM(s.new_entries + s.re_entries) AS total_entries,
       SUM(CASE WHEN s.is_high_turnover_day THEN 1 ELSE 0 END) AS high_turnover_days,
       AVG(s.unique_primary_artists)    AS avg_unique_lead_artists
FROM daily_summary s
JOIN dim_date d ON d.date = s.date
GROUP BY d.year_month, d.year, d.month;


-- ---------- 6. one row per weekday (the first snapshot has no "yesterday", so snapshot_index > 1 for entries) ----------
CREATE VIEW vw_weekday_pattern AS
SELECT d.day_of_week, d.day_name,
       COUNT(*)                         AS snapshots,
       AVG(s.new_entries + s.re_entries) AS avg_entries,
       AVG(s.turnover_rate_pct)         AS avg_turnover_rate_pct,
       AVG(s.explicit_share)            AS explicit_share,
       AVG(s.collab_share)              AS collab_share,
       AVG(s.avg_popularity)            AS avg_popularity
FROM daily_summary s
JOIN dim_date d ON d.date = s.date
WHERE s.snapshot_index > 1
GROUP BY d.day_of_week, d.day_name;


-- ---------- 7. how the chart moved, per month (long format: one row per month x movement) ----------
CREATE VIEW vw_movement_by_month AS
SELECT d.year_month, f.movement, COUNT(*) AS rows_count,
       100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY d.year_month) AS share_of_month_pct
FROM fact_chart_daily f
JOIN dim_date d ON d.date = f.date
GROUP BY d.year_month, f.movement;


-- ---------- 8. longevity classes by release format (use run_status = 'Complete history observed' for fair comparisons) ----------
CREATE VIEW vw_longevity_summary AS
SELECT t.longevity_class, t.release_format, t.run_status,
       COUNT(*)                                             AS songs,
       AVG(t.days_on_chart)                                 AS avg_days_on_chart,
       AVG(t.avg_rank)                                      AS avg_rank,
       AVG(CASE WHEN t.is_explicit THEN 1.0 ELSE 0.0 END)   AS explicit_share,
       AVG(CASE WHEN t.is_collab   THEN 1.0 ELSE 0.0 END)   AS collab_share
FROM dim_track t
GROUP BY t.longevity_class, t.release_format, t.run_status;
