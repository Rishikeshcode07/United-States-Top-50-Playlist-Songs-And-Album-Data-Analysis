-- =====================================================================================================
-- sql/01_schema.sql  -  the STAR SCHEMA of the Atlantic US Top 50 warehouse
--
-- WHAT THIS FILE IS
--     The CREATE TABLE statements of six tables:
--
--         dim_date              one row per calendar day              (the calendar; "dimension")
--         dim_track             one row per song                      (song attributes and song KPIs; "dimension")
--         dim_artist            one row per artist                    (artist KPIs; "dimension")
--         bridge_track_artist   one row per (song, artist)            (a song can have several artists; "bridge")
--         fact_chart_daily      one row per date x chart position     (what happened on the chart; "fact")
--         daily_summary         one row per snapshot date             (daily KPIs: turnover, explicit share ...)
--
--     Picture of the star (arrows = foreign keys, "many" side points to the "one" side):
--
--                 dim_date <---------- fact_chart_daily ----------> dim_track <---- bridge_track_artist ----> dim_artist
--                    ^
--                    '----------- daily_summary
--
-- WHY WE NEED IT
--     Power BI and SQL are fastest and least error-prone on a star schema: small descriptive tables around one big
--     table of events. Primary keys stop duplicates, foreign keys stop orphan rows (a fact row for a song that does
--     not exist), CHECK rules stop impossible values (a position of 57). The database itself becomes a second
--     quality gate, after the Python one.
--
-- WHAT HAPPENS WHEN IT RUNS
--     1. the views are dropped (they depend on the tables), then the six tables (children first), then recreated.
--        This makes every pipeline run a clean "full refresh": running it twice gives exactly the same database.
--     2. indexes are created for the two joins that analysts use most.
--     The loader (src/etl/load.py) then inserts the rows and recreates the views (02_views.sql).
--
-- EXPECTED OUTCOME
--     Six empty tables with keys and rules. After the load: 559 + 944 + 283 + 1,105 + 27,750 + 555 rows.
--
-- MOTIVE
--     Data that breaks a rule should be refused by the database at the door, not discovered in a chart.
--
-- Portable SQL: runs unchanged on SQLite, PostgreSQL and MySQL 8. Column types were chosen from the data:
-- SMALLINT for small whole numbers, DOUBLE PRECISION for measurements, VARCHAR with generous lengths.
-- =====================================================================================================

-- ---------- 1. drop old objects (views first because they depend on the tables, then children before parents) ----------
DROP VIEW IF EXISTS vw_chart_daily;
DROP VIEW IF EXISTS vw_song_scorecard;
DROP VIEW IF EXISTS vw_artist_leaderboard;
DROP VIEW IF EXISTS vw_artist_songs;
DROP VIEW IF EXISTS vw_monthly_trend;
DROP VIEW IF EXISTS vw_weekday_pattern;
DROP VIEW IF EXISTS vw_movement_by_month;
DROP VIEW IF EXISTS vw_longevity_summary;

DROP TABLE IF EXISTS daily_summary;
DROP TABLE IF EXISTS fact_chart_daily;
DROP TABLE IF EXISTS bridge_track_artist;
DROP TABLE IF EXISTS dim_artist;
DROP TABLE IF EXISTS dim_track;
DROP TABLE IF EXISTS dim_date;


-- ---------- 2. dim_date: every calendar day between the first and the last snapshot (also the 4 days without data) ----------
CREATE TABLE dim_date (
    date                      DATE          NOT NULL,
    date_key                  INTEGER       NOT NULL,   -- yyyymmdd as a number, e.g. 20240518
    has_snapshot              BOOLEAN       NOT NULL,   -- FALSE for calendar days the playlist was not captured
    snapshot_index            INTEGER,                  -- 1, 2, 3 ... counted over snapshots; empty on days without one
    days_since_prev_snapshot  SMALLINT,                 -- 1 normally, 2 right after a missing day; empty for the first
    year                      SMALLINT      NOT NULL,
    month                     SMALLINT      NOT NULL,
    month_name                VARCHAR(12)   NOT NULL,
    year_month                VARCHAR(7)    NOT NULL,   -- text such as 2024-05 (sorts correctly)
    quarter                   SMALLINT      NOT NULL,
    iso_week                  SMALLINT      NOT NULL,
    day_of_week               SMALLINT      NOT NULL,   -- Monday = 1 ... Sunday = 7
    day_name                  VARCHAR(12)   NOT NULL,
    is_weekend                BOOLEAN       NOT NULL,
    season                    VARCHAR(10)   NOT NULL,
    CONSTRAINT pk_dim_date           PRIMARY KEY (date),
    CONSTRAINT uq_dim_date_date_key  UNIQUE (date_key),
    CONSTRAINT ck_dim_date_month     CHECK (month BETWEEN 1 AND 12),
    CONSTRAINT ck_dim_date_quarter   CHECK (quarter BETWEEN 1 AND 4),
    CONSTRAINT ck_dim_date_dow       CHECK (day_of_week BETWEEN 1 AND 7)
);


-- ---------- 3. dim_track: one row per song (the song-level KPIs live here) ----------
CREATE TABLE dim_track (
    track_id                  INTEGER           NOT NULL,
    track_label               VARCHAR(500)      NOT NULL,   -- "song title by artist credit"
    song_clean                VARCHAR(400)      NOT NULL,
    artist_credit             VARCHAR(300)      NOT NULL,
    primary_artist            VARCHAR(200)      NOT NULL,
    artist_count              SMALLINT          NOT NULL,
    is_collab                 BOOLEAN           NOT NULL,
    artists_all               VARCHAR(300)      NOT NULL,   -- artists joined with " | "
    days_on_chart             INTEGER           NOT NULL,   -- KPI: Days on Chart
    first_seen                DATE              NOT NULL,
    last_seen                 DATE              NOT NULL,
    span_days                 INTEGER           NOT NULL,
    span_snapshots            INTEGER           NOT NULL,
    presence_ratio            DOUBLE PRECISION  NOT NULL,   -- days_on_chart / span_snapshots
    n_runs                    SMALLINT          NOT NULL,
    longest_run_days          INTEGER           NOT NULL,
    avg_rank                  DOUBLE PRECISION  NOT NULL,   -- KPI: Average Rank
    median_rank               DOUBLE PRECISION  NOT NULL,
    best_rank                 SMALLINT          NOT NULL,   -- KPI: Best Rank Achieved
    worst_rank                SMALLINT          NOT NULL,
    rank_volatility_index     DOUBLE PRECISION,             -- KPI: Rank Volatility Index (empty for one-day songs)
    avg_daily_rank_move       DOUBLE PRECISION,
    days_in_top10             INTEGER           NOT NULL,
    days_in_top20             INTEGER           NOT NULL,
    days_at_number_one        INTEGER           NOT NULL,
    chart_points              INTEGER           NOT NULL,
    entry_rank                SMALLINT          NOT NULL,
    exit_rank                 SMALLINT          NOT NULL,
    peak_date                 DATE              NOT NULL,
    days_to_peak              INTEGER           NOT NULL,
    avg_popularity            DOUBLE PRECISION,             -- empty when the song has no real popularity value at all
    peak_popularity           DOUBLE PRECISION,
    popularity_std            DOUBLE PRECISION,
    on_chart_at_window_start  BOOLEAN           NOT NULL,
    on_chart_at_window_end    BOOLEAN           NOT NULL,
    run_status                VARCHAR(40)       NOT NULL,
    longevity_class           VARCHAR(30)       NOT NULL,
    reached_top10             BOOLEAN           NOT NULL,
    reached_number_one        BOOLEAN           NOT NULL,
    duration_ms               INTEGER           NOT NULL,
    duration_min              DOUBLE PRECISION  NOT NULL,
    album_type                VARCHAR(20)       NOT NULL,
    release_format            VARCHAR(10)       NOT NULL,
    total_tracks              SMALLINT          NOT NULL,
    release_size_band         VARCHAR(10)       NOT NULL,
    is_explicit               BOOLEAN           NOT NULL,
    album_cover_url           VARCHAR(300),
    CONSTRAINT pk_dim_track           PRIMARY KEY (track_id),
    CONSTRAINT ck_dim_track_days      CHECK (days_on_chart >= 1),
    CONSTRAINT ck_dim_track_best      CHECK (best_rank BETWEEN 1 AND 50),
    CONSTRAINT ck_dim_track_worst     CHECK (worst_rank BETWEEN 1 AND 50),
    CONSTRAINT ck_dim_track_best_worst CHECK (best_rank <= worst_rank),
    CONSTRAINT ck_dim_track_presence  CHECK (presence_ratio > 0 AND presence_ratio <= 1),
    CONSTRAINT ck_dim_track_dates     CHECK (first_seen <= last_seen)
);


-- ---------- 4. dim_artist: one row per individual artist ----------
CREATE TABLE dim_artist (
    artist_name                   VARCHAR(200)      NOT NULL,
    unique_tracks                 INTEGER           NOT NULL,   -- KPI: number of unique songs
    unique_tracks_as_primary      INTEGER           NOT NULL,
    chart_slot_days               INTEGER           NOT NULL,
    chart_slot_days_primary       INTEGER           NOT NULL,
    days_present                  INTEGER           NOT NULL,
    first_seen                    DATE              NOT NULL,
    last_seen                     DATE              NOT NULL,
    avg_rank                      DOUBLE PRECISION  NOT NULL,
    best_rank                     SMALLINT          NOT NULL,
    days_in_top10                 INTEGER           NOT NULL,
    days_at_number_one            INTEGER           NOT NULL,
    chart_points                  INTEGER           NOT NULL,
    chart_points_primary          INTEGER           NOT NULL,
    dominance_index_primary       DOUBLE PRECISION  NOT NULL,   -- KPI: Artist Dominance Index (% of all chart points; sums to 100)
    dominance_index_all_credits   DOUBLE PRECISION  NOT NULL,
    share_of_slots_primary_pct    DOUBLE PRECISION  NOT NULL,
    CONSTRAINT pk_dim_artist          PRIMARY KEY (artist_name),
    CONSTRAINT ck_dim_artist_share    CHECK (dominance_index_primary >= 0 AND dominance_index_primary <= 100),
    CONSTRAINT ck_dim_artist_tracks   CHECK (unique_tracks >= 1)
);


-- ---------- 5. bridge_track_artist: resolves the many-to-many relation between songs and artists ----------
CREATE TABLE bridge_track_artist (
    track_id       INTEGER       NOT NULL,
    artist_name    VARCHAR(200)  NOT NULL,
    artist_order   SMALLINT      NOT NULL,     -- 1 = first (lead) artist in the credit
    is_primary     BOOLEAN       NOT NULL,
    CONSTRAINT pk_bridge_track_artist  PRIMARY KEY (track_id, artist_order),
    CONSTRAINT uq_bridge_track_artist  UNIQUE (track_id, artist_name),
    CONSTRAINT fk_bridge_track         FOREIGN KEY (track_id)    REFERENCES dim_track (track_id),
    CONSTRAINT fk_bridge_artist        FOREIGN KEY (artist_name) REFERENCES dim_artist (artist_name),
    CONSTRAINT ck_bridge_order         CHECK (artist_order >= 1)
);


-- ---------- 6. fact_chart_daily: THE fact table. One row = one song at one position on one day ----------
CREATE TABLE fact_chart_daily (
    date                    DATE              NOT NULL,
    track_id                INTEGER           NOT NULL,
    position                SMALLINT          NOT NULL,   -- 1 = top of the chart
    chart_points            SMALLINT          NOT NULL,   -- 51 - position
    popularity_clean        DOUBLE PRECISION,             -- empty for the few rows that could not be repaired
    popularity_trend_7d     DOUBLE PRECISION,             -- KPI: Popularity Score Trend
    popularity_fill_method  VARCHAR(20)       NOT NULL,
    rank_change             SMALLINT,                     -- positive = climbed; empty on the first day of a run
    rank_gain_7d            SMALLINT,
    avg_rank_7d             DOUBLE PRECISION  NOT NULL,
    rank_volatility_7d      DOUBLE PRECISION,
    movement                VARCHAR(12)       NOT NULL,
    rank_tier               VARCHAR(10)       NOT NULL,
    in_top10                BOOLEAN           NOT NULL,
    in_top20                BOOLEAN           NOT NULL,
    is_number_one           BOOLEAN           NOT NULL,
    chart_day_number        INTEGER           NOT NULL,
    days_in_stint           INTEGER           NOT NULL,
    is_stint_start          BOOLEAN           NOT NULL,
    is_stint_end            BOOLEAN           NOT NULL,
    is_explicit             BOOLEAN           NOT NULL,
    CONSTRAINT pk_fact_chart_daily     PRIMARY KEY (date, position),          -- one song per position per day
    CONSTRAINT uq_fact_date_track      UNIQUE (date, track_id),               -- one position per song per day
    CONSTRAINT fk_fact_date            FOREIGN KEY (date)     REFERENCES dim_date (date),
    CONSTRAINT fk_fact_track           FOREIGN KEY (track_id) REFERENCES dim_track (track_id),
    CONSTRAINT ck_fact_position        CHECK (position BETWEEN 1 AND 50),
    CONSTRAINT ck_fact_points          CHECK (chart_points = 51 - position),
    CONSTRAINT ck_fact_popularity      CHECK (popularity_clean IS NULL OR (popularity_clean >= 0 AND popularity_clean <= 100)),
    CONSTRAINT ck_fact_stint_day       CHECK (days_in_stint >= 1 AND chart_day_number >= 1)
);


-- ---------- 7. daily_summary: one row per snapshot date ----------
CREATE TABLE daily_summary (
    date                    DATE              NOT NULL,
    snapshot_index          INTEGER           NOT NULL,
    unique_tracks           SMALLINT          NOT NULL,
    new_entries             SMALLINT          NOT NULL,
    re_entries              SMALLINT          NOT NULL,
    exits                   SMALLINT          NOT NULL,
    turnover_rate_pct       DOUBLE PRECISION  NOT NULL,
    is_high_turnover_day    BOOLEAN           NOT NULL,
    avg_popularity          DOUBLE PRECISION  NOT NULL,
    median_popularity       DOUBLE PRECISION  NOT NULL,
    explicit_share          DOUBLE PRECISION  NOT NULL,   -- KPI: Explicit Content Share (0 to 1)
    collab_share            DOUBLE PRECISION  NOT NULL,
    single_share            DOUBLE PRECISION  NOT NULL,
    avg_duration_min        DOUBLE PRECISION  NOT NULL,
    unique_primary_artists  SMALLINT          NOT NULL,
    CONSTRAINT pk_daily_summary        PRIMARY KEY (date),
    CONSTRAINT fk_summary_date         FOREIGN KEY (date) REFERENCES dim_date (date),
    CONSTRAINT ck_summary_shares       CHECK (explicit_share BETWEEN 0 AND 1 AND collab_share BETWEEN 0 AND 1
                                              AND single_share BETWEEN 0 AND 1),
    CONSTRAINT ck_summary_entries      CHECK (new_entries >= 0 AND re_entries >= 0 AND exits >= 0)
);


-- ---------- 8. indexes for the joins analysts use most (primary keys are indexed automatically) ----------
CREATE INDEX ix_fact_track  ON fact_chart_daily (track_id);          -- "all days of one song"
CREATE INDEX ix_bridge_artist ON bridge_track_artist (artist_name);  -- "all songs of one artist"
