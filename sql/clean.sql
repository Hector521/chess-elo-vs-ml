-- clean.sql: turn games_raw (text, as in the file) into games (typed, cleaned, with features).
-- Every rule here was chosen after exploring the raw data (see explore.py).

CREATE OR REPLACE TABLE games AS
WITH parsed AS (
    -- Step 1: convert text to proper types. TRY_ functions give NULL instead of an error.
    SELECT
        regexp_extract(site, '[^/]+$')                              AS game_id,
        try_strptime(utc_date || ' ' || utc_time, '%Y.%m.%d %H:%M:%S') AS played_at,
        TRY_CAST(white_elo AS INTEGER)                              AS white_elo,
        TRY_CAST(black_elo AS INTEGER)                              AS black_elo,
        result,
        TRY_CAST(split_part(time_control, '+', 1) AS INTEGER)       AS base_seconds,
        TRY_CAST(split_part(time_control, '+', 2) AS INTEGER)       AS increment_seconds,
        eco,
        opening,
        termination
    FROM games_raw
)
SELECT
    game_id,
    played_at,

    -- Ratings and the features built from them
    white_elo,
    black_elo,
    white_elo - black_elo                      AS rating_diff,
    (white_elo + black_elo) / 2.0              AS avg_rating,

    -- Time control, using Lichess's rule: estimated seconds = base + 40 x increment
    base_seconds,
    increment_seconds,
    CASE
        WHEN base_seconds + 40 * increment_seconds < 30   THEN 'ultrabullet'
        WHEN base_seconds + 40 * increment_seconds < 180  THEN 'bullet'
        WHEN base_seconds + 40 * increment_seconds < 480  THEN 'blitz'
        WHEN base_seconds + 40 * increment_seconds < 1500 THEN 'rapid'
        ELSE 'classical'
    END                                        AS time_class,

    -- Opening
    eco,
    LEFT(eco, 1)                               AS eco_group,       -- A-E: broad opening family
    split_part(opening, ':', 1)                AS opening_family,  -- e.g. "French Defense"

    -- Target: White's score
    CASE result WHEN '1-0' THEN 1.0 WHEN '0-1' THEN 0.0 ELSE 0.5 END AS white_score,
    CASE result WHEN '1-0' THEN 'white' WHEN '0-1' THEN 'black' ELSE 'draw' END AS outcome,

    -- For analysis only. NOT a model feature: it's known only after the game (data leakage).
    termination
FROM parsed
WHERE white_elo IS NOT NULL
  AND black_elo IS NOT NULL
  AND result IN ('1-0', '0-1', '1/2-1/2')
  AND base_seconds IS NOT NULL
  AND increment_seconds IS NOT NULL
  AND played_at IS NOT NULL
ORDER BY played_at;
