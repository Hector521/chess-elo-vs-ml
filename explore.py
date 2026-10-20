"""explore.py - Ask the raw data questions with SQL before deciding how to clean it."""
import duckdb

from build_db import DB_PATH

con = duckdb.connect(DB_PATH, read_only=True)  # read-only: exploring can't change the data

print("1) Are any ratings missing or not numbers? What range do they cover?")
print(con.sql("""
    SELECT
        COUNT(*) FILTER (WHERE TRY_CAST(white_elo AS INTEGER) IS NULL
                            OR TRY_CAST(black_elo AS INTEGER) IS NULL) AS bad_ratings,
        MIN(TRY_CAST(white_elo AS INTEGER)) AS lowest,
        MAX(TRY_CAST(white_elo AS INTEGER)) AS highest,
        ROUND(AVG(TRY_CAST(white_elo AS INTEGER))) AS average
    FROM games_raw
"""))

print("2) How did games end?")
print(con.sql("""
    SELECT termination, COUNT(*) AS games
    FROM games_raw
    GROUP BY termination
    ORDER BY games DESC
"""))

print("3) Most common time controls (seconds + increment per move)")
print(con.sql("""
    SELECT time_control, COUNT(*) AS games
    FROM games_raw
    GROUP BY time_control
    ORDER BY games DESC
    LIMIT 10
"""))

print("4) The big question: does the rating gap predict the result the way Elo says?")
print(con.sql("""
    WITH g AS (
        SELECT
            TRY_CAST(white_elo AS INTEGER) - TRY_CAST(black_elo AS INTEGER) AS diff,
            CASE result WHEN '1-0' THEN 1.0
                        WHEN '0-1' THEN 0.0
                        WHEN '1/2-1/2' THEN 0.5 END AS white_score
        FROM games_raw
    )
    SELECT
        CAST(FLOOR(diff / 100.0) * 100 AS INTEGER) AS gap_from,
        COUNT(*) AS games,
        ROUND(AVG(white_score), 3) AS white_actual,
        ROUND(AVG(1 / (1 + POWER(10, -diff / 400.0))), 3) AS elo_expected
    FROM g
    WHERE diff BETWEEN -400 AND 399 AND white_score IS NOT NULL
    GROUP BY gap_from
    ORDER BY gap_from
"""))

con.close()