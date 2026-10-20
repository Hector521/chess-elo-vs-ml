"""clean_data.py - Run sql/clean.sql to build the cleaned `games` table, then check the result."""
from pathlib import Path

import duckdb

from build_db import DB_PATH

CLEAN_SQL = Path("sql/clean.sql")

con = duckdb.connect(DB_PATH)
con.execute(CLEAN_SQL.read_text())  # run every statement in the SQL file

raw = con.sql("SELECT COUNT(*) FROM games_raw").fetchone()[0]
clean = con.sql("SELECT COUNT(*) FROM games").fetchone()[0]
print(f"Kept {clean:,} of {raw:,} games ({raw - clean:,} removed)\n")

print("Games by time class, with White's average score:")
print(con.sql("""
    SELECT time_class,
           COUNT(*) AS games,
           ROUND(AVG(white_score), 3) AS white_avg_score,
           ROUND(100.0 * AVG(CASE WHEN outcome = 'draw' THEN 1 ELSE 0 END), 1) AS draw_percent
    FROM games
    GROUP BY time_class
    ORDER BY games DESC
"""))

print("Sample of the cleaned table:")
print(con.sql("""
    SELECT game_id, played_at, white_elo, black_elo, rating_diff, time_class,
           eco_group, opening_family, white_score
    FROM games
    LIMIT 5
"""))
con.close()
