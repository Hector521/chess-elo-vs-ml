"""build_db.py - Read the game headers from the Lichess file into a DuckDB table.

Creates data/chess.duckdb with one table, games_raw: one row per game, every
header stored exactly as it appears in the file (as text). Cleaning and type
conversion happen later, in SQL, so the raw copy never has to be re-read.
"""
import io
import time

import chess.pgn
import duckdb
import pandas as pd
import zstandard

from download_data import OUT_FILE

DB_PATH = "data/chess.duckdb"

# PGN header name -> column name in our table
HEADER_FIELDS = {
    "Site": "site",
    "Event": "event",
    "White": "white",
    "Black": "black",
    "Result": "result",
    "UTCDate": "utc_date",
    "UTCTime": "utc_time",
    "WhiteElo": "white_elo",
    "BlackElo": "black_elo",
    "WhiteRatingDiff": "white_rating_diff",
    "BlackRatingDiff": "black_rating_diff",
    "WhiteTitle": "white_title",
    "BlackTitle": "black_title",
    "ECO": "eco",
    "Opening": "opening",
    "TimeControl": "time_control",
    "Termination": "termination",
}


def read_headers(path) -> list[dict]:
    """Stream through the compressed PGN and collect each game's headers (moves are skipped)."""
    rows = []
    with open(path, "rb") as fh:
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        while True:
            headers = chess.pgn.read_headers(text)  # reads one game's headers, skips its moves
            if headers is None:  # end of file
                break
            rows.append({column: headers.get(tag) for tag, column in HEADER_FIELDS.items()})
            if len(rows) % 20_000 == 0:
                print(f"  ...{len(rows):,} games read")
    return rows


if __name__ == "__main__":
    start = time.time()
    print(f"Reading headers from {OUT_FILE} ...")
    games = pd.DataFrame(read_headers(OUT_FILE))
    print(f"Read {len(games):,} games in {time.time() - start:.0f} seconds")

    con = duckdb.connect(DB_PATH)
    con.register("games_df", games)  # let SQL see the pandas table by name
    con.execute("CREATE OR REPLACE TABLE games_raw AS SELECT * FROM games_df")

    print(f"\nSaved to {DB_PATH}. Quick look with SQL:")
    print(con.sql("SELECT COUNT(*) AS games FROM games_raw"))
    print(con.sql("""
        SELECT result, COUNT(*) AS games,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS percent
        FROM games_raw
        GROUP BY result
        ORDER BY games DESC
    """))
    con.close()