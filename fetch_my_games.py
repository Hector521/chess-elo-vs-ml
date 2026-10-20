"""fetch_my_games.py - Download my own rated games from the Chess.com public API.

The API is free and needs no key: https://www.chess.com/news/view/published-data-api
It returns one "archive" per month of games. This script walks every archive,
keeps rated standard-chess games, and saves one row per game to data/my_games.csv
(the data/ folder is git-ignored, so my game history is not committed).
"""
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

USERNAME = "chessking45235"
API = "https://api.chess.com/pub/player"
OUT_FILE = Path("data/my_games.csv")
# Chess.com asks API users to identify themselves with a descriptive User-Agent
HEADERS = {"User-Agent": f"chess-elo-vs-ml portfolio project (github.com/Hector521/chess-elo-vs-ml; user {USERNAME})"}

# Chess.com result codes that mean the game was drawn
DRAW_CODES = {"agreed", "repetition", "stalemate", "insufficient", "50move", "timevsinsufficient"}


def get_json(url: str) -> dict:
    """Fetch one API URL and parse the JSON response."""
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def opening_name(eco_url: str | None) -> str:
    """Chess.com gives the opening as a URL, e.g. .../openings/Italian-Game-Two-Knights...;
    keep the first two words (e.g. "Italian Game", "Sicilian Defense") as the opening family."""
    if not eco_url or "/openings/" not in eco_url:
        return "Unknown"
    words = eco_url.rstrip("/").split("/openings/")[-1].split("-")
    return " ".join(words[:2])


def to_row(game: dict) -> dict | None:
    """Turn one API game into one row from MY point of view; None if it doesn't qualify."""
    if not game.get("rated") or game.get("rules") != "chess":
        return None  # skip casual games and variants (Chess960, bughouse, ...)
    white, black = game["white"], game["black"]
    my_color = "white" if white["username"].lower() == USERNAME.lower() else "black"
    me, opponent = (white, black) if my_color == "white" else (black, white)

    if me["result"] == "win":
        score = 1.0
    elif me["result"] in DRAW_CODES:
        score = 0.5
    else:
        score = 0.0

    return {
        "played_at": datetime.fromtimestamp(game["end_time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "time_class": game.get("time_class"),
        "my_color": my_color,
        "my_rating": me["rating"],
        "opponent_rating": opponent["rating"],
        "rating_diff": me["rating"] - opponent["rating"],   # positive = I'm the higher-rated player
        "my_score": score,
        "opening": opening_name(game.get("eco")),
        "url": game.get("url"),
    }


if __name__ == "__main__":
    archives = get_json(f"{API}/{USERNAME}/games/archives")["archives"]
    print(f"Found {len(archives)} months of games for {USERNAME}")

    rows = []
    for i, archive_url in enumerate(archives, start=1):
        games = get_json(archive_url).get("games", [])
        rows.extend(row for row in map(to_row, games) if row is not None)
        print(f"  month {i}/{len(archives)}: {archive_url[-7:]}  ({len(rows):,} rated games so far)")
        time.sleep(0.3)  # be polite: one request at a time, with a short pause

    df = pd.DataFrame(rows)
    OUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"\nSaved {len(df):,} rated games to {OUT_FILE}")
    if len(df):
        print("\nGames by time control:")
        print(df["time_class"].value_counts().to_string())
        print(f"\nFirst game: {df['played_at'].min()}   Latest game: {df['played_at'].max()}")
