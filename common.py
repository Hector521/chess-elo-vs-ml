"""common.py - Shared tools for every model: load data, split by time, baselines, scoring."""
import duckdb
import numpy as np
import pandas as pd

from build_db import DB_PATH

TEST_SHARE = 0.2   # the last 20% of games (by time) are held out for testing
ELO_SCALE = 400    # the 400 in Elo's formula


def load_games() -> pd.DataFrame:
    """Load the cleaned games table from DuckDB, oldest first."""
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.sql("""
        SELECT played_at, white_elo, black_elo, rating_diff, avg_rating,
               base_seconds, increment_seconds, time_class, eco_group, opening_family,
               CAST(white_score AS DOUBLE) AS white_score
        FROM games
        ORDER BY played_at
    """).df()
    con.close()
    return df


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earlier games, test on the later ones (like predicting the future)."""
    cut = int(len(df) * (1 - TEST_SHARE))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def elo_expected(rating_diff) -> np.ndarray:
    """Elo's predicted score for White: 1 / (1 + 10^(-gap/400))."""
    return 1 / (1 + 10 ** (-np.asarray(rating_diff, dtype=float) / ELO_SCALE))


def win_loss_with_weights(df: pd.DataFrame):
    """Turn each draw into half a win plus half a loss, so a yes/no classifier learns expected score.

    Returns (rows, labels, weights): wins and losses keep weight 1; each draw appears
    twice (once as a win, once as a loss), each with weight 0.5.
    """
    decisive = df[df["white_score"] != 0.5]
    draws = df[df["white_score"] == 0.5]
    rows = pd.concat([decisive, draws, draws], ignore_index=True)
    labels = np.concatenate([
        decisive["white_score"].to_numpy(),
        np.ones(len(draws)),
        np.zeros(len(draws)),
    ]).astype(int)
    weights = np.concatenate([np.ones(len(decisive)), np.full(2 * len(draws), 0.5)])
    return rows, labels, weights


def log_loss(y, p) -> float:
    """Average penalty; confident wrong predictions are punished hardest. Lower is better."""
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def brier(y, p) -> float:
    """Average squared gap between prediction and result. Lower is better."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Group games by predicted score and compare the average prediction to what really happened."""
    frame = pd.DataFrame({"predicted": p, "actual": y})
    frame["bucket"] = pd.cut(frame["predicted"], np.linspace(0, 1, bins + 1), include_lowest=True)
    return (frame.groupby("bucket", observed=True)
                 .agg(games=("actual", "size"), predicted=("predicted", "mean"), actual=("actual", "mean"))
                 .round(3))


if __name__ == "__main__":
    games = load_games()
    train, test = time_split(games)
    print(f"Train: {len(train):,} games  ({train['played_at'].min()} to {train['played_at'].max()})")
    print(f"Test:  {len(test):,} games  ({test['played_at'].min()} to {test['played_at'].max()})\n")

    y = test["white_score"]
    naive = np.full(len(test), train["white_score"].mean())
    elo = elo_expected(test["rating_diff"])
    print(f"{'Baseline':<10}{'Log loss':>10}{'Brier':>10}")
    print(f"{'Naive':<10}{log_loss(y, naive):>10.4f}{brier(y, naive):>10.4f}")
    print(f"{'Elo':<10}{log_loss(y, elo):>10.4f}{brier(y, elo):>10.4f}")
