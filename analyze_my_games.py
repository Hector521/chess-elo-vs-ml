"""analyze_my_games.py - Apply the project's findings to my own Chess.com games.

Questions:
  1. Do I score more or less than my rating predicts?
  2. Does the correction learned from Lichess games (Elo is overconfident) also fit my
     Chess.com games, years later on a different site?
  3. How do I do as White vs. Black, by time control, against stronger/weaker players,
     and with my most-played openings?
  4. Fit the model on MY games: how much does the rating gap matter for me?

Run fetch_my_games.py first (it writes data/my_games.csv).
"""
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import elo_expected, load_games, log_loss, time_split, win_loss_with_weights

MY_GAMES = "data/my_games.csv"
ELO_SLOPE = math.log(10) / 400
MIN_GAMES = 50   # don't report a group smaller than this; too likely to be luck

# ---- The Lichess-trained model (Model A from Session 3), from White's point of view ----
train, _ = time_split(load_games())
rows, labels, weights = win_loss_with_weights(train)
lichess_model = LogisticRegression(C=1e6, max_iter=1000)
lichess_model.fit(rows[["rating_diff"]], labels, sample_weight=weights)

# ---- My games, from my point of view ----
me = pd.read_csv(MY_GAMES).sort_values("played_at").reset_index(drop=True)

# DATA LEAKAGE FIX: Chess.com's API records each player's rating AFTER the game, so the raw
# rating gap already contains the result. (Checked on my data: my rating moves about +8 after
# a win and -8 after a loss in that same game.) Rebuild pre-game ratings instead:
#   - my rating before a game = my rating after my previous game in the same time control
#   - my opponent gained or lost about what I lost or gained (an approximation)
raw_games = len(me)
me["my_pre"] = me.groupby("time_class")["my_rating"].shift()
rating_change = me["my_rating"] - me["my_pre"]
me["opponent_pre"] = me["opponent_rating"] + rating_change
me = me.dropna(subset=["my_pre"]).copy()  # the first game in each time control has no "before"
me["rating_diff"] = me["my_pre"] - me["opponent_pre"]
print(f"Rebuilt pre-game ratings ({raw_games - len(me)} first-of-time-control games dropped)")

# Convert to White's point of view, predict, then flip back for the games where I was Black
white_diff = np.where(me["my_color"] == "white", me["rating_diff"], -me["rating_diff"])
p_white_model = lichess_model.predict_proba(pd.DataFrame({"rating_diff": white_diff}))[:, 1]
me["expected_elo"] = elo_expected(me["rating_diff"])          # Elo: no color effect
me["expected_model"] = np.where(me["my_color"] == "white", p_white_model, 1 - p_white_model)


def summary(group: pd.DataFrame) -> pd.Series:
    """Actual vs. expected score for a group of my games."""
    return pd.Series({
        "games": len(group),
        "my_score": group["my_score"].mean(),
        "elo_expected": group["expected_elo"].mean(),
        "model_expected": group["expected_model"].mean(),
        "vs_elo": group["my_score"].mean() - group["expected_elo"].mean(),
    })


def show(title: str, table: pd.DataFrame) -> None:
    table = table[table["games"] >= MIN_GAMES].copy()
    table["games"] = table["games"].astype(int)
    print(f"\n{title}")
    print(table.round(3).to_string())


print(f"My rated Chess.com games: {len(me):,}  ({me['played_at'].min()[:10]} to {me['played_at'].max()[:10]})")

# 1-2. Overall, and which formula fits my games better
overall = summary(me)
print(f"\nOverall: I scored {overall['my_score']:.3f}; Elo expected {overall['elo_expected']:.3f}; "
      f"the Lichess-trained model expected {overall['model_expected']:.3f}")
print(f"  -> I score {overall['vs_elo']:+.3f} per game versus what Elo predicts")
elo_ll = log_loss(me["my_score"], me["expected_elo"])
model_ll = log_loss(me["my_score"], me["expected_model"])
print(f"Log loss on my games: Elo {elo_ll:.4f}  vs  Lichess-trained model {model_ll:.4f} "
      f"({(elo_ll - model_ll) / elo_ll:+.1%})")

# 3. Breakdowns
show("By color:", me.groupby("my_color").apply(summary, include_groups=False))
show("By time control:", me.groupby("time_class").apply(summary, include_groups=False))

strength = pd.cut(
    me["rating_diff"],
    [-10_000, -200, -50, 50, 200, 10_000],
    labels=["opponent 200+ higher", "opponent 50-200 higher", "within 50", "opponent 50-200 lower", "opponent 200+ lower"],
)
show("By opponent strength (vs_elo > 0 means I beat my expected score):",
     me.groupby(strength, observed=True).apply(summary, include_groups=False))

top_openings = me["opening"].value_counts().head(8).index
show("My most-played openings:",
     me[me["opening"].isin(top_openings)].groupby("opening").apply(summary, include_groups=False)
       .sort_values("games", ascending=False))

# 4. The model refit on MY games: rating gap plus a "playing White" input
mine = pd.DataFrame({"rating_diff": me["rating_diff"], "is_white": (me["my_color"] == "white").astype(int),
                     "white_score": me["my_score"]})
my_rows, my_labels, my_weights = win_loss_with_weights(mine)
my_model = LogisticRegression(C=1e6, max_iter=1000)
my_model.fit(my_rows[["rating_diff", "is_white"]], my_labels, sample_weight=my_weights)
gap_coef, white_coef = my_model.coef_[0]
print("\nModel fit on my own games:")
print(f"  A rating gap matters {gap_coef / ELO_SLOPE:.0%} as much as Elo assumes "
      f"(Lichess 2013 average: 78%)")
# is_white compares playing White with playing Black; half of that is the edge over an
# even game, which is what the Lichess "White edge" measured.
print(f"  Playing White is worth about {white_coef / 2 / gap_coef:.0f} rating points to me "
      f"(Lichess 2013 average: about 19)")
