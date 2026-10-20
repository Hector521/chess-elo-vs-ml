"""train_boosting.py - Does a non-linear model find anything Elo and logistic regression miss?

Gradient boosting builds hundreds of small decision trees, each one correcting the
mistakes of the ones before it. Unlike logistic regression, it can learn interactions
on its own, such as "a rating gap matters less in bullet than in classical."

Also answers: how much does the rating gap matter in each time control?
Saves test-set predictions to data/test_predictions.csv for the charts.
"""
import math

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from common import brier, elo_expected, load_games, log_loss, time_split, win_loss_with_weights

ELO_SLOPE = math.log(10) / 400   # Elo's fixed slope, about 0.00576
NUMERIC = ["rating_diff", "avg_rating", "base_seconds", "increment_seconds"]
CATEGORICAL = ["time_class", "eco_group"]
FEATURES = NUMERIC + CATEGORICAL
PREDICTIONS_FILE = "data/test_predictions.csv"


def fit_gap_only(df: pd.DataFrame) -> LogisticRegression:
    """Model A from Session 3: Elo's formula shape, slope and White edge learned from data."""
    rows, labels, weights = win_loss_with_weights(df)
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(rows[["rating_diff"]], labels, sample_weight=weights)
    return model


games = load_games()
for column in CATEGORICAL:
    games[column] = games[column].astype("category")  # same categories in train and test
train, test = time_split(games)
rows, labels, weights = win_loss_with_weights(train)
y_test = test["white_score"]

# ---- 1. How much does the rating gap matter in each time control? ----
print("How much a rating gap matters in each time control (textbook Elo = 100%):")
for time_class in ["bullet", "blitz", "rapid", "classical"]:
    part = train[train["time_class"] == time_class]
    model = fit_gap_only(part)
    slope, intercept = model.coef_[0][0], model.intercept_[0]
    print(f"  {time_class:<10}{slope / ELO_SLOPE:>6.0%} of Elo   "
          f"White edge {intercept / slope:>4.0f} pts   ({len(part):,} training games)")

# ---- 2. Train the models ----
model_a = fit_gap_only(train)

boost = HistGradientBoostingClassifier(
    learning_rate=0.05,          # small steps: each tree corrects only a little
    max_iter=500,                # at most 500 trees...
    early_stopping=True,         # ...but stop when a held-back slice stops improving
    validation_fraction=0.1,
    n_iter_no_change=20,
    max_leaf_nodes=15,           # small trees, to avoid memorizing noise
    min_samples_leaf=200,        # each final branch must cover at least 200 games
    l2_regularization=1.0,
    categorical_features="from_dtype",
    random_state=42,             # same result every run
)
boost.fit(rows[FEATURES], labels, sample_weight=weights)
print(f"\nGradient boosting stopped after {boost.n_iter_} trees")

# ---- 3. Scoreboard on unseen test games ----
test["p_elo"] = elo_expected(test["rating_diff"])
test["p_model_a"] = model_a.predict_proba(test[["rating_diff"]])[:, 1]
test["p_boost"] = boost.predict_proba(test[FEATURES])[:, 1]
models = {"Elo formula": "p_elo", "Model A (gap only)": "p_model_a", "Gradient boosting": "p_boost"}

elo_ll = log_loss(y_test, test["p_elo"])
print(f"\nScoreboard on {len(test):,} unseen test games (lower is better):")
print(f"  {'Model':<22}{'Log loss':>10}{'Brier':>9}{'vs Elo':>9}")
for name, column in models.items():
    ll = log_loss(y_test, test[column])
    print(f"  {name:<22}{ll:>10.4f}{brier(y_test, test[column]):>9.4f}{(elo_ll - ll) / elo_ll:>+9.1%}")

# ---- 4. Log loss within each time control ----
print("\nLog loss by time control (lower is better):")
by_class = []
for time_class, group in test.groupby("time_class", observed=True):
    row = {"time_class": time_class, "games": len(group)}
    for name, column in models.items():
        row[name] = log_loss(group["white_score"], group[column])
    by_class.append(row)
print(pd.DataFrame(by_class).set_index("time_class").round(4).to_string())

# ---- 5. Which inputs does the boosting model actually rely on? ----
test_rows, test_labels, test_weights = win_loss_with_weights(test)
importance = permutation_importance(
    boost, test_rows[FEATURES], test_labels, sample_weight=test_weights,
    scoring="neg_log_loss", n_repeats=5, random_state=42,
)
ranked = pd.Series(importance.importances_mean, index=FEATURES).sort_values(ascending=False)
print("\nFeature importance (how much worse log loss gets when that input is scrambled):")
for feature, value in ranked.items():
    print(f"  {feature:<20}{value:>9.4f}")

test[["played_at", "rating_diff", "time_class", "white_score", "p_elo", "p_model_a", "p_boost"]].to_csv(
    PREDICTIONS_FILE, index=False)
print(f"\nSaved test predictions to {PREDICTIONS_FILE} (used by charts.py)")
