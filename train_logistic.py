"""train_logistic.py - Can logistic regression beat the Elo formula?

Model A: rating gap only. Elo's own formula shape, but the slope and White's advantage
         are learned from real games instead of fixed by the textbook.
Model B: rating gap + average rating + time control.
"""
import math

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (brier, calibration_table, elo_expected, load_games, log_loss,
                    time_split, win_loss_with_weights)

train, test = time_split(load_games())
rows, labels, weights = win_loss_with_weights(train)  # draws = half win + half loss
y_test = test["white_score"]

# ---- Model A: Elo's formula shape, refit to real games ----
# C=1e6 means essentially no regularization, so the learned numbers stay easy to read.
model_a = LogisticRegression(C=1e6, max_iter=1000)
model_a.fit(rows[["rating_diff"]], labels, sample_weight=weights)
slope = model_a.coef_[0][0]
intercept = model_a.intercept_[0]
elo_slope = math.log(10) / 400  # Elo's fixed slope, about 0.00576

print("What Model A learned (Elo's formula, refit to real games):")
print(f"  Slope per rating point: {slope:.5f}   (textbook Elo: {elo_slope:.5f})")
print(f"  -> a rating gap matters {slope / elo_slope:.0%} as much as Elo assumes")
print(f"  White's first-move edge is worth about {intercept / slope:.0f} rating points\n")

# ---- Model B: add average rating and time control ----
features_b = ["rating_diff", "avg_rating", "time_class"]
preprocess = ColumnTransformer([
    ("numbers", StandardScaler(), ["rating_diff", "avg_rating"]),
    ("categories", OneHotEncoder(handle_unknown="ignore"), ["time_class"]),
])
model_b = make_pipeline(preprocess, LogisticRegression(max_iter=1000))
model_b.fit(rows[features_b], labels, logisticregression__sample_weight=weights)

# ---- Scoreboard on the unseen test games ----
predictions = {
    "Naive": np.full(len(test), train["white_score"].mean()),
    "Elo formula": elo_expected(test["rating_diff"]),
    "Model A (gap only)": model_a.predict_proba(test[["rating_diff"]])[:, 1],
    "Model B (+ level, time)": model_b.predict_proba(test[features_b])[:, 1],
}

elo_ll = log_loss(y_test, predictions["Elo formula"])
print(f"Scoreboard on {len(test):,} unseen test games (lower is better):")
print(f"  {'Model':<26}{'Log loss':>10}{'Brier':>9}{'vs Elo':>9}")
for name, p in predictions.items():
    ll = log_loss(y_test, p)
    print(f"  {name:<26}{ll:>10.4f}{brier(y_test, p):>9.4f}{(elo_ll - ll) / elo_ll:>+9.1%}")

print("\nCalibration: when a model predicts a score, what actually happens?")
print("Elo formula:")
print(calibration_table(y_test, predictions["Elo formula"]))
print("\nModel B:")
print(calibration_table(y_test, predictions["Model B (+ level, time)"]))
