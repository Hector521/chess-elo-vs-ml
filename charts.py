"""charts.py - Draw the two README charts from the saved test-set predictions.

Run train_boosting.py first; it writes data/test_predictions.csv.
Charts are saved to docs/ so they can be committed and shown in the README.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw straight to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PREDICTIONS_FILE = "data/test_predictions.csv"
DOCS = Path("docs")
ELO_COLOR, MODEL_COLOR, ACTUAL_COLOR = "#c2410c", "#1d4ed8", "#111827"

test = pd.read_csv(PREDICTIONS_FILE)
DOCS.mkdir(exist_ok=True)

# ---- Chart 1: rating gap vs. White's score ----
bins = np.arange(-500, 501, 50)
test["gap_bin"] = pd.cut(test["rating_diff"], bins)
by_gap = test.groupby("gap_bin", observed=True).agg(
    games=("white_score", "size"),
    actual=("white_score", "mean"),
    elo=("p_elo", "mean"),
    model=("p_model_a", "mean"),
)
by_gap = by_gap[by_gap["games"] >= 100]  # skip buckets too small to trust
centers = [interval.mid for interval in by_gap.index]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(centers, by_gap["elo"], color=ELO_COLOR, linewidth=2, label="Elo formula")
ax.plot(centers, by_gap["model"], color=MODEL_COLOR, linewidth=2, label="Model (learned from data)")
ax.scatter(centers, by_gap["actual"], color=ACTUAL_COLOR, s=28, zorder=3, label="Actual result")
ax.axhline(0.5, color="#9ca3af", linewidth=0.8)
ax.axvline(0, color="#9ca3af", linewidth=0.8)
ax.set_xlabel("Rating gap (White minus Black)")
ax.set_ylabel("White's average score")
ax.set_title("Elo is overconfident: real results are flatter than the formula")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(DOCS / "rating_gap_curve.png", dpi=150)
plt.close(fig)

# ---- Chart 2: calibration ----
edges = np.linspace(0, 1, 11)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1, label="Perfect calibration")
for column, color, label in [("p_elo", ELO_COLOR, "Elo formula"), ("p_model_a", MODEL_COLOR, "Model")]:
    bucket = pd.cut(test[column], edges, include_lowest=True)
    table = test.groupby(bucket, observed=True).agg(predicted=(column, "mean"), actual=("white_score", "mean"))
    ax.plot(table["predicted"], table["actual"], marker="o", color=color, linewidth=2, label=label)
ax.set_xlabel("Predicted score for White")
ax.set_ylabel("Actual score for White")
ax.set_title("Calibration on unseen games")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.grid(alpha=0.25)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(DOCS / "calibration.png", dpi=150)
plt.close(fig)

print("Saved docs/rating_gap_curve.png and docs/calibration.png")
