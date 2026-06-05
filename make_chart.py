"""Generate a clean LinkedIn-ready chart of the final Finals prediction.

Reads the headline numbers from finals_prediction.json (written by the notebook's
final readout) so the chart never silently desyncs from the model. Falls back to the
last-known values if the JSON hasn't been generated yet.
"""
import json
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

_DEFAULTS = {"p_spurs_series": 47.6, "p_knicks_series": 52.4, "ensemble_accuracy": 68.4}
if os.path.exists("finals_prediction.json"):
    with open("finals_prediction.json") as _f:
        _d = json.load(_f)
else:
    print("finals_prediction.json not found -- run the notebook; using fallback values.")
    _d = _DEFAULTS

P_SPURS, P_KNICKS = _d["p_spurs_series"], _d["p_knicks_series"]
ACC = _d["ensemble_accuracy"]

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("white")

teams = ["San Antonio\nSpurs", "New York\nKnicks"]
probs = [P_SPURS, P_KNICKS]
colors = ["#5A6065", "#1D428A"]   # Spurs silver-black, Knicks blue

bars = ax.bar(teams, probs, color=colors, width=0.55, zorder=3)
for bar, p in zip(bars, probs):
    ax.text(bar.get_x() + bar.get_width() / 2, p + 1.2, f"{p:.1f}%",
            ha="center", va="bottom", fontsize=26, fontweight="bold",
            color=bar.get_facecolor())

ax.axhline(50, color="#999", ls="--", lw=1.2, zorder=2)
ax.text(-0.42, 50.6, "coin-flip line (50%)", ha="left", va="bottom",
        fontsize=9, color="#777")

ax.set_ylim(0, 65)
ax.set_ylabel("Probability of winning the series", fontsize=12)
ax.set_yticks([0, 25, 50])
ax.set_yticklabels(["0%", "25%", "50%"])
ax.set_title("2026 NBA Finals Prediction", fontsize=20, fontweight="bold", pad=18)
ax.text(0.5, 1.012, "Machine learning model  ·  XGBoost + Logistic Regression",
        transform=ax.transAxes, ha="center", fontsize=11, color="#666")

# Verdict banner + footnotes
ax.text(0.5, 0.93, "VERDICT:  TOSS-UP", transform=ax.transAxes, ha="center",
        fontsize=15, fontweight="bold", color="#222",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF3CD", ec="#E0C97F"))
ax.text(0.5, -0.13,
        f"Model accuracy on unseen games: {ACC:.1f}%   ·   "
        f"3 seasons of NBA data (2023-26)   ·   built in Python",
        transform=ax.transAxes, ha="center", fontsize=10, color="#666")

for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(left=False)
ax.grid(axis="y", color="#EEE", zorder=0)

plt.tight_layout()
plt.savefig("finals_prediction.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved finals_prediction.png")
