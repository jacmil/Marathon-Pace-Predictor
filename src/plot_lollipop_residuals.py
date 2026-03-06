"""
Lollipop residual chart for the best model (XGBoost).

Each runner in the test set is a horizontal stem from zero.
Sorted by actual finish time, colored by over/under prediction.
Works well with ~80-100 data points.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ── Load data and model ────────────────────────────────────────────────────
test = pd.read_parquet('../data/processed/test.parquet')
gb_model = joblib.load('../models/gradient_boosting_tuned.pkl')

TARGET = 'mf_ti_adj'
FEATURE_COLS = [
    "age", "bmi", "female", "injury", "footwear",
    "mh_ti_adj_final", "mh_ti_adj_imputed_flag",
    "tempo", "sprint", "typical",
    "mean_vdot", "vdot_consistency",
]

X_test = test[FEATURE_COLS]
actual = test[TARGET].values
predicted = gb_model.predict(X_test)
residuals = predicted - actual  # positive = overpredicted, negative = underpredicted

# ── Sort by actual finish time ─────────────────────────────────────────────
sort_idx = np.argsort(actual)
actual_sorted = actual[sort_idx]
residuals_sorted = residuals[sort_idx]

# ── Colors: blue for underprediction, red for overprediction ───────────────
colors = ['#1f77b4' if r < 0 else '#d62728' for r in residuals_sorted]

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

y_pos = range(len(residuals_sorted))

# Stems
ax.hlines(
    y=y_pos,
    xmin=0,
    xmax=residuals_sorted,
    colors=colors,
    linewidth=1.5,
    alpha=0.7,
)

# Dots at the end of each stem
ax.scatter(
    residuals_sorted, y_pos,
    c=colors, s=20, zorder=3, alpha=0.8,
)

# Zero line
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

# Y-axis: show actual finish time instead of index
# Pick ~8 evenly spaced ticks
n_ticks = 8
tick_positions = np.linspace(0, len(y_pos) - 1, n_ticks, dtype=int)
tick_labels = [f"{actual_sorted[i]:.0f}" for i in tick_positions]
ax.set_yticks(tick_positions)
ax.set_yticklabels(tick_labels)

ax.set_xlabel('Prediction Error (min)', fontsize=12)
ax.set_ylabel('Actual Finish Time (min)', fontsize=12)
ax.set_title('XGBoost Prediction Residuals by Runner', fontsize=14)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='#d62728', label='Overpredicted',
           markersize=6, linestyle='-', linewidth=1.5),
    Line2D([0], [0], marker='o', color='#1f77b4', label='Underpredicted',
           markersize=6, linestyle='-', linewidth=1.5),
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(
    '../results/figures/lollipop_residuals.png',
    dpi=150, bbox_inches='tight',
)
plt.show()
