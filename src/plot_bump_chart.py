"""
Bump chart: model MAE rank across 20-minute finish-time buckets.

Shows how each model's relative performance shifts across runner speed tiers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from src.baselines import riegel_predict, vdot_predict

# ── Load data and models ────────────────────────────────────────────────────
test = pd.read_parquet('../data/processed/test.parquet')

lr_model = joblib.load('../models/linear_regression.pkl')
rf_model = joblib.load('../models/random_forest_tuned.pkl')
gb_model = joblib.load('../models/gradient_boosting_tuned.pkl')

TARGET = 'mf_ti_adj'
FEATURE_COLS = [
    "age", "bmi", "female", "injury", "footwear",
    "mh_ti_adj_final", "mh_ti_adj_imputed_flag",
    "tempo", "sprint", "typical",
    "mean_vdot", "vdot_consistency",
]

X_test = test[FEATURE_COLS]
y_test = test[TARGET]
actual = y_test.values

# ── Generate predictions ────────────────────────────────────────────────────
preds = {
    'VDOT':          test["mean_vdot"].apply(lambda v: vdot_predict(vdot=v, race_distance=26.2)).values,
    'Riegel':        test["mh_ti_adj_final"].apply(lambda t: riegel_predict(race_time=t, race_distance=13.1, target_distance=26.2)).values,
    'Linear Reg.':   lr_model.predict(X_test),
    'Random Forest':  rf_model.predict(X_test),
    'XGBoost':       gb_model.predict(X_test),
}

# ── Bucket by 20-minute intervals ──────────────────────────────────────────
bin_width = 20
bin_min = int(np.floor(actual.min() / bin_width) * bin_width)
bin_max = int(np.ceil(actual.max() / bin_width) * bin_width) + bin_width
bins = np.arange(bin_min, bin_max, bin_width)

bucket_labels = [f"{b}-{b+bin_width}" for b in bins[:-1]]
bucket_idx = np.digitize(actual, bins) - 1

# Compute MAE per model per bucket
mae_by_bucket = {}
for name, pred in preds.items():
    abs_err = np.abs(actual - pred)
    bucket_maes = []
    for i in range(len(bins) - 1):
        mask = bucket_idx == i
        if mask.sum() >= 2:  # need at least 2 runners in a bucket
            bucket_maes.append(abs_err[mask].mean())
        else:
            bucket_maes.append(np.nan)
    mae_by_bucket[name] = bucket_maes

mae_df = pd.DataFrame(mae_by_bucket, index=bucket_labels)

# Drop buckets with any NaN (too few runners)
mae_df = mae_df.dropna()

# Rank: 1 = lowest MAE (best)
rank_df = mae_df.rank(axis=1, method='min')

# ── Plot ────────────────────────────────────────────────────────────────────
colors = {
    'VDOT':          '#d62728',
    'Riegel':        '#ff7f0e',
    'Linear Reg.':   '#2ca02c',
    'Random Forest': '#1f77b4',
    'XGBoost':       '#9467bd',
}

fig, ax = plt.subplots(figsize=(10, 6))

x_positions = range(len(rank_df))

for name in rank_df.columns:
    ranks = rank_df[name].values
    ax.plot(
        x_positions, ranks,
        marker='o', markersize=8,
        linewidth=2.5,
        color=colors[name],
        label=name,
        zorder=3,
    )
    # Label the right end
    ax.text(
        x_positions[-1] + 0.15, ranks[-1], name,
        va='center', fontsize=9, color=colors[name], fontweight='bold',
    )

ax.set_xticks(x_positions)
ax.set_xticklabels(rank_df.index, rotation=45, ha='right')
ax.set_xlabel('Actual Finish Time Bucket (min)', fontsize=12)
ax.set_ylabel('Rank (1 = best)', fontsize=12)
ax.set_title('Model Performance Rank by Finish Time', fontsize=14)

# Invert y so rank 1 is on top
ax.invert_yaxis()
ax.set_yticks(range(1, len(preds) + 1))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# Legend on the left instead of overlapping labels
ax.legend(loc='lower left', frameon=True, fontsize=9)

plt.tight_layout()
plt.savefig('../results/figures/bump_chart_model_rank.png', dpi=150, bbox_inches='tight')
plt.show()
