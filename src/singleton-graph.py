"""
Plot smoothed MAE across the marathon finish-time distribution for all 5 models.

Assumes:
  - train.parquet / test.parquet in data/processed/
  - Models are imported
  - Target column: 'mf_ti_adj' (minutes)
  - Riegel prediction column: 'riegel_pred'
  - VDOT prediction column: 'vdot_pred'

Adjust column names to match your actual data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import joblib
from src.baselines import riegel_predict, vdot_predict
lr_model = joblib.load('../models/linear_regression.pkl')
rf_model = joblib.load('../models/random_forest_tuned.pkl')
gb_model = joblib.load('../models/gradient_boosting_tuned.pkl')

# ── 1. Load data ────────────────────────────────────────────────────────────
train = pd.read_parquet('../data/processed/train.parquet')
test  = pd.read_parquet('../data/processed/test.parquet')

# ── 2. Define target and features ───────────────────────────────────────────
TARGET = 'mf_ti_adj'        
RIEGEL_COL = 'riegel_pred'        
VDOT_COL   = 'vdot_pred'         

# Features used by ML models — adjust to match your feature set
FEATURE_COLS = [
    "age",
    "bmi",
    "female",
    "injury",
    "footwear",
    "mh_ti_adj_final",
    "mh_ti_adj_imputed_flag",
    "tempo",
    "sprint",
    "typical",
    "mean_vdot",
    "vdot_consistency",
]

X_train = train[FEATURE_COLS]
y_train = train[TARGET]
X_test  = test[FEATURE_COLS]
y_test  = test[TARGET]

# ── 3. Get baseline predictions ────────────────────────────────────────────
# If these columns already exist in test, just grab them.
# Otherwise compute them from your baselines module.
riegel_preds = test["mh_ti_adj_final"].apply(lambda t: riegel_predict(race_time=t, race_distance=13.1, target_distance=26.2))
vdot_preds   = test["mean_vdot"].apply(lambda v: vdot_predict(vdot=v, race_distance=26.2))

# ── 4. Train ML models (or load saved ones) ────────────────────────────────
lr = lr_model
lr_preds = lr.predict(X_test)

rf = rf_model
rf_preds = rf.predict(X_test)

# Use your tuned hyperparams here
xgb = gb_model
xgb_preds = xgb.predict(X_test)

# ── 5. Compute absolute errors ─────────────────────────────────────────────
actual = y_test.values

models = {
    'VDOT':            np.abs(actual - vdot_preds),
    'Riegel':          np.abs(actual - riegel_preds),
    'Linear Reg.':     np.abs(actual - lr_preds),
    'Random Forest':   np.abs(actual - rf_preds),
    'XGBoost':         np.abs(actual - xgb_preds),
}

# ── 6. LOWESS smoothing and plot ───────────────────────────────────────────
# frac controls smoothing bandwidth — higher = smoother.
# With ~100 test points you'll want something in the 0.3–0.5 range;
# with more data you can go lower (0.15–0.25) for more local detail.
LOWESS_FRAC = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    'VDOT':           '#d62728',   # red
    'Riegel':         '#ff7f0e',   # orange
    'Linear Reg.':    '#2ca02c',   # green
    'Random Forest':  '#1f77b4',   # blue
    'XGBoost':        '#9467bd',   # purple
}

for name, abs_errors in models.items():
    smoothed = lowess(
        abs_errors,
        actual,
        frac=LOWESS_FRAC,
        return_sorted=True,
    )
    ax.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        label=name,
        color=colors[name],
        linewidth=2,
    )

# Optional: add a light rug or scatter of actual test points
ax.scatter(
    actual,
    [0] * len(actual),
    alpha=0.3,
    s=10,
    color='grey',
    zorder=0,
    label='_nolegend_',
)

# ── 7. Format ──────────────────────────────────────────────────────────────
ax.set_xlabel('Actual Marathon Time (min)', fontsize=12)
ax.set_ylabel('Absolute Error (min)', fontsize=12)
ax.set_title('Prediction Error by Finish Time', fontsize=14)
ax.legend(frameon=True, fontsize=10)
ax.set_ylim(bottom=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../results/figures/mae_by_finish_time.png', dpi=150)
plt.show()

