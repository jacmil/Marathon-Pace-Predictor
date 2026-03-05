"""
SHAP waterfall plot for a single personal prediction.

Shows how each feature pushed the prediction up or down from
the base value to the final predicted marathon time.

Reads personal features from data/processed/personal_features.parquet.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

# ── Load model and training data (SHAP needs background data) ──────────────
gb_model = joblib.load('../models/gradient_boosting_tuned.pkl')
train = pd.read_parquet('../data/processed/train.parquet')

FEATURE_COLS = [
    "age", "bmi", "female", "injury", "footwear",
    "mh_ti_adj_final", "mh_ti_adj_imputed_flag",
    "tempo", "sprint", "typical",
    "mean_vdot", "vdot_consistency",
]

X_train = train[FEATURE_COLS]

# ── Load personal feature vector ───────────────────────────────────────────
personal = pd.read_parquet('../data/processed/personal_features.parquet')
X_personal = personal[FEATURE_COLS]
X_train = X_train.astype(float)
X_personal = X_personal.astype(float)

# ── Compute SHAP values ───────────────────────────────────────────────────
explainer = shap.TreeExplainer(gb_model, data=X_train)
shap_values = explainer(X_personal)

# ── Plot waterfall for the single prediction ──────────────────────────────
# shap_values[0] is the first (and only) row
fig = plt.figure(figsize=(10, 7))
shap.waterfall_plot(shap_values[0], show=False, max_display=12)

plt.title('How Each Feature Shaped My Marathon Prediction', fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(
    '../results/figures/shap_waterfall_personal.png',
    dpi=150, bbox_inches='tight',
)
plt.show()
