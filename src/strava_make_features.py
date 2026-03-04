"""Build a single-row feature vector for personal marathon prediction.

Uses the filtered Strava training window to calculate typical weekly
mileage, and hardcodes the remaining features from user-provided values.
"""

import pandas as pd
import numpy as np

# --- Load training window ---
df = pd.read_parquet("../data/raw/personal_strava_data/strava_training_window.parquet")

# --- Calculate typical weekly mileage (excluding the marathon) ---
MARATHON_DISTANCE_THRESHOLD = 40_000  # metres; marathon is ~42,195m
training_runs = df[df["distance"] < MARATHON_DISTANCE_THRESHOLD].copy()

print(f"Total activities: {len(df)}")
print(f"Activities after excluding marathon: {len(training_runs)}")

# Assign each activity to an ISO week
training_runs["week"] = training_runs["start_date"].dt.isocalendar().week
training_runs["year"] = training_runs["start_date"].dt.isocalendar().year

# Convert distance from metres to miles
METRES_PER_MILE = 1609.344
weekly_miles = (
    training_runs
    .groupby(["year", "week"])["distance"]
    .sum()
    / METRES_PER_MILE
)

# Drop weeks under 15 miles (not real training weeks)
# Drop weeks under 20 miles (taper/race week)
low_weeks = weekly_miles[weekly_miles < 25]
if len(low_weeks) > 0:
    print(f"\nDropping {len(low_weeks)} week(s) under 25 miles:")
    for (y, w), val in low_weeks.items():
        print(f"  Year {y}, Week {w}: {val:.1f} miles")

weekly_miles = weekly_miles[weekly_miles >= 25]
typical_mileage = weekly_miles.mean()
print(f"\nWeekly mileage stats (after filtering):")
print(f"  Mean:   {weekly_miles.mean():.1f} miles")
print(f"  Median: {weekly_miles.median():.1f} miles")
print(f"  Std:    {weekly_miles.std():.1f} miles")
print(f"  Weeks:  {len(weekly_miles)}")

# --- VDOT consistency ---
vdot_a, vdot_b = 53, 54
mean_vdot = 53.5
vdot_consistency = 1 - (abs(vdot_a - vdot_b) / mean_vdot)
print(f"\nVDOT consistency: {vdot_consistency:.4f}")

# --- Assemble feature row ---
features = pd.DataFrame([{
    "age": 19,
    "bmi": 23.2,
    "female": 0,
    "injury": 0,
    "footwear": 1,
    "mh_ti_adj_final": 88,
    "mh_ti_adj_imputed_flag": 1,
    "tempo": 1,
    "sprint": 1,
    "typical": round(typical_mileage, 2),
    "mean_vdot": mean_vdot,
    "vdot_consistency": round(vdot_consistency, 4),
}])

print(f"\nFeature vector:")
print(features.to_string(index=False))

# --- Save ---
output_path = "../data/processed/personal_features.parquet"
features.to_parquet("../data/processed/personal_features.parquet", index=False)
features.to_csv("../data/processed/personal_features.csv", index=False)
print("\nSaved to personal_features.parquet and personal_features.csv")

