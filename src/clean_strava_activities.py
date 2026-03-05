"""Filter Strava activity data with 16-week marathon training window

Keeps only activities between 2024-09-21 and 2025-01-11, and retains
only start_date, distance, name, and moving_time columns
"""

import json
import pandas as pd
from datetime import datetime, timedelta, timezone

# --- Config ---
MARATHON_DATE = datetime(2025, 1, 12, tzinfo=timezone.utc)
TRAINING_WEEKS = 16
CUTOFF_DATE = MARATHON_DATE - timedelta(weeks=TRAINING_WEEKS)
KEEP_COLS = ["start_date", "distance", "name", "moving_time"]

# --- Load ---
data_path = "../data/raw/personal_strava_data/strava_data.jsonl"
with open(data_path, "r") as f:
    activities = [json.loads(line) for line in f if line.strip()]

df = pd.DataFrame(activities)
print(f"Total activities loaded: {len(df)}")

# --- Filter to training window ---
df["start_date"] = pd.to_datetime(df["start_date"])
mask = (df["start_date"] >= CUTOFF_DATE) & (df["start_date"] <= MARATHON_DATE)
df_filtered = df.loc[mask, KEEP_COLS].copy()
df_filtered = df_filtered.sort_values("start_date").reset_index(drop=True)

print(f"Activities in training window ({CUTOFF_DATE.date()} to {MARATHON_DATE.date()}): {len(df_filtered)}")
print(f"\nColumns: {list(df_filtered.columns)}")
print(f"\nFirst 5 rows:")
print(df_filtered.head())
print(f"\nLast 5 rows:")
print(df_filtered.tail())

# --- Save ---
output_path = "../data/raw/personal_strava_data/strava_training_window.parquet"
df_filtered.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")

