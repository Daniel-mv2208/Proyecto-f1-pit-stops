import pandas as pd
import numpy as np
import os

print("Starting Phase 1: Extraction, outlier filtering (double filter), and aggregation...")

# 1. Load data
pit_stops = pd.read_csv('data/pit_stops.csv')
results = pd.read_csv('data/results.csv')

def parse_duration(val):
    try:
        return float(val)
    except ValueError:
        parts = str(val).split(':')
        if len(parts) == 2:
            return (float(parts[0]) * 60) + float(parts[1])
        return np.nan

pit_stops['duration_sec'] = pit_stops['duration'].apply(parse_duration)

results_subset = results[['raceId', 'driverId', 'constructorId']]
pit_stops_merged = pd.merge(pit_stops, results_subset, on=['raceId', 'driverId'], how='inner')

# 2. Pit stop count (prior to filtering)
# Count all stops to reflect strategic race occurrences and tire wear
stop_counts = pit_stops_merged.groupby(['raceId', 'constructorId'])['stop'].count().reset_index()
stop_counts.rename(columns={'stop': 'total_stops'}, inplace=True)

# 3. Two-stage outlier filtering
# Filter A: Physical threshold (removes red flags and extended garage stops)
pit_stops_pre_cleaned = pit_stops_merged[pit_stops_merged['duration_sec'] <= 100]

# Filter B: IQR method per race (removes abnormal mechanical delay outliers)
bounds = pit_stops_pre_cleaned.groupby('raceId')['duration_sec'].agg(
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75)
).reset_index()

bounds['IQR'] = bounds['Q3'] - bounds['Q1']
bounds['Upper_Bound'] = bounds['Q3'] + 1.5 * bounds['IQR']

# Merge bounds and apply the statistical threshold
pit_stops_pre_cleaned = pd.merge(pit_stops_pre_cleaned, bounds[['raceId', 'Upper_Bound']], on='raceId', how='left')
pit_stops_cleaned = pit_stops_pre_cleaned[pit_stops_pre_cleaned['duration_sec'] <= pit_stops_pre_cleaned['Upper_Bound']]

# 4. Mean calculation (focusing on pure pit crew performance)
means = pit_stops_cleaned.groupby(['raceId', 'constructorId'])['duration_sec'].mean().reset_index()
means.rename(columns={'duration_sec': 'mean_pit_stop'}, inplace=True)

# 5. Merge and export
pit_stops_agg = pd.merge(means, stop_counts, on=['raceId', 'constructorId'], how='inner')
pit_stops_agg['mean_pit_stop'] = pit_stops_agg['mean_pit_stop'].round(3)

output_path = os.path.join('data', 'pit_stops_means.csv')
pit_stops_agg.to_csv(output_path, index=False)

print("Process complete: Physical limit and IQR filtering applied.")
print(f"Dataset saved to: {output_path}")