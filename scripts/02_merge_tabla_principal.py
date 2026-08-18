import pandas as pd
import numpy as np
import os

print("Starting Phase 2: Master assembly with circuit normalization (Delta)...")

# 1. Load required datasets
pit_stops_means = pd.read_csv('data/pit_stops_means.csv')
races = pd.read_csv('data/races.csv')
constructor_results = pd.read_csv('data/constructor_results.csv')
constructor_standings = pd.read_csv('data/constructor_standings.csv')
constructors = pd.read_csv('data/constructors.csv')

# Calculate race benchmark to obtain the relative delta
# 'transform' computes the race mean and broadcasts it to match the original group size
pit_stops_means['race_baseline'] = pit_stops_means.groupby('raceId')['mean_pit_stop'].transform('mean')
pit_stops_means['delta_pit_stop'] = (pit_stops_means['mean_pit_stop'] - pit_stops_means['race_baseline']).round(3)

# 2. Filter races (2011 - 2024) and include 'round' for chronological sorting
filtered_races = races[(races['year'] >= 2011) & (races['year'] <= 2024)][['raceId', 'year', 'round', 'name']]
filtered_races.rename(columns={'name': 'race_name'}, inplace=True)

# 3. Prepare constructors reference table
constructors_subset = constructors[['constructorId', 'name']].rename(columns={'name': 'constructor_name'})

# 4. Sequential merges
master_df = pd.merge(pit_stops_means, filtered_races, on='raceId', how='inner')

master_df = pd.merge(
    master_df, 
    constructor_results[['raceId', 'constructorId', 'points']], 
    on=['raceId', 'constructorId'], 
    how='left'
)
master_df.rename(columns={'points': 'race_points'}, inplace=True)

master_df = pd.merge(
    master_df, 
    constructor_standings[['raceId', 'constructorId', 'points', 'position']], 
    on=['raceId', 'constructorId'], 
    how='left'
)
master_df.rename(columns={'points': 'season_points', 'position': 'championship_standing'}, inplace=True)

master_df = pd.merge(master_df, constructors_subset, on='constructorId', how='left')

# 5. Create target feature based on Top 5 championship threshold
master_df['target_top5'] = np.where(master_df['championship_standing'] <= 5, 1, 0)

# 6. Sort by season calendar and standings
master_df.sort_values(by=['year', 'round', 'championship_standing'], ascending=[True, True, True], inplace=True)

# 7. Select and order final strategic columns
final_columns = [
    'year', 'race_name', 'constructor_name', 
    'total_stops', 'mean_pit_stop', 'delta_pit_stop',
    'race_points', 'season_points', 'championship_standing', 'target_top5'
]
master_df = master_df[final_columns]

# 8. Save the final master dataset
output_path = os.path.join('data', 'f1_master_dataset.csv')
master_df.to_csv(output_path, index=False)

print(f"\nETL pipeline completed successfully. Master dataset saved to: {output_path}")
print("\nSample of the final dataset with the delta feature:")
print(master_df[['race_name', 'constructor_name', 'mean_pit_stop', 'delta_pit_stop', 'target_top5']].head(10))