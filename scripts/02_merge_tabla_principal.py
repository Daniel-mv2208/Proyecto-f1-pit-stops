import pandas as pd
import numpy as np
import os

print("Iniciando Fase 2: Ensamblaje Maestro y Creación de Variable Target (Top 5)...")

pit_stops_medias = pd.read_csv('data/pit_stops_medias.csv')
races = pd.read_csv('data/races.csv')
constructor_results = pd.read_csv('data/constructor_results.csv')
constructor_standings = pd.read_csv('data/constructor_standings.csv')
constructors = pd.read_csv('data/constructors.csv')

races_filtradas = races[(races['year'] >= 2011) & (races['year'] <= 2024)][['raceId', 'year', 'round', 'name']]
races_filtradas.rename(columns={'name': 'race_name'}, inplace=True)

constructors_subset = constructors[['constructorId', 'name']].rename(columns={'name': 'constructor_name'})

df_maestro = pd.merge(pit_stops_medias, races_filtradas, on='raceId', how='inner')

df_maestro = pd.merge(df_maestro, constructor_results[['raceId', 'constructorId', 'points']], 
                      on=['raceId', 'constructorId'], how='left')
df_maestro.rename(columns={'points': 'race_points'}, inplace=True)

df_maestro = pd.merge(df_maestro, constructor_standings[['raceId', 'constructorId', 'points', 'position']], 
                      on=['raceId', 'constructorId'], how='left')
df_maestro.rename(columns={'points': 'season_points', 'position': 'championship_standing'}, inplace=True)

df_maestro = pd.merge(df_maestro, constructors_subset, on='constructorId', how='left')

# MODIFICACIÓN CLAVE: Creación de la variable objetivo (Target)
# 1 si está en el Top 5, 0 si no lo está.
df_maestro['target_top5'] = np.where(df_maestro['championship_standing'] <= 5, 1, 0)

df_maestro.sort_values(by=['year', 'round', 'championship_standing'], ascending=[True, True, True], inplace=True)

# Actualizamos las columnas finales para incluir total_stops y target_top5
columnas_finales = [
    'year', 'race_name', 'constructor_name', 
    'total_stops', 'mean_pit_stop', 
    'race_points', 'season_points', 'championship_standing', 'target_top5'
]
df_maestro = df_maestro[columnas_finales]

ruta_salida = os.path.join('data', 'dataset_maestro_f1.csv')
df_maestro.to_csv(ruta_salida, index=False)

print(f"\n¡Ensamblaje completado! Dataset maestro guardado en: {ruta_salida}")
print("\nPrimeras 5 filas del dataset final:")
print(df_maestro[['constructor_name', 'total_stops', 'mean_pit_stop', 'championship_standing', 'target_top5']].head())