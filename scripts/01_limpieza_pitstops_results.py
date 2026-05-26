import pandas as pd
import numpy as np
import os

print("Iniciando Fase 1: Extracción de Medias y Total de Paradas...")

pit_stops = pd.read_csv('data/pit_stops.csv')
results = pd.read_csv('data/results.csv')

def limpiar_duracion(valor):
    try:
        return float(valor)
    except ValueError:
        partes = str(valor).split(':')
        if len(partes) == 2:
            return (float(partes[0]) * 60) + float(partes[1])
        return np.nan

pit_stops['duration_seg'] = pit_stops['duration'].apply(limpiar_duracion)

results_subset = results[['raceId', 'driverId', 'constructorId']]
pit_stops_merged = pd.merge(pit_stops, results_subset, on=['raceId', 'driverId'], how='inner')

# MODIFICACIÓN CLAVE: Calculamos la media Y el conteo total de paradas del equipo
pit_stops_agg = pit_stops_merged.groupby(['raceId', 'constructorId']).agg(
    mean_pit_stop=('duration_seg', 'mean'),
    total_stops=('stop', 'count')
).reset_index()

pit_stops_agg['mean_pit_stop'] = pit_stops_agg['mean_pit_stop'].round(3)

ruta_salida = os.path.join('data', 'pit_stops_medias.csv')
pit_stops_agg.to_csv(ruta_salida, index=False)

print(f"¡Proceso completado! Dataset guardado en: {ruta_salida}")
print(pit_stops_agg.head())