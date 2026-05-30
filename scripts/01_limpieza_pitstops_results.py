import pandas as pd
import numpy as np
import os

print("Iniciando Fase 1: Extracción, Limpieza de Outliers (Doble Filtro) y Agrupación...")

# 1. Cargar datos
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

# 2. CONTEO DE PARADAS (Antes de filtrar)
# Contamos todas las paradas porque a nivel estratégico ocurrieron y hubo desgaste
conteos = pit_stops_merged.groupby(['raceId', 'constructorId'])['stop'].count().reset_index()
conteos.rename(columns={'stop': 'total_stops'}, inplace=True)

# 3. DOBLE FILTRO DE OUTLIERS
# FILTRO A: Límite Físico (Elimina banderas rojas y detenciones en garaje)
pit_stops_pre_limpios = pit_stops_merged[pit_stops_merged['duration_seg'] <= 100]

# FILTRO B: Método RIC por Carrera (Elimina errores lentos estándar de los mecánicos)
limites = pit_stops_pre_limpios.groupby('raceId')['duration_seg'].agg(
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75)
).reset_index()

limites['IQR'] = limites['Q3'] - limites['Q1']
limites['Upper_Bound'] = limites['Q3'] + 1.5 * limites['IQR']

# Unimos y aplicamos el filtro estadístico
pit_stops_pre_limpios = pd.merge(pit_stops_pre_limpios, limites[['raceId', 'Upper_Bound']], on='raceId', how='left')
pit_stops_limpios = pit_stops_pre_limpios[pit_stops_pre_limpios['duration_seg'] <= pit_stops_pre_limpios['Upper_Bound']]

# 4. CÁLCULO DE LA MEDIA (Solo con el rendimiento mecánico puro)
medias = pit_stops_limpios.groupby(['raceId', 'constructorId'])['duration_seg'].mean().reset_index()
medias.rename(columns={'duration_seg': 'mean_pit_stop'}, inplace=True)

# 5. ENSAMBLAJE Y GUARDADO
pit_stops_agg = pd.merge(medias, conteos, on=['raceId', 'constructorId'], how='inner')
pit_stops_agg['mean_pit_stop'] = pit_stops_agg['mean_pit_stop'].round(3)

ruta_salida = os.path.join('data', 'pit_stops_medias.csv')
pit_stops_agg.to_csv(ruta_salida, index=False)

print(f"¡Proceso completado! Se ha aplicado el Límite Físico y el RIC.")
print(f"Dataset guardado en: {ruta_salida}")