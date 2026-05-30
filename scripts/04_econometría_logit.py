import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Iniciando análisis econométrico (Modelo Logit)...")

# 1. Asegurar que la carpeta 'outputs' existe
os.makedirs('outputs', exist_ok=True)

# 2. Cargar el dataset maestro
df = pd.read_csv('data/dataset_maestro_f1.csv')

# 3. ESPECIFICACIÓN DEL MODELO LOGIT 
# target_top5 es nuestra variable dependiente dicotómica
# delta_pit_stop y total_stops son nuestras variables independientes
formula = 'target_top5 ~ delta_pit_stop + total_stops'
modelo_logit = smf.logit(formula=formula, data=df).fit()

# 4. EXPORTAR EL REPORTE ESTADÍSTICO TRADICIONAL
# Generamos el resumen estadístico 
resumen = modelo_logit.summary()
ruta_reporte = os.path.join('outputs', 'reporte_econometria_logit.txt')

with open(ruta_reporte, 'w') as f:
    f.write("=== REPORTE ECONOMÉTRICO: REGRESIÓN LOGÍSTICA (F1) ===\n")
    f.write("Basado en la metodología de D. Gujarati (Modelos de respuesta cualitativa)\n\n")
    f.write(str(resumen))
    
    # Añadimos el cálculo de Odds Ratios (Exponencial de los coeficientes)
    f.write("\n\n=== ODDS RATIOS (Razón de Probabilidades) ===\n")
    odds_ratios = np.exp(modelo_logit.params)
    f.write(str(odds_ratios))

print(f"Reporte estadístico guardado en: {ruta_reporte}")

# 5. VISUALIZACIONES
sns.set_theme(style="whitegrid")

# --- Visual 1: La Curva Sigmoidea (Probabilidad vs Delta) ---
plt.figure(figsize=(10, 6))
# Fijamos total_stops en su mediana (ej. 2 paradas) para aislar el efecto del Delta
mediana_stops = df['total_stops'].median()
rango_delta = np.linspace(df['delta_pit_stop'].min(), df['delta_pit_stop'].max(), 300)
datos_pred = pd.DataFrame({'delta_pit_stop': rango_delta, 'total_stops': mediana_stops})

# Predecir probabilidades
probabilidades = modelo_logit.predict(datos_pred)

plt.plot(rango_delta, probabilidades, color='crimson', linewidth=3, label=f'Curva Logit (Paradas = {mediana_stops})')
plt.scatter(df['delta_pit_stop'], df['target_top5'], alpha=0.1, color='black', label='Observaciones reales')
plt.axvline(x=0, color='gray', linestyle='--', label='Promedio de Carrera (Delta = 0)')
plt.title('Probabilidad de entrar al Top 5 según el Delta en Boxes', fontsize=14, fontweight='bold')
plt.xlabel('Delta Pit Stop (Segundos respecto al promedio)', fontsize=12)
plt.ylabel('Probabilidad Estimada P(Top 5 = 1)', fontsize=12)
plt.legend()
plt.tight_layout()
ruta_grafico_1 = os.path.join('outputs', 'curva_probabilidad_logit.png')
plt.savefig(ruta_grafico_1, dpi=300)
plt.close()

# --- Visual 2: Forest Plot de Odds Ratios (Impacto Visual) ---
plt.figure(figsize=(8, 4))
# Extraemos intervalos de confianza y calculamos exp()
conf = modelo_logit.conf_int()
conf['Odds Ratio'] = odds_ratios
conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
# Omitimos el intercepto para el gráfico
conf = conf.drop('Intercept')

# Calculamos los errores para las barras
errores_bajos = conf['Odds Ratio'] - np.exp(conf['2.5%'])
errores_altos = np.exp(conf['97.5%']) - conf['Odds Ratio']

plt.errorbar(x=conf['Odds Ratio'], y=conf.index, 
             xerr=[errores_bajos, errores_altos], 
             fmt='o', color='navy', markersize=10, capsize=5, linewidth=2)
plt.axvline(x=1, color='red', linestyle='--', label='Sin efecto (OR = 1)')
plt.title('Impacto de Variables (Odds Ratios)', fontsize=14, fontweight='bold')
plt.xlabel('Multiplicador de Probabilidad (Odds Ratio)', fontsize=12)
plt.legend()
plt.tight_layout()
ruta_grafico_2 = os.path.join('outputs', 'odds_ratios_impacto.png')
plt.savefig(ruta_grafico_2, dpi=300)
plt.close()

print(f"Gráficos econométricos guardados en la carpeta 'outputs/'. ¡Proceso completado!")