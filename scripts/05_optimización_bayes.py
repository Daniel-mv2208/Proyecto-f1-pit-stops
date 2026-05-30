import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Iniciando Fase de Optimización Empresarial (Teorema de Bayes)...")

# 1. Asegurar que la subcarpeta 'optimizacion' existe
carpeta_salida = os.path.join('outputs', 'optimizacion')
os.makedirs(carpeta_salida, exist_ok=True)

# 2. Cargar el dataset maestro
df = pd.read_csv('data/dataset_maestro_f1.csv')

# 3. DEFINICIÓN DE LOS EVENTOS ESTRATÉGICOS
total_carreras_equipos = len(df)
es_top5 = df['target_top5'] == 1
es_eficiente = df['delta_pit_stop'] < 0.0

# 4. CÁLCULO DE LAS PROBABILIDADES
p_A = df[es_top5].shape[0] / total_carreras_equipos
p_B = df[es_eficiente].shape[0] / total_carreras_equipos
p_B_dado_A = df[es_top5 & es_eficiente].shape[0] / df[es_top5].shape[0]

# 5. APLICACIÓN DEL TEOREMA DE BAYES
p_A_dado_B_bayes = (p_B_dado_A * p_A) / p_B
incremento_estrategico = p_A_dado_B_bayes - p_A

# 6. GENERACIÓN DEL INFORME TEXTUAL
ruta_reporte = os.path.join(carpeta_salida, 'reporte_optimizacion_bayes.txt')
with open(ruta_reporte, 'w', encoding='utf-8') as f:
    f.write("=== INFORME DE OPTIMIZACIÓN EMPRESARIAL: TEOREMA DE BAYES ===\n")
    f.write("Variable de decisión: Mantenimiento de un Delta de Pit Stop Negativo (< 0.0s)\n")
    f.write("-" * 70 + "\n\n")
    f.write("1. PROBABILIDADES BASE:\n")
    f.write(f"   - P(A)  | Probabilidad Base de ser Top 5:            {p_A:.1%}\n")
    f.write(f"   - P(B)  | Probabilidad de lograr un pit stop rápido: {p_B:.1%}\n")
    f.write(f"   - P(B|A)| De los equipos Top 5, % que son rápidos:   {p_B_dado_A:.1%}\n\n")
    f.write("2. APLICACIÓN DEL TEOREMA DE BAYES P(A|B):\n")
    f.write(f"   - Probabilidad de ser Top 5 SI se opera con Delta negativo: {p_A_dado_B_bayes:.1%}\n\n")
    f.write("3. CONCLUSIÓN ESTRATÉGICA:\n")
    f.write(f"   El incremento estratégico absoluto es del {incremento_estrategico:.1%}.\n")

# 7. VISUALIZACIÓN 1: IMPACTO ESTRATÉGICO (Mantenemos el gráfico original)
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

escenarios_impacto = ['Probabilidad Base\n(P(A) General)', 'Probabilidad Optimizada\n(P(A|B) Bayesiana)']
prob_impacto = [p_A * 100, p_A_dado_B_bayes * 100]
colores_impacto = ['#95a5a6', '#27ae60']

barras_impacto = plt.bar(escenarios_impacto, prob_impacto, color=colores_impacto, width=0.5)

for barra in barras_impacto:
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2., altura + 1.5,
             f'{altura:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.annotate(f'Incremento:\n+{incremento_estrategico*100:.1f}%', 
             xy=(0.5, p_A * 100), 
             xytext=(0.5, (p_A * 100) + 10),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             ha='center', va='center', fontweight='bold', color='black',
             bbox=dict(boxstyle="round,pad=0.3", fc="#f1c40f", ec="black", lw=1))

plt.title('Impacto Estratégico de la Eficiencia en Boxes', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Probabilidad de clasificar al Top 5 (%)', fontsize=12)
plt.ylim(0, 100)
plt.tight_layout()
ruta_grafico_1 = os.path.join(carpeta_salida, 'comparativa_bayesiana.png')
plt.savefig(ruta_grafico_1, dpi=300)
plt.close()

# 8. VISUALIZACIÓN 2: PROBABILIDADES FUNDAMENTALES (NUEVO GRÁFICO)
plt.figure(figsize=(9, 6))

etiquetas_base = [
    'P(A)\nProbabilidad de\nser Top 5', 
    'P(B)\nProbabilidad de\nDelta Negativo', 
    'P(B|A)\nDelta Negativo dado\nque es Top 5'
]
valores_base = [p_A * 100, p_B * 100, p_B_dado_A * 100]
# Usamos colores azules y púrpuras para diferenciarlo visualmente del gráfico de conclusiones
colores_base = ['#3498db', '#34495e', '#8e44ad'] 

barras_base = plt.bar(etiquetas_base, valores_base, color=colores_base, width=0.6)

for barra in barras_base:
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2., altura + 1.5,
             f'{altura:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.title('Desglose de Probabilidades Fundamentales (Bayes)', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Probabilidad (%)', fontsize=12)
plt.ylim(0, 100)
plt.tight_layout()
ruta_grafico_2 = os.path.join(carpeta_salida, 'probabilidades_base_bayes.png')
plt.savefig(ruta_grafico_2, dpi=300)
plt.close()

print(f"¡Cálculos Bayesianos finalizados!")
print(f"Informe guardado en: {ruta_reporte}")
print(f"Gráfico de impacto guardado en: {ruta_grafico_1}")
print(f"Gráfico de desglose base guardado en: {ruta_grafico_2}")