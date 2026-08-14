import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score
)

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE DIRECTORIOS Y ESTILO VISUAL
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "dataset_maestro_f1.csv")
OUTPUT_DIR = os.path.join("outputs", "machine_learning")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# -----------------------------------------------------------------------------
# 2. CARGA Y PREPARACIÓN DE DATOS (ENFOQUE OPERATIVO PURO)
# -----------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encontró el archivo de datos en: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Variables operativas para evitar fuga de información (Data Leakage)
FEATURE_COLS = ["delta_pit_stop", "total_stops"]
TARGET_COL = "target_top5"

missing_cols = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df.columns]
if missing_cols:
    raise KeyError(f"Faltan las siguientes columnas en el dataset: {missing_cols}")

# Limpieza de nulos
df_model = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

X = df_model[FEATURE_COLS]
y = df_model[TARGET_COL]

# -----------------------------------------------------------------------------
# 3. DIVISIÓN DE DATOS (TRAIN / TEST)
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y
)

# -----------------------------------------------------------------------------
# 4. ENTRENAMIENTO DEL MODELO RANDOM FOREST
# -----------------------------------------------------------------------------
rf_operativo = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

rf_operativo.fit(X_train, y_train)

# Predicciones
y_pred = rf_operativo.predict(X_test)
y_proba = rf_operativo.predict_proba(X_test)[:, 1]

# -----------------------------------------------------------------------------
# 5. GENERACIÓN Y GUARDADO DEL REPORTE EN ARCHIVO .TXT
# -----------------------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
report_str = classification_report(y_test, y_pred, target_names=["Fuera (0)", "Top 5 (1)"])

reporte_contenido = f"""=== MODELO RANDOM FOREST: ENFOQUE OPERATIVO ===
Variables utilizadas: {', '.join(FEATURE_COLS)}
------------------------------------------------------------

Exactitud Global (Accuracy): {acc * 100:.2f}%
ROC-AUC Score: {auc:.4f}

MÉTRICAS DETALLADAS:
{report_str}
"""

txt_report_path = os.path.join(OUTPUT_DIR, "reporte_random_forest_operativo.txt")
with open(txt_report_path, "w", encoding="utf-8") as f:
    f.write(reporte_contenido)

print(f"[OK] Reporte de texto guardado en: {txt_report_path}")

# -----------------------------------------------------------------------------
# 6. GENERACIÓN DE ARTEFACTOS VISUALES
# -----------------------------------------------------------------------------

# A. Matriz de Confusión
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=["Fuera (0)", "Top 5 (1)"],
    yticklabels=["Fuera (0)", "Top 5 (1)"]
)
plt.title("Matriz de Confusión - Random Forest Operativo", fontsize=12, fontweight="bold")
plt.xlabel("Predicción del Modelo")
plt.ylabel("Valor Real")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "matriz_confusion_operativo.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"[OK] Matriz de confusión guardada en: {cm_path}")

# B. Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Random Forest (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Clasificador Aleatorio")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
plt.title("Curva ROC - Enfoque Operativo de Pit Stops", fontsize=12, fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "curva_roc_operativo.png")
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"[OK] Curva ROC guardada en: {roc_path}")

# C. Importancia de Características con Etiquetas de Valor en las Barras
importances = rf_operativo.feature_importances_
df_importance = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 4.5))
ax = sns.barplot(data=df_importance, x="Importance", y="Feature", palette="viridis")

# Ampliar margen del eje X para que no se corten las etiquetas de texto
max_importance = df_importance["Importance"].max()
ax.set_xlim(0, max_importance * 1.25)

# Añadir el valor numérico y porcentaje exacto al extremo de cada barra
for p in ax.patches:
    val = p.get_width()
    ax.annotate(
        f"{val:.3f} ({val * 100:.1f}%)",
        (val, p.get_y() + p.get_height() / 2.0),
        ha="left",
        va="center",
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold"
    )

plt.title("Importancia de Variables (Feature Importance Operativo)", fontsize=12, fontweight="bold")
plt.xlabel("Peso Predictivo Relativo")
plt.ylabel("Variable Operativa")
plt.tight_layout()

imp_path = os.path.join(OUTPUT_DIR, "importancia_caracteristicas_operativo.png")
plt.savefig(imp_path, dpi=300)
plt.close()
print(f"[OK] Gráfico de importancias guardado en: {imp_path}")