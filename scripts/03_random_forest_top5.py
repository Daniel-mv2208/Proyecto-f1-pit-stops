# ============================================================
# 04_random_forest_top5.py
# Modelo de Machine Learning: Random Forest para predecir Top 5
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
# 1. Crear carpetas de salida si no existen
# ------------------------------------------------------------

os.makedirs("outputs/graficos", exist_ok=True)
os.makedirs("outputs/modelos", exist_ok=True)
os.makedirs("outputs/metricas", exist_ok=True)


# ------------------------------------------------------------
# 2. Cargar dataset maestro
# ------------------------------------------------------------

df = pd.read_csv("data/dataset_maestro_f1.csv")

print("Dataset cargado correctamente")
print(df.head())
print(df.columns)


# ------------------------------------------------------------
# 3. Crear variables adicionales
# ------------------------------------------------------------

df["delta_negativo"] = (df["delta_pit_stop"] < 0).astype(int)
df["delta_abs"] = df["delta_pit_stop"].abs()


# ------------------------------------------------------------
# 4. Definir variable objetivo
# ------------------------------------------------------------

y = df["target_top5"]


# ------------------------------------------------------------
# 5. Definir variables predictoras
# OJO: No usamos championship_standing ni season_points
# para evitar fuga de información.
# ------------------------------------------------------------

X = df[
    [
        "year",
        "race_name",
        "constructor_name",
        "mean_pit_stop",
        "delta_pit_stop",
        "delta_negativo",
        "delta_abs",
        "total_stops"
    ]
]


# ------------------------------------------------------------
# 6. Separar variables numéricas y categóricas
# ------------------------------------------------------------

numeric_features = [
    "year",
    "mean_pit_stop",
    "delta_pit_stop",
    "delta_negativo",
    "delta_abs",
    "total_stops"
]

categorical_features = [
    "race_name",
    "constructor_name"
]


# ------------------------------------------------------------
# 7. Preprocesamiento
# OneHotEncoder convierte textos en variables numéricas.
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)


# ------------------------------------------------------------
# 8. Crear modelo Random Forest
# ------------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)


# ------------------------------------------------------------
# 9. Crear pipeline completo
# ------------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ]
)


# ------------------------------------------------------------
# 10. Dividir datos en entrenamiento y prueba
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 11. Entrenar modelo
# ------------------------------------------------------------

pipeline.fit(X_train, y_train)

print("Modelo entrenado correctamente")


# ------------------------------------------------------------
# 12. Hacer predicciones
# ------------------------------------------------------------

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 13. Calcular métricas
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

metricas = pd.DataFrame({
    "Metrica": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    "Valor": [accuracy, precision, recall, f1, roc_auc]
})

print("\nMétricas del modelo:")
print(metricas)

metricas.to_csv("outputs/metricas/random_forest_metricas.csv", index=False)


# ------------------------------------------------------------
# 14. Reporte de clasificación
# ------------------------------------------------------------

reporte = classification_report(y_test, y_pred)

print("\nReporte de clasificación:")
print(reporte)

with open("outputs/metricas/random_forest_reporte.txt", "w", encoding="utf-8") as f:
    f.write(reporte)


# ------------------------------------------------------------
# 15. Matriz de confusión
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Top 5", "Top 5"],
    yticklabels=["No Top 5", "Top 5"]
)
plt.title("Matriz de Confusión - Random Forest")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.savefig("outputs/graficos/matriz_confusion_random_forest.png")
plt.close()


# ------------------------------------------------------------
# 16. Importancia de variables
# ------------------------------------------------------------

model = pipeline.named_steps["model"]
encoder = pipeline.named_steps["preprocessor"]

feature_names_cat = encoder.named_transformers_["cat"].get_feature_names_out(categorical_features)
feature_names = list(feature_names_cat) + numeric_features

importances = model.feature_importances_

importancia_df = pd.DataFrame({
    "Variable": feature_names,
    "Importancia": importances
}).sort_values(by="Importancia", ascending=False)

importancia_df.to_csv("outputs/metricas/importancia_variables_random_forest.csv", index=False)

top_importancia = importancia_df.head(15)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_importancia,
    x="Importancia",
    y="Variable"
)
plt.title("Top 15 Variables más importantes - Random Forest")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
plt.savefig("outputs/graficos/importancia_variables_random_forest.png")
plt.close()


# ------------------------------------------------------------
# 17. Guardar modelo entrenado
# ------------------------------------------------------------

joblib.dump(pipeline, "outputs/modelos/random_forest_top5.pkl")

print("\nProceso finalizado correctamente.")
print("Modelo guardado en: outputs/modelos/random_forest_top5.pkl")
print("Métricas guardadas en: outputs/metricas/")
print("Gráficos guardados en: outputs/graficos/")