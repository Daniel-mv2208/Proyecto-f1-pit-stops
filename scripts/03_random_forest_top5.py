import os
import pandas as pd
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

# 1. Directory and plotting configuration
DATA_PATH = os.path.join("data", "f1_master_dataset.csv")
OUTPUT_DIR = os.path.join("outputs", "machine_learning")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# 2. Data loading and preparation (operational features only to prevent data leakage)
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

FEATURE_COLS = ["delta_pit_stop", "total_stops"]
TARGET_COL = "target_top5"

missing_cols = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df.columns]
if missing_cols:
    raise KeyError(f"Missing required columns in dataset: {missing_cols}")

# Drop missing values
df_model = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

X = df_model[FEATURE_COLS]
y = df_model[TARGET_COL]

# 3. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y
)

# 4. Train Random Forest classifier
rf_operational = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

rf_operational.fit(X_train, y_train)

# Model predictions
y_pred = rf_operational.predict(X_test)
y_proba = rf_operational.predict_proba(X_test)[:, 1]

# 5. Generate and export performance report (.txt)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
report_str = classification_report(y_test, y_pred, target_names=["Outside Top 5 (0)", "Top 5 (1)"])

report_content = f"""=== RANDOM FOREST MODEL: OPERATIONAL FOCUS ===
Features used: {', '.join(FEATURE_COLS)}
------------------------------------------------------------

Overall Accuracy: {acc * 100:.2f}%
ROC-AUC Score: {auc:.4f}

DETAILED METRICS:
{report_str}
"""

txt_report_path = os.path.join(OUTPUT_DIR, "operational_random_forest_report.txt")
with open(txt_report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"[OK] Text report saved to: {txt_report_path}")

# 6. Generate visual artifacts

# A. Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=["Outside Top 5 (0)", "Top 5 (1)"],
    yticklabels=["Outside Top 5 (0)", "Top 5 (1)"]
)
plt.title("Confusion Matrix - Operational Random Forest", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "operational_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"[OK] Confusion matrix saved to: {cm_path}")

# B. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Random Forest (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve - Pit Stop Operational Model", fontsize=12, fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "operational_roc_curve.png")
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"[OK] ROC curve saved to: {roc_path}")

# C. Feature Importance with bar value labels
importances = rf_operational.feature_importances_
df_importance = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 4.5))
ax = sns.barplot(data=df_importance, x="Importance", y="Feature", palette="viridis")

# Extend x-axis margin so value annotations are not clipped
max_importance = df_importance["Importance"].max()
ax.set_xlim(0, max_importance * 1.25)

# Annotate each bar with numerical and percentage values
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

plt.title("Feature Importance - Operational Model", fontsize=12, fontweight="bold")
plt.xlabel("Relative Predictive Weight")
plt.ylabel("Operational Feature")
plt.tight_layout()

imp_path = os.path.join(OUTPUT_DIR, "operational_feature_importance.png")
plt.savefig(imp_path, dpi=300)
plt.close()
print(f"[OK] Feature importance plot saved to: {imp_path}")