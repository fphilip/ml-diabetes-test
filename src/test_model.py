import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

# --- 1. Cargar artefactos del modelo ---
models_dir = 'models'
imputer = joblib.load(os.path.join(models_dir, 'imputer.pkl'))
scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
model = joblib.load(os.path.join(models_dir, 'model.pkl'))

# --- 2. Cargar datos de test ---
test_path = "datasets/test.csv"
df_test = pd.read_csv(test_path)

# Separar features y target
X_test = df_test.drop("Outcome", axis=1)
y_true = df_test["Outcome"]

# --- 3. Preprocesamiento (mismo que en entrenamiento) ---
# Reemplazar ceros por NaN en columnas biologicas
cols_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X_test[cols_to_replace] = X_test[cols_to_replace].replace(0, np.nan)

# Aplicar imputacion y escalado
X_imputed = imputer.transform(X_test)
X_scaled = scaler.transform(X_imputed)

# --- 4. Predicciones ---
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

# --- 5. Metricas ---
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

# --- 6. Generar reporte MD ---
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Crear DataFrame con resultados detallados
df_results = df_test.copy()
df_results['Predicted'] = y_pred
df_results['Probability'] = np.round(y_proba, 4)
df_results['Correct'] = (y_true == y_pred).map({True: 'Yes', False: 'No'})

# Contar aciertos y errores
correct_count = (y_true == y_pred).sum()
incorrect_count = len(y_true) - correct_count

# Contar por clase
total_negative = (y_true == 0).sum()
total_positive = (y_true == 1).sum()
predicted_negative = (y_pred == 0).sum()
predicted_positive = (y_pred == 1).sum()

# Generar contenido MD
md_content = f"""# Test Results - Diabetes Prediction Model

**Execution Date:** {timestamp}

## Summary

| Metric | Value |
|--------|-------|
| Total Samples | {len(y_true)} |
| Correct Predictions | {correct_count} |
| Incorrect Predictions | {incorrect_count} |
| **Accuracy** | {accuracy:.4f} ({accuracy * 100:.2f}%) |
| **Precision** | {precision:.4f} |
| **Recall** | {recall:.4f} |
| **F1-Score** | {f1:.4f} |

## Dataset Distribution (Outcome)

| Class | Actual (Test Data) | Predicted |
|-------|-------------------|-----------|
| Non-diabetic (0) | {total_negative} | {predicted_negative} |
| Diabetic (1) | {total_positive} | {predicted_positive} |
| **Total** | **{len(y_true)}** | **{len(y_pred)}** |

## Confusion Matrix

|  | Predicted Negative (0) | Predicted Positive (1) |
|--|------------------------|------------------------|
| **Actual Negative (0)** | {cm[0][0]} (TN) | {cm[0][1]} (FP) |
| **Actual Positive (1)** | {cm[1][0]} (FN) | {cm[1][1]} (TP) |

### Interpretation
- **True Negatives (TN):** {cm[0][0]} - Correctly predicted as non-diabetic
- **True Positives (TP):** {cm[1][1]} - Correctly predicted as diabetic
- **False Positives (FP):** {cm[0][1]} - Incorrectly predicted as diabetic (Type I error)
- **False Negatives (FN):** {cm[1][0]} - Incorrectly predicted as non-diabetic (Type II error)

## Detailed Results

| # | Pregnancies | Glucose | BloodPressure | BMI | Age | Actual | Predicted | Probability | Correct |
|---|-------------|---------|---------------|-----|-----|--------|-----------|-------------|---------|
"""

# Agregar filas de resultados
for idx, row in df_results.iterrows():
    md_content += f"| {idx + 1} | {int(row['Pregnancies'])} | {int(row['Glucose'])} | {int(row['BloodPressure'])} | {row['BMI']:.1f} | {int(row['Age'])} | {int(row['Outcome'])} | {int(row['Predicted'])} | {row['Probability']:.4f} | {row['Correct']} |\n"

# Agregar analisis de errores
errors = df_results[df_results['Correct'] == 'No']
if len(errors) > 0:
    md_content += f"""
## Error Analysis

The model made {len(errors)} incorrect prediction(s):

"""
    for idx, row in errors.iterrows():
        actual = "Diabetic" if row['Outcome'] == 1 else "Non-diabetic"
        predicted = "Diabetic" if row['Predicted'] == 1 else "Non-diabetic"
        md_content += f"- **Sample {idx + 1}:** Actual: {actual}, Predicted: {predicted} (Probability: {row['Probability']:.4f})\n"

md_content += f"""
## Model Information

- **Model Type:** XGBClassifier (XGBoost)
- **Hyperparameters:** n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.86
- **Preprocessing:** KNNImputer (k=5) + RobustScaler
- **Test Dataset:** `datasets/test.csv`
- **Artifacts Used:** `models/model.pkl`, `models/imputer.pkl`, `models/scaler.pkl`

---
*Report generated automatically by test_model.py*
"""

# Guardar reporte
report_filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
report_path = os.path.join(results_dir, report_filename)
with open(report_path, 'w') as f:
    f.write(md_content)

# Imprimir resumen en consola
print(f"{'=' * 60}")
print(f"TEST RESULTS - Diabetes Prediction Model")
print(f"{'=' * 60}")
print(f"Execution: {timestamp}")
print(f"Samples: {len(y_true)}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"{'=' * 60}")
print(f"Report saved to: {report_path}")
