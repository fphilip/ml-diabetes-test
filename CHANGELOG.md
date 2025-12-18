# Changelog

## 2025-12-18

### Added: Test Suite para el modelo

**Nuevos archivos:**
- `datasets/test.csv` - Dataset de prueba con 30 casos del dataset original
- `src/test_model.py` - Script de testing que evalua el modelo y genera reportes

**Funcionalidad:**
- Carga los artefactos del modelo (`model.pkl`, `imputer.pkl`, `scaler.pkl`)
- Aplica el mismo preprocesamiento que en entrenamiento
- Genera predicciones y calcula metricas (Accuracy, Precision, Recall, F1)
- Crea matriz de confusion
- Analiza errores de prediccion
- Genera reporte MD automatico en `results/test_results_YYYYMMDD_HHMMSS.md`

**Ejecucion:**
```bash
python src/test_model.py
```

---

### Fixed: train_model.py - Error en carga de CSV

**Problema:** El script fallaba con el error:
```
ValueError: The least populated class in y has only 1 member, which is too few.
```

**Causa:** El archivo `datasets/diabetes.csv` ya contiene encabezados en la primera fila, pero el script estaba configurado con `header=None` y definiendo nombres de columnas manualmente. Esto causaba que la fila de encabezados se interpretara como datos, creando una clase con un solo miembro.

**Solución:** Se simplificó la carga del CSV para usar los encabezados existentes.

**Antes:**
```python
file_path = "datasets/diabetes.csv"
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df = pd.read_csv(file_path, header=None, names=column_names)
```

**Despues:**
```python
file_path = "datasets/diabetes.csv"
df = pd.read_csv(file_path)
```

**Archivo modificado:** `src/train_model.py:15-16`

**Resultado:** El modelo se entrena correctamente y genera los archivos en `models/`:
- `model.pkl`
- `imputer.pkl`
- `scaler.pkl`