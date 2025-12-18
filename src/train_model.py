
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# --- 1. Carga de Datos ---
# Descargar el dataset desde la URL de UCI o cargarlo localmente
# En este caso, lo cargamos desde un archivo local para reproducibilidad.
file_path = "datasets/diabetes.csv"
df = pd.read_csv(file_path)

# --- 2. Preprocesamiento ---
# Variables biológicas donde un cero es fisiológicamente improbable
cols_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)

# Separar características (X) y objetivo (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# --- 3. Definición del Pipeline de Entrenamiento ---
# El pipeline encapsula los pasos de preprocesamiento y el modelo.
# Esto asegura que los mismos pasos se apliquen tanto en entrenamiento como en predicción.

# Paso 1: Imputación de valores nulos con KNNImputer
# Se utiliza KNN para encontrar los vecinos más cercanos y estimar los valores faltantes.
imputer = KNNImputer(n_neighbors=5)

# Paso 2: Escalado de características con RobustScaler
# Es menos sensible a outliers que StandardScaler.
scaler = RobustScaler()

# Paso 3: Modelo de clasificación
# XGBoost - Ganador del análisis comparativo del notebook
# Hiperparámetros optimizados mediante RandomizedSearchCV
model = XGBClassifier(
    n_estimators=300,           # Número de árboles (optimizado: 100-500)
    learning_rate=0.05,         # Velocidad de aprendizaje (optimizado: 0.01-0.2)
    max_depth=5,                # Profundidad máxima (optimizado: 3-10)
    subsample=0.8,              # Muestreo de filas para evitar overfitting
    colsample_bytree=0.8,       # Muestreo de columnas por árbol
    scale_pos_weight=1.86,      # Balanceo de clases (ratio 65/35)
    eval_metric='logloss',      # Métrica de evaluación
    random_state=42
)

# Construcción del pipeline completo
pipeline = Pipeline([
    ('imputer', imputer),
    ('scaler', scaler),
    ('model', model)
])

# --- 4. Entrenamiento del Modelo ---
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entrenar el pipeline con los datos de entrenamiento
pipeline.fit(X_train, y_train)

# --- 5. Guardado de Artefactos ---
# Crear el directorio 'models' si no existe
output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)

# Guardar el modelo completo (pipeline)
joblib.dump(pipeline.named_steps['model'], os.path.join(output_dir, 'model.pkl'))
# Guardar el imputer y el scaler por separado para usarlos en la API
joblib.dump(pipeline.named_steps['imputer'], os.path.join(output_dir, 'imputer.pkl'))
joblib.dump(pipeline.named_steps['scaler'], os.path.join(output_dir, 'scaler.pkl'))

print("✅ Modelo y preprocesadores entrenados y guardados en la carpeta 'models/'.")
print(f"Archivos guardados: model.pkl, imputer.pkl, scaler.pkl")
