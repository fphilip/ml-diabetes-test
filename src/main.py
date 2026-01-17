
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

# --- 1. Carga de Modelos y Preprocesadores ---
# Cargar los artefactos guardados durante el entrenamiento.
# Es crucial usar try-except para manejar errores si los archivos no se encuentran.
try:
    model = joblib.load("models/model.pkl")
    imputer = joblib.load("models/imputer.pkl")
    scaler = joblib.load("models/scaler.pkl")
    # Cargar datos para LIME
    X_train_scaled = joblib.load("models/X_train_scaled.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
except FileNotFoundError:
    raise RuntimeError("Model files not found. Please run train_model.py first.")
except Exception as e:
    raise RuntimeError(f"Error loading model files: {e}")

# --- 1.1 Configurar LIME Explainer ---
lime_explainer = LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=feature_names,
    class_names=['Healthy', 'Diabetic'],
    mode='classification'
)

# --- 2. Definición de la Aplicación FastAPI ---
app = FastAPI(
    title="API de Predicción de Diabetes",
    description="Una API para predecir el riesgo de diabetes basado en datos clínicos.",
    version="1.0.0"
)

# --- 3. Definición del Modelo de Datos de Entrada (Pydantic) ---
# Esto asegura que los datos de entrada tengan el formato correcto y realiza validación automática.
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

    class Config:
        schema_extra = {
            "example": {
                "Pregnancies": 6,
                "Glucose": 148,
                "BloodPressure": 72,
                "SkinThickness": 35,
                "Insulin": 0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50
            }
        }

# --- 4. Endpoint de Predicción ---
@app.post("/predict")
def predict(data: PatientData):
    """
    Endpoint para realizar predicciones de diabetes.

    Recibe un JSON con los datos del paciente y devuelve la predicción,
    la probabilidad y el nivel de riesgo.
    """
    try:
        # Convertir los datos de entrada a un DataFrame de Pandas
        input_data = pd.DataFrame([data.dict()])
        
        # Reemplazar ceros por NaN en variables biológicas, igual que en el entrenamiento
        cols_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        input_data[cols_to_replace] = input_data[cols_to_replace].replace(0, np.nan)

        # --- Aplicar el mismo preprocesamiento que en el entrenamiento ---
        # 1. Imputación de valores nulos
        imputed_data = imputer.transform(input_data)

        # 2. Escalado de características
        scaled_data = scaler.transform(imputed_data)
        
        # --- Realizar la Predicción ---
        # Usar predict_proba para obtener la probabilidad de la clase positiva (diabetes)
        probability = model.predict_proba(scaled_data)[:, 1][0]

        # Aplicar el umbral de decisión personalizado de 0.18
        prediction_value = 1 if probability >= 0.18 else 0
        prediction_label = "Diabetic" if prediction_value == 1 else "Healthy"
        
        # Determinar el nivel de riesgo
        risk_level = "High" if prediction_label == "Diabetic" else "Low"

        # --- Devolver la Respuesta ---
        return {
            "prediction": prediction_label,
            "probability": float(probability),
            "risk_level": risk_level
        }

    except Exception as e:
        # Capturar cualquier excepción y devolver un error HTTP 500
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. Endpoint de Explicación con LIME ---
@app.post("/explain")
def explain(data: PatientData):
    """
    Endpoint para obtener explicaciones LIME de una predicción.

    Recibe un JSON con los datos del paciente y devuelve la predicción
    junto con las contribuciones de cada característica según LIME.
    """
    try:
        # Convertir los datos de entrada a un DataFrame de Pandas
        input_data = pd.DataFrame([data.dict()])

        # Reemplazar ceros por NaN en variables biológicas
        cols_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        input_data[cols_to_replace] = input_data[cols_to_replace].replace(0, np.nan)

        # Aplicar preprocesamiento
        imputed_data = imputer.transform(input_data)
        scaled_data = scaler.transform(imputed_data)

        # Obtener predicción
        probability = model.predict_proba(scaled_data)[:, 1][0]
        prediction_value = 1 if probability >= 0.18 else 0
        prediction_label = "Diabetic" if prediction_value == 1 else "Healthy"

        # Generar explicación LIME
        explanation = lime_explainer.explain_instance(
            data_row=scaled_data[0],
            predict_fn=model.predict_proba,
            num_features=len(feature_names)
        )

        # Obtener contribuciones como lista de diccionarios
        lime_contributions = []
        for feature, contribution in explanation.as_list():
            lime_contributions.append({
                "feature": feature,
                "contribution": round(contribution, 4)
            })

        # Ordenar por valor absoluto de contribución (mayor impacto primero)
        lime_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "prediction": prediction_label,
            "probability": float(probability),
            "lime_explanation": lime_contributions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 6. Endpoint Raíz (Opcional) ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API. Go to /docs for documentation."}
