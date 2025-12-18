
import streamlit as st
import requests

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predicción de Diabetes",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Barra Lateral (Sidebar) Institucional ---
with st.sidebar:
    st.title("TFM - Máster en Inteligencia Artificial")
    st.markdown("---")
    st.write("**Institución:** CEMP (Centro Europeo de Másteres y Posgrados)")
    st.write("**Alumno:** [TU NOMBRE AQUÍ]")
    st.write("**Director:** Cristian Rodríguez")
    st.write("**Proyecto:** Clasificación de Enfermos Diabéticos")
    st.markdown("---")
    st.info("Esta aplicación utiliza un modelo de Machine Learning para predecir el riesgo de diabetes.")

# --- Título y Descripción Principal ---
st.title("🩺 Sistema de Predicción de Diabetes")
st.markdown("""
Esta herramienta permite al personal médico introducir los datos de un paciente para evaluar 
el riesgo de desarrollar diabetes. El modelo subyacente es un `Gradient Boosting Classifier`
optimizado para minimizar falsos negativos.
""")

# --- Formulario de Entrada de Datos ---
st.header("Formulario de Datos del Paciente")

# Crear columnas para una mejor disposición
col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input("Embarazos (Pregnancies)", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucosa (Glucose)", min_value=0, max_value=250, value=120)
    blood_pressure = st.number_input("Presión Arterial (BloodPressure)", min_value=0, max_value=150, value=70)

with col2:
    skin_thickness = st.number_input("Grosor de Piel (SkinThickness)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulina (Insulin)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=0.0, max_value=70.0, value=32.0, format="%.1f")

with col3:
    dpf = st.number_input("Función de Pedigrí de Diabetes (DPF)", min_value=0.0, max_value=2.5, value=0.47, format="%.3f")
    age = st.number_input("Edad (Age)", min_value=1, max_value=120, value=30)

# --- Botón de Predicción y Lógica de la API ---
if st.button("Evaluar Paciente", type="primary"):
    # Crear el payload para la API
    patient_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    # URL de la API (usando el nombre del servicio de Docker Compose)
    api_url = "http://api:8000/predict"

    with st.spinner("Evaluando..."):
        try:
            response = requests.post(api_url, json=patient_data)
            response.raise_for_status()  # Lanza un error para respuestas 4xx/5xx

            result = response.json()
            
            # --- Visualización del Resultado ---
            st.subheader("Resultado de la Evaluación")
            
            prediction = result.get("prediction")
            probability = result.get("probability")
            risk_level = result.get("risk_level")

            if prediction == "Diabetic":
                st.error(f"**Predicción:** {prediction} (Riesgo Alto)")
                st.write(f"**Probabilidad de Diabetes:** {probability:.2%}")
                st.warning("""
                **Recomendación:** El modelo ha clasificado al paciente en la categoría de 'Alto Riesgo'. 
                Se recomienda realizar pruebas confirmatorias y seguimiento médico.
                """)
            else:
                st.success(f"**Predicción:** {prediction} (Riesgo Bajo)")
                st.write(f"**Probabilidad de Diabetes:** {probability:.2%}")
                st.info("""
                **Recomendación:** El modelo ha clasificado al paciente en la categoría de 'Bajo Riesgo'. 
                Se aconseja mantener un estilo de vida saludable y controles periódicos.
                """)

        except requests.exceptions.RequestException as e:
            st.error(f"Error de conexión con la API: {e}")
            st.warning("Asegúrate de que el servicio de la API esté en funcionamiento. Si usas Docker, verifica que el contenedor 'api' esté corriendo.")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado: {e}")
