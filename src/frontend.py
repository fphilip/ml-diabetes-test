
import streamlit as st
import requests
import pandas as pd

# --- Datos de Pacientes Preset (del dataset) ---
# Colores semáforo: verde (#28a745), naranja (#fd7e14), rojo (#dc3545)
PRESET_PATIENTS = {
    "P10": {
        "pregnancies": 8, "glucose": 125, "blood_pressure": 96,
        "skin_thickness": 0, "insulin": 0, "bmi": 0.0,
        "dpf": 0.232, "age": 54, "color": "#5cb85c"  # Verde 70%
    },
    "P11": {
        "pregnancies": 4, "glucose": 110, "blood_pressure": 92,
        "skin_thickness": 0, "insulin": 0, "bmi": 37.6,
        "dpf": 0.191, "age": 30, "color": "#dc3545"  # Rojo
    },
    "P12": {
        "pregnancies": 10, "glucose": 168, "blood_pressure": 74,
        "skin_thickness": 0, "insulin": 0, "bmi": 38.0,
        "dpf": 0.537, "age": 34, "color": "#e85a1e"  # Naranja-rojo
    },
    "P13": {
        "pregnancies": 10, "glucose": 139, "blood_pressure": 80,
        "skin_thickness": 0, "insulin": 0, "bmi": 27.1,
        "dpf": 1.441, "age": 57, "color": "#28a745"  # Verde 100%
    },
}

# --- Inicializar Session State ---
default_values = {
    "pregnancies": 0, "glucose": 0, "blood_pressure": 0,
    "skin_thickness": 0, "insulin": 0, "bmi": 0.0,
    "dpf": 0.0, "age": 0
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predicción de Diabetes",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CSS Global personalizado ---
st.markdown("""
    <style>
    /* Ocultar botones +/- de number_input */
    [data-testid="stNumberInput"] button,
    [data-testid="stNumberInputContainer"] button,
    .stNumberInput button {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Barra Lateral (Sidebar) Institucional ---
with st.sidebar:
    st.title("TFM - Máster en Inteligencia Artificial")
    st.markdown("---")
    st.write("**Institución:** CEMP (Centro Europeo de Másteres y Posgrados)")
    st.write("**Alumno:** Francisco Philip")
    st.write("**Proyecto:** Clasificación de Enfermos Diabéticos")
    st.markdown("---")
    st.info("Esta aplicación utiliza un modelo de Machine Learning para predecir el riesgo de diabetes.")

# --- Título y Descripción Principal ---
st.title("🩺 Sistema de Predicción de Diabetes")
st.markdown("""
Esta herramienta permite al personal médico introducir los datos de un paciente para evaluar 
el riesgo de desarrollar diabetes. El modelo subyacente es **XGBoost** (`eXtreme Gradient Boosting`),
optimizado para minimizar falsos negativos.
""")

# --- Formulario de Entrada de Datos ---
st.header("Formulario de Datos del Paciente")

# --- Botones de Pacientes Preset con colores semáforo ---
# CSS para botones compactos con colores personalizados
st.markdown("""
    <style>
    .preset-container { margin-bottom: 0.5rem; }
    .preset-container p { margin-bottom: 0.3rem !important; font-size: 0.9rem; }
    </style>
    <div class="preset-container">
    <p><strong>Cargar paciente de ejemplo:</strong></p>
    </div>
""", unsafe_allow_html=True)

preset_cols = st.columns([1, 1, 1, 1, 1, 2])
for i, (name, data) in enumerate(PRESET_PATIENTS.items()):
    with preset_cols[i]:
        st.markdown(f"""
            <style>
            div[data-testid="stHorizontalBlock"] > div:nth-child({i+1}) button {{
                background-color: {data['color']} !important;
                color: white !important;
                font-weight: bold !important;
                border: none !important;
                padding: 0.2rem 0.6rem !important;
                min-height: 0 !important;
                line-height: 1.2 !important;
            }}
            div[data-testid="stHorizontalBlock"] > div:nth-child({i+1}) button:hover {{
                opacity: 0.85;
                filter: brightness(1.1);
            }}
            </style>
        """, unsafe_allow_html=True)
        if st.button(name, key=f"preset_{i}"):
            st.session_state.pregnancies = data["pregnancies"]
            st.session_state.glucose = data["glucose"]
            st.session_state.blood_pressure = data["blood_pressure"]
            st.session_state.skin_thickness = data["skin_thickness"]
            st.session_state.insulin = data["insulin"]
            st.session_state.bmi = data["bmi"]
            st.session_state.dpf = data["dpf"]
            st.session_state.age = data["age"]
            st.rerun()

# Botón Reset en la quinta columna
with preset_cols[4]:
    st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] > div:nth-child(5) button {
            background-color: #6c757d !important;
            color: white !important;
            font-weight: bold !important;
            border: none !important;
            padding: 0.2rem 0.6rem !important;
            min-height: 0 !important;
            line-height: 1.2 !important;
        }
        div[data-testid="stHorizontalBlock"] > div:nth-child(5) button:hover {
            opacity: 0.85;
            filter: brightness(1.1);
        }
        </style>
    """, unsafe_allow_html=True)
    if st.button("Reset", key="reset_btn"):
        st.session_state.pregnancies = default_values["pregnancies"]
        st.session_state.glucose = default_values["glucose"]
        st.session_state.blood_pressure = default_values["blood_pressure"]
        st.session_state.skin_thickness = default_values["skin_thickness"]
        st.session_state.insulin = default_values["insulin"]
        st.session_state.bmi = default_values["bmi"]
        st.session_state.dpf = default_values["dpf"]
        st.session_state.age = default_values["age"]
        st.rerun()

st.markdown("---")

# Crear columnas para una mejor disposición
col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input("Embarazos (Pregnancies)", min_value=0, max_value=20,
                                   step=1, key="pregnancies")
    glucose = st.number_input("Glucosa (Glucose)", min_value=0, max_value=250,
                               key="glucose")
    blood_pressure = st.number_input("Presión Arterial (BloodPressure)", min_value=0, max_value=150,
                                      key="blood_pressure")

with col2:
    skin_thickness = st.number_input("Grosor de Piel (SkinThickness)", min_value=0, max_value=100,
                                      key="skin_thickness")
    insulin = st.number_input("Insulina (Insulin)", min_value=0, max_value=900,
                               key="insulin")
    bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=0.0, max_value=70.0,
                           format="%.1f", key="bmi")

with col3:
    dpf = st.number_input("Función de Pedigrí de Diabetes (DPF)", min_value=0.0, max_value=2.5,
                           format="%.3f", key="dpf")
    age = st.number_input("Edad (Age)", min_value=0, max_value=120,
                           key="age")

# --- Opción para mostrar explicación LIME ---
show_lime = st.checkbox("Mostrar explicación LIME (Interpretabilidad del modelo)", value=True)

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
    # Usar /explain si se solicita LIME, sino /predict
    api_url = "http://api:8000/explain" if show_lime else "http://api:8000/predict"

    with st.spinner("Evaluando..."):
        try:
            response = requests.post(api_url, json=patient_data)
            response.raise_for_status()

            result = response.json()

            # --- Visualización del Resultado ---
            st.subheader("Resultado de la Evaluación")

            prediction = result.get("prediction")
            probability = result.get("probability")

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

            # --- Mostrar explicación LIME si está habilitado ---
            if show_lime and "lime_explanation" in result:
                st.subheader("Explicación LIME")
                st.markdown("""
                El siguiente gráfico muestra la contribución de cada característica a la predicción.
                - **Valores positivos (verde):** Aumentan la probabilidad de diabetes
                - **Valores negativos (rojo):** Disminuyen la probabilidad de diabetes
                """)

                # Crear DataFrame para visualización
                lime_data = result.get("lime_explanation", [])
                df_lime = pd.DataFrame(lime_data)

                if not df_lime.empty:
                    # Ordenar por contribución absoluta
                    df_lime = df_lime.sort_values(by="contribution", key=abs, ascending=True)

                    # Crear gráfico de barras horizontal con colores
                    colors = ['#d73027' if x < 0 else '#1a9850' for x in df_lime['contribution']]

                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(df_lime['feature'], df_lime['contribution'], color=colors)
                    ax.set_xlabel('Contribución a la predicción')
                    ax.set_title('Contribución de cada característica (LIME)')
                    ax.axvline(x=0, color='black', linewidth=0.5)
                    plt.tight_layout()

                    st.pyplot(fig)

                    # Mostrar tabla de datos detallados
                    st.markdown("**Datos detallados:**")
                    df_display = df_lime.copy()
                    df_display = df_display.sort_values(by="contribution", key=abs, ascending=False)
                    df_display.columns = ['Característica', 'Contribución']
                    st.dataframe(df_display, use_container_width=True, hide_index=True)

        except requests.exceptions.RequestException as e:
            st.error(f"Error de conexión con la API: {e}")
            st.warning("Asegúrate de que el servicio de la API esté en funcionamiento. Si usas Docker, verifica que el contenedor 'api' esté corriendo.")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado: {e}")
