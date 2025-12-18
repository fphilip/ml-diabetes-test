
# Usar una imagen base de Python delgada y moderna
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requerimientos primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY ./src /app/src
COPY ./models /app/models

# Exponer los puertos que usarán las aplicaciones
# 8000 para la API (FastAPI) y 8501 para el Frontend (Streamlit)
EXPOSE 8000
EXPOSE 8501

# Nota: El comando para ejecutar la aplicación se especificará en docker-compose.yml
# Esto permite que la misma imagen se use para múltiples servicios (api, frontend).
