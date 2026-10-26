# Dockerfile de reproducibilidad del entorno
# Sistema de Recomendacion Dual (BPE + FastText + CNN-1D)
# Imagen CPU; el entrenamiento con GPU se hace fuera del contenedor.

FROM python:3.11-slim

LABEL description="Entorno de reproducibilidad - Tesis sistema de recomendacion dual"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY README.md .

CMD ["python", "src/verificar_entorno.py"]
