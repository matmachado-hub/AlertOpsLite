# AlertOpsLite - API de Inteligencia Preditiva (Sprint 3 MVP)
# Imagem self-contained: treina os modelos no build e sobe a API FastAPI.
FROM python:3.12-slim

WORKDIR /app

# Dependencias Python (camada cacheavel)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo-fonte + dataset de entrada
COPY src/ ./src/
COPY data/LW-DATASET.xlsx ./data/LW-DATASET.xlsx

# Pipeline de dados + treino dos modelos (gera data/*.csv e models/*.joblib)
RUN python src/data_pipeline.py && python src/model_training.py

EXPOSE 8000

# Sobe a API (Swagger em http://localhost:8000/docs)
CMD ["uvicorn", "src.app_api:app", "--host", "0.0.0.0", "--port", "8000"]
