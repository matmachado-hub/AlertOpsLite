# -*- coding: utf-8 -*-
"""
AlertOpsLite - Backend FastAPI (Servico de Inteligencia Preditiva)
Challenge Locaweb - FIAP 2TSCOA (Grupo Irmaos)
Autores: Marcos Machado (RM566099) e Matheus Machado (RM564991)

API REST modular com validacao Pydantic v2, servindo os modelos treinados:
  GET  /api/v1/health              -> verificacao de integridade
  GET  /api/v1/metrics             -> metricas de acuracia/validacao dos modelos
  POST /api/v1/predict/volume      -> estimativa de volume D+1 e D+7 (+ curva 7 dias)
  POST /api/v1/predict/ola-risk    -> probabilidade e nivel de risco de violacao de OLA

Execucao (a partir da raiz do projeto):
  uvicorn src.app_api:app --reload
  ou:  python src/app_api.py
Docs interativas (Swagger): http://127.0.0.1:8000/docs
"""

import os
import json
import sys
from contextlib import asynccontextmanager
from typing import List, Optional, Literal

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_training import recursive_forecast, FEATURES_VOLUME, CAT_FEATURES_OLA, NUM_FEATURES_OLA

MODELS_DIR = "models"
METRICS_PATH = os.path.join("data", "metrics_report.json")

STATE = {}  # cache de modelos/metadata carregados no startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: carrega modelos e metadados uma unica vez
    STATE["model_d1"] = joblib.load(os.path.join(MODELS_DIR, "volume_d1_model.joblib"))
    STATE["model_d7"] = joblib.load(os.path.join(MODELS_DIR, "volume_d7_model.joblib"))
    STATE["model_ola"] = joblib.load(os.path.join(MODELS_DIR, "ola_risk_model.joblib"))
    with open(os.path.join(MODELS_DIR, "metadata.json"), encoding="utf-8") as f:
        STATE["meta"] = json.load(f)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as f:
            STATE["metrics"] = json.load(f)
    else:
        STATE["metrics"] = {}
    print("[API] Modelos e metadados carregados com sucesso.")
    yield
    STATE.clear()


app = FastAPI(
    title="AlertOpsLite API",
    description="Inteligencia Preditiva para Operacoes de TI - Challenge Locaweb (FIAP 2TSCOA / Grupo Irmaos)",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------- SCHEMAS --- #
class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    service: str = "AlertOpsLite API"
    version: str = "1.0.0"
    models_loaded: bool


class VolumeRequest(BaseModel):
    recent_history: Optional[List[float]] = Field(
        default=None,
        description="Serie dos ultimos totais diarios (>=14 valores, ordem cronologica). "
                    "Se omitido, usa o historico mais recente do treino.",
        examples=[[41, 38, 52, 47, 60, 12, 8, 55, 49, 61, 58, 44, 15, 9]],
    )


class ForecastPoint(BaseModel):
    date: str
    forecast: float


class VolumeResponse(BaseModel):
    volume_d1: float = Field(description="Estimativa de incidentes elegiveis para o dia D+1")
    volume_d7: float = Field(description="Estimativa acumulada/pontual para o dia D+7")
    horizon: List[ForecastPoint] = Field(description="Curva de projecao D+1 a D+7")
    model: str = "RandomForestRegressor (validacao temporal)"


class OlaRiskRequest(BaseModel):
    prioridade: str = Field(examples=["2 - Alta"], description="Prioridade do incidente (ex: '2 - Alta', '3 - Media')")
    produto: str = Field(examples=["lhco"], description="Produto associado")
    categoria: str = Field(examples=["cat71"], description="Categoria do incidente")
    grupo_designado: str = Field(examples=["Team11"], description="Equipe responsavel (Grupo designado)")
    hour: int = Field(ge=0, le=23, examples=[14], description="Hora de abertura (0-23)")
    day_of_week: int = Field(ge=0, le=6, examples=[0], description="Dia da semana (0=segunda ... 6=domingo)")
    month: int = Field(ge=1, le=12, examples=[8], description="Mes (1-12)")
    is_weekend: int = Field(ge=0, le=1, examples=[0], description="1 se fim de semana, 0 caso contrario")


class OlaRiskResponse(BaseModel):
    violation_probability: float = Field(description="Probabilidade estimada de violacao de OLA (0-1)")
    risk_level: Literal["Baixo", "Medio", "Alto"]
    recommendation: str


# --------------------------------------------------------------- ROTAS --- #
@app.get("/api/v1/health", response_model=HealthResponse, tags=["Infra"])
def health():
    return HealthResponse(status="ok", models_loaded=bool(STATE.get("model_d1")))


@app.get("/api/v1/metrics", tags=["Modelos"])
def metrics():
    if not STATE.get("metrics"):
        raise HTTPException(status_code=404, detail="Relatorio de metricas indisponivel. Rode src/model_training.py.")
    return STATE["metrics"]


@app.post("/api/v1/predict/volume", response_model=VolumeResponse, tags=["Predicao"])
def predict_volume(req: VolumeRequest):
    meta = STATE["meta"]
    history = req.recent_history if req.recent_history else meta["history_recent"]
    if len(history) < 14:
        raise HTTPException(status_code=422, detail="recent_history precisa de pelo menos 14 valores.")
    fc = recursive_forecast(STATE["model_d1"], history, meta["last_date"], n_days=7)
    return VolumeResponse(
        volume_d1=fc[0]["forecast"],
        volume_d7=fc[6]["forecast"],
        horizon=[ForecastPoint(**p) for p in fc],
    )


@app.post("/api/v1/predict/ola-risk", response_model=OlaRiskResponse, tags=["Predicao"])
def predict_ola_risk(req: OlaRiskRequest):
    row = pd.DataFrame([{
        "Prioridade": req.prioridade,
        "Produto": req.produto,
        "Categoria": req.categoria,
        "Grupo designado": req.grupo_designado,
        "hour": req.hour,
        "day_of_week": req.day_of_week,
        "month": req.month,
        "is_weekend": req.is_weekend,
    }])[CAT_FEATURES_OLA + NUM_FEATURES_OLA]

    proba = float(STATE["model_ola"].predict_proba(row)[0, 1])
    # Limiares data-driven (P70/P90 da distribuicao de risco); fallback fixo se ausente
    t_low, t_high = STATE["meta"].get("risk_thresholds", [0.05, 0.20])
    if proba < t_low:
        level, rec = "Baixo", "Fluxo padrao de atendimento."
    elif proba < t_high:
        level, rec = "Medio", "Priorizar triagem e acompanhar prazo de OLA."
    else:
        level, rec = "Alto", "Acionar escalonamento preventivo para evitar violacao de OLA."
    return OlaRiskResponse(
        violation_probability=round(proba, 4),
        risk_level=level,
        recommendation=rec,
    )


@app.get("/", tags=["Infra"])
def root():
    return {"service": "AlertOpsLite API", "docs": "/docs", "health": "/api/v1/health"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app_api:app", host="127.0.0.1", port=8000, reload=False)
