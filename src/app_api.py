# -*- coding: utf-8 -*-
"""
AlertOpsLite - Backend FastAPI (Servico de Inteligencia Preditiva)
Challenge Locaweb - FIAP 2TSCOA (Grupo Irmaos)
Autores: Marcos Machado (RM566099) e Matheus Machado (RM564991)

O QUE ESTE ARQUIVO FAZ (visão para júnior):
  É a "porta de entrada" do sistema — uma API web. Outros programas (ou o navegador)
  fazem um pedido HTTP com os dados de um incidente, e a API responde com a previsão
  (volume futuro ou risco de OLA). Ela carrega os modelos .joblib UMA vez ao ligar e
  fica pronta para responder rápido.

  Endpoints (as "URLs" que a API atende):
    GET  /api/v1/health              -> a API está viva? (checagem de saúde)
    GET  /api/v1/metrics             -> métricas de validação dos modelos
    POST /api/v1/predict/volume      -> previsão de volume D+1 e D+7 (+ curva de 7 dias)
    POST /api/v1/predict/ola-risk    -> probabilidade e nível de risco de violação de OLA

Execucao (a partir da raiz do projeto):
  uvicorn src.app_api:app --reload           (uvicorn é o "servidor" que roda a API)
  ou:  python src/app_api.py
Docs interativas (Swagger): http://127.0.0.1:8000/docs   (página que testa a API no navegador)
"""

import os
import json
import sys
from contextlib import asynccontextmanager      # ajuda a rodar código no "ligar/desligar" da API
from typing import List, Optional, Literal       # anotações de tipo (List=lista, Optional=pode ser None, Literal=valores fixos)

import numpy as np
import pandas as pd
import joblib                                     # carrega os modelos salvos (.joblib)
from fastapi import FastAPI, HTTPException         # FastAPI = framework da API; HTTPException = erros HTTP (ex.: 404)
from fastapi.responses import FileResponse        # FileResponse = entrega arquivos HTML estáticos
from pydantic import BaseModel, Field             # Pydantic = valida os dados de entrada/saída automaticamente

# Adiciona a pasta deste arquivo ao "caminho de busca" do Python, para conseguir importar model_training.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reaproveita funções/constantes já definidas no treino (não reescreve a lógica de forecast).
from model_training import recursive_forecast, FEATURES_VOLUME, CAT_FEATURES_OLA, NUM_FEATURES_OLA

MODELS_DIR = "models"                                     # pasta dos modelos
METRICS_PATH = os.path.join("data", "metrics_report.json")  # relatório de métricas

STATE = {}  # "cache" na memória: guarda os modelos carregados para não reler do disco a cada pedido


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Roda no LIGAR e no DESLIGAR da API. Aqui carregamos os modelos uma única vez (startup)."""
    # --- startup (antes de atender qualquer pedido) ---
    STATE["model_d1"] = joblib.load(os.path.join(MODELS_DIR, "volume_d1_model.joblib"))   # modelo volume D+1
    STATE["model_d7"] = joblib.load(os.path.join(MODELS_DIR, "volume_d7_model.joblib"))   # modelo volume D+7
    STATE["model_ola"] = joblib.load(os.path.join(MODELS_DIR, "ola_risk_model.joblib"))   # modelo risco de OLA
    with open(os.path.join(MODELS_DIR, "metadata.json"), encoding="utf-8") as f:
        STATE["meta"] = json.load(f)                     # metadados (features, histórico, limiares)
    if os.path.exists(METRICS_PATH):                     # se o relatório de métricas existir...
        with open(METRICS_PATH, encoding="utf-8") as f:
            STATE["metrics"] = json.load(f)              # ...carrega para o endpoint /metrics
    else:
        STATE["metrics"] = {}
    print("[API] Modelos e metadados carregados com sucesso.")
    yield                                                # <- aqui a API fica "no ar" atendendo pedidos
    # --- shutdown (ao desligar) ---
    STATE.clear()                                        # limpa a memória


# Cria a aplicação. title/description/version aparecem na documentação Swagger. lifespan liga o startup acima.
app = FastAPI(
    title="AlertOpsLite API",
    description="Inteligencia Preditiva para Operacoes de TI - Challenge Locaweb (FIAP 2TSCOA / Grupo Irmaos)",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------- SCHEMAS --- #
# "Schemas" = contratos dos dados. O Pydantic valida automaticamente e gera a documentação.
# Se alguém mandar um dado no formato errado, a API já responde erro sem executar nada.

class HealthResponse(BaseModel):
    """Formato da resposta do /health."""
    status: str = Field(examples=["ok"])       # str = texto
    service: str = "AlertOpsLite API"          # valor padrão
    version: str = "1.0.0"
    models_loaded: bool                        # bool = Verdadeiro/Falso


class VolumeRequest(BaseModel):
    """Entrada do /predict/volume (opcional: pode mandar o próprio histórico)."""
    recent_history: Optional[List[float]] = Field(   # Optional = pode ser omitido (fica None)
        default=None,
        description="Serie dos ultimos totais diarios (>=14 valores, ordem cronologica). "
                    "Se omitido, usa o historico mais recente do treino.",
        examples=[[41, 38, 52, 47, 60, 12, 8, 55, 49, 61, 58, 44, 15, 9]],
    )


class ForecastPoint(BaseModel):
    """Um ponto da curva de previsão: uma data e o valor previsto."""
    date: str
    forecast: float                             # float = número com casas decimais


class VolumeResponse(BaseModel):
    """Formato da resposta do /predict/volume."""
    volume_d1: float = Field(description="Estimativa de incidentes elegiveis para o dia D+1")
    volume_d7: float = Field(description="Estimativa acumulada/pontual para o dia D+7")
    horizon: List[ForecastPoint] = Field(description="Curva de projecao D+1 a D+7")  # lista de pontos
    model: str = "RandomForestRegressor (validacao temporal)"


class OlaRiskRequest(BaseModel):
    """Entrada do /predict/ola-risk: os atributos de UM incidente na abertura."""
    # Field(...) documenta e VALIDA: ge=maior-ou-igual, le=menor-ou-igual. Ex.: hora entre 0 e 23.
    prioridade: str = Field(examples=["2 - Alta"], description="Prioridade do incidente (ex: '2 - Alta', '3 - Media')")
    produto: str = Field(examples=["lhco"], description="Produto associado")
    categoria: str = Field(examples=["cat71"], description="Categoria do incidente")
    grupo_designado: str = Field(examples=["Team11"], description="Equipe responsavel (Grupo designado)")
    hour: int = Field(ge=0, le=23, examples=[14], description="Hora de abertura (0-23)")
    day_of_week: int = Field(ge=0, le=6, examples=[0], description="Dia da semana (0=segunda ... 6=domingo)")
    month: int = Field(ge=1, le=12, examples=[8], description="Mes (1-12)")
    is_weekend: int = Field(ge=0, le=1, examples=[0], description="1 se fim de semana, 0 caso contrario")


class OlaRiskResponse(BaseModel):
    """Formato da resposta do /predict/ola-risk."""
    violation_probability: float = Field(description="Probabilidade estimada de violacao de OLA (0-1)")
    risk_level: Literal["Baixo", "Medio", "Alto"]   # Literal = só aceita um destes três textos
    recommendation: str


# --------------------------------------------------------------- ROTAS --- #
# Cada @app.get / @app.post é um "decorator" que liga a URL à função abaixo dele.
# response_model diz o formato da resposta; tags agrupam os endpoints na documentação.

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Infra"])
def health():
    """Checagem de saúde: responde 'ok' e diz se os modelos foram carregados."""
    return HealthResponse(status="ok", models_loaded=bool(STATE.get("model_d1")))


@app.get("/api/v1/metrics", tags=["Modelos"])
def metrics():
    """Devolve o relatório de métricas (R², AUC, matriz de confusão...)."""
    if not STATE.get("metrics"):                    # se não houver relatório carregado...
        # raise HTTPException = devolve um erro HTTP com código e mensagem (404 = não encontrado).
        raise HTTPException(status_code=404, detail="Relatorio de metricas indisponivel. Rode src/model_training.py.")
    return STATE["metrics"]


@app.post("/api/v1/predict/volume", response_model=VolumeResponse, tags=["Predicao"])
def predict_volume(req: VolumeRequest):
    """Prevê o volume dos próximos 7 dias. 'req' já vem validado pelo Pydantic."""
    meta = STATE["meta"]
    # Usa o histórico enviado; se não vier, usa o histórico recente salvo no treino.
    history = req.recent_history if req.recent_history else meta["history_recent"]
    if len(history) < 14:                           # precisa de pelo menos 14 dias (por causa do lag_14)
        raise HTTPException(status_code=422, detail="recent_history precisa de pelo menos 14 valores.")  # 422 = entrada inválida
    # Chama o forecast recursivo (a mesma função do treino) para gerar a curva de 7 dias.
    fc = recursive_forecast(STATE["model_d1"], history, meta["last_date"], n_days=7)
    return VolumeResponse(
        volume_d1=fc[0]["forecast"],                # 1º ponto = D+1
        volume_d7=fc[6]["forecast"],                # 7º ponto = D+7
        horizon=[ForecastPoint(**p) for p in fc],   # converte cada dict {date, forecast} em ForecastPoint
    )


@app.post("/api/v1/predict/ola-risk", response_model=OlaRiskResponse, tags=["Predicao"])
def predict_ola_risk(req: OlaRiskRequest):
    """Prevê a probabilidade de violação de OLA de um incidente e classifica o risco."""
    # Monta um DataFrame de 1 linha, na ORDEM de colunas que o modelo espera.
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

    # predict_proba retorna [prob_classe_0, prob_classe_1]; pegamos [0, 1] = prob de VIOLAR.
    proba = float(STATE["model_ola"].predict_proba(row)[0, 1])
    # Limiares "data-driven" (P70/P90) salvos no metadata; se faltarem, usa 0.05/0.20 como reserva.
    t_low, t_high = STATE["meta"].get("risk_thresholds", [0.05, 0.20])
    if proba < t_low:                               # abaixo do P70 -> Baixo
        level, rec = "Baixo", "Fluxo padrao de atendimento."
    elif proba < t_high:                            # entre P70 e P90 -> Médio
        level, rec = "Medio", "Priorizar triagem e acompanhar prazo de OLA."
    else:                                           # acima do P90 -> Alto
        level, rec = "Alto", "Acionar escalonamento preventivo para evitar violacao de OLA."
    return OlaRiskResponse(
        violation_probability=round(proba, 4),
        risk_level=level,
        recommendation=rec,
    )


@app.get("/dash", response_class=FileResponse, tags=["Dashboard"])
@app.get("/dashboard", response_class=FileResponse, tags=["Dashboard"], include_in_schema=False)
def dashboard():
    """Entrega o painel visual interativo HTML do AlertOpsLite."""
    for p in ["dashboard.html", os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard.html")]:
        if os.path.exists(p):
            return FileResponse(p, media_type="text/html")
    raise HTTPException(status_code=404, detail="Arquivo dashboard.html não encontrado no container.")


@app.get("/", tags=["Infra"])
def root():
    """Rota raiz: um 'mapa' rápido apontando para a documentação, o dashboard e o health."""
    return {
        "service": "AlertOpsLite API",
        "dashboard": "/dash",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Permite rodar com 'python src/app_api.py' (sobe o servidor uvicorn na porta 8000).
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app_api:app", host="127.0.0.1", port=8000, reload=False)
