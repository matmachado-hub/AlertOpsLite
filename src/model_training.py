# -*- coding: utf-8 -*-
"""
AlertOpsLite - Modelagem Preditiva (Treino, Validacao e Serializacao)
Challenge Locaweb - FIAP 2TSCOA (Grupo Irmaos)
Autores: Marcos Machado (RM566099) e Matheus Machado (RM564991)
Fundamentacao: Capitulos 10 (CRISP-DM), 11 (Regressao Linear) e 12 (Arvores de Decisao)

Gera dois blocos de modelos:
  1. Regressao de Volume Diario (D+1 e D+7) -> LinearRegression / Ridge / RandomForest
     Validacao TEMPORAL (holdout dos ultimos 20% dias, sem embaralhar).
  2. Classificacao de Risco de Violacao de OLA -> DecisionTree / RandomForest
     class_weight balanceado para mitigar ~1% de eventos positivos.

Artefatos salvos:
  models/volume_d1_model.joblib, models/volume_d7_model.joblib, models/ola_risk_model.joblib
  models/metadata.json   -> features, historico recente p/ forecast, sumario do dataset
  data/metrics_report.json -> metricas completas p/ o dashboard e a apresentacao
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score,
)

RANDOM_STATE = 42

DAILY_PATH = os.path.join("data", "daily_incidents_processed.csv")
KPI_PATH = os.path.join("data", "kpi_incidents_processed.csv")
MODELS_DIR = "models"
METRICS_PATH = os.path.join("data", "metrics_report.json")

# Features da serie temporal (Cap. 11) - todas conhecidas em D para prever D+1/D+7
FEATURES_VOLUME = [
    "day_of_week", "day_of_month", "month", "is_weekend",
    "lag_1", "lag_2", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
]

# Features do classificador de OLA (conhecidas na ABERTURA do incidente - sem vazamento)
CAT_FEATURES_OLA = ["Prioridade", "Produto", "Categoria", "Grupo designado"]
NUM_FEATURES_OLA = ["hour", "day_of_week", "month", "is_weekend"]


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# --------------------------------------------------------------------------- #
# 1. REGRESSAO DE VOLUME (Cap. 10 e 11)
# --------------------------------------------------------------------------- #
def train_volume_regressor(daily, target_col, label):
    """Treina 3 regressores com validacao temporal e retorna (melhor_modelo, metricas)."""
    print(f"\n[MODELO] Regressao de Volume -> {label} (target='{target_col}')")
    data = daily.dropna(subset=[target_col]).reset_index(drop=True)

    X = data[FEATURES_VOLUME].values
    y = data[target_col].values

    # Split TEMPORAL: ultimos 20% dos dias como teste (sem embaralhar) - Cap. 10 (validacao temporal)
    split = int(len(data) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"         treino={len(X_train)} dias | teste={len(X_test)} dias")

    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    results = {}
    best_name, best_model, best_r2 = None, None, -np.inf
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = float(r2_score(y_test, pred))
        mae = float(mean_absolute_error(y_test, pred))
        rmse = _rmse(y_test, pred)
        results[name] = {"r2": round(r2, 4), "mae": round(mae, 3), "rmse": round(rmse, 3)}
        print(f"         {name:16s} | R2={r2:6.3f} | MAE={mae:6.2f} | RMSE={rmse:6.2f}")
        if r2 > best_r2:
            best_name, best_model, best_r2 = name, model, r2

    print(f"         => Melhor modelo: {best_name} (R2={best_r2:.3f})")

    # Residuo do melhor modelo no teste -> usado para bandas de confianca no dashboard
    resid_std = float(np.std(y_test - best_model.predict(X_test)))

    metrics = {
        "target": target_col,
        "label": label,
        "best_model": best_name,
        "resid_std": round(resid_std, 3),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "candidates": results,
    }
    return best_model, metrics


# --------------------------------------------------------------------------- #
# 2. CLASSIFICACAO DE RISCO DE OLA (Cap. 12)
# --------------------------------------------------------------------------- #
def _build_ola_pipeline(estimator):
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CAT_FEATURES_OLA),
        ],
        remainder="passthrough",  # features numericas passam direto
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def train_ola_classifier(kpi):
    """Treina classificadores de violacao de OLA (dados desbalanceados) e retorna (melhor, metricas)."""
    print("\n[MODELO] Classificacao de Risco de Violacao de OLA")
    df = kpi.copy()
    for c in CAT_FEATURES_OLA:
        df[c] = df[c].fillna("Desconhecido").astype(str)

    X = df[CAT_FEATURES_OLA + NUM_FEATURES_OLA]
    y = df["ola_violada"].astype(int).values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"         treino={len(X_train)} | teste={len(X_test)} | positivos_teste={int(y_test.sum())}")

    candidates = {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    results, best_name, best_pipe, best_f1 = {}, None, None, -np.inf
    best_cm = None
    for name, est in candidates.items():
        pipe = _build_ola_pipeline(est)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, pred)
        prec = float(precision_score(y_test, pred, zero_division=0))
        rec = float(recall_score(y_test, pred, zero_division=0))
        f1 = float(f1_score(y_test, pred, zero_division=0))
        auc = float(roc_auc_score(y_test, proba))
        results[name] = {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "roc_auc": round(auc, 4),
            "confusion_matrix": cm.tolist(),
        }
        print(f"         {name:14s} | Prec={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f} | AUC={auc:.3f}")
        if f1 > best_f1:
            best_name, best_pipe, best_f1, best_cm = name, pipe, f1, cm

    print(f"         => Melhor modelo: {best_name} (F1={best_f1:.3f})")

    metrics = {
        "best_model": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positives_total": int(y.sum()),
        "positives_rate": round(float(y.mean()), 5),
        "best_confusion_matrix": best_cm.tolist(),
        "candidates": results,
    }
    return best_pipe, metrics


# --------------------------------------------------------------------------- #
# Forecast recursivo D+1..D+n (reaproveitado pelo dashboard e pela API)
# --------------------------------------------------------------------------- #
def recursive_forecast(model, history, last_date, n_days=7):
    """
    Projeta os proximos n_days aplicando o modelo D+1 recursivamente.
    history: lista dos ultimos totais diarios (>= 14 valores, ordem cronologica).
    Retorna lista de dicts {date, forecast}.
    """
    hist = list(history)
    last = pd.Timestamp(last_date)
    out = []
    for step in range(1, n_days + 1):
        d = last + pd.Timedelta(days=step)
        feat = {
            "day_of_week": d.dayofweek,
            "day_of_month": d.day,
            "month": d.month,
            "is_weekend": 1 if d.dayofweek >= 5 else 0,
            "lag_1": hist[-1],
            "lag_2": hist[-2],
            "lag_7": hist[-7],
            "lag_14": hist[-14],
            "rolling_mean_7": float(np.mean(hist[-7:])),
            "rolling_mean_14": float(np.mean(hist[-14:])),
            "rolling_std_7": float(np.std(hist[-7:])),
        }
        row = np.array([[feat[f] for f in FEATURES_VOLUME]])
        yhat = float(model.predict(row)[0])
        yhat = max(0.0, yhat)  # volume nao pode ser negativo
        out.append({"date": d.strftime("%Y-%m-%d"), "forecast": round(yhat, 1)})
        hist.append(yhat)
    return out


# --------------------------------------------------------------------------- #
# Orquestracao
# --------------------------------------------------------------------------- #
def run_training():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    kpi = pd.read_csv(KPI_PATH)

    # 1) Regressores de volume
    model_d1, metrics_d1 = train_volume_regressor(daily, "target_d1", "Volume D+1")
    model_d7, metrics_d7 = train_volume_regressor(daily, "target_d7", "Volume D+7")

    # 2) Classificador de OLA
    ola_model, metrics_ola = train_ola_classifier(kpi)

    # Serializacao dos modelos
    joblib.dump(model_d1, os.path.join(MODELS_DIR, "volume_d1_model.joblib"))
    joblib.dump(model_d7, os.path.join(MODELS_DIR, "volume_d7_model.joblib"))
    joblib.dump(ola_model, os.path.join(MODELS_DIR, "ola_risk_model.joblib"))

    # Historico recente para o forecast recursivo (ultimos 21 dias)
    recent = daily.sort_values("date").tail(21)
    history = recent["total_incidentes"].tolist()
    last_date = recent["date"].max().strftime("%Y-%m-%d")

    # Limiares de risco data-driven: o classificador balanceado infla as probabilidades,
    # entao Baixo/Medio/Alto sao definidos por PERCENTIS da distribuicao real de risco (P70/P90).
    X_all = kpi.copy()
    for c in CAT_FEATURES_OLA:
        X_all[c] = X_all[c].fillna("Desconhecido").astype(str)
    proba_all = ola_model.predict_proba(X_all[CAT_FEATURES_OLA + NUM_FEATURES_OLA])[:, 1]
    q70, q90 = np.quantile(proba_all, [0.70, 0.90])
    risk_thresholds = [round(float(q70), 4), round(float(q90), 4)]
    print(f"[MODELO] Limiares de risco (P70/P90): {risk_thresholds}")

    metadata = {
        "features_volume": FEATURES_VOLUME,
        "cat_features_ola": CAT_FEATURES_OLA,
        "num_features_ola": NUM_FEATURES_OLA,
        "history_recent": history,
        "last_date": last_date,
        "eligible_count": int(len(kpi)),
        "raw_count": 122543,
        "violation_rate": metrics_ola["positives_rate"],
        "risk_thresholds": risk_thresholds,
        "date_range": [daily["date"].min().strftime("%Y-%m-%d"),
                       daily["date"].max().strftime("%Y-%m-%d")],
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Predicoes de vitrine (KPIs do painel) via forecast recursivo
    fc = recursive_forecast(model_d1, history, last_date, n_days=7)
    kpi_d1 = fc[0]["forecast"]
    kpi_d7 = fc[6]["forecast"]

    report = {
        "volume_d1": metrics_d1,
        "volume_d7": metrics_d7,
        "ola_risk": metrics_ola,
        "forecast_next_7": fc,
        "kpi_cards": {
            "volume_d1": kpi_d1,
            "volume_d7": kpi_d7,
            "ola_risk_index_pct": round(metrics_ola["positives_rate"] * 100, 2),
            "eligible_incidents": int(len(kpi)),
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[MODELO] Modelos serializados em '{MODELS_DIR}/' e metricas em '{METRICS_PATH}'.")
    print(f"[MODELO] KPI Previsto -> D+1={kpi_d1} | D+7={kpi_d7} | Risco OLA={report['kpi_cards']['ola_risk_index_pct']}%")
    return report


if __name__ == "__main__":
    run_training()
