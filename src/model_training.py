# -*- coding: utf-8 -*-
"""
AlertOpsLite - Modelagem Preditiva (Treino, Validacao e Serializacao)
Challenge Locaweb - FIAP 2TSCOA (Grupo Irmaos)
Autores: Marcos Machado (RM566099) e Matheus Machado (RM564991)
Fundamentacao: Capitulos 10 (CRISP-DM), 11 (Regressao Linear) e 12 (Arvores de Decisao)

O QUE ESTE ARQUIVO FAZ (visão para júnior):
  Ensina 2 tipos de "modelo" (um programa que aprende padrões nos dados):
  1) REGRESSÃO: prevê um NÚMERO — quantos incidentes teremos amanhã (D+1) e em 7 dias (D+7).
  2) CLASSIFICAÇÃO: prevê SIM/NÃO — este incidente tem risco de estourar a OLA?
  Depois salva os modelos treinados em arquivos .joblib (para a API usar) e um
  relatório de métricas em JSON (para o dashboard e os slides).
"""

import os          # caminhos e pastas
import json        # ler/escrever arquivos .json
import numpy as np # matemática com vetores
import pandas as pd# tabelas (DataFrame)
import joblib      # salvar/carregar modelos treinados em disco (serialização)

# --- Importes do scikit-learn (biblioteca de Machine Learning) ---
from sklearn.linear_model import LinearRegression, Ridge                 # modelos de regressão lineares
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # "floresta" de árvores (regressão e classificação)
from sklearn.tree import DecisionTreeClassifier                          # árvore de decisão única
from sklearn.pipeline import Pipeline                                    # encadeia pré-processamento + modelo num objeto só
from sklearn.compose import ColumnTransformer                           # aplica transformações diferentes por grupo de colunas
from sklearn.preprocessing import OneHotEncoder                          # transforma texto (categorias) em colunas de 0/1
from sklearn.metrics import (                                            # "réguas" para medir a qualidade dos modelos
    r2_score, mean_absolute_error, mean_squared_error,                   # métricas de regressão
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score,  # métricas de classificação
)

# Semente aleatória fixa: garante que rodar de novo dá o MESMO resultado (reprodutibilidade).
RANDOM_STATE = 42

# --- Caminhos de entrada/saída ---
DAILY_PATH = os.path.join("data", "daily_incidents_processed.csv")   # série diária (para regressão)
KPI_PATH = os.path.join("data", "kpi_incidents_processed.csv")       # incidentes (para classificação)
MODELS_DIR = "models"                                                # pasta onde os modelos são salvos
METRICS_PATH = os.path.join("data", "metrics_report.json")          # relatório de métricas

# Colunas de ENTRADA da regressão (tudo conhecido HOJE para prever o futuro).
FEATURES_VOLUME = [
    "day_of_week", "day_of_month", "month", "is_weekend",   # calendário
    "lag_1", "lag_2", "lag_7", "lag_14",                    # valores de dias passados
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",   # tendência recente
]

# Colunas de ENTRADA do classificador de OLA (conhecidas na ABERTURA do incidente).
# Importante: só usamos o que se sabe NO MOMENTO da abertura -> evita "vazamento" (trapaça).
CAT_FEATURES_OLA = ["Prioridade", "Produto", "Categoria", "Grupo designado"]  # categóricas (texto)
NUM_FEATURES_OLA = ["hour", "day_of_week", "month", "is_weekend"]             # numéricas


def _rmse(y_true, y_pred):
    """RMSE = raiz do erro quadrático médio. Erro médio de previsão, na mesma unidade (incidentes)."""
    # mean_squared_error dá o erro ao quadrado; np.sqrt tira a raiz para voltar à unidade original.
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# --------------------------------------------------------------------------- #
# 1. REGRESSAO DE VOLUME (Cap. 10 e 11) — prever um NÚMERO
# --------------------------------------------------------------------------- #
def train_volume_regressor(daily, target_col, label):
    """Treina 3 regressores com validacao temporal e retorna (melhor_modelo, metricas)."""
    print(f"\n[MODELO] Regressao de Volume -> {label} (target='{target_col}')")
    # Remove linhas sem "resposta certa" (target vazio nos últimos dias, por causa do shift negativo).
    data = daily.dropna(subset=[target_col]).reset_index(drop=True)

    X = data[FEATURES_VOLUME].values   # X = ENTRADAS (features). .values converte para matriz numpy.
    y = data[target_col].values        # y = SAÍDA que queremos prever (o alvo)

    # --- Divisão TEMPORAL treino/teste (Cap. 10) ---
    # Em série temporal NÃO se embaralha: treina no passado e testa no futuro (como na vida real).
    split = int(len(data) * 0.80)           # ponto de corte: 80% para treino
    X_train, X_test = X[:split], X[split:]  # treino = primeiros 80% dos dias; teste = últimos 20%
    y_train, y_test = y[:split], y[split:]
    print(f"         treino={len(X_train)} dias | teste={len(X_test)} dias")

    # --- 3 modelos candidatos (vamos comparar e ficar com o melhor) ---
    candidates = {
        "LinearRegression": LinearRegression(),                 # reta simples (Cap. 11)
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),   # regressão linear com "freio" (regularização)
        "RandomForest": RandomForestRegressor(                  # floresta: combina muitas árvores (captura não-linearidades)
            n_estimators=300,     # 300 árvores
            max_depth=12,         # profundidade máxima de cada árvore (evita decorar demais = overfitting)
            min_samples_leaf=3,   # cada "folha" precisa de >= 3 exemplos
            random_state=RANDOM_STATE,
            n_jobs=-1,            # usa todos os núcleos da CPU (mais rápido)
        ),
    }

    results = {}                                    # guarda as métricas de cada modelo
    best_name, best_model, best_r2 = None, None, -np.inf  # controle do melhor (começa em -infinito)
    for name, model in candidates.items():          # percorre cada candidato
        model.fit(X_train, y_train)                 # .fit = TREINAR (o modelo aprende com treino)
        pred = model.predict(X_test)                # .predict = PREVER no teste (dados nunca vistos)
        r2 = float(r2_score(y_test, pred))          # R²: 0 a 1, quanto maior melhor (quão bem explica a variação)
        mae = float(mean_absolute_error(y_test, pred))  # MAE: erro médio absoluto (em incidentes/dia)
        rmse = _rmse(y_test, pred)                  # RMSE: erro que penaliza mais os erros grandes
        results[name] = {"r2": round(r2, 4), "mae": round(mae, 3), "rmse": round(rmse, 3)}
        print(f"         {name:16s} | R2={r2:6.3f} | MAE={mae:6.2f} | RMSE={rmse:6.2f}")
        if r2 > best_r2:                            # se este modelo é melhor (R² maior)...
            best_name, best_model, best_r2 = name, model, r2  # ...guarda como o novo melhor

    print(f"         => Melhor modelo: {best_name} (R2={best_r2:.3f})")

    # Desvio-padrão dos "resíduos" (erros) no teste -> vira a largura da banda de incerteza no dashboard.
    resid_std = float(np.std(y_test - best_model.predict(X_test)))

    metrics = {                        # dicionário com o resumo desta regressão
        "target": target_col,
        "label": label,
        "best_model": best_name,
        "resid_std": round(resid_std, 3),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "candidates": results,         # métricas de TODOS os candidatos (para os slides)
    }
    return best_model, metrics         # devolve o melhor modelo + as métricas


# --------------------------------------------------------------------------- #
# 2. CLASSIFICACAO DE RISCO DE OLA (Cap. 12) — prever SIM/NÃO
# --------------------------------------------------------------------------- #
def _build_ola_pipeline(estimator):
    """Monta um 'Pipeline': pré-processa as colunas e depois aplica o modelo, num objeto só."""
    # ColumnTransformer aplica transformações diferentes por tipo de coluna.
    pre = ColumnTransformer(
        transformers=[
            # OneHotEncoder transforma texto em colunas 0/1 (ex.: Prioridade='2 - Alta' vira uma coluna).
            # handle_unknown='ignore': categorias novas (não vistas no treino) não quebram a previsão.
            # min_frequency=30: categorias muito raras (<30 ocorrências) são agrupadas (evita explodir colunas).
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CAT_FEATURES_OLA),
        ],
        remainder="passthrough",  # as colunas numéricas passam direto, sem transformação
    )
    # Pipeline = "esteira": primeiro 'pre' (pré-processamento), depois 'clf' (o classificador).
    return Pipeline([("pre", pre), ("clf", estimator)])


def train_ola_classifier(kpi):
    """Treina classificadores de violacao de OLA (dados desbalanceados) e retorna (melhor, metricas)."""
    print("\n[MODELO] Classificacao de Risco de Violacao de OLA")
    df = kpi.copy()                                    # cópia para não alterar a tabela original
    for c in CAT_FEATURES_OLA:                         # garante texto sem vazios nas categóricas
        df[c] = df[c].fillna("Desconhecido").astype(str)

    X = df[CAT_FEATURES_OLA + NUM_FEATURES_OLA]        # entradas (categóricas + numéricas)
    y = df["ola_violada"].astype(int).values          # alvo: 1 = violou OLA, 0 = não

    # train_test_split divide em treino/teste. Importado aqui dentro (perto do uso).
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,          # 25% para teste
        random_state=RANDOM_STATE,
        stratify=y,              # mantém a MESMA proporção de violações no treino e no teste (importante em dados raros)
    )
    print(f"         treino={len(X_train)} | teste={len(X_test)} | positivos_teste={int(y_test.sum())}")

    # --- 2 classificadores candidatos ---
    # PROBLEMA: só ~1% dos incidentes violam OLA (dados "desbalanceados").
    # SOLUÇÃO: class_weight='balanced' faz o modelo dar mais PESO à classe rara (violações).
    candidates = {
        "DecisionTree": DecisionTreeClassifier(          # 1 árvore de decisão (Cap. 12)
            max_depth=8, min_samples_leaf=20, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(          # floresta de árvores (mais robusta)
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    results, best_name, best_pipe, best_f1 = {}, None, None, -np.inf  # controle do melhor (por F1)
    best_cm = None
    for name, est in candidates.items():
        pipe = _build_ola_pipeline(est)          # monta pré-processamento + modelo
        pipe.fit(X_train, y_train)               # treina
        pred = pipe.predict(X_test)              # previsão SIM/NÃO no teste
        proba = pipe.predict_proba(X_test)[:, 1] # probabilidade da classe 1 (violar) — 0 a 1
        cm = confusion_matrix(y_test, pred)      # matriz de confusão (acertos e erros)
        prec = float(precision_score(y_test, pred, zero_division=0))  # dos que ele disse "violar", quantos violaram
        rec = float(recall_score(y_test, pred, zero_division=0))      # das violações reais, quantas ele pegou
        f1 = float(f1_score(y_test, pred, zero_division=0))           # equilíbrio entre precisão e recall
        auc = float(roc_auc_score(y_test, proba))                    # ROC-AUC: poder de separar 0 de 1 (0.5=sorte, 1=perfeito)
        results[name] = {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "roc_auc": round(auc, 4),
            "confusion_matrix": cm.tolist(),   # .tolist() converte matriz numpy -> lista (para salvar em JSON)
        }
        print(f"         {name:14s} | Prec={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f} | AUC={auc:.3f}")
        if f1 > best_f1:                        # escolhemos o melhor pelo F1 (bom para dados desbalanceados)
            best_name, best_pipe, best_f1, best_cm = name, pipe, f1, cm

    print(f"         => Melhor modelo: {best_name} (F1={best_f1:.3f})")

    metrics = {
        "best_model": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positives_total": int(y.sum()),                # total de violações no dataset
        "positives_rate": round(float(y.mean()), 5),    # proporção de violações (~0.0097 = 0,97%)
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
    IDEIA: prevê amanhã; usa essa previsão como se fosse real para prever depois de amanhã; e assim por diante.
    history: lista dos ultimos totais diarios (>= 14 valores, ordem cronologica).
    Retorna lista de dicts {date, forecast}.
    """
    hist = list(history)                 # cópia da lista de valores recentes (vamos ir acrescentando previsões)
    last = pd.Timestamp(last_date)       # última data conhecida (como objeto de data)
    out = []                             # lista de saída
    for step in range(1, n_days + 1):    # para cada dia futuro (1, 2, ... n)
        d = last + pd.Timedelta(days=step)   # a data desse passo futuro
        # Monta as features desse dia futuro (mesma "receita" do pipeline, mas em memória):
        feat = {
            "day_of_week": d.dayofweek,
            "day_of_month": d.day,
            "month": d.month,
            "is_weekend": 1 if d.dayofweek >= 5 else 0,
            "lag_1": hist[-1],           # ontem = último valor da lista
            "lag_2": hist[-2],           # anteontem
            "lag_7": hist[-7],           # 7 dias atrás
            "lag_14": hist[-14],         # 14 dias atrás
            "rolling_mean_7": float(np.mean(hist[-7:])),    # média dos últimos 7
            "rolling_mean_14": float(np.mean(hist[-14:])),  # média dos últimos 14
            "rolling_std_7": float(np.std(hist[-7:])),      # desvio dos últimos 7
        }
        # Monta a linha na ORDEM exata que o modelo espera (FEATURES_VOLUME).
        row = np.array([[feat[f] for f in FEATURES_VOLUME]])
        yhat = float(model.predict(row)[0])  # previsão para o dia
        yhat = max(0.0, yhat)                # volume não pode ser negativo -> corta em 0
        out.append({"date": d.strftime("%Y-%m-%d"), "forecast": round(yhat, 1)})  # guarda {data, previsão}
        hist.append(yhat)                    # acrescenta a previsão ao histórico -> alimenta o próximo passo
    return out


# --------------------------------------------------------------------------- #
# Orquestracao — junta tudo e salva os artefatos
# --------------------------------------------------------------------------- #
def run_training():
    os.makedirs(MODELS_DIR, exist_ok=True)   # garante a pasta 'models'
    os.makedirs("data", exist_ok=True)       # garante a pasta 'data'

    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])  # carrega série diária (parse_dates: 'date' vira tipo data)
    kpi = pd.read_csv(KPI_PATH)                             # carrega incidentes

    # 1) Dois regressores de volume (um para D+1, outro para D+7)
    model_d1, metrics_d1 = train_volume_regressor(daily, "target_d1", "Volume D+1")
    model_d7, metrics_d7 = train_volume_regressor(daily, "target_d7", "Volume D+7")

    # 2) Classificador de risco de OLA
    ola_model, metrics_ola = train_ola_classifier(kpi)

    # --- Serialização: salva os modelos treinados em disco (a API vai carregá-los depois) ---
    joblib.dump(model_d1, os.path.join(MODELS_DIR, "volume_d1_model.joblib"))
    joblib.dump(model_d7, os.path.join(MODELS_DIR, "volume_d7_model.joblib"))
    joblib.dump(ola_model, os.path.join(MODELS_DIR, "ola_risk_model.joblib"))

    # Guarda os últimos 21 dias reais -> semente para o forecast recursivo da API/dashboard.
    recent = daily.sort_values("date").tail(21)
    history = recent["total_incidentes"].tolist()
    last_date = recent["date"].max().strftime("%Y-%m-%d")

    # --- Limiares de risco "data-driven" (baseados nos dados) ---
    # Como o modelo balanceado INFLA as probabilidades, não dá para usar 5%/20% fixos.
    # Em vez disso, definimos Baixo/Médio/Alto pelos PERCENTIS 70 e 90 da distribuição real de risco.
    X_all = kpi.copy()
    for c in CAT_FEATURES_OLA:
        X_all[c] = X_all[c].fillna("Desconhecido").astype(str)
    proba_all = ola_model.predict_proba(X_all[CAT_FEATURES_OLA + NUM_FEATURES_OLA])[:, 1]  # risco de todos os incidentes
    q70, q90 = np.quantile(proba_all, [0.70, 0.90])  # corte dos 70% e 90%
    risk_thresholds = [round(float(q70), 4), round(float(q90), 4)]
    print(f"[MODELO] Limiares de risco (P70/P90): {risk_thresholds}")

    # metadata.json: informações que a API precisa (features, histórico, limiares, resumo do dataset).
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
        # json.dump grava o dicionário como arquivo. ensure_ascii=False mantém acentos; indent=2 deixa legível.
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Previsões "de vitrine" para os KPIs do painel (usa o forecast recursivo do modelo D+1).
    fc = recursive_forecast(model_d1, history, last_date, n_days=7)
    kpi_d1 = fc[0]["forecast"]   # previsão de amanhã (D+1)
    kpi_d7 = fc[6]["forecast"]   # previsão do 7º dia (D+7)

    # metrics_report.json: relatório completo consumido pelo dashboard e pelos slides.
    report = {
        "volume_d1": metrics_d1,
        "volume_d7": metrics_d7,
        "ola_risk": metrics_ola,
        "forecast_next_7": fc,
        "kpi_cards": {                                       # os 4 números grandes do painel
            "volume_d1": kpi_d1,
            "volume_d7": kpi_d7,
            "ola_risk_index_pct": round(metrics_ola["positives_rate"] * 100, 2),  # taxa de violação em %
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
