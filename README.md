# AlertOpsLite — Inteligência Preditiva para Operações de TI

**Challenge Locaweb · FIAP 2TSCOA · Grupo Irmãos · Sprint 3 (MVP)**
Integrantes: Marcos Machado (RM566099) · Matheus Machado (RM564991)

MVP funcional que transforma **122.543 chamados** da operação 24×7 da Locaweb em
inteligência acionável: **previsão de volume** de incidentes (D+1 / D+7) e
**sinalização de risco de violação de OLA**, servidos por uma **API FastAPI** e um
**painel de Data Viz** interativo e didático.

Fundamentação: capítulos **1–6** (Data Viz) e **9–12** (Data Science) da FIAP.

---

## 🧱 Estrutura do projeto

```
Sprint3asset/
├── src/
│   ├── data_pipeline.py       # Cap. 9  — limpeza + feature engineering
│   ├── model_training.py      # Cap.10-12 — regressão + classificação
│   ├── dashboard.py           # Cap.1-6 — painel Plotly didático + PNGs
│   └── app_api.py             # Backend FastAPI (Pydantic v2)
├── builder.py                 # Orquestra o pipeline completo (5 passos)
├── capture_dashboard.py       # Prints do dashboard em execução (Chrome/Edge)
├── build_sprint3_from_sprint2.py  # Gera o PPTX a partir do template da Sprint 2
├── data/
│   └── LW-DATASET.xlsx        # dataset real da Locaweb (input)
├── templates/
│   └── EC_Sprint_2_..._GrupoIrmaos.pptx   # template do deck (input)
├── models/                    # modelos .joblib + metadata (gerado)
├── img/                       # evidências gráficas PNG (gerado)
├── dashboard.html             # painel interativo didático (gerado)
├── EC_Sprint_3_..._GrupoIrmaos.pptx        # apresentação (gerado)
├── requirements.txt
├── Dockerfile · .dockerignore · .gitignore
```

## 🚀 Como rodar (local)

```bash
# 1. dependências (Python 3.12)
pip install -r requirements.txt

# 2. pipeline completo: dados → modelos → painel → prints → PPTX
python builder.py
```

Ou passo a passo, **a partir da raiz do projeto**:

```bash
python src/data_pipeline.py      # data/*.csv
python src/model_training.py     # models/* + data/metrics_report.json
python src/dashboard.py          # img/*.png + dashboard.html
python capture_dashboard.py      # prints do dashboard (requer Chrome/Edge)
python build_sprint3_from_sprint2.py   # EC_Sprint_3_...pptx
```

## 🌐 API (FastAPI)

```bash
uvicorn src.app_api:app --reload
```

- Swagger interativo: <http://127.0.0.1:8000/docs>
- `GET  /api/v1/health` — integridade
- `GET  /api/v1/metrics` — métricas de validação dos modelos
- `POST /api/v1/predict/volume` — volume D+1 e D+7 (+ curva 7 dias)
- `POST /api/v1/predict/ola-risk` — probabilidade e nível de risco (Baixo/Médio/Alto)

Exemplo:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict/ola-risk \
  -H "Content-Type: application/json" \
  -d '{"prioridade":"2 - Alta","produto":"lhco","categoria":"cat71","grupo_designado":"Team11","hour":14,"day_of_week":0,"month":8,"is_weekend":0}'
```

## 🐳 Docker (uso na próxima sprint)

O `Dockerfile` empacota o serviço: instala dependências, **treina os modelos no
build** (a partir do dataset em `data/`) e sobe a API.

```bash
# build da imagem
docker build -t alertopslite:sprint3 .

# subir a API (Swagger em http://localhost:8000/docs)
docker run --rm -p 8000:8000 alertopslite:sprint3
```

> Débito técnico da Sprint 4: orquestração de alertas via **n8n** (Telegram/E-mail),
> deploy conteinerizado na VPS Hostinger e persistência em PostgreSQL para retreino.

## 📊 Resultados (validação)

| Modelo | Algoritmo | Métrica | Resultado |
| :--- | :--- | :--- | :--- |
| Volume D+1 | RandomForest | R² · MAE | **0,58** · 13,0 |
| Volume D+7 | RandomForest | R² | **0,54** |
| Risco de OLA | RandomForest | ROC-AUC · Recall | **0,82** · 0,29 |

- **25.600** incidentes elegíveis (P2/P3) após remover ~79% de ruído.
- **248** violações de OLA (0,97%) — `class_weight='balanced'`.
- Regressão validada com **holdout temporal**; risco por **percentis P70/P90**.

## 🖥️ Painel didático (`dashboard.html`)

Painel **autoexplicativo** para usuário leigo: onboarding "Como usar", KPI cards com
tooltip, e cada gráfico com um ícone **ⓘ** que abre **O que é · De onde vem · Como
ler · O que você tira daqui**. Tooltips do Plotly mostram valores exatos.
