# -*- coding: utf-8 -*-
"""
AlertOpsLite - Painel Operacional & Tatico (Data Viz)
Challenge Locaweb - FIAP 2TSCOA (Grupo Irmaos)
Autores: Marcos Machado (RM566099) e Matheus Machado (RM564991)
Fundamentacao: Capitulos 1 a 6 (Data Viz, tipos de graficos, Pareto, desenho de paineis)

Design alinhado ao sistema visual do deck AlertOpsLite (Sprint 2/3):
  paleta indigo/teal/crimson/vermelho, fontes Trebuchet MS/Calibri, cards claros
  (#F5F6FB) e cabecalho estilo "painel" (barra navy #12152E).

Gera as evidencias graficas (PNG alta resolucao) + dashboard interativo (HTML):
  img/timeseries_forecast.png  -> Serie historica + projecao D+1..D+7 com banda
  img/pareto_categories.png    -> Diagrama de Pareto 80/20 por Categoria
  img/risk_matrix.png          -> Heatmap de risco (equipes x criticidade)
  img/model_metrics.png        -> Matriz de confusao + metricas dos modelos
  img/dashboard_full.png       -> Painel operacional consolidado (hero)
  dashboard.html               -> Painel interativo consolidado
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_training import recursive_forecast  # reaproveita o forecast recursivo

# ------------------------------------ paleta do deck AlertOpsLite ---------- #
C_NAVY = "#12152E"     # navy escuro (barra do painel / cabecalho)
C_INK = "#1E2236"      # texto principal
C_INDIGO = "#4338CA"   # primaria
C_TEAL = "#0EA5A4"     # secundaria (dados)
C_CRIMSON = "#92223A"  # modelagem / projecao
C_RED = "#F23237"      # alerta / risco
C_AMBER = "#E8991C"    # atencao
C_MUTE = "#6B7088"     # texto secundario
C_GRID = "#E9ECF3"     # grade
C_CARD = "#F5F6FB"     # fundo de card claro
C_WHITE = "#FFFFFF"

FONT = "Trebuchet MS, Calibri, Segoe UI, sans-serif"

DAILY_PATH = os.path.join("data", "daily_incidents_processed.csv")
KPI_PATH = os.path.join("data", "kpi_incidents_processed.csv")
METRICS_PATH = os.path.join("data", "metrics_report.json")
MODELS_DIR = "models"
IMG_DIR = "img"

BASE_LAYOUT = dict(
    font=dict(family=FONT, size=13, color=C_INK),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=30, t=70, b=50),
    colorway=[C_INDIGO, C_TEAL, C_AMBER, C_CRIMSON, C_RED],
)


def _title(text, size=18):
    return dict(text=f"<b>{text}</b>", x=0.02, xanchor="left",
               font=dict(family=FONT, size=size, color=C_INK))


def _load():
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    kpi = pd.read_csv(KPI_PATH)
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    with open(os.path.join(MODELS_DIR, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return daily, kpi, metrics, meta


def _forecast_frame(daily, metrics, meta):
    """Reconstroi a projecao 7 dias com banda de confianca (widening com o horizonte)."""
    model_d1 = joblib.load(os.path.join(MODELS_DIR, "volume_d1_model.joblib"))
    fc = recursive_forecast(model_d1, meta["history_recent"], meta["last_date"], n_days=7)
    resid = metrics["volume_d1"]["resid_std"]
    fdf = pd.DataFrame(fc)
    fdf["date"] = pd.to_datetime(fdf["date"])
    fdf["k"] = np.arange(1, len(fdf) + 1)
    fdf["upper"] = (fdf["forecast"] + 1.28 * resid * np.sqrt(fdf["k"])).round(1)
    fdf["lower"] = (fdf["forecast"] - 1.28 * resid * np.sqrt(fdf["k"])).clip(lower=0).round(1)
    return fdf


# ---------------------------------------------------- 1) SERIE + PROJECAO --- #
def build_timeseries(daily, fdf, tail=120, standalone=True):
    hist = daily.sort_values("date").tail(tail)
    last_date = daily["date"].max()
    last_val = daily.loc[daily["date"] == last_date, "total_incidentes"].iloc[0]

    fx = pd.concat([
        pd.DataFrame({"date": [last_date], "forecast": [last_val],
                      "upper": [last_val], "lower": [last_val]}),
        fdf[["date", "forecast", "upper", "lower"]],
    ], ignore_index=True)

    traces = []
    traces.append(go.Scatter(
        x=pd.concat([fx["date"], fx["date"][::-1]]),
        y=pd.concat([fx["upper"], fx["lower"][::-1]]),
        fill="toself", fillcolor="rgba(146,34,58,0.13)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        name="Banda ~80%", showlegend=True,
    ))
    traces.append(go.Scatter(
        x=hist["date"], y=hist["total_incidentes"], mode="lines",
        line=dict(color=C_INDIGO, width=2.5), name="Histórico (real)",
    ))
    traces.append(go.Scatter(
        x=fx["date"], y=fx["forecast"], mode="lines+markers",
        line=dict(color=C_CRIMSON, width=3, dash="dot"),
        marker=dict(size=6, color=C_CRIMSON), name="Projeção D+1..D+7",
    ))
    if standalone:
        fig = go.Figure(traces)
        fig.update_layout(
            **BASE_LAYOUT,
            title=_title("Volume Diário de Incidentes — Histórico e Projeção (D+1 a D+7)"),
            legend=dict(orientation="h", y=1.06, x=0.02, font=dict(size=12)),
            height=460, width=1100,
        )
        fig.update_xaxes(gridcolor=C_GRID, title="Data")
        fig.update_yaxes(gridcolor=C_GRID, title="Incidentes elegíveis / dia")
        return fig
    return traces


# ------------------------------------------------------------ 2) PARETO --- #
def build_pareto(kpi, top=10, standalone=True):
    vc = kpi["Categoria"].fillna("Outras").value_counts().head(top)
    cats = vc.index.tolist()
    vals = vc.values
    cum = np.cumsum(vals) / vals.sum() * 100

    bar = go.Bar(x=cats, y=vals, name="Incidentes",
                 marker_color=C_INDIGO, text=vals, textposition="outside")
    line = go.Scatter(x=cats, y=cum, name="% Acumulado", yaxis="y2",
                      mode="lines+markers", line=dict(color=C_CRIMSON, width=3),
                      marker=dict(size=7))
    if standalone:
        fig = go.Figure([bar, line])
        fig.add_hline(y=80, line_dash="dash", line_color=C_AMBER, yref="y2",
                      annotation_text="80%", annotation_position="top left")
        fig.update_layout(
            **BASE_LAYOUT,
            title=_title("Diagrama de Pareto — Concentração de Incidentes por Categoria (80/20)"),
            yaxis=dict(title="Nº de incidentes", gridcolor=C_GRID),
            yaxis2=dict(title="% acumulado", overlaying="y", side="right",
                        range=[0, 105], showgrid=False),
            legend=dict(orientation="h", y=1.08, x=0.02, font=dict(size=12)),
            height=460, width=1100,
        )
        fig.update_xaxes(title="Categoria", tickangle=-30)
        return fig
    return bar, line, cum


# -------------------------------------------------------- 3) RISK MATRIX --- #
def _risk_pivot(kpi, top_teams=8):
    df = kpi.copy()
    df["prio"] = np.where(df["Prioridade"].astype(str).str.contains("2"), "P2 (Alta)", "P3 (Média)")
    top = df["Grupo designado"].value_counts().head(top_teams).index.tolist()
    df = df[df["Grupo designado"].isin(top)]
    rate = df.groupby(["Grupo designado", "prio"])["ola_violada"].mean().mul(100).round(1)
    cnt = df.groupby(["Grupo designado", "prio"])["ola_violada"].size()
    z = rate.unstack("prio").reindex(top)
    c = cnt.unstack("prio").reindex(top)
    return z, c


def build_risk_matrix(kpi, standalone=True):
    z, c = _risk_pivot(kpi)
    cols = list(z.columns)
    rows = list(z.index)
    text = [[f"{z.iloc[i,j]:.1f}%<br>({int(c.iloc[i,j]) if not pd.isna(c.iloc[i,j]) else 0})"
             for j in range(len(cols))] for i in range(len(rows))]
    heat = go.Heatmap(
        z=z.values, x=cols, y=rows,
        colorscale=[[0, C_CARD], [0.5, C_AMBER], [1, C_RED]],
        text=text, texttemplate="%{text}", textfont=dict(size=11, family=FONT),
        showscale=standalone, colorbar=dict(title="Risco<br>OLA (%)"), hoverongaps=False,
    )
    if standalone:
        fig = go.Figure(heat)
        fig.update_layout(
            **BASE_LAYOUT,
            title=_title("Matriz de Risco Operacional — Violação de OLA por Equipe × Criticidade", 17),
            height=460, width=760,
        )
        fig.update_xaxes(title="Criticidade")
        fig.update_yaxes(title="Equipe (Grupo designado)")
        return fig
    return heat


# ------------------------------------------------------- 4) MODEL METRICS --- #
def build_model_metrics(metrics, standalone=True):
    cm = np.array(metrics["ola_risk"]["best_confusion_matrix"])
    best = metrics["ola_risk"]["best_model"]
    m = metrics["ola_risk"]["candidates"][best]

    cm_labels = [["VN", "FP"], ["FN", "VP"]]
    cm_text = [[f"{cm_labels[i][j]}<br><b>{cm[i,j]}</b>" for j in range(2)] for i in range(2)]
    heat = go.Heatmap(
        z=cm, x=["Prev: Não", "Prev: Sim"], y=["Real: Não", "Real: Sim"],
        colorscale=[[0, C_CARD], [1, C_INDIGO]], text=cm_text, texttemplate="%{text}",
        textfont=dict(size=14, family=FONT), showscale=False,
    )

    reg_names = ["R² D+1", "R² D+7"]
    reg_vals = [metrics["volume_d1"]["candidates"][metrics["volume_d1"]["best_model"]]["r2"],
                metrics["volume_d7"]["candidates"][metrics["volume_d7"]["best_model"]]["r2"]]
    clf_names = ["Precisão", "Recall", "F1", "ROC-AUC"]
    clf_vals = [m["precision"], m["recall"], m["f1"], m["roc_auc"]]

    bars_reg = go.Bar(x=reg_names, y=reg_vals, marker_color=C_INDIGO,
                      text=[f"{v:.2f}" for v in reg_vals], textposition="outside", name="Regressão")
    bars_clf = go.Bar(x=clf_names, y=clf_vals, marker_color=[C_TEAL, C_TEAL, C_TEAL, C_INDIGO],
                      text=[f"{v:.2f}" for v in clf_vals], textposition="outside", name="Classificação")

    if standalone:
        fig = make_subplots(
            rows=1, cols=3, column_widths=[0.34, 0.30, 0.36],
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
            subplot_titles=("Matriz de Confusão — Risco de OLA",
                            "Qualidade Regressão (R²)",
                            f"Classificação OLA ({best})"),
        )
        fig.add_trace(heat, row=1, col=1)
        fig.add_trace(bars_reg, row=1, col=2)
        fig.add_trace(bars_clf, row=1, col=3)
        fig.update_yaxes(range=[0, 1], row=1, col=2, gridcolor=C_GRID)
        fig.update_yaxes(range=[0, 1], row=1, col=3, gridcolor=C_GRID)
        fig.update_layout(
            font=dict(family=FONT, size=13, color=C_INK),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=50, r=30, t=80, b=40),
            title=_title("Avaliação dos Modelos Preditivos (Validação)"),
            showlegend=False, height=440, width=1150,
        )
        for ann in fig.layout.annotations:
            ann.font.family = FONT
            ann.font.size = 14
        return fig
    return heat, bars_reg, bars_clf


# ------------------------------------------------- PAINEL CONSOLIDADO (hero) - #
def build_full_dashboard(daily, kpi, metrics, meta, fdf):
    cards = metrics["kpi_cards"]
    fig = make_subplots(
        rows=4, cols=4,
        row_heights=[0.13, 0.32, 0.28, 0.27],
        vertical_spacing=0.085, horizontal_spacing=0.08,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "xy", "colspan": 4}, None, None, None],
            [{"type": "xy", "colspan": 2}, None, {"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None, {"type": "xy", "colspan": 2}, None],
        ],
        subplot_titles=("", "", "", "",
                        "Volume Diário — Histórico e Projeção D+1..D+7",
                        "Pareto de Incidentes por Categoria (80/20)",
                        "Risco de OLA por Equipe × Criticidade",
                        "Matriz de Confusão — Risco de OLA",
                        "Qualidade dos Modelos"),
    )

    # --- KPI cards (linha 1) com cores do deck
    def card(val, title, suffix="", color=C_INDIGO):
        return go.Indicator(
            mode="number", value=val,
            number=dict(suffix=suffix, font=dict(size=36, color=color, family=FONT)),
            title=dict(text=title, font=dict(size=13, color=C_MUTE, family=FONT)),
        )
    fig.add_trace(card(cards["volume_d1"], "Volume Previsto D+1", color=C_INDIGO), row=1, col=1)
    fig.add_trace(card(cards["volume_d7"], "Volume Previsto D+7", color=C_TEAL), row=1, col=2)
    fig.add_trace(card(cards["ola_risk_index_pct"], "Índice de Risco de OLA", suffix="%", color=C_RED), row=1, col=3)
    fig.add_trace(card(cards["eligible_incidents"], "Incidentes Elegíveis", color=C_AMBER), row=1, col=4)

    # fundo de card claro atras de cada KPI (alinhado ao dominio do indicador)
    for tr in fig.data:
        if tr.type == "indicator":
            dx, dy = tr.domain.x, tr.domain.y
            fig.add_shape(type="rect", xref="paper", yref="paper",
                          x0=dx[0] - 0.006, x1=dx[1] + 0.006, y0=dy[0] - 0.01, y1=dy[1] + 0.005,
                          fillcolor=C_CARD, line=dict(width=0), layer="below")

    # --- Serie temporal (linha 2)
    for tr in build_timeseries(daily, fdf, tail=100, standalone=False):
        fig.add_trace(tr, row=2, col=1)

    # --- Pareto (linha 3, esq)
    bar, line, _ = build_pareto(kpi, standalone=False)
    fig.add_trace(bar, row=3, col=1)

    # --- Risk matrix (linha 3, dir)
    fig.add_trace(build_risk_matrix(kpi, standalone=False), row=3, col=3)

    # --- Confusion matrix (linha 4, esq) + metricas (linha 4, dir)
    heat, bars_reg, bars_clf = build_model_metrics(metrics, standalone=False)
    fig.add_trace(heat, row=4, col=1)
    best = metrics["ola_risk"]["best_model"]
    m = metrics["ola_risk"]["candidates"][best]
    fig.add_trace(go.Bar(
        x=["R² D+1", "R² D+7", "Recall", "ROC-AUC"],
        y=[metrics["volume_d1"]["candidates"][metrics["volume_d1"]["best_model"]]["r2"],
           metrics["volume_d7"]["candidates"][metrics["volume_d7"]["best_model"]]["r2"],
           m["recall"], m["roc_auc"]],
        marker_color=[C_INDIGO, C_TEAL, C_AMBER, C_CRIMSON],
        text=[f"{v:.2f}" for v in [
            metrics["volume_d1"]["candidates"][metrics["volume_d1"]["best_model"]]["r2"],
            metrics["volume_d7"]["candidates"][metrics["volume_d7"]["best_model"]]["r2"],
            m["recall"], m["roc_auc"]]],
        textposition="outside",
    ), row=4, col=3)
    fig.update_yaxes(range=[0, 1.05], row=4, col=3, gridcolor=C_GRID)

    # cabecalho estilo "painel": pontos coloridos + nome do app (assinatura do deck)
    dots = ('<span style="color:#F23237">●</span> '
            '<span style="color:#E8991C">●</span> '
            '<span style="color:#0EA5A4">●</span>')
    fig.update_layout(
        font=dict(family=FONT, size=12, color=C_INK),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=1220, width=1400,
        title=dict(
            text=f"{dots}  <b>AlertOpsLite — Painel Operacional &amp; Tático</b>"
                 "<br><span style='font-size:13px;color:#6B7088'>"
                 "Inteligência preditiva · Challenge Locaweb · FIAP 2TSCOA · Grupo Irmãos</span>",
            x=0.02, y=0.985, xanchor="left", font=dict(size=22, color=C_NAVY, family=FONT)),
        margin=dict(l=55, r=35, t=110, b=45),
    )
    fig.update_xaxes(gridcolor=C_GRID)
    fig.update_yaxes(gridcolor=C_GRID)
    fig.update_xaxes(tickangle=-30, row=3, col=1)
    for ann in fig.layout.annotations:
        ann.font.family = FONT
        if ann.text and ann.text not in ("",):
            ann.font.color = C_INK
    return fig


def build_confusion(metrics):
    cm = np.array(metrics["ola_risk"]["best_confusion_matrix"])
    labels = [["VN", "FP"], ["FN", "VP"]]
    text = [[f"{labels[i][j]}<br><b>{cm[i,j]}</b>" for j in range(2)] for i in range(2)]
    heat = go.Heatmap(
        z=cm, x=["Prev: Não", "Prev: Sim"], y=["Real: Não", "Real: Sim"],
        colorscale=[[0, C_CARD], [1, C_INDIGO]], text=text, texttemplate="%{text}",
        textfont=dict(size=15, family=FONT), showscale=False,
    )
    fig = go.Figure(heat)
    fig.update_layout(**BASE_LAYOUT, title=_title("Matriz de Confusão — Risco de OLA", 15), height=360)
    return fig


def build_quality(metrics):
    best = metrics["ola_risk"]["best_model"]
    m = metrics["ola_risk"]["candidates"][best]
    names = ["R² D+1", "R² D+7", "Recall", "F1", "ROC-AUC"]
    vals = [metrics["volume_d1"]["candidates"][metrics["volume_d1"]["best_model"]]["r2"],
            metrics["volume_d7"]["candidates"][metrics["volume_d7"]["best_model"]]["r2"],
            m["recall"], m["f1"], m["roc_auc"]]
    colors = [C_INDIGO, C_INDIGO, C_TEAL, C_TEAL, C_CRIMSON]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors,
                           text=[f"{v:.2f}" for v in vals], textposition="outside"))
    fig.update_layout(**BASE_LAYOUT, title=_title("Qualidade dos Modelos (escala 0 a 1)", 15), height=360)
    fig.update_yaxes(range=[0, 1.08], gridcolor=C_GRID)
    return fig


def _fig_div(fig, height=430):
    fig.update_layout(width=None, autosize=True, height=height)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, default_width="100%",
                       config={"responsive": True, "displayModeBar": False})


# ---- conteudo didatico (usuario leigo): O que e / De onde vem / Como ler / O que tira -- #
def _explanations(metrics):
    cards = metrics["kpi_cards"]
    rd1 = metrics["volume_d1"]["candidates"][metrics["volume_d1"]["best_model"]]
    kpi_help = {
        "d1": "Quantos incidentes P2/P3 devem ser abertos amanhã (D+1). Serve para dimensionar a escala e o plantão do próximo dia.",
        "d7": "Estimativa de incidentes daqui a 7 dias (D+7). Serve para planejar a capacidade da semana.",
        "risk": "Percentual histórico de incidentes elegíveis que estouraram o prazo de OLA. É a taxa-base de risco da operação.",
        "elig": "Total de incidentes P2/P3 usados na análise, após remover ~79% de ruído (alertas de monitoramento sem intervenção).",
    }
    charts = [
        dict(key="ts", full=True, title="Volume Diário — Histórico e Projeção (D+1 a D+7)",
             tip="Passado real + previsão dos próximos 7 dias.",
             oque="A quantidade de incidentes abertos por dia ao longo do tempo e a previsão para os próximos 7 dias.",
             origem="Contagem diária dos incidentes elegíveis (dataset real da Locaweb), processada por um modelo RandomForest de regressão.",
             ler="Linha cheia (indigo) = histórico real. Linha pontilhada (vinho) = previsão. A faixa clara ao redor da previsão é a margem de incerteza (~80%). Eixo horizontal = datas; eixo vertical = incidentes por dia. Passe o mouse para ver o valor exato de cada dia.",
             tirar="A tendência (subindo ou caindo), o padrão semanal (quedas em fins de semana) e a antecipação de picos — para agir antes que aconteçam."),
        dict(key="pareto", full=False, title="Pareto — Incidentes por Categoria (80/20)",
             tip="As poucas categorias que causam a maioria dos incidentes.",
             oque="Quais categorias de problema concentram a maior parte dos incidentes, seguindo a regra 80/20 (poucas causas geram a maioria dos casos).",
             origem="Contagem de incidentes agrupada por categoria, ordenada da maior para a menor.",
             ler="As barras (indigo) mostram o nº de incidentes de cada categoria, em ordem decrescente. A linha (vinho) é o % acumulado; a linha tracejada marca os 80%. Tudo à esquerda de onde a linha cruza 80% são as categorias que mais pesam.",
             tirar="Onde concentrar esforço: resolver as poucas categorias da esquerda ataca a maior parte do problema."),
        dict(key="risk", full=False, title="Matriz de Risco — Violação de OLA por Equipe × Criticidade",
             tip="Onde o risco de estourar a OLA se concentra.",
             oque="A taxa de violação de OLA de cada equipe, separada por prioridade (P2 Alta e P3 Média).",
             origem="Para cada equipe e prioridade: % de incidentes que violaram a OLA. O número entre parênteses é o volume de incidentes daquela célula.",
             ler="Cores indicam o risco: claro = baixo, âmbar = médio, vermelho = alto. Linhas = equipes; colunas = criticidade. Foque nas células vermelhas com volume relevante.",
             tirar="Quais equipes/prioridades precisam de reforço ou atenção preventiva para não estourar prazos."),
        dict(key="confusion", full=False, title="Matriz de Confusão — Acertos e Erros do Modelo de Risco",
             tip="Como o modelo acerta e erra ao prever violações.",
             oque="Um resumo de quantas vezes o modelo de risco de OLA acertou e errou, comparado com o que realmente aconteceu.",
             origem="Comparação entre a previsão do modelo e a realidade, em dados de teste que o modelo não viu no treino.",
             ler="VP = previu violação e violou (acerto). VN = previu que não violaria e não violou (acerto). FP = alarme falso. FN = violação que passou batido. Quanto maiores VP e VN, melhor.",
             tirar="O equilíbrio entre pegar violações de verdade (recall) e não gerar alarmes falsos demais."),
        dict(key="quality", full=False, title="Qualidade dos Modelos — O quão confiáveis são",
             tip="Notas que medem o desempenho dos modelos.",
             oque="As métricas que medem se as previsões são boas. Todas vão de 0 a 1 (quanto maior, melhor).",
             origem="Avaliação dos modelos em dados de teste separados do treino.",
             ler="R² (previsão de volume): perto de 1 = previsão muito aderente. Recall: fração das violações que o modelo detecta. ROC-AUC: 0,5 = chute; 1,0 = perfeito. F1: equilíbrio entre precisão e recall.",
             tirar=f"A confiança para usar os números: volume com R²≈{rd1['r2']:.2f} e risco de OLA com ROC-AUC≈{metrics['ola_risk']['candidates'][metrics['ola_risk']['best_model']]['roc_auc']:.2f} (bom para um MVP)."),
    ]
    return kpi_help, charts


_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#EEF0F7;color:#1E2236;font-family:'Trebuchet MS','Calibri','Segoe UI',sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:22px}
header.top{background:#12152E;color:#fff;border-radius:16px;padding:20px 24px;display:flex;align-items:center;gap:16px}
.dots{display:flex;gap:7px}.dot{width:12px;height:12px;border-radius:50%}
header .tt{font-size:22px;font-weight:bold;line-height:1.1}
header .ss{color:#AEB6E0;font-size:13px;margin-top:3px}
.onb{background:#fff;border-radius:16px;padding:18px 22px;margin:16px 0;box-shadow:0 6px 22px rgba(18,21,46,.07)}
.onb h2{margin:0 0 12px;font-size:16px;color:#4338CA}
.steps{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:12px}
.step{flex:1;min-width:230px;display:flex;gap:11px;align-items:flex-start}
.step .badge{background:#4338CA;color:#fff;width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:bold;flex:none;font-size:14px}
.step b{display:block;font-size:14px}.step span{font-size:13px;color:#5A6172}
.legend{display:flex;gap:16px;flex-wrap:wrap;border-top:1px solid #E9ECF3;padding-top:11px;font-size:12.5px;color:#5A6172}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:middle}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:16px 0}
.kpi{background:#fff;border-radius:16px;padding:16px 18px;box-shadow:0 6px 22px rgba(18,21,46,.07)}
.kpi .lab{color:#6B7088;font-size:13px;display:flex;align-items:center;gap:7px}
.kpi .val{font-size:36px;font-weight:bold;margin-top:6px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
.card{background:#fff;border-radius:16px;padding:15px 17px;box-shadow:0 6px 22px rgba(18,21,46,.07);overflow:visible}
.card.full{grid-column:1/-1}
.card-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:4px}
.card-head h3{margin:0;font-size:15.5px;color:#1E2236}
.help{border:none;background:#EEF0F7;color:#4338CA;width:27px;height:27px;border-radius:50%;font-weight:bold;cursor:pointer;font-size:14px;font-style:italic;flex:none}
.help:hover,.help.active{background:#4338CA;color:#fff}
.tipwrap{position:relative;display:inline-flex}
.tip{position:absolute;right:0;top:34px;width:240px;background:#12152E;color:#fff;font-size:12px;line-height:1.45;padding:9px 11px;border-radius:9px;opacity:0;pointer-events:none;transition:.15s;z-index:30;box-shadow:0 8px 26px rgba(0,0,0,.25)}
.tipwrap:hover .tip{opacity:1}
.help-panel{display:none;background:#F5F6FB;border-radius:12px;padding:14px 16px;margin:8px 0 12px}
.help-panel.open{display:block}
.hp{display:grid;grid-template-columns:1fr 1fr;gap:12px 20px}
.hp h4{margin:0 0 3px;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
.hp p{margin:0;font-size:13px;color:#3A414F;line-height:1.5}
.c-oque h4{color:#4338CA}.c-origem h4{color:#0EA5A4}.c-ler h4{color:#E8991C}.c-tirar h4{color:#F23237}
footer{color:#6B7088;font-size:12px;text-align:center;padding:20px}
@media(max-width:860px){.grid{grid-template-columns:1fr}.hp{grid-template-columns:1fr}}
"""

_JS = """
document.querySelectorAll('.help').forEach(function(b){
  b.addEventListener('click',function(){
    var card=b.closest('.card');
    if(!card)return;
    var p=card.querySelector('.help-panel');
    if(p){var open=p.classList.toggle('open');b.classList.toggle('active',open);
      b.textContent=open?'\\u2715':'i';}
  });
});
"""


def build_educational_html(daily, kpi, metrics, meta, fdf, path):
    kpi_help, charts = _explanations(metrics)
    cards = metrics["kpi_cards"]

    divs = {
        "ts": _fig_div(build_timeseries(daily, fdf), 440),
        "pareto": _fig_div(build_pareto(kpi), 430),
        "risk": _fig_div(build_risk_matrix(kpi), 430),
        "confusion": _fig_div(build_confusion(metrics), 360),
        "quality": _fig_div(build_quality(metrics), 360),
    }

    kpi_defs = [
        ("Volume Previsto D+1", f"{cards['volume_d1']:.0f}", C_INDIGO, kpi_help["d1"]),
        ("Volume Previsto D+7", f"{cards['volume_d7']:.0f}", C_TEAL, kpi_help["d7"]),
        ("Índice de Risco de OLA", f"{cards['ola_risk_index_pct']:.2f}%".replace(".", ","), C_RED, kpi_help["risk"]),
        ("Incidentes Elegíveis", f"{cards['eligible_incidents']:,}".replace(",", "."), C_AMBER, kpi_help["elig"]),
    ]
    kpi_html = ""
    for lab, val, col, tip in kpi_defs:
        kpi_html += (
            f'<div class="kpi"><div class="lab">{lab}'
            f'<span class="tipwrap"><button class="help" aria-label="Ajuda">i</button>'
            f'<span class="tip">{tip}</span></span></div>'
            f'<div class="val" style="color:{col}">{val}</div></div>'
        )

    cards_html = ""
    for c in charts:
        cards_html += (
            f'<section class="card{" full" if c["full"] else ""}">'
            f'<div class="card-head"><h3>{c["title"]}</h3>'
            f'<span class="tipwrap"><button class="help" aria-label="Como ler este gráfico">i</button>'
            f'<span class="tip">{c["tip"]} Clique para os detalhes.</span></span></div>'
            f'<div class="help-panel"><div class="hp">'
            f'<div class="c-oque"><h4>O que é</h4><p>{c["oque"]}</p></div>'
            f'<div class="c-origem"><h4>De onde vem</h4><p>{c["origem"]}</p></div>'
            f'<div class="c-ler"><h4>Como ler</h4><p>{c["ler"]}</p></div>'
            f'<div class="c-tirar"><h4>O que você tira daqui</h4><p>{c["tirar"]}</p></div>'
            f'</div></div>'
            f'<div class="plot">{divs[c["key"]]}</div></section>'
        )

    header = (
        '<header class="top"><div class="dots">'
        '<span class="dot" style="background:#F23237"></span>'
        '<span class="dot" style="background:#E8991C"></span>'
        '<span class="dot" style="background:#0EA5A4"></span></div>'
        '<div><div class="tt">AlertOpsLite — Painel Operacional</div>'
        '<div class="ss">Inteligência preditiva para operações de TI · Challenge Locaweb · FIAP 2TSCOA · Grupo Irmãos</div>'
        '</div></header>'
    )
    onboarding = (
        '<div class="onb"><h2>Como usar este painel</h2><div class="steps">'
        '<div class="step"><span class="badge">1</span><div><b>Comece pelos KPIs</b>'
        '<span>Os quatro números no topo resumem a previsão do dia e o risco atual.</span></div></div>'
        '<div class="step"><span class="badge">2</span><div><b>Explore os gráficos</b>'
        '<span>Passe o mouse sobre linhas, barras e células para ver os valores exatos.</span></div></div>'
        '<div class="step"><span class="badge">3</span><div><b>Entenda cada gráfico</b>'
        '<span>Clique no ícone <i>i</i> de qualquer gráfico para ver o que é, de onde vem e como ler.</span></div></div>'
        '</div><div class="legend">'
        '<span><i style="background:#4338CA"></i>Histórico / volume</span>'
        '<span><i style="background:#92223A"></i>Projeção (previsão)</span>'
        '<span><i style="background:#F23237"></i>Risco alto</span>'
        '<span><i style="background:#E8991C"></i>Atenção</span>'
        '<span><i style="background:#0EA5A4"></i>Indicador secundário</span>'
        '</div></div>'
    )
    footer = ('<footer>AlertOpsLite · gerado a partir do dataset real da Locaweb '
              '(122.543 chamados → 25.600 incidentes elegíveis) · Sprint 3 (MVP)</footer>')

    page = (
        '<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>AlertOpsLite — Painel Operacional</title>'
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'
        f'<style>{_CSS}</style></head><body><div class="wrap">'
        f'{header}{onboarding}<div class="kpis">{kpi_html}</div>'
        f'<div class="grid">{cards_html}</div>{footer}'
        f'</div><script>{_JS}</script></body></html>'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def run_dashboard():
    os.makedirs(IMG_DIR, exist_ok=True)
    daily, kpi, metrics, meta = _load()
    fdf = _forecast_frame(daily, metrics, meta)

    figs = {
        "timeseries_forecast": build_timeseries(daily, fdf),
        "pareto_categories": build_pareto(kpi),
        "risk_matrix": build_risk_matrix(kpi),
        "model_metrics": build_model_metrics(metrics),
    }
    for name, fig in figs.items():
        out = os.path.join(IMG_DIR, f"{name}.png")
        fig.write_image(out, scale=2)
        print(f"[DASHBOARD] Evidencia salva: {out}")

    full = build_full_dashboard(daily, kpi, metrics, meta, fdf)
    full.write_image(os.path.join(IMG_DIR, "dashboard_full.png"), scale=2)
    print(f"[DASHBOARD] Evidencia salva: {os.path.join(IMG_DIR, 'dashboard_full.png')}")

    build_educational_html(daily, kpi, metrics, meta, fdf, "dashboard.html")
    print("[DASHBOARD] Painel interativo (didatico) salvo: dashboard.html")


if __name__ == "__main__":
    run_dashboard()
