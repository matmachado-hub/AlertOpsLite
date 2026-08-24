# -*- coding: utf-8 -*-
"""
AlertOpsLite - Gera a Sprint 3 a partir do template EXATO da Sprint 2.
Challenge Locaweb - FIAP 2TSCOA (Grupo Irmaos)

Estrategia (aprovada): copia identica do .pptx da Sprint 2 (mesmo tema/fontes/icones)
e substitui APENAS os slides ilustrativos/prototipo pela EVIDENCIA REAL construida:
  - Slide 1  (capa): re-versiona para "Sprint 3 - Evidencias de construcao".
  - Slide 10 (modelagem): grafico "ilustrativo" -> previsao REAL + metricas reais.
  - Slide 11 (prototipo painel): KPIs/graficos mock -> painel REAL (dados reais).
Os demais slides (contexto, problema, arquitetura, gestao, roadmap, obrigado)
permanecem IDENTICOS.
"""

import os
import sys
import copy
import json
import shutil
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model_training import recursive_forecast  # noqa

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE


def _rgb(h):
    return RGBColor.from_string(h.lstrip("#"))

SRC_PPTX = os.path.join("templates", "EC_Sprint_2_2TSCOA_arqsolucao_AlertOpsLite_GrupoIrmaos.pptx")
OUT_PPTX = "EC_Sprint_3_2TSCOA_Evidencias_Construcao_AlertOpsLite_GrupoIrmaos.pptx"
IMG = "img"
REPO_URL = "https://github.com/matmachado-hub/AlertOpsLite"


def _ensure_qr():
    """Gera img/qr_github.png (reprodutível). Requer o pacote 'qrcode'."""
    out = os.path.join(IMG, "qr_github.png")
    try:
        import qrcode
    except ImportError:
        print("[S3] pacote 'qrcode' ausente; slide do GitHub usará QR existente (se houver).")
        return
    qr = qrcode.QRCode(box_size=12, border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(REPO_URL); qr.make(fit=True)
    qr.make_image(fill_color=(18, 21, 46), back_color="white").save(out)

# paleta espelhando os graficos da Sprint 2
C_ORANGE = "#E8991C"
C_CRIMSON = "#92223A"
C_INDIGO = "#4338CA"
C_RED = "#F23237"
C_INK = "#1E2236"
C_MUTE = "#6B7088"
C_GRID = "#E9ECF3"
FONT = "Trebuchet MS, Calibri, sans-serif"


# --------------------------------------------------------------------------- #
# 1) Imagens de evidencia REAL, no aspect ratio EXATO de cada caixa do template
# --------------------------------------------------------------------------- #
def _load_real():
    daily = pd.read_csv(os.path.join("data", "daily_incidents_processed.csv"), parse_dates=["date"])
    kpi = pd.read_csv(os.path.join("data", "kpi_incidents_processed.csv"))
    with open(os.path.join("data", "metrics_report.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    with open(os.path.join("models", "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    model_d1 = joblib.load(os.path.join("models", "volume_d1_model.joblib"))
    return daily, kpi, metrics, meta, model_d1


def build_forecast_strip(daily, meta, model_d1, resid, path):
    """Faixa larga e baixa (11.70 x 1.50 in) para o slide 10 (sem titulo interno)."""
    hist = daily.sort_values("date").tail(45)
    last_date = daily["date"].max()
    last_val = daily.loc[daily["date"] == last_date, "total_incidentes"].iloc[0]
    fc = recursive_forecast(model_d1, meta["history_recent"], meta["last_date"], n_days=7)
    fdf = pd.DataFrame(fc); fdf["date"] = pd.to_datetime(fdf["date"])
    # datetimes nativos (Timestamp nao e serializavel pelo kaleido/orjson)
    hist_x = list(hist["date"].dt.to_pydatetime())
    fx_x = [last_date.to_pydatetime()] + list(fdf["date"].dt.to_pydatetime())
    fx_y = [float(last_val)] + [float(v) for v in fdf["forecast"]]
    k = np.arange(0, len(fx_x))
    upper = [v + 1.28 * resid * np.sqrt(max(i, 0.3)) for i, v in zip(k, fx_y)]
    lower = [max(0, v - 1.28 * resid * np.sqrt(max(i, 0.3))) for i, v in zip(k, fx_y)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fx_x + fx_x[::-1], y=upper + lower[::-1], fill="toself",
                             fillcolor="rgba(146,34,58,0.12)", line=dict(color="rgba(0,0,0,0)"),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=hist_x, y=[float(v) for v in hist["total_incidentes"]], mode="lines",
                             line=dict(color=C_ORANGE, width=2.5), showlegend=False))
    fig.add_trace(go.Scatter(x=fx_x, y=fx_y, mode="lines+markers",
                             line=dict(color=C_CRIMSON, width=3, dash="dot"),
                             marker=dict(size=5, color=C_CRIMSON), showlegend=False))
    fig.update_layout(
        font=dict(family=FONT, size=13, color=C_MUTE), plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=34, r=12, t=6, b=18), height=300, width=2340,
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=12), showline=True, linecolor=C_GRID)
    fig.update_yaxes(showgrid=True, gridcolor=C_GRID, zeroline=False, tickfont=dict(size=12), nticks=4)
    fig.write_image(path, scale=1)
    return fc


def build_panel_volume(meta, model_d1, path):
    """Area de volume previsto D+1..D+14 (4.70 x 2.55 in) para o slide 11 (com titulo)."""
    fc = recursive_forecast(model_d1, meta["history_recent"], meta["last_date"], n_days=14)
    xs = [f"D+{i+1}" for i in range(len(fc))]
    ys = [p["forecast"] for p in fc]
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="lines+markers+text", fill="tozeroy",
        fillcolor="rgba(67,56,202,0.18)", line=dict(color=C_INDIGO, width=2.5),
        marker=dict(size=5, color=C_INDIGO),
        text=[f"{v:.0f}" for v in ys], textposition="top center", textfont=dict(size=10, color=C_INK),
    ))
    fig.update_layout(
        title=dict(text="Volume previsto (D+1..D+14)", x=0.5, xanchor="center",
                   font=dict(family=FONT, size=15, color=C_INK)),
        font=dict(family=FONT, size=12, color=C_MUTE), plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=36, r=14, t=40, b=24), height=510, width=940, showlegend=False,
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10.5))
    fig.update_yaxes(showgrid=True, gridcolor=C_GRID, zeroline=False, rangemode="tozero", nticks=5)
    fig.write_image(path, scale=1)


def build_panel_risk(kpi, path):
    """Barras horizontais de risco de OLA por equipe (2.95 x 2.55 in) para o slide 11 (com titulo)."""
    tr = kpi.groupby("Grupo designado")["ola_violada"].agg(["mean", "count"])
    tr = tr[tr["count"] >= 200].sort_values("mean").tail(5)  # top-5 por taxa, asc p/ maior no topo
    teams = tr.index.tolist()
    rates = (tr["mean"] * 100).round(1).tolist()
    fig = go.Figure(go.Bar(
        x=rates, y=teams, orientation="h", marker_color=C_RED,
        text=[f"{r:.1f}%" for r in rates], textposition="outside",
        textfont=dict(size=11, color=C_INK), cliponaxis=False,
    ))
    fig.update_layout(
        title=dict(text="Risco de OLA por equipe (%)", x=0.5, xanchor="center",
                   font=dict(family=FONT, size=15, color=C_INK)),
        font=dict(family=FONT, size=12, color=C_MUTE), plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=8, r=40, t=40, b=22), height=510, width=590, showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=C_GRID, zeroline=False, range=[0, max(rates) * 1.25])
    fig.update_yaxes(tickfont=dict(size=11.5, color=C_INK))
    fig.write_image(path, scale=1)


# --------------------------------------------------------------------------- #
# 2) Edicao do PPTX (copia identica da Sprint 2)
# --------------------------------------------------------------------------- #
def shp(slide, sid):
    for s in slide.shapes:
        if s.shape_id == sid:
            return s
    raise KeyError(f"shape id={sid} nao encontrado")


def set_run0(shape, text):
    """Substitui o texto do 1o run preservando formatacao; zera runs extras do paragrafo 0."""
    para = shape.text_frame.paragraphs[0]
    if para.runs:
        para.runs[0].text = text
        for r in para.runs[1:]:
            r.text = ""
    else:
        para.add_run().text = text


def set_paras(shape, texts):
    """Ajusta texto por paragrafo (preserva bullets/formatacao); zera paragrafos sobrando."""
    paras = shape.text_frame.paragraphs
    for i, para in enumerate(paras):
        if i < len(texts):
            if para.runs:
                para.runs[0].text = texts[i]
                for r in para.runs[1:]:
                    r.text = ""
            else:
                para.add_run().text = texts[i]
        else:
            for r in para.runs:
                r.text = ""


def replace_chart_with_image(slide, sid, img_path):
    s = shp(slide, sid)
    L, T, W, H = s.left, s.top, s.width, s.height
    el = s._element
    el.getparent().remove(el)
    slide.shapes.add_picture(img_path, L, T, W, H)


# --------------------------------------------------------------------------- #
# 3) Slides de EVIDÊNCIA (código-fonte + prints do dashboard em execução)
#    Criados duplicando o cabeçalho/rodapé de um slide-modelo (badge + kicker +
#    título herdados = template idêntico) e preenchendo com conteúdo novo.
# --------------------------------------------------------------------------- #
_RNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _relink_images(src_part, dest_part, tree):
    for el in tree.iter():
        for name, val in list(el.attrib.items()):
            if name.startswith("{" + _RNS + "}") and val in src_part.rels:
                rel = src_part.rels[val]
                if rel.is_external:
                    new = dest_part.rels.get_or_add_ext_rel(rel.reltype, rel.target_ref)
                else:
                    new = dest_part.relate_to(rel.target_part, rel.reltype)
                el.set(name, new)


def _new_slide_from(prs, src_index):
    """Novo slide que herda só o cabeçalho (badge/kicker/título) e o rodapé do modelo."""
    src = prs.slides[src_index]
    dest = prs.slides.add_slide(src.slide_layout)
    for sh in list(dest.shapes):
        sh._element.getparent().remove(sh._element)
    thresh = int(6.8 * 914400)  # rodapé fica em y > 6.8in
    for sh in src.shapes:
        if sh.shape_id in (2, 3, 4, 5) or (sh.top is not None and sh.top > thresh):
            dest.shapes._spTree.append(copy.deepcopy(sh._element))
    _relink_images(src.part, dest.part, dest.shapes._spTree)
    return dest


def _set_kt(slide, kicker, title):
    set_run0(shp(slide, 4), kicker)
    set_run0(shp(slide, 5), title)


def _codebox(slide, code, l, t, w, h, size=11):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    box.fill.solid(); box.fill.fore_color.rgb = _rgb("12152E")
    box.line.fill.background(); box.shadow.inherit = False
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Pt(12); tf.margin_right = Pt(8); tf.margin_top = Pt(10); tf.margin_bottom = Pt(8)
    for i, ln in enumerate(code.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT; p.space_after = Pt(0)
        r = p.add_run(); r.text = ln if ln else " "
        r.font.name = "Consolas"; r.font.size = Pt(size)
        r.font.color.rgb = _rgb("8FD69C") if ln.strip().startswith("#") else _rgb("E6EDF3")
    return box


def _bullets(slide, items, l, t, w, h, size=13.5, accent="4338CA"):
    tf = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(8)
        r = p.add_run(); r.text = "▸ "; r.font.size = Pt(size); r.font.bold = True
        r.font.color.rgb = _rgb(accent); r.font.name = "Trebuchet MS"
        r2 = p.add_run(); r2.text = it; r2.font.size = Pt(size)
        r2.font.color.rgb = _rgb("1E2236"); r2.font.name = "Calibri"


def _callout(slide, title, items, l, t, w, h, accent="F23237"):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    box.fill.solid(); box.fill.fore_color.rgb = _rgb("F5F6FB")
    box.line.fill.background(); box.shadow.inherit = False
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Pt(14); tf.margin_top = Pt(12); tf.margin_right = Pt(12)
    p0 = tf.paragraphs[0]; p0.alignment = PP_ALIGN.LEFT
    r = p0.add_run(); r.text = title
    r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = _rgb(accent); r.font.name = "Trebuchet MS"
    p0.space_after = Pt(9)
    for it in items:
        p = tf.add_paragraph(); p.space_after = Pt(7)
        r = p.add_run(); r.text = "▸ " + it; r.font.size = Pt(12.5)
        r.font.color.rgb = _rgb("1E2236"); r.font.name = "Calibri"


def _stat(slide, l, t, w, value, label, color):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(1.2))
    box.fill.solid(); box.fill.fore_color.rgb = _rgb("F5F6FB")
    box.line.fill.background(); box.shadow.inherit = False
    tf = box.text_frame; tf.margin_left = Pt(13); tf.margin_top = Pt(10)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = value
    r.font.size = Pt(25); r.font.bold = True; r.font.color.rgb = _rgb(color); r.font.name = "Trebuchet MS"
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.LEFT
    r2 = p2.add_run(); r2.text = label
    r2.font.size = Pt(10.5); r2.font.color.rgb = _rgb("6B7088"); r2.font.name = "Calibri"


def _caption(slide, text, l, t, w):
    tf = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(0.35)).text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = text
    r.font.size = Pt(10); r.font.italic = True; r.font.color.rgb = _rgb("6B7088"); r.font.name = "Calibri"


def _evolution_card(slide, l, w, color, sprint, label, status, status_color, bullets):
    y, h = 2.55, 3.55
    bg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(y), Inches(w), Inches(h))
    bg.fill.solid(); bg.fill.fore_color.rgb = _rgb("F5F6FB")
    bg.line.fill.background(); bg.shadow.inherit = False
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(y), Inches(w), Inches(0.11))
    bar.fill.solid(); bar.fill.fore_color.rgb = _rgb(color)
    bar.line.fill.background(); bar.shadow.inherit = False
    head = slide.shapes.add_textbox(Inches(l + 0.24), Inches(y + 0.26), Inches(w - 0.45), Inches(1.0)).text_frame
    head.word_wrap = True
    p = head.paragraphs[0]; r = p.add_run(); r.text = sprint
    r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = _rgb(color); r.font.name = "Trebuchet MS"
    p2 = head.add_paragraph(); p2.space_before = Pt(1)
    r2 = p2.add_run(); r2.text = label
    r2.font.size = Pt(13); r2.font.bold = True; r2.font.color.rgb = _rgb("1E2236"); r2.font.name = "Trebuchet MS"
    p3 = head.add_paragraph(); p3.space_before = Pt(3)
    r3 = p3.add_run(); r3.text = status
    r3.font.size = Pt(10.5); r3.font.bold = True; r3.font.color.rgb = _rgb(status_color); r3.font.name = "Calibri"
    _bullets(slide, bullets, l + 0.24, y + 1.55, w - 0.46, h - 1.7, size=11, accent=color)


def rebuild_evolution_slide(slide):
    """Slide 5 (Proposta e evolução): 2 cards (S1, S2) -> 3 cards (S1, S2, S3-MVP)."""
    for sid in (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17):
        try:
            el = shp(slide, sid)._element
            el.getparent().remove(el)
        except KeyError:
            pass
    gap = 0.3
    w = (12.05 - 2 * gap) / 3.0
    x1 = 0.70
    x2 = x1 + w + gap
    x3 = x2 + w + gap
    _evolution_card(
        slide, x1, w, "6B7088", "Sprint 1", "Ideação", "✓ concluída", "2E8B57",
        ["Contagem de incidentes por período.",
         "Média e tendência simples.",
         "Previsões básicas ('alta chance amanhã').",
         "Sem arquitetura nem tratamento de OLA."])
    _evolution_card(
        slide, x2, w, "2E75B6", "Sprint 2", "Arquitetura", "✓ concluída", "2E8B57",
        ["Arquitetura definida (FastAPI, PostgreSQL, Plotly, n8n).",
         "Protótipos do painel e dos alertas.",
         "Escopo: previsão D+1/D+7 + risco de OLA."])
    _evolution_card(
        slide, x3, w, "C00000", "Sprint 3", "MVP funcional", "● entrega atual", "C00000",
        ["Pipeline real: 122.543 → 25.600 elegíveis.",
         "Modelos treinados: R²=0,58 (volume) · AUC=0,82 (OLA).",
         "API FastAPI + painel didático (código rodando).",
         "Evidências de construção validadas."])
    # atualiza a legenda (id=18)
    set_run0(shp(slide, 18),
             "O escopo evoluiu da análise descritiva (Sprint 1) à arquitetura (Sprint 2) "
             "e agora ao MVP funcional com código-fonte validado (Sprint 3).")


def _pagenum_all(prs):
    th = int(6.8 * 914400); lt = int(9 * 914400)
    for i, slide in enumerate(prs.slides, 1):
        for sh in slide.shapes:
            if (sh.top is not None and sh.left is not None and sh.top > th
                    and sh.left > lt and sh.has_text_frame):
                set_run0(sh, f"{i:02d}")
                break


CODE_PIPELINE = (
    "# src/data_pipeline.py  (Cap. 9)\n"
    "df = pd.read_excel(DATA_PATH)      # 122.543 chamados\n"
    "mask = (df['Entrou para KPI?']\n"
    "        .str.upper().str.contains('SIM'))\n"
    "df_kpi = df[mask].copy()           # 25.600 elegiveis\n"
    "\n"
    "# engenharia de features temporais\n"
    "daily['lag_1'] = daily['total'].shift(1)\n"
    "daily['lag_7'] = daily['total'].shift(7)\n"
    "daily['rolling_mean_7'] = (\n"
    "    daily['total'].shift(1).rolling(7).mean())\n"
    "daily['target_d1'] = daily['total'].shift(-1)\n"
    "daily['target_d7'] = daily['total'].shift(-7)"
)
CODE_MODEL = (
    "# src/model_training.py  (Cap. 10-12)\n"
    "# Regressao de volume (validacao temporal)\n"
    "rf = RandomForestRegressor(\n"
    "        n_estimators=300, max_depth=12)\n"
    "rf.fit(X_tr, y_tr)\n"
    "r2 = r2_score(y_te, rf.predict(X_te))   # 0.58\n"
    "\n"
    "# Classificacao de OLA (~1% positivos)\n"
    "clf = RandomForestClassifier(\n"
    "        class_weight='balanced_subsample')\n"
    "clf.fit(X_tr, y_tr)\n"
    "auc = roc_auc_score(y_te,\n"
    "        clf.predict_proba(X_te)[:, 1])   # 0.82"
)
CODE_API = (
    "# src/app_api.py  (FastAPI + Pydantic v2)\n"
    "class OlaRiskRequest(BaseModel):\n"
    "    prioridade: str\n"
    "    grupo_designado: str\n"
    "    hour: int = Field(ge=0, le=23)\n"
    "\n"
    "@app.post('/api/v1/predict/ola-risk')\n"
    "def predict_ola_risk(req: OlaRiskRequest):\n"
    "    p = model.predict_proba(row)[0, 1]\n"
    "    return {'violation_probability': p,\n"
    "            'risk_level': nivel(p)}"
)
CODE_API_IO = (
    "# POST /api/v1/predict/ola-risk\n"
    "{\n"
    "  'prioridade': '2 - Alta',\n"
    "  'produto': 'lhco', 'categoria': 'cat71',\n"
    "  'grupo_designado': 'Team11',\n"
    "  'hour': 14, 'day_of_week': 0,\n"
    "  'month': 8, 'is_weekend': 0\n"
    "}\n"
    "# --> 200 OK\n"
    "{\n"
    "  'violation_probability': 0.3246,\n"
    "  'risk_level': 'Medio',\n"
    "  'recommendation': 'Priorizar triagem\n"
    "     e acompanhar o prazo de OLA.'\n"
    "}"
)
CODE_DOCKER = (
    "# Dockerfile (resumo)\n"
    "FROM python:3.12-slim\n"
    "COPY requirements.txt .\n"
    "RUN pip install -r requirements.txt\n"
    "COPY src/ ./src/\n"
    "COPY data/LW-DATASET.xlsx ./data/\n"
    "RUN python src/data_pipeline.py && \\\n"
    "    python src/model_training.py\n"
    "CMD [\"uvicorn\", \"src.app_api:app\",\n"
    "  \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n"
    "\n"
    "# build + run\n"
    "docker build -t alertopslite:sprint3 .\n"
    "docker run -p 8000:8000 alertopslite:sprint3"
)


def build_evidence_slides(prs, metrics):
    rd1 = metrics["volume_d1"]["candidates"][metrics["volume_d1"]["best_model"]]
    rd7 = metrics["volume_d7"]["candidates"][metrics["volume_d7"]["best_model"]]
    ola = metrics["ola_risk"]["candidates"][metrics["ola_risk"]["best_model"]]
    cards = metrics["kpi_cards"]
    risk_txt = f"{cards['ola_risk_index_pct']:.2f}%".replace(".", ",")

    # E1 — Engenharia de dados (modelo: slide 6 DADOS teal)
    s = _new_slide_from(prs, 5)
    _set_kt(s, "MVP · DADOS", "Engenharia de dados (Cap. 9)")
    _bullets(s, [
        "Fonte real: LW-DATASET.xlsx — 122.543 chamados.",
        "Filtragem de ruído → 25.600 incidentes elegíveis (P2/P3).",
        "Alvo de violação de OLA derivado de 'KPI Violado?'.",
        "Features: lags D-1/D-7/D-14 e médias móveis de 7/14 dias.",
        "Série diária contínua e reindexada: 1.081 dias.",
    ], 0.6, 1.8, 5.6, 4.6, accent="0EA5A4")
    _codebox(s, CODE_PIPELINE, 6.45, 1.72, 6.25, 4.9)

    # E2 — Modelagem (modelo: slide 10 MODELAGEM crimson)
    s = _new_slide_from(prs, 9)
    _set_kt(s, "MVP · MODELAGEM", "Modelagem preditiva — código e validação (Cap. 10–12)")
    _bullets(s, [
        "Regressão de volume D+1/D+7: Linear, Ridge e RandomForest.",
        f"Melhor: RandomForest — R²={rd1['r2']:.2f} (D+1) e {rd7['r2']:.2f} (D+7).",
        "Classificador de OLA com class_weight balanceado (~1% positivos).",
        f"ROC-AUC={ola['roc_auc']:.2f} · recall={ola['recall']:.2f} (dados de teste).",
        "Modelos serializados (.joblib) e consumidos pela API.",
    ], 0.6, 1.8, 5.6, 4.6, accent="92223A")
    _codebox(s, CODE_MODEL, 6.45, 1.72, 6.25, 4.9)

    # E3 — Backend FastAPI (modelo: slide 8 ARQUITETURA dark)
    s = _new_slide_from(prs, 7)
    _set_kt(s, "MVP · BACKEND", "Backend FastAPI — código e endpoints")
    _bullets(s, [
        "API REST modular com validação Pydantic v2.",
        "Modelos carregados uma vez no startup (lifespan).",
        "Documentação Swagger automática em /docs.",
    ], 0.6, 1.8, 5.6, 2.1, accent="4338CA")
    _callout(s, "Endpoints", [
        "POST /api/v1/predict/volume",
        "POST /api/v1/predict/ola-risk",
        "GET  /api/v1/metrics",
        "GET  /api/v1/health",
    ], 0.6, 3.95, 5.6, 2.5, accent="4338CA")
    _codebox(s, CODE_API, 6.45, 1.72, 6.25, 4.9)

    # E4 — Painel em execução (modelo: slide 11 PROTÓTIPOS red)
    s = _new_slide_from(prs, 10)
    _set_kt(s, "MVP · EXECUÇÃO", "Painel operacional em execução (amostra real)")
    _pic(s, os.path.join(IMG, "dash_exec_help.png"), 0.55, 1.7, w=7.05)
    _caption(s, "Print real do dashboard.html em execução — ajuda do gráfico aberta.", 0.55, 5.42, 7.05)
    _callout(s, "O que a evidência mostra", [
        "Painel HTML real (Plotly), gerado do dataset real.",
        f"KPIs reais: D+1={cards['volume_d1']:.0f} · D+7={cards['volume_d7']:.0f} · risco {risk_txt}.",
        "Onboarding 'Como usar' + legenda de cores para leigos.",
        "Cada gráfico tem ajuda (ⓘ): o que é, de onde vem, como ler.",
        "Tooltips mostram o valor exato ao passar o mouse.",
    ], 7.8, 1.7, 4.95, 4.5, accent="F23237")

    # E5 — Resultados & hipóteses (modelo: slide 6 DADOS teal)
    s = _new_slide_from(prs, 5)
    _set_kt(s, "MVP · RESULTADOS", "Resultados & validação de hipóteses")
    _stat(s, 0.6, 1.75, 2.9, f"{cards['volume_d1']:.0f}", "Volume previsto D+1", "4338CA")
    _stat(s, 3.65, 1.75, 2.9, f"{cards['volume_d7']:.0f}", "Volume previsto D+7", "0EA5A4")
    _stat(s, 6.7, 1.75, 2.9, f"{ola['roc_auc']:.2f}", "ROC-AUC risco de OLA", "92223A")
    _stat(s, 9.75, 1.75, 2.95, "25.600", "Incidentes elegíveis", "E8991C")
    _bullets(s, [
        "H1 — Volume é previsível: confirmada (R²=0,58, validação temporal).",
        "H2 — Risco de OLA é detectável: confirmada (AUC=0,82, ~1% positivos).",
        "H3 — Incidentes se concentram: confirmada (Pareto 80/20).",
        "Ganho: antecipar picos e priorizar incidentes de alto risco.",
    ], 0.6, 3.35, 7.35, 3.2, size=13, accent="0EA5A4")
    _pic(s, os.path.join(IMG, "dash_exec_charts.png"), 8.2, 3.2, w=4.55)

    # E6 — Código-fonte no GitHub (modelo: slide 8 dark)
    s = _new_slide_from(prs, 7)
    _set_kt(s, "MVP · REPOSITÓRIO", "Código-fonte público no GitHub")
    _pic(s, os.path.join(IMG, "qr_github.png"), 1.35, 2.55, w=2.7)
    _caption(s, "Aponte a câmera do celular", 1.35, 5.35, 2.7)
    tb = s.shapes.add_textbox(Inches(0.7), Inches(5.75), Inches(4.0), Inches(0.5)).text_frame
    tb.word_wrap = True
    pr = tb.paragraphs[0]; pr.alignment = PP_ALIGN.CENTER
    rr = pr.add_run(); rr.text = "github.com/matmachado-hub/AlertOpsLite"
    rr.font.size = Pt(12.5); rr.font.bold = True; rr.font.color.rgb = _rgb("4338CA"); rr.font.name = "Consolas"
    _callout(s, "O repositório entrega", [
        "Projeto aplicado autocontido (git + Docker ready).",
        "Estrutura: src/, builder.py, Dockerfile, README.",
        "Público — clone e execução reprodutível.",
        "git clone → docker run → API no ar em minutos.",
    ], 5.2, 2.55, 7.55, 3.7, accent="4338CA")

    # E7 — API em funcionamento (modelo: slide 8 dark)
    s = _new_slide_from(prs, 7)
    _set_kt(s, "MVP · API", "API em funcionamento — predições reais")
    _bullets(s, [
        "4 endpoints REST testados em /api/v1.",
        "Documentação Swagger interativa em /docs.",
        "Modelos .joblib servidos via FastAPI (Pydantic v2).",
        "Exemplo real de requisição e resposta ao lado →",
    ], 0.6, 1.8, 5.6, 2.4, accent="4338CA")
    _callout(s, "Saídas reais validadas", [
        "predict/volume → D+1=43.6 · D+7=98.7",
        "predict/ola-risk (Team11 P2) → 'Médio' (0,32)",
        "health → ok · metrics → R²/AUC do relatório",
    ], 0.6, 4.1, 5.6, 2.35, accent="0EA5A4")
    _codebox(s, CODE_API_IO, 6.45, 1.72, 6.25, 4.9, size=10.5)

    # E8 — Empacotamento & Docker (modelo: slide 8 dark)
    s = _new_slide_from(prs, 7)
    _set_kt(s, "MVP · ENTREGA", "Empacotamento, reprodutibilidade e Docker")
    _bullets(s, [
        "builder.py: 1 comando regenera dados → modelos → painel → PPTX.",
        "Dockerfile: treina no build e sobe a API (porta 8000).",
        "requirements.txt fixado + README com passo a passo.",
        "Base pronta para o deploy da Sprint 4 (VPS Hostinger).",
    ], 0.6, 1.8, 5.6, 3.4, accent="4338CA")
    _codebox(s, CODE_DOCKER, 6.45, 1.72, 6.25, 4.9, size=10.5)

    # reordena: move os 8 novos slides para logo após o slide 11 (posições 11..18)
    lst = prs.slides._sldIdLst
    block = list(lst)[-8:]
    for el in block:
        lst.remove(el)
    for i, el in enumerate(block):
        lst.insert(11 + i, el)

    _pagenum_all(prs)


def _pic(slide, path, l, t, w=None, h=None):
    if os.path.exists(path):
        slide.shapes.add_picture(path, Inches(l), Inches(t),
                                 Inches(w) if w else None, Inches(h) if h else None)


def main():
    daily, kpi, metrics, meta, model_d1 = _load_real()
    resid = metrics["volume_d1"]["resid_std"]
    rd1 = metrics["volume_d1"]["candidates"][metrics["volume_d1"]["best_model"]]
    rd7 = metrics["volume_d7"]["candidates"][metrics["volume_d7"]["best_model"]]
    ola = metrics["ola_risk"]["candidates"][metrics["ola_risk"]["best_model"]]
    cards = metrics["kpi_cards"]

    os.makedirs(IMG, exist_ok=True)
    _ensure_qr()
    strip = os.path.join(IMG, "slide_forecast_strip.png")
    pvol = os.path.join(IMG, "slide_panel_volume.png")
    prisk = os.path.join(IMG, "slide_panel_risk.png")
    print("[S3] Gerando imagens de evidencia real (fit-to-box)...")
    build_forecast_strip(daily, meta, model_d1, resid, strip)
    build_panel_volume(meta, model_d1, pvol)
    build_panel_risk(kpi, prisk)

    print(f"[S3] Copiando template exato: {SRC_PPTX} -> {OUT_PPTX}")
    shutil.copyfile(SRC_PPTX, OUT_PPTX)
    prs = Presentation(OUT_PPTX)
    sl = list(prs.slides)

    # ---- SLIDE 1 (capa): re-versiona ----
    s1 = sl[0]
    e = shp(s1, 8).text_frame.paragraphs[0]
    e.runs[0].text = "Sprint 3"
    e.runs[1].text = "  ·  Evidências de construção do MVP"

    # ---- SLIDE 10 (modelagem): mock -> real ----
    s10 = sl[9]
    set_run0(shp(s10, 5), "Modelagem preditiva — resultados")
    set_run0(shp(s10, 10), f"R²={rd1['r2']:.2f} (D+1) e {rd7['r2']:.2f} (D+7) · MAE≈{rd1['mae']:.0f}/dia (validação temporal).")
    set_run0(shp(s10, 15), f"RandomForest balanceado · ROC-AUC={ola['roc_auc']:.2f} · recall={ola['recall']:.2f}.")
    set_run0(shp(s10, 19), "Engenharia de dados")
    set_run0(shp(s10, 20), "122k → 25.600 elegíveis · lags D-1/D-7/D-14 e médias móveis.")
    set_run0(shp(s10, 24), "Validação")
    set_run0(shp(s10, 25), "Holdout temporal (20%) · métricas R², MAE, RMSE, F1 e AUC.")
    set_run0(shp(s10, 27), "Previsão de volume — resultado real (RandomForest, validação temporal)")
    replace_chart_with_image(s10, 29, strip)

    # ---- SLIDE 11 (prototipo painel): mock -> real ----
    s11 = sl[10]
    set_run0(shp(s11, 4), "EVIDÊNCIA")
    set_run0(shp(s11, 5), "Painel operacional — evidência real")
    set_run0(shp(s11, 14), f"{cards['volume_d1']:.0f}")
    set_run0(shp(s11, 18), f"{cards['volume_d7']:.0f}")
    set_run0(shp(s11, 22), f"{cards['ola_risk_index_pct']:.2f}%".replace(".", ","))
    set_run0(shp(s11, 23), "índice global de violação")
    set_paras(shp(s11, 28), [
        "Painel HTML (Plotly) + evidências exportadas em PNG.",
        "KPIs reais: volume D+1/D+7 e índice de risco de OLA.",
        "Série histórica + projeção com banda, Pareto e heatmap.",
        "Gerado do dataset real: 25.600 incidentes elegíveis.",
    ])
    replace_chart_with_image(s11, 24, pvol)
    replace_chart_with_image(s11, 25, prisk)

    # ---- Slide 5: Proposta e evolução -> inclui a Sprint 3 ----
    rebuild_evolution_slide(sl[4])

    # ---- Slides de EVIDÊNCIA (código-fonte + prints do dashboard em execução) ----
    build_evidence_slides(prs, metrics)

    prs.save(OUT_PPTX)
    total = len(prs.slides._sldIdLst)
    print(f"[S3] Concluido: {OUT_PPTX} ({total} slides = 15 base + 8 evidencias).")


if __name__ == "__main__":
    main()
