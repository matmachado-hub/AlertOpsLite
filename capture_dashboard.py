# -*- coding: utf-8 -*-
"""
AlertOpsLite - Captura de prints do dashboard em execucao
Challenge Locaweb - FIAP 2TSCOA (Grupo Irmaos)

Renderiza o dashboard.html em um navegador headless (Google Chrome ou Microsoft
Edge) e recorta as regioes usadas como EVIDENCIA no PPTX:
  img/dash_exec_overview.png  -> cabecalho + onboarding + KPIs
  img/dash_exec_help.png      -> card da serie com a AJUDA aberta (recurso didatico)
  img/dash_exec_charts.png    -> Pareto + Risco + Confusao + Qualidade

Requer Chrome/Edge instalado + Pillow. Se o navegador nao existir (ex.: container
Linux enxuto), o passo e PULADO sem quebrar o build.
"""

import os
import shutil
import subprocess
import tempfile
import pathlib

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, "img")
DASH = os.path.join(ROOT, "dashboard.html")

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
]


def _find_browser():
    for b in BROWSERS:
        if os.path.isfile(b):
            return b
        found = shutil.which(b)
        if found:
            return found
    return None


def _shot(browser, url, out, w, h):
    subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
         "--force-device-scale-factor=1.5", f"--window-size={w},{h}",
         "--virtual-time-budget=9000", f"--screenshot={out}", url],
        check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main():
    if not os.path.exists(DASH):
        print("[CAPTURE] dashboard.html nao encontrado; rode src/dashboard.py antes.")
        return
    browser = _find_browser()
    if not browser:
        print("[CAPTURE] Chrome/Edge nao encontrado; pulando prints (usa imagens existentes).")
        return
    try:
        from PIL import Image
    except ImportError:
        print("[CAPTURE] Pillow indisponivel; pulando recortes.")
        return

    os.makedirs(IMG, exist_ok=True)
    tmp = tempfile.mkdtemp()
    full_png = os.path.join(tmp, "full.png")
    help_png = os.path.join(tmp, "help.png")

    # variante com o painel de ajuda do 1o grafico (serie) ABERTO
    html = open(DASH, encoding="utf-8").read().replace(
        '<div class="help-panel">', '<div class="help-panel open">', 1)
    help_html = os.path.join(tmp, "dash_help.html")
    with open(help_html, "w", encoding="utf-8") as f:
        f.write(html)

    try:
        _shot(browser, pathlib.Path(DASH).as_uri(), full_png, 1360, 2960)
        _shot(browser, pathlib.Path(help_html).as_uri(), help_png, 1360, 1500)
        Image.open(full_png).crop((0, 0, 2040, 650)).save(os.path.join(IMG, "dash_exec_overview.png"))
        Image.open(full_png).crop((0, 1395, 2040, 2880)).save(os.path.join(IMG, "dash_exec_charts.png"))
        Image.open(help_png).crop((0, 655, 2040, 1690)).save(os.path.join(IMG, "dash_exec_help.png"))
        print("[CAPTURE] Prints salvos: img/dash_exec_overview.png, dash_exec_help.png, dash_exec_charts.png")
    except Exception as e:  # navegador falhou -> nao quebra o build
        print(f"[CAPTURE] Falha ao capturar ({type(e).__name__}); mantendo imagens existentes.")


if __name__ == "__main__":
    main()
