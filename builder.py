# -*- coding: utf-8 -*-
"""
AlertOpsLite - Orquestrador End-to-End (Sprint 3)
Challenge Locaweb - FIAP 2TSCOA (Grupo Irmaos)

Executa o pipeline completo na ordem correta:
  1. src/data_pipeline.py    -> limpeza + feature engineering (data/*.csv)
  2. src/model_training.py   -> treino e serializacao dos modelos (models/*)
  3. src/dashboard.py        -> evidencias graficas (img/*.png) + dashboard.html
  4. generate_sprint3_pptx.py-> apresentacao oficial (.pptx)

Uso:
  python builder.py
"""

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("1/5 · Engenharia de dados (Cap. 9)", "src/data_pipeline.py"),
    ("2/5 · Treino dos modelos (Cap. 10-12)", "src/model_training.py"),
    ("3/5 · Painel & evidencias Data Viz (Cap. 1-6)", "src/dashboard.py"),
    # Prints do dashboard em execucao (requer Chrome/Edge; pulado se ausente)
    ("4/5 · Prints do dashboard (amostra de execucao)", "capture_dashboard.py"),
    # PPTX gerado a partir do template EXATO da Sprint 2 (mock -> evidencia real)
    ("5/5 · Apresentacao oficial (PPTX)", "build_sprint3_from_sprint2.py"),
]


def main():
    os.chdir(ROOT)
    print("=" * 64)
    print(" AlertOpsLite - Build da Sprint 3 (MVP)")
    print("=" * 64)
    for title, script in STEPS:
        print(f"\n>>> {title}")
        print("-" * 64)
        result = subprocess.run([sys.executable, script], cwd=ROOT)
        if result.returncode != 0:
            print(f"\n[ERRO] Falha em '{script}' (codigo {result.returncode}). Abortando.")
            sys.exit(result.returncode)
    print("\n" + "=" * 64)
    print(" Build concluido com sucesso!")
    print(" Artefatos: data/  models/  img/  dashboard.html  *.pptx")
    print("=" * 64)


if __name__ == "__main__":
    main()
