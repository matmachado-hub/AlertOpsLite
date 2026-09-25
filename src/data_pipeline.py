# -*- coding: utf-8 -*-
# ^ Diz ao Python que este arquivo está salvo em UTF-8 (permite acentos: ç, ã, é...).
"""
AlertOpsLite - Pipeline de Engenharia e Tratamento de Dados
Challenge Locaweb - FIAP 2TSCOA (Grupo Irmaos)
Autores: Marcos Machado (RM566099) e Matheus Machado (RM564991)
Fundamentacao: Capitulos 1, 2, 9 e 10 (Data Viz e Data Science)

O QUE ESTE ARQUIVO FAZ (visão para júnior):
  Ele é a "cozinha" dos dados. Pega a planilha bruta da Locaweb (122 mil chamados),
  joga fora o "lixo" (alertas automáticos que ninguém tratou), separa o que interessa
  (25.600 incidentes que contam para o KPI) e cria colunas novas ("features") que
  ajudam o modelo a aprender. No fim, salva dois CSVs prontos para o treino.
"""

import os               # 'os' = funções do sistema operacional (montar caminhos de arquivo, criar pastas)
import pandas as pd     # 'pandas' = biblioteca de tabelas (DataFrame). O apelido 'pd' é convenção do mercado.
import numpy as np      # 'numpy' = matemática com vetores/matrizes. Apelido 'np' (aqui usado indiretamente).

# --- Caminhos dos arquivos (constantes: MAIÚSCULAS por convenção) ---
# os.path.join junta pastas do jeito certo em qualquer sistema (Windows usa '\', Linux usa '/').
DATA_PATH = os.path.join("data", "LW-DATASET.xlsx")                        # entrada: planilha bruta
PROCESSED_DAILY_PATH = os.path.join("data", "daily_incidents_processed.csv")   # saída 1: série diária (para prever volume)
PROCESSED_INCIDENTS_PATH = os.path.join("data", "kpi_incidents_processed.csv")  # saída 2: incidentes (para prever risco de OLA)


def load_raw_data(filepath=DATA_PATH):
    """Etapa 1 — Ler a planilha Excel e devolver uma tabela (DataFrame)."""
    print(f"[DATA PIPELINE] Carregando dataset bruto de: {filepath}")  # log para acompanhar a execução
    df = pd.read_excel(filepath)   # lê o .xlsx inteiro e transforma numa tabela na memória
    # len(df) = nº de linhas; len(df.columns) = nº de colunas. O ':,' formata milhar (122,543).
    print(f"[DATA PIPELINE] Registros carregados: {len(df):,} linhas e {len(df.columns)} colunas.")
    return df                      # devolve a tabela para quem chamou a função


def clean_and_filter_data(df):
    """Etapa 2 — Limpar, filtrar o ruído e criar as features por incidente."""
    print("[DATA PIPELINE] Iniciando limpeza e filtragem de ruido...")

    # Remove espaços sobrando no NOME de cada coluna (ex.: ' Prioridade ' -> 'Prioridade').
    df.columns = [str(c).strip() for c in df.columns]

    # Converte as colunas de data (que vêm como texto) para o tipo "data/hora" de verdade.
    date_cols = ['Aberto', 'Resolvido', 'Encerrado']
    for col in date_cols:                     # percorre cada nome de coluna de data
        if col in df.columns:                 # só age se a coluna existir na planilha
            # to_datetime converte texto -> data. errors='coerce' = o que não der, vira "vazio" (NaT) em vez de quebrar.
            df[col] = pd.to_datetime(df[col], errors='coerce')

    total_raw = len(df)   # guarda o total bruto (antes de filtrar) para o log

    # --- Descobrir qual coluna diz se o chamado "Entrou para o KPI" ---
    # Procura uma coluna cujo nome tenha 'KPI' e 'Entrou' (robusto a pequenas variações de nome).
    kpi_cols = [c for c in df.columns if 'KPI' in c and ('Entrou' in c or 'entrou' in c)]
    kpi_col = kpi_cols[0] if kpi_cols else 'Entrou para KPI?'   # usa a 1ª achada; senão, um nome padrão

    # Cria uma "máscara" (lista de Verdadeiro/Falso): True onde o valor contém 'SIM'.
    # .astype(str) garante texto; .str.upper() deixa maiúsculo; .str.contains busca o padrão.
    kpi_mask = df[kpi_col].astype(str).str.upper().str.contains('SIM|S')
    df_kpi = df[kpi_mask].copy()   # fica só com as linhas elegíveis. .copy() evita avisos do pandas.

    print(f"[DATA PIPELINE] Total bruto: {total_raw:,} | Elegiveis para KPI: {len(df_kpi):,} (Foco P2/P3)")

    # --- Criar o "alvo" (target) que o modelo de risco vai aprender a prever ---
    # Descobre a coluna que informa se a OLA foi violada.
    viol_cols = [c for c in df.columns if 'KPI' in c and ('Violado' in c or 'violado' in c)]
    viol_col = viol_cols[0] if viol_cols else 'KPI Violado?'

    # Transforma o texto ('SIM'/'NAO') em número: 1 = violou a OLA, 0 = não violou.
    # Modelos de ML só entendem números, por isso essa conversão é obrigatória.
    df_kpi['ola_violada'] = df_kpi[viol_col].astype(str).str.upper().apply(
        lambda x: 1 if ('SIM' in x or 'S' in x or 'TRUE' in x or '1' in x) else 0
    )

    # --- Preencher valores vazios (NaN) com um rótulo padrão, para não perder linhas ---
    # fillna troca "vazio" por um valor. Assim nenhum incidente é descartado por falta de categoria.
    df_kpi['Prioridade'] = df_kpi['Prioridade'].fillna('3 - Media')
    df_kpi['Produto'] = df_kpi['Produto'].fillna('Outros_Produtos')
    df_kpi['Categoria'] = df_kpi['Categoria'].fillna('Outras_Categorias')
    df_kpi['Grupo designado'] = df_kpi['Grupo designado'].fillna('Team_Default')

    # --- Features temporais de CADA incidente (extraídas da data de abertura) ---
    # .dt dá acesso a partes da data. Essas colunas ajudam o modelo a captar padrões por horário/dia.
    df_kpi['date'] = df_kpi['Aberto'].dt.date            # só a data (sem a hora)
    df_kpi['hour'] = df_kpi['Aberto'].dt.hour            # hora do dia (0 a 23)
    df_kpi['day_of_week'] = df_kpi['Aberto'].dt.dayofweek  # dia da semana (0=segunda ... 6=domingo)
    df_kpi['month'] = df_kpi['Aberto'].dt.month          # mês (1 a 12)
    # is_weekend = 1 se for sábado(5) ou domingo(6); senão 0. Fim de semana costuma ter menos incidentes.
    df_kpi['is_weekend'] = df_kpi['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    return df_kpi   # devolve a tabela limpa e enriquecida (nível "incidente")


def generate_daily_aggregated_features(df_kpi):
    """Etapa 3 — Agrupar por DIA e criar features de série temporal (para prever o volume)."""
    print("[DATA PIPELINE] Gerando agregacao diaria de series temporais...")

    # Acha a coluna de identificador do incidente (o "Número"), usada só para CONTAR quantos por dia.
    id_col = [c for c in df_kpi.columns if 'mero' in c or 'Numero' in c or 'N' in c][0]

    # groupby('date').agg(...) = "para cada dia, calcule estes resumos":
    daily = df_kpi.groupby('date').agg(
        total_incidentes=(id_col, 'count'),   # quantos incidentes no dia (conta as linhas)
        # incidentes de prioridade 2 (Alta): soma quantos têm '2' no texto da prioridade
        incidentes_p2=('Prioridade', lambda x: (x.astype(str).str.contains('2')).sum()),
        # incidentes de prioridade 3 (Média)
        incidentes_p3=('Prioridade', lambda x: (x.astype(str).str.contains('3')).sum()),
        total_violacoes=('ola_violada', 'sum')  # quantas violações de OLA no dia
    ).reset_index()   # reset_index transforma o índice 'date' de volta em coluna normal

    daily['date'] = pd.to_datetime(daily['date'])                 # garante tipo data
    daily = daily.sort_values('date').reset_index(drop=True)      # ordena do dia mais antigo ao mais novo

    # --- Calendário contínuo: preencher dias que não tiveram nenhum incidente ---
    # date_range cria TODAS as datas entre a mínima e a máxima (sem "buracos").
    idx = pd.date_range(daily['date'].min(), daily['date'].max())
    # reindex encaixa a tabela nesse calendário; dias faltantes entram com 0 (fill_value=0).
    daily = daily.set_index('date').reindex(idx, fill_value=0).reset_index()
    daily.rename(columns={'index': 'date'}, inplace=True)         # renomeia a coluna do calendário para 'date'

    # --- Features de calendário (o modelo aprende sazonalidade com elas) ---
    daily['day_of_week'] = daily['date'].dt.dayofweek   # dia da semana
    daily['day_of_month'] = daily['date'].dt.day        # dia do mês
    daily['month'] = daily['date'].dt.month             # mês
    daily['is_weekend'] = daily['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # --- LAGS: "quantos incidentes houve N dias atrás?" ---
    # shift(N) desloca a coluna para baixo N linhas, trazendo o valor do passado para a linha de hoje.
    # Isso dá "memória" ao modelo: ontem e a semana passada ajudam a prever amanhã.
    daily['lag_1'] = daily['total_incidentes'].shift(1)    # 1 dia atrás
    daily['lag_2'] = daily['total_incidentes'].shift(2)    # 2 dias atrás
    daily['lag_7'] = daily['total_incidentes'].shift(7)    # 7 dias atrás (mesmo dia da semana passada)
    daily['lag_14'] = daily['total_incidentes'].shift(14)  # 14 dias atrás

    # --- MÉDIAS MÓVEIS: tendência recente (suaviza o "sobe e desce" diário) ---
    # .shift(1) antes do rolling garante usar só o PASSADO (não "espiar" o dia de hoje) — evita vazamento.
    daily['rolling_mean_7'] = daily['total_incidentes'].shift(1).rolling(7).mean()    # média dos últimos 7 dias
    daily['rolling_mean_14'] = daily['total_incidentes'].shift(1).rolling(14).mean()  # média dos últimos 14 dias
    daily['rolling_std_7'] = daily['total_incidentes'].shift(1).rolling(7).std()      # variabilidade (desvio-padrão) 7 dias

    # --- TARGETS (o que queremos prever): valores do FUTURO ---
    # shift(-N) traz o valor de N dias À FRENTE para a linha de hoje. É a "resposta certa" do treino.
    daily['target_d1'] = daily['total_incidentes'].shift(-1)   # volume de amanhã (D+1)
    daily['target_d7'] = daily['total_incidentes'].shift(-7)   # volume daqui a 7 dias (D+7)

    # Os primeiros dias não têm lag_14/média_14 (viram NaN). dropna remove essas linhas incompletas.
    daily_clean = daily.dropna(subset=['lag_14', 'rolling_mean_14']).copy()
    print(f"[DATA PIPELINE] Agregacao diaria concluida: {len(daily_clean)} dias uteis estruturados.")
    return daily_clean


def run_pipeline():
    """Orquestra as 3 etapas e salva os CSVs de saída."""
    os.makedirs('data', exist_ok=True)   # cria a pasta 'data' se não existir (exist_ok evita erro se já existe)
    os.makedirs('img', exist_ok=True)    # cria a pasta 'img'

    df_raw = load_raw_data()                                   # etapa 1: carregar
    df_kpi = clean_and_filter_data(df_raw)                     # etapa 2: limpar/filtrar
    daily_df = generate_daily_aggregated_features(df_kpi)      # etapa 3: agregar por dia

    df_kpi.to_csv(PROCESSED_INCIDENTS_PATH, index=False)      # salva incidentes (index=False: não grava a numeração das linhas)
    daily_df.to_csv(PROCESSED_DAILY_PATH, index=False)        # salva série diária

    print(f"[DATA PIPELINE] Arquivos salvos com sucesso em {PROCESSED_INCIDENTS_PATH} e {PROCESSED_DAILY_PATH}")
    return df_kpi, daily_df


# Este bloco só roda quando você executa 'python src/data_pipeline.py' diretamente.
# Se outro arquivo apenas IMPORTAR este, o run_pipeline() NÃO dispara sozinho.
if __name__ == '__main__':
    run_pipeline()
