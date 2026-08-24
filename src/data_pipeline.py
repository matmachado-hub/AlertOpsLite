# -*- coding: utf-8 -*-
"""
AlertOpsLite - Pipeline de Engenharia e Tratamento de Dados
Challenge Locaweb - FIAP 2TSCOA (Grupo Irmaos)
Autores: Marcos Machado (RM566099) e Matheus Machado (RM564991)
Fundamentacao: Capitulos 1, 2, 9 e 10 (Data Viz e Data Science)
"""

import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join("data", "LW-DATASET.xlsx")
PROCESSED_DAILY_PATH = os.path.join("data", "daily_incidents_processed.csv")
PROCESSED_INCIDENTS_PATH = os.path.join("data", "kpi_incidents_processed.csv")

def load_raw_data(filepath=DATA_PATH):
    print(f"[DATA PIPELINE] Carregando dataset bruto de: {filepath}")
    df = pd.read_excel(filepath)
    print(f"[DATA PIPELINE] Registros carregados: {len(df):,} linhas e {len(df.columns)} colunas.")
    return df

def clean_and_filter_data(df):
    print("[DATA PIPELINE] Iniciando limpeza e filtragem de ruido...")
    df.columns = [str(c).strip() for c in df.columns]
    
    date_cols = ['Aberto', 'Resolvido', 'Encerrado']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    total_raw = len(df)
    
    # Identificar coluna de elegibilidade ao KPI
    kpi_cols = [c for c in df.columns if 'KPI' in c and ('Entrou' in c or 'entrou' in c)]
    kpi_col = kpi_cols[0] if kpi_cols else 'Entrou para KPI?'
    
    kpi_mask = df[kpi_col].astype(str).str.upper().str.contains('SIM|S')
    df_kpi = df[kpi_mask].copy()
    
    print(f"[DATA PIPELINE] Total bruto: {total_raw:,} | Elegiveis para KPI: {len(df_kpi):,} (Foco P2/P3)")
    
    # Target de Violacao de OLA
    viol_cols = [c for c in df.columns if 'KPI' in c and ('Violado' in c or 'violado' in c)]
    viol_col = viol_cols[0] if viol_cols else 'KPI Violado?'
    
    df_kpi['ola_violada'] = df_kpi[viol_col].astype(str).str.upper().apply(
        lambda x: 1 if ('SIM' in x or 'S' in x or 'TRUE' in x or '1' in x) else 0
    )
    
    # Preenchimento semantico
    df_kpi['Prioridade'] = df_kpi['Prioridade'].fillna('3 - Media')
    df_kpi['Produto'] = df_kpi['Produto'].fillna('Outros_Produtos')
    df_kpi['Categoria'] = df_kpi['Categoria'].fillna('Outras_Categorias')
    df_kpi['Grupo designado'] = df_kpi['Grupo designado'].fillna('Team_Default')
    
    # Features temporais dos incidentes individuais
    df_kpi['date'] = df_kpi['Aberto'].dt.date
    df_kpi['hour'] = df_kpi['Aberto'].dt.hour
    df_kpi['day_of_week'] = df_kpi['Aberto'].dt.dayofweek
    df_kpi['month'] = df_kpi['Aberto'].dt.month
    df_kpi['is_weekend'] = df_kpi['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df_kpi

def generate_daily_aggregated_features(df_kpi):
    print("[DATA PIPELINE] Gerando agregacao diaria de series temporais...")
    
    id_col = [c for c in df_kpi.columns if 'mero' in c or 'Numero' in c or 'N' in c][0]
    
    daily = df_kpi.groupby('date').agg(
        total_incidentes=(id_col, 'count'),
        incidentes_p2=('Prioridade', lambda x: (x.astype(str).str.contains('2')).sum()),
        incidentes_p3=('Prioridade', lambda x: (x.astype(str).str.contains('3')).sum()),
        total_violacoes=('ola_violada', 'sum')
    ).reset_index()
    
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)
    
    # Reindexar calendario continuo
    idx = pd.date_range(daily['date'].min(), daily['date'].max())
    daily = daily.set_index('date').reindex(idx, fill_value=0).reset_index()
    daily.rename(columns={'index': 'date'}, inplace=True)
    
    # Features de Calendario (Capitulo 10 e 11)
    daily['day_of_week'] = daily['date'].dt.dayofweek
    daily['day_of_month'] = daily['date'].dt.day
    daily['month'] = daily['date'].dt.month
    daily['is_weekend'] = daily['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Lags temporais
    daily['lag_1'] = daily['total_incidentes'].shift(1)
    daily['lag_2'] = daily['total_incidentes'].shift(2)
    daily['lag_7'] = daily['total_incidentes'].shift(7)
    daily['lag_14'] = daily['total_incidentes'].shift(14)
    
    # Medias moveis e desvio
    daily['rolling_mean_7'] = daily['total_incidentes'].shift(1).rolling(7).mean()
    daily['rolling_mean_14'] = daily['total_incidentes'].shift(1).rolling(14).mean()
    daily['rolling_std_7'] = daily['total_incidentes'].shift(1).rolling(7).std()
    
    # Targets futuros (D+1 e D+7)
    daily['target_d1'] = daily['total_incidentes'].shift(-1)
    daily['target_d7'] = daily['total_incidentes'].shift(-7)
    
    daily_clean = daily.dropna(subset=['lag_14', 'rolling_mean_14']).copy()
    print(f"[DATA PIPELINE] Agregacao diaria concluida: {len(daily_clean)} dias uteis estruturados.")
    return daily_clean

def run_pipeline():
    os.makedirs('data', exist_ok=True)
    os.makedirs('img', exist_ok=True)
    
    df_raw = load_raw_data()
    df_kpi = clean_and_filter_data(df_raw)
    daily_df = generate_daily_aggregated_features(df_kpi)
    
    df_kpi.to_csv(PROCESSED_INCIDENTS_PATH, index=False)
    daily_df.to_csv(PROCESSED_DAILY_PATH, index=False)
    
    print(f"[DATA PIPELINE] Arquivos salvos com sucesso em {PROCESSED_INCIDENTS_PATH} e {PROCESSED_DAILY_PATH}")
    return df_kpi, daily_df

if __name__ == '__main__':
    run_pipeline()
