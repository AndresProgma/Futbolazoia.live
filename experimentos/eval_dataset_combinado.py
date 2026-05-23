"""
Prueba del dataset combinado: UCL + Premier League 24-25 + 25-26.

Compara accuracy walk-forward CV en:
  A) Solo UCL (baseline)
  B) Dataset combinado (UCL + PL)
  C) Dataset combinado con feature Liga

Uso:
    python experimentos/eval_dataset_combinado.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.knime_workflow_converter import (
    compute_uefa_coef_features,
    compute_elo_features,
    compute_form_features,
    compute_h2h_features,
    compute_xg_features,
    compute_media11_rolling_features,
    select_columns,
    handle_missing_values,
    create_derived_variables,
    filter_rows,
    aggregate_by_team,
    join_team_stats,
    prepare_for_modeling,
    cross_validate_models,
)


def cargar_y_preparar(df):
    """Aplica el pipeline completo a un DataFrame ya cargado."""
    # Orden cronológico
    if 'Fecha' in df.columns:
        df['_fecha_dt'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.sort_values('_fecha_dt', na_position='last').reset_index(drop=True)
        df = df.drop(columns=['_fecha_dt'])

    # Sede neutral para finales UCL
    if 'Fase' in df.columns and 'Es_Local_E1' in df.columns:
        mask = df['Fase'].astype(str).str.strip().str.lower() == 'final'
        df.loc[mask, 'Es_Local_E1'] = 0

    # Features pre-partido
    df = compute_uefa_coef_features(df)
    df, _ = compute_elo_features(df)
    df, _ = compute_form_features(df)
    df, _ = compute_h2h_features(df)
    df, _ = compute_xg_features(df)
    df, _ = compute_media11_rolling_features(df)

    return df


def pipeline_cv(df, label='', n_splits=5):
    """Corre select_columns → limpieza → CV y devuelve resultados."""
    df2 = select_columns(df.copy())
    df2 = handle_missing_values(df2, strategy='mean')
    df2 = create_derived_variables(df2)
    df2 = filter_rows(df2)
    team_stats = aggregate_by_team(df2)
    df2 = join_team_stats(df2, team_stats)
    df_model, le_dict = prepare_for_modeling(df2)

    if 'Resultado_E1' not in df_model.columns:
        print(f'  [{label}] ERROR: sin Resultado_E1')
        return None

    y = df_model['Resultado_E1']
    exclude = [
        'Partido_id', 'Fase', 'Equipo1', 'Equipo2',
        'EQUIPO1_GOLES', 'EQUIPO2_GOLES', 'Resultado_E1', 'Diferencia_Goles',
        'Goles_dentro_area_E1', 'Goles_Fuera_Area_E1',
        'Goles_dentro_area_E2', 'Goles_Fuera_Area_E2',
        'goles_encajados_E1', 'goles_encajados_E2',
        'Goles_encajados_propia_puerta_E1', 'Goles_encajados_propia_puerta_E2',
        'Porterias_a_cero_E1', 'Porterias_a_cero_E2',
        'Eficiencia_Tiros_E1', 'Eficiencia_Tiros_E2',
    ]
    X = df_model.drop(columns=[c for c in exclude if c in df_model.columns])

    print(f'\n{"="*60}')
    print(f'  [{label}]  filas={len(df2)}  features={X.shape[1]}  clases={list(le_dict["Resultado_E1"].classes_)}')
    print(f'{"="*60}')

    # df_dc=None omite Dixon-Coles del ensemble (evita problemas de índices con el combinado)
    return cross_validate_models(X, y, df_dc=None, n_splits=n_splits)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Cargar datasets ───────────────────────────────────────────────────────
    print('Cargando datasets...')
    ucl = pd.read_excel(ROOT / 'data' / 'creando_dataset_modificado.xlsx')
    pl1 = pd.read_csv(ROOT / 'data' / 'premier_2024-25_enriquecido.csv')
    pl2 = pd.read_csv(ROOT / 'data' / 'premier_2025-26_enriquecido.csv')

    # Columna Liga
    ucl['Liga'] = 0
    pl1['Liga'] = 1
    pl2['Liga'] = 1

    print(f'  UCL : {len(ucl)} partidos')
    print(f'  PL 24-25: {len(pl1)} partidos')
    print(f'  PL 25-26: {len(pl2)} partidos')

    combinado = pd.concat([ucl, pl1, pl2], ignore_index=True)
    print(f'  TOTAL   : {len(combinado)} partidos')

    # ── A) Baseline: solo UCL ─────────────────────────────────────────────────
    print('\n\n### A) SOLO UCL (baseline) ###')
    df_ucl = cargar_y_preparar(ucl.copy())
    res_ucl = pipeline_cv(df_ucl, label='UCL solo', n_splits=5)

    # ── B) Combinado sin feature Liga ─────────────────────────────────────────
    print('\n\n### B) UCL + PL (sin feature Liga) ###')
    df_comb = cargar_y_preparar(combinado.copy())
    res_comb = pipeline_cv(df_comb, label='UCL+PL sin Liga', n_splits=5)

    # ── C) Combinado con feature Liga en select_columns ───────────────────────
    print('\n\n### C) UCL + PL (con feature Liga) ###')
    # Parchear select_columns para incluir Liga temporalmente
    import ml.knime_workflow_converter as kwc
    _orig_select = kwc.select_columns

    def select_con_liga(df):
        result = _orig_select(df)
        if 'Liga' in df.columns:
            result['Liga'] = df['Liga'].values
        return result

    kwc.select_columns = select_con_liga
    try:
        df_comb2 = cargar_y_preparar(combinado.copy())
        res_liga = pipeline_cv(df_comb2, label='UCL+PL con Liga', n_splits=5)
    finally:
        kwc.select_columns = _orig_select

    # ── Resumen final ─────────────────────────────────────────────────────────
    print('\n\n' + '='*60)
    print('RESUMEN COMPARATIVO')
    print('='*60)

    def _fmt(res):
        if not res:
            return 'N/A'
        accs = [r.get('accuracy', 0) for r in res.values() if isinstance(r, dict)]
        if not accs:
            return 'N/A'
        mean = np.mean(accs)
        std  = np.std(accs)
        best_model = max(res.items(), key=lambda x: x[1].get('accuracy', 0) if isinstance(x[1], dict) else 0)
        return f'{mean*100:.1f}% ± {std*100:.1f}%  (mejor: {best_model[0]} {best_model[1].get("accuracy",0)*100:.1f}%)'

    print(f'  A) Solo UCL      : {_fmt(res_ucl)}')
    print(f'  B) UCL+PL s/Liga : {_fmt(res_comb)}')
    print(f'  C) UCL+PL c/Liga : {_fmt(res_liga)}')
    print()


if __name__ == '__main__':
    main()
