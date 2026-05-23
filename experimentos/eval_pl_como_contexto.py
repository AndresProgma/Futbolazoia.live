"""
Prueba: Premier League como contexto de forma para UCL.

Pipeline:
  1. Cargar UCL + PL combinados, ordenar cronológicamente
  2. Calcular features (ELO, forma, xG, media11, H2H) sobre TODO el dataset
  3. Filtrar a partidos UCL únicamente
  4. Entrenar y evaluar SOLO en UCL

Comparación:
  A) UCL solo (baseline — features calculadas solo con UCL)
  B) UCL+PL como contexto (features ricas, modelo entrenado solo en UCL)

Uso:
    python experimentos/eval_pl_como_contexto.py
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


def preparar_features(df):
    """Aplica el pipeline de features sobre el df completo (puede ser UCL+PL)."""
    if 'Fecha' in df.columns:
        df['_f'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.sort_values('_f', na_position='last').reset_index(drop=True)
        df = df.drop(columns=['_f'])

    if 'Fase' in df.columns and 'Es_Local_E1' in df.columns:
        mask = df['Fase'].astype(str).str.strip().str.lower() == 'final'
        df.loc[mask, 'Es_Local_E1'] = 0

    df = compute_uefa_coef_features(df)
    df, _ = compute_elo_features(df)
    df, _ = compute_form_features(df)
    df, _ = compute_h2h_features(df)
    df, _ = compute_xg_features(df)
    df, _ = compute_media11_rolling_features(df)
    return df


def pipeline_cv(df, label='', n_splits=5):
    """select_columns → limpieza → CV sobre el df dado."""
    df2 = select_columns(df.copy())
    df2 = handle_missing_values(df2, strategy='mean')
    df2 = create_derived_variables(df2)
    df2 = filter_rows(df2)
    ts  = aggregate_by_team(df2)
    df2 = join_team_stats(df2, ts)
    df_model, le_dict = prepare_for_modeling(df2)

    if 'Resultado_E1' not in df_model.columns:
        print(f'[{label}] ERROR: sin Resultado_E1'); return None

    y = df_model['Resultado_E1']
    excl = [
        'Partido_id', 'Fase', 'Equipo1', 'Equipo2',
        'EQUIPO1_GOLES', 'EQUIPO2_GOLES', 'Resultado_E1', 'Diferencia_Goles',
        'Goles_dentro_area_E1', 'Goles_Fuera_Area_E1',
        'Goles_dentro_area_E2', 'Goles_Fuera_Area_E2',
        'goles_encajados_E1', 'goles_encajados_E2',
        'Goles_encajados_propia_puerta_E1', 'Goles_encajados_propia_puerta_E2',
        'Porterias_a_cero_E1', 'Porterias_a_cero_E2',
        'Eficiencia_Tiros_E1', 'Eficiencia_Tiros_E2',
    ]
    X = df_model.drop(columns=[c for c in excl if c in df_model.columns])

    print(f'\n{"="*60}')
    print(f'  [{label}]  filas={len(df2)}  features={X.shape[1]}')
    print(f'{"="*60}')

    return cross_validate_models(X, y, df_dc=None, n_splits=n_splits)


def main():
    print('Cargando datasets...')
    ucl = pd.read_excel(ROOT / 'data' / 'creando_dataset_modificado.xlsx')
    pl1 = pd.read_csv(ROOT / 'data' / 'premier_2024-25_enriquecido.csv')
    pl2 = pd.read_csv(ROOT / 'data' / 'premier_2025-26_enriquecido.csv')

    print(f'  UCL: {len(ucl)}  PL 24-25: {len(pl1)}  PL 25-26: {len(pl2)}')

    # ── A) Baseline: UCL solo ─────────────────────────────────────────────────
    print('\n\n### A) BASELINE — UCL solo ###')
    df_a = preparar_features(ucl.copy())
    res_a = pipeline_cv(df_a, 'UCL solo (baseline)')

    # ── B) PL como contexto ───────────────────────────────────────────────────
    print('\n\n### B) PL como contexto — train/test solo UCL ###')
    # Marcar origen
    ucl['_origen'] = 'UCL'
    pl1['_origen'] = 'PL'
    pl2['_origen'] = 'PL'

    # Combinar para calcular features con historial completo
    combinado = pd.concat([ucl, pl1, pl2], ignore_index=True)
    df_b_full = preparar_features(combinado.copy())

    # Filtrar: solo partidos UCL para entrenar y evaluar
    df_b = df_b_full[df_b_full['_origen'] == 'UCL'].copy()
    print(f'\n  Features calculadas con {len(df_b_full)} partidos, '
          f'entrenamiento/test en {len(df_b)} partidos UCL')

    res_b = pipeline_cv(df_b, 'UCL+PL contexto → solo UCL')

    # ── Resumen ───────────────────────────────────────────────────────────────
    print('\n\n' + '='*60)
    print('RESUMEN')
    print('='*60)

    def _mejor(res):
        if res is None: return 'N/A'
        mejor = max(
            [(m, v) for m, v in res.items() if isinstance(v, dict)],
            key=lambda x: x[1].get('accuracy', 0)
        )
        accs = [v.get('accuracy', 0) for v in res.values() if isinstance(v, dict)]
        return (f"mejor={mejor[0]} {mejor[1]['accuracy']*100:.1f}%  "
                f"| media modelos={np.mean(accs)*100:.1f}%")

    print(f'\n  A) UCL solo         : {_mejor(res_a)}')
    print(f'  B) UCL+PL contexto  : {_mejor(res_b)}')

    if res_a and res_b:
        accs_a = [v.get('accuracy', 0) for v in res_a.values() if isinstance(v, dict)]
        accs_b = [v.get('accuracy', 0) for v in res_b.values() if isinstance(v, dict)]
        delta = (np.mean(accs_b) - np.mean(accs_a)) * 100
        print(f'\n  Diferencia media: {delta:+.1f}pp')
        print(f'  {"✅ El contexto PL mejora el modelo UCL" if delta > 0 else "❌ El contexto PL no mejora (o empeora) el modelo UCL"}')


if __name__ == '__main__':
    main()
