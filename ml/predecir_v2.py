"""
predecir_v2 — predicción honesta de un partido específico.

Diferencia clave con `predecir_partido` original: filtra el dataset para usar
SOLO los partidos JUGADOS ANTES de la fecha del partido a predecir.

Es decir, simula la predicción "como si fuera nuevo de hoy" — el modelo no ve:
  1. El partido que estás prediciendo (no hay leakage del resultado).
  2. Los partidos jugados DESPUÉS de ese (no usa info del futuro).
  3. Los partidos del MISMO DÍA (puede ser que ya jugaron pero asumimos que no).

Uso desde Python:
    from predecir_v2 import predecir_partido_v2
    predecir_partido_v2("Real Madrid", "Barcelona", fecha="2026-02-25")
    predecir_partido_v2("Real Madrid", "Barcelona")  # busca el más reciente

Uso desde CLI:
    python predecir_v2.py "Real Madrid" "Barcelona"
    python predecir_v2.py "Real Madrid" "Barcelona" --fecha 2026-02-25
    python predecir_v2.py "Real Madrid" "Barcelona" --fecha 2026-02-25 --fase Cuartos
    python predecir_v2.py "Real Madrid" "Barcelona" --n-runs 30
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ml.knime_workflow_converter import main as run_pipeline, predecir_partido

DATASET = _PROJECT_ROOT / "data" / "creando_dataset_modificado.xlsx"

# CSVs de Premier League usados como contexto de forma (ELO, xG, forma rolling).
# El modelo sigue entrenando/evaluando solo en UCL — PL enriquece las features.
PL_CONTEXT = [
    p for p in [
        _PROJECT_ROOT / "data" / "premier_2024-25_enriquecido.csv",
        _PROJECT_ROOT / "data" / "premier_2025-26_enriquecido.csv",
    ]
    if p.exists()
]


def _norm_fecha(f):
    """YYYY-M-D ó YYYY-MM-DD → datetime."""
    return pd.to_datetime(f, errors='coerce')


def _buscar_partido(df, equipo1, equipo2, fecha=None):
    """Localiza el partido en el dataset. Devuelve (fila, fecha_corte_dt) o (None, None)."""
    e1, e2 = str(equipo1), str(equipo2)
    df = df.copy()
    df['_fecha_dt'] = df['Fecha'].apply(_norm_fecha)

    mask_a = (df['Equipo1'].astype(str) == e1) & (df['Equipo2'].astype(str) == e2)
    mask_b = (df['Equipo1'].astype(str) == e2) & (df['Equipo2'].astype(str) == e1)
    cand = df[mask_a | mask_b].sort_values('_fecha_dt')

    if fecha:
        fdt = _norm_fecha(fecha)
        match = cand[cand['_fecha_dt'].dt.date == fdt.date()] if pd.notna(fdt) else cand
        if not match.empty:
            row = match.iloc[0]
            return row, row['_fecha_dt']
        # Fecha dada pero no hay match concreto → corte por fecha, predicción "nueva"
        return None, fdt

    # Sin fecha: el más reciente
    if not cand.empty:
        row = cand.iloc[-1]
        return row, row['_fecha_dt']
    return None, None


def predecir_partido_v2(equipo1, equipo2, fecha=None, fase=None,
                       n_runs=20, dataset_path=None):
    """
    Predice un partido entrenando SOLO con datos previos a su fecha.

    Args:
        equipo1, equipo2: nombres exactos del dataset
        fecha: 'YYYY-MM-DD' del partido. Si None, se busca el más reciente
               en el dataset entre los dos equipos.
        fase: 'Liga', 'Octavos', 'Cuartos', 'Semifinal', 'Final'. Si None,
              se infiere del partido encontrado (o 'Liga' por defecto).
        n_runs: corridas por modelo en el ensemble (default 20).
        dataset_path: override del Excel fuente.

    Returns:
        dict del predecir_partido() pero entrenado solo con los partidos
        ANTERIORES a la fecha del partido predicho.
    """
    path = Path(dataset_path) if dataset_path else DATASET
    df_full = pd.read_excel(path)

    # 1. Encontrar el partido (o usar la fecha provista) → fecha de corte
    fila_partido, fecha_corte = _buscar_partido(df_full, equipo1, equipo2, fecha)

    if fila_partido is not None:
        g1 = fila_partido.get('EQUIPO1_GOLES')
        g2 = fila_partido.get('EQUIPO2_GOLES')
        g1_str = int(g1) if pd.notna(g1) else '?'
        g2_str = int(g2) if pd.notna(g2) else '?'
        fase_real = str(fila_partido.get('Fase', 'Liga'))
        print(f"\n📍 Partido localizado: {fila_partido['Equipo1']} {g1_str}–{g2_str} "
              f"{fila_partido['Equipo2']}  |  {fecha_corte.date()}  |  Fase: {fase_real}")
        if fase is None:
            fase = fase_real
    else:
        print(f"\n📍 Partido NO encontrado en el dataset.")
        if fecha_corte is not None:
            print(f"   Usando corte temporal por fecha: < {fecha_corte.date()}")
        else:
            print(f"   Sin fecha de corte → se usa el dataset completo (mismo que predecir_partido original).")

    fase = fase or 'Liga'

    # 2. Filtrar dataset: solo partidos antes de la fecha de corte
    df_full['_fecha_dt'] = df_full['Fecha'].apply(_norm_fecha)
    if fecha_corte is not None:
        mask_antes = df_full['_fecha_dt'] < fecha_corte
        df_filtrado = df_full[mask_antes].drop(columns=['_fecha_dt']).reset_index(drop=True)
        print(f"\n🔪 Dataset filtrado: {len(df_full)} → {len(df_filtrado)} partidos "
              f"(antes de {fecha_corte.date()}, mismo día incluido como futuro)")
    else:
        df_filtrado = df_full.drop(columns=['_fecha_dt']).reset_index(drop=True)
        print(f"\nℹ️  Sin corte aplicado, training sobre {len(df_filtrado)} partidos.")

    if len(df_filtrado) < 30:
        print(f"⚠️  Solo {len(df_filtrado)} partidos para entrenar — pocos datos, predicción muy ruidosa.")

    # 3. Entrenar pipeline sobre el subset filtrado (tempfile)
    tmp = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    tmp.close()
    try:
        df_filtrado.to_excel(tmp.name, index=False)
        print(f"\n🧠 Entrenando pipeline sobre el dataset filtrado...\n")
        results = run_pipeline(tmp.name, context_filepaths=[str(p) for p in PL_CONTEXT])

        # 4. Predecir
        print(f"\n{'='*60}")
        print(f"  PREDICCIÓN HONESTA (v2): {equipo1} vs {equipo2}")
        print(f"  Fase: {fase}")
        if fecha_corte is not None:
            print(f"  Training: {len(df_filtrado)} partidos previos a {fecha_corte.date()}")
        print(f"{'='*60}")
        pred = predecir_partido(equipo1, equipo2, results, n_runs=n_runs, fase=fase)

        # 5. Si conocemos el resultado real, compararlo
        if fila_partido is not None:
            g1 = fila_partido.get('EQUIPO1_GOLES')
            g2 = fila_partido.get('EQUIPO2_GOLES')
            if pd.notna(g1) and pd.notna(g2):
                g1, g2 = int(g1), int(g2)
                if g1 > g2:
                    real = 'Win'
                elif g1 < g2:
                    real = 'Loss'
                else:
                    real = 'Draw'
                ok = pred['consenso']['pred'] == real
                print(f"\n📊 Resultado REAL: {fila_partido['Equipo1']} {g1}–{g2} {fila_partido['Equipo2']}  →  {real}")
                print(f"   Consenso predijo: {pred['consenso']['pred']}  →  {'✅ ACIERTO' if ok else '❌ FALLO'}")
                pred['resultado_real'] = real
                pred['acierto_consenso'] = ok
                pred['goles_real'] = (g1, g2)
        return pred
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('equipo1')
    p.add_argument('equipo2')
    p.add_argument('--fecha', help='Fecha YYYY-MM-DD del partido a predecir')
    p.add_argument('--fase', help='Liga | Playoffs | Octavos | Cuartos | Semifinal | Final')
    p.add_argument('--n-runs', type=int, default=20)
    args = p.parse_args()

    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

    predecir_partido_v2(args.equipo1, args.equipo2,
                       fecha=args.fecha, fase=args.fase,
                       n_runs=args.n_runs)


if __name__ == '__main__':
    main()
