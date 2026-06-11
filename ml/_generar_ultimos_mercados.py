"""
Genera data/ultimos_mercados.json con los MERCADOS COMPLETOS (todas las
estadísticas apostables) de los últimos N amistosos resueltos, para el
desplegable bajo "Estadísticas clave" en la tarjeta del pick.

A diferencia del record (solo 1X2 + 3 patas), aquí se guarda el dict `mercados`
completo de la predicción + los resultados reales, para poder mostrar cada
mercado con su ✓/✗.

Uso:  python ml/_generar_ultimos_mercados.py [N]
Salida: data/ultimos_mercados.json
"""
import json, sys
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ml.pipeline_mundial import main_mundial, predecir_partido_mundial
from ml._generar_record_mundial import construir_apuesta

RESULTADOS = _PROJECT_ROOT / "data" / "amistosos_2026_resultados.csv"
OUTPUT     = _PROJECT_ROOT / "data" / "ultimos_mercados.json"
N          = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_RUNS     = 20


def main():
    df = pd.read_csv(RESULTADOS)
    df = df.dropna(subset=['goles1', 'goles2'])
    df = df.drop_duplicates(subset=['fecha', 'equipo1', 'equipo2'])
    # Solo partidos con córners + amarillas para un showcase completo
    df = df.dropna(subset=['corners1', 'corners2', 'amarillas1', 'amarillas2'])
    df = df.sort_values('fecha', ascending=False).head(N).reset_index(drop=True)
    print(f"Últimos {len(df)} amistosos con stats completas:")
    for _, r in df.iterrows():
        print(f"  {r['fecha']}  {r['equipo1']} {int(r['goles1'])}-{int(r['goles2'])} {r['equipo2']}")

    print("\n🧠 Entrenando pipeline Mundial...")
    results = main_mundial()
    print("✅ Pipeline entrenado\n")

    partidos = []
    for _, row in df.iterrows():
        e1, e2 = str(row['equipo1']), str(row['equipo2'])
        g1, g2 = int(row['goles1']), int(row['goles2'])
        fecha  = str(row['fecha'])[:10]
        res    = 'Win' if g1 > g2 else 'Loss' if g1 < g2 else 'Draw'
        goles_real = {'str': f'{g1}–{g2}', 'res': res}
        c1, c2 = int(row['corners1']), int(row['corners2'])
        a1, a2 = int(row['amarillas1']), int(row['amarillas2'])
        corners_real, amarillas_real = c1 + c2, a1 + a2

        print(f"Prediciendo {e1} vs {e2} ({fecha})...")
        pred = predecir_partido_mundial(e1, e2, results, n_runs=N_RUNS)
        cons = pred['consenso']
        apuesta = construir_apuesta(pred, e1, e2, goles_real, corners_real, amarillas_real)

        partidos.append({
            'equipo1': e1, 'equipo2': e2, 'fecha': fecha,
            'goles_real': goles_real['str'], 'resultado_real': res,
            'goles_real_e1': g1, 'goles_real_e2': g2,
            'corners_real': corners_real, 'corners_real_e1': c1, 'corners_real_e2': c2,
            'amarillas_real': amarillas_real, 'amarillas_real_e1': a1, 'amarillas_real_e2': a2,
            'consenso': {
                'pred': cons['pred'],
                'win': round(cons['win'], 4), 'draw': round(cons['draw'], 4), 'loss': round(cons['loss'], 4),
            },
            'mercados': pred.get('mercados'),
            'stats_equipos': pred.get('stats_equipos'),
            'apuesta': apuesta,
        })

    OUTPUT.write_text(json.dumps(partidos, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n✅ Guardado {len(partidos)} partidos en {OUTPUT}")


if __name__ == '__main__':
    main()
