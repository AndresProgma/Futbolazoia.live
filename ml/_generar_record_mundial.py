"""
Genera record_historico.json a partir de los AMISTOSOS de selecciones 2026.

- Entrena el pipeline Mundial UNA vez con selecciones_combinado.csv (hasta dic-2025).
- Predice cada amistoso 2026 (1X2 + mercados de córners/amarillas).
- Arma la "apuesta recomendada" combinada (1X2 + mejor Under córners + mejor
  Under amarillas) replicando la lógica de autoValores()/_bestUnder() del front.
- Evalúa cada pata contra el resultado real (goles/córners/amarillas scrapeados)
  para poder mostrar si la apuesta acertó completa o qué pata falló.

Uso:  python ml/_generar_record_mundial.py
Salida: data/record_historico.json
"""
import json, sys, os
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ml.pipeline_mundial import main_mundial, predecir_partido_mundial

RESULTADOS = _PROJECT_ROOT / "data" / "amistosos_2026_resultados.csv"
OUTPUT     = _PROJECT_ROOT / "data" / "record_historico.json"
N_RUNS     = 15


# ─── Lógica de selección de líneas (port de static/index.html) ────────────────

def _best_under(dist: dict):
    """Mejor línea Under con prob en [0.62, 0.75]. Devuelve (line, prob) o None."""
    if not dist:
        return None
    cands = []
    for k, v in dist.items():
        if not k.startswith('over_') or not isinstance(v, (int, float)):
            continue
        try:
            line = float(k.replace('over_', '').replace('_', '.'))
        except ValueError:
            continue
        under = 1 - v
        if 0.62 <= under <= 0.75:
            cands.append((line, under))
    if not cands:
        return None
    cands.sort(key=lambda x: -x[1])
    return cands[0]


def construir_apuesta(pred, e1, e2, goles_real, corners_real, amarillas_real):
    """Arma la apuesta combinada y evalúa cada pata contra lo real."""
    cons = pred['consenso']
    mercados = pred.get('mercados') or {}
    patas = []

    # ── Pata 1X2 (consenso) ──────────────────────────────────────────────
    if cons['win'] >= cons['draw'] and cons['win'] >= cons['loss']:
        pick1x2, nombre, desc, prob = 'Win', f'1X2 · {e1}', 'Victoria local', cons['win']
    elif cons['loss'] >= cons['win'] and cons['loss'] >= cons['draw']:
        pick1x2, nombre, desc, prob = 'Loss', f'1X2 · {e2}', 'Victoria visitante', cons['loss']
    else:
        pick1x2, nombre, desc, prob = 'Draw', '1X2 · Empate', 'Partido igualado', cons['draw']

    real_1x2 = goles_real['res']
    patas.append({
        'tipo': '1X2', 'nombre': nombre, 'descripcion': desc,
        'prob': round(prob, 4), 'pick': pick1x2,
        'real': real_1x2,
        'resultado': 'acierto' if pick1x2 == real_1x2 else 'fallo',
        'detalle': f"Predijo {desc.lower()} · resultado real {goles_real['str']}",
    })

    # ── Pata Under córners ───────────────────────────────────────────────
    cu = _best_under(mercados.get('corners'))
    if cu:
        line, p = cu
        if corners_real is None:
            res = 'sin_dato'
        else:
            res = 'acierto' if corners_real < line else 'fallo'
        patas.append({
            'tipo': 'corners', 'nombre': f'Córners U{line}', 'descripcion': 'Mejor under córners',
            'prob': round(p, 4), 'linea': line, 'real': corners_real,
            'resultado': res,
            'detalle': (f"Menos de {line} córners · reales {corners_real}"
                        if corners_real is not None else f"Menos de {line} córners · sin dato real"),
        })

    # ── Pata Under amarillas ─────────────────────────────────────────────
    au = _best_under(mercados.get('amarillas'))
    if au:
        line, p = au
        if amarillas_real is None:
            res = 'sin_dato'
        else:
            res = 'acierto' if amarillas_real < line else 'fallo'
        patas.append({
            'tipo': 'amarillas', 'nombre': f'Amarillas U{line}', 'descripcion': 'Mejor under amarillas',
            'prob': round(p, 4), 'linea': line, 'real': amarillas_real,
            'resultado': res,
            'detalle': (f"Menos de {line} amarillas · reales {amarillas_real}"
                        if amarillas_real is not None else f"Menos de {line} amarillas · sin dato real"),
        })

    evaluables = [x for x in patas if x['resultado'] in ('acierto', 'fallo')]
    n_acierto  = sum(1 for x in evaluables if x['resultado'] == 'acierto')
    fallos     = [x for x in patas if x['resultado'] == 'fallo']
    completa   = len(evaluables) > 0 and len(fallos) == 0

    return {
        'patas': patas,
        'n_patas': len(patas),
        'n_evaluables': len(evaluables),
        'n_acierto': n_acierto,
        'completa': completa,
        'que_fallo': [x['nombre'] for x in fallos],
    }


def main():
    # 1. Cargar amistosos y deduplicar
    df = pd.read_csv(RESULTADOS)
    df = df.dropna(subset=['goles1', 'goles2'])
    df = df.drop_duplicates(subset=['fecha', 'equipo1', 'equipo2']).reset_index(drop=True)
    df = df.sort_values('fecha', ascending=False).reset_index(drop=True)
    print(f"Amistosos a predecir: {len(df)}")

    # 2. Entrenar pipeline Mundial una sola vez
    print("\n🧠 Entrenando pipeline Mundial (selecciones_combinado)...")
    results = main_mundial()
    print("✅ Pipeline entrenado\n")

    # 3. Predecir cada amistoso
    registros = []
    for i, row in df.iterrows():
        e1, e2 = str(row['equipo1']), str(row['equipo2'])
        g1, g2 = int(row['goles1']), int(row['goles2'])
        fecha  = str(row['fecha'])[:10]
        res    = 'Win' if g1 > g2 else 'Loss' if g1 < g2 else 'Draw'
        goles_real = {'str': f'{g1}–{g2}', 'res': res}

        def _opt(v):
            return int(v) if pd.notna(v) else None
        c1, c2 = _opt(row.get('corners1')), _opt(row.get('corners2'))
        a1, a2 = _opt(row.get('amarillas1')), _opt(row.get('amarillas2'))
        corners_real   = (c1 + c2) if (c1 is not None and c2 is not None) else None
        amarillas_real = (a1 + a2) if (a1 is not None and a2 is not None) else None

        print(f"[{i+1}/{len(df)}] {e1} vs {e2} ({fecha})... ", end='', flush=True)
        try:
            pred = predecir_partido_mundial(e1, e2, results, n_runs=N_RUNS)
            cons = pred['consenso']
            apuesta = construir_apuesta(pred, e1, e2, goles_real, corners_real, amarillas_real)
            ok = cons['pred'] == res
            tag = '✅' if ok else '❌'
            comb = '🟢' if apuesta['completa'] else f"🔴{apuesta['n_acierto']}/{apuesta['n_evaluables']}"
            print(f"{tag} 1X2 pred={cons['pred']} real={res} | combi {comb}")

            registros.append({
                'equipo1': e1, 'equipo2': e2, 'fecha': fecha, 'fase': 'Amistoso',
                'goles_real': goles_real['str'], 'resultado_real': res,
                'corners_real': corners_real, 'amarillas_real': amarillas_real,
                'consenso': {
                    'pred': cons['pred'],
                    'win':  round(cons['win'], 4),
                    'draw': round(cons['draw'], 4),
                    'loss': round(cons['loss'], 4),
                },
                'modelos': [
                    {'modelo': m['modelo'], 'pred': m['pred'],
                     'win': round(m['win'], 4), 'draw': round(m['draw'], 4), 'loss': round(m['loss'], 4)}
                    for m in pred.get('modelos', [])
                ],
                'apuesta': apuesta,
                'acierto': ok,
            })
        except Exception as exc:
            print(f"ERROR: {exc}")
            registros.append({'equipo1': e1, 'equipo2': e2, 'fecha': fecha, 'error': str(exc)})

    # 4. Estadísticas
    resueltos = [r for r in registros if 'acierto' in r]
    aciertos  = [r for r in resueltos if r['acierto']]
    pct = len(aciertos) / len(resueltos) * 100 if resueltos else 0
    combis = [r for r in registros if r.get('apuesta', {}).get('n_evaluables', 0) > 0]
    combis_ok = [r for r in combis if r['apuesta']['completa']]
    pct_combi = len(combis_ok) / len(combis) * 100 if combis else 0
    print(f"\n📊 1X2: {len(aciertos)}/{len(resueltos)} = {pct:.1f}%")
    print(f"📊 Apuesta combinada completa: {len(combis_ok)}/{len(combis)} = {pct_combi:.1f}%")

    OUTPUT.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"✅ Guardado en {OUTPUT}")


if __name__ == '__main__':
    main()
