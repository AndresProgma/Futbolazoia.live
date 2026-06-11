"""
Scrapea resultados de amistosos internacionales 2026 para el record histórico.
Captura SOLO lo necesario para evaluar la apuesta recomendada:
  fecha, equipos, goles, corners (total por equipo), amarillas (total por equipo).

No construye las 140 columnas del dataset (estos partidos NO entran al
entrenamiento; el modelo se entrena con selecciones_combinado hasta dic-2025).

Uso:
    python scraper/scrape_amistosos.py [max_partidos]
Salida:
    data/amistosos_2026_resultados.csv
"""
import sys, os
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scraper.scraper_selecciones import listar_partidos
from scraper.scraper_premier import SofaScoreSession, obtener_stats

TORNEO_AMISTOSOS = 851
SEASON_2026      = 87155
OUT = os.path.join(_ROOT, 'data', 'amistosos_2026_resultados.csv')

max_partidos = int(sys.argv[1]) if len(sys.argv) > 1 else None

sess = SofaScoreSession()
filas = []
try:
    partidos = listar_partidos(sess, TORNEO_AMISTOSOS, SEASON_2026)
    # Más recientes primero (los últimos vistos hasta hoy)
    partidos = sorted(partidos, key=lambda p: p['fecha'] or '', reverse=True)
    if max_partidos:
        partidos = partidos[:max_partidos]
    print(f'\nScrapeando stats de {len(partidos)} amistosos...\n')

    for i, p in enumerate(partidos, 1):
        eid = p['event_id']
        label = f"{p['fecha']}  {p['home']} {p['goles_home']}-{p['goles_away']} {p['away']}"
        print(f'[{i:3}/{len(partidos)}] {label}', end=' ', flush=True)
        try:
            sh, sa = obtener_stats(sess, eid)
        except Exception as e:
            print(f'[ERROR stats: {e}] — guardando parcial y saliendo')
            break

        filas.append({
            'fecha':        p['fecha'],
            'equipo1':      p['home'],
            'equipo2':      p['away'],
            'goles1':       p['goles_home'],
            'goles2':       p['goles_away'],
            'corners1':     sh.get('Saques_de_esquina_sacados'),
            'corners2':     sa.get('Saques_de_esquina_sacados'),
            'amarillas1':   sh.get('Tarjetas_amarillas'),
            'amarillas2':   sa.get('Tarjetas_amarillas'),
        })
        c = (sh.get('Saques_de_esquina_sacados'), sa.get('Saques_de_esquina_sacados'))
        a = (sh.get('Tarjetas_amarillas'), sa.get('Tarjetas_amarillas'))
        print(f'corners={c} amarillas={a}  ✓')

        # Guardado parcial cada 20
        if i % 20 == 0:
            pd.DataFrame(filas).to_csv(OUT, index=False)
finally:
    sess.close()

df = pd.DataFrame(filas)
df.to_csv(OUT, index=False)
con_stats = df['corners1'].notna().sum() if len(df) else 0
print(f'\n✅ Guardado {len(df)} amistosos en {OUT}')
print(f'   Con córners/tarjetas: {con_stats}/{len(df)}')
