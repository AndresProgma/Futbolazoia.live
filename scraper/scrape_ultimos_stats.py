"""
Scrapea las ESTADÍSTICAS REALES (parte completo) de los partidos que están en
data/ultimos_mercados.json y las fusiona como `stats_reales` en ese mismo JSON,
para mostrar en el desplegable los números reales del partido (no los del modelo).

Uso:  python scraper/scrape_ultimos_stats.py
"""
import sys, os, json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scraper.scraper_selecciones import listar_partidos
from scraper.scraper_premier import SofaScoreSession, obtener_stats

TORNEO_AMISTOSOS = 851
SEASON_2026      = 87155
JSON_PATH = os.path.join(_ROOT, 'data', 'ultimos_mercados.json')


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _stats_team(s):
    """Mapea el dict crudo de obtener_stats a las claves del front."""
    comp = _num(s.get('Pases_completados'))
    real = _num(s.get('Pases_realizados'))
    prec = round(comp / real * 100, 1) if (comp and real) else None
    return {
        'posesion':       _num(s.get('Posesion')),
        'tiros':          _num(s.get('Disparos_totales')),
        'tiros_puerta':   _num(s.get('Disparos_a_puerta')),
        'corners_for':    _num(s.get('Saques_de_esquina_sacados')),
        'amarillas':      _num(s.get('Tarjetas_amarillas')),
        'faltas':         _num(s.get('Faltas_cometidas')),
        'precision_pase': prec,
        'oportunidades':  _num(s.get('Oportunidades_claras')),
    }


def main():
    partidos = json.load(open(JSON_PATH, encoding='utf-8'))
    print(f'Partidos en ultimos_mercados.json: {len(partidos)}')

    sess = SofaScoreSession()
    try:
        lista = listar_partidos(sess, TORNEO_AMISTOSOS, SEASON_2026)
        # índice (fecha, home, away) -> event_id
        idx = {(p['fecha'], p['home'], p['away']): p['event_id'] for p in lista}

        for p in partidos:
            key = (p['fecha'], p['equipo1'], p['equipo2'])
            eid = idx.get(key)
            if not eid:
                print(f"  ⚠ sin event_id para {key}")
                continue
            sh, sa = obtener_stats(sess, eid)
            e1 = _stats_team(sh)
            e2 = _stats_team(sa)
            # Goles reales desde el propio JSON
            e1['goles_for'] = p.get('goles_real_e1')
            e2['goles_for'] = p.get('goles_real_e2')
            p['stats_reales'] = {'e1': e1, 'e2': e2}
            print(f"  ✓ {p['equipo1']} vs {p['equipo2']}: pos {e1['posesion']}-{e2['posesion']}, "
                  f"tiros {e1['tiros']}-{e2['tiros']}, faltas {e1['faltas']}-{e2['faltas']}")
    finally:
        sess.close()

    json.dump(partidos, open(JSON_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n✅ stats_reales fusionadas en {JSON_PATH}')


if __name__ == '__main__':
    main()
