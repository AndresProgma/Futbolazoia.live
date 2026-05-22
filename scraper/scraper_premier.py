"""
Scraper Premier League — SofaScore via Playwright (sesión persistente).

Mantiene UN solo browser abierto para toda la sesión; cada llamada a la API
de SofaScore se hace desde el contexto del browser (evita Cloudflare).

Cobertura: ~90-100 / 140 columnas dependiendo de la riqueza de stats del partido.

Uso:
    python scraper/scraper_premier.py --temporadas
    python scraper/scraper_premier.py --scrape <season_id> <nombre_ej:2024-25>
    python scraper/scraper_premier.py --partido <event_id>   # debug de un partido
    python scraper/scraper_premier.py --combinar             # une data/premier_*.csv
"""

import re
import sys
import time
import json

import pandas as pd
from playwright.sync_api import sync_playwright

# ─── Constantes ───────────────────────────────────────────────────────────────

PL_TOURNAMENT = 17
SS_BASE       = 'https://api.sofascore.com/api/v1'
DELAY_API     = 0.8   # segundos entre llamadas fetch dentro del browser

# ─── Mapeo equipos ────────────────────────────────────────────────────────────

MAPEO_EQUIPOS = {
    'manchester city':          'Man City',
    'man city':                 'Man City',
    'manchester united':        'Man United',
    'manchester utd':           'Man United',
    'man united':               'Man United',
    'man utd':                  'Man United',
    'liverpool':                'Liverpool',
    'liverpool fc':             'Liverpool',
    'arsenal':                  'Arsenal',
    'arsenal fc':               'Arsenal',
    'chelsea':                  'Chelsea',
    'chelsea fc':               'Chelsea',
    'tottenham hotspur':        'Tottenham',
    'tottenham':                'Tottenham',
    'spurs':                    'Tottenham',
    'newcastle united':         'Newcastle',
    'newcastle united fc':      'Newcastle',
    'newcastle':                'Newcastle',
    'aston villa':              'Aston Villa',
    'aston villa fc':           'Aston Villa',
    'west ham united':          'West Ham',
    'west ham':                 'West Ham',
    'brighton & hove albion':   'Brighton',
    'brighton':                 'Brighton',
    'brentford':                'Brentford',
    'brentford fc':             'Brentford',
    'fulham':                   'Fulham',
    'fulham fc':                'Fulham',
    'crystal palace':           'Crystal Palace',
    'wolverhampton wanderers':  'Wolves',
    'wolves':                   'Wolves',
    'everton':                  'Everton',
    'everton fc':               'Everton',
    'nottingham forest':        'Nottm Forest',
    "nottingham forest fc":     'Nottm Forest',
    "nott'm forest":            'Nottm Forest',
    'leicester city':           'Leicester',
    'leicester city fc':        'Leicester',
    'leicester':                'Leicester',
    'leeds united':             'Leeds',
    'leeds':                    'Leeds',
    'southampton':              'Southampton',
    'southampton fc':           'Southampton',
    'burnley':                  'Burnley',
    'burnley fc':               'Burnley',
    'sheffield united':         'Sheffield Utd',
    'sheffield united fc':      'Sheffield Utd',
    'afc bournemouth':          'Bournemouth',
    'bournemouth':              'Bournemouth',
    'luton town':               'Luton',
    'luton':                    'Luton',
    'ipswich town':             'Ipswich',
    'ipswich town fc':          'Ipswich',
    'ipswich':                  'Ipswich',
    'sunderland':               'Sunderland',
    'sunderland afc':           'Sunderland',
    'norwich city':             'Norwich',
    'watford':                  'Watford',
    'stoke city':               'Stoke',
    'west bromwich albion':     'West Brom',
    'west brom':                'West Brom',
    'swansea city':             'Swansea',
    'cardiff city':             'Cardiff',
    'huddersfield town':        'Huddersfield',
}

def _norm_equipo(nombre):
    if not nombre:
        return nombre
    c = nombre.lower().strip()
    if c in MAPEO_EQUIPOS:
        return MAPEO_EQUIPOS[c]
    for k, v in MAPEO_EQUIPOS.items():
        if k in c or c in k:
            return v
    return nombre.title()


# ─── Parser de valores SofaScore ──────────────────────────────────────────────

def _parsear_simple(val):
    """'55%' → 55 | '14' → 14 | None → None"""
    if val is None:
        return None
    s = str(val).strip().rstrip('%').replace(',', '.')
    try:
        v = float(s)
        return int(v) if v == int(v) else v
    except (ValueError, TypeError):
        return None


def _parsear_ratio(val):
    """
    '23/40 (58%)' → (23, 40)
    '8/19 (42%)'  → (8, 19)
    '14'          → (14, None)
    None          → (None, None)
    """
    if val is None:
        return None, None
    s = str(val).strip()
    m = re.match(r'(\d+)/(\d+)', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    v = _parsear_simple(s)
    return v, None


# ─── Orden exacto de columnas del dataset ─────────────────────────────────────

COLUMNAS_DATASET = [
    'Partido_id', 'Fase', 'Fecha', 'Equipo1', 'Es_Local_E1', 'Equipo2',
    'EQUIPO1_GOLES', 'EQUIPO2_GOLES',
    'Goles_dentro_area_E1', 'Goles_Fuera_Area_E1',
    'Disparos_totales_E1', 'Disparos_a_puerta_E1', 'Disparos_fuera_E1',
    'Disparo_Bloqueados_E1', 'Disparos_Al palo_E1', 'Disparos_Larguero_E1',
    'Disparos_Poste_E1', 'Disparos_a_puerta_fuera_del_area_E1',
    'Disparos_fuera_desde_fuera_del_area_E1',
    'Asistencias_E1', 'Penaltis_marcados_E1', 'Penaltis_fallados_E1',
    'Penaltis_forzados_E1',
    'Ataques_E1', 'Oportunidades_claras_E1', 'Saques_de_esquina_sacados_E1',
    'Fueras_de_juego_E1', 'Regates_E1',
    'Ataques_tercio_ofensivo_E1', 'Ataques_zonas_clave_E1',
    'Carreras_hacia_el_area_E1',
    'Posesion_E1', 'Precision_pase_E1',
    'Pases_completados_E1', 'Pases_realizados_E1',
    'Pases_cortos_completados_E1', 'Pases_media_distancia_completados_E1',
    'Pases_en_largo_completados_E1',
    'Pases_completados_atras_E1', 'Pases_completadosa_izquierda_E1',
    'Pases_completados_derecha_E1',
    'Libres_directos_sacados_E1',
    'Centros__tercio_ofensivo_E1', 'Pases_zonas_clave_E1',
    'Pases_al_area_E1', 'Precision_en_el_centro_E1',
    'Centros_completados_E1', 'Centros_realizados_E1',
    'Tiempo_de_posesion_E1',
    'Balones_recuperados_E1', 'Bloqueos_E1', 'Penaltis_cometidos_E1',
    'Entradas_E1', 'Entradas_con_exito_E1', 'Entradas_perdidas_E1',
    'Despejes_completados_E1', 'Despejes_realizados_E1',
    'goles_encajados_E1', 'Goles_encajados_propia_puerta_E1',
    'Porterias_a_cero_E1',
    'Paradas_E1', 'Paradas_en_libres_directo_E1',
    'Paradas-tras_libre_indirecto_E1', 'Penaltis_parados_E1',
    'Balones_blocados_E1', 'Balones_blocados_por arriba_E1',
    'Balones_blocados_por_abajo_E1',
    'Despejes_de_puños_E1',
    'Tarjetas_amarillas_E1', 'Tarjetas_rojas_E1',
    'Faltas_cometidas_E1', 'Faltas_cometidas_tercio_def_E1',
    'Faltas_cometidas_en_campo_propio_E1',
    # ── E2 ──────────────────────────────────────────────────────────────────
    'Goles_dentro_area_E2', 'Goles_Fuera_Area_E2',
    'Disparos_totales_E2', 'Disparos_a_puerta_E2', 'Disparos_fuera_E2',
    'Disparo_Bloqueados_E2', 'Disparos_Al palo_E2', 'Disparos_Larguero_E2',
    'Disparos_Poste_E2', 'Disparos_a_puerta_fuera_del_area_E2',
    'Disparos_fuera_desde_fuera_del_area_E2',
    'Asistencias_E2', 'Penaltis_marcados_E2', 'Penaltis_fallados_E2',
    'Penaltis_forzados_E2',
    'Ataques_E2', 'Oportunidades_claras_E2', 'Saques_de_esquina_sacados_E2',
    'Fueras_de_juego_E2', 'Regates_E2',
    'Ataques_tercio_ofensivo_E2', 'Ataques_zonas_clave_E2',
    'Carreras_hacia_el_area_E2',
    'Posesion_E2', 'Precision_pase_E2',
    'Pases_completados_E2', 'Pases_realizados_E2',
    'Pases_cortos_completados_E2', 'Pases_media_distancia_completados_E2',
    'Pases_en_largo_completados_E2',
    'Pases_completados_atras_E2', 'Pases_completadosa_izquierda_E2',
    'Pases_completados_derecha_E2',
    'Libres_directos_sacados_E2',
    'Centros__tercio_ofensivo_E2', 'Pases_zonas_clave_E2',
    'Pases_al_area_E2', 'Precision_en_el_centro_E2',
    'Centros_completados_E2', 'Centros_realizados_E2',
    'Tiempo_de_posesion_E2',
    'Balones_recuperados_E2', 'Bloqueos_E2', 'Penaltis_cometidos_E2',
    'Entradas_E2', 'Entradas_con_exito_E2', 'Entradas_perdidas_E2',
    'Despejes_completados_E2', 'Despejes_realizados_E2',
    'goles_encajados_E2', 'Goles_encajados_propia_puerta_E2',
    'Porterias_a_cero_E2',
    'Paradas_E2', 'Paradas_en_libres_directo_E2',
    'Paradas-tras_libre_indirecto_E2', 'Penaltis_parados_E2',
    'Balones_blocados_E2', 'Balones_blocados_por arriba_E2',
    'Balones_blocados_por_abajo_E2',
    'Despejes_de_puños_E2',
    'Tarjetas_amarillas_E2', 'Tarjetas_rojas_E2',
    'Faltas_cometidas_E2', 'Faltas_cometidas_tercio_def_E2',
    'Faltas_cometidas_en_campo_propio_E2',
    'media11_titular_E1', 'media11_titular_E2',
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Extracción de stats de SofaScore
# ═══════════════════════════════════════════════════════════════════════════════

def _extraer_stats(grupos):
    """
    Dado el JSON de groups de SofaScore, devuelve (stats_home, stats_away)
    con columnas base del dataset → valor numérico.
    """
    h, a = {}, {}

    for grupo in grupos:
        for item in grupo.get('statisticsItems', []):
            name = item.get('name', '')
            hv   = item.get('home')
            av   = item.get('away')
            _aplicar_stat(name, hv, av, h, a)

    return h, a


def _aplicar_stat(name, hv, av, h, a):
    """Mapea una stat de SofaScore a columnas del dataset."""

    # ── Stats simples (valor directo) ─────────────────────────────────────
    SIMPLE = {
        'Ball possession':      'Posesion',
        'Total shots':          'Disparos_totales',
        'Shots on target':      'Disparos_a_puerta',
        'Shots off target':     'Disparos_fuera',
        'Blocked shots':        'Disparo_Bloqueados',
        'Hit woodwork':         'Disparos_Al palo',
        'Big chances':          'Oportunidades_claras',
        'Big chances created':  'Oportunidades_claras',
        'Final third entries':  'Ataques_tercio_ofensivo',
        'Passes into final third': 'Ataques_tercio_ofensivo',
        'Touches in penalty area': 'Carreras_hacia_el_area',
        'Offsides':             'Fueras_de_juego',
        'Corner kicks':         'Saques_de_esquina_sacados',
        'Passes':               'Pases_realizados',
        'Accurate passes':      'Pases_completados',
        'Key passes':           'Pases_zonas_clave',
        'Passes into the penalty area': 'Pases_al_area',
        'Free kicks':           'Libres_directos_sacados',
        'Fouls':                'Faltas_cometidas',
        'Yellow cards':         'Tarjetas_amarillas',
        'Red cards':            'Tarjetas_rojas',
        'Total tackles':        'Entradas',
        'Tackles':              'Entradas',
        'Interceptions':        'Balones_recuperados',
        'Recoveries':           'Balones_recuperados',
        'Clearances':           'Despejes_realizados',
        'Total saves':          'Paradas',
        'Goalkeeper saves':     'Paradas',
        'High claims':          'Balones_blocados_por arriba',
        'Punches':              'Despejes_de_puños',
        'Goals inside box':     'Goles_dentro_area',
        'Goals outside box':    'Goles_Fuera_Area',
        'Assists':              'Asistencias',
        'Penalties scored':     'Penaltis_marcados',
        'Penalties missed':     'Penaltis_fallados',
        'Penalties won':        'Penaltis_forzados',
        'Penalties committed':  'Penaltis_cometidos',
        'Penalty saves':        'Penaltis_parados',
        'Attacks':              'Ataques',
        'Dangerous attacks':    'Ataques_zonas_clave',
        'Time in possession':   'Tiempo_de_posesion',
        'Blocks':               'Bloqueos',
    }

    if name in SIMPLE:
        col = SIMPLE[name]
        if col not in h:
            h[col] = _parsear_simple(hv)
            a[col] = _parsear_simple(av)
        return

    # ── Long balls: "23/40 (58%)" → completados=23, realizados totales ────
    if name == 'Long balls':
        hc, ht = _parsear_ratio(hv)
        ac, at = _parsear_ratio(av)
        if 'Pases_en_largo_completados' not in h:
            h['Pases_en_largo_completados'] = hc
            a['Pases_en_largo_completados'] = ac
        return

    # ── Crosses: "3/18 (17%)" → completados=3, realizados=18 ─────────────
    if name == 'Crosses':
        hc, ht = _parsear_ratio(hv)
        ac, at = _parsear_ratio(av)
        if 'Centros_completados' not in h:
            h['Centros_completados'] = hc
            a['Centros_completados'] = ac
        if 'Centros_realizados' not in h:
            h['Centros_realizados'] = ht
            a['Centros_realizados'] = at
        return

    # ── Dribbles: "8/19 (42%)" → exitosos=8, intentados=19 ───────────────
    if name == 'Dribbles':
        hc, ht = _parsear_ratio(hv)
        ac, at = _parsear_ratio(av)
        if 'Regates' not in h:
            h['Regates'] = hc          # exitosos
            a['Regates'] = ac
        return

    # ── Tackles won: "67%" → derivar Entradas_con_exito ──────────────────
    if name == 'Tackles won':
        hp = _parsear_simple(hv)       # porcentaje
        ap = _parsear_simple(av)
        if hp is not None and 'Entradas' in h and h['Entradas'] is not None:
            h['Entradas_con_exito'] = round(h['Entradas'] * hp / 100)
        if ap is not None and 'Entradas' in a and a['Entradas'] is not None:
            a['Entradas_con_exito'] = round(a['Entradas'] * ap / 100)
        return

    # ── Accurate crosses: si SofaScore las lista por separado ─────────────
    if name == 'Accurate crosses':
        if 'Centros_completados' not in h:
            h['Centros_completados'] = _parsear_simple(hv)
            a['Centros_completados'] = _parsear_simple(av)
        return


def _extraer_stats_periodo(data_stats):
    """Itera sobre los periodos y devuelve stats del periodo ALL."""
    for periodo in data_stats.get('statistics', []):
        if periodo.get('period') == 'ALL':
            return _extraer_stats(periodo.get('groups', []))
    return {}, {}


# ═══════════════════════════════════════════════════════════════════════════════
#  Sesión de Playwright (una sola instancia para toda la ejecución)
# ═══════════════════════════════════════════════════════════════════════════════

class SofaScoreSession:
    """Browser persistente que ejecuta todas las llamadas a la API de SofaScore."""

    def __init__(self, headless=True):
        self._pw      = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=headless,
            args=['--disable-blink-features=AutomationControlled'],
        )
        self._ctx  = self._browser.new_context(
            user_agent=(
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/131.0.0.0 Safari/537.36'
            ),
            locale='es-ES',
        )
        self._page = self._ctx.new_page()
        self._page.goto('https://www.sofascore.com/', wait_until='domcontentloaded', timeout=30000)
        print('  [Browser] SofaScore listo')

    def api(self, path):
        """Ejecuta una llamada fetch a la API de SofaScore desde el browser."""
        time.sleep(DELAY_API)
        js = f"""
            async () => {{
                const r = await fetch('{SS_BASE}{path}',
                    {{headers: {{'Accept': 'application/json'}}}});
                if (!r.ok) return {{_error: r.status}};
                return r.json();
            }}
        """
        result = self._page.evaluate(js)
        if isinstance(result, dict) and '_error' in result:
            print(f'  [SS] HTTP {result["_error"]} en {path}')
            return None
        return result

    def close(self):
        self._browser.close()
        self._pw.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  Funciones de consulta
# ═══════════════════════════════════════════════════════════════════════════════

def listar_temporadas(sess):
    data = sess.api(f'/unique-tournament/{PL_TOURNAMENT}/seasons')
    if not data:
        return []
    return [
        {'id': s['id'], 'name': s.get('name', ''), 'year': s.get('year', '')}
        for s in data.get('seasons', [])
    ]


def listar_partidos(sess, season_id, solo_terminados=True):
    data_rounds = sess.api(f'/unique-tournament/{PL_TOURNAMENT}/season/{season_id}/rounds')
    if not data_rounds:
        return []

    rounds = [r['round'] for r in data_rounds.get('rounds', [])]
    partidos = []

    for rnd in rounds:
        data = sess.api(
            f'/unique-tournament/{PL_TOURNAMENT}/season/{season_id}/events/round/{rnd}'
        )
        if not data:
            continue
        for ev in data.get('events', []):
            status = ev.get('status', {}).get('type', '')
            if solo_terminados and status != 'finished':
                continue
            ts  = ev.get('startTimestamp')
            from datetime import datetime
            fecha = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else None
            partidos.append({
                'event_id':   ev['id'],
                'home':       _norm_equipo(ev.get('homeTeam', {}).get('name', '')),
                'away':       _norm_equipo(ev.get('awayTeam', {}).get('name', '')),
                'fecha':      fecha,
                'goles_home': ev.get('homeScore', {}).get('current'),
                'goles_away': ev.get('awayScore', {}).get('current'),
                'round':      rnd,
            })
        print(f'    Jornada {rnd:2}: {sum(1 for p in partidos if p["round"] == rnd)} partidos terminados')

    return partidos


def obtener_stats(sess, event_id):
    """Devuelve (stats_home, stats_away) con columnas base → valor."""
    data = sess.api(f'/event/{event_id}/statistics')
    if not data:
        return {}, {}
    return _extraer_stats_periodo(data)


def obtener_ratings(sess, event_id):
    """Ratings promedio del once titular. Returns (rating_home, rating_away)."""
    data = sess.api(f'/event/{event_id}/lineups')
    if not data:
        return None, None

    def avg(players):
        vals = [
            float(p['statistics']['rating'])
            for p in players
            if not p.get('substitute', False)
            and p.get('statistics', {}).get('rating') is not None
        ]
        return round(sum(vals) / len(vals), 2) if vals else None

    return (
        avg(data.get('home', {}).get('players', [])),
        avg(data.get('away', {}).get('players', [])),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Construcción de fila del dataset
# ═══════════════════════════════════════════════════════════════════════════════

def construir_fila(partido, stats_h, stats_a, rating_h, rating_a, pid):
    row = {c: None for c in COLUMNAS_DATASET}

    # Metadatos
    row['Partido_id']    = pid
    row['Fase']          = f"Jornada {partido['round']}"
    row['Fecha']         = partido['fecha']
    row['Equipo1']       = partido['home']
    row['Es_Local_E1']   = 1
    row['Equipo2']       = partido['away']
    row['EQUIPO1_GOLES'] = partido['goles_home']
    row['EQUIPO2_GOLES'] = partido['goles_away']
    row['media11_titular_E1'] = rating_h
    row['media11_titular_E2'] = rating_a

    # Stats de SofaScore
    for col, val in stats_h.items():
        key = f'{col}_E1'
        if key in row:
            row[key] = val
    for col, val in stats_a.items():
        key = f'{col}_E2'
        if key in row:
            row[key] = val

    # Derivadas
    g1 = partido['goles_home']
    g2 = partido['goles_away']
    if g1 is not None and g2 is not None:
        row['goles_encajados_E1']  = g2
        row['goles_encajados_E2']  = g1
        row['Porterias_a_cero_E1'] = 1 if g2 == 0 else 0
        row['Porterias_a_cero_E2'] = 1 if g1 == 0 else 0

    for sfx in ('E1', 'E2'):
        cmp = row.get(f'Pases_completados_{sfx}')
        att = row.get(f'Pases_realizados_{sfx}')
        if cmp and att:
            row[f'Precision_pase_{sfx}'] = round(cmp / att * 100, 1)

        cc = row.get(f'Centros_completados_{sfx}')
        cr = row.get(f'Centros_realizados_{sfx}')
        if cc and cr:
            row[f'Precision_en_el_centro_{sfx}'] = round(cc / cr * 100, 1)

        tot  = row.get(f'Entradas_{sfx}')
        exito = row.get(f'Entradas_con_exito_{sfx}')
        if tot is not None and exito is not None:
            row[f'Entradas_perdidas_{sfx}'] = tot - exito

    return row


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline principal
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_temporada(season_id, season_name, headless=True):
    """
    Scrapea una temporada completa de Premier League y guarda en CSV.
    """
    print(f'\n=== Premier League {season_name} (SofaScore id={season_id}) ===')

    sess = SofaScoreSession(headless=headless)
    try:
        partidos = listar_partidos(sess, season_id, solo_terminados=True)
        print(f'\n  Total partidos terminados: {len(partidos)}')

        filas = []
        for i, p in enumerate(partidos, 1):
            eid   = p['event_id']
            label = f'{p["home"]} vs {p["away"]} ({p["fecha"]})'
            print(f'  [{i:3}/{len(partidos)}] {label}', end=' ', flush=True)

            stats_h, stats_a = obtener_stats(sess, eid)
            rating_h, rating_a = obtener_ratings(sess, eid)

            pid  = f'PL-{season_name}-{eid}'
            fila = construir_fila(p, stats_h, stats_a, rating_h, rating_a, pid)
            filas.append(fila)

            n_datos = sum(1 for v in fila.values() if v is not None)
            print(f'[{n_datos}/140 cols]  ✓')

    finally:
        sess.close()

    df = pd.DataFrame(filas, columns=COLUMNAS_DATASET)
    out = f'data/premier_{season_name}.csv'
    df.to_csv(out, index=False)

    n_no_nan = int(df.notna().sum().sum())
    total    = df.shape[0] * df.shape[1]
    print(f'\n  Guardado: {out}')
    print(f'  Filas: {len(df)} | Cobertura media: {n_no_nan/total*100:.1f}%')

    cobertura = (df.notna().mean() * 100).sort_values(ascending=False)
    cols_ok = cobertura[cobertura > 0]
    print(f'\n  Columnas con datos ({len(cols_ok)}/140):')
    print(cols_ok.to_string())

    return df


def combinar_csvs():
    import glob
    archivos = sorted(glob.glob('data/premier_*.csv'))
    if not archivos:
        print('No se encontraron archivos data/premier_*.csv')
        return
    df = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
    out = 'data/premier_combinado.csv'
    df.to_csv(out, index=False)
    print(f'Combinado: {len(df)} partidos → {out}')


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if '--temporadas' in sys.argv:
        sess = SofaScoreSession()
        try:
            print('\nTemporadas Premier League:')
            for t in listar_temporadas(sess):
                print(f"  id={t['id']:7}  {t['name']:30}  {t['year']}")
        finally:
            sess.close()

    elif '--partido' in sys.argv:
        idx = sys.argv.index('--partido')
        eid = int(sys.argv[idx + 1])
        sess = SofaScoreSession()
        try:
            sh, sa = obtener_stats(sess, eid)
            rh, ra = obtener_ratings(sess, eid)
            print(f'\nStats HOME: {json.dumps(sh, indent=2, ensure_ascii=False)}')
            print(f'\nStats AWAY: {json.dumps(sa, indent=2, ensure_ascii=False)}')
            print(f'\nRating HOME: {rh}  |  AWAY: {ra}')
        finally:
            sess.close()

    elif '--scrape' in sys.argv:
        idx = sys.argv.index('--scrape')
        try:
            season_id = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print('ERROR: falta el season_id.')
            print('Usa --temporadas para ver los IDs disponibles.')
            sys.exit(1)
        season_name = sys.argv[idx + 2] if idx + 2 < len(sys.argv) else str(season_id)
        visible = '--visible' in sys.argv
        scrape_temporada(season_id, season_name, headless=not visible)

    elif '--combinar' in sys.argv:
        combinar_csvs()

    else:
        print(__doc__)
        print('\nTemporadas conocidas:')
        print('  2024-25 → id=61627   (380 partidos)')
        print('  2023-24 → id=52186   (380 partidos)')
        print('  2022-23 → id=41886   (380 partidos)')
        print('\nEjemplo:')
        print('  python scraper/scraper_premier.py --scrape 61627 2024-25')
