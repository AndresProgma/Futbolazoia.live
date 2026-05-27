"""
Scraper de selecciones nacionales — SofaScore via Playwright.

Competiciones soportadas (usar clave en --scrape):
  nations_league      UEFA Nations League (todas las ligas)
  euro                UEFA Eurocopa
  wc_quals_uefa       Clasificatorias Mundial UEFA
  world_cup           FIFA World Cup
  copa_america        Copa América CONMEBOL
  eliminatorias       CONMEBOL Eliminatorias WC
  afcon               Copa África (AFCON)
  asian_cup           AFC Copa Asiática
  gold_cup            CONCACAF Gold Cup
  concacaf_nl         CONCACAF Nations League

Uso:
    python scraper/scraper_selecciones.py --torneos
    python scraper/scraper_selecciones.py --temporadas <clave_torneo>
    python scraper/scraper_selecciones.py --scrape <clave_torneo> <season_id> <nombre>
    python scraper/scraper_selecciones.py --partido <event_id>
    python scraper/scraper_selecciones.py --combinar
"""

import re
import sys
import json
import time
import glob

import pandas as pd
from playwright.sync_api import sync_playwright

# ─── Importar utilidades del scraper PL ───────────────────────────────────────
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scraper.scraper_premier import (
    _parsear_simple,
    _parsear_ratio,
    _extraer_stats,
    _extraer_stats_periodo,
    obtener_stats,
    obtener_ratings,
    COLUMNAS_DATASET,
    SofaScoreSession,
)

# ─── Constantes ───────────────────────────────────────────────────────────────

SS_BASE    = 'https://api.sofascore.com/api/v1'
DELAY_API  = 0.9

# IDs de torneos en SofaScore (unique-tournament IDs) — verificados en URLs del sitio
TORNEOS = {
    'nations_league': {'id': 10783, 'nombre': 'UEFA Nations League',          'conf': 'UEFA'},
    'euro':           {'id': 1,     'nombre': 'UEFA Eurocopa',                'conf': 'UEFA'},
    'wc_quals_uefa':  {'id': 11,    'nombre': 'Clasificatorias WC UEFA',      'conf': 'UEFA'},
    'world_cup':      {'id': 16,    'nombre': 'FIFA World Cup',               'conf': 'FIFA'},
    'copa_america':   {'id': 133,   'nombre': 'Copa América',                 'conf': 'CONMEBOL'},
    'eliminatorias':  {'id': 295,   'nombre': 'CONMEBOL Eliminatorias',       'conf': 'CONMEBOL'},
    'afcon':          {'id': 270,   'nombre': 'AFCON',                        'conf': 'CAF'},
    'asian_cup':      {'id': 246,   'nombre': 'AFC Asian Cup',                'conf': 'AFC'},
    'gold_cup':       {'id': 140,   'nombre': 'CONCACAF Gold Cup',            'conf': 'CONCACAF'},
    'concacaf_nl':    {'id': 14100, 'nombre': 'CONCACAF Nations League',      'conf': 'CONCACAF'},
}

# ─── Columnas extra para selecciones ──────────────────────────────────────────

COLUMNAS_EXTRA = ['Torneo', 'Confederacion_E1', 'Confederacion_E2']
COLUMNAS_SELECCIONES = COLUMNAS_DATASET + COLUMNAS_EXTRA

# ─── Normalización de nombres de selecciones ──────────────────────────────────

MAPEO_SELECCIONES = {
    # UEFA — Europa
    'spain': 'España', 'españa': 'España',
    'france': 'Francia', 'francia': 'Francia',
    'germany': 'Alemania', 'alemania': 'Alemania',
    'england': 'Inglaterra', 'inglaterra': 'Inglaterra',
    'portugal': 'Portugal',
    'netherlands': 'Países Bajos', 'holland': 'Países Bajos',
    'italy': 'Italia', 'italia': 'Italia',
    'belgium': 'Bélgica', 'bélgica': 'Bélgica',
    'croatia': 'Croacia', 'hrvatska': 'Croacia',
    'denmark': 'Dinamarca', 'dinamarca': 'Dinamarca',
    'austria': 'Austria',
    'switzerland': 'Suiza', 'suiza': 'Suiza',
    'sweden': 'Suecia', 'suecia': 'Suecia',
    'norway': 'Noruega', 'noruega': 'Noruega',
    'poland': 'Polonia', 'polonia': 'Polonia',
    'ukraine': 'Ucrania', 'ucrania': 'Ucrania',
    'türkiye': 'Turquía', 'turkey': 'Turquía', 'turquía': 'Turquía',
    'serbia': 'Serbia',
    'czech republic': 'República Checa', 'czechia': 'República Checa',
    'slovakia': 'Eslovaquia',
    'hungary': 'Hungría',
    'romania': 'Rumania', 'românia': 'Rumania',
    'scotland': 'Escocia',
    'wales': 'Gales',
    'northern ireland': 'Irlanda del Norte',
    'republic of ireland': 'Irlanda', 'ireland': 'Irlanda',
    'greece': 'Grecia',
    'russia': 'Rusia',
    'israel': 'Israel',
    'finland': 'Finlandia',
    'slovenia': 'Eslovenia',
    'albania': 'Albania',
    'georgia': 'Georgia',
    'iceland': 'Islandia',
    'north macedonia': 'Macedonia del Norte',
    'bulgaria': 'Bulgaria',
    'bosnia and herzegovina': 'Bosnia', 'bosnia & herzegovina': 'Bosnia',
    'montenegro': 'Montenegro',
    'kosovo': 'Kosovo',
    'moldova': 'Moldavia',
    'belarus': 'Bielorrusia',
    'estonia': 'Estonia',
    'latvia': 'Letonia',
    'lithuania': 'Lituania',
    'luxembourg': 'Luxemburgo',
    'malta': 'Malta',
    'cyprus': 'Chipre',
    'faroe islands': 'Islas Feroe',
    'liechtenstein': 'Liechtenstein',
    'andorra': 'Andorra',
    'san marino': 'San Marino',
    'gibraltar': 'Gibraltar',
    'armenia': 'Armenia',
    'azerbaijan': 'Azerbaiyán',
    'kazakhstan': 'Kazajistán',

    # CONMEBOL — Sudamérica
    'brazil': 'Brasil', 'brasil': 'Brasil',
    'argentina': 'Argentina',
    'colombia': 'Colombia',
    'uruguay': 'Uruguay',
    'chile': 'Chile',
    'ecuador': 'Ecuador',
    'peru': 'Perú', 'perú': 'Perú',
    'bolivia': 'Bolivia',
    'paraguay': 'Paraguay',
    'venezuela': 'Venezuela',

    # CONCACAF — Norte/Centro América + Caribe
    'mexico': 'México', 'méxico': 'México',
    'united states': 'USA', 'usa': 'USA', 'us': 'USA',
    'canada': 'Canadá',
    'costa rica': 'Costa Rica',
    'panama': 'Panamá', 'panamá': 'Panamá',
    'jamaica': 'Jamaica',
    'honduras': 'Honduras',
    'el salvador': 'El Salvador',
    'trinidad and tobago': 'Trinidad y Tobago',
    'haiti': 'Haití',
    'cuba': 'Cuba',
    'guatemala': 'Guatemala',
    'curaçao': 'Curazao', 'curacao': 'Curazao',

    # CAF — África
    'morocco': 'Marruecos', 'marruecos': 'Marruecos',
    'senegal': 'Senegal',
    'nigeria': 'Nigeria',
    'egypt': 'Egipto', 'egipto': 'Egipto',
    'ivory coast': 'Costa de Marfil', "côte d'ivoire": 'Costa de Marfil',
    'cameroon': 'Camerún',
    'ghana': 'Ghana',
    'south africa': 'Sudáfrica',
    'tunisia': 'Túnez',
    'algeria': 'Argelia',
    'mali': 'Mali',
    'burkina faso': 'Burkina Faso',
    'guinea': 'Guinea',
    'democratic republic of congo': 'RD Congo', 'dr congo': 'RD Congo',
    'zambia': 'Zambia',
    'cape verde': 'Cabo Verde',
    'equatorial guinea': 'Guinea Ecuatorial',
    'mozambique': 'Mozambique',
    'angola': 'Angola',
    'kenya': 'Kenia',
    'ethiopia': 'Etiopía',
    'tanzania': 'Tanzania',
    'uganda': 'Uganda',
    'zimbabwe': 'Zimbabue',
    'gabon': 'Gabón',
    'namibia': 'Namibia',
    'benin': 'Benín',

    # AFC — Asia
    'japan': 'Japón', 'japón': 'Japón',
    'south korea': 'Corea del Sur', 'korea republic': 'Corea del Sur',
    'saudi arabia': 'Arabia Saudita', 'arabia saudita': 'Arabia Saudita',
    'iran': 'Irán', 'islamic republic of iran': 'Irán',
    'australia': 'Australia',
    'qatar': 'Catar',
    'iraq': 'Iraq',
    'united arab emirates': 'EAU',
    'uzbekistan': 'Uzbekistán',
    'china pr': 'China', 'china': 'China',
    'thailand': 'Tailandia',
    'vietnam': 'Vietnam',
    'india': 'India',
    'bahrain': 'Baréin',
    'oman': 'Omán',
    'jordan': 'Jordania',
    'north korea': 'Corea del Norte', 'korea dpr': 'Corea del Norte',
    'indonesia': 'Indonesia',
    'tajikistan': 'Tayikistán',
    'palestine': 'Palestina',
    'hong kong': 'Hong Kong',
    'kyrgyzstan': 'Kirguistán',

    # OFC — Oceanía
    'new zealand': 'Nueva Zelanda',
    'fiji': 'Fiyi',
    'papua new guinea': 'Papúa Nueva Guinea',
}

# Confederación por equipo (normalizado)
CONFEDERACION_EQUIPO = {
    # UEFA
    'España': 'UEFA', 'Francia': 'UEFA', 'Alemania': 'UEFA', 'Inglaterra': 'UEFA',
    'Portugal': 'UEFA', 'Países Bajos': 'UEFA', 'Italia': 'UEFA', 'Bélgica': 'UEFA',
    'Croacia': 'UEFA', 'Dinamarca': 'UEFA', 'Austria': 'UEFA', 'Suiza': 'UEFA',
    'Suecia': 'UEFA', 'Noruega': 'UEFA', 'Polonia': 'UEFA', 'Ucrania': 'UEFA',
    'Turquía': 'UEFA', 'Serbia': 'UEFA', 'República Checa': 'UEFA',
    'Eslovaquia': 'UEFA', 'Hungría': 'UEFA', 'Rumania': 'UEFA', 'Escocia': 'UEFA',
    'Gales': 'UEFA', 'Irlanda del Norte': 'UEFA', 'Irlanda': 'UEFA',
    'Grecia': 'UEFA', 'Rusia': 'UEFA', 'Israel': 'UEFA', 'Finlandia': 'UEFA',
    'Eslovenia': 'UEFA', 'Albania': 'UEFA', 'Georgia': 'UEFA', 'Islandia': 'UEFA',
    'Macedonia del Norte': 'UEFA', 'Bulgaria': 'UEFA', 'Bosnia': 'UEFA',
    'Montenegro': 'UEFA', 'Kosovo': 'UEFA', 'Moldavia': 'UEFA',
    'Bielorrusia': 'UEFA', 'Estonia': 'UEFA', 'Letonia': 'UEFA',
    'Lituania': 'UEFA', 'Luxemburgo': 'UEFA', 'Malta': 'UEFA', 'Chipre': 'UEFA',
    'Islas Feroe': 'UEFA', 'Liechtenstein': 'UEFA', 'Andorra': 'UEFA',
    'San Marino': 'UEFA', 'Gibraltar': 'UEFA', 'Armenia': 'UEFA',
    'Azerbaiyán': 'UEFA', 'Kazajistán': 'UEFA',
    # CONMEBOL
    'Brasil': 'CONMEBOL', 'Argentina': 'CONMEBOL', 'Colombia': 'CONMEBOL',
    'Uruguay': 'CONMEBOL', 'Chile': 'CONMEBOL', 'Ecuador': 'CONMEBOL',
    'Perú': 'CONMEBOL', 'Bolivia': 'CONMEBOL', 'Paraguay': 'CONMEBOL',
    'Venezuela': 'CONMEBOL',
    # CONCACAF
    'México': 'CONCACAF', 'USA': 'CONCACAF', 'Canadá': 'CONCACAF',
    'Costa Rica': 'CONCACAF', 'Panamá': 'CONCACAF', 'Jamaica': 'CONCACAF',
    'Honduras': 'CONCACAF', 'El Salvador': 'CONCACAF', 'Trinidad y Tobago': 'CONCACAF',
    'Haití': 'CONCACAF', 'Cuba': 'CONCACAF', 'Guatemala': 'CONCACAF',
    'Curazao': 'CONCACAF',
    # CAF
    'Marruecos': 'CAF', 'Senegal': 'CAF', 'Nigeria': 'CAF', 'Egipto': 'CAF',
    'Costa de Marfil': 'CAF', 'Camerún': 'CAF', 'Ghana': 'CAF',
    'Sudáfrica': 'CAF', 'Túnez': 'CAF', 'Argelia': 'CAF', 'Mali': 'CAF',
    'Burkina Faso': 'CAF', 'Guinea': 'CAF', 'RD Congo': 'CAF', 'Zambia': 'CAF',
    'Cabo Verde': 'CAF', 'Guinea Ecuatorial': 'CAF', 'Mozambique': 'CAF',
    'Angola': 'CAF', 'Kenia': 'CAF', 'Etiopía': 'CAF', 'Tanzania': 'CAF',
    'Uganda': 'CAF', 'Zimbabue': 'CAF', 'Gabón': 'CAF', 'Namibia': 'CAF',
    'Benín': 'CAF',
    # AFC
    'Japón': 'AFC', 'Corea del Sur': 'AFC', 'Arabia Saudita': 'AFC',
    'Irán': 'AFC', 'Australia': 'AFC', 'Catar': 'AFC', 'Iraq': 'AFC',
    'EAU': 'AFC', 'Uzbekistán': 'AFC', 'China': 'AFC', 'Tailandia': 'AFC',
    'Vietnam': 'AFC', 'India': 'AFC', 'Baréin': 'AFC', 'Omán': 'AFC',
    'Jordania': 'AFC', 'Corea del Norte': 'AFC', 'Indonesia': 'AFC',
    'Tayikistán': 'AFC', 'Palestina': 'AFC', 'Hong Kong': 'AFC',
    'Kirguistán': 'AFC',
    # OFC
    'Nueva Zelanda': 'OFC', 'Fiyi': 'OFC', 'Papúa Nueva Guinea': 'OFC',
}


def _norm_seleccion(nombre):
    if not nombre:
        return nombre
    c = nombre.lower().strip()
    if c in MAPEO_SELECCIONES:
        return MAPEO_SELECCIONES[c]
    for k, v in MAPEO_SELECCIONES.items():
        if k in c or c in k:
            return v
    return nombre.title()


def _confederacion(equipo_norm):
    return CONFEDERACION_EQUIPO.get(equipo_norm, 'Desconocido')


# ═══════════════════════════════════════════════════════════════════════════════
#  Listado de temporadas y partidos
# ═══════════════════════════════════════════════════════════════════════════════

def listar_temporadas(sess, torneo_id):
    data = sess.api(f'/unique-tournament/{torneo_id}/seasons')
    if not data:
        return []
    return [
        {'id': s['id'], 'name': s.get('name', ''), 'year': s.get('year', '')}
        for s in data.get('seasons', [])
    ]


def listar_partidos(sess, torneo_id, season_id):
    data_rounds = sess.api(f'/unique-tournament/{torneo_id}/season/{season_id}/rounds')
    if not data_rounds:
        return []

    rounds_raw = data_rounds.get('rounds', [])
    if not rounds_raw:
        return []

    # Construir mapa round_num → fase_nombre
    fase_map = {}
    for r in rounds_raw:
        rnum = r.get('round')
        prefix = r.get('prefix', '')
        name   = r.get('name', '')
        if prefix:
            fase_map[rnum] = prefix
        elif name:
            fase_map[rnum] = name
        else:
            fase_map[rnum] = f'Ronda {rnum}'

    rounds = [r['round'] for r in rounds_raw]
    partidos = []

    for rnd in rounds:
        data = sess.api(
            f'/unique-tournament/{torneo_id}/season/{season_id}/events/round/{rnd}'
        )
        if not data:
            continue
        for ev in data.get('events', []):
            status = ev.get('status', {}).get('type', '')
            if status != 'finished':
                continue
            ts    = ev.get('startTimestamp')
            from datetime import datetime
            fecha = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else None
            partidos.append({
                'event_id':   ev['id'],
                'home':       _norm_seleccion(ev.get('homeTeam', {}).get('name', '')),
                'away':       _norm_seleccion(ev.get('awayTeam', {}).get('name', '')),
                'fecha':      fecha,
                'goles_home': ev.get('homeScore', {}).get('current'),
                'goles_away': ev.get('awayScore', {}).get('current'),
                'round':      rnd,
                'fase':       fase_map.get(rnd, f'Ronda {rnd}'),
            })

        n_rnd = sum(1 for p in partidos if p['round'] == rnd)
        if n_rnd:
            print(f'    {fase_map.get(rnd, f"Ronda {rnd}"):30} {n_rnd} partidos')

    return partidos


# ═══════════════════════════════════════════════════════════════════════════════
#  Construcción de fila
# ═══════════════════════════════════════════════════════════════════════════════

def construir_fila(partido, stats_h, stats_a, rating_h, rating_a, pid, nombre_torneo):
    row = {c: None for c in COLUMNAS_SELECCIONES}

    e1 = partido['home']
    e2 = partido['away']

    # Metadatos
    row['Partido_id']       = pid
    row['Fase']             = partido['fase']
    row['Fecha']            = partido['fecha']
    row['Equipo1']          = e1
    row['Es_Local_E1']      = 1
    row['Equipo2']          = e2
    row['EQUIPO1_GOLES']    = partido['goles_home']
    row['EQUIPO2_GOLES']    = partido['goles_away']
    row['media11_titular_E1'] = rating_h
    row['media11_titular_E2'] = rating_a

    # Extra selecciones
    row['Torneo']           = nombre_torneo
    row['Confederacion_E1'] = _confederacion(e1)
    row['Confederacion_E2'] = _confederacion(e2)

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

        tot   = row.get(f'Entradas_{sfx}')
        exito = row.get(f'Entradas_con_exito_{sfx}')
        if tot is not None and exito is not None:
            row[f'Entradas_perdidas_{sfx}'] = tot - exito

    return row


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline principal
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_torneo(clave, season_id, season_name, headless=True):
    if clave not in TORNEOS:
        print(f'ERROR: clave "{clave}" no reconocida. Claves válidas:')
        for k in TORNEOS:
            print(f'  {k}')
        return None

    t = TORNEOS[clave]
    torneo_id     = t['id']
    nombre_torneo = t['nombre']

    print(f'\n=== {nombre_torneo} — {season_name} (SofaScore id={torneo_id}, season={season_id}) ===')

    sess = SofaScoreSession(headless=headless)
    try:
        partidos = listar_partidos(sess, torneo_id, season_id)
        print(f'\n  Total partidos terminados: {len(partidos)}')

        if not partidos:
            print('  Sin partidos encontrados. Verifica el season_id con --temporadas.')
            return None

        filas = []
        for i, p in enumerate(partidos, 1):
            eid   = p['event_id']
            label = f'{p["home"]} vs {p["away"]} ({p["fecha"]})'
            print(f'  [{i:3}/{len(partidos)}] {label}', end=' ', flush=True)

            try:
                sh, sa = obtener_stats(sess, eid)
                rh, ra = obtener_ratings(sess, eid)
            except Exception as e:
                print(f'[ERROR: {e}] — guardando parcial y saliendo')
                break

            pid  = f'SEL-{clave.upper()}-{season_name}-{eid}'
            fila = construir_fila(p, sh, sa, rh, ra, pid, nombre_torneo)
            filas.append(fila)

            n_datos = sum(1 for v in fila.values() if v is not None)
            print(f'[{n_datos} cols]  ✓')

    finally:
        sess.close()

    if not filas:
        print('  Sin datos capturados.')
        return None

    df  = pd.DataFrame(filas, columns=COLUMNAS_SELECCIONES)
    out = f'data/selecciones_{clave}_{season_name}.csv'
    df.to_csv(out, index=False)

    n_ok  = int(df.notna().sum().sum())
    total = df.shape[0] * df.shape[1]
    print(f'\n  Guardado: {out}')
    print(f'  Filas: {len(df)} | Cobertura media: {n_ok/total*100:.1f}%')

    cobertura = (df.notna().mean() * 100).sort_values(ascending=False)
    cols_ok   = cobertura[cobertura > 0]
    print(f'\n  Columnas con datos ({len(cols_ok)}/{len(COLUMNAS_SELECCIONES)}):')
    print(cols_ok.to_string())

    return df


def combinar_csvs():
    archivos = sorted(
        f for f in glob.glob('data/selecciones_*.csv')
        if not f.endswith('selecciones_combinado.csv')
    )
    if not archivos:
        print('No se encontraron archivos data/selecciones_*.csv')
        return
    dfs = []
    for f in archivos:
        df = pd.read_csv(f)
        print(f'  {f}: {len(df)} partidos')
        dfs.append(df)
    combinado = pd.concat(dfs, ignore_index=True)
    out = 'data/selecciones_combinado.csv'
    combinado.to_csv(out, index=False)
    print(f'\nCombinado: {len(combinado)} partidos → {out}')
    print(f'Torneos: {combinado["Torneo"].value_counts().to_dict()}')


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    if '--torneos' in sys.argv:
        print('\nTorneos disponibles:')
        print(f'  {"Clave":<25} {"ID":>6}  Nombre')
        print('  ' + '-' * 60)
        for k, v in TORNEOS.items():
            print(f'  {k:<25} {v["id"]:>6}  {v["nombre"]}')
        print('\nUsa --temporadas <clave> para ver los IDs de cada temporada.')

    elif '--temporadas' in sys.argv:
        idx   = sys.argv.index('--temporadas')
        clave = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if not clave or clave not in TORNEOS:
            print('ERROR: especifica una clave válida. Ejemplo:')
            print('  python scraper/scraper_selecciones.py --temporadas nations_league_a')
            sys.exit(1)
        t = TORNEOS[clave]
        sess = SofaScoreSession()
        try:
            temporadas = listar_temporadas(sess, t['id'])
            print(f'\nTemporadas de {t["nombre"]}:')
            for s in temporadas:
                print(f"  id={s['id']:7}  {s['name']:40}  {s['year']}")
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
            clave      = sys.argv[idx + 1]
            season_id  = int(sys.argv[idx + 2])
            season_name = sys.argv[idx + 3]
        except (IndexError, ValueError):
            print('ERROR: uso correcto:')
            print('  python scraper/scraper_selecciones.py --scrape <clave> <season_id> <nombre>')
            print('  Ejemplo:')
            print('  python scraper/scraper_selecciones.py --scrape nations_league 57478 2024-25')
            sys.exit(1)
        visible = '--visible' in sys.argv
        scrape_torneo(clave, season_id, season_name, headless=not visible)

    elif '--combinar' in sys.argv:
        combinar_csvs()

    else:
        print(__doc__)
        print('\nClaves de torneos disponibles:')
        for k, v in TORNEOS.items():
            print(f'  {k:<25} {v["nombre"]}')
        print('\nPaso 1: descubrir season_ids')
        print('  python scraper/scraper_selecciones.py --temporadas nations_league_a')
        print('\nPaso 2: scrape')
        print('  python scraper/scraper_selecciones.py --scrape nations_league 57478 2024-25')
        print('\nPaso 3: combinar todo')
        print('  python scraper/scraper_selecciones.py --combinar')
