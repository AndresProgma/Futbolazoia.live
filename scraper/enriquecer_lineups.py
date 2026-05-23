"""
Enriquece los datasets con estadísticas de lineups desde SofaScore.

Agrega por partido y por equipo (E1/E2):
  Formacion, Formacion_Def
  GK: Rating, Saves, GoalsPrevented
  DEF promedio: Rating, Tackles, Clearances
  MID promedio: Rating, KeyPasses, Recovery
  FWD promedio: Rating, Shots, Goals
  Top1/2/3 Rating (mejores 3 jugadores)
  Rating_Std (dispersión del equipo)

Total: 18 columnas × 2 equipos = 36 columnas nuevas.

Uso:
    python scraper/enriquecer_lineups.py --ucl          # solo UCL Excel
    python scraper/enriquecer_lineups.py --pl           # solo PL CSVs
    python scraper/enriquecer_lineups.py --todo         # ambos
"""

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent

SS_BASE       = 'https://api.sofascore.com/api/v1'
UCL_TOURN     = 7       # UEFA Champions League en SofaScore
UCL_SEASONS   = {
    '2024-25': 61644,
    '2025-26': 76953,
}
DELAY         = 0.8     # segundos entre llamadas API

# ─── Mapeo equipos UCL (SofaScore → dataset) ─────────────────────────────────

MAPEO_UCL = {
    'bsc young boys':          'Young Boys',
    'young boys':              'Young Boys',
    'bsc young boys fc':       'Young Boys',
    'aston villa':             'Aston Villa',
    'aston villa fc':          'Aston Villa',
    'juventus':                'Juventus',
    'juventus fc':             'Juventus',
    'psv eindhoven':           'PSV',
    'psv':                     'PSV',
    'ac milan':                'Milan',
    'milan':                   'Milan',
    'liverpool':               'Liverpool',
    'liverpool fc':            'Liverpool',
    'fc bayern münchen':       'Bayern Munchen',
    'fc bayern munich':        'Bayern Munchen',
    'bayern munich':           'Bayern Munchen',
    'gnk dinamo zagreb':       'Dinamo Zagreb',
    'dinamo zagreb':           'Dinamo Zagreb',
    'real madrid':             'Real Madrid',
    'real madrid cf':          'Real Madrid',
    'vfb stuttgart':           'Stuttgart',
    'stuttgart':               'Stuttgart',
    'rb salzburg':             'Salzburg',
    'red bull salzburg':       'Salzburg',
    'fc salzburg':             'Salzburg',
    'stade brestois 29':       'Brest',
    'stade brestois':          'Brest',
    'brest':                   'Brest',
    'paris saint-germain':     'Paris',
    'paris saint germain':     'Paris',
    'psg':                     'Paris',
    'paris':                   'Paris',
    'atletico de madrid':      'Atleti',
    'atletico madrid':         'Atleti',
    'atlético de madrid':      'Atleti',
    'atletico':                'Atleti',
    'atleti':                  'Atleti',
    'borussia dortmund':       'BDortmund',
    'bvb':                     'BDortmund',
    'club brugge':             'Club Brugge',
    'club brugge kv':          'Club Brugge',
    'atalanta':                'Atlanta',
    'atalanta bc':             'Atlanta',
    'feyenoord':               'Feyenoord',
    'fc barcelona':            'Barcelona',
    'barcelona':               'Barcelona',
    'arsenal':                 'Arsenal',
    'arsenal fc':              'Arsenal',
    'chelsea':                 'Chelsea',
    'chelsea fc':              'Chelsea',
    'slavia prague':           'Slavia Praha',
    'sk slavia praha':         'Slavia Praha',
    'slavia praha':            'Slavia Praha',
    'rb leipzig':              'Leipzig',
    'red bull leipzig':        'Leipzig',
    'as monaco':               'Monaco',
    'monaco':                  'Monaco',
    'benfica':                 'Benfica',
    'sl benfica':              'Benfica',
    'sporting cp':             'Sporting CP',
    'sporting clube de portugal': 'Sporting CP',
    'inter':                   'Inter',
    'fc internazionale milano': 'Inter',
    'internazionale':          'Inter',
    'inter milan':             'Inter',
    'bayer leverkusen':        'Leverkursen',
    'leverkusen':              'Leverkursen',
    'manchester city':         'Man City',
    'man city':                'Man City',
    'tottenham hotspur':       'Tottenham',
    'tottenham':               'Tottenham',
    'newcastle united':        'Newcastle',
    'newcastle':               'Newcastle',
    'girona':                  'Girona',
    'girona fc':               'Girona',
    'losc lille':              'Lille',
    'lille':                   'Lille',
    'eintracht frankfurt':     'Frankfurt',
    'frankfurt':               'Frankfurt',
    'galatasaray':             'Galatasaray',
    'galatasaray sk':          'Galatasaray',
    'fc shakhtar donetsk':     'Shakhtar',
    'shakhtar donetsk':        'Shakhtar',
    'shakhtar':                'Shakhtar',
    'ajax':                    'Ajax',
    'afc ajax':                'Ajax',
    'ajax amsterdam':          'Ajax',
    'olympiacos':              'Olympiacos',
    'olympiacos fc':           'Olympiacos',
    'athletic club':           'Athletic Club',
    'athletic bilbao':         'Athletic Club',
    'villarreal cf':           'Villareal',
    'villarreal':              'Villareal',
    'napoli':                  'Napoli',
    'ssc napoli':              'Napoli',
    'celtic':                  'Celtic',
    'celtic fc':               'Celtic',
    'fc bodo/glimt':           'Bodo',
    'bodo/glimt':              'Bodo',
    'bodo glimt':              'Bodo',
    'fk bodo/glimt':           'Bodo',
    'qarabag fk':              'Qarabag',
    'qarabag':                 'Qarabag',
    'pafos fc':                'Pafos',
    'pafos':                   'Pafos',
    'ac sparta praha':         'Sparta Praha',
    'sparta prague':           'Sparta Praha',
    'sparta praha':            'Sparta Praha',
    'sk slovan bratislava':    'Slovan Bratislava',
    'slovan bratislava':       'Slovan Bratislava',
    'fc red bull salzburg':    'Salzburg',
    'olympique de marseille':  'Marseille',
    'olympique marseille':     'Marseille',
    'marseille':               'Marseille',
    'qarabag fk':              'Qarabag',
    'qarabag':                 'Qarabag',
    'sk rapid vienna':         'Rapid Wien',
    'rapid wien':              'Rapid Wien',
    'fc midtjylland':          'Midtjylland',
    'midtjylland':             'Midtjylland',
    'rb bragantino':           'Bragantino',
    'vfb stuttgart':           'Stuttgart',
    'vfl bochum':              'Bochum',
    'sc freiburg':             'Freiburg',
    'borussia monchengladbach':'Gladbach',
    'fc porto':                'Porto',
    'porto':                   'Porto',
    'lazio':                   'Lazio',
    'ss lazio':                'Lazio',
    'as roma':                 'Roma',
    'roma':                    'Roma',
    'sevilla fc':              'Sevilla',
    'sevilla':                 'Sevilla',
    'real sociedad':           'Real Sociedad',
    'psv eindhoven':           'PSV',
    'sturm graz':              'Sturm Graz',
    'sk sturm graz':           'Sturm Graz',
    'bologna fc 1909':         'Bologna',
    'bologna fc':              'Bologna',
    'bologna':                 'Bologna',
    'royale union saint-gilloise': 'Union SG',
    'union saint-gilloise':    'Union SG',
    'union sg':                'Union SG',
    'everton':                 'Everton',
    'kairat almaty':           'Kairat Almaty',
    'fc kairat':               'Kairat Almaty',
}

MAPEO_PL = {
    'manchester city':          'Man City',
    'manchester united':        'Man United',
    'liverpool':                'Liverpool',
    'arsenal':                  'Arsenal',
    'chelsea':                  'Chelsea',
    'tottenham hotspur':        'Tottenham',
    'newcastle united':         'Newcastle',
    'aston villa':              'Aston Villa',
    'west ham united':          'West Ham',
    'brighton & hove albion':   'Brighton',
    'brentford':                'Brentford',
    'fulham':                   'Fulham',
    'crystal palace':           'Crystal Palace',
    'wolverhampton wanderers':  'Wolves',
    'everton':                  'Everton',
    'nottingham forest':        'Nottm Forest',
    'leicester city':           'Leicester',
    'southampton':              'Southampton',
    'burnley':                  'Burnley',
    'sheffield united':         'Sheffield Utd',
    'afc bournemouth':          'Bournemouth',
    'luton town':               'Luton',
    'ipswich town':             'Ipswich',
    'sunderland':               'Sunderland',
}


def _norm(nombre, mapeo):
    if not nombre:
        return nombre
    c = nombre.lower().strip()
    if c in mapeo:
        return mapeo[c]
    for k, v in mapeo.items():
        if k in c or c in k:
            return v
    return nombre.title()


# ─── Extracción de stats de lineups ──────────────────────────────────────────

def _stat(player, key, default=0):
    return player.get('statistics', {}).get(key) or default


def extraer_lineup_stats(data_lineups, side='home'):
    """
    Dado el JSON del endpoint /lineups de SofaScore, extrae stats agregadas
    para un equipo (side='home' o 'away').

    Returns dict con columnas base (sin sufijo _E1/_E2).
    """
    team = data_lineups.get(side, {})
    formation = team.get('formation', '')
    players   = [p for p in team.get('players', []) if not p.get('substitute', False)]

    if not players:
        return {}

    # Número de defensas de la formación ("4-3-3" → 4)
    def_count = 0
    if formation:
        parts = formation.split('-')
        try:
            def_count = int(parts[0])
        except (ValueError, IndexError):
            def_count = 0

    # Separar por posición
    gks   = [p for p in players if p.get('player', {}).get('position') == 'G']
    defs  = [p for p in players if p.get('player', {}).get('position') == 'D']
    mids  = [p for p in players if p.get('player', {}).get('position') == 'M']
    fwds  = [p for p in players if p.get('player', {}).get('position') == 'F']

    def _avg(lst, key):
        vals = [_stat(p, key) for p in lst if _stat(p, key) is not None]
        return round(float(np.mean(vals)), 3) if vals else np.nan

    # Portero
    gk = gks[0] if gks else None
    gk_rating = _stat(gk, 'rating') if gk else np.nan
    gk_saves  = _stat(gk, 'saves') if gk else np.nan
    gk_gp     = _stat(gk, 'goalsPrevented') if gk else np.nan

    # Ratings de todos para top-N y std
    all_ratings = [_stat(p, 'rating') for p in players if _stat(p, 'rating')]
    all_ratings_sorted = sorted(all_ratings, reverse=True)

    def _topn(n):
        return all_ratings_sorted[n-1] if len(all_ratings_sorted) >= n else np.nan

    return {
        'Formacion':          formation,
        'Formacion_Def':      def_count,
        'GK_Rating':          gk_rating,
        'GK_Saves':           gk_saves,
        'GK_GoalsPrevented':  round(float(gk_gp), 3) if gk_gp else np.nan,
        'Def_Avg_Rating':     _avg(defs, 'rating'),
        'Def_Avg_Tackles':    _avg(defs, 'totalTackle'),
        'Def_Avg_Clearances': _avg(defs, 'totalClearance'),
        'Mid_Avg_Rating':     _avg(mids, 'rating'),
        'Mid_Avg_KeyPasses':  _avg(mids, 'keyPass'),
        'Mid_Avg_Recovery':   _avg(mids, 'ballRecovery'),
        'Fwd_Avg_Rating':     _avg(fwds, 'rating'),
        'Fwd_Avg_Shots':      _avg(fwds, 'totalShots'),
        'Fwd_Avg_Goals':      _avg(fwds, 'goals'),
        'Top1_Rating':        _topn(1),
        'Top2_Rating':        _topn(2),
        'Top3_Rating':        _topn(3),
        'Rating_Std':         round(float(np.std(all_ratings)), 3) if len(all_ratings) > 1 else np.nan,
    }


# ─── Sesión Playwright ────────────────────────────────────────────────────────

class SSSession:
    def __init__(self):
        self._pw  = sync_playwright().start()
        self._br  = self._pw.chromium.launch(headless=True)
        self._ctx = self._br.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        self._pg  = self._ctx.new_page()
        self._pg.goto('https://www.sofascore.com/', wait_until='domcontentloaded', timeout=30000)
        print('  [Browser] listo')

    def api(self, path):
        time.sleep(DELAY)
        result = self._pg.evaluate(f"""
            async () => {{
                const r = await fetch('{SS_BASE}{path}',
                    {{headers:{{'Accept':'application/json'}}}});
                if (!r.ok) return {{_error: r.status}};
                return r.json();
            }}
        """)
        if isinstance(result, dict) and '_error' in result:
            return None
        return result

    def close(self):
        self._br.close()
        self._pw.stop()


# ─── Índice de partidos UCL en SofaScore ─────────────────────────────────────

def construir_indice_ucl(sess):
    """
    Descarga todos los partidos UCL (24-25 y 25-26) de SofaScore y construye
    {(home_norm, away_norm, fecha) → event_id}.
    """
    print('  Construyendo índice UCL en SofaScore...')
    indice = {}

    for season_name, season_id in UCL_SEASONS.items():
        data_rounds = sess.api(f'/unique-tournament/{UCL_TOURN}/season/{season_id}/rounds')
        if not data_rounds:
            continue
        rounds = [r['round'] for r in data_rounds.get('rounds', [])]

        for rnd in rounds:
            data = sess.api(
                f'/unique-tournament/{UCL_TOURN}/season/{season_id}/events/round/{rnd}'
            )
            if not data:
                continue
            for ev in data.get('events', []):
                if ev.get('status', {}).get('type') != 'finished':
                    continue
                home = _norm(ev.get('homeTeam', {}).get('name', ''), MAPEO_UCL)
                away = _norm(ev.get('awayTeam', {}).get('name', ''), MAPEO_UCL)
                ts   = ev.get('startTimestamp')
                if ts:
                    fecha = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                else:
                    continue
                eid = ev['id']
                indice[(home, away, fecha)] = eid

    print(f'  Índice UCL: {len(indice)} partidos encontrados')
    return indice


# ─── Pipeline de enriquecimiento ─────────────────────────────────────────────

LINEUP_COLS = [
    'Formacion', 'Formacion_Def',
    'GK_Rating', 'GK_Saves', 'GK_GoalsPrevented',
    'Def_Avg_Rating', 'Def_Avg_Tackles', 'Def_Avg_Clearances',
    'Mid_Avg_Rating', 'Mid_Avg_KeyPasses', 'Mid_Avg_Recovery',
    'Fwd_Avg_Rating', 'Fwd_Avg_Shots', 'Fwd_Avg_Goals',
    'Top1_Rating', 'Top2_Rating', 'Top3_Rating', 'Rating_Std',
]


def _norm_fecha(f):
    """Normaliza fecha a YYYY-MM-DD con ceros (2025-9-16 → 2025-09-16)."""
    try:
        return pd.to_datetime(str(f)).strftime('%Y-%m-%d')
    except Exception:
        return str(f)[:10]


def _agregar_columnas_vacias(df):
    for col in LINEUP_COLS:
        for sfx in ('_E1', '_E2'):
            c = f'{col}{sfx}'
            if c not in df.columns:
                # Formacion es string; el resto numérico
                df[c] = '' if col == 'Formacion' else np.nan
    return df


def enriquecer_dataset(df, sess, indice_ucl=None, es_pl=False):
    """
    Agrega stats de lineups a cada fila del DataFrame.
    - es_pl=True  → extrae event_id del Partido_id (formato PL-XXXX-{eid})
    - es_pl=False → busca en indice_ucl por (home, away, fecha)
    """
    df = _agregar_columnas_vacias(df)
    total = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        # Obtener event_id
        if es_pl:
            m = re.search(r'-(\d+)$', str(row.get('Partido_id', '')))
            eid = int(m.group(1)) if m else None
        else:
            home  = str(row.get('Equipo1', ''))
            away  = str(row.get('Equipo2', ''))
            fecha = str(row.get('Fecha', ''))
            eid   = indice_ucl.get((home, away, fecha))

        if not eid:
            # Segundo intento: normalizar fecha
            if not es_pl:
                fecha_norm = _norm_fecha(row.get('Fecha', ''))
                eid = indice_ucl.get((home, away, fecha_norm))
        if not eid:
            continue

        data = sess.api(f'/event/{eid}/lineups')
        if not data:
            continue

        stats_h = extraer_lineup_stats(data, 'home')
        stats_a = extraer_lineup_stats(data, 'away')

        for col, val in stats_h.items():
            df.at[idx, f'{col}_E1'] = val
        for col, val in stats_a.items():
            df.at[idx, f'{col}_E2'] = val

        total += 1
        if (i + 1) % 30 == 0:
            print(f'    {i+1}/{len(df)} procesados  ({total} con datos)')

    print(f'  Total enriquecidos: {total}/{len(df)}')
    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────

def enriquecer_ucl():
    print('\n=== UCL — enriqueciendo lineups ===')
    xlsx = ROOT / 'data' / 'creando_dataset_modificado.xlsx'
    df   = pd.read_excel(xlsx)
    print(f'  Cargado: {len(df)} partidos')

    sess = SSSession()
    try:
        indice = construir_indice_ucl(sess)
        df = enriquecer_dataset(df, sess, indice_ucl=indice, es_pl=False)
    finally:
        sess.close()

    out = ROOT / 'data' / 'creando_dataset_modificado_lineups.xlsx'
    df.to_excel(out, index=False)
    cobertura = df[[f'{c}_E1' for c in LINEUP_COLS if c != 'Formacion']].notna().mean().mean()
    print(f'  Guardado: {out}')
    print(f'  Cobertura lineup numérica: {cobertura*100:.1f}%')


def enriquecer_pl():
    print('\n=== PL — enriqueciendo lineups ===')
    csvs = [
        ROOT / 'data' / 'premier_2024-25_enriquecido.csv',
        ROOT / 'data' / 'premier_2025-26_enriquecido.csv',
    ]

    sess = SSSession()
    try:
        for csv_path in csvs:
            if not csv_path.exists():
                print(f'  No encontrado: {csv_path}')
                continue
            df = pd.read_csv(csv_path)
            print(f'\n  Procesando {csv_path.name} ({len(df)} partidos)...')
            df = enriquecer_dataset(df, sess, es_pl=True)
            out = str(csv_path).replace('.csv', '_lineups.csv')
            df.to_csv(out, index=False)
            cobertura = df[[f'{c}_E1' for c in LINEUP_COLS if c != 'Formacion']].notna().mean().mean()
            print(f'  Guardado: {out}  |  Cobertura: {cobertura*100:.1f}%')
    finally:
        sess.close()


if __name__ == '__main__':
    modo = sys.argv[1] if len(sys.argv) > 1 else '--todo'

    if modo == '--ucl':
        enriquecer_ucl()
    elif modo == '--pl':
        enriquecer_pl()
    elif modo == '--todo':
        enriquecer_ucl()
        enriquecer_pl()
    else:
        print(__doc__)
