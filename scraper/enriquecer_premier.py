"""
Enriquece data/premier_XXXX-XX.csv con stats adicionales de 2 fuentes:

  1. SofaScore /incidents  → Goles_encajados_propia_puerta,
                             Penaltis_marcados/forzados (desde incidencias reales)
  2. Understat.com         → Goles_dentro_area, Goles_Fuera_Area,
                             Disparos_a_puerta_fuera_del_area,
                             Disparos_fuera_desde_fuera_del_area

Uso:
    python scraper/enriquecer_premier.py data/premier_2024-25.csv
    python scraper/enriquecer_premier.py data/premier_2024-25.csv --solo-ss
    python scraper/enriquecer_premier.py data/premier_2024-25.csv --solo-understat
"""

import re
import sys
import time
import json
import requests

import pandas as pd
from playwright.sync_api import sync_playwright

SS_BASE  = 'https://api.sofascore.com/api/v1'
US_BASE  = 'https://understat.com'
DELAY_SS = 0.8
DELAY_US = 3.5

US_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept':     'text/html,application/xhtml+xml',
}

# ─── Understat team name → dataset name ──────────────────────────────────────

US_NORM = {
    'Manchester City':          'Man City',
    'Manchester United':        'Man United',
    'Liverpool':                'Liverpool',
    'Arsenal':                  'Arsenal',
    'Chelsea':                  'Chelsea',
    'Tottenham':                'Tottenham',
    'Newcastle United':         'Newcastle',
    'Aston Villa':              'Aston Villa',
    'West Ham':                 'West Ham',
    'Brighton':                 'Brighton',
    'Brentford':                'Brentford',
    'Fulham':                   'Fulham',
    'Crystal Palace':           'Crystal Palace',
    'Wolverhampton Wanderers':  'Wolves',
    'Everton':                  'Everton',
    'Nottingham Forest':        'Nottm Forest',
    'Leicester':                'Leicester',
    'Southampton':              'Southampton',
    'Burnley':                  'Burnley',
    'Sheffield United':         'Sheffield Utd',
    'Bournemouth':              'Bournemouth',
    'Luton':                    'Luton',
    'Ipswich':                  'Ipswich',
}

# Understat coords: X es distancia desde la portería atacada (0=línea de meta, 1=otro extremo)
# El área grande empieza aprox en X=0.83 (16.5m del gol sobre campo de 105m)
AREA_X_MIN = 0.83


# ═══════════════════════════════════════════════════════════════════════════════
#  Sesión SofaScore (Playwright persistente)
# ═══════════════════════════════════════════════════════════════════════════════

class BrowserSession:
    """
    Sesión Playwright compartida para SofaScore (via fetch) y Understat (via goto).
    Abre dos páginas: una anclada en SofaScore, otra navegable para Understat.
    """
    def __init__(self):
        self._pw      = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._ctx     = self._browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        # Página 1: anclada en SofaScore para fetch API
        self._ss_page = self._ctx.new_page()
        self._ss_page.goto('https://www.sofascore.com/', wait_until='domcontentloaded', timeout=30000)
        # Página 2: libre para navegar a Understat
        self.us_page  = self._ctx.new_page()
        print('  [Browser] Listo (SofaScore + Understat)')

    def api(self, path):
        time.sleep(DELAY_SS)
        result = self._ss_page.evaluate(f"""
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

    def ss_api(self, path):
        return self.api(path)

    def close(self):
        self._browser.close()
        self._pw.stop()


# Alias para retrocompatibilidad con enriquecer_con_incidents
class SofaScoreSession(BrowserSession):
    def api(self, path):
        return self.ss_api(path)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. SOFASCORE INCIDENTS
# ═══════════════════════════════════════════════════════════════════════════════

def _extraer_incidents(data):
    """
    Extrae de la lista de incidencias:
      - Goles_encajados_propia_puerta por equipo (isOwnGoal + isHome)
      - Penaltis_marcados por equipo (goalType == 'penalty')
      - Penaltis_forzados por equipo (el rival marcó penalti)
      - Penaltis_parados por equipo (penaltyMissed con resultado 'save')
    """
    h = {k: 0 for k in [
        'Goles_encajados_propia_puerta', 'Penaltis_marcados',
        'Penaltis_forzados', 'Penaltis_parados',
    ]}
    a = {k: 0 for k in h}

    for inc in data.get('incidents', []):
        itype  = inc.get('incidentType', '')
        is_home = inc.get('isHome')      # True=home, False=away

        # ── Gol (normal o en propia) ──────────────────────────────────────
        if itype == 'goal':
            own  = inc.get('isOwnGoal', False)
            gtype = inc.get('goalType', '')

            if own:
                # Propia puerta: el error fue del equipo is_home
                if is_home is True:
                    h['Goles_encajados_propia_puerta'] += 1
                elif is_home is False:
                    a['Goles_encajados_propia_puerta'] += 1

            if gtype == 'penalty':
                if is_home is True:
                    h['Penaltis_marcados'] += 1
                    a['Penaltis_forzados'] += 1
                elif is_home is False:
                    a['Penaltis_marcados'] += 1
                    h['Penaltis_forzados'] += 1

        # ── Penalti fallado / parado ──────────────────────────────────────
        # SofaScore usa incidentType='penaltyMissed' tanto para parados como
        # para disparos fuera; el campo 'reason' indica si fue 'save' o 'missed'
        if itype == 'penaltyMissed':
            reason = inc.get('reason', '')      # 'save', 'missed', 'post'
            if reason == 'save':
                # El portero contrario paró → portero is del equipo contrario
                if is_home is True:    # El delantero home disparó → portero away paró
                    a['Penaltis_parados'] += 1
                elif is_home is False:
                    h['Penaltis_parados'] += 1

    return h, a


def enriquecer_con_incidents(df, sess):
    """Agrega stats desde SofaScore /incidents."""
    print('\n[SS incidents] Procesando...')
    total = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        pid = str(row.get('Partido_id', ''))
        m   = re.search(r'-(\d+)$', pid)
        if not m:
            continue
        eid = m.group(1)

        data = sess.api(f'/event/{eid}/incidents')
        if not data:
            continue

        ih, ia = _extraer_incidents(data)

        changed = False
        for col, val in ih.items():
            key = f'{col}_E1'
            if key in df.columns and val > 0:
                df.at[idx, key] = val
                changed = True
        for col, val in ia.items():
            key = f'{col}_E2'
            if key in df.columns and val > 0:
                df.at[idx, key] = val
                changed = True

        if changed:
            total += 1
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(df)} procesados ({total} con nuevos datos)')

    print(f'  [SS incidents] {total} partidos con datos nuevos')
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  2. UNDERSTAT — shot data para goles y disparos por zona
# ═══════════════════════════════════════════════════════════════════════════════

def construir_indice_understat(page, season_year='2024'):
    """
    Usa Playwright para extraer window.datesData de la liga EPL.
    Devuelve {(home_norm, fecha) → understat_match_id}.
    """
    print(f'  [Understat] Cargando índice temporada {season_year}...')
    try:
        page.goto(f'{US_BASE}/league/EPL/{season_year}',
                  wait_until='networkidle', timeout=40000)
        time.sleep(1)
        data = page.evaluate("() => typeof window.datesData !== 'undefined' ? window.datesData : null")
    except Exception as e:
        print(f'  [Understat] ERROR cargando índice: {e}')
        return {}

    if not data:
        print('  [Understat] datesData no encontrado en window')
        return {}

    indice = {}
    for match in data:
        home_raw = match.get('h', {}).get('title', '') if isinstance(match.get('h'), dict) else ''
        home     = US_NORM.get(home_raw, home_raw)
        fecha    = (match.get('datetime', '') or '')[:10]
        mid      = match.get('id')
        if home and fecha and mid:
            indice[(home, fecha)] = mid

    print(f'  [Understat] Índice: {len(indice)} partidos')
    return indice


def _calcular_stats_shots(shots_dict, side='h'):
    """
    Convierte shots de un lado en stats para el dataset.
    shots_dict: {'h': [...], 'a': [...]}
    side: 'h' (home) o 'a' (away)
    """
    stats = {
        'Goles_dentro_area':                   0,
        'Goles_Fuera_Area':                    0,
        'Disparos_a_puerta_fuera_del_area':    0,
        'Disparos_fuera_desde_fuera_del_area': 0,
        'Goles_encajados_propia_puerta':       0,
    }

    # shotsData puede ser {'h': [...], 'a': [...]} o una lista plana
    if isinstance(shots_dict, dict):
        disparos = shots_dict.get(side, [])
    elif isinstance(shots_dict, list):
        disparos = [s for s in shots_dict if s.get('h_a') == side]
    else:
        return stats

    for s in disparos:
        try:
            x = float(s.get('X', 0))
        except (ValueError, TypeError):
            x = 0
        result    = s.get('result', '')
        situation = s.get('situation', '')
        dentro    = x >= AREA_X_MIN

        if result == 'Goal':
            if situation == 'OwnGoal':
                stats['Goles_encajados_propia_puerta'] += 1
            elif dentro:
                stats['Goles_dentro_area'] += 1
            else:
                stats['Goles_Fuera_Area'] += 1
        elif result == 'SavedShot' and not dentro:
            stats['Disparos_a_puerta_fuera_del_area'] += 1
        elif result in ('MissedShots', 'BlockedShot', 'ShotOnPost') and not dentro:
            stats['Disparos_fuera_desde_fuera_del_area'] += 1

    return stats


def enriquecer_con_understat(df, page, season_year='2024'):
    """Agrega stats de zona de disparo desde Understat usando Playwright."""
    print(f'\n[Understat] Procesando...')
    indice = construir_indice_understat(page, season_year)
    if not indice:
        return df

    total = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        home  = str(row.get('Equipo1', ''))
        fecha = str(row.get('Fecha', ''))
        mid   = indice.get((home, fecha))
        if not mid:
            continue

        try:
            time.sleep(DELAY_US)
            page.goto(f'{US_BASE}/match/{mid}', wait_until='networkidle', timeout=30000)
            shots = page.evaluate(
                "() => typeof window.shotsData !== 'undefined' ? window.shotsData : null"
            )
        except Exception as e:
            print(f'  [Understat] ERROR match {mid}: {e}')
            continue

        if not shots:
            continue

        sh = _calcular_stats_shots(shots, 'h')
        sa = _calcular_stats_shots(shots, 'a')

        changed = False
        for col, val in sh.items():
            key = f'{col}_E1'
            if key in df.columns and (pd.isna(df.at[idx, key]) or df.at[idx, key] == 0):
                df.at[idx, key] = val
                changed = True
        for col, val in sa.items():
            key = f'{col}_E2'
            if key in df.columns and (pd.isna(df.at[idx, key]) or df.at[idx, key] == 0):
                df.at[idx, key] = val
                changed = True

        if changed:
            total += 1
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(df)} procesados ({total} con nuevos datos)')

    print(f'  [Understat] {total} partidos enriquecidos')
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline principal
# ═══════════════════════════════════════════════════════════════════════════════

def enriquecer(csv_path, fuentes=('ss', 'understat')):
    print(f'\n=== Enriqueciendo {csv_path} ===')
    df = pd.read_csv(csv_path)
    print(f'  Filas: {len(df)} | Cobertura inicial: '
          f'{int(df.notna().sum().sum())}/{df.shape[0]*df.shape[1]} celdas')

    sess = BrowserSession()
    try:
        # 1. SofaScore incidents
        if 'ss' in fuentes:
            df = enriquecer_con_incidents(df, sess)

        # 2. Understat (usa la segunda página del mismo browser)
        if 'understat' in fuentes:
            m2 = re.search(r'(\d{4})-\d{2}', csv_path)
            year = m2.group(1) if m2 else '2024'
            df = enriquecer_con_understat(df, sess.us_page, year)
    finally:
        sess.close()

    # Guardar
    out = csv_path.replace('.csv', '_enriquecido.csv')
    df.to_csv(out, index=False)

    total   = df.shape[0] * df.shape[1]
    celdas  = int(df.notna().sum().sum())
    print(f'\n=== Resultado ===')
    print(f'  Cobertura final: {celdas}/{total} ({celdas/total*100:.1f}%)')
    print(f'  Guardado en    : {out}')

    cob = (df.notna().mean() * 100).sort_values(ascending=False)
    print(f'\n  Columnas con datos ({(cob>0).sum()}/140):')
    print(cob[cob > 0].to_string())
    return df


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    csv_path = sys.argv[1]
    fuentes  = {'ss', 'understat'}
    if '--solo-ss' in sys.argv:
        fuentes = {'ss'}
    elif '--solo-understat' in sys.argv:
        fuentes = {'understat'}

    enriquecer(csv_path, fuentes)
