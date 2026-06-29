"""
Scraper puntual del Mundial 2026 (SofaScore, season 58210).

SofaScore endureció Cloudflare: el fetch manual a /event/{id}/statistics
devuelve 403 "challenge", pero la propia página del evento sí carga esa API
con 200. Por eso aquí:
  1. Listamos los partidos TERMINADOS con fetch a events/last/{page} (sí pasa).
  2. Para cada partido navegamos a su página y CAPTURAMOS el cuerpo de las
     respuestas /statistics y /lineups que la web dispara sola.

Reutiliza los parsers del scraper de selecciones.
"""

import os
import re
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from playwright.sync_api import sync_playwright

from scraper.scraper_premier import _extraer_stats_periodo
from scraper.scraper_selecciones import (
    _norm_seleccion, _confederacion, construir_fila, COLUMNAS_SELECCIONES,
)

SEASON_ID = 58210
TOURNAMENT_ID = 16
NOMBRE_TORNEO = 'FIFA World Cup'
SEASON_NAME = '2026'
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')


def _ratings_from_lineups(data):
    """Promedio de rating del once titular (misma lógica que obtener_ratings)."""
    if not data:
        return None, None

    def avg(players):
        vals = [float(p['statistics']['rating']) for p in players
                if not p.get('substitute', False)
                and p.get('statistics', {}).get('rating') is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    return (avg(data.get('home', {}).get('players', [])),
            avg(data.get('away', {}).get('players', [])))


def _fase_from_round(ev):
    info = ev.get('roundInfo') or {}
    name = info.get('name') or ''
    if name:
        return name
    grp = ev.get('tournament', {}).get('name', '')
    m = re.search(r'Group\s+([A-L])', grp)
    if m:
        return f'Grupo {m.group(1)}'
    return 'Fase de grupos'


def main():
    pw = sync_playwright().start()
    br = pw.chromium.launch(headless=True,
                            args=['--disable-blink-features=AutomationControlled'])
    ctx = br.new_context(user_agent=UA, locale='es-ES')
    pg = ctx.new_page()

    # Warm-up: navegar la web resuelve el challenge de Cloudflare.
    pg.goto('https://www.sofascore.com/', wait_until='domcontentloaded', timeout=45000)
    time.sleep(4)

    def fetch(path, intentos=5):
        # El challenge de Cloudflare es flaky: un 403 puede pasar al reintentar.
        js = ("async () => { const r = await fetch("
              f"'https://www.sofascore.com/api/v1{path}',"
              "{headers:{'Accept':'application/json'}});"
              " if(!r.ok) return {_error:r.status}; return r.json(); }")
        data = None
        for _ in range(intentos):
            data = pg.evaluate(js)
            if data and '_error' not in data:
                return data
            time.sleep(2)
        return data

    # 1. Listar partidos terminados (events/last sí responde 200)
    eventos = []
    page = 0
    while True:
        data = fetch(f'/unique-tournament/{TOURNAMENT_ID}/season/{SEASON_ID}/events/last/{page}')
        if not data or '_error' in data:
            print(f'  events/last/{page}: {data}')
            break
        evs = data.get('events', [])
        eventos.extend(evs)
        if not data.get('hasNextPage'):
            break
        page += 1

    finished = [e for e in eventos if e.get('status', {}).get('type') == 'finished']
    print(f'  Partidos terminados: {len(finished)}')

    filas = []
    for i, ev in enumerate(finished, 1):
        eid = ev['id']
        slug = ev.get('slug')
        cid = ev.get('customId')
        home = _norm_seleccion(ev.get('homeTeam', {}).get('name', ''))
        away = _norm_seleccion(ev.get('awayTeam', {}).get('name', ''))
        from datetime import datetime
        ts = ev.get('startTimestamp')
        fecha = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else None

        partido = {
            'home': home, 'away': away, 'fecha': fecha,
            'goles_home': ev.get('homeScore', {}).get('current'),
            'goles_away': ev.get('awayScore', {}).get('current'),
            'fase': _fase_from_round(ev),
        }
        print(f'  [{i}/{len(finished)}] {home} {partido["goles_home"]}-{partido["goles_away"]} {away} ({fecha})', end=' ', flush=True)

        # Capturar /statistics y /lineups que la página dispara sola.
        # El challenge de Cloudflare es flaky → reintentar con scroll y esperas.
        url = f'https://www.sofascore.com/{slug}/{cid}' if slug and cid else None
        resp = {}

        def on_resp(r, eid=eid):
            if r.url.endswith(f'/event/{eid}/statistics'):
                resp['stats'] = r
            elif r.url.endswith(f'/event/{eid}/lineups'):
                resp['lineups'] = r

        pg.on('response', on_resp)
        try:
            for intento in range(3):
                if not url:
                    break
                resp.clear()
                pg.goto(url, wait_until='domcontentloaded', timeout=45000)
                # scroll para forzar el lazy-load de la pestaña de estadísticas
                for y in (700, 1500, 2300, 1000):
                    pg.mouse.wheel(0, y)
                    time.sleep(1.2)
                for _ in range(16):
                    if 'stats' in resp and 'lineups' in resp:
                        break
                    time.sleep(0.5)
                if 'stats' in resp:
                    break
        finally:
            pg.remove_listener('response', on_resp)

        def _json(r):
            try:
                return r.json() if r is not None else None
            except Exception:
                return None

        sj = _json(resp.get('stats'))
        sh, sa = _extraer_stats_periodo(sj) if sj else ({}, {})
        rh, ra = _ratings_from_lineups(_json(resp.get('lineups')))

        pid = f'SEL-WORLD_CUP-{SEASON_NAME}-{eid}'
        fila = construir_fila(partido, sh, sa, rh, ra, pid, NOMBRE_TORNEO)
        filas.append(fila)
        n = sum(1 for v in fila.values() if v is not None)
        print(f'[{n} cols] ✓')

    br.close()
    pw.stop()

    if not filas:
        print('Sin datos.')
        return

    df = pd.DataFrame(filas, columns=COLUMNAS_SELECCIONES)
    out = f'data/selecciones_world_cup_{SEASON_NAME}.csv'
    df.to_csv(out, index=False)
    n_ok = int(df.notna().sum().sum())
    print(f'\n  Guardado: {out} | {len(df)} filas | cobertura {n_ok/(df.shape[0]*df.shape[1])*100:.1f}%')


if __name__ == '__main__':
    main()
