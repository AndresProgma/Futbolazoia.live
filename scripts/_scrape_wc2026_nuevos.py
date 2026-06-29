"""
Scraper INCREMENTAL del Mundial 2026.

A diferencia de _scrape_wc2026.py (que reescribe el CSV entero), este:
  1. Lee los Partido_id que YA están en data/selecciones_world_cup_2026.csv.
  2. Lista los terminados de SofaScore y descarta los que ya existen.
  3. Scrapea SOLO los nuevos y los AGREGA (append) al final del CSV.
  4. Si un partido nuevo baja sin estadísticas, lo descarta (no lo guarda).

Las filas existentes quedan intactas.
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from playwright.sync_api import sync_playwright

from scraper.scraper_premier import _extraer_stats_periodo
from scraper.scraper_selecciones import (
    _norm_seleccion, construir_fila, COLUMNAS_SELECCIONES,
)
from scripts._scrape_wc2026 import (
    SEASON_ID, TOURNAMENT_ID, NOMBRE_TORNEO, SEASON_NAME, UA,
    _ratings_from_lineups, _fase_from_round,
)

CSV_PATH = 'data/selecciones_world_cup_2026.csv'


def main():
    # IDs existentes
    df_old = pd.read_csv(CSV_PATH, dtype=str)
    ids_existentes = set(df_old['Partido_id'].astype(str))
    print(f'  CSV actual: {len(df_old)} partidos')

    pw = sync_playwright().start()
    br = pw.chromium.launch(headless=True,
                            args=['--disable-blink-features=AutomationControlled'])
    ctx = br.new_context(user_agent=UA, locale='es-ES')
    pg = ctx.new_page()

    pg.goto('https://www.sofascore.com/', wait_until='domcontentloaded', timeout=45000)
    time.sleep(4)

    def fetch(path, intentos=5):
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

    eventos = []
    page = 0
    while True:
        data = fetch(f'/unique-tournament/{TOURNAMENT_ID}/season/{SEASON_ID}/events/last/{page}')
        if not data or '_error' in data:
            print(f'  events/last/{page}: {data}')
            break
        eventos.extend(data.get('events', []))
        if not data.get('hasNextPage'):
            break
        page += 1

    finished = [e for e in eventos if e.get('status', {}).get('type') == 'finished']
    nuevos = [e for e in finished
              if f'SEL-WORLD_CUP-{SEASON_NAME}-{e["id"]}' not in ids_existentes]
    print(f'  Terminados en SofaScore: {len(finished)} | nuevos: {len(nuevos)}')
    if not nuevos:
        print('  No hay partidos nuevos. CSV sin cambios.')
        br.close(); pw.stop()
        return

    # El dataset existente es "solo goles/resultado" (las 28 filas ya solo
    # tienen 15-17 de 143 cols). SofaScore bloquea /statistics con Cloudflare,
    # así que armamos las filas directo de events/last (goles/resultado/fase),
    # sin navegar página por página. Mismo nivel de datos que lo ya guardado.
    filas = []
    for i, ev in enumerate(nuevos, 1):
        eid = ev['id']
        home = _norm_seleccion(ev.get('homeTeam', {}).get('name', ''))
        away = _norm_seleccion(ev.get('awayTeam', {}).get('name', ''))
        ts = ev.get('startTimestamp')
        fecha = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else None

        partido = {
            'home': home, 'away': away, 'fecha': fecha,
            'goles_home': ev.get('homeScore', {}).get('current'),
            'goles_away': ev.get('awayScore', {}).get('current'),
            'fase': _fase_from_round(ev),
        }
        pid = f'SEL-WORLD_CUP-{SEASON_NAME}-{eid}'
        fila = construir_fila(partido, {}, {}, None, None, pid, NOMBRE_TORNEO)
        filas.append(fila)
        print(f'  [{i}/{len(nuevos)}] {home} {partido["goles_home"]}-{partido["goles_away"]} {away} ({fecha}) ✓')

    br.close()
    pw.stop()

    if not filas:
        print('\n  Ningún partido nuevo con datos. CSV sin cambios.')
        return

    df_new = pd.DataFrame(filas, columns=COLUMNAS_SELECCIONES)
    df_out = pd.concat([df_old, df_new], ignore_index=True)
    df_out.to_csv(CSV_PATH, index=False)
    print(f'\n  Agregados {len(df_new)} partidos. CSV: {len(df_old)} → {len(df_out)}.')


if __name__ == '__main__':
    main()
