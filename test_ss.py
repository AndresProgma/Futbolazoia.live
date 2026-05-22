"""Chequear partidos disponibles en PL 25/26."""
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    br  = p.chromium.launch(headless=True)
    ctx = br.new_context(user_agent='Mozilla/5.0 Chrome/131')
    pg  = ctx.new_page()
    pg.goto('https://www.sofascore.com/', wait_until='domcontentloaded', timeout=30000)

    def api(path):
        return pg.evaluate(f"""
            async () => {{
                const r = await fetch('https://api.sofascore.com/api/v1{path}',
                    {{headers:{{'Accept':'application/json'}}}});
                return r.json();
            }}
        """)

    rounds_data = api('/unique-tournament/17/season/76986/rounds')
    rounds = [r['round'] for r in rounds_data.get('rounds', [])]
    print(f'Rounds en 25/26: {len(rounds)} → {rounds}')

    terminados = 0
    pendientes = 0
    for rnd in rounds:
        data = api(f'/unique-tournament/17/season/76986/events/round/{rnd}')
        for ev in data.get('events', []):
            status = ev.get('status', {}).get('type', '')
            if status == 'finished':
                terminados += 1
            else:
                pendientes += 1

    print(f'\nTerminados : {terminados}')
    print(f'Pendientes : {pendientes}')
    print(f'Total      : {terminados + pendientes}')
    br.close()
