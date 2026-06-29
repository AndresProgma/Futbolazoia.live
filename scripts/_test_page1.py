import time
from playwright.sync_api import sync_playwright

SEASON_ID = 58210
TOURNAMENT_ID = 16
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')

pw = sync_playwright().start()
br = pw.chromium.launch(headless=True, args=['--disable-blink-features=AutomationControlled'])
ctx = br.new_context(user_agent=UA, locale='es-ES')
pg = ctx.new_page()
pg.goto('https://www.sofascore.com/', wait_until='domcontentloaded', timeout=45000)
time.sleep(4)

def fetch(path):
    js = ("async () => { const r = await fetch("
          f"'https://www.sofascore.com/api/v1{path}',"
          "{headers:{'Accept':'application/json'}}); return {status:r.status, ok:r.ok}; }")
    return pg.evaluate(js)

print('home warmup, page0:', fetch(f'/unique-tournament/{TOURNAMENT_ID}/season/{SEASON_ID}/events/last/0'))
print('home warmup, page1:', fetch(f'/unique-tournament/{TOURNAMENT_ID}/season/{SEASON_ID}/events/last/1'))

# Navegar a la pagina del torneo y reintentar
pg.goto(f'https://www.sofascore.com/tournament/football/world/world-championship/16', wait_until='domcontentloaded', timeout=45000)
time.sleep(5)
print('after tournament nav, page1:', fetch(f'/unique-tournament/{TOURNAMENT_ID}/season/{SEASON_ID}/events/last/1'))
print('after tournament nav, page2:', fetch(f'/unique-tournament/{TOURNAMENT_ID}/season/{SEASON_ID}/events/last/2'))
br.close(); pw.stop()
