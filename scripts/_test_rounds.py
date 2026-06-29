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
time.sleep(3)

def fetch(path):
    js = ("async () => { const r = await fetch("
          f"'https://www.sofascore.com/api/v1{path}',"
          "{headers:{'Accept':'application/json'}}); return r.ok ? r.json() : {_e:r.status}; }")
    return pg.evaluate(js)

# rounds metadata
meta = fetch(f'/unique-tournament/{TOURNAMENT_ID}/season/{SEASON_ID}/rounds')
print('rounds meta:', meta if '_e' in (meta or {'_e':1}) else [r.get('round') for r in meta.get('rounds',[])])

for n in range(1, 9):
    d = fetch(f'/unique-tournament/{TOURNAMENT_ID}/season/{SEASON_ID}/events/round/{n}')
    if d and '_e' not in d:
        evs = d.get('events', [])
        fin = sum(1 for e in evs if e.get('status',{}).get('type')=='finished')
        print(f'round {n}: {len(evs)} eventos, {fin} finished')
    else:
        print(f'round {n}: {d}')
    time.sleep(0.4)
br.close(); pw.stop()
