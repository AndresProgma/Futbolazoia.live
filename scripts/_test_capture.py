import time, json
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

# obtener slug del torneo desde page0
js = ("async () => { const r = await fetch("
      f"'https://www.sofascore.com/api/v1/unique-tournament/{TOURNAMENT_ID}/season/{SEASON_ID}/events/last/0',"
      "{headers:{'Accept':'application/json'}}); return r.ok ? r.json() : {_e:r.status}; }")
d = pg.evaluate(js)
ev0 = d['events'][0]
ut = ev0['tournament']['uniqueTournament']
print('uniqueTournament:', ut.get('slug'), ut.get('id'))

captured = []
def on_resp(r):
    if '/events/last/' in r.url:
        try:
            captured.append((r.url, r.json()))
        except Exception:
            captured.append((r.url, None))
pg.on('response', on_resp)

url = f"https://www.sofascore.com/tournament/football/{ut.get('category',{}).get('slug','world') if isinstance(ut.get('category'),dict) else 'world'}/{ut.get('slug')}/{TOURNAMENT_ID}#id:{SEASON_ID},tab:matches"
print('nav:', url)
pg.goto(url, wait_until='domcontentloaded', timeout=45000)
time.sleep(5)
for _ in range(25):
    pg.mouse.wheel(0, 1600)
    time.sleep(1.0)

ids = set()
for u, j in captured:
    if j and 'events' in j:
        for e in j['events']:
            ids.add(e['id'])
    print('captured', u, '->', len(j.get('events',[])) if j else j)
print('total eventos unicos capturados:', len(ids))
br.close(); pw.stop()
